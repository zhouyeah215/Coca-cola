#!/usr/bin/env python3
"""Convert Search-o1 web-search data to latent SFT JSONL.

This script writes ``agent_prompt_full`` directly into the output JSONL as the
full chat-formatted teacher prompt, so training does not need to reconstruct it.
The prompt format matches the Search-o1 runtime prompt shape: one user message
whose content is ``dataset-specific search instruction + task prompt``, followed
by the visible reasoning/search-result context accumulated so far.

It supports two source formats:

1. ``trajectory``: raw Search-o1 trajectories that still need prompt rebuilding.
2. ``latent2``: existing latent compression rows where ``prompt`` already
   contains the previous-step reasoning and ``agent_prompt_full`` already
   contains the full teacher context accumulated during data collection.
"""

from __future__ import annotations

import argparse
import json
import logging
import glob
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_FULL_PAGE = "<|begin_full_page|>"
END_FULL_PAGE = "<|end_full_page|>"
BEGIN_URL = "<|begin_url|>"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (TRAIN_DIR, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_COMPRESSION_PROMPT = """You are a web search result compression assistant for an AI agent.

## Your Task
The agent below is solving a multi-step reasoning task using web search.
It has just received search results from the web. Your job is to compress
these search results into a compact latent representation that preserves
ALL information the agent needs to decide its next action.

## Global Task Goal
{question}

## Agent's Current Reasoning
{reasoning}

## Search Query Executed
{search_query}

## Original Web Search Results
{tool_output}

## Compression Requirement
Compress the above search results. You must preserve:
- All factual information directly relevant to the task goal
- Entity names, dates, numbers, relationships mentioned
- Any information that would affect the agent's next decision
"""

_FALLBACK_MAX_SEARCH_LIMIT = 5


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _dataset_inference_defaults(dataset_name: str) -> dict[str, int]:
    qa = {"nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki", "medmcqa", "pubhealth"}
    out = {
        "max_search_limit": _FALLBACK_MAX_SEARCH_LIMIT,
    }
    if dataset_name not in qa:
        return out
    if dataset_name in {"hotpotqa", "musique", "bamboogle", "2wiki"}:
        out["max_search_limit"] = 10
    return out


def _render_prompt(question: str, reasoning: str, search_query: str, tool_output: str) -> str:
    return _COMPRESSION_PROMPT.format(
        question=question or "",
        reasoning=reasoning or "",
        search_query=search_query or "",
        tool_output=tool_output or "",
    )


def _extract_latest_local_reasoning(current_reasoning_prefix: str, search_query: str) -> str:
    text = str(current_reasoning_prefix or "")
    if not text.strip():
        return ""

    last_boundary = -1
    for marker in (END_SEARCH_RESULT, END_FULL_PAGE):
        pos = text.rfind(marker)
        if pos >= 0:
            pos += len(marker)
            if pos > last_boundary:
                last_boundary = pos

    if last_boundary >= 0:
        text = text[last_boundary:].lstrip()

    return text.strip()


def _build_searcho1_user_content(
    dataset_name: str | None,
    question: str,
    *,
    question_title: str | None = None,
) -> str:
    from prompts import (
        get_code_search_o1_instruction,
        get_gpqa_search_o1_instruction,
        get_math_search_o1_instruction,
        get_multiqa_search_o1_instruction,
        get_singleqa_search_o1_instruction,
        get_task_instruction_code,
        get_task_instruction_math,
        get_task_instruction_multi_choice,
        get_task_instruction_openqa,
    )

    dataset_key = (dataset_name or "").lower()
    defaults = _dataset_inference_defaults(dataset_key)
    max_search_limit = defaults["max_search_limit"]

    if dataset_key in {"nq", "triviaqa"}:
        instruction = get_singleqa_search_o1_instruction(max_search_limit)
        task = get_task_instruction_openqa(question)
    elif dataset_key in {"hotpotqa", "musique", "bamboogle", "2wiki", "medmcqa", "pubhealth"}:
        instruction = get_multiqa_search_o1_instruction(max_search_limit)
        task = get_task_instruction_openqa(question)
    elif dataset_key in {"math500", "aime", "amc"}:
        instruction = get_math_search_o1_instruction(max_search_limit)
        task = get_task_instruction_math(question)
    elif dataset_key == "gpqa":
        instruction = get_gpqa_search_o1_instruction(max_search_limit)
        task = get_task_instruction_multi_choice(question)
    elif dataset_key == "livecode":
        instruction = get_code_search_o1_instruction(max_search_limit)
        task = get_task_instruction_code(question, question_title=question_title)
    else:
        instruction = get_multiqa_search_o1_instruction(max_search_limit)
        task = get_task_instruction_openqa(question)
    return instruction + task


def _build_agent_prompt_full(
    tokenizer: Any,
    dataset_name: str | None,
    question: str,
    reasoning_prefix: str,
    tool_output: str,
    *,
    question_title: str | None = None,
) -> str:
    user_content = _build_searcho1_user_content(
        dataset_name,
        question,
        question_title=question_title,
    )
    chat_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    reasoning = (reasoning_prefix or "").rstrip()
    search_block = f"{BEGIN_SEARCH_RESULT}\n{tool_output}\n{END_SEARCH_RESULT}\n"
    suffix_parts = []
    if reasoning:
        suffix_parts.append(reasoning)
    suffix_parts.append(search_block.rstrip())
    suffix = "\n".join(suffix_parts).rstrip() + "\n"
    return chat_prefix + suffix


def _has_bad_completion_prefix(text: object) -> bool:
    if text is None:
        return False
    stripped = str(text).lstrip()
    return stripped.startswith(BEGIN_FULL_PAGE) or stripped.startswith(BEGIN_SEARCH_RESULT)


def _has_begin_url_completion(text: object) -> bool:
    if text is None:
        return False
    return str(text).lstrip().startswith(BEGIN_URL)


def _count_tokens(text: str, tokenizer: Any | None) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def _percentile_from_sorted(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _print_token_distribution(name: str, values: list[int]) -> None:
    if not values:
        print(f"{name}: count=0")
        return
    vals = sorted(int(v) for v in values)
    count = len(vals)
    total = sum(vals)
    mean = total / count
    print(
        f"{name}: count={count}, total={total}, min={vals[0]}, max={vals[-1]}, "
        f"mean={mean:.2f}, p50={_percentile_from_sorted(vals, 0.50):.1f}, "
        f"p90={_percentile_from_sorted(vals, 0.90):.1f}, "
        f"p95={_percentile_from_sorted(vals, 0.95):.1f}, "
        f"p99={_percentile_from_sorted(vals, 0.99):.1f}"
    )


def _iter_jsonl_rows(path_pattern: str) -> tuple[Path, dict]:
    matches = sorted(
        Path(p).resolve()
        for p in glob.glob(str(_resolve_repo_path(path_pattern)), recursive=True)
    )
    if not matches:
        raise FileNotFoundError(f"No files matched: {path_pattern}")
    for path in matches:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                yield path, json.loads(line)


def _build_record_from_latent2_row(row: dict) -> dict | None:
    prompt = str(row.get("prompt") or "").strip()
    completion = str(row.get("completion") or "").strip()
    agent_prompt_full = str(row.get("agent_prompt_full") or "").strip()
    if not prompt or not completion or not agent_prompt_full:
        return None
    if _has_bad_completion_prefix(completion):
        return None
    return {
        "prompt": prompt,
        "completion": completion,
        "agent_prompt_full": agent_prompt_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory JSONL -> latent SFT JSONL for Search-o1 web search")
    parser.add_argument(
        "--trajectory-jsonl",
        type=str,
        default="data/traces/musique_train_1k_0.jsonl",
        help="Input trajectory JSONL. Relative paths are resolved from the Search-o1 repo root.",
    )
    parser.add_argument(
        "--source-format",
        type=str,
        choices=("trajectory", "latent2"),
        default="trajectory",
        help="Input format: raw trajectories or existing latent2 JSONL rows.",
    )
    parser.add_argument(
        "--latent2-glob",
        type=str,
        default="data/sft/latent2/*.jsonl",
        help="Glob for latent2 JSONL files when --source-format=latent2.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/latent_web_search_sft.jsonl",
        help="Output JSONL path. Relative paths are resolved from the Search-o1 repo root.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(TRAIN_DIR / "config.yaml"),
        help="Latent training config used to resolve the teacher agent model tokenizer.",
    )
    parser.add_argument(
        "--agent-model-path",
        type=str,
        default=None,
        help="Optional override for the teacher agent model path used to build chat-formatted prompts.",
    )
    parser.add_argument(
        "--allow-missing-kind",
        action="store_true",
        help="Keep rows with missing trajectory_kind as web_search. URL rows are always skipped.",
    )
    parser.add_argument(
        "--min-tool-output-tokens",
        type=int,
        default=10,
        help="Keep samples whose tool output token count is strictly greater than this value.",
    )
    parser.add_argument(
        "--skip-begin-url-completion",
        action="store_true",
        help="Drop rows whose completion starts with <|begin_url|>, matching filter_final_data_begin_url.py.",
    )
    args = parser.parse_args()

    out_path = _resolve_repo_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.source_format == "trajectory":
        from transformers import AutoTokenizer

        from config import SearchLatentConfig

        cfg = SearchLatentConfig.load(args.config)
        agent_model_path = args.agent_model_path or cfg.latent.agent_model_path
        tokenizer = AutoTokenizer.from_pretrained(agent_model_path, trust_remote_code=True)
        in_path = _resolve_repo_path(args.trajectory_jsonl)

    n_in = 0
    n_out = 0
    n_final = 0
    n_skipped_non_search = 0
    n_skipped_empty = 0
    n_skipped_bad_completion_prefix = 0
    n_skipped_begin_url = 0
    n_skipped_short_tool_output = 0
    compressor_prompt_token_lengths: list[int] = []
    agent_prompt_full_token_lengths: list[int] = []

    with open(out_path, "w", encoding="utf-8") as fout:
        if args.source_format == "latent2":
            for source_path, row in _iter_jsonl_rows(args.latent2_glob):
                n_in += 1
                record = _build_record_from_latent2_row(row)
                if record is None:
                    n_skipped_empty += 1
                    continue
                if args.skip_begin_url_completion and _has_begin_url_completion(record["completion"]):
                    n_skipped_begin_url += 1
                    continue
                record["source_file"] = source_path.name
                if tokenizer is not None:
                    compressor_prompt_token_lengths.append(_count_tokens(record["prompt"], tokenizer))
                    agent_prompt_full_token_lengths.append(_count_tokens(record["agent_prompt_full"], tokenizer))
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_out += 1
        else:
            assert tokenizer is not None
            with open(in_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    row = json.loads(line)

                    if row.get("record_type") == "final":
                        n_final += 1
                        continue

                    kind = row.get("trajectory_kind")
                    if kind is None and args.allow_missing_kind:
                        kind = "web_search"
                    if kind != "web_search":
                        n_skipped_non_search += 1
                        continue

                    completion = row.get("next_reasoning_continuation") or row.get("next_generation")
                    if not completion or not str(completion).strip():
                        n_skipped_empty += 1
                        continue
                    if _has_bad_completion_prefix(completion):
                        n_skipped_bad_completion_prefix += 1
                        continue
                    if args.skip_begin_url_completion and _has_begin_url_completion(completion):
                        n_skipped_begin_url += 1
                        continue

                    question = str(row.get("question") or "").strip()
                    dataset_name = row.get("dataset_name")
                    reasoning_full = str(row.get("reasoning") or row.get("current_reasoning_prefix") or "").rstrip()
                    search_query = str(row.get("search_query") or "").strip()
                    tool_output = str(row.get("tool_output") or row.get("retrieved_docs") or "").strip()
                    question_title = row.get("question_title")
                    if not question or not search_query or not tool_output:
                        n_skipped_empty += 1
                        continue
                    reasoning_local = _extract_latest_local_reasoning(reasoning_full, search_query)
                    tool_output_token_count = _count_tokens(tool_output, tokenizer)
                    if tool_output_token_count <= args.min_tool_output_tokens:
                        n_skipped_short_tool_output += 1
                        continue

                    record = {
                        "prompt": _render_prompt(question, reasoning_local, search_query, tool_output),
                        "completion": str(completion).strip(),
                        "agent_prompt_full": _build_agent_prompt_full(
                            tokenizer,
                            dataset_name,
                            question,
                            reasoning_full,
                            tool_output,
                            question_title=question_title,
                        ),
                        "agent_current_reasoning": reasoning_local,
                        "agent_reasoning_full_prefix": reasoning_full,
                        "tool_output_token_count": tool_output_token_count,
                    }
                    if question_title is not None:
                        record["question_title"] = question_title

                    compressor_prompt_token_lengths.append(_count_tokens(record["prompt"], tokenizer))
                    agent_prompt_full_token_lengths.append(_count_tokens(record["agent_prompt_full"], tokenizer))
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_out += 1

    print(
        f"Done: {n_out} samples -> {out_path} "
        f"(source_format={args.source_format}, in={n_in}, final_skipped={n_final}, "
        f"non_search_skipped={n_skipped_non_search}, empty_skipped={n_skipped_empty}, "
        f"bad_completion_skipped={n_skipped_bad_completion_prefix}, "
        f"begin_url_skipped={n_skipped_begin_url}, "
        f"short_tool_output_skipped={n_skipped_short_tool_output})"
    )
    if tokenizer is not None:
        _print_token_distribution("compressor_prompt_tokens", compressor_prompt_token_lengths)
        _print_token_distribution("agent_prompt_full_tokens", agent_prompt_full_token_lengths)


if __name__ == "__main__":
    main()
