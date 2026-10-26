#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent / "scripts"
for _path in (_HERE, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config import SearchLatentConfig
from model import CompressorAdapterTrainable, LatentForward, LatentModelBuilder
from prompts import (
    get_multiqa_search_o1_instruction,
    get_singleqa_search_o1_instruction,
    get_task_instruction_openqa,
)
from evaluate import evaluate_predictions, extract_answer
from web_search import bing_web_search, extract_relevant_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

TARGET_DATASETS = ("2wiki", "nq", "musique", "hotpotqa", "triviaqa")
_FALLBACK_MAX_SEARCH_LIMIT = 5
_FALLBACK_MAX_TURN = 5
_FALLBACK_TOP_K = 5
_COMPRESSION_PROMPT = """You are a web search result compression assistant for an AI agent.

## Your Task
The agent below is solving a multi-step reasoning task using web search.
It has just received search results from the web. Your job is to compress
these search results into a compact latent representation that preserves
ALL information the agent needs to decide its next action.

## Global Task Goal
{question}

## Agent Current Step
{current_step}

## Original Web Search Results
{tool_output}

## Compression Requirement
Compress the above search results. You must preserve:
- All factual information directly relevant to the task goal
- Entity names, dates, numbers, relationships mentioned
- Any information that would affect the agent's next decision
"""


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]) -> None:
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if input_ids.numel() == 0:
            return False
        seq = input_ids[0].tolist()
        for stop_ids in self.stop_sequences:
            if len(seq) >= len(stop_ids) and seq[-len(stop_ids):] == stop_ids:
                return True
        return False


def _dataset_inference_defaults(dataset_name: str) -> dict[str, int]:
    qa = {"nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki", "medmcqa", "pubhealth"}
    out = {
        "max_search_limit": _FALLBACK_MAX_SEARCH_LIMIT,
        "max_turn": _FALLBACK_MAX_TURN,
        "top_k": _FALLBACK_TOP_K,
    }
    if dataset_name not in qa:
        return out
    if dataset_name in {"hotpotqa", "musique", "bamboogle", "2wiki"}:
        out["max_search_limit"] = 10
        out["max_turn"] = 150
    else:
        out["max_search_limit"] = 5
        out["max_turn"] = 100
    out["top_k"] = 5
    return out


def _resolve_int(cli_value: int | None, default_from_dataset: int) -> int:
    return default_from_dataset if cli_value is None else cli_value


def _build_user_content(dataset_name: str, question: str, model_path: str) -> str:
    defaults = _dataset_inference_defaults(dataset_name)
    max_search_limit = defaults["max_search_limit"]
    if dataset_name in {"nq", "triviaqa"}:
        instruction = get_singleqa_search_o1_instruction(max_search_limit)
    else:
        instruction = get_multiqa_search_o1_instruction(max_search_limit)
    if "qwq" in model_path.lower():
        task = get_task_instruction_openqa(question, model_name="qwq")
    else:
        task = get_task_instruction_openqa(question)
    return instruction + task


def _build_initial_prompt(agent_tokenizer, dataset_name: str, question: str, model_path: str) -> str:
    user_content = _build_user_content(dataset_name, question, model_path)
    return agent_tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _render_compressor_input(question: str, current_step: str, tool_output: str) -> str:
    return _COMPRESSION_PROMPT.format(
        question=question or "",
        current_step=current_step or "",
        tool_output=tool_output or "",
    )


def _extract_current_step_text(step_text: str, search_query: str) -> str:
    text = (step_text or "").rstrip()
    query_block = f"{BEGIN_SEARCH_QUERY}{search_query or ''}{END_SEARCH_QUERY}"
    if query_block in text:
        end_idx = text.find(query_block) + len(query_block)
        step = text[:end_idx].rstrip()
    elif text:
        step = text
    else:
        step = query_block
    duplicated = re.escape(query_block) + r"(?:\s*" + re.escape(query_block) + r")+"
    return re.sub(duplicated, query_block, step).rstrip()


def _split_visible_tool_block(agent_prompt_full: str) -> tuple[str, str] | None:
    end_idx = agent_prompt_full.rfind(END_SEARCH_RESULT)
    if end_idx < 0:
        return None
    begin_idx = agent_prompt_full.rfind(BEGIN_SEARCH_RESULT, 0, end_idx)
    if begin_idx < 0:
        return None
    after_idx = end_idx + len(END_SEARCH_RESULT)
    prefix = agent_prompt_full[:begin_idx].rstrip() + "\n" + BEGIN_SEARCH_RESULT + "\n"
    suffix = "\n" + agent_prompt_full[end_idx:after_idx] + agent_prompt_full[after_idx:]
    return prefix, suffix


def _extract_all_between(text: str, start_tag: str, end_tag: str) -> list[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [m.strip() for m in matches if m and str(m).strip()]


def _normalize_search_query(query: str) -> str:
    if not query:
        return ""
    return " ".join(str(query).strip().lower().split())


def _build_search_history_text(records: list[dict[str, Any]], *, max_results_per_query: int = 3) -> str:
    lines: list[str] = []
    for idx, rec in enumerate(records, start=1):
        query = str(rec.get("query") or "").strip()
        lines.append(f"{idx}. {query}")
        for j, item in enumerate((rec.get("results") or [])[:max_results_per_query], start=1):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = " ".join(str(item.get("snippet") or "").split())
            lines.append(f"   - Result {j}: {title}")
            if url:
                lines.append(f"     URL: {url}")
            if snippet:
                lines.append(f"     Snippet: {snippet}")
    return "\n".join(lines).strip()


def _build_finalization_prompt(question: str, search_history_text: str) -> str:
    return (
        "You have reached the maximum number of searches. You are not allowed to search again.\n\n"
        "Answer the question using only the search history below.\n\n"
        f"Question:\n{question}\n\n"
        f"Search history:\n{search_history_text}\n\n"
        "Complete the following answer prefix using only the final answer text.\n"
        "Do not output thinking process, analysis, bullet points, or any extra words.\n"
        "Output only the answer span that should go inside the box.\n\n"
        "Finally, the answer is \\boxed{"
    )


def _has_valid_boxed_answer(output_text: str) -> bool:
    return bool(extract_answer(output_text, mode="qa"))


def _is_placeholder_boxed_answer(answer_text: str) -> bool:
    normalized = " ".join(str(answer_text or "").strip().lower().split())
    return normalized in {"", "answer", "your answer", "actual answer", "the answer"}


def _build_retry_finalization_prompt(question: str, search_history_text: str, bad_output: str) -> str:
    return (
        "Your previous response was invalid.\n"
        "It included extra reasoning or a placeholder answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Search history:\n{search_history_text}\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        "Respond again.\n"
        "Output only the answer span that should go inside the box below, and nothing else.\n\n"
        "Finally, the answer is \\boxed{"
    )


def _clean_generated_answer_span(text: str) -> str:
    span = str(text or "").strip()
    if "}" in span:
        span = span.split("}", 1)[0]
    if "\n" in span:
        span = span.split("\n", 1)[0]
    span = span.strip().strip(".").strip()
    span = span.strip("{} ").strip()
    return span


def _looks_like_explanatory_span(span: str) -> bool:
    text = " ".join(str(span or "").strip().split())
    lower = text.lower()
    if not text:
        return True
    bad_prefixes = (
        "the user wants",
        "the question asks",
        "based on the search history",
        "from the search history",
        "the answer is",
        "thinking process",
    )
    if any(lower.startswith(prefix) for prefix in bad_prefixes):
        return True
    if any(token in lower for token in (" because ", " which ", " who ", " that ", " based on ", " identify ", " compare ")):
        return True
    return len(text.split()) > 12


def _build_answer_extraction_prompt(question: str, search_history_text: str, bad_span: str) -> str:
    return (
        "Extract the final short answer only.\n\n"
        f"Question:\n{question}\n\n"
        f"Search history:\n{search_history_text}\n\n"
        f"Verbose or invalid answer:\n{bad_span}\n\n"
        "Output only the short answer span that should go inside the box.\n\n"
        "Finally, the answer is \\boxed{"
    )


def _truncate_prompt_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_prompt_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_prompt_tokens > 0 and input_ids.size(1) > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
        attention_mask = attention_mask[:, -max_prompt_tokens:]
    return input_ids, attention_mask


def _truncate_student_segment_ids(
    segment_ids: list[torch.Tensor],
    num_latent_tokens: int,
    max_prompt_tokens: int,
) -> list[torch.Tensor]:
    available_text = max(0, max_prompt_tokens - num_latent_tokens)
    total_text = sum(seg.numel() for seg in segment_ids)
    trim = max(0, total_text - available_text)
    out = [seg.clone() for seg in segment_ids]
    seg_idx = 0
    while trim > 0 and seg_idx < len(out):
        seg = out[seg_idx]
        if seg.numel() <= trim:
            trim -= seg.numel()
            out[seg_idx] = seg[:0]
        else:
            out[seg_idx] = seg[trim:]
            trim = 0
        seg_idx += 1
    return out


def _decode_new_tokens(tokenizer, generated_ids: torch.Tensor, prompt_len: int) -> str:
    if generated_ids.dim() != 2:
        generated_ids = generated_ids.view(1, -1)
    seq = generated_ids[0]
    if prompt_len >= seq.size(0):
        return ""
    return tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)


def _decode_after_inputs_embeds(tokenizer, out: torch.Tensor) -> str:
    if out.dim() != 2:
        out = out.view(1, -1)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _embed_token_ids(agent_model, token_ids: torch.Tensor) -> torch.Tensor:
    embedding = agent_model.get_input_embeddings()
    model_dtype = next(agent_model.parameters()).dtype
    return embedding(token_ids).to(dtype=model_dtype)


def _truncate_embeds_history(
    history_embeds: torch.Tensor,
    history_token_ids: torch.Tensor,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_prompt_tokens > 0 and history_embeds.size(1) > max_prompt_tokens:
        history_embeds = history_embeds[:, -max_prompt_tokens:, :]
        history_token_ids = history_token_ids[:, -max_prompt_tokens:]
    return history_embeds, history_token_ids


def _append_history_chunk(
    history_embeds: torch.Tensor,
    history_token_ids: torch.Tensor,
    chunk_embeds: torch.Tensor,
    chunk_token_ids: torch.Tensor,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_embeds.dim() != 3:
        raise ValueError(f"chunk_embeds must be rank-3, got shape={tuple(chunk_embeds.shape)}")
    if chunk_token_ids.dim() != 2:
        raise ValueError(f"chunk_token_ids must be rank-2, got shape={tuple(chunk_token_ids.shape)}")
    if history_embeds.size(0) != 1 or chunk_embeds.size(0) != 1:
        raise ValueError("Only batch size 1 is supported for latent history evaluation")
    history_embeds = torch.cat([history_embeds, chunk_embeds], dim=1)
    history_token_ids = torch.cat([history_token_ids, chunk_token_ids], dim=1)
    return _truncate_embeds_history(history_embeds, history_token_ids, max_prompt_tokens)

def _eos_token_ids(tokenizer) -> int | list[int] | None:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def _eos_token_id_set(tokenizer) -> set[int]:
    eos = _eos_token_ids(tokenizer)
    if eos is None:
        return set()
    if isinstance(eos, list):
        return set(eos)
    return {int(eos)}

def _stop_reached(generated_ids: list[int], stop_sequences: list[list[int]], eos_token_ids: set[int]) -> bool:
    if eos_token_ids and generated_ids and generated_ids[-1] in eos_token_ids:
        return True
    for stop_ids in stop_sequences:
        if stop_ids and len(generated_ids) >= len(stop_ids) and generated_ids[-len(stop_ids):] == stop_ids:
            return True
    return False


def _generate_with_latent_history(
    agent_model,
    agent_tokenizer,
    history_embeds: torch.Tensor,
    history_token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_sequences: list[list[int]],
) -> tuple[str, int, int, float, torch.Tensor, torch.Tensor]:
    device = next(agent_model.parameters()).device
    history_embeds, history_token_ids = _truncate_embeds_history(history_embeds, history_token_ids, max_prompt_tokens)
    prompt_len = int(history_embeds.size(1))
    current_embeds = history_embeds
    current_attention = torch.ones((1, prompt_len), dtype=torch.long, device=device)
    past_key_values = None
    generated_ids: list[int] = []
    seq_len = prompt_len
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = agent_model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            next_token_id = int(torch.argmax(out.logits[0, -1, :]).item())
            generated_ids.append(next_token_id)
            if _stop_reached(generated_ids, stop_sequences, _eos_token_id_set(agent_tokenizer)):
                break
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            current_embeds = _embed_token_ids(agent_model, next_token)
            seq_len += 1
            current_attention = torch.ones((1, seq_len), dtype=torch.long, device=device)

    elapsed = time.perf_counter() - t0
    if generated_ids:
        generated_ids_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        generated_embeds = _embed_token_ids(agent_model, generated_ids_tensor)
        history_embeds, history_token_ids = _append_history_chunk(
            history_embeds,
            history_token_ids,
            generated_embeds,
            generated_ids_tensor,
            max_prompt_tokens,
        )
        text = agent_tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        text = ""
    return text, prompt_len, len(generated_ids), elapsed, history_embeds, history_token_ids


def _load_agent_only(cfg: SearchLatentConfig):
    model_path = cfg.latent.agent_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def _load_latent_stack(cfg: SearchLatentConfig, checkpoint_dir: Path):
    builder = LatentModelBuilder(cfg.latent)
    components = builder.load_from_checkpoint(checkpoint_dir)
    n_agent_gpus = getattr(cfg.latent, "agent_num_gpus", 0) or 0
    if n_agent_gpus > 0 and torch.cuda.is_available():
        if getattr(components, "trainable_compress", None) is None:
            components.trainable_compress = CompressorAdapterTrainable(
                components.compressor_model,
                components.adapter,
            )
        forward = LatentForward(
            components,
            cfg.latent,
            device_agent=next(components.agent_model.parameters()).device,
            device_compressor=next(components.compressor_model.parameters()).device,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        components.agent_model.to(device)
        components.compressor_model.to(device)
        components.adapter.to(device)
        forward = LatentForward(components, cfg.latent).to(device)
    components.agent_model.eval()
    components.compressor_model.eval()
    components.adapter.eval()
    return components, forward


def _build_stop_criteria(tokenizer) -> StoppingCriteriaList:
    stop_sequences: list[list[int]] = []
    stop_ids = tokenizer(END_SEARCH_QUERY, add_special_tokens=False)["input_ids"]
    if stop_ids:
        stop_sequences.append(stop_ids)
    return StoppingCriteriaList([StopOnSubsequence(stop_sequences)])


def _build_stop_sequences(tokenizer) -> list[list[int]]:
    stop_sequences: list[list[int]] = []
    stop_ids = tokenizer(END_SEARCH_QUERY, add_special_tokens=False)["input_ids"]
    if stop_ids:
        stop_sequences.append(stop_ids)
    return stop_sequences


def _generate_original(
    agent_model,
    agent_tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_criteria: StoppingCriteriaList,
) -> tuple[str, int, int, float]:
    device = next(agent_model.parameters()).device
    enc = agent_tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    input_ids, attn = _truncate_prompt_ids(input_ids, attn, max_prompt_tokens)
    prompt_len = int(input_ids.shape[1])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = agent_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=stop_criteria,
            pad_token_id=agent_tokenizer.pad_token_id or agent_tokenizer.eos_token_id,
            eos_token_id=agent_tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = _decode_new_tokens(agent_tokenizer, out, prompt_len)
    completion_tokens = max(0, int(out.shape[1]) - prompt_len)
    return text, prompt_len, completion_tokens, elapsed


def _generate_plain_completion(
    agent_model,
    agent_tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    stop_criteria = StoppingCriteriaList(
        [
            StopOnSubsequence(
                [agent_tokenizer(BEGIN_SEARCH_QUERY, add_special_tokens=False)["input_ids"]]
            )
        ]
    )
    return _generate_original(
        agent_model,
        agent_tokenizer,
        prompt_text,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max(1, getattr(agent_model.config, "max_position_embeddings", 4096) - max_new_tokens),
        stop_criteria=stop_criteria,
    )


def _generate_boxed_answer_span(
    agent_model,
    agent_tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    stop_sequences = []
    for stop_text in ("}", "}\n"):
        stop_ids = agent_tokenizer(stop_text, add_special_tokens=False)["input_ids"]
        if stop_ids:
            stop_sequences.append(stop_ids)
    stop_criteria = StoppingCriteriaList([StopOnSubsequence(stop_sequences)])
    return _generate_original(
        agent_model,
        agent_tokenizer,
        prompt_text,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max(1, getattr(agent_model.config, "max_position_embeddings", 4096) - max_new_tokens),
        stop_criteria=stop_criteria,
    )


# Legacy helper retained for reference; the current eval path uses
# `_generate_with_latent_history` for latent_hf and does not call this function.
def _generate_latent(
    components,
    forward: LatentForward,
    compressor_input: str,
    agent_prompt_full: str,
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_criteria: StoppingCriteriaList,
) -> tuple[str, int, int, float]:
    split_prompt = _split_visible_tool_block(agent_prompt_full)
    if split_prompt is None:
        return _generate_original(
            components.agent_model,
            components.agent_tokenizer,
            agent_prompt_full,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            stop_criteria=stop_criteria,
        )

    compressor_tokenizer = components.compressor_tokenizer
    agent_tokenizer = components.agent_tokenizer
    device_agent = next(components.agent_model.parameters()).device
    device_compressor = next(components.compressor_model.parameters()).device

    comp_enc = compressor_tokenizer(
        compressor_input,
        truncation=True,
        max_length=getattr(forward.cfg, "compressor_max_length", 4096),
        padding=False,
        return_tensors="pt",
    )
    comp_ids = comp_enc["input_ids"].to(device_compressor)
    comp_attn = comp_enc["attention_mask"].to(device_compressor)

    prefix_text, suffix_text = split_prompt
    segment_ids = [
        agent_tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0],
        agent_tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0],
    ]
    segment_ids = _truncate_student_segment_ids(
        segment_ids,
        getattr(forward.cfg, "num_latent_tokens", 0),
        max_prompt_tokens,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        latent_hidden = forward._compressor_to_latent_hidden(comp_ids, comp_attn)
        latent_hidden = torch.nan_to_num(latent_hidden, nan=0.0, posinf=0.0, neginf=0.0)

        agent_dtype = next(components.agent_model.parameters()).dtype
        embedding = components.agent_model.get_input_embeddings()
        parts: list[torch.Tensor] = []
        for seg_idx, seg_ids_cpu in enumerate(segment_ids):
            seg_ids = seg_ids_cpu.to(device_agent)
            if seg_ids.numel() > 0:
                seg_embeds = embedding(seg_ids.unsqueeze(0)).squeeze(0).to(dtype=agent_dtype)
                parts.append(seg_embeds)
            if seg_idx == 0:
                parts.append(latent_hidden[0].to(device=device_agent, dtype=agent_dtype))
        inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)
        attention_mask = torch.ones((1, inputs_embeds.size(1)), dtype=torch.long, device=device_agent)
        input_len = int(inputs_embeds.shape[1])
        out = components.agent_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=stop_criteria,
            pad_token_id=agent_tokenizer.pad_token_id or agent_tokenizer.eos_token_id,
            eos_token_id=agent_tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = _decode_after_inputs_embeds(agent_tokenizer, out)
    completion_tokens = int(out.shape[1])
    return text, input_len, completion_tokens, elapsed


def _dataset_path(dataset_name: str, split: str) -> Path:
    repo_root = _HERE.parent
    return repo_root / "data" / "QA_Datasets" / f"{dataset_name}_{split}.json"


def _is_placeholder_answer(answer: Any) -> bool:
    if isinstance(answer, list):
        values = [str(v).strip().lower() for v in answer]
        return bool(values) and all(v in {"<unk>", "unk", ""} for v in values)
    text = str(answer).strip().lower()
    return text in {"<unk>", "unk", ""}


def _load_dataset_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    path = _dataset_path(dataset_name, split)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    bad_examples = []
    for idx, row in enumerate(rows):
        if _is_placeholder_answer(row.get("answer")):
            bad_examples.append((idx, row.get("Question", "")))
            if len(bad_examples) >= 3:
                break
    if bad_examples:
        details = "; ".join(f"idx={idx} question={question!r}" for idx, question in bad_examples)
        raise ValueError(
            f"Dataset file contains placeholder answers in {path}. "
            f"Examples: {details}. This is usually a bad download/export rather than an eval call issue."
        )
    return rows


def _resolve_arg(cli_value, config_value):
    return config_value if cli_value is None else cli_value


def _print_step(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _append_trace(trace_file: TextIO | None, payload: dict[str, Any]) -> None:
    if trace_file is None:
        return
    trace_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    trace_file.flush()


def _append_trajectory_events(
    trajectory_file: TextIO | None,
    *,
    dataset_name: str,
    question_idx: int,
    question: str,
    baseline: str,
    trajectory: list[dict[str, Any]],
) -> None:
    if trajectory_file is None:
        return
    for event_idx, event in enumerate(trajectory):
        payload = {
            "dataset": dataset_name,
            "question_idx": question_idx,
            "question": question,
            "baseline": baseline,
            "event_idx": event_idx,
            **event,
        }
        trajectory_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    trajectory_file.flush()


def _evaluate_question(
    *,
    baseline: str,
    item: dict[str, Any],
    dataset_name: str,
    cfg: SearchLatentConfig,
    agent_model=None,
    agent_tokenizer=None,
    components=None,
    forward=None,
    search_provider: str,
    search_api_key: str,
    bing_endpoint: str,
    top_k: int,
    max_turn: int,
    max_search_limit: int,
    max_new_tokens: int,
    search_cache: dict[str, Any],
    stop_criteria: StoppingCriteriaList,
    stop_sequences: list[list[int]],
    print_steps: bool,
    trace_file: TextIO | None,
    trajectory_file: TextIO | None,
    question_idx: int,
) -> dict[str, Any]:
    question = item["Question"]
    prompt_full = _build_initial_prompt(
        agent_tokenizer if agent_tokenizer is not None else components.agent_tokenizer,
        dataset_name,
        question,
        cfg.latent.agent_model_path,
    )
    prompt_latent = prompt_full
    history_embeds: torch.Tensor | None = None
    history_token_ids: torch.Tensor | None = None
    seq_output = ""
    pending_operations: list[dict[str, str]] = []
    executed_search_queries: set[str] = set()
    search_records: list[dict[str, Any]] = []
    search_count = 0
    gen_round = 0
    current_compression_context: dict[str, str] | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_generation_time = 0.0
    trajectory: list[dict[str, Any]] = []

    start_time = time.perf_counter()
    _print_step(print_steps, f"\n=== [{dataset_name} #{question_idx}] Question ===\n{question}")
    if baseline == "latent_hf":
        device_agent = next(components.agent_model.parameters()).device
        init_ids = components.agent_tokenizer(
            prompt_latent,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device_agent)
        history_token_ids = init_ids
        history_embeds = _embed_token_ids(components.agent_model, init_ids)
        history_embeds, history_token_ids = _truncate_embeds_history(
            history_embeds,
            history_token_ids,
            max(1, cfg.latent.agent_max_length - max_new_tokens),
        )

    def _generate_standalone_answer(prompt_body: str, *, max_tokens_local: int) -> tuple[str, int, int, float]:
        standalone_prompt = (agent_tokenizer if agent_tokenizer is not None else components.agent_tokenizer).apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if baseline == "original_hf":
            return _generate_plain_completion(
                agent_model,
                agent_tokenizer,
                standalone_prompt,
                max_new_tokens=max_tokens_local,
            )
        return _generate_plain_completion(
            components.agent_model,
            components.agent_tokenizer,
            standalone_prompt,
            max_new_tokens=max_tokens_local,
        )

    def _generate_standalone_boxed_span(prompt_body: str, *, max_tokens_local: int) -> tuple[str, int, int, float]:
        standalone_prompt = (agent_tokenizer if agent_tokenizer is not None else components.agent_tokenizer).apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if baseline == "original_hf":
            return _generate_boxed_answer_span(
                agent_model,
                agent_tokenizer,
                standalone_prompt,
                max_new_tokens=max_tokens_local,
            )
        return _generate_boxed_answer_span(
            components.agent_model,
            components.agent_tokenizer,
            standalone_prompt,
            max_new_tokens=max_tokens_local,
        )

    def _inject_agent_message(message: str, *, event: str, **extra: Any) -> None:
        nonlocal prompt_full, prompt_latent, seq_output, history_embeds, history_token_ids
        prompt_full += message
        prompt_latent += message
        seq_output += message
        if baseline == "latent_hf" and history_embeds is not None and history_token_ids is not None:
            device_agent = next(components.agent_model.parameters()).device
            msg_ids = components.agent_tokenizer(
                message,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device_agent)
            msg_embeds = _embed_token_ids(components.agent_model, msg_ids)
            max_pt = max(1, cfg.latent.agent_max_length - max_new_tokens)
            history_embeds, history_token_ids = _append_history_chunk(
                history_embeds,
                history_token_ids,
                msg_embeds,
                msg_ids,
                max_pt,
            )
        trajectory.append(
            {
                "event": event,
                "turn": gen_round,
                "message": message.strip(),
                **extra,
            }
        )

    def _forced_finalize(reason: str) -> None:
        nonlocal seq_output, prompt_full, prompt_latent, termination_reason
        nonlocal total_prompt_tokens, total_completion_tokens, total_generation_time

        search_history_text = _build_search_history_text(search_records)
        if not search_history_text:
            search_history_text = "No search results were successfully retrieved."

        def _run_boxed_prompt(prompt_body: str, *, max_tokens_local: int = 512) -> tuple[str, int, int, float]:
            return _generate_standalone_boxed_span(prompt_body, max_tokens_local=max_tokens_local)

        prompt_body = _build_finalization_prompt(question, search_history_text)
        span_raw, ptok, ctok, elapsed = _run_boxed_prompt(prompt_body)
        total_prompt_tokens += ptok
        total_completion_tokens += ctok
        total_generation_time += elapsed

        cleaned = _clean_generated_answer_span(span_raw)
        if not cleaned or _is_placeholder_boxed_answer(cleaned) or _looks_like_explanatory_span(cleaned):
            retry_prompt = _build_retry_finalization_prompt(question, search_history_text, span_raw)
            span_raw, ptok, ctok, elapsed = _run_boxed_prompt(retry_prompt)
            total_prompt_tokens += ptok
            total_completion_tokens += ctok
            total_generation_time += elapsed
            cleaned = _clean_generated_answer_span(span_raw)

        boxed_suffix = f"Finally, the answer is \\boxed{{{cleaned}}}" if cleaned else ""
        if cleaned and (_looks_like_explanatory_span(cleaned) or not _has_valid_boxed_answer(boxed_suffix)):
            extract_prompt = _build_answer_extraction_prompt(question, search_history_text, cleaned)
            span_raw, ptok, ctok, elapsed = _run_boxed_prompt(extract_prompt)
            total_prompt_tokens += ptok
            total_completion_tokens += ctok
            total_generation_time += elapsed
            cleaned = _clean_generated_answer_span(span_raw)

        final_text = f"\nFinally, the answer is \\boxed{{{cleaned}}}\n" if cleaned else ""
        if final_text:
            seq_output += final_text
            prompt_full += final_text
            prompt_latent += final_text

        termination_reason = reason
        trajectory.append(
            {
                "event": "forced_finalization",
                "turn": gen_round,
                "reason": reason,
                "search_history": search_history_text,
                "final_answer": cleaned,
            }
        )
        _print_step(
            print_steps,
            f"\n--- [{dataset_name} #{question_idx}] Forced Finalization ({reason}) ---\n"
            f"{cleaned if cleaned else '[EMPTY]'}",
        )

    termination_reason = "loop_exit"

    while True:
        if pending_operations:
            operation = pending_operations.pop(0)
            query = operation["content"]
            _print_step(print_steps, f"\n--- [{dataset_name} #{question_idx}] Search Query ---\n{query}")
            cache_key = f"{search_provider}::{query}"
            if cache_key in search_cache:
                results = search_cache[cache_key]
            else:
                try:
                    results = bing_web_search(
                        query,
                        search_api_key,
                        bing_endpoint,
                        market="en-US",
                        language="en",
                        provider=search_provider,
                    )
                except Exception as e:
                    logger.warning("Search failed for query '%s': %s", query, e)
                    results = {}
                search_cache[cache_key] = results
            relevant_info = extract_relevant_info(results)[:top_k]
            search_result_str = json.dumps(relevant_info, ensure_ascii=False, indent=2)
            append_text_full = f"\n{BEGIN_SEARCH_RESULT}\n{search_result_str}\n{END_SEARCH_RESULT}\n"
            append_text_latent = f"\n{BEGIN_SEARCH_RESULT}\n{END_SEARCH_RESULT}\n"
            prompt_full += append_text_full
            prompt_latent += append_text_latent
            seq_output += append_text_full
            search_count += 1
            search_records.append(
                {
                    "query": query,
                    "results": relevant_info,
                }
            )
            trajectory.append(
                {
                    "event": "search_result",
                    "turn": gen_round,
                    "search_index": search_count,
                    "query": query,
                    "search_result": relevant_info,
                }
            )
            _print_step(print_steps, f"--- [{dataset_name} #{question_idx}] Search Result ---\n{search_result_str}")
            if baseline == "latent_hf":
                current_compression_context = {
                    "question": question,
                    "current_step": _extract_current_step_text(operation.get("step_text", ""), query),
                    "tool_output": search_result_str,
                }
                if history_embeds is None or history_token_ids is None:
                    raise RuntimeError("latent history must be initialized before appending search results")
                compressor_input = _render_compressor_input(
                    current_compression_context["question"],
                    current_compression_context["current_step"],
                    current_compression_context["tool_output"],
                )
                compressor_tokenizer = components.compressor_tokenizer
                agent_tokenizer_local = components.agent_tokenizer
                device_agent = next(components.agent_model.parameters()).device
                device_compressor = next(components.compressor_model.parameters()).device
                comp_enc = compressor_tokenizer(
                    compressor_input,
                    truncation=True,
                    max_length=getattr(forward.cfg, "compressor_max_length", 4096),
                    padding=False,
                    return_tensors="pt",
                )
                comp_ids = comp_enc["input_ids"].to(device_compressor)
                comp_attn = comp_enc["attention_mask"].to(device_compressor)
                with torch.no_grad():
                    latent_hidden = forward._compressor_to_latent_hidden(comp_ids, comp_attn)
                    latent_hidden = torch.nan_to_num(latent_hidden, nan=0.0, posinf=0.0, neginf=0.0)
                latent_hidden = latent_hidden.to(device=device_agent, dtype=next(components.agent_model.parameters()).dtype)
                begin_ids = agent_tokenizer_local(
                    f"\n{BEGIN_SEARCH_RESULT}\n",
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device_agent)
                end_ids = agent_tokenizer_local(
                    f"\n{END_SEARCH_RESULT}\n",
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device_agent)
                latent_placeholder_ids = torch.full(
                    (1, latent_hidden.size(1)),
                    fill_value=agent_tokenizer_local.pad_token_id or agent_tokenizer_local.eos_token_id or 0,
                    dtype=torch.long,
                    device=device_agent,
                )
                chunk_token_ids = torch.cat([begin_ids, latent_placeholder_ids, end_ids], dim=1)
                begin_embeds = _embed_token_ids(components.agent_model, begin_ids)
                end_embeds = _embed_token_ids(components.agent_model, end_ids)
                chunk_embeds = torch.cat([begin_embeds, latent_hidden.to(begin_embeds.dtype), end_embeds], dim=1)
                history_embeds, history_token_ids = _append_history_chunk(
                    history_embeds,
                    history_token_ids,
                    chunk_embeds,
                    chunk_token_ids,
                    max(1, cfg.latent.agent_max_length - max_new_tokens),
                )
            continue

        if gen_round >= max_turn:
            _forced_finalize("forced_finalization_max_turn")
            break

        gen_round += 1
        max_prompt_tokens = max(1, cfg.latent.agent_max_length - max_new_tokens)
        if baseline == "original_hf":
            text, prompt_tokens, completion_tokens, elapsed = _generate_original(
                agent_model,
                agent_tokenizer,
                prompt_full,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                stop_criteria=stop_criteria,
            )
        else:
            if history_embeds is None or history_token_ids is None:
                raise RuntimeError("latent history should be initialized before generation")
            text, prompt_tokens, completion_tokens, elapsed, history_embeds, history_token_ids = _generate_with_latent_history(
                components.agent_model,
                components.agent_tokenizer,
                history_embeds,
                history_token_ids,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                stop_sequences=stop_sequences,
            )
        current_compression_context = None
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_generation_time += elapsed
        trajectory.append(
            {
                "event": "agent_generation",
                "turn": gen_round,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_seconds": elapsed,
                "text": text,
            }
        )
        _print_step(
            print_steps,
            f"\n--- [{dataset_name} #{question_idx}] Agent Output Turn {gen_round} ---\n{text if text else '[EMPTY]'}",
        )

        prompt_full += text
        seq_output += text
        search_queries = _extract_all_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        pending_search_ops = sum(1 for op in pending_operations if op["type"] == "search")
        hit_search_limit = False
        for search_query in search_queries:
            if search_count + pending_search_ops >= max_search_limit:
                hit_search_limit = True
                break
            if search_query in executed_search_queries:
                duplicate_message = (
                    f"\n{BEGIN_SEARCH_RESULT}\n"
                    "You have searched this query. Please refer to previous results.\n"
                    f"{END_SEARCH_RESULT}\n"
                )
                _inject_agent_message(
                    duplicate_message,
                    event="duplicate_search_query",
                    query=search_query,
                )
                _print_step(
                    print_steps,
                    f"--- [{dataset_name} #{question_idx}] Duplicate Search Query ---\n{search_query}",
                )
                continue
            pending_operations.append(
                {
                    "type": "search",
                    "content": search_query,
                    "step_text": _extract_current_step_text(text, search_query),
                }
            )
            executed_search_queries.add(search_query)
            pending_search_ops += 1

        if hit_search_limit:
            _forced_finalize("forced_finalization_search_limit")
            break

        if not search_queries:
            termination_reason = "natural_stop"
            trajectory.append(
                {
                    "event": "terminated",
                    "turn": gen_round,
                    "reason": termination_reason,
                }
            )
            break

    if not _has_valid_boxed_answer(seq_output):
        _forced_finalize("forced_finalization_no_boxed_answer")

    total_elapsed = time.perf_counter() - start_time
    metric, pred_answer = evaluate_predictions(output=seq_output, labeled_answer=item["answer"], mode="qa")
    result = {
        "question": question,
        "answer": item["answer"],
        "prediction": pred_answer,
        "output": seq_output,
        "metric": metric,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "time_seconds": total_elapsed,
        "generation_time_seconds": total_generation_time,
        "search_count": search_count,
        "turns": gen_round,
        "terminated_by": termination_reason,
        "trajectory": trajectory,
    }
    _print_step(
        print_steps,
        f"\n=== [{dataset_name} #{question_idx}] Final ===\nanswer={item['answer']}\nprediction={pred_answer}\nacc={metric['acc']}\n",
    )
    _append_trace(
        trace_file,
        {
            "dataset": dataset_name,
            "question_idx": question_idx,
            **result,
        },
    )
    _append_trajectory_events(
        trajectory_file,
        dataset_name=dataset_name,
        question_idx=question_idx,
        question=question,
        baseline=baseline,
        trajectory=trajectory,
    )
    return result


def _summarize_results(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "accuracy": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "total_time_seconds": 0.0,
            "avg_time_seconds": 0.0,
        }
    accuracy = sum(float(r["metric"]["acc"]) for r in rows) / n
    total_prompt_tokens = sum(int(r["prompt_tokens"]) for r in rows)
    total_completion_tokens = sum(int(r["completion_tokens"]) for r in rows)
    total_time_seconds = sum(float(r["time_seconds"]) for r in rows)
    return {
        "accuracy": accuracy,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "total_time_seconds": total_time_seconds,
        "avg_time_seconds": total_time_seconds / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end HF evaluation for Search-o1 QA baselines")
    parser.add_argument("--config", type=str, default=str(_HERE / "config.yaml"))
    parser.add_argument("--baseline", type=str, choices=["original_hf", "latent_hf"], default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Required for latent_hf.")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-questions", type=int, default=None, help="Per dataset.")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-search-limit", type=int, default=None)
    parser.add_argument("--max-turn", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--search-api-key", type=str, default=None)
    parser.add_argument("--search-provider", type=str, default=None, choices=["tavily", "exa"])
    parser.add_argument("--bing-endpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--trace-output-dir", type=str, default=None)
    parser.add_argument("--print-steps", action="store_true", default=None)
    parser.add_argument("--no-print-steps", action="store_false", dest="print_steps")
    args = parser.parse_args()

    cfg = SearchLatentConfig.load(args.config)
    baseline = _resolve_arg(args.baseline, cfg.eval.baseline)
    if baseline not in {"original_hf", "latent_hf"}:
        raise ValueError(f"Unsupported baseline: {baseline}")
    datasets_arg = _resolve_arg(args.datasets, cfg.eval.datasets)
    split = _resolve_arg(args.split, cfg.eval.split)
    start_idx = _resolve_arg(args.start_idx, cfg.eval.start_idx)
    max_questions = _resolve_arg(args.max_questions, cfg.eval.max_questions)
    max_new_tokens = _resolve_arg(args.max_new_tokens, cfg.eval.max_new_tokens)
    search_api_key = _resolve_arg(args.search_api_key, cfg.eval.search_api_key)
    search_provider = _resolve_arg(args.search_provider, cfg.eval.search_provider)
    bing_endpoint = _resolve_arg(args.bing_endpoint, cfg.eval.bing_endpoint)
    output_json = _resolve_arg(args.output_json, cfg.eval.output_json)
    trace_output_dir = Path(_resolve_arg(args.trace_output_dir, cfg.eval.trace_output_dir))
    print_steps = _resolve_arg(args.print_steps, cfg.eval.print_steps)

    dataset_names = [d.strip() for d in datasets_arg.split(",") if d.strip()]
    if dataset_names == ["all"]:
        dataset_names = list(TARGET_DATASETS)
    for dataset_name in dataset_names:
        if dataset_name not in TARGET_DATASETS:
            raise ValueError(f"Unsupported dataset for this script: {dataset_name}")

    agent_model = agent_tokenizer = None
    components = forward = None
    if baseline == "original_hf":
        agent_model, agent_tokenizer = _load_agent_only(cfg)
        stop_criteria = _build_stop_criteria(agent_tokenizer)
        stop_sequences = _build_stop_sequences(agent_tokenizer)
    else:
        checkpoint_dir = Path(
            _resolve_arg(args.checkpoint, cfg.eval.checkpoint)
            or cfg.latent.eval_checkpoint_dir
            or (Path(cfg.paths.checkpoint_dir) / "final")
        )
        components, forward = _load_latent_stack(cfg, checkpoint_dir)
        stop_criteria = _build_stop_criteria(components.agent_tokenizer)
        stop_sequences = _build_stop_sequences(components.agent_tokenizer)

    all_results: dict[str, list[dict[str, Any]]] = {}
    search_cache: dict[str, Any] = {}
    trace_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        data = _load_dataset_rows(dataset_name, split)
        data = data[start_idx:]
        if max_questions is not None:
            data = data[: max_questions]

        defaults = _dataset_inference_defaults(dataset_name)
        max_search_limit = _resolve_int(_resolve_arg(args.max_search_limit, cfg.eval.max_search_limit), defaults["max_search_limit"])
        max_turn = _resolve_int(_resolve_arg(args.max_turn, cfg.eval.max_turn), defaults["max_turn"])
        top_k = _resolve_int(_resolve_arg(args.top_k, cfg.eval.top_k), defaults["top_k"])

        logger.info(
            "Evaluating dataset=%s baseline=%s questions=%d max_search_limit=%d max_turn=%d top_k=%d",
            dataset_name,
            baseline,
            len(data),
            max_search_limit,
            max_turn,
            top_k,
        )

        rows: list[dict[str, Any]] = []
        trace_path = trace_output_dir / f"{dataset_name}.{baseline}.{split}.jsonl"
        trajectory_path = trace_output_dir / f"{dataset_name}.{baseline}.{split}.trajectory.jsonl"
        with trace_path.open("w", encoding="utf-8") as trace_file, trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            for idx, item in enumerate(data, 1):
                row = _evaluate_question(
                    baseline=baseline,
                    item=item,
                    dataset_name=dataset_name,
                    cfg=cfg,
                    agent_model=agent_model,
                    agent_tokenizer=agent_tokenizer,
                    components=components,
                    forward=forward,
                    search_provider=search_provider,
                    search_api_key=search_api_key,
                    bing_endpoint=bing_endpoint,
                    top_k=top_k,
                    max_turn=max_turn,
                    max_search_limit=max_search_limit,
                    max_new_tokens=max_new_tokens,
                    search_cache=search_cache,
                    stop_criteria=stop_criteria,
                    stop_sequences=stop_sequences,
                    print_steps=print_steps,
                    trace_file=trace_file,
                    trajectory_file=trajectory_file,
                    question_idx=start_idx + idx - 1,
                )
                rows.append(row)
                if idx % 10 == 0:
                    logger.info(
                        "[%s] done %d/%d acc_so_far=%.4f",
                        dataset_name,
                        idx,
                        len(data),
                        sum(float(r["metric"]["acc"]) for r in rows) / len(rows),
                    )
        logger.info("Saved full question traces to %s", trace_path)
        logger.info("Saved step-level trajectories to %s", trajectory_path)
        all_results[dataset_name] = rows

    summary = {dataset_name: _summarize_results(rows) for dataset_name, rows in all_results.items()}
    overall_rows = [row for rows in all_results.values() for row in rows]
    summary["overall"] = _summarize_results(overall_rows)

    print(f"baseline={baseline}")
    if baseline == "latent_hf":
        print(
            f"checkpoint={_resolve_arg(args.checkpoint, cfg.eval.checkpoint) or cfg.latent.eval_checkpoint_dir or (Path(cfg.paths.checkpoint_dir) / 'final')}"
        )
    print(f"datasets={','.join(dataset_names)}")
    print(f"split={split}")
    for dataset_name in dataset_names + ["overall"]:
        stats = summary[dataset_name]
        print(f"\n[{dataset_name}]")
        print(f"accuracy={stats['accuracy']:.4f}")
        print(f"avg_prompt_tokens={stats['avg_prompt_tokens']:.2f}")
        print(f"avg_completion_tokens={stats['avg_completion_tokens']:.2f}")
        print(f"total_time_seconds={stats['total_time_seconds']:.2f}")
        print(f"avg_time_seconds={stats['avg_time_seconds']:.2f}")

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": baseline,
            "datasets": dataset_names,
            "split": split,
            "summary": summary,
            "results": all_results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
