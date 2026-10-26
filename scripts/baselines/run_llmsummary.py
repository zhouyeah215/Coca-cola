# run_rag_agent.py
import os
import json
import time
import re
from typing import Optional, List, Dict
import argparse
from pathlib import Path
from runtime_metrics import new_usage, add_agent_usage, add_summary_usage, finish_usage, progress
from search_config import resolve_search_args
from LLMSummary.web_search_adapter import LLMSummarySearchSummarizer, SUMMARY_SYSTEM_PROMPT, SUMMARY_COMPLETION_PREFILL

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from web_search import SearchPaymentRequiredError, bing_web_search, extract_relevant_info
from evaluate import evaluate_predictions, run_evaluation
from prompts import (
    get_singleqa_search_o1_instruction,
    get_multiqa_search_o1_instruction,
    get_gpqa_search_o1_instruction,
    get_math_search_o1_instruction,
    get_code_search_o1_instruction,
    get_task_instruction_openqa,
    get_task_instruction_math,
    get_task_instruction_multi_choice,
    get_task_instruction_code,
)



BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

_FALLBACK_MAX_SEARCH_LIMIT = 5
_FALLBACK_MAX_TURN = 50
_FALLBACK_TOP_K = 10
_SEARCH_LIMIT_STALL_MAX = 2


def _dataset_inference_defaults(dataset_name: str) -> Dict[str, int]:
    qa = ["nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki", "medmcqa", "pubhealth"]
    out = {
        "max_search_limit": _FALLBACK_MAX_SEARCH_LIMIT,
        "max_turn": _FALLBACK_MAX_TURN,
        "top_k": _FALLBACK_TOP_K,
    }
    if dataset_name not in qa:
        return out
    if dataset_name in ["hotpotqa", "musique", "bamboogle", "2wiki"]:
        out["max_search_limit"] = 10
        out["max_turn"] = 150
    else:
        out["max_search_limit"] = 5
        out["max_turn"] = 100
    out["top_k"] = 5
    return out


def _resolve_int(cli_value: Optional[int], default_from_dataset: int) -> int:
    return default_from_dataset if cli_value is None else cli_value


def extract_all_between(text: str, start_tag: str, end_tag: str) -> List[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [m.strip() for m in matches if m and str(m).strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run search-only QA agent without URL fetching.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "gpqa",
            "math500",
            "aime",
            "amc",
            "livecode",
            "nq",
            "triviaqa",
            "hotpotqa",
            "2wiki",
            "musique",
            "bamboogle",
            "medmcqa",
            "pubhealth",
        ],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["test", "diamond", "main", "extended", "train"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--subset_num",
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of examples to process to completion in each batch. Omit to process all examples together.",
    )
    parser.add_argument(
        "--max_search_limit",
        type=int,
        default=None,
        help="Max searches per question. Omit to use dataset-specific default.",
    )
    parser.add_argument(
        "--max_turn",
        type=int,
        default=None,
        help="Max turns. Omit to use dataset-specific default.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Max retrieved search entries to return into the prompt. Omit to use dataset-specific default.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top_k_sampling",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8196,
        help="Max tokens per generation call. Omit to use model/dataset heuristic.",
    )
    parser.add_argument(
        "--search_api_key",
        type=str,
        default=None,
        help="Search API key; defaults to SEARCH_API_KEY or the selected provider API key environment variable.",
    )
    parser.add_argument(
        "--search_provider",
        type=str,
        default="exa",
        choices=["tavily", "exa"],
        help="Search backend provider.",
    )
    parser.add_argument(
        "--bing_endpoint",
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Unused Bing Search API endpoint.",
    )
    parser.add_argument(
        "--trajectory_jsonl",
        type=str,
        default="./data/sft/rag_agent_wo_url.jsonl",
        help="If set, append JSONL per retrieval step + one final row with final_answer.",
    )
    parser.add_argument(
        "--evaluate_trajectory",
        action="store_true",
        help="Evaluate final trajectory rows and write acc/evaluate fields into them.",
    )
    parser.add_argument(
        "--target_acc_count",
        type=int,
        default=None,
        help="Stop after collecting this many evaluated final rows with acc=1.",
    )
    parser.add_argument(
        "--positive_trajectory_jsonl",
        type=str,
        default=None,
        help="If set, write full trajectories only for examples whose final acc=1.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size. Default: number of visible CUDA devices.",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip benchmark evaluation at the end.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.80,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=8196,
        help="vLLM scheduler/chunked-prefill batch cap.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of examples to process.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index of examples to process, exclusive.",
    )
    parser.add_argument("--context_compressor", choices=["llmsummary"], default="llmsummary")
    parser.add_argument("--summary_model_name", required=True)
    parser.add_argument("--summary_api_base", required=True)
    parser.add_argument("--summary_max_tokens", type=int, default=1024)
    parser.add_argument("--summary_temperature", type=float, default=0)
    parser.add_argument("--summary_timeout", type=float, default=600)
    parser.add_argument("--summary_workers", type=int, default=16)
    parser.add_argument("--summary_max_input_chars", type=int, default=24000)
    parser.add_argument("--resume_completed", default=None, help="Immutable JSONL containing completed trajectories to carry over")
    return resolve_search_args(parser, parser.parse_args())


def main():
    args = parse_args()
    progress("loading")
    summarizer = LLMSummarySearchSummarizer(args.summary_model_name, api_base=args.summary_api_base,
        max_tokens=args.summary_max_tokens, temperature=args.summary_temperature,
        timeout=args.summary_timeout, max_workers=args.summary_workers,
        max_input_chars=args.summary_max_input_chars)

    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    batch_size = args.batch_size
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    search_provider = args.search_provider
    bing_subscription_key = args.search_api_key
    bing_endpoint = args.bing_endpoint
    trajectory_jsonl_path = args.trajectory_jsonl
    positive_trajectory_jsonl_path = args.positive_trajectory_jsonl
    evaluate_trajectory = (
        args.evaluate_trajectory
        or args.target_acc_count is not None
        or positive_trajectory_jsonl_path is not None
    )
    tensor_parallel_size = args.tensor_parallel_size
    skip_evaluation = args.skip_evaluation
    gpu_memory_utilization = args.gpu_memory_utilization
    max_num_batched_tokens = args.max_num_batched_tokens

    ds_def = _dataset_inference_defaults(dataset_name)
    max_search_limit = _resolve_int(args.max_search_limit, ds_def["max_search_limit"])
    max_turn = _resolve_int(args.max_turn, ds_def["max_turn"])
    top_k = _resolve_int(args.top_k, ds_def["top_k"])

    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif "qwq" in model_path.lower():
        if dataset_name in ["aime", "amc", "livecode"]:
            max_tokens = 32768
        else:
            max_tokens = 20480
    else:
        max_tokens = 8196

    gpu_mu = 0.95 if gpu_memory_utilization is None else gpu_memory_utilization

    trajectory_logging_enabled = bool(trajectory_jsonl_path or positive_trajectory_jsonl_path)
    resume_rows = []
    completed_indices = set()
    if args.resume_completed:
        with open(args.resume_completed, encoding="utf-8") as seed_file:
            resume_rows = [json.loads(line) for line in seed_file if line.strip()]
        for row in resume_rows:
            if row.get("record_type") == "final":
                idx = row["example_idx"]
                if idx in completed_indices: raise ValueError(f"Duplicate completed question {idx}")
                if idx < args.start_idx or (args.end_idx is not None and idx >= args.end_idx):
                    raise ValueError(f"Resume question outside shard: {idx}")
                completed_indices.add(idx)
        if any(row["example_idx"] not in completed_indices for row in resume_rows):
            raise ValueError("Resume file contains an incomplete trajectory")

    if trajectory_jsonl_path:
        traj_abs = os.path.abspath(trajectory_jsonl_path)
        os.makedirs(os.path.dirname(traj_abs) or ".", exist_ok=True)
        with open(traj_abs, "x", encoding="utf-8") as seed_out:
            for row in resume_rows: seed_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Carried over {len(completed_indices)} completed questions; they will not be generated again.")
        print(f"Trajectory logging enabled: {traj_abs}")

    if positive_trajectory_jsonl_path:
        positive_traj_abs = os.path.abspath(positive_trajectory_jsonl_path)
        os.makedirs(os.path.dirname(positive_traj_abs) or ".", exist_ok=True)
        open(positive_traj_abs, "w", encoding="utf-8").close()
        print(f"Positive trajectory logging enabled: {positive_traj_abs}")

    def append_trajectory_record(record: Dict) -> None:
        if not trajectory_jsonl_path:
            return
        with open(os.path.abspath(trajectory_jsonl_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_positive_trajectory_record(record: Dict) -> None:
        if not positive_trajectory_jsonl_path:
            return
        with open(os.path.abspath(positive_trajectory_jsonl_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_labeled_answer_and_mode(item: Dict):
        if dataset_name in ["gpqa", "medmcqa"]:
            return item["Correct Choice"], "choose"
        if dataset_name in ["math500", "aime", "amc"]:
            return item["answer"], "gen"
        if dataset_name in ["nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki"]:
            return item["answer"], "qa"
        if dataset_name in ["pubhealth"]:
            return item["answer"], "choose"
        raise ValueError(f"Unknown dataset_name for trajectory evaluation: {dataset_name}")

    def contains_402_status(value) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_str = str(key).lower()
                if key_str in {"status", "status_code", "code", "error_code"} and str(nested) == "402":
                    return True
                if contains_402_status(nested):
                    return True
        elif isinstance(value, list):
            return any(contains_402_status(item) for item in value)
        return False

    positive_acc_count = 0

    def build_final_trajectory_record(seq: Dict, terminated_by: str) -> Dict:
        record = {
            "record_type": "final",
            "instance_id": f"{dataset_name}_{split}_{seq['idx']}_final",
            "dataset_name": dataset_name,
            "split": split,
            "example_idx": seq["idx"],
            "question": seq["item"]["Question"],
            "current_reasoning_prefix": None,
            "search_query": None,
            "retrieved_docs": None,
            "next_reasoning_continuation": None,
            "final_answer": seq["output"],
            "terminated_by": terminated_by,
            "context_compressor": "llmsummary",
            "search_provider": search_provider,
            "scaffold": "search_o1_plain_continuation",
            "uses_chat_template": False,
            "initial_prompt": seq["input_prompt"],
            "assistant_output": "\n".join(seq["assistant_outputs"]),
            "token_usage": finish_usage(seq["token_usage"]),
            "generation_usage": seq["generation_usage"],
            "search_count": seq["search_count"],
            "elapsed_seconds": time.monotonic()-seq["_started_monotonic"],
            "started_at": seq["_started_at"],
            "completed_at": time.time(),
        }
        if evaluate_trajectory:
            labeled_answer, mode = get_labeled_answer_and_mode(seq["item"])
            metric, pred_answer = evaluate_predictions(
                output="\n".join(seq["assistant_outputs"]),
                labeled_answer=labeled_answer,
                mode=mode,
            )
            record.update({
                "evaluate": bool(metric["acc"]),
                "acc": int(metric["acc"]),
                "em": int(metric["em"]),
                "f1": float(metric["f1"]),
                # "math_equal": metric["math_equal"],
                "is_valid_answer": bool(metric["is_valid_answer"]),
                "pred_answer": pred_answer,
                "gold_answer": labeled_answer,
            })
        return record

    def write_final_trajectory(seq: Dict, terminated_by: str) -> None:
        nonlocal positive_acc_count
        if not trajectory_logging_enabled or seq.get("_trajectory_final_written"):
            return

        record = build_final_trajectory_record(seq, terminated_by)
        append_trajectory_record(record)
        seq.setdefault("_trajectory_records", []).append(record)
        seq["_trajectory_final_written"] = True
        progress("sampling", latest_completed_index=seq["idx"])

        if record.get("acc") == 1 and (
            args.target_acc_count is None or positive_acc_count < args.target_acc_count
        ):
            positive_acc_count += 1
            if positive_trajectory_jsonl_path and not seq.get("_positive_trajectory_written"):
                for traj_record in seq.get("_trajectory_records", []):
                    append_positive_trajectory_record(traj_record)
                seq["_positive_trajectory_written"] = True
            if args.target_acc_count is not None:
                print(f"Collected acc=1 trajectories: {positive_acc_count}/{args.target_acc_count}")

    print(
        f"Inference limits: max_search_limit={max_search_limit}, "
        f"max_turn={max_turn}, top_k={top_k}, max_tokens={max_tokens}, "
        f"batch_size={batch_size or 'all'} "
        f"(CLI overrides: search={args.max_search_limit}, turn={args.max_turn}, "
        f"top_k={args.top_k}, max_tokens={args.max_tokens}, batch_size={args.batch_size})"
    )

    if repetition_penalty is None:
        repetition_penalty = 1.05 if "qwq" in model_path.lower() else 1.0

    if dataset_name == "livecode":
        data_path = f"./data/LiveCodeBench/{split}.json"
    elif dataset_name in ["math500", "gpqa", "aime", "amc"]:
        data_path = f"./data/{dataset_name.upper()}/{split}.json"
    else:
        data_path = f"./data/QA_Datasets/{dataset_name}_{split}.json"
    print(f"QA data from {data_path} is already!")

    cache_dir = "./cache"
    search_cache_path = os.path.join(cache_dir, "search_cache.json")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(search_cache_path):
        with open(search_cache_path, "r", encoding="utf-8") as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    def save_caches():
        with open(search_cache_path, "w", encoding="utf-8") as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if "qwq" in model_path.lower():
        if dataset_name in ["math500", "gpqa", "aime", "amc", "livecode"]:
            output_dir = f"./outputs/{dataset_name}.qwq.rag_agent_wo_url"
        else:
            output_dir = f"./outputs/runs.qa/{dataset_name}.qwq.rag_agent_wo_url"
    else:
        model_short_name = model_path.split("/")[-1].lower().replace("-instruct", "")
        output_dir = f"./outputs/runs.baselines/{dataset_name}.{model_short_name}.rag_agent_wo_url"
    os.makedirs(output_dir, exist_ok=True)

    tp = tensor_parallel_size if tensor_parallel_size is not None else torch.cuda.device_count()
    if tp < 1:
        tp = 1
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mu,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=16384, max_num_seqs=32, enforce_eager=True,
        language_model_only=True, dtype="bfloat16", seed=42, generation_config="vllm",
    )

    with open(data_path, "r", encoding="utf-8") as json_file:
        filtered_data = json.load(json_file)

    input_list = []
    for item in filtered_data:
        question = item["Question"]
        if dataset_name in ["nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki"]:
            if dataset_name in ["nq", "triviaqa"]:
                instruction = get_singleqa_search_o1_instruction(max_search_limit)
            else:
                instruction = get_multiqa_search_o1_instruction(max_search_limit)
            if "qwq" in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name="qwq")
            else:
                user_prompt = get_task_instruction_openqa(question)
        elif dataset_name in ["math500", "aime", "amc"]:
            instruction = get_math_search_o1_instruction(max_search_limit)
            if "qwq" in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name="qwq")
            else:
                user_prompt = get_task_instruction_math(question)
        elif dataset_name == "gpqa":
            instruction = get_gpqa_search_o1_instruction(max_search_limit)
            if "qwq" in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name="qwq")
            elif "llama" in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name="llama")
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        elif dataset_name == "livecode":
            instruction = get_code_search_o1_instruction(max_search_limit)
            question_title = item.get("question_title", "")
            if "qwq" in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name="qwq")
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""
            instruction = ""

        prompt = instruction + user_prompt
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    start_idx = args.start_idx
    end_idx = args.end_idx
    input_list = input_list[start_idx:end_idx]
    filtered_data = filtered_data[start_idx:end_idx]

    active_sequences = [{
        "item": item,
        "input_prompt": prompt,
        "prompt": prompt,
        "output": "",
        "finished": False,
        "history": [],
        "pending_operations": [],
        "executed_search_queries": set(),
        "search_count": 0,
        "gen_round": 0,
        "search_limit_stall_turns": 0,
        "idx": i,
        "traj_step": 0,
        "_traj_pending_records": [],
        "_trajectory_records": [],
        "_trajectory_final_written": False,
        "token_usage": new_usage(),
        "generation_usage": [],
        "assistant_outputs": [],
        "_positive_trajectory_written": False,
    } for i, (item, prompt) in enumerate(zip(filtered_data, input_list), start=start_idx)]

    active_sequences = [seq for seq in active_sequences if seq["idx"] not in completed_indices]

    if batch_size is None or batch_size <= 0:
        batch_size = len(active_sequences) if active_sequences else 1

    def run_generation(sequences, max_tokens_):
        prompts = [s["prompt"] for s in sequences]
        sampling_params = SamplingParams(
            max_tokens=max_tokens_,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            repetition_penalty=repetition_penalty,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        progress("generating", active_questions=len(sequences))
        return llm.generate(prompts, sampling_params=sampling_params)

    def process_sequence_batch(batch_sequences):
        for seq in batch_sequences:
            seq["_started_monotonic"] = time.monotonic()
            seq["_started_at"] = time.time()
        marker=Path("sampling_started_at.json")
        if not marker.exists(): marker.write_text(json.dumps({"started_at":time.time()}))
        main_loop_iter = 0
        last_cache_size_at_save = -1

        while True:
            sequences_with_pending_ops = [
                seq for seq in batch_sequences if not seq["finished"] and seq["pending_operations"]
            ]
            sequences_needing_generation = [
                seq
                for seq in batch_sequences
                if not seq["finished"] and not seq["pending_operations"] and seq["gen_round"] < max_turn
            ]

            if sequences_with_pending_ops:
                print(f"{len(sequences_with_pending_ops)} sequences have pending operations. Executing...")
                pending_observations = []
                for seq in sequences_with_pending_ops:
                    operation = seq["pending_operations"].pop(0)
                    if operation["type"] != "search":
                        continue

                    query = operation["content"]
                    cache_key = f"{search_provider}::web::{query}"
                    reasoning_prefix = seq["output"] if trajectory_logging_enabled else None

                    if cache_key in search_cache:
                        results = search_cache[cache_key]
                        print(f"Using cached search results for query: {query}")
                    else:
                        try:
                            results = bing_web_search(
                                query,
                                bing_subscription_key,
                                bing_endpoint,
                                market="en-US",
                                language="en",
                                provider=search_provider,
                            )
                            search_cache[cache_key] = results
                            print(f"Executed and cached search for query: {query}")
                            if contains_402_status(results):
                                raise SearchPaymentRequiredError("402ERROR")
                        except SearchPaymentRequiredError:
                            raise SystemExit(42)
                        except Exception as e:
                            print(f"Error during search query '{query}': {e}")
                            search_cache[cache_key] = {}
                            results = {}

                    relevant_info = extract_relevant_info(results)[:top_k]
                    search_result_str = json.dumps(relevant_info, ensure_ascii=False, indent=2)
                    pending_observations.append((seq,query,reasoning_prefix,relevant_info,search_result_str))
                summary_results=summarizer.summarize_batch([
                    dict(documents=docs,question=seq["item"]["Question"],search_query=query)
                    for seq,query,reasoning_prefix,docs,raw in pending_observations])
                for (seq,query,reasoning_prefix,relevant_info,search_result_str),summary_result in zip(pending_observations,summary_results):
                    summary_metadata=summary_result.metadata()
                    add_summary_usage(seq["token_usage"],summary_metadata)
                    summary_metadata.update(agent_original_tokens=len(tokenizer.encode(search_result_str,add_special_tokens=False)),
                        agent_summary_tokens=len(tokenizer.encode(summary_result.summary_text,add_special_tokens=False)))
                    append_text = f"\n{BEGIN_SEARCH_RESULT}\n{summary_result.summary_text}\n{END_SEARCH_RESULT}\n"
                    print(f"LLMSummary summarized search evidence: {summary_metadata['agent_original_tokens']} -> {summary_metadata['agent_summary_tokens']} tokens (plain completion)")

                    if trajectory_logging_enabled:
                        seq["traj_step"] += 1
                        seq.setdefault("_traj_pending_records", []).append({
                            "record_type": "retrieval_step",
                            "trajectory_kind": "web_search",
                            "instance_id": f"{dataset_name}_{split}_{seq['idx']}_t{seq['traj_step']}",
                            "dataset_name": dataset_name,
                            "split": split,
                            "example_idx": seq["idx"],
                            "question": seq["item"]["Question"],
                            "current_reasoning_prefix": reasoning_prefix,
                            "search_query": query,
                            "search_provider": search_provider,
                            "retrieved_docs": search_result_str,
                            "summarized_retrieved_docs": summary_result.summary_text,
                            "summary": summary_metadata,
                            "summary_input_prompt": SUMMARY_SYSTEM_PROMPT + "\n\n" + summarizer._build_user_prompt(relevant_info,question=seq["item"]["Question"],search_query=query)[0] + SUMMARY_COMPLETION_PREFILL,
                            "observation_processor": "llmsummary",
                            "uses_chat_template": False,
                            "next_reasoning_continuation": None,
                            "final_answer": None,
                        })

                    seq["prompt"] += append_text
                    seq["output"] += append_text
                    seq["history"].append(append_text)
                    seq["search_count"] += 1

            if sequences_with_pending_ops:
                continue

            if sequences_needing_generation:
                for seq in sequences_needing_generation:
                    seq["gen_round"] += 1
                gr_max = max(s["gen_round"] for s in sequences_needing_generation)
                print(
                    f"Generation round (per-sequence max gen_round={gr_max}/{max_turn}): "
                    f"{len(sequences_needing_generation)} sequences. Generating with LLM..."
                )
                outputs = run_generation(sequences_needing_generation, max_tokens)
                print("Generation complete. Processing outputs...")

                for seq, out in zip(sequences_needing_generation, outputs):
                    add_agent_usage(seq["token_usage"],out)
                    text = out.outputs[0].text
                    seq["generation_usage"].append(dict(input_tokens=len(out.prompt_token_ids),output_tokens=len(out.outputs[0].token_ids),
                        finish_reason=out.outputs[0].finish_reason,prompt_text=seq["prompt"],output_text=text))
                    seq["assistant_outputs"].append(text)

                    if trajectory_logging_enabled:
                        pending = seq.pop("_traj_pending_records", None) or []
                        for rec in pending:
                            rec["next_reasoning_continuation"] = text
                            append_trajectory_record(rec)
                            seq.setdefault("_trajectory_records", []).append(rec)

                    seq["_traj_pending_records"] = []
                    seq["history"].append(text)
                    seq["prompt"] += text
                    seq["output"] += text

                    search_queries = extract_all_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                    pending_search_ops = sum(1 for p in seq["pending_operations"] if p["type"] == "search")
                    for search_query in search_queries:
                        if seq["search_count"] + pending_search_ops >= max_search_limit:
                            if search_query not in seq["executed_search_queries"]:
                                limit_message = (
                                    f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. "
                                    f"You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                                )
                                seq["prompt"] += limit_message
                                seq["output"] += limit_message
                                seq["history"].append(limit_message)
                                print(
                                    f"Search limit exceeded for query: {search_query}; "
                                    "limit message injected (sequence not finished, model may continue)."
                                )
                            else:
                                print(f"Search query already executed: {search_query}")
                            continue

                        if search_query not in seq["executed_search_queries"]:
                            seq["pending_operations"].append({"type": "search", "content": search_query})
                            seq["executed_search_queries"].add(search_query)
                            pending_search_ops += 1
                            print(f"Added pending search operation for query: {search_query}")
                        else:
                            print(f"Search query already executed: {search_query}")

                    at_search_limit = seq["search_count"] >= max_search_limit
                    no_pending_after = len(seq["pending_operations"]) == 0
                    if search_queries and at_search_limit and no_pending_after:
                        seq["search_limit_stall_turns"] += 1
                    else:
                        seq["search_limit_stall_turns"] = 0

                    if not search_queries:
                        seq["finished"] = True
                        print("Sequence marked as finished (no search tags in this generation).")
                        write_final_trajectory(seq, "max_new_tokens" if out.outputs[0].finish_reason=="length" else "natural_stop")
                    elif seq["search_limit_stall_turns"] >= _SEARCH_LIMIT_STALL_MAX:
                        seq["finished"] = True
                        seq["pending_operations"] = []
                        print(
                            f"Sequence idx={seq['idx']} stopped: search quota exhausted but model kept emitting "
                            f"search tags ({_SEARCH_LIMIT_STALL_MAX} consecutive rounds with no new retrieval queued)."
                        )
                        write_final_trajectory(seq, "search_limit_stalemate")
                    elif seq["gen_round"] >= max_turn:
                        seq["finished"] = True
                        seq["pending_operations"] = []
                        print(
                            f"Sequence idx={seq['idx']} stopped: reached max_turn ({max_turn}) for this example; "
                            "other sequences continue."
                        )
                        write_final_trajectory(seq, "max_turn")

            unfinished = [seq for seq in batch_sequences if not seq["finished"]]
            if not unfinished:
                break

            main_loop_iter += 1
            if main_loop_iter % 30 == 0:
                save_caches()
                print(f"Periodic cache save (main-loop iter {main_loop_iter}).")
            if len(search_cache) > 0 and len(search_cache) % 50 == 0 and len(search_cache) != last_cache_size_at_save:
                save_caches()
                last_cache_size_at_save = len(search_cache)
                print(f"Periodic cache save (search cache entries={len(search_cache)}).")

    start_time = time.time()
    processed_sequences = []
    total_batches = (len(active_sequences) + batch_size - 1) // batch_size if active_sequences else 0
    for batch_idx, batch_start in enumerate(range(0, len(active_sequences), batch_size), start=1):
        batch_sequences = active_sequences[batch_start:batch_start + batch_size]
        batch_end = batch_start + len(batch_sequences) - 1
        print(
            f"Processing batch {batch_idx}/{total_batches}: "
            f"example indices {batch_start}-{batch_end} ({len(batch_sequences)} sequences)"
        )
        process_sequence_batch(batch_sequences)
        processed_sequences.extend(batch_sequences)
        if args.target_acc_count is not None and positive_acc_count >= args.target_acc_count:
            print(f"Target reached: collected {positive_acc_count} acc=1 trajectories.")
            break

    if trajectory_logging_enabled:
        for seq in processed_sequences:
            if seq.get("_trajectory_final_written"):
                continue
            write_final_trajectory(seq, "natural_stop" if seq["finished"] else "incomplete")

    total_time = time.time() - start_time
    eval_sequences = processed_sequences
    filtered_data_for_eval = [seq["item"] for seq in eval_sequences]
    input_list_for_eval = [seq["input_prompt"] for seq in eval_sequences]
    output_list = ["\n".join(seq["assistant_outputs"]) for seq in eval_sequences]

    if not skip_evaluation:
        print("Evaluating generated answers...")
        run_evaluation(
            filtered_data=filtered_data_for_eval,
            input_list=input_list_for_eval,
            output_list=output_list,
            dataset_name=dataset_name,
            output_dir=output_dir,
            total_time=total_time,
            split=split,
        )
    else:
        print("Skipping evaluation (--skip_evaluation).")

    print("Updating search cache...")
    if os.path.exists(search_cache_path):
        with open(search_cache_path, "r", encoding="utf-8") as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}
    search_cache.update(search_cache_new)
    save_caches()

    progress("complete")
    print("Process completed.")


if __name__ == "__main__":
    main()
