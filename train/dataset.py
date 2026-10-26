from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

SECTION_TOOL_OUTPUT_HEADERS = (
    "## Original Web Search Results",
    "## Original Tool Output",
)
SECTION_BOUNDARY_HEADERS = (
    "## Compression Requirement",
    "## Instructions",
)

def _pad_1d(t: torch.Tensor, target_len: int, pad_value: int | float) -> torch.Tensor:
    if t.size(0) >= target_len:
        return t[:target_len]
    pad = torch.full((target_len - t.size(0),), pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)


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


def _fallback_student_segments(prompt: str) -> list[str]:
    for header in SECTION_TOOL_OUTPUT_HEADERS:
        if header in prompt:
            head, _, _tail = prompt.partition(header)
            return [head + header + "\n", "\n"]
    return [prompt, ""]


def _reconstruct_teacher_prompt(
    _tokenizer: PreTrainedTokenizerBase,
    _row: dict[str, Any],
    _prompt: str,
    legacy_agent_prompt_full: str | None,
) -> str | None:
    base_prompt = str(legacy_agent_prompt_full or "").strip()
    return base_prompt or None


def latent_collate_fn(
    batch: list[dict[str, Any]],
    agent_pad_token_id: int = 0,
    compressor_pad_token_id: int = 0,
    max_length: int | None = None,
) -> dict[str, Any]:
    if not batch:
        return {}

    max_len_full = max(item["agent_input_full"].size(0) for item in batch)
    max_comp_len = max(item["compressor_input_ids"].size(0) for item in batch)
    if max_length is not None and max_length > 0:
        max_len_full = min(max_len_full, max_length)
        max_comp_len = min(max_comp_len, max_length)

    out: dict[str, Any] = {}
    out["compressor_input_ids"] = torch.stack(
        [_pad_1d(b["compressor_input_ids"], max_comp_len, compressor_pad_token_id) for b in batch],
        dim=0,
    )
    out["compressor_attention_mask"] = torch.stack(
        [_pad_1d(b["compressor_attention_mask"], max_comp_len, 0) for b in batch],
        dim=0,
    )
    out["agent_input_full"] = torch.stack(
        [_pad_1d(b["agent_input_full"], max_len_full, agent_pad_token_id) for b in batch],
        dim=0,
    )
    out["agent_attention_full"] = torch.stack(
        [_pad_1d(b["agent_attention_full"], max_len_full, 0) for b in batch],
        dim=0,
    )
    out["labels"] = torch.stack(
        [_pad_1d(b["labels"], max_len_full, -100) for b in batch],
        dim=0,
    )
    out["student_segment_ids"] = [b["student_segment_ids"] for b in batch]
    out["completion_ids"] = [b["completion_ids"] for b in batch]
    return out


@dataclass
class LatentSample:
    compressor_input: str
    agent_prompt_full: str
    student_segment_texts: list[str]
    completion: str


class LatentSFTDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        compressor_tokenizer: PreTrainedTokenizerBase,
        agent_tokenizer: PreTrainedTokenizerBase,
        num_latent_tokens: int,
        compressor_max_length: int = 4096,
        agent_max_length: int = 8192,
    ) -> None:
        self.data_path = Path(data_path)
        self.compressor_tokenizer = compressor_tokenizer
        self.agent_tokenizer = agent_tokenizer
        self.num_latent_tokens = num_latent_tokens
        self.compressor_max_length = compressor_max_length
        self.agent_max_length = agent_max_length
        self.samples: list[LatentSample] = []
        self._load()

    def _truncate_teacher_prompt_and_completion(
        self,
        prompt_ids_full: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if completion_ids.numel() > self.agent_max_length:
            completion_ids = completion_ids[: self.agent_max_length]
        available_prompt = max(0, self.agent_max_length - completion_ids.numel())
        if prompt_ids_full.numel() > available_prompt:
            prompt_ids_full = prompt_ids_full[-available_prompt:] if available_prompt > 0 else prompt_ids_full[:0]
        return prompt_ids_full, completion_ids

    def _truncate_student_segments(
        self,
        student_segment_ids: list[torch.Tensor],
        completion_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if completion_ids.numel() > self.agent_max_length:
            completion_ids = completion_ids[: self.agent_max_length]
        latent_budget = self.num_latent_tokens
        available_text = max(0, self.agent_max_length - completion_ids.numel() - latent_budget)
        segment_ids = [seg.clone() for seg in student_segment_ids]
        total_text = sum(seg.numel() for seg in segment_ids)
        trim = max(0, total_text - available_text)
        seg_idx = 0
        while trim > 0 and seg_idx < len(segment_ids):
            seg = segment_ids[seg_idx]
            if seg.numel() <= trim:
                trim -= seg.numel()
                segment_ids[seg_idx] = seg[:0]
            else:
                segment_ids[seg_idx] = seg[trim:]
                trim = 0
            seg_idx += 1
        return segment_ids, completion_ids

    def _load(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Latent SFT data not found: {self.data_path}")
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", "")
                completion = row.get("completion", "")
                if not prompt or not completion:
                    continue
                agent_prompt_full = _reconstruct_teacher_prompt(
                    self.agent_tokenizer,
                    row,
                    prompt,
                    row.get("agent_prompt_full"),
                )

                if not agent_prompt_full or not str(agent_prompt_full).strip():
                    continue

                split_prompt = _split_visible_tool_block(agent_prompt_full)
                if split_prompt is not None:
                    student_segment_texts = [split_prompt[0], split_prompt[1]]
                else:
                    student_segment_texts = _fallback_student_segments(prompt)

                self.samples.append(
                    LatentSample(
                        compressor_input=prompt,
                        agent_prompt_full=agent_prompt_full,
                        student_segment_texts=student_segment_texts,
                        completion=completion,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        comp_enc = self.compressor_tokenizer(
            s.compressor_input,
            truncation=True,
            max_length=self.compressor_max_length,
            padding=False,
            return_tensors="pt",
        )

        prompt_ids_full = self.agent_tokenizer(
            s.agent_prompt_full,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        completion_text = s.completion + (self.agent_tokenizer.eos_token or "")
        completion_ids = self.agent_tokenizer(
            completion_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        prompt_ids_full, completion_ids = self._truncate_teacher_prompt_and_completion(
            prompt_ids_full,
            completion_ids,
        )
        if completion_ids.numel() == 0:
            raise IndexError(f"Sample {idx} has empty completion after truncation")

        agent_input_full = torch.cat([prompt_ids_full, completion_ids], dim=0)
        attention_full = torch.ones_like(agent_input_full)
        labels = agent_input_full.clone()
        prompt_len_full = min(len(prompt_ids_full), int(agent_input_full.size(0)))
        labels[:prompt_len_full] = -100

        student_segment_ids = [
            self.agent_tokenizer(seg, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            for seg in s.student_segment_texts
        ]
        student_segment_ids, completion_ids = self._truncate_student_segments(
            student_segment_ids,
            completion_ids,
        )
        if completion_ids.numel() == 0:
            raise IndexError(f"Sample {idx} has empty completion after student truncation")

        return {
            "compressor_input_ids": comp_enc["input_ids"][0],
            "compressor_attention_mask": comp_enc["attention_mask"][0],
            "agent_input_full": agent_input_full,
            "agent_attention_full": attention_full,
            "labels": labels,
            "student_segment_ids": student_segment_ids,
            "completion_ids": completion_ids,
        }
