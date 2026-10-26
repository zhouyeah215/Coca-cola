"""LongLLMLingua adapter for Latte's iterative web-search agent.

The retriever output remains available in its original JSON form for auditing,
while this module converts the documents to LongLLMLingua contexts and returns
the compressed text that should be sent to the agent as a tool message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch


LLMLINGUA_ROOT = Path(__file__).resolve().parent
if str(LLMLINGUA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLMLINGUA_ROOT))


@dataclass(frozen=True)
class CompressionResult:
    compressed_text: str
    origin_tokens: int
    compressed_tokens: int
    ratio: str
    rate: str
    elapsed_seconds: float
    method: str
    model_name: str
    document_count: int

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("compressed_text")
        return data


def format_search_document(document: Mapping[str, Any], position: int) -> str:
    """Turn a search-result dictionary into a stable compressor context."""
    title = str(document.get("title") or "").strip()
    url = str(document.get("url") or "").strip()
    site_name = str(document.get("site_name") or "").strip()
    evidence = str(document.get("context") or document.get("snippet") or "").strip()

    fields = [f"Document {position}"]
    if title:
        fields.append(f"Title: {title}")
    if site_name:
        fields.append(f"Site: {site_name}")
    if url:
        fields.append(f"URL: {url}")
    if evidence:
        fields.append(f"Evidence: {evidence}")
    return "\n".join(fields)


class LongLLMLinguaSearchCompressor:
    """Question-conditioned LongLLMLingua compression for retrieved documents."""

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "cpu",
        rate: float = 0.5,
        target_token: int = -1,
        dynamic_context_compression_ratio: float = 0.4,
        context_budget: str = "+100",
        local_files_only: bool = False,
        compressor: Any | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("LongLLMLingua requires a compressor model name or path")
        if not 0.0 < rate <= 1.0:
            raise ValueError("LongLLMLingua compression rate must be in (0, 1]")
        if target_token == 0 or target_token < -1:
            raise ValueError("target_token must be -1 or a positive integer")
        if not 0.0 <= dynamic_context_compression_ratio <= 1.0:
            raise ValueError("dynamic_context_compression_ratio must be in [0, 1]")

        self.model_name = model_name
        self.device_map = device_map
        self.rate = float(rate)
        self.target_token = int(target_token)
        self.dynamic_context_compression_ratio = float(
            dynamic_context_compression_ratio
        )
        self.context_budget = context_budget

        if compressor is None:
            from llmlingua import PromptCompressor

            model_config: dict[str, Any] = {
                "local_files_only": bool(local_files_only),
                # Transformers 5 no longer accepts the legacy string value
                # torch_dtype="auto" used by upstream LLMLingua here.  An
                # explicit dtype also halves the 7B compressor's GPU memory.
                "torch_dtype": (
                    torch.bfloat16 if "cuda" in device_map else torch.float32
                ),
            }
            compressor = PromptCompressor(
                model_name=model_name,
                device_map=device_map,
                model_config=model_config,
            )
        self.compressor = compressor

    def compress(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        question: str,
    ) -> CompressionResult:
        contexts = [
            format_search_document(document, position)
            for position, document in enumerate(documents, start=1)
        ]
        contexts = [context for context in contexts if context.strip()]
        if not contexts:
            return CompressionResult(
                compressed_text="No search evidence was retrieved.",
                origin_tokens=0,
                compressed_tokens=0,
                ratio="1.0x",
                rate="100.0%",
                elapsed_seconds=0.0,
                method="longllmlingua",
                model_name=self.model_name,
                document_count=0,
            )

        started_at = time.perf_counter()
        result = self.compressor.compress_prompt(
            contexts,
            question=str(question),
            rate=self.rate,
            target_token=self.target_token,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=self.dynamic_context_compression_ratio,
            condition_compare=True,
            context_budget=self.context_budget,
            rank_method="longllmlingua",
            use_sentence_level_filter=False,
            use_context_level_filter=True,
            use_token_level_filter=True,
            concate_question=False,
        )
        elapsed = time.perf_counter() - started_at
        compressed_text = str(result.get("compressed_prompt") or "").strip()
        if not compressed_text:
            raise RuntimeError("LongLLMLingua returned an empty compressed prompt")

        return CompressionResult(
            compressed_text=compressed_text,
            origin_tokens=int(result.get("origin_tokens") or 0),
            compressed_tokens=int(result.get("compressed_tokens") or 0),
            ratio=str(result.get("ratio") or ""),
            rate=str(result.get("rate") or ""),
            elapsed_seconds=elapsed,
            method="longllmlingua",
            model_name=self.model_name,
            document_count=len(contexts),
        )
