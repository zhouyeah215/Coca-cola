"""LLM-based observation summarization for Latte's web-search agent.

The summarizer is exposed through a dedicated OpenAI-compatible vLLM server.
Keeping it in a separate process lets the main agent and summarizer use disjoint
GPU sets and makes the baseline easy to reproduce.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUMMARY_SYSTEM_PROMPT = 'Summarize the web-search observation for the question-answering agent. Preserve important information that may help answer the question, together with the relevant supporting evidence. Retain key facts, names, dates, numbers, relationships, and qualifications. Preserve supporting statements from the original text and enough context to explain what they support; retain source or document attribution when available. Do not add information or conclusions that are not supported by the observation.'


SUMMARY_COMPLETION_PREFILL = "\n<think>\n\n</think>\n\n"

@dataclass(frozen=True)
class SummaryResult:
    summary_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    method: str
    model_name: str
    document_count: int
    fallback_used: bool = False
    fallback_mode: str = "none"
    reasoning_completion_tokens: int = 0
    finalization_completion_tokens: int = 0
    finish_reason: str = ""

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("summary_text")
        return data


def format_search_document(document: Mapping[str, Any], position: int) -> str:
    """Render one retrieved document without losing source attribution."""
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


def _visible_answer(text: str) -> str:
    """Drop hidden reasoning if a reasoning-capable server returns it."""
    value = str(text or "").strip()
    if value.startswith("<think>") and "</think>" not in value:
        return ""
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1].strip()
    if value.startswith("<think>"):
        value = value[len("<think>") :].strip()
    return value


class LLMSummarySearchSummarizer:
    """Summarize retrieved observations through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str,
        *,
        api_base: str = "http://127.0.0.1:18081/v1",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_workers: int = 16,
        max_input_chars: int = 24000,
        max_retries: int = 3,
        request_fn: Any | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("LLMSummary requires a served model name")
        if max_tokens <= 0:
            raise ValueError("summary max_tokens must be positive")
        if temperature < 0:
            raise ValueError("summary temperature must be non-negative")
        if timeout <= 0 or max_workers <= 0 or max_input_chars <= 0:
            raise ValueError("timeout, max_workers, and max_input_chars must be positive")

        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.max_workers = int(max_workers)
        self.max_input_chars = int(max_input_chars)
        self.max_retries = max(1, int(max_retries))
        self.request_fn = request_fn or self._post_completion

    @staticmethod
    def _deterministic_evidence_fallback(
        documents: Sequence[Mapping[str, Any]],
        max_chars: int = 6000,
    ) -> str:
        """Return bounded source evidence when the model emits no visible text."""
        rendered = [
            format_search_document(document, position)
            for position, document in enumerate(documents, start=1)
        ]
        value = "\n\n".join(part for part in rendered if part.strip()).strip()
        if not value:
            return "Retrieved documents contained no usable textual evidence."
        if len(value) > max_chars:
            value = value[:max_chars].rstrip() + "\n[Fallback evidence truncated.]"
        return value

    def _post_completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.api_base}/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(
            f"LLMSummary request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _build_user_prompt(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        question: str,
        search_query: str,
    ) -> tuple[str, int]:
        contexts = [
            format_search_document(document, position)
            for position, document in enumerate(documents, start=1)
        ]
        contexts = [context for context in contexts if context.strip()]
        observation = "\n\n".join(contexts)
        if len(observation) > self.max_input_chars:
            observation = observation[: self.max_input_chars].rstrip()
            observation += "\n[Observation truncated to fit the summarizer context window.]"
        prompt = (
            f"Question: {str(question).strip()}\n"
            f"Search query: {str(search_query).strip()}\n\n"
            f"Observation:\n{observation}\n\n"
            "Summary:"
        )
        return prompt, len(contexts)

    def summarize(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        question: str,
        search_query: str,
    ) -> SummaryResult:
        prompt, document_count = self._build_user_prompt(
            documents,
            question=question,
            search_query=search_query,
        )
        if document_count == 0:
            return SummaryResult(
                summary_text="No useful search evidence was retrieved.",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                elapsed_seconds=0.0,
                method="llmsummary",
                model_name=self.model_name,
                document_count=0,
            )

        payload = {
            "model": self.model_name,
            "prompt": SUMMARY_SYSTEM_PROMPT + "\n\n" + prompt + SUMMARY_COMPLETION_PREFILL,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        started_at = time.perf_counter()
        response = self.request_fn(payload)
        try:
            raw_completion = str(response["choices"][0]["text"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed LLMSummary response: {response}") from exc

        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        summary_text = _visible_answer(raw_completion)
        fallback_used = False
        fallback_mode = "none"

        elapsed = time.perf_counter() - started_at
        if not summary_text:
            # The model has already been tried twice. Keep the run alive and
            # preserve attributable evidence instead of aborting the whole
            # sampling batch because of one empty generation.
            summary_text = self._deterministic_evidence_fallback(documents)
            fallback_used = True
            fallback_mode = "deterministic_evidence"

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        total_tokens = prompt_tokens + completion_tokens
        return SummaryResult(
            summary_text=summary_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            method="llmsummary",
            model_name=self.model_name,
            document_count=document_count,
            fallback_used=fallback_used,
            fallback_mode=fallback_mode,
            reasoning_completion_tokens=0,
            finalization_completion_tokens=0,
            finish_reason=str(response["choices"][0].get("finish_reason") or ""),
        )

    def summarize_batch(
        self,
        jobs: Sequence[Mapping[str, Any]],
    ) -> list[SummaryResult]:
        """Summarize independent observations concurrently, preserving order."""
        if not jobs:
            return []
        results: list[SummaryResult | None] = [None] * len(jobs)
        worker_count = min(self.max_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    self.summarize,
                    job["documents"],
                    question=str(job["question"]),
                    search_query=str(job["search_query"]),
                ): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
        return [result for result in results if result is not None]
