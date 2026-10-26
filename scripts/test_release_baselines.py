"""Offline integration checks for the actual Search-o1 baseline loops."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parent / "baselines"
sys.path.insert(0, str(CODE))
# Load the real retriever formatting/scoring dependencies before mocking GPU libraries.
import web_search
import evaluate


class BaselineTests(unittest.TestCase):
    def run_method(self, method, provider="exa"):
        calls, summary_requests, search_requests = [], [], []

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"

            def encode(self, text, **kwargs):
                return list(text.encode())

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("Search-o1 must not call a chat template")

        class Model:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, sampling_params):
                calls.append(list(prompts))
                result = []
                for prompt in prompts:
                    if "Second question" in prompt:
                        text = r"\boxed{Paris}"
                    elif len(calls) > 1:
                        text = r"\boxed{Lyon}"
                    else:
                        text = "<|begin_search_query|>capital of France<|end_search_query|>"
                    result.append(NS(prompt_token_ids=list(prompt.encode()), outputs=[
                        NS(text=text, token_ids=list(text.encode()), finish_reason="stop")]))
                return result

        class Compressor:
            def __init__(self, *args, **kwargs):
                pass

            def compress(self, docs, question):
                self_question = question
                assert self_question == "First question"
                return NS(compressed_text="Compressed Paris evidence", metadata=lambda: {"method": "longllmlingua"})

        def summarize(_, payload):
            self.assertNotIn("messages", payload)
            self.assertIsInstance(payload["prompt"], str)
            summary_requests.append(payload)
            return {"choices": [{"text": "Summary Paris evidence", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 5}}

        class ExaClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def search_and_contents(self, query, **kwargs):
                search_requests.append(("exa", self.api_key, query, kwargs))
                return NS(results=[NS(title="France", text="Paris is the capital of France.",
                                      url="https://example.test/france", highlights=["Paris"], score=1)])

        class TavilyClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def search(self, query, **kwargs):
                search_requests.append(("tavily", self.api_key, query, kwargs))
                return {"results": [{"title": "France", "content": "Paris is the capital of France.",
                                     "url": "https://example.test/france"}]}

        mocks = {
            "exa_py": NS(Exa=ExaClient),
            "tavily": NS(TavilyClient=TavilyClient),
            "torch": NS(cuda=NS(device_count=lambda: 1)),
            "transformers": NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: Tokenizer())),
            "vllm": NS(LLM=Model, SamplingParams=lambda **kw: NS(**kw)),
            "LLMLingua.web_search_adapter": NS(LongLLMLinguaSearchCompressor=Compressor),
        }
        from LLMSummary.web_search_adapter import LLMSummarySearchSummarizer
        filename = "run_llmsummary.py" if method == "llmsummary" else "run_no_compress.py"
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, mocks), \
                patch.object(LLMSummarySearchSummarizer, "_post_completion", summarize), \
                patch.dict(os.environ, {"SEARCH_API_KEY": "", "EXA_API_KEY": "test-exa-key", "TAVILY_API_KEY": "test-tavily-key"}):
            spec = importlib.util.spec_from_file_location("release_agent", CODE / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                os.chdir(tmp)
                Path("data/QA_Datasets").mkdir(parents=True)
                Path("data/QA_Datasets/nq_test.json").write_text(json.dumps([
                    {"Question": "First question", "answer": ["Paris"]},
                    {"Question": "Second question", "answer": ["Paris"]}]))
                argv = [filename, "--dataset_name", "nq", "--split", "test", "--model_path", "dummy",
                        "--context_compressor", method, "--batch_size", "2", "--max_turn", "3",
                        "--trajectory_jsonl", "rows.jsonl", "--evaluate_trajectory", "--skip_evaluation"]
                if provider != "exa":
                    argv += ["--search_provider", provider]
                if method == "llmsummary":
                    argv += ["--summary_model_name", "dummy", "--summary_api_base", "http://unused/v1"]
                elif method == "longllmlingua":
                    argv += ["--compression_model_name", "dummy"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                rows = [json.loads(line) for line in Path("rows.jsonl").read_text().splitlines()]
                finals = {r["example_idx"]: r for r in rows if r["record_type"] == "final"}
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(finals), 2)
                self.assertEqual(finals[0]["em"], 0, "Retrieved gold answer leaked into scoring")
                self.assertEqual(finals[1]["em"], 1)
                retrieval = next(r for r in rows if r["record_type"] == "retrieval_step")
                self.assertIn("Paris is the capital of France", retrieval["retrieved_docs"])
                self.assertEqual(retrieval["search_provider"], provider)
                self.assertEqual(finals[0]["search_provider"], provider)
                self.assertNotIn("test-exa-key", Path("rows.jsonl").read_text())
                self.assertNotIn("test-tavily-key", Path("rows.jsonl").read_text())
                self.assertEqual(len(search_requests), 1)
                actual_provider, key, query, options = search_requests[0]
                self.assertEqual(actual_provider, provider)
                self.assertEqual(key, f"test-{provider}-key")
                self.assertEqual(query, "capital of France")
                if provider == "exa":
                    self.assertEqual(options["num_results"], 20)
                    self.assertEqual(options["text"], {"max_characters": 1200})
                else:
                    self.assertEqual(options["max_results"], 20)
                    self.assertEqual(options["search_depth"], "advanced")
                observation = {"none": retrieval["retrieved_docs"], "longllmlingua": "Compressed Paris evidence",
                               "llmsummary": "Summary Paris evidence"}[method]
                expected = (calls[0][0] + "<|begin_search_query|>capital of France<|end_search_query|>"
                            + "\n<|begin_search_result|>\n" + observation + "\n<|end_search_result|>\n")
                self.assertEqual(calls[1][0], expected)
                self.assertNotIn("<tool_response>", expected)
                self.assertEqual(finals[0]["token_usage"]["agent_input_tokens"],
                                 len(calls[0][0].encode()) + len(expected.encode()))
                self.assertEqual(len(summary_requests), int(method == "llmsummary"))
                self.assertEqual(json.loads(Path("progress.json").read_text())["phase"], "complete")
            finally:
                os.chdir(old_cwd)

    def test_vanilla(self):
        for provider in ("exa", "tavily"):
            with self.subTest(provider=provider):
                self.run_method("none", provider)

    def test_llmsummary(self):
        for provider in ("exa", "tavily"):
            with self.subTest(provider=provider):
                self.run_method("llmsummary", provider)

    def test_llmlingua(self):
        for provider in ("exa", "tavily"):
            with self.subTest(provider=provider):
                self.run_method("longllmlingua", provider)


class SearchConfigTests(unittest.TestCase):
    def test_missing_credentials_fail(self):
        from search_config import resolve_search_args
        with patch.dict(os.environ, {}, clear=True):
            for provider in ("exa", "tavily"):
                with self.subTest(provider=provider), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    resolve_search_args(argparse.ArgumentParser(), NS(search_provider=provider, search_api_key=None))

    def test_credential_precedence_and_provider_isolation(self):
        from search_config import resolve_search_args
        env = {"SEARCH_API_KEY": "generic", "EXA_API_KEY": "exa-only", "TAVILY_API_KEY": "tavily-only"}
        with patch.dict(os.environ, env, clear=True):
            for provider in ("exa", "tavily"):
                for explicit, expected in (("explicit", "explicit"), (None, "generic")):
                    args = NS(search_provider=provider, search_api_key=explicit)
                    self.assertEqual(resolve_search_args(argparse.ArgumentParser(), args).search_api_key, expected)
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tavily-only"}, clear=True):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resolve_search_args(argparse.ArgumentParser(), NS(search_provider="exa", search_api_key=None))


if __name__ == "__main__":
    unittest.main()
