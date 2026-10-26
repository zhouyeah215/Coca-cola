#!/usr/bin/env python3
"""Launch the existing Search-o1 baseline in an isolated output directory."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["vanilla", "llmsummary", "llmlingua"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/QA_Datasets")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    reserved = ("--trajectory_jsonl", "--context_compressor", "--evaluate_trajectory", "--skip_evaluation")
    if any(token.split("=", 1)[0] in reserved for token in extra):
        parser.error("The launcher manages trajectories, method selection and evaluation flags.")
    run_dir = args.run_dir.expanduser().resolve()
    script = "run_llmsummary.py" if args.method == "llmsummary" else "run_no_compress.py"
    method = {"vanilla": "none", "llmsummary": "llmsummary", "llmlingua": "longllmlingua"}[args.method]
    command = [sys.executable, "-u", str(ROOT / "scripts/baselines" / script),
               "--context_compressor", method, "--trajectory_jsonl", str(run_dir / "trajectories.jsonl"),
               "--evaluate_trajectory", "--skip_evaluation", *extra]
    # The default backend is Exa; credentials are normally inherited via the environment.
    # Never serialize credentials explicitly passed on the command line.
    redacted = list(command)
    for i, token in enumerate(redacted):
        if token == "--search_api_key" and i + 1 < len(redacted):
            redacted[i + 1] = "<redacted>"
        elif token.startswith("--search_api_key="):
            redacted[i] = "--search_api_key=<redacted>"
    print(shlex.join(redacted), flush=True)
    print(f"Working directory: {run_dir}", flush=True)
    if args.dry_run:
        return
    data_dir = args.data_dir.expanduser().resolve(strict=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data").mkdir()
    (run_dir / "data/QA_Datasets").symlink_to(data_dir, target_is_directory=True)
    (run_dir / "command.json").write_text(json.dumps(redacted, indent=2) + "\n")
    env = dict(os.environ)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(command, cwd=run_dir, env=env)
    (run_dir / "exit_code").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
