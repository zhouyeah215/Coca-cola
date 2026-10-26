#!/usr/bin/env python3
"""Summarize scored final rows; optionally require exact dataset coverage."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trajectory.read_text().splitlines() if line.strip()]
    finals = [r for r in rows if r.get("record_type") == "final"]
    ids = [(r.get("dataset_name"), r.get("split"), r["example_idx"]) for r in finals]
    if not finals or len(set(ids)) != len(ids):
        raise SystemExit("Missing final rows or duplicate example indices")
    if args.expected_count is not None and len(finals) != args.expected_count:
        raise SystemExit(f"Incomplete evaluation: {len(finals)}/{args.expected_count}")
    result = {"completed": len(finals)}
    for key in ("em", "f1", "acc"):
        result[key] = sum(row[key] for row in finals) / len(finals)
    for key in ("agent_input_tokens", "agent_output_tokens", "summary_input_tokens", "summary_output_tokens"):
        result["total_" + key] = sum(row["token_usage"][key] for row in finals)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
