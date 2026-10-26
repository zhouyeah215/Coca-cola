#!/usr/bin/env python3
"""Export the curated web-search release without development experiments or runtime artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def source_directory():
    if (ROOT / "release_manifest.json").is_file():
        return ROOT
    candidates = sorted(path.parent for path in (ROOT / "release").glob("*/release_manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError("Run the exporter from the curated release, or keep exactly one release manifest under release/.")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    source = source_directory()
    if destination == source or destination in source.parents:
        raise ValueError("Destination must not replace the release source or its parents")
    files = {Path("README.md"), Path(".gitignore"), Path("configs/latte.example.yaml")}
    files.update(Path("docs") / name for name in (
        "INSTALL.md", "METHODS.md", "TRAINING.md"))
    for directory, pattern in (("requirements", "*.txt"), ("train", "*.py")):
        files.update(p.relative_to(source) for p in (source / directory).glob(pattern))
    for name in ("run_baseline.py", "summarize_baseline.py",
                 "test_release_baselines.py", "export_release.py", "rag_agent_wo_url.py",
                 "sampling_runtime.py", "web_search.py", "evaluate.py",
                 "prompts.py", "prepare_latent_qa_sft_data.py", "utils/math_equivalence.py"):
        files.add(Path("scripts") / name)
    files.update(p.relative_to(source) for p in (source / "scripts/baselines").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and
                 (p.suffix in (".py", ".md") or p.name == "LICENSE"))
    for path in files:
        if not (source / path).is_file() or (source / path).is_symlink():
            raise ValueError(f"Missing source or unexpected symlink: {path}")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(files):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / path, target)
        manifest[str(path)] = hashlib.sha256(target.read_bytes()).hexdigest()
    # The legacy default configuration contains machine-specific paths.
    shutil.copy2(source / "configs/latte.example.yaml", destination / "train/config.yaml")
    manifest["train/config.yaml"] = hashlib.sha256((destination / "train/config.yaml").read_bytes()).hexdigest()
    (destination / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest)} files to {destination}")
    print("Data, checkpoints, internal notes and Git history excluded.")


if __name__ == "__main__":
    main()
