from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


@dataclass
class PathsConfig:
    train_data_file: str = "data/sft/latent_web_search_sft.jsonl"
    eval_data_file: str | None = None
    checkpoint_dir: str = "checkpoints"


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "QA latent agent"
    entity: str | None = None
    name: str | None = None


@dataclass
class TrainRuntimeConfig:
    epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    resume_from_checkpoint: str | None = None
    eval_split_ratio: float = 0.1
    gradient_checkpointing: bool = True
    learning_rate: float = 1.0e-4
    learning_rate_end: float = 5.0e-6
    learning_rate_schedule: str = "linear"
    optimizer_type: str = "paged_adamw_8bit"
    weight_decay: float = 0.0
    kl_skip_threshold: float = 2.0
    eval_steps: int = 100
    save_steps: int = 200
    log_steps: int = 1
    raw_batch_log_steps: int = 1
    batch_stage_log: bool = False
    num_workers: int = 2
    seed: int = 42


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    keep_last_n_layers: int = 8


@dataclass
class LatentConfig:
    agent_model_path: str = "/path/to/Qwen3.5-35B-A3B"
    compressor_model_path: str = "/path/to/Qwen3.5-4B"
    compressor_num_hidden_layers: int | None = None
    agent_max_length: int = 8192
    compressor_max_length: int = 4096
    num_latent_tokens: int = 32
    compressor_load_in_4bit: bool = False
    compressor_bnb_4bit_quant_type: str = "nf4"
    compressor_bnb_4bit_use_double_quant: bool = True
    compressor_bnb_4bit_compute_dtype: str = "bfloat16"
    kl_weight: float = 1.0
    task_weight: float = 0.1
    kl_warmup_ratio: float = 0.1
    agent_num_gpus: int = 0
    compressor_num_gpus: int = 0
    compressor_balance_tail_layers: int = 0
    compressor_adapter_placement: str = "first"
    save_on_cpu_for_backward: bool = True
    eval_checkpoint_dir: str | None = None


@dataclass
class EvalConfig:
    baseline: str = "latent_hf"
    checkpoint: str | None = None
    datasets: str = "2wiki"
    split: str = "dev"
    max_questions: int | None = None
    start_idx: int = 0
    max_new_tokens: int = 256
    max_search_limit: int | None = None
    max_turn: int | None = None
    top_k: int | None = None
    search_api_key: str | None = None
    search_provider: str = "exa"
    bing_endpoint: str | None = None
    output_json: str | None = None
    trace_output_dir: str = "outputs/eval_traces"
    print_steps: bool = False


@dataclass
class SearchLatentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    train: TrainRuntimeConfig = field(default_factory=TrainRuntimeConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path) -> "SearchLatentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        latent_raw = raw.get("latent", {}) or {}
        if "num_latent_tokens" not in latent_raw and "num_latent_tokens_per_result" in latent_raw:
            latent_raw["num_latent_tokens"] = latent_raw["num_latent_tokens_per_result"]
        latent_raw.pop("num_latent_tokens_per_result", None)
        paths_cfg = PathsConfig(**raw.get("paths", {}))
        paths_cfg.train_data_file = _resolve_repo_path(paths_cfg.train_data_file) or paths_cfg.train_data_file
        paths_cfg.eval_data_file = _resolve_repo_path(paths_cfg.eval_data_file)
        paths_cfg.checkpoint_dir = _resolve_repo_path(paths_cfg.checkpoint_dir) or paths_cfg.checkpoint_dir
        train_cfg = TrainRuntimeConfig(**raw.get("train", {}))
        train_cfg.resume_from_checkpoint = _resolve_repo_path(train_cfg.resume_from_checkpoint)
        eval_cfg = EvalConfig(**raw.get("eval", {}))
        eval_cfg.checkpoint = _resolve_repo_path(eval_cfg.checkpoint)
        eval_cfg.output_json = _resolve_repo_path(eval_cfg.output_json)
        eval_cfg.trace_output_dir = _resolve_repo_path(eval_cfg.trace_output_dir) or eval_cfg.trace_output_dir
        return cls(
            paths=paths_cfg,
            wandb=WandbConfig(**raw.get("wandb", {})),
            train=train_cfg,
            lora=LoraConfig(**raw.get("lora", {})),
            latent=LatentConfig(**latent_raw),
            eval=eval_cfg,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
