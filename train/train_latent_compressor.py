#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import sys

import torch
from torch.optim.lr_scheduler import LambdaLR

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import SearchLatentConfig
from trainer import LatentTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent compressor for Search-o1 QA search traces")
    parser.add_argument("--config", type=str, default=str(_HERE / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    args = parser.parse_args()

    cfg = SearchLatentConfig.load(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    accelerator = None
    try:
        from accelerate import Accelerator

        accelerator = Accelerator()
        logger.info("Accelerate active: device=%s, processes=%s", accelerator.device, accelerator.num_processes)
    except ImportError:
        pass

    if cfg.wandb.enabled and not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name,
                config=cfg.to_dict(),
            )
        except ImportError:
            logger.warning("wandb is not installed; continuing without it.")

    trainer = LatentTrainer(cfg)
    train_loader, val_loader = trainer.build_dataloaders()

    if accelerator is not None:
        if val_loader is not None:
            trainer.state.forward, trainer.state.optimizer, train_loader, val_loader = accelerator.prepare(
                trainer.state.forward,
                trainer.state.optimizer,
                train_loader,
                val_loader,
            )
        else:
            trainer.state.forward, trainer.state.optimizer, train_loader = accelerator.prepare(
                trainer.state.forward,
                trainer.state.optimizer,
                train_loader,
            )
        trainer.state.device = accelerator.device
        trainer._accelerator = accelerator

    accum = max(1, cfg.train.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, steps_per_epoch * cfg.train.epochs)
    kl_warmup_ratio = float(getattr(cfg.latent, "kl_warmup_ratio", 0.1) or 0.0)
    kl_warmup_steps = max(0, int(total_steps * kl_warmup_ratio))
    trainer.state.forward.kl_warmup_steps = kl_warmup_steps
    base_lr = trainer.state.optimizer.param_groups[0]["lr"]
    end_factor = float(cfg.train.learning_rate_end) / float(base_lr)
    if not (0.0 < end_factor <= 1.0):
        raise ValueError(f"Invalid LR schedule: base_lr={base_lr}, end_factor={end_factor}")
    sched_kind = (cfg.train.learning_rate_schedule or "linear").strip().lower()
    if sched_kind not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported learning_rate_schedule: {sched_kind}")

    def lr_factor(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        if sched_kind == "linear":
            return 1.0 - (1.0 - end_factor) * progress
        cos_w = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_factor + (1.0 - end_factor) * cos_w

    trainer.scheduler = LambdaLR(
        trainer.state.optimizer,
        lr_lambda=lambda step: lr_factor(min(step / total_steps, 1.0)),
    )

    logger.info(
        "Starting training: epochs=%d steps_per_epoch=%d eval_steps=%d save_steps=%d lr=%g->%g kl_warmup_steps=%d kl_skip_threshold=%g",
        cfg.train.epochs,
        steps_per_epoch,
        cfg.train.eval_steps,
        cfg.train.save_steps,
        base_lr,
        base_lr * end_factor,
        kl_warmup_steps,
        float(getattr(cfg.train, "kl_skip_threshold", 0.0) or 0.0),
    )
    is_main = accelerator is None or accelerator.is_main_process
    for epoch in range(cfg.train.epochs):
        stats = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            val_loader=val_loader,
            eval_steps=cfg.train.eval_steps,
            save_steps=cfg.train.save_steps,
        )
        if is_main:
            logger.info(
                "Epoch %d train: loss=%.4f kl=%.4f task=%.4f teacher_task=%.4f alpha=%.6g",
                epoch,
                stats["loss"],
                stats["kl"],
                stats["task"],
                stats["teacher_task"],
                stats.get("alpha", float("nan")),
            )
        if val_loader is not None:
            val_stats = trainer.evaluate(val_loader)
            if is_main:
                logger.info(
                    "Epoch %d val: loss=%.4f kl=%.4f task=%.4f teacher_task=%.4f alpha=%.6g",
                    epoch,
                    val_stats["loss"],
                    val_stats["kl"],
                    val_stats["task"],
                    val_stats["teacher_task"],
                    val_stats.get("alpha", float("nan")),
                )
        if is_main:
            if accelerator is not None:
                accelerator.wait_for_everyone()
            trainer.save_checkpoint(Path(cfg.paths.checkpoint_dir) / f"epoch_{epoch}")

    out_dir = Path(cfg.paths.checkpoint_dir) / "final"
    if accelerator is not None:
        accelerator.wait_for_everyone()
    trainer.save_checkpoint(out_dir)
    if is_main:
        logger.info("Saved final checkpoint to %s", out_dir)


if __name__ == "__main__":
    main()
