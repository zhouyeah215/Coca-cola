from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
from typing import Any
import sys

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from torch.utils.data import DataLoader, random_split
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import SearchLatentConfig
from dataset import LatentSFTDataset, latent_collate_fn
from model import (
    CompressorAdapterTrainable,
    LatentComponents,
    LatentForward,
    LatentModelBuilder,
    _hf_first_last_device,
    _find_transformer_layers,
    _select_adapter_device,
)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


logger = logging.getLogger(__name__)


@dataclass
class LatentTrainingState:
    components: LatentComponents
    forward: LatentForward
    optimizer: torch.optim.Optimizer
    device: torch.device


class LatentTrainer:
    def __init__(self, cfg: SearchLatentConfig) -> None:
        self.cfg = cfg
        builder = LatentModelBuilder(cfg.latent)
        resume_from_checkpoint = getattr(cfg.train, "resume_from_checkpoint", None)
        if resume_from_checkpoint:
            logger.info("Initializing trainable weights from checkpoint: %s", resume_from_checkpoint)
            components = builder.load_from_checkpoint(resume_from_checkpoint)
        else:
            components = builder.build()

        lora_cfg = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        components.compressor_model = get_peft_model(components.compressor_model, lora_cfg)
        self._freeze_compressor_layers(components)
        if hasattr(components.compressor_model, "print_trainable_parameters"):
            components.compressor_model.print_trainable_parameters()

        n_agent_gpus = getattr(cfg.latent, "agent_num_gpus", 0) or 0
        n_compressor_gpus = getattr(cfg.latent, "compressor_num_gpus", 0) or 0
        if n_agent_gpus > 0:
            device_agent_first = next(components.agent_model.parameters()).device
            if n_compressor_gpus > 0:
                device_compressor_first, device_compressor_last = _hf_first_last_device(components.compressor_model)
            else:
                device_compressor_first = torch.device(f"cuda:{n_agent_gpus}")
                device_compressor_last = device_compressor_first
                if not bool(getattr(cfg.latent, "compressor_load_in_4bit", False)):
                    components.compressor_model.to(device_compressor_first)
            if device_compressor_first is None:
                raise RuntimeError("Could not determine compressor input device")
            if device_compressor_last is None:
                device_compressor_last = device_compressor_first
            adapter_device = _select_adapter_device(
                getattr(cfg.latent, "compressor_adapter_placement", "first"),
                device_compressor_first,
                device_compressor_last,
            )
            components.adapter.to(adapter_device)
            components.trainable_compress = CompressorAdapterTrainable(
                components.compressor_model,
                components.adapter,
            )
            device = device_compressor_first
            forward = LatentForward(
                components,
                cfg.latent,
                device_agent=device_agent_first,
                device_compressor=device_compressor_first,
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            components.agent_model.to(device)
            components.compressor_model.to(device)
            components.adapter.to(device)
            components.trainable_compress = CompressorAdapterTrainable(
                components.compressor_model,
                components.adapter,
            )
            forward = LatentForward(components, cfg.latent).to(device)

        if cfg.train.gradient_checkpointing:
            if hasattr(components.agent_model, "gradient_checkpointing_enable"):
                components.agent_model.gradient_checkpointing_enable()
            if hasattr(components.compressor_model, "gradient_checkpointing_enable"):
                components.compressor_model.gradient_checkpointing_enable()

        target_dtype = next(components.compressor_model.parameters()).dtype
        components.adapter = components.adapter.to(dtype=target_dtype)
        for p in components.compressor_model.parameters():
            if p.requires_grad and p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)

        trainable_params = [p for p in components.trainable_compress.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(trainable_params)
        self.state = LatentTrainingState(
            components=components,
            forward=forward,
            optimizer=optimizer,
            device=device,
        )
        self.global_step = 0
        self.scheduler = None

    def _is_main_process(self) -> bool:
        accelerator = getattr(self, "_accelerator", None)
        return accelerator is None or accelerator.is_main_process

    def _build_optimizer(self, trainable_params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        optimizer_type = str(getattr(self.cfg.train, "optimizer_type", "paged_adamw_8bit")).strip().lower()
        lr = self.cfg.train.learning_rate
        weight_decay = getattr(self.cfg.train, "weight_decay", 0.0)

        if optimizer_type in {"adamw_8bit", "paged_adamw_8bit"}:
            if bnb is None:
                raise ImportError(
                    "bitsandbytes is required for 8-bit optimizers. Install it or set train.optimizer_type=adamw."
                )
            optimizer_cls = (
                bnb.optim.PagedAdamW8bit if optimizer_type == "paged_adamw_8bit" else bnb.optim.AdamW8bit
            )
            return optimizer_cls(trainable_params, lr=lr, weight_decay=weight_decay)
        if optimizer_type == "adamw":
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer type: {self.cfg.train.optimizer_type}")

    def _adapter_alpha_value(self) -> float:
        alpha = getattr(self.state.components.adapter, "alpha", None)
        if alpha is None:
            return float("nan")
        if isinstance(alpha, torch.Tensor):
            return float(alpha.detach().float().mean().cpu().item())
        try:
            return float(alpha)
        except (TypeError, ValueError):
            return float("nan")

    def _freeze_compressor_layers(self, components: LatentComponents) -> None:
        keep_last_n = max(0, int(self.cfg.lora.keep_last_n_layers))
        try:
            layers = _find_transformer_layers(components.compressor_model)
            if layers is None or not hasattr(layers, "__len__"):
                return
            cutoff = max(0, len(layers) - keep_last_n)
            for idx, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = idx >= cutoff
                    else:
                        param.requires_grad = False
            self._install_compressor_prefix_cutoff(components, cutoff)
        except Exception:
            return

    def _install_compressor_prefix_cutoff(self, components: LatentComponents, cutoff: int) -> None:
        layers = _find_transformer_layers(components.compressor_model)
        if layers is None or not hasattr(layers, "__len__"):
            return

        old_handle = getattr(components.compressor_model, "_latent_prefix_detach_handle", None)
        if old_handle is not None:
            try:
                old_handle.remove()
            except Exception:
                pass
            components.compressor_model._latent_prefix_detach_handle = None

        if cutoff <= 0 or cutoff >= len(layers):
            return

        def _detach_hidden_pre_hook(module, args):
            if not args:
                return args
            hidden = args[0]
            if torch.is_tensor(hidden) and hidden.requires_grad:
                hidden = hidden.detach()
            return (hidden, *args[1:])

        handle = layers[cutoff].register_forward_pre_hook(_detach_hidden_pre_hook)
        components.compressor_model._latent_prefix_detach_handle = handle

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        dataset = LatentSFTDataset(
            data_path=self.cfg.paths.train_data_file,
            compressor_tokenizer=self.state.components.compressor_tokenizer,
            agent_tokenizer=self.state.components.agent_tokenizer,
            num_latent_tokens=self.cfg.latent.num_latent_tokens,
            compressor_max_length=self.cfg.latent.compressor_max_length,
            agent_max_length=self.cfg.latent.agent_max_length,
        )
        collate = partial(
            latent_collate_fn,
            agent_pad_token_id=self.state.components.agent_tokenizer.pad_token_id
            or self.state.components.agent_tokenizer.eos_token_id
            or 0,
            compressor_pad_token_id=self.state.components.compressor_tokenizer.pad_token_id
            or self.state.components.compressor_tokenizer.eos_token_id
            or 0,
            max_length=self.cfg.latent.agent_max_length,
        )
        if 0 < self.cfg.train.eval_split_ratio < 1 and len(dataset) > 10:
            n_val = max(1, int(len(dataset) * self.cfg.train.eval_split_ratio))
            n_train = len(dataset) - n_val
            gen = torch.Generator().manual_seed(self.cfg.train.seed)
            train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.cfg.train.per_device_train_batch_size,
                shuffle=True,
                num_workers=self.cfg.train.num_workers,
                pin_memory=True,
                persistent_workers=self.cfg.train.num_workers > 0,
                collate_fn=collate,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg.train.per_device_train_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate,
            )
            return train_loader, val_loader

        train_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.train.num_workers > 0,
            collate_fn=collate,
        )
        return train_loader, None

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        val_loader: DataLoader | None = None,
        eval_steps: int | None = None,
        save_steps: int | None = None,
    ) -> dict[str, Any]:
        self.state.forward.train()
        accum = max(1, self.cfg.train.gradient_accumulation_steps)
        total_loss = total_kl = total_task = total_teacher = 0.0
        steps = 0
        run_loss = run_kl = run_task = run_teacher = 0.0
        micro_count_since_step = 0
        num_batches = len(dataloader)
        log_steps = max(1, int(getattr(self.cfg.train, "log_steps", 1) or 1))
        raw_batch_log_steps = int(getattr(self.cfg.train, "raw_batch_log_steps", log_steps) or 0)
        batch_stage_log = bool(getattr(self.cfg.train, "batch_stage_log", False))
        kl_skip_threshold = float(getattr(self.cfg.train, "kl_skip_threshold", 0.0) or 0.0)

        for step, batch in enumerate(dataloader):
            self.state.forward.global_step = self.global_step
            if self._is_main_process() and batch_stage_log:
                print(
                    f"Epoch {epoch} batch {step + 1}/{num_batches}: entering forward "
                    f"(global_step={self.global_step})",
                    flush=True,
                )
            try:
                kl, task_loss, loss, teacher_task = self.state.forward(batch)
            except torch.OutOfMemoryError as e:
                self.state.optimizer.zero_grad(set_to_none=True)
                run_loss = run_kl = run_task = run_teacher = 0.0
                micro_count_since_step = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self._is_main_process():
                    print(f"Epoch {epoch} batch {step + 1}: OOM during forward, skipping batch ({e})", flush=True)
                continue
            if self._is_main_process() and batch_stage_log:
                alpha_value = self._adapter_alpha_value()
                print(
                    f"Epoch {epoch} batch {step + 1}/{num_batches}: forward done "
                    f"loss={loss.item():.4f} kl={kl.item():.4f} task={task_loss.item():.4f} "
                    f"teacher_task={teacher_task.item():.4f} alpha={alpha_value:.6g}",
                    flush=True,
                )
            if kl_skip_threshold > 0.0 and torch.isfinite(kl) and kl.item() > kl_skip_threshold:
                self.state.optimizer.zero_grad(set_to_none=True)
                run_loss = run_kl = run_task = run_teacher = 0.0
                micro_count_since_step = 0
                if self._is_main_process():
                    print(
                        f"Epoch {epoch} batch {step + 1}: skipping batch because kl={kl.item():.4f} "
                        f"exceeds threshold={kl_skip_threshold:.4f}",
                        flush=True,
                    )
                continue
            if not all(torch.isfinite(x) for x in (kl, task_loss, loss, teacher_task)):
                self.state.optimizer.zero_grad(set_to_none=True)
                run_loss = run_kl = run_task = run_teacher = 0.0
                micro_count_since_step = 0
                if self._is_main_process():
                    print(
                        "Epoch "
                        f"{epoch} batch {step + 1}: non-finite metrics detected, skipping batch "
                        f"(kl={kl} task={task_loss} loss={loss} teacher={teacher_task})",
                        flush=True,
                    )
                continue

            try:
                (loss / accum).backward()
                torch.nn.utils.clip_grad_norm_(self.state.optimizer.param_groups[0]["params"], max_norm=1.0)
            except torch.OutOfMemoryError as e:
                self.state.optimizer.zero_grad(set_to_none=True)
                run_loss = run_kl = run_task = run_teacher = 0.0
                micro_count_since_step = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self._is_main_process():
                    print(f"Epoch {epoch} batch {step + 1}: OOM during backward, skipping batch ({e})", flush=True)
                continue
            if self._is_main_process() and batch_stage_log:
                print(
                    f"Epoch {epoch} batch {step + 1}/{num_batches}: backward done "
                    f"(micro_step={(step + 1) % accum or accum}/{accum})",
                    flush=True,
                )
            run_loss += loss.item()
            run_kl += kl.item()
            run_task += task_loss.item()
            run_teacher += teacher_task.item()
            micro_count_since_step += 1

            do_step = micro_count_since_step >= accum or (step == num_batches - 1 and micro_count_since_step > 0)
            if (
                self._is_main_process()
                and raw_batch_log_steps > 0
                and ((step + 1) % raw_batch_log_steps == 0)
                and not do_step
            ):
                pending_micro = micro_count_since_step
                current_lr = float(self.state.optimizer.param_groups[0]["lr"])
                alpha_value = self._adapter_alpha_value()
                print(
                    f"Epoch {epoch} batch {step + 1}/{num_batches}: "
                    f"micro_step={pending_micro}/{accum} global_step={self.global_step} "
                    f"loss={loss.item():.4f} kl={kl.item():.4f} task={task_loss.item():.4f} "
                    f"teacher_task={teacher_task.item():.4f} alpha={alpha_value:.6g} lr={current_lr:.6g}",
                    flush=True,
                )
            if do_step:
                self.state.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self._is_main_process() and batch_stage_log:
                    print(
                        f"Epoch {epoch} batch {step + 1}/{num_batches}: optimizer step done "
                        f"(global_step={self.global_step})",
                        flush=True,
                    )

                if save_steps and self.global_step % save_steps == 0:
                    acc = getattr(self, "_accelerator", None)
                    if acc is not None:
                        acc.wait_for_everyone()
                    if acc is None or acc.is_main_process:
                        self.save_checkpoint(Path(self.cfg.paths.checkpoint_dir) / f"step_{self.global_step}")

                total_loss += run_loss
                total_kl += run_kl
                total_task += run_task
                total_teacher += run_teacher
                steps += 1
                n_accum = micro_count_since_step
                avg_loss = run_loss / n_accum
                avg_kl = run_kl / n_accum
                avg_task = run_task / n_accum
                avg_teacher = run_teacher / n_accum
                current_lr = float(self.state.optimizer.param_groups[0]["lr"])
                alpha_value = self._adapter_alpha_value()
                if _HAS_WANDB and wandb.run is not None:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/kl": avg_kl,
                            "train/task": avg_task,
                            "train/teacher_task": avg_teacher,
                            "train/adapter_alpha": alpha_value,
                            "train/loss_ratio": avg_task / (avg_teacher + 1e-9),
                            "train/global_step": self.global_step,
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/batch": step + 1,
                        }
                    )
                if self._is_main_process() and (self.global_step % log_steps == 0):
                    print(
                        f"Epoch {epoch} step {self.global_step} batch {step + 1}/{num_batches}: "
                        f"loss={avg_loss:.4f} kl={avg_kl:.4f} task={avg_task:.4f} "
                        f"teacher_task={avg_teacher:.4f} ratio={avg_task / (avg_teacher + 1e-9):.4f} "
                        f"alpha={alpha_value:.6g} lr={current_lr:.6g}",
                        flush=True,
                    )
                if val_loader is not None and eval_steps and self.global_step % eval_steps == 0:
                    val_stats = self.evaluate(val_loader)
                    if self._is_main_process():
                        print(
                            f"Eval epoch {epoch} step {self.global_step}: "
                            f"loss={val_stats['loss']:.4f} kl={val_stats['kl']:.4f} "
                            f"task={val_stats['task']:.4f} teacher_task={val_stats['teacher_task']:.4f} "
                            f"ratio={val_stats['task'] / (val_stats['teacher_task'] + 1e-9):.4f} "
                            f"alpha={val_stats['alpha']:.6g}",
                            flush=True,
                        )
                    if _HAS_WANDB and wandb.run is not None:
                        wandb.log(
                            {f"val/{k}": v for k, v in val_stats.items()}
                            | {
                                "val/loss_ratio": val_stats["task"] / (val_stats["teacher_task"] + 1e-9),
                                "train/global_step": self.global_step,
                                "train/epoch": epoch,
                            }
                        )

                run_loss = run_kl = run_task = run_teacher = 0.0
                micro_count_since_step = 0

        if steps == 0:
            return {"loss": 0.0, "kl": 0.0, "task": 0.0, "teacher_task": 0.0, "alpha": self._adapter_alpha_value()}
        return {
            "loss": total_loss / steps,
            "kl": total_kl / steps,
            "task": total_task / steps,
            "teacher_task": total_teacher / steps,
            "alpha": self._adapter_alpha_value(),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.state.forward.eval()
        total_loss = total_kl = total_task = total_teacher = 0.0
        steps = 0
        for batch in dataloader:
            kl, task_loss, loss, teacher_task = self.state.forward(batch)
            if not all(torch.isfinite(x) for x in (kl, task_loss, loss, teacher_task)):
                continue
            total_loss += loss.item()
            total_kl += kl.item()
            total_task += task_loss.item()
            total_teacher += teacher_task.item()
            steps += 1
        self.state.forward.train()
        if steps == 0:
            return {"loss": 0.0, "kl": 0.0, "task": 0.0, "teacher_task": 0.0, "alpha": self._adapter_alpha_value()}
        return {
            "loss": total_loss / steps,
            "kl": total_kl / steps,
            "task": total_task / steps,
            "teacher_task": total_teacher / steps,
            "alpha": self._adapter_alpha_value(),
        }

    def save_checkpoint(self, save_dir: str | Path) -> None:
        accelerator = getattr(self, "_accelerator", None)
        if accelerator is not None and not accelerator.is_main_process:
            return
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            forward = accelerator.unwrap_model(self.state.forward)
            compressor = forward.components.compressor_model
            adapter = forward.components.adapter
        else:
            compressor = self.state.components.compressor_model
            adapter = self.state.components.adapter
        if hasattr(compressor, "save_pretrained"):
            compressor.save_pretrained(save_dir / "compressor")
        else:
            torch.save(compressor.state_dict(), save_dir / "compressor.pt")
        torch.save(adapter.state_dict(), save_dir / "adapter.pt")
