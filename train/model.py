from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import logging
import math
import os
import sys

import torch
import transformers
from peft import PeftModel, prepare_model_for_kbit_training
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import LatentConfig

logger = logging.getLogger(__name__)


def _parse_torch_dtype(name: str | None) -> torch.dtype:
    raw = (name or "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[raw]


def _hf_first_last_device(model: nn.Module) -> tuple[torch.device | None, torch.device | None]:
    hf_map = getattr(model, "hf_device_map", None)
    if not hf_map:
        try:
            dev = next(model.parameters()).device
            return dev, dev
        except StopIteration:
            return None, None

    cuda_indices: list[int] = []
    cpu_seen = False
    for dev in hf_map.values():
        if isinstance(dev, int):
            cuda_indices.append(dev)
        elif isinstance(dev, str):
            if dev.startswith("cuda:"):
                try:
                    cuda_indices.append(int(dev.split(":")[1]))
                except Exception:
                    pass
            elif dev == "cpu":
                cpu_seen = True
    if cuda_indices:
        return torch.device(f"cuda:{min(cuda_indices)}"), torch.device(f"cuda:{max(cuda_indices)}")
    if cpu_seen:
        return torch.device("cpu"), torch.device("cpu")
    try:
        dev = next(model.parameters()).device
        return dev, dev
    except StopIteration:
        return None, None


def _split_counts(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    base = total // buckets
    rem = total % buckets
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def _get_num_hidden_layers(cfg: Any) -> int:
    seen: list[str] = []
    candidates: list[Any] = [cfg]
    for attr in ("text_config", "llm_config", "language_config"):
        value = getattr(cfg, attr, None)
        if value is not None:
            candidates.append(value)
    keys = (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        "hidden_layers",
        "num_decoder_layers",
        "decoder_layers",
    )
    for candidate in candidates:
        for attr in keys:
            value = getattr(candidate, attr, None)
            if value is not None:
                return int(value)
            seen.append(f"{type(candidate).__name__}.{attr}=None")
        if isinstance(candidate, dict):
            for key in keys:
                value = candidate.get(key)
                if value is not None:
                    return int(value)
                seen.append(f"dict[{key!r}]=None")
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                for key in keys:
                    value = data.get(key)
                    if value is not None:
                        return int(value)
                    seen.append(f"to_dict()[{key!r}]=None")
    raise ValueError(
        "Could not determine compressor hidden layer count from config "
        f"type={type(cfg).__name__}; tried: {', '.join(seen[:40])}"
    )


def get_hidden_size(model: nn.Module) -> int:
    cfg = model.config

    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)

    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)

    if hasattr(cfg, "language_config") and hasattr(cfg.language_config, "hidden_size"):
        return int(cfg.language_config.hidden_size)

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return int(embeddings.weight.shape[-1])

    config_keys = list(cfg.to_dict().keys()) if hasattr(cfg, "to_dict") else dir(cfg)
    raise AttributeError(
        f"Cannot infer hidden_size from config type {type(cfg)}. "
        f"Available config keys: {config_keys}"
    )


def _select_adapter_device(
    placement: str | None,
    first_device: torch.device | None,
    last_device: torch.device | None,
) -> torch.device | None:
    mode = (placement or "first").strip().lower()
    if mode == "last":
        return last_device or first_device
    return first_device or last_device


def _build_balanced_compressor_device_map(
    model_cfg: Any,
    start_gpu: int,
    n_compressor_gpus: int,
    balance_tail_layers: int,
    explicit_num_hidden_layers: int | None = None,
) -> dict[str, int] | None:
    if n_compressor_gpus <= 1:
        return None

    total_layers = int(explicit_num_hidden_layers) if explicit_num_hidden_layers else _get_num_hidden_layers(model_cfg)
    tail_layers = max(0, min(int(balance_tail_layers or 0), total_layers))
    if tail_layers <= 0:
        return None

    devices = [start_gpu + i for i in range(n_compressor_gpus)]
    device_map: dict[str, int] = {
        "model": devices[0],
        "model.embed_tokens": devices[0],
        "lm_head": devices[-1],
    }

    head_layers = total_layers - tail_layers
    head_devices = devices[:-1] if len(devices) > 1 else devices
    head_counts = _split_counts(head_layers, len(head_devices))
    tail_counts = _split_counts(tail_layers, len(devices))

    layer_to_device: list[int] = []
    for device, count in zip(head_devices, head_counts):
        layer_to_device.extend([device] * count)
    for device, count in zip(devices, tail_counts):
        layer_to_device.extend([device] * count)
    if len(layer_to_device) != total_layers:
        raise RuntimeError(
            f"Invalid compressor device assignment: expected {total_layers} layers, got {len(layer_to_device)}"
        )

    for idx, device in enumerate(layer_to_device):
        device_map[f"model.layers.{idx}"] = device
    device_map["model.norm"] = layer_to_device[-1]
    return device_map


def _log_device_map_summary(name: str, device_map: dict[str, Any] | None) -> None:
    if not device_map:
        logger.info("%s device_map: <none>", name)
        return
    layer_items: list[tuple[int, Any]] = []
    for module_name, device in device_map.items():
        if module_name.startswith("model.layers."):
            try:
                layer_idx = int(module_name.rsplit(".", 1)[1])
            except ValueError:
                continue
            layer_items.append((layer_idx, device))
    layer_items.sort()
    ranges: list[str] = []
    if layer_items:
        start_idx, current_idx, current_device = layer_items[0][0], layer_items[0][0], layer_items[0][1]
        for idx, device in layer_items[1:]:
            if idx == current_idx + 1 and device == current_device:
                current_idx = idx
                continue
            ranges.append(f"{start_idx}-{current_idx}@cuda:{current_device}")
            start_idx = current_idx = idx
            current_device = device
        ranges.append(f"{start_idx}-{current_idx}@cuda:{current_device}")
    extra = {
        key: value
        for key, value in device_map.items()
        if key in {"model", "model.embed_tokens", "model.norm", "lm_head"}
    }
    logger.info("%s device_map summary: %s; extras=%s", name, ", ".join(ranges) or "<no layers>", extra)


def _find_transformer_layers(model: nn.Module):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(base, "model", None),
        getattr(getattr(base, "base_model", None), "model", None),
        getattr(base, "base_model", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None) if candidate is not None else None
        if layers is not None and hasattr(layers, "__len__"):
            return layers
    return None


def _find_backbone_model(model: nn.Module) -> nn.Module | None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(base, "model", None),
        getattr(getattr(base, "base_model", None), "model", None),
        getattr(base, "base_model", None),
        getattr(getattr(getattr(base, "model", None), "model", None), "model", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def compressor_last_hidden_state(
    compressor_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    try:
        backbone = _find_backbone_model(compressor_model)
        if backbone is not None:
            out = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            return out[0]
    except Exception as e:
        logger.warning("Fast hidden-state path failed (%s); falling back to full hidden states.", e)
    out = compressor_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    if getattr(out, "hidden_states", None) is not None:
        return out.hidden_states[-1]
    raise RuntimeError("Could not obtain compressor hidden states")


class CompressorAdapterTrainable(nn.Module):
    def __init__(self, compressor: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.compressor = compressor
        self.adapter = adapter

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = compressor_last_hidden_state(self.compressor, input_ids, attention_mask)
        adapter_device = next(self.adapter.parameters()).device
        if last_hidden.device != adapter_device:
            last_hidden = last_hidden.to(adapter_device)
        if attention_mask.device != adapter_device:
            attention_mask = attention_mask.to(adapter_device)
        return self.adapter(last_hidden, attention_mask=attention_mask)


class LatentAdapter(nn.Module):
    def __init__(
        self,
        d_compressor: int,
        d_agent: int,
        num_latent_tokens: int,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads is None:
            for h in (16, 8, 4, 2):
                if d_compressor % h == 0:
                    num_heads = h
                    break
            if num_heads is None:
                num_heads = 1
        self.num_latent_tokens = num_latent_tokens
        self.latent_queries = nn.Parameter(
            torch.randn(num_latent_tokens, d_compressor) / math.sqrt(d_compressor)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_compressor,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_compressor, d_agent)
        self.norm = nn.LayerNorm(d_agent)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz = hidden.size(0)
        target_dtype = self.attn.in_proj_weight.dtype
        queries = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1).to(dtype=target_dtype)
        hidden = hidden.to(dtype=target_dtype)
        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        pooled, _ = self.attn(
            query=queries,
            key=hidden,
            value=hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latent = self.out_proj(pooled)
        latent = self.norm(latent)
        latent = latent * self.alpha
        return latent


@dataclass
class LatentComponents:
    agent_model: nn.Module
    agent_tokenizer: AutoTokenizer
    compressor_model: nn.Module
    compressor_tokenizer: AutoTokenizer
    adapter: nn.Module
    trainable_compress: nn.Module | None = None


class LatentModelBuilder:
    def __init__(self, cfg: LatentConfig, attn_implementation: str | None = None) -> None:
        self.cfg = cfg
        self.attn_implementation = attn_implementation

    def build(self) -> LatentComponents:
        logger.info(
            "Loading agent from %s (transformers=%s)",
            self.cfg.agent_model_path,
            getattr(transformers, "__version__", "?"),
        )
        agent_tokenizer = AutoTokenizer.from_pretrained(self.cfg.agent_model_path, trust_remote_code=True)
        compressor_tokenizer = AutoTokenizer.from_pretrained(self.cfg.compressor_model_path, trust_remote_code=True)
        if agent_tokenizer.pad_token is None:
            agent_tokenizer.pad_token = agent_tokenizer.eos_token
        if compressor_tokenizer.pad_token is None:
            compressor_tokenizer.pad_token = compressor_tokenizer.eos_token

        n_agent_gpus = getattr(self.cfg, "agent_num_gpus", 0) or 0
        attn_impl = self.attn_implementation or (os.getenv("SEARCHO1_ATTENTION_IMPL") or "").strip() or None

        agent_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if attn_impl:
            agent_kwargs["attn_implementation"] = attn_impl
        if n_agent_gpus > 0:
            agent_kwargs["device_map"] = "auto"
            agent_kwargs["max_memory"] = {i: "60GiB" for i in range(n_agent_gpus)}
        agent_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.agent_model_path,
            **agent_kwargs,
        )

        n_compressor_gpus = getattr(self.cfg, "compressor_num_gpus", 0) or 0
        compressor_load_in_4bit = bool(getattr(self.cfg, "compressor_load_in_4bit", False))
        compressor_model_config = AutoConfig.from_pretrained(
            self.cfg.compressor_model_path,
            trust_remote_code=True,
        )
        compressor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if not compressor_load_in_4bit:
            compressor_kwargs["torch_dtype"] = torch.bfloat16
        if attn_impl:
            compressor_kwargs["attn_implementation"] = attn_impl
        if compressor_load_in_4bit:
            compute_dtype = _parse_torch_dtype(getattr(self.cfg, "compressor_bnb_4bit_compute_dtype", "bfloat16"))
            compressor_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=getattr(self.cfg, "compressor_bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(
                    getattr(self.cfg, "compressor_bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        if n_compressor_gpus > 0:
            start_gpu = n_agent_gpus if n_agent_gpus > 0 else 0
            try:
                manual_device_map = _build_balanced_compressor_device_map(
                    model_cfg=compressor_model_config,
                    start_gpu=start_gpu,
                    n_compressor_gpus=n_compressor_gpus,
                    balance_tail_layers=getattr(self.cfg, "compressor_balance_tail_layers", 0),
                    explicit_num_hidden_layers=getattr(self.cfg, "compressor_num_hidden_layers", None),
                )
            except Exception as e:
                manual_device_map = None
                logger.warning("Falling back to compressor auto device_map: %s", e)
            if manual_device_map is not None:
                compressor_kwargs["device_map"] = manual_device_map
                _log_device_map_summary("compressor(manual)", manual_device_map)
            else:
                compressor_kwargs["device_map"] = "auto"
                compressor_kwargs["max_memory"] = {
                    start_gpu + i: "60GiB" for i in range(n_compressor_gpus)
                }
        elif compressor_load_in_4bit and torch.cuda.is_available():
            target_gpu = n_agent_gpus if n_agent_gpus > 0 else torch.cuda.current_device()
            compressor_kwargs["device_map"] = {"": target_gpu}
        compressor_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.compressor_model_path,
            **compressor_kwargs,
        )
        if compressor_load_in_4bit:
            compressor_model = prepare_model_for_kbit_training(compressor_model)

        if hasattr(agent_model.config, "use_cache"):
            agent_model.config.use_cache = False
        if hasattr(compressor_model.config, "use_cache"):
            compressor_model.config.use_cache = False
        if hasattr(compressor_model, "gradient_checkpointing_enable"):
            try:
                compressor_model.gradient_checkpointing_enable()
            except Exception:
                logger.warning("Failed to enable compressor gradient checkpointing", exc_info=True)
        for p in agent_model.parameters():
            p.requires_grad = False

        d_c = get_hidden_size(compressor_model)
        d_a = get_hidden_size(agent_model)
        compressor_input_device, compressor_output_device = _hf_first_last_device(compressor_model)
        adapter_device = _select_adapter_device(
            getattr(self.cfg, "compressor_adapter_placement", "first"),
            compressor_input_device,
            compressor_output_device,
        )
        adapter = LatentAdapter(d_c, d_a, self.cfg.num_latent_tokens).to(
            device=adapter_device,
            dtype=torch.bfloat16,
        )
        logger.info(
            "compressor devices: first=%s last=%s adapter=%s",
            compressor_input_device,
            compressor_output_device,
            adapter_device,
        )

        return LatentComponents(
            agent_model=agent_model,
            agent_tokenizer=agent_tokenizer,
            compressor_model=compressor_model,
            compressor_tokenizer=compressor_tokenizer,
            adapter=adapter,
        )

    def load_from_checkpoint(self, checkpoint_dir: str | Path) -> LatentComponents:
        components = self.build()
        checkpoint_dir = Path(checkpoint_dir)
        comp_path = checkpoint_dir / "compressor"
        adapter_path = checkpoint_dir / "adapter.pt"
        if comp_path.exists():
            components.compressor_model = PeftModel.from_pretrained(
                components.compressor_model,
                comp_path,
                is_trainable=False,
            )
        if adapter_path.exists():
            state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            components.adapter.load_state_dict(state, strict=True)
        components.trainable_compress = CompressorAdapterTrainable(
            components.compressor_model,
            components.adapter,
        )
        return components


class LatentForward(nn.Module):
    def __init__(
        self,
        components: LatentComponents,
        cfg: LatentConfig,
        device_agent: torch.device | None = None,
        device_compressor: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.components = components
        self.cfg = cfg
        self.device_agent = device_agent
        self.device_compressor = device_compressor
        self.global_step = 0
        self.kl_warmup_steps = 0

    def _compressor_to_latent_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tc = getattr(self.components, "trainable_compress", None)
        if tc is not None:
            return tc(input_ids, attention_mask)
        h = compressor_last_hidden_state(self.components.compressor_model, input_ids, attention_mask)
        return self.components.adapter(h, attention_mask=attention_mask)

    def _build_student_inputs(
        self,
        batch: dict[str, Any],
        latent_hidden_flat: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        agent_embedding = self.components.agent_model.get_input_embeddings()
        agent_dtype = next(self.components.agent_model.parameters()).dtype

        segment_lists = batch["student_segment_ids"]
        completion_ids_list = batch["completion_ids"]

        sample_embeds: list[torch.Tensor] = []
        sample_attn: list[torch.Tensor] = []
        sample_labels: list[torch.Tensor] = []
        completion_starts: list[int] = []

        for sample_idx, segments in enumerate(segment_lists):
            completion_ids = completion_ids_list[sample_idx].to(device)
            parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []

            for seg_idx, seg_ids_cpu in enumerate(segments):
                seg_ids = seg_ids_cpu.to(device)
                if seg_ids.numel() > 0:
                    seg_embeds = agent_embedding(seg_ids.unsqueeze(0)).squeeze(0).to(dtype=agent_dtype)
                    parts.append(seg_embeds)
                    label_parts.append(torch.full((seg_ids.size(0),), -100, device=device, dtype=torch.long))
                if seg_idx == 0:
                    latent_piece = latent_hidden_flat[sample_idx].to(device=device, dtype=agent_dtype)
                    parts.append(latent_piece)
                    label_parts.append(
                        torch.full((latent_piece.size(0),), -100, device=device, dtype=torch.long)
                    )

            completion_start = sum(part.size(0) for part in parts)
            completion_starts.append(completion_start)
            if completion_ids.numel() > 0:
                completion_embeds = agent_embedding(completion_ids.unsqueeze(0)).squeeze(0).to(dtype=agent_dtype)
                parts.append(completion_embeds)
                label_parts.append(completion_ids.long())

            sample_embed = torch.cat(parts, dim=0)
            sample_label = torch.cat(label_parts, dim=0)
            sample_embeds.append(sample_embed)
            sample_labels.append(sample_label)
            sample_attn.append(torch.ones(sample_embed.size(0), device=device, dtype=torch.long))

        max_len = max(embed.size(0) for embed in sample_embeds)
        hidden_size = sample_embeds[0].size(-1)
        bsz = len(sample_embeds)
        inputs_embeds = torch.zeros((bsz, max_len, hidden_size), device=device, dtype=agent_dtype)
        attention_mask = torch.zeros((bsz, max_len), device=device, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, device=device, dtype=torch.long)

        for i, (embed, attn, sample_label) in enumerate(zip(sample_embeds, sample_attn, sample_labels)):
            seq_len = embed.size(0)
            inputs_embeds[i, :seq_len] = embed
            attention_mask[i, :seq_len] = attn
            labels[i, :seq_len] = sample_label
        return inputs_embeds, attention_mask, labels, completion_starts

    @staticmethod
    def _slice_teacher_completion_logits(
        teacher_logits: torch.Tensor,
        prompt_lens_full: list[int],
        completion_lens: list[int],
    ) -> list[torch.Tensor]:
        teacher_completion_logits: list[torch.Tensor] = []
        for i, (prompt_len_full, completion_len) in enumerate(zip(prompt_lens_full, completion_lens)):
            if completion_len <= 0:
                teacher_completion_logits.append(teacher_logits.new_empty((0, teacher_logits.size(-1))))
                continue
            start = max(0, prompt_len_full - 1)
            end = start + completion_len
            teacher_completion_logits.append(teacher_logits[i, start:end, :].detach().clone().cpu())
        return teacher_completion_logits

    @staticmethod
    def _completion_meta_from_labels(labels_full: torch.Tensor) -> tuple[list[int], list[int]]:
        prompt_lens: list[int] = []
        completion_lens: list[int] = []
        for row in labels_full:
            non_ignore = row.ne(-100)
            if non_ignore.any():
                prompt_len = int(non_ignore.long().argmax().item())
                completion_len = int(non_ignore.long().sum().item())
            else:
                prompt_len = int(row.size(0))
                completion_len = 0
            prompt_lens.append(prompt_len)
            completion_lens.append(completion_len)
        return prompt_lens, completion_lens

    @staticmethod
    def _compute_teacher_ce_loss(
        teacher_logits: torch.Tensor,
        labels_teacher: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        teacher_seq_len = teacher_logits.size(1)
        if labels_teacher.dim() > 1 and labels_teacher.size(1) == teacher_seq_len:
            shift_logits = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels_teacher[..., 1:].contiguous()
            return LatentForward._compute_masked_ce_loss(shift_logits, shift_labels, device)
        return torch.zeros((), device=device)

    @staticmethod
    def _compute_masked_ce_loss(
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            return torch.zeros((), device=device)

        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask].long().to(valid_logits.device)
        return F.cross_entropy(valid_logits.float(), valid_labels)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.device_agent is not None and self.device_compressor is not None:
            return self._forward_split(batch)
        device = next(self.components.adapter.parameters()).device
        return self._forward_single(batch, device)

    def _student_save_context(self):
        if (
            bool(getattr(self.cfg, "save_on_cpu_for_backward", True))
            and hasattr(torch.autograd, "graph")
            and hasattr(torch.autograd.graph, "save_on_cpu")
        ):
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
        return nullcontext()

    def _forward_single(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        agent_input_full = batch["agent_input_full"].to(device)
        agent_attention_full = batch["agent_attention_full"].to(device)
        compressor_input_ids = batch["compressor_input_ids"].to(device)
        compressor_attention_mask = batch["compressor_attention_mask"].to(device)
        labels_full = batch["labels"].to(device)
        prompt_lens_full, completion_lens = self._completion_meta_from_labels(labels_full)

        with torch.no_grad():
            teacher_out = self.components.agent_model(
                input_ids=agent_input_full,
                attention_mask=agent_attention_full,
            )
            teacher_completion_logits = self._slice_teacher_completion_logits(
                teacher_out.logits,
                prompt_lens_full,
                completion_lens,
            )
            teacher_task_loss = self._compute_teacher_ce_loss(
                teacher_out.logits,
                batch["labels"],
                device,
            )
        del teacher_out
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        latent_agent_hidden = self._compressor_to_latent_hidden(compressor_input_ids, compressor_attention_mask)
        latent_agent_hidden = torch.nan_to_num(latent_agent_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        
        inputs_embeds, attention, student_labels, completion_starts = self._build_student_inputs(
            batch,
            latent_agent_hidden,
            device,
        )
        with self._student_save_context():
            student_out = self.components.agent_model(inputs_embeds=inputs_embeds, attention_mask=attention)
            student_logits = student_out.logits
            return self._loss_and_metrics(
                batch=batch,
                device=device,
                teacher_completion_logits=teacher_completion_logits,
                student_logits=student_logits,
                labels=student_labels,
                teacher_task_loss=teacher_task_loss,
                prompt_lens_full=prompt_lens_full,
                completion_lens=completion_lens,
                completion_starts=completion_starts,
            )

    def _forward_split(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dc = self.device_compressor
        first_device = next(self.components.agent_model.parameters()).device
        agent_input_full = batch["agent_input_full"].to(first_device)
        agent_attention_full = batch["agent_attention_full"].to(first_device)
        compressor_input_ids = batch["compressor_input_ids"].to(dc)
        compressor_attention_mask = batch["compressor_attention_mask"].to(dc)
        labels_full = batch["labels"].to(first_device)
        prompt_lens_full, completion_lens = self._completion_meta_from_labels(labels_full)

        with torch.no_grad():
            teacher_out = self.components.agent_model(
                input_ids=agent_input_full,
                attention_mask=agent_attention_full,
            )
            teacher_completion_logits = self._slice_teacher_completion_logits(
                teacher_out.logits,
                prompt_lens_full,
                completion_lens,
            )
            teacher_task_loss = self._compute_teacher_ce_loss(
                teacher_out.logits,
                batch["labels"],
                first_device,
            )
        del teacher_out
        if first_device.type == "cuda":
            with torch.cuda.device(first_device):
                torch.cuda.empty_cache()

        latent_agent_hidden = self._compressor_to_latent_hidden(compressor_input_ids, compressor_attention_mask)
        latent_agent_hidden = torch.nan_to_num(latent_agent_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        if latent_agent_hidden.device != first_device:
            latent_agent_hidden = latent_agent_hidden.to(first_device)
        del compressor_input_ids, compressor_attention_mask
        if dc is not None and dc != first_device and dc.type == "cuda":
            with torch.cuda.device(dc):
                torch.cuda.empty_cache()
        inputs_embeds, attention, student_labels, completion_starts = self._build_student_inputs(
            batch,
            latent_agent_hidden,
            first_device,
        )
        with self._student_save_context():
            student_out = self.components.agent_model(inputs_embeds=inputs_embeds, attention_mask=attention)
            student_logits = student_out.logits
            return self._loss_and_metrics(
                batch=batch,
                device=first_device,
                teacher_completion_logits=teacher_completion_logits,
                student_logits=student_logits,
                labels=student_labels,
                teacher_task_loss=teacher_task_loss,
                prompt_lens_full=prompt_lens_full,
                completion_lens=completion_lens,
                completion_starts=completion_starts,
            )

    def _loss_and_metrics(
        self,
        batch: dict[str, Any],
        device: torch.device,
        teacher_completion_logits: list[torch.Tensor],
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_task_loss: torch.Tensor,
        prompt_lens_full: list[int],
        completion_lens: list[int],
        completion_starts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vocab_size = student_logits.size(-1)
        kl_terms: list[torch.Tensor] = []
        for i, (completion_len, student_start, teacher_completion) in enumerate(
            zip(completion_lens, completion_starts, teacher_completion_logits)
        ):
            if completion_len <= 0:
                continue
            student_start = max(0, student_start - 1)
            student_completion = student_logits[i, student_start : student_start + completion_len, :]
            teacher_completion = torch.clamp(
                torch.nan_to_num(teacher_completion.to(device), nan=0.0, posinf=30.0, neginf=-30.0),
                -30.0,
                30.0,
            )
            student_completion = torch.clamp(
                torch.nan_to_num(student_completion, nan=0.0, posinf=30.0, neginf=-30.0),
                -30.0,
                30.0,
            )
            student_log_probs = F.log_softmax(student_completion, dim=-1)
            teacher_probs = F.softmax(teacher_completion, dim=-1)
            kl_i = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            kl_terms.append(torch.clamp(kl_i, max=10.0))
        kl = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)

        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = self._compute_masked_ce_loss(shift_student_logits, shift_labels, device)

        warmup_steps = max(0, int(getattr(self, "kl_warmup_steps", 0)))
        warmup_factor = min(1.0, float(self.global_step) / float(warmup_steps)) if warmup_steps > 0 else 1.0
        kl_weight = self.cfg.kl_weight * warmup_factor
        total_loss = kl_weight * kl + self.cfg.task_weight * task_loss
        return kl, task_loss, total_loss, teacher_task_loss
