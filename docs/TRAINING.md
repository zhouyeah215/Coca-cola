# Latte 轨迹采集与训练

本节介绍 `train/model.py` 的 latent adapter 训练流程。先安装 `requirements/train.txt`，准备 `configs/latte.local.yaml`，并根据 [METHODS.md](METHODS.md)配置 agent/compressor 本地模型路径。

## 1. 用当前训练格式采集 teacher 轨迹

先设置 `SEARCH_API_KEY` 为 Exa key。采集命令也使用网页检索；若更换 Tavily，应同时更换 provider 与 key。

为保持已有训练流程的初始 wrapper 一致，使用原 `scripts/rag_agent_wo_url.py`。不要把新的纯文本 baseline 轨迹直接交给会重建 chat wrapper 的旧转换器，然后称为完全一致的蒸馏。

```bash
mkdir -p outputs/teacher
CUDA_VISIBLE_DEVICES=0 python scripts/rag_agent_wo_url.py \
  --dataset_name nq --split train --model_path "$AGENT_MODEL" \
  --search_provider exa \
  --search_api_key "$SEARCH_API_KEY" --tensor_parallel_size 1 --batch_size 1 \
  --max_model_len 16384 --max_tokens 1024 \
  --max_search_limit 5 --max_turn 7 --top_k 5 \
  --temperature 0.3 --top_p 0.8 --top_k_sampling 20 \
  --trajectory_jsonl outputs/teacher/all.jsonl \
  --positive_trajectory_jsonl outputs/teacher/positive.jsonl \
  --evaluate_trajectory --skip_evaluation \
  --cache_dir outputs/teacher/cache --output_dir outputs/teacher/results \
  --progress_json outputs/teacher/progress.json
```

默认完整 train split。先 smoke 时添加 `--subset_num 10`；采集指定正例数量可用 `--target_acc_count N`，但应检查真实正例数。正例筛选使用当前 ACC，不能称为严格 EM 正例。请勿从 test split 采集训练数据。

## 2. 轨迹转 SFT

```bash
python scripts/prepare_latent_qa_sft_data.py \
  --source-format trajectory \
  --trajectory-jsonl outputs/teacher/positive.jsonl \
  --config configs/latte.local.yaml \
  --agent-model-path "$AGENT_MODEL" \
  --output data/sft/latte_train.jsonl
```

保留 question、当前推理、检索结果、后续 reasoning continuation，生成 `prompt`、`completion`、`agent_prompt_full` 等字段。转换器会过滤无后续推理、格式异常或过短观察。检查日志中输入/输出数量，不要以空文件开始训练。

当前转换器会重新构建提示，数据集默认搜索上限也是其现有代码的一部分；训练轨迹使用非默认搜索预算时，必须核对生成的 `agent_prompt_full`。训练数据的自动划分是已有实现，不保证按原问题分组；正式研究需要另行实现按问题划分的 train/dev 加载。当前 trainer 没有使用 `paths.eval_data_file`，不能靠设置该字段保证问题隔离。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 SEARCHO1_ATTENTION_IMPL=sdpa \
python train/train_latent_compressor.py \
  --config configs/latte.local.yaml --epochs 1 --no-wandb
```

这是单进程按配置分配 agent/compressor 的命令。不要直接加 `torchrun --nproc_per_node=...` 并沿用手工模型 GPU 分配，把模型并行与数据并行混为一谈。checkpoint 默认写到 `checkpoints/latte/`。训练成功后保存准确配置：

```bash
cp configs/latte.local.yaml checkpoints/latte/final/train_config.yaml
```

```text
checkpoints/latte/final/
├── compressor/        # PEFT adapter 权重与 adapter_config.json
├── adapter.pt         # latent adapter 参数
└── train_config.yaml  # 需要手工复制保存；trainer 不自动保存 config
```

用该 checkpoint 按 [METHODS.md](METHODS.md) 的 Latte 命令评测。

## 配置与检查点

示例配置用于说明路径和参数，不是已发布权重的固定配置。保存检查点时同时记录模型 revision、训练数据版本、prompt policy、latent token 数与评测命令。
