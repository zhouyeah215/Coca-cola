# Latte: Search-o1 检索与上下文压缩

Latte 面向多轮检索问答，将检索上下文压缩为潜在表示，供模型继续推理。本仓库包含 Latte 的训练与评测代码，以及 Vanilla、LLMSummary 和 LLMLingua 三种基线方法。

## 支持的方法

| 方法 | 上下文处理方式 |
| --- | --- |
| Vanilla | 直接使用检索结果 |
| LLMSummary | 使用语言模型生成文本摘要 |
| LLMLingua | 使用 LongLLMLingua 压缩文本 |
| Latte | 将上下文压缩为潜在表示 |

## 快速开始

先按[安装说明](docs/INSTALL.md)准备环境、模型和数据，然后在项目根目录运行以下示例。网页检索默认使用 Exa，需配置 API key。

```bash
conda activate latte
export AGENT_MODEL=/absolute/path/to/agent-model
export SEARCH_API_KEY=YOUR_EXA_API_KEY

CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py vanilla \
  --run-dir outputs/vanilla-nq-smoke \
  --dataset_name nq --split test --subset_num 10 \
  --model_path "$AGENT_MODEL" --tensor_parallel_size 1 --batch_size 1 \
  --search_provider exa \
  --max_tokens 1024 --max_search_limit 5 --max_turn 7 --top_k 5 \
  --temperature 0 --top_p 1 --top_k_sampling -1 --repetition_penalty 1

python scripts/summarize_baseline.py \
  outputs/vanilla-nq-smoke/trajectories.jsonl --expected-count 10
```

## 文档

- [环境安装与数据准备](docs/INSTALL.md)
- [各方法运行与评测](docs/METHODS.md)
- [Latte 数据准备与训练](docs/TRAINING.md)
