# Latte: Search-o1 检索与上下文压缩

本目录整理 Vanilla（no compression）、LLMSummary、LLMLingua 和 Latte 的环境安装、启动与训练流程。所有命令均从 **本 README 所在目录**执行，发布目录为 `release/Coca-cola/`。

四种方法的文档命令现统一使用 Exa 网页检索；三个 baseline 默认 `--search_provider exa`，也可切换 `tavily`。发布版仅支持这两个网页检索服务，运行时需提供对应 API key。

保留 Search-o1 查询/结果标记、搜索循环和 `\boxed{answer}` 约定，没有迁移为模型原生的 tool calling 或逐轮 chat messages。

| 方法 | 入口 | 返回到后续推理的观察信息 |
| --- | --- | --- |
| Vanilla / no compress | `scripts/run_baseline.py vanilla` | 原始检索结果 |
| LLMSummary | `scripts/run_baseline.py llmsummary` | 独立模型通过 `/v1/completions` 生成的摘要 |
| LLMLingua | `scripts/run_baseline.py llmlingua` | **LongLLMLingua** 问题相关文本压缩，不是 LLMLingua-2 |
| Latte | `train/eval_all.py --baseline latent_hf` | compressor + latent adapter 的向量；历史观察持续以 latent 保留 |

**现有格式差异：**三个 baseline 使用 Search-o1 纯文本续写，不调用 agent chat template。Latte 评测入口和训练数据转换器对初始提示/独立收尾提示调用 `apply_chat_template`，之后仍使用 Search-o1 连续上下文。不能把这些命令直接视为四种方法完全同条件的对比，详见[方法与复现边界](docs/METHODS.md)。

## 使用顺序

1. [安装环境、模型、数据和网页检索 API](docs/INSTALL.md)
2. [四种方法的完整命令与结果汇总](docs/METHODS.md)
3. [Latte 轨迹采集、数据准备与训练](docs/TRAINING.md)

```bash
cd /path/to/release/Coca-cola
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

加 `--dry-run` 只打印实际命令，不加载模型或创建输出目录。GPU 和 batch size 是示例，不代表最低显存要求。

## 目录

```text
configs/latte.example.yaml        # 无凭据/开发机路径的示例配置
requirements/                    # 实验环境依赖
scripts/run_baseline.py           # 三种 baseline 的隔离运行入口
scripts/baselines/                # Search-o1 推理与压缩模块
scripts/summarize_baseline.py     # 汇总 final rows 的 EM/F1/ACC 和 token
train/                           # 现有 Latte latent adapter 训练/评测
scripts/prepare_latent_qa_sft_data.py
```

