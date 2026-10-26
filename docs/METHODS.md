# 四种方法的启动命令

先完成 [INSTALL.md](INSTALL.md)。以下为 Bash 命令，从项目根目录（发布目录 `Coca-cola/`）执行，每次运行使用新的输出目录。

## 公共参数

```bash
export AGENT_MODEL=/absolute/path/to/agent-model
export SEARCH_PROVIDER=exa
export SEARCH_API_KEY=YOUR_EXA_API_KEY
export DATASET=nq
COMMON=(
  --dataset_name "$DATASET" --split test --subset_num 10
  --model_path "$AGENT_MODEL" --tensor_parallel_size 1 --batch_size 1
  --search_provider "$SEARCH_PROVIDER"
  --max_tokens 1024 --max_search_limit 5 --max_turn 7 --top_k 5
  --temperature 0 --top_p 1 --top_k_sampling -1 --repetition_penalty 1
  --gpu_memory_utilization 0.80 --max_num_batched_tokens 2048
)
```

三个 baseline 的默认 provider 为 Exa；省略 `--search_provider` 也会使用 Exa。要切换 Tavily，先设置 `SEARCH_PROVIDER=tavily` 与对应的 `SEARCH_API_KEY`，再构建 `COMMON`。key 优先级为显式 `--search_api_key`、`SEARCH_API_KEY`、所选服务的 `EXA_API_KEY`/`TAVILY_API_KEY`。缺少 key 会在加载模型前报错；key 不写入轨迹或缓存。

以上是使用 greedy 解码的 10 条样本启动示例。扩大 GPU 数量时同时设置 `CUDA_VISIBLE_DEVICES` 与 `--tensor_parallel_size`。

## 1. Vanilla（no compress）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py vanilla \
  --run-dir "outputs/vanilla-${DATASET}-smoke" "${COMMON[@]}"
```

Exa/Tavily 返回的网页标题、URL 和内容片段组成原始检索 JSON，放入 `<|begin_search_result|>...<|end_search_result|>`，继续同一文本上下文。

## 2. LLMSummary

在独立终端启动摘要服务，示例使用物理 GPU 1：

```bash
conda activate latte
export SUMMARY_MODEL=/absolute/path/to/summary-model
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "$SUMMARY_MODEL" --served-model-name latte-summary \
  --host 127.0.0.1 --port 18081 --tensor-parallel-size 1 \
  --dtype bfloat16 --gpu-memory-utilization 0.60 \
  --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --generation-config vllm --language-model-only --enforce-eager
```

服务就绪后在运行 agent 的终端执行：

```bash
curl --fail http://127.0.0.1:18081/health
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py llmsummary \
  --run-dir "outputs/llmsummary-${DATASET}-smoke" "${COMMON[@]}" \
  --summary_model_name latte-summary --summary_api_base http://127.0.0.1:18081/v1 \
  --summary_max_tokens 1024 --summary_temperature 0 \
  --summary_timeout 600 --summary_workers 16 --summary_max_input_chars 24000
```

摘要端调用 `/v1/completions`，传入字符串 `prompt`，不调用 `/chat/completions`，不设置 `--chat-template`。当前摘要提示要求保留证据，并使用固定 `\n<think>\n\n</think>\n\n` prefill；换摘要模型需要验证该 prefill 的适用性，不能宣称模型无关。原文与摘要都写入轨迹，token 统计分别记录 agent 和 summarizer。

## 3. LLMLingua（实际为 LongLLMLingua）

CPU 压缩器：

```bash
export LINGUA_MODEL=/absolute/path/to/causal-lm-compressor
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py llmlingua \
  --run-dir "outputs/llmlingua-${DATASET}-smoke" "${COMMON[@]}" \
  --compression_model_name "$LINGUA_MODEL" --compression_device cpu \
  --compression_rate 0.5 --compression_target_token -1
```

也可让 agent 使用逻辑 GPU 0，压缩器使用逻辑 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_baseline.py llmlingua \
  --run-dir "outputs/llmlingua-gpu-${DATASET}-smoke" "${COMMON[@]}" \
  --compression_model_name "$LINGUA_MODEL" --compression_device cuda:1 \
  --compression_rate 0.5 --compression_target_token -1
```

这里 `--tensor_parallel_size 1` 只分配 agent；若 agent TP=2，则至少另留第三张 GPU 给压缩器。rate 是保留比例；正数 `--compression_target_token` 指定目标 token 数。适配器保留 `rank_method=longllmlingua`、`condition_compare=True`、`context_budget=+100`、dynamic ratio=0.4，不添加模型 chat template。CPU 运行可能很慢且占用较大内存。

LongLLMLingua 在 Search-o1 读取网页检索结果时压缩观察文本。压缩器的 token/时间记录在 retrieval row 的 `compression` 字段，不应将它忽略后计算“总计算成本”。

## 4. Latte（latent adapter）

本节运行 `train/model.py` 的 latent adapter 方案。需已有训练好的 checkpoint，或先按 [TRAINING.md](TRAINING.md)训练。

```bash
cp configs/latte.example.yaml configs/latte.local.yaml
export COMPRESSOR_MODEL=/absolute/path/to/compressor-model
export CHECKPOINT=/absolute/path/to/latte-checkpoint
# API key 由用户在本机设置：export SEARCH_API_KEY=...
python - <<'PY'
import os, yaml
from pathlib import Path
p = Path('configs/latte.local.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['latent']['agent_model_path'] = os.environ['AGENT_MODEL']
cfg['latent']['compressor_model_path'] = os.environ['COMPRESSOR_MODEL']
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
ckpt = Path(os.environ['CHECKPOINT'])
for name in ('adapter.pt', 'compressor/adapter_config.json'):
    assert (ckpt / name).is_file(), f'Missing checkpoint component: {name}'
PY

CUDA_VISIBLE_DEVICES=0,1 SEARCHO1_ATTENTION_IMPL=sdpa \
python train/eval_all.py \
  --config configs/latte.local.yaml --baseline latent_hf --checkpoint "$CHECKPOINT" \
  --datasets "$DATASET" --split test --max-questions 10 --start-idx 0 \
  --max-new-tokens 1024 --max-search-limit 5 --max-turn 7 --top-k 5 \
  --search-provider "$SEARCH_PROVIDER" --search-api-key "$SEARCH_API_KEY" \
  --output-json "outputs/latte-${DATASET}-smoke/results.json" \
  --trace-output-dir "outputs/latte-${DATASET}-smoke/traces"
```

示例 config 将 agent 与 compressor 各放一张 GPU，模型大小不同需调整 GPU 数量。`num_latent_tokens`、compressor 截断层数、模型与 LoRA 设置必须匹配 checkpoint；示例 128 tokens 不是所有权重通用。从训练时另存的配置恢复这些字段；当前 trainer 不自动保存 config。

当前这个 Latte CLI 只接受 `tavily`/`exa`。四种方法的文档示例现在统一为 Exa；切换服务时应四种一起切换。其配对的无压缩 HF 入口是同一脚本的 `--baseline original_hf`：使用同一 config、模型、在线检索、生成与停止逻辑，省略 checkpoint 参数即可。它和第 1 节 vLLM Vanilla 不是同一评测后端。

网页模式沿用当前 `web_search.py` 的 API 调用：Exa 请求 20 条结果、每条正文片段最多 1200 字符并请求 highlights；Tavily 使用 advanced、最多 20 条且不返回 raw content，随后统一截取 `--top_k` 条。不会额外抓取整页正文。LLMSummary 的本地 `/v1/completions` 仍是摘要模型服务，与网页检索 API 分开。

## 全量与输出

三个 baseline：删除 `COMMON` 中的 `--subset_num 10`，或改为 `--subset_num -1`。换数据集时先重新构建 `COMMON` 数组，修改 `DATASET` 不会自动改变已展开的数组。Latte 全量评测需把本地配置 `eval.max_questions` 改为 `null`，并省略 `--max-questions`，不能传 `-1`（会触发 Python 切片）。

baseline 的 `--start_idx`、`--end_idx` 为数据原始序号的半开区间。手工分片需为每个进程使用独立 run-dir。输出包括：

- `trajectories.jsonl`：retrieval_step 与 scored final rows。
- `progress.json`：进度；只有 `phase=complete` 且 final 数量符合预期才视为完成。
- `command.json`：实际命令（API key 已脱敏）；`exit_code`：进程退出码。
- `cache/`：该运行独立的检索缓存；不要让并发进程覆盖同一缓存。

```bash
python scripts/summarize_baseline.py outputs/vanilla-nq-smoke/trajectories.jsonl \
  --expected-count 10
```

`EM` 是规范化答案精确匹配，`F1` 是答案 token 重合，`ACC` 是本项目的答案包含判断，不等于 EM。baseline 的评分只使用 assistant 输出，不将检索原文作为预测。完整评测请把 expected count 改为对应 split 的实际长度。

## 复现边界

| 项目 | 整理后的三个 baseline | 当前 Latte `train/eval_all.py` |
| --- | --- | --- |
| 推理 | vLLM | HF/PyTorch |
| 提示 | 既有纯文本 Search-o1 continuation | 现有初始化 chat wrapper + Search-o1 continuation；独立收尾也有 wrapper |
| 检索示例 | Exa 网页检索（也支持 Tavily） | Exa 网页检索（也支持 Tavily） |
| 解码示例 | 显式 greedy | 代码现有 greedy |
| 长度 | Vanilla/LongLLMLingua 上限 32768、最多 64 seq；Summary 16384、最多 32 seq（源码固定） | config 的 agent_max_length |
| 训练格式 | 不训练 | 现有转换器保留初始 chat wrapper |

不能仅凭方法名称合并结果。网页检索 provider 已统一；在线结果会随时间变化，需记录服务、日期与缓存。严格四方对比仍需统一 prompt policy、检索参数与缓存、后端/计时口径、长度上限及 checkpoint 训练格式。
