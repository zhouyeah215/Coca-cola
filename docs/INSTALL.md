# 环境安装

命令从项目根目录（发布目录 `Coca-cola/`）执行。目标环境为 Linux x86_64、Python 3.12、NVIDIA GPU。

## 1. 创建环境

```bash
conda create -n latte python=3.12 -y
conda activate latte
python -m pip install --upgrade pip
```

三个 baseline 默认使用 Exa 网页检索，也支持 Tavily。发布版仅提供这两个网页检索服务。

## 2. 安装依赖

本机实验环境为 `torch==2.11.0+cu129`、`vllm==0.24.0+cu129`。下列 vLLM wheel 地址来自已安装包的 `direct_url.json`。需要宿主驱动支持 CUDA 12.9。

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
python -m pip install -r requirements/runtime.txt
python -m pip check
```

其他平台/CUDA 组合按 [vLLM 官方安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)选择匹配构建，不要任意升级 torch 后继续用旧 vLLM 二进制。这里的版本是本机环境观测，不是跨平台 lockfile；尚未在全新环境完整重装验证。

`requirements/runtime.txt` 已包含 Exa/Tavily SDK（通过 `online.txt`）。Latte 训练/评测另需：

```bash
python -m pip install -r requirements/train.txt
python -m nltk.downloader punkt punkt_tab
```

LongLLMLingua 代码与本地 Transformers 5 兼容补丁位于 `scripts/baselines/LLMLingua/`，入口直接加载该版本，无需另装 PyPI `llmlingua`。其余依赖已在 runtime requirements 中。上述四个入口不要求 DeepSpeed 或 ICAE；FlashAttention 不是必需项。

```bash
python -c 'import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__); print("CUDA:", torch.cuda.is_available())'
python -m pip freeze > /tmp/latte-environment.txt
```

## 3. 模型

准备 Hugging Face 格式的完整模型目录，包括配置、tokenizer、权重：

```bash
export AGENT_MODEL=/absolute/path/to/agent-model
export SUMMARY_MODEL=/absolute/path/to/summary-model
export LINGUA_MODEL=/absolute/path/to/causal-lm-compressor
```

当前提供的摘要提示与 prefill 用于 Qwen3.5-4B，可搭配 Qwen3.5-35B-A3B agent。LongLLMLingua 使用 causal LM，例如 Llama-2-7b 系列；不能用 LLMLingua-2 的 token classifier 替代。模型许可、下载权限、revision 与 tokenizer revision 应随实验记录，仓库不打包权重。

LLMSummary 服务与 agent 使用独立 GPU。LongLLMLingua 可以放 CPU；放 GPU 时 `cuda:N` 指 **CUDA_VISIBLE_DEVICES 重映射后的逻辑编号**。

## 4. QA 数据

文件名为 `data/QA_Datasets/<dataset>_<split>.json`，内容为 JSON 数组：

```json
[{"Question": "What is the capital of France?", "answer": ["Paris"]}]
```

主要数据集名称：`nq`、`triviaqa`、`hotpotqa`、`2wiki`、`musique`。单个答案也应放在字符串列表中。文件名的 `test` 不自动证明对应官方 test，应明确官方 split、转换方法与文件 hash。

发布包不包含 QA 数据，也未提供统一的下载/转换脚本；使用者需准备上述格式的数据。外部目录可通过 baseline launcher 的 `--data-dir /path/to/QA_Datasets` 指定。

```bash
sha256sum data/QA_Datasets/nq_test.json
```

## 5. 网页检索 API（默认）

```bash
export SEARCH_API_KEY=YOUR_EXA_API_KEY
# 三个 baseline 默认 --search_provider exa，也可显式指定。
```

使用 Tavily 时改为对应 key，并给三个 baseline 设置 `--search_provider tavily`；Latte 使用 `--search-provider tavily`。请使用同一 provider 对比四种方法。

baseline 也支持 `EXA_API_KEY` 和 `TAVILY_API_KEY`，只读取所选服务的变量；通用 `SEARCH_API_KEY` 优先。缺少凭据会在加载模型前报错。原 Latte 入口仍按命令显式传 `--search-api-key "$SEARCH_API_KEY"`。

网页模式返回 API 提供的标题、URL、正文片段或 highlights，不需要 Jina。在线结果会随时间变化，应保存运行日期与各次运行的搜索缓存。API key 由服务商发放，仓库不包含凭据。

## 故障定位

| 问题 | 检查 |
| --- | --- |
| 缺少 API key | 设置 `SEARCH_API_KEY` 或所选 provider 的 key 变量 |
| 缺少 Exa/Tavily SDK | `python -m pip install -r requirements/online.txt` |
| vLLM undefined symbol | torch/vLLM/CUDA wheel 是否匹配 |
| OOM | 减小 batch；增加 tensor parallel；模型权重本身也需装入显存 |
| QA 文件不存在 | 工作目录、数据文件名、`--data-dir` |
| run directory 已存在 | 换新目录，避免混合或覆盖结果 |
| 摘要连接失败 | 服务 `/health`、端口、served model name |
| Latte checkpoint 格式不符 | 必须有 `adapter.pt` 和 `compressor/adapter_config.json`，不能用 ICAE checkpoint |
