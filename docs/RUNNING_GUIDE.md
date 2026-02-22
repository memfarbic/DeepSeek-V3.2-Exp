# DSA Top-2048 轨迹采集 —— 运行指南

## 目录

- [0. 前提条件](#0-前提条件)
- [1. 环境准备](#1-环境准备)
- [2. 数据集准备](#2-数据集准备)
  - [D1: RULER 风格（合成 needle-in-a-haystack）](#d1-ruler-风格合成-needle-in-a-haystack)
  - [D2: LongBench v2](#d2-longbench-v2)
  - [D3: BurstGPT（长度驱动合成 prompt）](#d3-burstgpt长度驱动合成-prompt)
  - [D4: ShareGPT（对话）](#d4-sharegpt对话)
- [3. 运行轨迹采集](#3-运行轨迹采集)
  - [方式 A: 交互模式（generate.py）](#方式-a-交互模式generatepy)
  - [方式 B: 批量数据集模式（run_trace.py）](#方式-b-批量数据集模式run_tracepy)
- [4. 汇总分析](#4-汇总分析)
- [5. 完整端到端示例](#5-完整端到端示例)
- [6. CLI 参数速查表](#6-cli-参数速查表)
- [7. 输出文件格式说明](#7-输出文件格式说明)
- [8. 常见问题](#8-常见问题)

---

## 0. 前提条件

| 条件 | 说明 |
|------|------|
| GPU | 需要可运行 DeepSeek-V3.2-Exp 671B 的 GPU 集群（推荐 8xH200/H100） |
| 模型权重 | 已从 HuggingFace 下载 `deepseek-ai/DeepSeek-V3.2-Exp` 并转换为本 demo 的分片格式 |
| Python 依赖 | `inference/requirements.txt` 中列出（torch, transformers, safetensors, fast_hadamard_transform, tilelang） |

## 1. 环境准备

```bash
# 1) 安装基础依赖
cd inference
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install git+https://github.com/tile-ai/tilelang

# 2) 转换模型权重（如果尚未转换）
export EXPERTS=256
export MP=8                           # 根据 GPU 数量调整
export HF_CKPT_PATH=/data/models/deepseek-v3.2-exp
export SAVE_PATH=/data/models/deepseek-v3.2-exp-s
python convert.py \
  --hf-ckpt-path ${HF_CKPT_PATH} \
  --save-path ${SAVE_PATH} \
  --n-experts ${EXPERTS} \
  --model-parallel ${MP}

# 3) 回到仓库根目录
cd ..

# 4) 创建输出目录（首次运行需要）
mkdir -p data/raw data/tasks artifacts/traces artifacts/summary
```

## 2. 数据集准备

所有数据集最终都预处理为统一的 `data/tasks/*.jsonl` 格式，每行一个 JSON 对象：

```json
{
  "dataset": "ruler_style",
  "request_id": "ruler_style_0",
  "prompt": "You will be given a long context...",
  "max_new_tokens": 64
}
```

或多轮对话格式：

```json
{
  "dataset": "sharegpt",
  "request_id": "sharegpt_0",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "Tell me about..."}
  ],
  "max_new_tokens": 64
}
```

### D1: RULER 风格（合成 needle-in-a-haystack）

RULER（arXiv:2404.06654）本身是合成样本生成器。这里提供一个最小复刻生成器，无需克隆 NVIDIA/RULER 仓库。

```bash
python scripts/datasets/gen_ruler_style.py \
  --out data/tasks/ruler_style.jsonl \
  --num-samples 200 \
  --context-words 50000 \
  --needles 4 \
  --seed 1234 \
  --max-new-tokens 64
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-samples` | 200 | 生成样本数 |
| `--context-words` | 50000 | 每个样本的上下文词数（约 50K 词 ≈ 65K~80K tokens） |
| `--needles` | 4 | 每个样本插入的 needle 数量（分散在不同位置） |
| `--seed` | 1234 | 随机种子（保证可复现） |
| `--max-new-tokens` | 64 | 每条 request 的 decode 步数上限 |

**调整上下文长度**：增加 `--context-words` 可以测更长上下文。例如 `--context-words 100000` 约产出 130K~160K tokens 的 prompt。

### D2: LongBench v2

来源论文：arXiv:2412.15204。503 条真实长上下文多选题。

```bash
# 步骤 1：下载原始数据
bash scripts/datasets/download_longbench_v2.sh
# 下载到 data/raw/longbench_v2/data.json

# 步骤 2：预处理为 tasks.jsonl
python scripts/datasets/prepare_longbench_v2.py \
  --in data/raw/longbench_v2/data.json \
  --out data/tasks/longbench_v2.jsonl \
  --max-samples 200 \
  --seed 1234 \
  --max-context-chars 200000 \
  --max-new-tokens 64
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | 200 | 从全集中随机采样的数量 |
| `--max-context-chars` | 200000 | 截断单条 context 的最大字符数 |
| `--max-new-tokens` | 64 | decode 步数上限 |

### D3: BurstGPT（长度驱动合成 prompt）

来源论文：arXiv:2401.17644。BurstGPT 本身是 serving workload trace（CSV），不含真实 prompt 文本。本预处理脚本读取其 `Request tokens / Response tokens` 分布，用固定词表 + 固定种子生成等长合成 prompt。

```bash
# 步骤 1：下载 CSV（约 50MB）
bash scripts/datasets/download_burstgpt.sh
# 下载到 data/raw/burstgpt/BurstGPT_without_fails_1.csv

# 步骤 2：预处理
python scripts/datasets/prepare_burstgpt.py \
  --in data/raw/burstgpt/BurstGPT_without_fails_1.csv \
  --out data/tasks/burstgpt_synth.jsonl \
  --max-samples 500 \
  --seed 1234 \
  --max-prompt-words 60000 \
  --default-max-new-tokens 64
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | 500 | 从全集中随机采样 |
| `--prompt-words-scale` | 1.0 | 词数 = Request tokens * scale |
| `--max-prompt-words` | 60000 | 单条 prompt 最大词数上限 |

### D4: ShareGPT（对话）

来源：OpenChat 维护的 ShareGPT4 快照。

```bash
# 步骤 1：下载 JSON（约 200MB）
bash scripts/datasets/download_sharegpt.sh
# 下载到 data/raw/sharegpt/openchat.train.json

# 步骤 2：预处理
python scripts/datasets/prepare_sharegpt.py \
  --in data/raw/sharegpt/openchat.train.json \
  --out data/tasks/sharegpt.jsonl \
  --max-samples 500 \
  --seed 1234 \
  --max-turns 0 \
  --max-new-tokens 64
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | 500 | 从全集中随机采样 |
| `--max-turns` | 0 | 最大保留轮次，0 表示保留全部 |

## 3. 运行轨迹采集

### 方式 A: 交互模式（generate.py）

在原有交互模式上加 `--trace-out` 即可边聊天边采集。**每条 request 自动分配 `interactive_N` 的 request_id**，输入 `/clear` 清空对话历史。

```bash
cd inference
torchrun --nproc-per-node ${MP} generate.py \
  --ckpt-path ${SAVE_PATH} \
  --config config_671B_v3.2.json \
  --interactive \
  --max-new-tokens 200 \
  --temperature 0.6 \
  --trace-out ../artifacts/traces/interactive.jsonl
```

不加 `--trace-out` 则行为与原版 demo 完全一致（零额外开销）。

**交互模式新增 CLI 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trace-out` | `""` | 空串表示不启用 tracing |
| `--trace-scores` | false | 是否落盘完整 top-2048 scores 列表 |
| `--trace-block-size` | 128 | token→block 映射的 block 大小 |
| `--trace-flush-every` | 1 | 每 N 条事件 flush 一次 |
| `--trace-all-ranks` | false | 默认只 rank0 写盘，加此参数所有 rank 都写 |

### 方式 B: 批量数据集模式（run_trace.py）

专为数据集跑数设计：从 `data/tasks/*.jsonl` 逐条读取，**每条 request 独立（等价自动 `/clear`）**，强制 `batch_size=1` 以保证可复现。

```bash
cd inference
torchrun --nproc-per-node ${MP} run_trace.py \
  --ckpt-path ${SAVE_PATH} \
  --config config_671B_v3.2.json \
  --tasks ../data/tasks/ruler_style.jsonl \
  --trace-out ../artifacts/traces/ruler_style.jsonl \
  --max-new-tokens 64 \
  --temperature 0.0
```

**run_trace.py 完整 CLI 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt-path` | 必填 | 转换后的模型权重目录 |
| `--config` | 必填 | `config_671B_v3.2.json` 路径 |
| `--tasks` | 必填 | tasks JSONL 文件路径 |
| `--trace-out` | 必填 | 输出轨迹 JSONL 文件路径 |
| `--max-new-tokens` | 64 | 全局默认 decode 步数（每条 task 可覆盖） |
| `--temperature` | 0.0 | 采样温度（0 = greedy，最可复现） |
| `--max-prompt-tokens` | 0 | 最大 prompt token 数，0 = 自动（model.max_seq_len - max_new_tokens） |
| `--truncate-strategy` | `head_tail` | 超长 prompt 截断策略：`head` / `tail` / `head_tail` |
| `--truncate-tail-tokens` | 512 | `head_tail` 策略中保留尾部的 token 数 |
| `--trace-scores` | false | 是否落盘完整 top-2048 scores |
| `--trace-block-size` | 128 | block 映射的 block 大小 |
| `--trace-flush-every` | 1 | 每 N 条事件 flush 一次 |

**截断策略说明**：

当 prompt 长度超过 `--max-prompt-tokens` 时：

- `head`：保留前 N 个 token
- `tail`：保留后 N 个 token
- `head_tail`（默认）：保留前 `N - tail_tokens` 个 + 后 `tail_tokens` 个，适合"保留系统指令/问题尾巴"的长上下文场景

## 4. 汇总分析

跑完轨迹后，用 `scripts/analyze_traces.py` 聚合统计。支持多个 JSONL 文件合并分析。

```bash
# 单个数据集
python scripts/analyze_traces.py \
  --in artifacts/traces/ruler_style.jsonl \
  --out artifacts/summary/ruler_style.json

# 多个数据集合并
python scripts/analyze_traces.py \
  --in artifacts/traces/ruler_style.jsonl \
      artifacts/traces/longbench_v2.jsonl \
      artifacts/traces/burstgpt_synth.jsonl \
      artifacts/traces/sharegpt.jsonl \
  --out artifacts/summary/all_datasets.json
```

输出的 summary JSON 结构：

```json
{
  "num_events": 12800,
  "num_requests": 200,
  "unique_token_pos_count": {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "offset_min":             {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "offset_p50":             {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "offset_p95":             {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "offset_max":             {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "unique_blocks":                    {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "tokens_per_touched_block_mean":    {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "tokens_per_touched_block_p50":     {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...},
  "tokens_per_touched_block_p95":     {"min": ..., "p50": ..., "p95": ..., "p99": ..., "max": ..., "mean": ...}
}
```

每个字段是"跨所有 step 事件聚合"的分位数统计，含义见下方 [7. 输出文件格式说明](#7-输出文件格式说明)。

## 5. 完整端到端示例

以 RULER 风格数据集为例的一键脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

export SAVE_PATH=/data/models/deepseek-v3.2-exp-s
export MP=8
export CONFIG=inference/config_671B_v3.2.json

# ---- 步骤 1: 生成数据集 ----
python scripts/datasets/gen_ruler_style.py \
  --out data/tasks/ruler_style.jsonl \
  --num-samples 50 \
  --context-words 50000 \
  --needles 4

# ---- 步骤 2: 采集轨迹 ----
cd inference
torchrun --nproc-per-node ${MP} run_trace.py \
  --ckpt-path ${SAVE_PATH} \
  --config ${CONFIG} \
  --tasks ../data/tasks/ruler_style.jsonl \
  --trace-out ../artifacts/traces/ruler_style.jsonl \
  --max-new-tokens 64 \
  --temperature 0.0
cd ..

# ---- 步骤 3: 汇总统计 ----
python scripts/analyze_traces.py \
  --in artifacts/traces/ruler_style.jsonl \
  --out artifacts/summary/ruler_style.json

echo "Done. Summary:"
cat artifacts/summary/ruler_style.json
```

跑完全部四个数据集只需把步骤 1/2 重复四次（替换不同的 `--tasks` / `--trace-out`），最后可以一起汇总或分别汇总。

## 6. CLI 参数速查表

### generate.py（交互 + trace）

```
--ckpt-path           模型权重目录（必填）
--config              config JSON 路径（必填）
--interactive         交互模式
--input-file          批量 prompt 文件（与 --interactive 二选一）
--max-new-tokens      最大生成 token 数（默认 200）
--temperature         采样温度（默认 0.6）
--trace-out           轨迹输出 JSONL 路径（空 = 不启用）
--trace-scores        落盘完整 score 列表
--trace-block-size    block 映射大小（默认 128）
--trace-flush-every   每 N 事件 flush（默认 1）
--trace-all-ranks     所有 rank 都写盘
```

### run_trace.py（批量数据集 + trace）

```
--ckpt-path           模型权重目录（必填）
--config              config JSON 路径（必填）
--tasks               tasks JSONL 路径（必填）
--trace-out           轨迹输出 JSONL 路径（必填）
--max-new-tokens      默认 decode 步数（默认 64）
--temperature         采样温度（默认 0.0）
--max-prompt-tokens   最大 prompt token 数（0 = 自动）
--truncate-strategy   截断策略：head / tail / head_tail（默认 head_tail）
--truncate-tail-tokens  head_tail 尾部保留量（默认 512）
--trace-scores        落盘完整 score 列表
--trace-block-size    block 映射大小（默认 128）
--trace-flush-every   每 N 事件 flush（默认 1）
```

### analyze_traces.py（汇总分析）

```
--in                  一个或多个轨迹 JSONL 文件（必填，支持多路径）
--out                 汇总 JSON 输出路径（必填）
```

### 数据集生成/预处理脚本

| 脚本 | 输入 | 输出 |
|------|------|------|
| `gen_ruler_style.py` | 无（合成） | `data/tasks/ruler_style.jsonl` |
| `download_longbench_v2.sh` + `prepare_longbench_v2.py` | HuggingFace 直链 JSON | `data/tasks/longbench_v2.jsonl` |
| `download_burstgpt.sh` + `prepare_burstgpt.py` | GitHub release CSV | `data/tasks/burstgpt_synth.jsonl` |
| `download_sharegpt.sh` + `prepare_sharegpt.py` | HuggingFace 直链 JSON | `data/tasks/sharegpt.jsonl` |

## 7. 输出文件格式说明

### 轨迹文件（JSONL，一行一个 step 事件）

```json
{
  "ts_us": 1708612345678901,
  "dataset": "ruler_style",
  "request_id": "ruler_style_0",
  "seq_idx": 0,
  "step_idx": 3,
  "seq_len_current": 5003,
  "query_pos": 5003,
  "topk": 2048,
  "selected_token_pos": [0, 42, 107, 256, ...],
  "stats": {
    "unique_token_pos_count": 2048,
    "offset": {
      "min": 0,
      "max": 4999,
      "p50": 2501,
      "p95": 4750
    },
    "score": {
      "min": -12.5,
      "mean": -3.2,
      "max": 0.8
    }
  },
  "block": {
    "block_size": 128,
    "selected_block_ids": [0, 1, 2, 8, 15, ...],
    "unique_blocks": 312,
    "tokens_per_touched_block": {
      "mean": 6.56,
      "min": 1,
      "max": 128,
      "p50": 3,
      "p95": 22
    }
  }
}
```

**字段含义：**

| 字段 | 含义 |
|------|------|
| `ts_us` | 事件微秒级时间戳 |
| `dataset` | 数据集标签 |
| `request_id` | 请求唯一 ID |
| `seq_idx` | batch 内序列索引（通常为 0） |
| `step_idx` | decode step 编号（第一个 decode step 为 0，prefill step 为负数） |
| `seq_len_current` | 当前已有序列长度（= prompt_len + step_idx，在 decode 阶段） |
| `query_pos` | 当前 query token 的绝对位置（= end_pos - 1） |
| `topk` | 实际选出的 top-k 数量（min(2048, seq_len_current)） |
| `selected_token_pos` | top-2048 选中的历史 token **绝对位置**列表 |
| `stats.unique_token_pos_count` | 去重后位置数（通常 = topk，但理论上可 < topk） |
| `stats.offset` | 距离分布：`offset = query_pos - token_pos`（值越大 = 越远的历史 token） |
| `stats.score` | top-2048 scores 的 min/mean/max |
| `block.block_size` | 用于映射的 block 大小 |
| `block.selected_block_ids` | 去重后的 block ID 列表 |
| `block.unique_blocks` | 触达的唯一 block 数 |
| `block.tokens_per_touched_block` | 每个被触达 block 中选中的 token 数分布 |

### 汇总文件（JSON）

对轨迹文件中所有 step 事件做跨 step 聚合。每个字段是 `{min, p50, p95, p99, max, mean}` 格式的分位数统计。例如 `unique_blocks.p95` 表示"95% 的 decode step 中 unique_blocks 的值"。

## 8. 常见问题

**Q: 不启用 tracing 时，对原有推理有性能影响吗？**

没有。`dsa_trace` 模块通过 `try/except` 导入，不传 `--trace-out` 时 `is_enabled()` 永远返回 `False`，`trace_indexer_topk()` 立即返回，不会执行任何计算。原始 `topk()` 调用现在返回 `(values, indices)` 而非仅 `[1]`，但 PyTorch `topk` 默认就返回 named tuple，不会有额外开销。

**Q: prefill 阶段也会记录吗？**

默认不会。`TraceConfig.decode_only=True`（默认）时，只在 `seqlen==1 && mask is None`（decode 路径）记录。如果你需要记录 prefill 阶段的 top-2048 选择（数据量极大），可以在代码中设置 `decode_only=False`。

**Q: 多 GPU 并行时怎么处理？**

默认只 rank 0 写盘（`--trace-rank0-only`，generate.py 中为默认行为）。由于 `Indexer.forward()` 中有 `dist.broadcast` 确保所有 rank 的 `topk_indices` 一致，rank 0 的数据即代表全局结果。如需全 rank 写盘（调试用），generate.py 加 `--trace-all-ranks`。

**Q: 轨迹文件很大怎么办？**

每条 step 事件中 `selected_token_pos` 列表有 2048 个整数（约 10~15 KB JSON）。200 条 request x 64 decode steps = 12800 条事件 ≈ 130~190 MB。压缩方法：
- 设置 `--trace-flush-every 100`（减少 IO 次数）
- 设置 `include_block_stats=False`（去掉 block 字段）
- 后处理时只解析 `stats` 字段，跳过 `selected_token_pos`

**Q: 如何只跑少量样本做 smoke test？**

数据集脚本都有 `--max-samples` / `--num-samples` 参数，设为 3~5 即可。例如：

```bash
python scripts/datasets/gen_ruler_style.py \
  --out data/tasks/ruler_smoke.jsonl \
  --num-samples 3 \
  --context-words 5000

cd inference
torchrun --nproc-per-node ${MP} run_trace.py \
  --ckpt-path ${SAVE_PATH} \
  --config config_671B_v3.2.json \
  --tasks ../data/tasks/ruler_smoke.jsonl \
  --trace-out ../artifacts/traces/ruler_smoke.jsonl \
  --max-new-tokens 8
```

**Q: `step_idx` 的含义？**

`step_idx = cur_pos - prompt_len`。在 prefill 阶段（`cur_pos < prompt_len`）step_idx 为负数；第一个真正的 decode step（生成第一个新 token）时 step_idx = 0。由于默认只记录 decode（`decode_only=True`），你看到的 step_idx 通常从 0 开始。
