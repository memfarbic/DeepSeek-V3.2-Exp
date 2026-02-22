# DSA Top-2048 轨迹插桩 —— 设计文档

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 整体架构](#2-整体架构)
- [3. 插桩设计](#3-插桩设计)
  - [3.1 插桩点定位：Indexer.forward()](#31-插桩点定位indexerforward)
  - [3.2 tracer 模块：dsa_trace.py](#32-tracer-模块dsa_tracepy)
  - [3.3 上下文传递：generate.py 改动](#33-上下文传递generatepy-改动)
  - [3.4 批量 runner：run_trace.py](#34-批量-runnerrun_tracepy)
- [4. 数据集脚本设计](#4-数据集脚本设计)
- [5. 汇总分析工具](#5-汇总分析工具)
- [6. 改动文件清单](#6-改动文件清单)
- [7. 设计决策与权衡](#7-设计决策与权衡)

---

## 1. 背景与目标

DeepSeek-V3.2-Exp 引入了 **DeepSeek Sparse Attention（DSA）**：每个 decode step 中，indexer 模块为每个 query token 从全部历史 token 中选出 top-2048 个"最相关"的位置，只对这些位置做 full attention。这是一种 **细粒度稀疏注意力**机制。

**本次插桩的唯一目标**：在不修改模型计算路径的前提下，导出可复现的、按 request/step 组织的 DSA top-2048 token 选取轨迹，用于后续分析稀疏访问 pattern（如 offset 分布、block 分散度、访问热点等）。

**明确不做的事情**：
- 不接入 vLLM / SGLang
- 不涉及 PagedAttention 或外部 KV 系统
- 不涉及 KV 搬运/取数延迟测量
- 不涉及 prefix cache 交互统计
- 不实现任何 KV 管理策略

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据集准备层                               │
│                                                                   │
│  gen_ruler_style.py    prepare_longbench_v2.py                   │
│  prepare_burstgpt.py   prepare_sharegpt.py                       │
│           │                    │                                  │
│           └──── data/tasks/*.jsonl ────┘                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        运行层                                     │
│                                                                   │
│  run_trace.py ──── 逐条读 tasks ──── 调用 generate() ────┐       │
│                                                             │       │
│  generate.py ──── decode loop ──── set_context() ────┐     │       │
│                                                       │     │       │
│  model.py → Indexer.forward()                         │     │       │
│      │                                                │     │       │
│      └── topk_scores, topk_indices = topk(...)        │     │       │
│      └── dsa_trace.trace_indexer_topk(...)  ◄─────────┘     │       │
│                      │                                       │       │
│                      ▼                                       │       │
│              dsa_trace.py                                    │       │
│              (TraceWriter)                                   │       │
│                      │                                       │       │
│                      ▼                                       │       │
│           artifacts/traces/*.jsonl                            │       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        分析层                                     │
│                                                                   │
│  analyze_traces.py                                               │
│           │                                                       │
│           ▼                                                       │
│  artifacts/summary/*.json                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 插桩设计

### 3.1 插桩点定位：Indexer.forward()

DeepSeek-V3.2-Exp 的 DSA 全部发生在 `inference/model.py` 的 `Indexer` 类中。其 `forward()` 方法的核心逻辑（简化）：

```python
# inference/model.py, class Indexer, forward()
index_score = fp8_index(q_fp8, weights, k, k_s)   # [bsz, seqlen, end_pos]
if mask is not None:
    index_score += mask
topk_scores, topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)
#           ↑ 插桩点：这里产出的 topk_indices 就是 selected_token_pos
return topk_indices
```

**关键观察**：

1. `topk_indices` 的 shape 为 `[bsz, seqlen, topk]`，其中 `topk = min(index_topk, end_pos)`，`index_topk` 在 config 中为 2048。
2. 在 decode 路径（`seqlen == 1, mask is None`），`topk_indices[:, 0, :]` 就是长度为 2048 的绝对位置索引，范围 `[0, end_pos - 1]`。
3. 在 prefill 路径（`seqlen > 1, mask is not None`），每个 query position 都会产出一组 top-2048，数据量为 `bsz * seqlen * 2048`——**默认不记录**。

**改动内容**：

原代码只取 `[1]`（indices）：
```python
topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
```

改为同时取 values 和 indices：
```python
topk_scores, topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)
```

然后调用 tracer（整个调用包裹在 `try/except` 中，保证即使 tracer 出错也不影响推理）：
```python
if dsa_trace is not None:
    try:
        dsa_trace.trace_indexer_topk(
            topk_indices, topk_scores,
            end_pos=end_pos, seqlen=seqlen, mask_is_none=(mask is None),
        )
    except Exception:
        pass
```

**对原始计算路径的影响**：零。`torch.topk()` 本身就返回 `(values, indices)` namedtuple，原代码取 `[1]` 只是忽略了 values；改为解包不会产生额外计算。tracer 未启用时，`dsa_trace is None`（如果 import 失败）或 `is_enabled() == False`（未调用 `enable()`），`trace_indexer_topk()` 在第一行就返回。

### 3.2 tracer 模块：dsa_trace.py

这是一个纯 Python 模块，不依赖 torch 之外的任何包。设计为全局单例模式（模块级状态），避免需要在模型构造函数中传递 tracer 实例。

#### 核心数据结构

```python
@dataclass(frozen=True)
class TraceConfig:
    out_path: str             # 输出 JSONL 文件路径
    flush_every: int = 1      # 每 N 条事件 flush 一次
    decode_only: bool = True  # 只记录 decode 路径
    include_scores: bool = False  # 是否落盘完整 scores 列表
    include_block_stats: bool = True  # 是否计算 block 级统计
    block_size: int = 128     # block 映射的块大小

@dataclass(frozen=True)
class TraceContext:
    request_id: Union[str, Sequence[str]]  # 当前 request 的 ID
    step_idx: int             # decode step 编号
    seq_len_current: int      # 当前已有序列长度
    dataset: str = ""         # 数据集标签
```

#### 模块级状态

```python
_tls = threading.local()     # 线程局部存储，用于 TraceContext
_cfg: Optional[TraceConfig]  # 全局配置
_fh                          # 输出文件句柄
_write_count                 # 已写事件计数（用于 flush 控制）
```

使用 `threading.local()` 存储 `TraceContext` 是为了安全性（虽然当前 demo 是单线程的，但不排除未来多线程场景）。

#### API 设计

| 函数 | 调用者 | 说明 |
|------|--------|------|
| `enable(config)` | 启动时（generate.py / run_trace.py） | 打开输出文件，设置配置 |
| `disable()` | 结束时 | flush 并关闭文件 |
| `is_enabled()` | trace_indexer_topk 内部 | 快速检查是否启用 |
| `set_context(ctx)` | generate() decode 循环中 | 设置当前 step 的元信息 |
| `trace_indexer_topk(...)` | Indexer.forward() | 核心记录函数 |

#### trace_indexer_topk 内部逻辑

```
1. is_enabled() == False → 立即返回
2. decode_only 且不在 decode 路径 → 立即返回
3. 获取 TraceContext → 为空则返回
4. 遍历 batch 中的每个 sequence：
   a. 取 topk_indices[seq_idx, -1] → selected_token_pos 列表
   b. 计算 unique_token_pos_count = len(set(selected_pos))
   c. 计算 offsets = [query_pos - p for p in selected_pos]
   d. 计算 offset_stats: min, max, p50, p95
   e. 可选：计算 score_stats: min, mean, max
   f. 可选：计算 block_stats:
      - block_id = token_pos // block_size
      - unique_blocks, tokens_per_touched_block 分布
   g. 序列化为 JSON → 写入文件
```

所有统计都在写盘前**就地计算**（纯 Python list 操作 + 排序），不引入额外 GPU 操作。`topk_indices` 已经在 CPU 上（通过 `.tolist()` 转换），不会触发 GPU 同步。

#### 事件 JSON schema

每条事件包含以下字段（详见运行指南）：

```
ts_us, dataset, request_id, seq_idx, step_idx, seq_len_current,
query_pos, topk, selected_token_pos[],
stats.unique_token_pos_count, stats.offset.{min,max,p50,p95},
stats.score.{min,mean,max},
block.{block_size, selected_block_ids[], unique_blocks,
       tokens_per_touched_block.{mean,min,max,p50,p95}}
```

### 3.3 上下文传递：generate.py 改动

原始 `generate()` 函数签名只有模型和 token 参数。改动后新增两个可选参数：

```python
def generate(
    model, prompt_tokens, max_new_tokens, eos_id,
    temperature=1.0,
    trace_request_id: Optional[str] = None,   # 新增
    trace_dataset: str = "",                    # 新增
) -> List[List[int]]:
```

在 decode 循环中，**每次调用 `model.forward()` 之前**设置 TraceContext：

```python
for cur_pos in range(min(prompt_lens), total_len):
    if trace_request_id is not None and dsa_trace is not None and dsa_trace.is_enabled():
        dsa_trace.set_context(
            dsa_trace.TraceContext(
                request_id=trace_request_id,
                step_idx=int(cur_pos - prompt_len0),
                seq_len_current=int(cur_pos),
                dataset=trace_dataset,
            )
        )
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    ...
```

**为什么在 `model.forward()` 之前设 context？**

因为 `model.forward()` 内部会调用 61 层 `Block.forward()` → `MLA.forward()` → `Indexer.forward()`。在 indexer 中 `trace_indexer_topk()` 需要读取当前的 `TraceContext` 来知道 `request_id` 和 `step_idx`。context 通过模块级 thread-local 变量传递，避免修改 `Transformer` / `Block` / `MLA` / `Indexer` 的函数签名链。

**交互模式的改动**：

`main()` 函数新增 `trace_out` 等参数。在交互模式中：
- 自动分配 `request_id = f"interactive_{counter}"`
- 每次用户输入后 counter 递增
- `/clear` 只清空 messages，不影响 counter（保证 request_id 全局唯一）

**`--trace-out` 为空时**：`trace_request_id` 不传给 `generate()`，`set_context()` 不会被调用，`trace_indexer_topk()` 在检查 `is_enabled()` 时返回 False —— 完全零开销。

### 3.4 批量 runner：run_trace.py

新增文件，专为数据集批量跑数设计。与 `generate.py` 的关键区别：

| 特性 | generate.py | run_trace.py |
|------|-------------|--------------|
| 输入 | 交互 / 文本文件 | JSONL tasks 文件 |
| 批处理 | batch_size 可 > 1（无 tracing 时） | **强制 batch_size = 1** |
| 请求隔离 | 交互模式可多轮累积 | 每条 task 独立（等价 `/clear`） |
| tracing | 可选 | **始终启用** |
| 截断策略 | 无 | 支持 head / tail / head_tail |

**截断策略**的必要性：数据集中（尤其 LongBench v2、RULER）可能包含超过 `model.max_seq_len` 的 prompt。`run_trace.py` 提供三种截断方式：

- `head`：保留前 N token（丢弃尾部上下文）
- `tail`：保留后 N token（丢弃头部上下文）
- `head_tail`（默认）：保留前 `N - tail_tokens` + 后 `tail_tokens`（兼顾系统 prompt 和 query 尾巴）

`max_prompt_tokens` 默认为 `model.max_seq_len - max_new_tokens`，确保 prompt + decode 长度不超出模型限制。

## 4. 数据集脚本设计

### 统一输出格式

所有预处理脚本输出统一的 `tasks.jsonl`，每行一个 JSON 对象，必须包含 `dataset` 字段，二选一提供 `prompt`（单轮）或 `messages`（多轮）。

这个设计使 `run_trace.py` 无需关心数据来源，只需解析统一格式。

### D1: RULER 风格生成器（gen_ruler_style.py）

RULER 本身是一个完整的评测框架（含 NIAH、variable tracking、aggregation 等多类任务），依赖较重。本项目只需要"足够长的上下文 + 分散在不同位置的 needle"来触发 DSA 的非局部访问，因此做了最小复刻：

- 用固定词表（`w00000` ~ `w19999`）生成可复现的随机上下文
- 在随机位置插入 `NEEDLE_j: <secret>`（secret 为随机字符串）
- prompt 格式：`Context: ...\nQuestion: What is the secret value?\nAnswer:`
- 所有随机操作使用 `random.Random(seed)` 确保可复现

### D2: LongBench v2（prepare_longbench_v2.py）

- 直接下载官方 `data.json`（503 条），无需 `datasets` 库
- 拼接为：`Context: {context}\nQuestion: {question}\nChoices: A) ... B) ... C) ... D) ...\nAnswer:`
- `--max-context-chars` 控制单条 context 字符数上限（避免单条超大）
- 随机采样 `--max-samples` 条

### D3: BurstGPT（prepare_burstgpt.py）

BurstGPT 是 serving trace，只有 `(timestamp, request_tokens, response_tokens, model, log_type)` 等元数据，没有真实文本。处理方式：

- 读取 CSV 中每行的 `Request tokens` → 作为合成 prompt 的目标词数
- 用 `_make_filler_words(n_words, seed=base_seed+row_idx)` 生成确定性填充文本
- `Response tokens` → 设为该 task 的 `max_new_tokens`
- 这样保留了 BurstGPT 的"请求长度分布"特征，同时保证可复现

### D4: ShareGPT（prepare_sharegpt.py）

- 支持多种 JSON 格式（array / jsonl，messages / conversations 键名，role / from 角色标记）
- 自动归一化为 `[{"role": "user", "content": ...}, {"role": "assistant", ...}, ...]`
- 过滤掉不以 user 开头的对话（避免 chat template 报错）
- `--max-turns` 可限制保留轮次（0 = 全保留）

## 5. 汇总分析工具

`scripts/analyze_traces.py` 读取一个或多个轨迹 JSONL 文件，对所有 step 事件做**跨 step 聚合**，输出每个指标的 `{min, p50, p95, p99, max, mean}` 分位数统计。

聚合的指标：

| 指标 | 含义 | 用途 |
|------|------|------|
| `unique_token_pos_count` | 每 step 选中的去重位置数 | 近似 topk（通常 = 2048），但如果有重复则 < topk |
| `offset_min` | 每 step 中 offset 最小值（最近的选中 token） | 局部性分析 |
| `offset_p50` | 每 step 中 offset 中位数 | 典型访问距离 |
| `offset_p95` | 每 step 中 offset 95 分位 | 长距离访问程度 |
| `offset_max` | 每 step 中 offset 最大值（最远的选中 token） | 是否触达序列开头 |
| `unique_blocks` | 每 step 触达的唯一 block 数 | 分散度（越大 = 越分散） |
| `tokens_per_touched_block_mean` | 每 step 中"每个 block 被选中多少 token"的均值 | 冗余度（越小 = 越冗余） |
| `tokens_per_touched_block_p50` | 同上的中位数 | |
| `tokens_per_touched_block_p95` | 同上的 95 分位 | 是否存在热点 block |

## 6. 改动文件清单

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `inference/dsa_trace.py` | 196 | tracer 核心模块 |
| `inference/run_trace.py` | 153 | 批量数据集 trace runner |
| `scripts/analyze_traces.py` | 133 | 轨迹汇总分析工具 |
| `scripts/datasets/README.md` | 69 | 数据集准备说明 |
| `scripts/datasets/gen_ruler_style.py` | 65 | RULER 风格合成生成器 |
| `scripts/datasets/prepare_longbench_v2.py` | 63 | LongBench v2 预处理 |
| `scripts/datasets/prepare_burstgpt.py` | 81 | BurstGPT 预处理 |
| `scripts/datasets/prepare_sharegpt.py` | 96 | ShareGPT 预处理 |
| `scripts/datasets/download_longbench_v2.sh` | 14 | LongBench v2 下载脚本 |
| `scripts/datasets/download_burstgpt.sh` | 23 | BurstGPT 下载脚本 |
| `scripts/datasets/download_sharegpt.sh` | 14 | ShareGPT 下载脚本 |

### 修改文件

| 文件 | 改动行数 | 说明 |
|------|----------|------|
| `inference/model.py` | +17 -1 | Indexer.forward() 插桩 + dsa_trace 导入 |
| `inference/generate.py` | +85 -5 | trace 上下文传递 + CLI 参数 + enable/disable |

### 目录结构

```
DeepSeek-V3.2-Exp/
├── inference/
│   ├── model.py              # 已改动：Indexer 插桩
│   ├── generate.py           # 已改动：trace 上下文 + CLI
│   ├── dsa_trace.py          # 新增：tracer 模块
│   ├── run_trace.py          # 新增：批量 runner
│   ├── kernel.py             # 未改动
│   ├── convert.py            # 未改动
│   ├── config_671B_v3.2.json # 未改动
│   ├── requirements.txt      # 未改动
│   └── README.md             # 未改动
├── scripts/
│   ├── analyze_traces.py     # 新增：汇总分析
│   └── datasets/
│       ├── README.md
│       ├── gen_ruler_style.py
│       ├── prepare_longbench_v2.py
│       ├── prepare_burstgpt.py
│       ├── prepare_sharegpt.py
│       ├── download_longbench_v2.sh
│       ├── download_burstgpt.sh
│       └── download_sharegpt.sh
├── data/                     # 运行时生成
│   ├── raw/                  # 原始下载数据
│   └── tasks/                # 预处理后的 tasks JSONL
├── artifacts/                # 运行时生成
│   ├── traces/               # 轨迹 JSONL
│   └── summary/              # 汇总 JSON
└── docs/
    ├── RUNNING_GUIDE.md
    └── DESIGN.md
```

## 7. 设计决策与权衡

### 决策 1：模块级全局状态 vs 实例传递

**选择**：模块级全局状态（`_cfg`, `_fh`, `_tls`）。

**理由**：如果用实例传递，需要修改 `Transformer → Block → MLA → Indexer` 整条调用链的函数签名，改动面大且容易引入 bug。模块级状态只需在顶层 `enable()` + `set_context()`，在 `Indexer.forward()` 中调用 `trace_indexer_topk()` 即可。

**风险**：不支持同一进程中同时运行多个 tracer 实例（但当前场景不需要）。

### 决策 2：默认只记录 decode 路径

**选择**：`decode_only=True`，只在 `seqlen == 1 and mask is None` 时记录。

**理由**：prefill 阶段每个 query position 都会产出 top-2048，对于一个 10K token 的 prompt，一次 prefill 就会产出 10K x 2048 = 20M 个位置 —— 写盘开销巨大。decode 阶段每 step 只产出 2048 个位置，可控。

### 决策 3：batch_size = 1 强制

**选择**：开启 tracing 时强制 `batch_size == 1`（`run_trace.py` 逐条跑）。

**理由**：原始 demo 支持 batch，但不同 prompt 长度的 batch 中 `step_idx` 和 `seq_len_current` 的语义因 padding 而变得复杂。强制 batch=1 保证 `step_idx = cur_pos - prompt_len` 有唯一且清晰的含义。对于采集轨迹来说，吞吐量不是瓶颈。

### 决策 4：block 映射作为可选派生

**选择**：在 `dsa_trace.py` 中用纯数学映射 `block_id = token_pos // block_size` 计算 block 级统计，不依赖任何 vLLM 概念。

**理由**：你说"只需统计稀疏 token 选取 pattern"。但 block 粒度分析（`unique_blocks`、`tokens_per_touched_block`）对后续设计 KV 管理策略非常有用，且计算开销极小（纯 Python dict 操作），默认开启。

### 决策 5：try/except 包裹所有 tracer 调用

**选择**：`import dsa_trace` 和所有 `trace_indexer_topk()` 调用都包裹在 `try/except` 中。

**理由**：保证即使 tracer 代码有 bug 或 import 失败，模型推理不受任何影响。这是插桩代码的基本原则：**观测不应影响被观测系统**。

### 决策 6：JSONL 而非 Parquet/Arrow

**选择**：输出为 JSONL（每行一个 JSON 对象）。

**理由**：
- 不引入额外依赖（pandas / pyarrow）
- append-friendly（断电不丢之前的数据）
- 人类可读，方便 `head -1 trace.jsonl | python -m json.tool` 快速检查
- 后续分析可用任意工具（pandas、jq、Python 脚本）

**代价**：文件体积较大（JSON 相比二进制格式冗余），但对于采集轨迹的场景（不是实时 serving 日志）可以接受。需要时可后处理转 Parquet。

### 决策 7：数据集脚本不引入新依赖

**选择**：所有数据集脚本只使用 Python 标准库（json, csv, random, argparse, string）。

**理由**：你选择了"手动下载"方式，不希望新增 `datasets` 等依赖。下载脚本只用 `curl`，预处理脚本只用标准库。

### 决策 8：RULER 用最小复刻而非引入完整框架

**选择**：自行实现一个 65 行的 RULER-风格 NIAH 生成器，而非 `pip install` 或 clone NVIDIA/RULER。

**理由**：RULER 的完整代码包含多种任务、多种模型模板、Docker 依赖等，远超本项目需求。本项目只需要"长上下文 + 分散 needle"来触发 DSA 的非局部访问，一个最小生成器就够了，且完全可复现（固定 seed + 固定词表）。
