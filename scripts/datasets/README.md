# 数据集准备（只为采集 DSA top-2048 pattern）

本目录提供 **手动下载** + **最小预处理** 脚本，把不同来源的数据统一写成 `data/tasks/*.jsonl`，供 `inference/run_trace.py` 逐条跑并导出 DSA 轨迹。

## 统一的 tasks.jsonl 格式

每行一个 request（等价每条都先 `/clear`），字段：

- `dataset`：字符串
- `request_id`：字符串（可省略，runner 会自动补）
- 二选一：
  - `prompt`：单轮 user 文本
  - `messages`：多轮对话数组，元素为 `{ "role": "user"|"assistant", "content": "..." }`
- 可选：`max_new_tokens`

## D1: RULER（arXiv:2404.06654）

RULER 本身是合成生成器。本项目这里提供 **RULER-风格的最小 NIAH/retrieval 生成器**（固定 seed、可控长度）：

```bash
python scripts/datasets/gen_ruler_style.py \
  --out data/tasks/ruler_style.jsonl \
  --num-samples 200 \
  --context-words 50000 \
  --needles 4
```

## D2: LongBench v2（arXiv:2412.15204）

下载官方 `data.json`（无需 `datasets` 依赖）：

```bash
bash scripts/datasets/download_longbench_v2.sh
python scripts/datasets/prepare_longbench_v2.py \
  --in data/raw/longbench_v2/data.json \
  --out data/tasks/longbench_v2.jsonl \
  --max-samples 200
```

数据源：`https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json`

## D3: BurstGPT（arXiv:2401.17644）

BurstGPT 是 **serving trace**（请求/响应 token 长度、时间戳），不含真实 prompt 文本。本项目用它来生成“长度驱动”的可复现合成 prompts：

```bash
bash scripts/datasets/download_burstgpt.sh
python scripts/datasets/prepare_burstgpt.py \
  --in data/raw/burstgpt/BurstGPT_without_fails_1.csv \
  --out data/tasks/burstgpt_synth.jsonl \
  --max-samples 500
```

数据源（GitHub release v1.2）：`https://github.com/HPMLL/BurstGPT/releases/tag/v1.2`

## D4: ShareGPT（对话 serving workload）

这里使用一个常见公开镜像（OpenChat 维护的 ShareGPT4 数据快照）：

```bash
bash scripts/datasets/download_sharegpt.sh
python scripts/datasets/prepare_sharegpt.py \
  --in data/raw/sharegpt/openchat.train.json \
  --out data/tasks/sharegpt.jsonl \
  --max-samples 500
```

数据源：`https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/resolve/main/openchat.train.json`

