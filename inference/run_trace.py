import json
import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

from generate import generate
from model import ModelArgs, Transformer

try:
    import dsa_trace
except Exception:
    dsa_trace = None


def _truncate_tokens(
    toks: List[int],
    *,
    max_len: int,
    strategy: str,
    tail_tokens: int = 512,
) -> List[int]:
    if len(toks) <= max_len:
        return toks
    if max_len <= 0:
        return []
    if strategy == "head":
        return toks[:max_len]
    if strategy == "tail":
        return toks[-max_len:]
    if strategy == "head_tail":
        tail = min(int(tail_tokens), max_len)
        head = max_len - tail
        return toks[:head] + toks[-tail:]
    raise ValueError(f"Unknown truncate strategy: {strategy}")


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True, help="Path to JSONL tasks file.")
    parser.add_argument("--trace-out", type=str, required=True, help="Output JSONL trace file.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=0, help="0 means auto (model.max_seq_len - max_new_tokens).")
    parser.add_argument("--truncate-strategy", type=str, default="head_tail", choices=["head", "tail", "head_tail"])
    parser.add_argument("--truncate-tail-tokens", type=int, default=512)
    parser.add_argument("--trace-scores", action="store_true")
    parser.add_argument("--trace-block-size", type=int, default=128)
    parser.add_argument("--trace-flush-every", type=int, default=1)
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(args.config, "r", encoding="utf-8") as f:
        margs = ModelArgs(**json.load(f))
    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    tasks = _load_tasks(args.tasks)
    if rank == 0:
        print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    if dsa_trace is None:
        raise RuntimeError("dsa_trace module is not available.")
    if rank == 0:
        dsa_trace.enable(
            dsa_trace.TraceConfig(
                out_path=args.trace_out,
                flush_every=int(args.trace_flush_every),
                decode_only=True,
                include_scores=bool(args.trace_scores),
                include_block_stats=True,
                block_size=int(args.trace_block_size),
            )
        )

    max_prompt_tokens = int(args.max_prompt_tokens)
    if max_prompt_tokens <= 0:
        max_prompt_tokens = int(model.max_seq_len - args.max_new_tokens)
    max_prompt_tokens = max(1, max_prompt_tokens)

    for i, task in enumerate(tasks):
        dataset = str(task.get("dataset", ""))
        request_id = str(task.get("request_id", f"{dataset}_{i}"))
        max_new_tokens = int(task.get("max_new_tokens", args.max_new_tokens))

        if "messages" in task:
            messages = task["messages"]
        else:
            prompt = str(task.get("prompt", ""))
            messages = [{"role": "user", "content": prompt}]

        prompt_tokens: List[int] = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_tokens = _truncate_tokens(
            prompt_tokens,
            max_len=max_prompt_tokens,
            strategy=str(args.truncate_strategy),
            tail_tokens=int(args.truncate_tail_tokens),
        )

        _ = generate(
            model,
            [prompt_tokens],
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            temperature=float(args.temperature),
            trace_request_id=request_id if rank == 0 else None,
            trace_dataset=dataset,
        )

        if rank == 0 and ((i + 1) % 10 == 0):
            print(f"Processed {i + 1}/{len(tasks)} tasks")

    if dsa_trace is not None:
        try:
            dsa_trace.disable()
        except Exception:
            pass
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

