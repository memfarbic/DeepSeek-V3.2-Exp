from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


@dataclass(frozen=True)
class TraceConfig:
    out_path: str
    flush_every: int = 1
    decode_only: bool = True
    include_scores: bool = False
    include_block_stats: bool = True
    block_size: int = 128


@dataclass(frozen=True)
class TraceContext:
    request_id: Union[str, Sequence[str]]
    step_idx: int
    seq_len_current: int
    dataset: str = ""


_tls = threading.local()
_cfg: Optional[TraceConfig] = None
_fh = None
_write_count = 0


def is_enabled() -> bool:
    return _cfg is not None and _fh is not None


def enable(config: TraceConfig) -> None:
    global _cfg, _fh, _write_count
    disable()
    os.makedirs(os.path.dirname(config.out_path) or ".", exist_ok=True)
    _cfg = config
    _fh = open(config.out_path, "a", encoding="utf-8")
    _write_count = 0


def disable() -> None:
    global _cfg, _fh, _write_count
    if _fh is not None:
        try:
            _fh.flush()
        finally:
            _fh.close()
    _cfg = None
    _fh = None
    _write_count = 0


def set_context(ctx: TraceContext) -> None:
    _tls.ctx = ctx


def _get_context() -> Optional[TraceContext]:
    return getattr(_tls, "ctx", None)


def _quantiles_from_sorted(values_sorted: List[int], ps: Sequence[float]) -> Dict[str, int]:
    if not values_sorted:
        return {f"p{int(p * 100)}": 0 for p in ps}
    n = len(values_sorted)
    out: Dict[str, int] = {}
    for p in ps:
        idx = int(round((n - 1) * p))
        out[f"p{int(p * 100)}"] = int(values_sorted[max(0, min(n - 1, idx))])
    return out


def _stats_int(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "p50": 0, "p95": 0}
    values_sorted = sorted(values)
    return {
        "min": int(values_sorted[0]),
        "max": int(values_sorted[-1]),
        **_quantiles_from_sorted(values_sorted, ps=[0.5, 0.95]),
    }


def _stats_float_from_tensor(x: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
    }


def trace_indexer_topk(
    topk_indices: torch.Tensor,
    topk_scores: Optional[torch.Tensor],
    *,
    end_pos: int,
    seqlen: int,
    mask_is_none: bool,
) -> None:
    if not is_enabled():
        return
    cfg = _cfg
    assert cfg is not None
    if cfg.decode_only and not (seqlen == 1 and mask_is_none):
        return

    ctx = _get_context()
    if ctx is None:
        return

    if topk_indices.dim() != 3:
        return

    bsz = int(topk_indices.size(0))
    k = int(topk_indices.size(-1))
    query_pos = int(end_pos - 1)

    request_ids: Sequence[str]
    if isinstance(ctx.request_id, str):
        request_ids = [ctx.request_id] * bsz
    else:
        request_ids = list(ctx.request_id)
        if len(request_ids) != bsz:
            request_ids = [request_ids[0]] * bsz

    # Record per-sequence (request) in the batch.
    for seq_idx in range(bsz):
        # Decode path: seqlen == 1, use the last row.
        selected_pos: List[int] = [int(x) for x in topk_indices[seq_idx, -1].tolist()]
        unique_pos_count = int(len(set(selected_pos)))
        offsets: List[int] = [int(query_pos - p) for p in selected_pos]
        offset_stats = _stats_int(offsets)

        event: Dict[str, Any] = {
            "ts_us": int(time.time_ns() // 1_000),
            "dataset": ctx.dataset,
            "request_id": request_ids[seq_idx],
            "seq_idx": int(seq_idx),
            "step_idx": int(ctx.step_idx),
            "seq_len_current": int(ctx.seq_len_current),
            "query_pos": int(query_pos),
            "topk": int(k),
            "selected_token_pos": selected_pos,
            "stats": {
                "unique_token_pos_count": unique_pos_count,
                "offset": offset_stats,
            },
        }

        if topk_scores is not None:
            try:
                scores_1d = topk_scores[seq_idx, -1]
                event["stats"]["score"] = _stats_float_from_tensor(scores_1d)
                if cfg.include_scores:
                    event["topk_scores"] = [float(x) for x in scores_1d.tolist()]
            except Exception:
                pass

        if cfg.include_block_stats and cfg.block_size > 0:
            bs = int(cfg.block_size)
            block_ids = [int(p // bs) for p in selected_pos]
            counts: Dict[int, int] = {}
            for bid in block_ids:
                counts[bid] = counts.get(bid, 0) + 1
            touched_blocks = sorted(counts.keys())
            tokens_per_block = [counts[bid] for bid in touched_blocks]
            event["block"] = {
                "block_size": bs,
                "selected_block_ids": touched_blocks,
                "unique_blocks": int(len(touched_blocks)),
                "tokens_per_touched_block": {
                    "mean": float(sum(tokens_per_block) / max(1, len(tokens_per_block))),
                    **_stats_int(tokens_per_block),
                },
            }

        _write_event(event)


def _write_event(event: Dict[str, Any]) -> None:
    global _write_count
    if _fh is None or _cfg is None:
        return
    _fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    _write_count += 1
    if _cfg.flush_every > 0 and (_write_count % _cfg.flush_every) == 0:
        _fh.flush()

