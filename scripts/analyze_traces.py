import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _quantile(values_sorted: List[float], p: float) -> float:
    if not values_sorted:
        return 0.0
    n = len(values_sorted)
    idx = int(round((n - 1) * p))
    idx = max(0, min(n - 1, idx))
    return float(values_sorted[idx])


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    vs = sorted(values)
    return {
        "min": float(vs[0]),
        "p50": _quantile(vs, 0.50),
        "p95": _quantile(vs, 0.95),
        "p99": _quantile(vs, 0.99),
        "max": float(vs[-1]),
        "mean": float(sum(vs) / len(vs)),
    }


def _get_nested(d: Dict[str, Any], keys: List[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def analyze(paths: List[str]) -> Dict[str, Any]:
    num_events = 0
    request_ids = set()

    unique_token_pos_counts: List[float] = []
    offset_min: List[float] = []
    offset_p50: List[float] = []
    offset_p95: List[float] = []
    offset_max: List[float] = []

    unique_blocks: List[float] = []
    tokens_per_block_mean: List[float] = []
    tokens_per_block_p50: List[float] = []
    tokens_per_block_p95: List[float] = []

    for path in paths:
        for ev in _iter_jsonl(path):
            num_events += 1
            rid = ev.get("request_id")
            if isinstance(rid, str):
                request_ids.add(rid)

            utc = _get_nested(ev, ["stats", "unique_token_pos_count"])
            if isinstance(utc, (int, float)):
                unique_token_pos_counts.append(float(utc))

            off = _get_nested(ev, ["stats", "offset"])
            if isinstance(off, dict):
                for key, dst in [
                    ("min", offset_min),
                    ("p50", offset_p50),
                    ("p95", offset_p95),
                    ("max", offset_max),
                ]:
                    v = off.get(key)
                    if isinstance(v, (int, float)):
                        dst.append(float(v))

            blk = ev.get("block")
            if isinstance(blk, dict):
                ub = blk.get("unique_blocks")
                if isinstance(ub, (int, float)):
                    unique_blocks.append(float(ub))
                tpb = blk.get("tokens_per_touched_block")
                if isinstance(tpb, dict):
                    v = tpb.get("mean")
                    if isinstance(v, (int, float)):
                        tokens_per_block_mean.append(float(v))
                    v = tpb.get("p50")
                    if isinstance(v, (int, float)):
                        tokens_per_block_p50.append(float(v))
                    v = tpb.get("p95")
                    if isinstance(v, (int, float)):
                        tokens_per_block_p95.append(float(v))

    return {
        "num_events": int(num_events),
        "num_requests": int(len(request_ids)),
        "unique_token_pos_count": _summary(unique_token_pos_counts),
        "offset_min": _summary(offset_min),
        "offset_p50": _summary(offset_p50),
        "offset_p95": _summary(offset_p95),
        "offset_max": _summary(offset_max),
        "unique_blocks": _summary(unique_blocks),
        "tokens_per_touched_block_mean": _summary(tokens_per_block_mean),
        "tokens_per_touched_block_p50": _summary(tokens_per_block_p50),
        "tokens_per_touched_block_p95": _summary(tokens_per_block_p95),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_paths", nargs="+", required=True, help="One or more trace JSONL files.")
    p.add_argument("--out", dest="out_path", required=True, help="Output summary JSON file.")
    args = p.parse_args()

    summary = analyze(list(args.in_paths))
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

