import argparse
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_records(path: str) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for x in data:
                yield x
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _normalize_messages(obj: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(obj, dict):
        return None

    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = obj["messages"]
    elif "conversations" in obj and isinstance(obj["conversations"], list):
        msgs = obj["conversations"]
    else:
        return None

    out: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from") or m.get("speaker")
        content = m.get("content") or m.get("value") or m.get("text")
        if role is None or content is None:
            continue
        role_s = str(role).lower()
        if role_s in ("human", "user"):
            role_s = "user"
        elif role_s in ("gpt", "assistant", "bot"):
            role_s = "assistant"
        else:
            continue
        out.append({"role": role_s, "content": str(content)})

    if not out:
        return None
    # Ensure it starts with user for chat template stability.
    if out[0]["role"] != "user":
        return None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-turns", type=int, default=0, help="0 means keep all turns.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    rnd = random.Random(int(args.seed))
    records = list(_iter_records(args.in_path))
    idxs = list(range(len(records)))
    rnd.shuffle(idxs)
    idxs = idxs[: max(0, min(len(idxs), int(args.max_samples)))]

    with open(args.out_path, "w", encoding="utf-8") as out:
        written = 0
        for i, idx in enumerate(idxs):
            rec = records[idx]
            msgs = _normalize_messages(rec)
            if msgs is None:
                continue
            if int(args.max_turns) > 0:
                msgs = msgs[: int(args.max_turns) * 2]
            task = {
                "dataset": "sharegpt",
                "request_id": f"sharegpt_{written}",
                "messages": msgs,
                "max_new_tokens": int(args.max_new_tokens),
            }
            out.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1


if __name__ == "__main__":
    main()

