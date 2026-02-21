import argparse
import json
import random
from typing import Any, Dict, List


def _format_prompt(ex: Dict[str, Any], *, max_context_chars: int) -> str:
    context = str(ex.get("context", ""))
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]
    question = str(ex.get("question", ""))
    a = str(ex.get("choice_A", ""))
    b = str(ex.get("choice_B", ""))
    c = str(ex.get("choice_C", ""))
    d = str(ex.get("choice_D", ""))
    return (
        "Read the following context and answer the multiple-choice question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n\n"
        "Answer:"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-context-chars", type=int, default=200000)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    rnd = random.Random(int(args.seed))
    indices = list(range(len(data)))
    rnd.shuffle(indices)
    indices = indices[: max(0, min(len(indices), int(args.max_samples)))]

    with open(args.out_path, "w", encoding="utf-8") as out:
        for i, idx in enumerate(indices):
            ex = data[idx]
            ex_id = str(ex.get("_id", f"row_{idx}"))
            prompt = _format_prompt(ex, max_context_chars=int(args.max_context_chars))
            task = {
                "dataset": "longbench_v2",
                "request_id": f"longbench_v2_{i}_{ex_id}",
                "prompt": prompt,
                "max_new_tokens": int(args.max_new_tokens),
            }
            out.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

