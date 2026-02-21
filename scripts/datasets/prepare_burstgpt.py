import argparse
import csv
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_int(row: Dict[str, str], keys: List[str], default: int = 0) -> int:
    for k in keys:
        if k in row and row[k] != "":
            try:
                return int(float(row[k]))
            except Exception:
                continue
    return int(default)


def _make_filler_words(n_words: int, *, seed: int) -> str:
    rnd = random.Random(int(seed))
    vocab = [f"w{i:06d}" for i in range(5000)]
    parts: List[str] = []
    for _ in range(n_words):
        parts.append(vocab[rnd.randrange(len(vocab))])
    return " ".join(parts)


def _format_prompt(n_words: int, *, seed: int) -> str:
    filler = _make_filler_words(n_words, seed=seed)
    return (
        "You will be given a long document.\n\n"
        f"Document:\n{filler}\n\n"
        "Question: What is the last word of the document?\n"
        "Answer:"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt-words-scale", type=float, default=1.0)
    p.add_argument("--max-prompt-words", type=int, default=60000)
    p.add_argument("--default-max-new-tokens", type=int, default=64)
    args = p.parse_args()

    rnd = random.Random(int(args.seed))
    rows: List[Dict[str, str]] = []
    with open(args.in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    indices = list(range(len(rows)))
    rnd.shuffle(indices)
    indices = indices[: max(0, min(len(indices), int(args.max_samples)))]

    with open(args.out_path, "w", encoding="utf-8") as out:
        for i, idx in enumerate(indices):
            row = rows[idx]
            req_tokens = _get_int(row, ["Request tokens", "Request Tokens", "request_tokens"], default=0)
            resp_tokens = _get_int(row, ["Response tokens", "Response Tokens", "response_tokens"], default=int(args.default_max_new_tokens))
            timestamp = _get_int(row, ["Timestamp", "timestamp"], default=0)

            n_words = int(max(0, req_tokens) * float(args.prompt_words_scale))
            n_words = min(int(args.max_prompt_words), n_words)
            prompt = _format_prompt(n_words, seed=int(args.seed) + idx)

            task: Dict[str, Any] = {
                "dataset": "burstgpt_synth",
                "request_id": f"burstgpt_{i}_ts{timestamp}",
                "prompt": prompt,
                "max_new_tokens": int(max(1, resp_tokens)),
            }
            out.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

