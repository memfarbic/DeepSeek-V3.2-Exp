import argparse
import json
import random
import string
from typing import List, Tuple


def _rand_token(rnd: random.Random, n: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rnd.choice(alphabet) for _ in range(n))


def _make_context_words(rnd: random.Random, n_words: int) -> List[str]:
    vocab = [f"w{i:05d}" for i in range(20000)]
    return [vocab[rnd.randrange(len(vocab))] for _ in range(n_words)]


def _insert_needles(words: List[str], needles: List[Tuple[int, str]]) -> None:
    # Insert from back to keep positions stable.
    for pos, needle_text in sorted(needles, key=lambda x: x[0], reverse=True):
        needle_words = needle_text.split(" ")
        words[pos:pos] = needle_words


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--context-words", type=int, default=50000)
    p.add_argument("--needles", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    rnd = random.Random(int(args.seed))

    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(int(args.num_samples)):
            secret = _rand_token(rnd, n=16)
            words = _make_context_words(rnd, int(args.context_words))

            needle_positions = sorted({rnd.randrange(0, max(1, len(words))) for _ in range(int(args.needles))})
            needles = [(pos, f"NEEDLE_{j}: {secret}") for j, pos in enumerate(needle_positions)]
            _insert_needles(words, needles)

            context = " ".join(words)
            prompt = (
                "You will be given a long context. Find the secret value in the context.\n\n"
                f"Context:\n{context}\n\n"
                "Question: What is the secret value?\n"
                "Answer:"
            )

            task = {
                "dataset": "ruler_style",
                "request_id": f"ruler_style_{i}",
                "prompt": prompt,
                "max_new_tokens": int(args.max_new_tokens),
            }
            out.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

