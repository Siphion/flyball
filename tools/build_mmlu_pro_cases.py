#!/usr/bin/env python3
"""Turn a sample of MMLU-Pro questions into benchmark cases.

The governor was measured on questions whose answer is a number. Many questions have
an answer that is a choice instead, and the probe can ask for that just as well: the
model boxes a letter. MMLU-Pro is the test bed for it: multiple choice with ten
options, across fourteen subjects, with questions that need reasoning rather than
recall. It is not among the benchmarks reserved for the public results.

Rows are read from Hugging Face's dataset viewer API as JSON, so no parquet reader is
needed. The sample is drawn by row offset with a fixed seed.

Usage:
    python tools/build_mmlu_pro_cases.py --n 150 --seed 7 --out reports/mmlu_pro_cases.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "TIGER-Lab/MMLU-Pro"
ROWS = "https://datasets-server.huggingface.co/rows?"
SIZE = "https://datasets-server.huggingface.co/size?"
INSTRUCTION = "\n\nAnswer with the letter of the correct option, and put it within \\boxed{}."


def fetch(url: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:  # noqa: BLE001 - the viewer API rate-limits (HTTP 429)
            if attempt + 1 == retries:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("reports/mmlu_pro_cases.json"))
    args = parser.parse_args()

    size = fetch(SIZE + urllib.parse.urlencode({"dataset": DATASET}))
    total = next(s["num_rows"] for s in size["size"]["splits"] if s["split"] == args.split)
    rng = random.Random(args.seed)
    offsets = sorted(rng.sample(range(total), args.n))

    cases = []
    for offset in offsets:
        time.sleep(1.0)  # one row a second keeps clear of the API's rate limit
        page = fetch(ROWS + urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": args.split, "offset": offset, "length": 1,
        }))
        row = page["rows"][0]["row"]
        letters = string.ascii_uppercase[: len(row["options"])]
        options = "\n".join("{}. {}".format(letter, option) for letter, option in zip(letters, row["options"]))
        cases.append({
            "name": "mmlu-pro-{}-{}".format(args.split, row["question_id"]),
            "class": "mmlu-pro",
            "subject": row["category"],
            "prompt": row["question"].strip() + "\n\n" + options + INSTRUCTION,
            "answer_type": "letter",
            "reference_answer": row["answer"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("{} questions in the {} split, {} sampled with seed {} -> {}".format(
        total, args.split, len(cases), args.seed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
