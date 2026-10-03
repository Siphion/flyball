#!/usr/bin/env python3
"""Turn a sample of MBPP training problems into code cases.

The governor's probe asks for a final answer, and code has no short one. Measuring
a probe for code needs code that can be checked, and MBPP brings it: a short task,
a reference solution and three assert tests per problem. The training split is
used, never the test split: MBPP+ is built on the latter and is reserved for the
public benchmarks.

Rows are read from Hugging Face's dataset viewer API as JSON.

Usage:
    python tools/build_mbpp_cases.py --n 150 --seed 7 --out reports/mbpp_cases.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "google-research-datasets/mbpp"
ROWS = "https://datasets-server.huggingface.co/rows?"
PROMPT = (
    "{task}\n\nYour code should pass these tests:\n\n{tests}\n\n"
    "Write the final code in a single ```python code block."
)


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
    parser.add_argument("--out", type=Path, default=Path("reports/mbpp_cases.json"))
    args = parser.parse_args()

    rows, offset = [], 0
    while True:
        page = fetch(ROWS + urllib.parse.urlencode({
            "dataset": DATASET, "config": "full", "split": "train", "offset": offset, "length": 100,
        }))
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            break
        time.sleep(1.0)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    cases = [{
        "name": "mbpp-train-{}".format(row["task_id"]),
        "class": "mbpp",
        "prompt": PROMPT.format(task=row["text"].strip(), tests="\n".join(row["test_list"])),
        "answer_type": "code",
        "tests": row["test_list"],
        "test_setup": row.get("test_setup_code") or "",
        "reference_solution": row["code"],
    } for row in sample]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("{} problems in the training split, {} sampled with seed {} -> {}".format(
        len(rows), len(cases), args.seed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
