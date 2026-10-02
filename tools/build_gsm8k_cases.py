#!/usr/bin/env python3
"""Turn a sample of GSM8K *training* problems into benchmark cases.

Gate B needs many problems whose answer can be checked automatically, and they must
not be problems written for this project. GSM8K is standard, has one numeric answer
per problem, and its training split can be used freely here: the test split stays
untouched for the public benchmarks that come before any release.

Each case gets two patterns:

- `expect`, for grading the final answer;
- `answer`, for finding the answer inside reasoning text.

Both accept the number with or without thousands separators, and refuse to match it
as part of a longer number.

Usage:
    python tools/build_gsm8k_cases.py --n 300 --seed 7 --out reports/gsm8k_cases.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

SOURCE = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)


def number_pattern(answer: str) -> str:
    """A regex for a numeric answer, tolerant of 1,234 versus 1234."""
    value = answer.replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", value):
        raise ValueError("not a plain number: " + answer)
    integer, _, fraction = value.lstrip("-").partition(".")
    grouped = "{:,}".format(int(integer)) if integer else integer
    forms = {integer, grouped}
    body = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
    if fraction:
        body += re.escape("." + fraction)
    sign = "-" if value.startswith("-") else ""
    # Not preceded or followed by another digit, comma-digit or dot-digit.
    return r"(?<![\d.,])" + re.escape(sign) + body + r"(?![\d]|[.,]\d)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("reports/gsm8k_cases.json"))
    args = parser.parse_args()

    raw = urllib.request.urlopen(SOURCE, timeout=60).read().decode("utf-8")
    problems = [json.loads(line) for line in raw.splitlines() if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(problems)

    cases = []
    for index, item in enumerate(problems):
        answer = item["answer"].rsplit("####", 1)[-1].strip()
        try:
            pattern = number_pattern(answer)
        except ValueError:
            continue
        cases.append(
            {
                "name": "gsm8k-train-{:04d}".format(index),
                "class": "gsm8k",
                "prompt": item["question"].strip(),
                "expect": pattern,
                "answer": pattern,
                "reference_answer": answer,
            }
        )
        if len(cases) == args.n:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("{} problems in the training split, {} sampled with seed {} -> {}".format(
        len(problems), len(cases), args.seed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
