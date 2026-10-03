#!/usr/bin/env python3
"""Turn a sample of hard MATH problems into benchmark cases.

GSM8K turned out to be too easy for Gate B: the model rarely needs more than a few
hundred characters of reasoning, so a fixed budget is as good as any governor. The
regime the governor exists for is problems whose right amount of reasoning varies
widely, and competition mathematics has it.

The problems come from the training split of OpenAI's PRM800K, which repartitions
MATH so that its test split is exactly MATH-500. Drawing from the other side of that
line keeps MATH-500 clean for the public benchmarks that come before any release.
The file is pinned by its SHA-256 and refused if it changes.

Only problems with an integer answer are kept, so that grading stays a comparison of
numbers rather than of LaTeX expressions. Levels are sampled evenly, so that easy and
hard problems are both represented whatever their share of the source.

Usage:
    python tools/build_math_cases.py --n 300 --seed 7 --out reports/math_cases.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from build_gsm8k_cases import number_pattern

SOURCE = (
    "https://media.githubusercontent.com/media/openai/prm800k/main/"
    "prm800k/math_splits/train.jsonl"
)
SOURCE_SHA256 = "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076"

# The instruction used by the usual MATH evaluations, so that the final answer is
# boxed and can be read without guessing.
INSTRUCTION = "\n\nPut your final answer within \\boxed{}."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--levels", default="3,4,5", help="comma-separated MATH levels")
    parser.add_argument("--out", type=Path, default=Path("reports/math_cases.json"))
    args = parser.parse_args()

    raw = urllib.request.urlopen(SOURCE, timeout=120).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit("source changed: sha256 {} instead of {}".format(digest, SOURCE_SHA256))
    problems = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    levels = [int(level) for level in args.levels.split(",")]
    pools = defaultdict(list)
    for item in problems:
        if item["level"] in levels and re.fullmatch(r"-?\d+", item["answer"].strip()):
            pools[item["level"]].append(item)

    rng = random.Random(args.seed)
    for level in levels:
        rng.shuffle(pools[level])

    cases = []
    # Round-robin over levels: an even spread of difficulty.
    while len(cases) < args.n and any(pools[level] for level in levels):
        for level in levels:
            if pools[level] and len(cases) < args.n:
                item = pools[level].pop()
                answer = item["answer"].strip()
                pattern = number_pattern(answer)
                cases.append(
                    {
                        "name": "math-" + item["unique_id"].removesuffix(".json").replace("/", "-"),
                        "class": "math-l{}".format(level),
                        "subject": item["subject"],
                        "prompt": item["problem"].strip() + INSTRUCTION,
                        "expect": pattern,
                        "answer": pattern,
                        "reference_answer": answer,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("{} problems in the source, {} sampled from levels {} with seed {} -> {}".format(
        len(problems), len(cases), args.levels, args.seed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
