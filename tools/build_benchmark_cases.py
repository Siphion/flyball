#!/usr/bin/env python3
"""Turn the standard benchmarks into cases, in full, for the published comparison.

Development used training splits (GSM8K train, PRM800K's MATH training problems,
MBPP train) so that these stay clean. Before anything is published, flyball is
measured on them in full, the way the model card of Bonsai and the papers it is
compared with report them:

| benchmark | source | answer |
|---|---|---|
| `math500` | HuggingFaceH4/MATH-500 | LaTeX, normalised (grading.py) |
| `gsm8k-test` | openai/grade-school-math, test split | number |
| `aime2024` | HuggingFaceH4/aime_2024 | integer |
| `aime2025` | MathArena/aime_2025 | integer |
| `humaneval-plus` | evalplus/humanevalplus | code, EvalPlus tests |
| `mbpp-plus` | evalplus/mbppplus | code, EvalPlus tests |
| `gpqa-diamond` | Idavidrein/gpqa (gated: needs HF_TOKEN) | option letter |

Every problem is kept, in the source's order; nothing is sampled. Prompts ask for
the final answer in a box, or for the code in one Python block, as the development
sets did. GPQA's options are shuffled with a fixed seed, since its source lists the
right answer first. The raw rows are cached, so a rebuild needs no network.

Usage:
    python tools/build_benchmark_cases.py --benchmark math500 --out reports/bench_math500_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROWS = "https://datasets-server.huggingface.co/rows?"
GSM8K_TEST = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
BOXED = "\n\nPut your final answer within \\boxed{}."
LETTER = "\n\nAnswer with the letter of the correct option, and put it within \\boxed{}."
CODE = "\n\nWrite the final code in a single ```python code block."

# Problems whose own reference solution fails the generic test: graded by them, a
# right answer would count as wrong. They are left out, and the count says so.
EXCLUDED = {
    "humaneval-plus-32": "EvalPlus grades find_zero with a checker of its own; its generic test "
                         "fails the reference solution",
}

SOURCES = {
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "aime2024": ("HuggingFaceH4/aime_2024", "default", "train"),
    "aime2025": ("MathArena/aime_2025", "default", "train"),
    "humaneval-plus": ("evalplus/humanevalplus", "default", "test"),
    "mbpp-plus": ("evalplus/mbppplus", "default", "test"),
    "gpqa-diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
}


def fetch(url: str, token: str = "", retries: int = 6):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as reply:
                return reply.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SystemExit(f"{url}: access denied; a gated dataset needs HF_TOKEN with its terms accepted")
            if attempt + 1 == retries:
                raise
        except Exception:  # noqa: BLE001 - the viewer API rate-limits (HTTP 429) and times out
            if attempt + 1 == retries:
                raise
        time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def rows_of(name: str, cache: Path, token: str) -> list[dict]:
    path = cache / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if name == "gsm8k-test":
        rows = [json.loads(line) for line in fetch(GSM8K_TEST).decode("utf-8").splitlines() if line.strip()]
    else:
        dataset, config, split = SOURCES[name]
        rows, offset = [], 0
        while True:
            query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                            "offset": offset, "length": 100})
            page = json.loads(fetch(ROWS + query, token))
            rows += [r["row"] for r in page["rows"]]
            offset += 100
            if offset >= page.get("num_rows_total", 0):
                break
            time.sleep(1.0)  # the viewer API throttles bursts
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def imports_of(code: str) -> str:
    """The import lines at the top of a HumanEval prompt: a reply may leave them out."""
    return "\n".join(line for line in code.splitlines() if line.startswith(("import ", "from ")))


def cases_for(name: str, rows: list[dict], seed: int) -> list[dict]:
    out = []
    if name == "math500":
        for r in rows:
            uid = r["unique_id"].replace("test/", "").replace(".json", "").replace("/", "-")
            out.append({"name": f"math500-{uid}", "class": "math500", "subject": r["subject"],
                        "level": int(r["level"]), "prompt": r["problem"] + BOXED,
                        "reference_answer": r["answer"], "answer_type": "latex"})
    elif name == "gsm8k-test":
        for i, r in enumerate(rows):
            answer = r["answer"].split("####")[-1].strip().replace(",", "")
            out.append({"name": f"gsm8k-test-{i:04d}", "class": "gsm8k-test", "prompt": r["question"] + BOXED,
                        "reference_answer": answer, "answer_type": "number"})
    elif name in ("aime2024", "aime2025"):
        for i, r in enumerate(rows):
            number = r.get("id", r.get("problem_idx", i))
            out.append({"name": f"{name}-{number}", "class": name, "prompt": r["problem"] + BOXED,
                        "reference_answer": str(r["answer"]), "answer_type": "number"})
    elif name == "humaneval-plus":
        for r in rows:
            out.append({"name": "humaneval-plus-" + r["task_id"].split("/")[-1], "class": name,
                        "prompt": "Complete this Python function. Write the whole function, with its signature."
                                  + "\n\n```python\n" + r["prompt"] + "```" + CODE,
                        "answer_type": "code", "entry_point": r["entry_point"], "test_harness": r["test"],
                        "code_prefix": imports_of(r["prompt"]), "reference_solution": r["prompt"] + r["canonical_solution"]})
    elif name == "mbpp-plus":
        for r in rows:
            tests = r["test_list"] if isinstance(r["test_list"], list) else eval_list(r["test_list"])
            imports = r["test_imports"] if isinstance(r["test_imports"], list) else eval_list(r["test_imports"])
            out.append({"name": "mbpp-plus-" + str(r["task_id"]), "class": name,
                        "prompt": r["prompt"] + "\nYour code should pass this test:\n\n" + tests[0] + CODE,
                        "answer_type": "code", "test_harness": r["test"], "entry_point": "",
                        "code_prefix": "\n".join(imports), "reference_solution": r["code"]})
    elif name == "gpqa-diamond":
        rng = random.Random(seed)
        for i, r in enumerate(rows):
            options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            letters = string.ascii_uppercase[:4]
            listed = "\n".join(f"{letters[k]}. {options[j].strip()}" for k, j in enumerate(order))
            out.append({"name": f"gpqa-diamond-{i:03d}", "class": name, "subject": r.get("High-level domain", ""),
                        "prompt": r["Question"].strip() + "\n\n" + listed + LETTER,
                        "reference_answer": letters[order.index(0)], "answer_type": "letter"})
    else:
        raise SystemExit(f"unknown benchmark {name!r}")
    return out


def eval_list(text: str) -> list[str]:
    """A Python list literal of strings, as the EvalPlus rows store it, read safely."""
    import ast

    value = ast.literal_eval(text)
    return [str(v) for v in value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True, choices=sorted(SOURCES) + ["gsm8k-test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("F:/ai/flyball/benchmarks/raw"))
    parser.add_argument("--seed", type=int, default=7, help="for the order of GPQA's options")
    args = parser.parse_args()

    rows = rows_of(args.benchmark, args.cache, os.environ.get("HF_TOKEN", ""))
    cases = cases_for(args.benchmark, rows, args.seed)
    left_out = [c["name"] for c in cases if c["name"] in EXCLUDED]
    cases = [c for c in cases if c["name"] not in EXCLUDED]
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print("{} cases from {} -> {}{}".format(len(cases), args.benchmark, args.out,
                                           "; left out: " + ", ".join(left_out) if left_out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
