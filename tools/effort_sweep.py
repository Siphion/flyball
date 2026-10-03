#!/usr/bin/env python3
"""Run the same problems at every reasoning level of a model's ladder.

Gate A decides how much a request should think. Before anything is taught to make
that decision, the data has to say how accuracy and time change with the level,
problem by problem, and how much a perfect per-problem choice could save over the
best fixed level (effort_ceiling.py). This tool collects that data: each problem is
answered at each level, the level rendered into request fields by the same server
adapter and model profile the proxy uses, so what is measured is what flyball
would send.

The levels of a problem run back to back, in an order that rotates from one
problem to the next; on llama.cpp prompt caching is off. Neither drift nor a warm
cache can then favour a level. Each run keeps its whole text, and grading happens
at summary time, so a fix to the grader reaches runs already made.

Resumable: runs already in the output file are skipped. With --stop-after-minutes
a run stops itself between two requests, so that a long sweep can be split into
chunks that fit a time limit.

Usage:
    python tools/effort_sweep.py --cases reports/live_cases.json \\
        --cases reports/mmlu_pro_cases_a.json --levels none,low --repeats 0 \\
        --api-key-file ../localqwen/scripts/api-key.txt --out reports/effort_sweep.jsonl
    python tools/effort_ceiling.py --runs reports/effort_sweep.jsonl ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from evaluation import set_of
from lab import api_key, client_for, run_completion

from flyball.profiles import all_profiles, load_profiles
from flyball.servers import adapter_for


def load_cases(paths: list[Path], only: str = "", limit: int = 0) -> list[dict]:
    cases: list[dict] = []
    seen = set()
    for path in paths:
        for case in json.loads(path.read_text(encoding="utf-8")):
            if case["name"] in seen or (only and not case["name"].startswith(only)):
                continue
            seen.add(case["name"])
            cases.append(case)
    return cases[:limit] if limit else cases


def done_runs(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done.add((row["case"], row["level"], row["repeat"]))
    return done


def request_for(case: dict, level, profile, adapter, model: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": case["prompt"]}],
    }
    if adapter.kind == "llama.cpp":
        body["cache_prompt"] = False  # a warm cache would favour whichever level ran second
    return adapter.render_level(body, level, profile)


async def sweep(args) -> int:
    profiles = all_profiles(load_profiles(args.profiles) if args.profiles else [])
    profile = next((p for p in profiles if p.id == args.profile), None)
    if profile is None:
        raise SystemExit(f"unknown profile {args.profile!r}; known: {', '.join(p.id for p in profiles)}")
    levels = [name.strip() for name in args.levels.split(",") if name.strip()]
    missing = [name for name in levels if profile.level(name) is None]
    if missing:
        raise SystemExit(f"profile {profile.id} has no level {missing}; it has {profile.level_names()}")
    repeats = [int(r) for r in args.repeats.split(",")]
    adapter = adapter_for(args.server)
    cases = load_cases(args.cases, args.only, args.limit)
    done = done_runs(args.out)
    deadline = time.time() + args.stop_after_minutes * 60 if args.stop_after_minutes else None

    todo = [(case, level, repeat) for repeat in repeats for index, case in enumerate(cases)
            for level in (levels[index % len(levels):] + levels[:index % len(levels)])
            if (case["name"], level, repeat) not in done]
    print("{} runs to do ({} already done), profile {}, levels {}, repeats {}".format(
        len(todo), len(done), profile.id, ",".join(levels), args.repeats), flush=True)

    async with client_for(api_key(args.api_key_file)) as client:
        with args.out.open("a", encoding="utf-8") as handle:
            for count, (case, level, repeat) in enumerate(todo, 1):
                if deadline and time.time() > deadline:
                    print("stopping: --stop-after-minutes reached; run again to resume", flush=True)
                    return 0
                row = {"case": case["name"], "set": set_of(case["name"]), "level": level, "repeat": repeat,
                       "profile": profile.id, "server": adapter.kind}
                try:
                    body = request_for(case, profile.level(level), profile, adapter, args.model, args.max_tokens)
                    row.update(await run_completion(client, args.upstream, body, max_seconds=args.max_seconds))
                except Exception as exc:  # noqa: BLE001 - a failed run is recorded and retried later
                    row["error"] = repr(exc)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print("[{:>4}/{}] {:<44} {:7} r{} {:>7}s reasoning {:>6} {}".format(
                    count, len(todo), case["name"][:44], level, repeat, row.get("seconds", "-"),
                    row.get("reasoning_chars", "-"), row.get("error", "")[:60]), flush=True)
    return 0


def export_traces(out: Path, cases: list[dict], level: str, traces: Path) -> int:
    """The runs at one level, as uncut reasoning traces for probe_traces.py and the
    other tools that read measure_reasoning_cut.py's format, graded on the way."""
    from measure_reasoning_cut import answer_in, reference_of

    by_name = {c["name"]: c for c in cases}
    count = 0
    with traces.open("w", encoding="utf-8") as handle:
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case = by_name.get(row["case"])
            if case is None or row["level"] != level or "error" in row:
                continue
            answer = answer_in(row["content"], case.get("answer_type", "number"))
            handle.write(json.dumps({
                "case": row["case"], "cut": 0, "repeat": row["repeat"], "level": level,
                "correct": answer is not None and answer == reference_of(case),
                "seconds": row["seconds"], "reasoning": row["reasoning"], "content": row["content"],
            }, ensure_ascii=False) + "\n")
            count += 1
    print("{} traces at {} written to {}".format(count, level, traces))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-traces", metavar="LEVEL",
                        help="do not run: write the runs at LEVEL as reasoning traces to --traces-out")
    parser.add_argument("--traces-out", type=Path)
    parser.add_argument("--profile", default="bonsai-2")
    parser.add_argument("--profiles", help="a JSON file of extra profiles")
    parser.add_argument("--server", default="llama.cpp", choices=["llama.cpp", "openai"])
    parser.add_argument("--levels", default="none,low,medium")
    parser.add_argument("--repeats", default="0", help="comma-separated repeat indices to run")
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key-file", help="read the upstream key from this file (else FLYBALL_UPSTREAM_API_KEY)")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-seconds", type=float, default=240.0, help="a run longer than this is cut and kept")
    parser.add_argument("--stop-after-minutes", type=float, default=105.0)
    parser.add_argument("--only", default="", help="only the cases whose name starts with this")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.export_traces:
        if not args.traces_out:
            parser.error("--export-traces needs --traces-out")
        return export_traces(args.out, load_cases(args.cases), args.export_traces, args.traces_out)
    return asyncio.run(sweep(args))


if __name__ == "__main__":
    raise SystemExit(main())
