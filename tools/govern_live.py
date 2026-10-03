#!/usr/bin/env python3
"""Run the Gate B governor live and measure it end to end.

Everything before this was simulated: a probe's boxed answer stood in for the reply
the model writes after a real cut. Here the cuts are real, made by the governor that
runs in the proxy (flyball.governor), through the server adapter the proxy uses.
Each problem is answered by every arm, an arm being a governor configuration:

- `never`: the reasoning runs to its natural end;
- `budget[:N]`: the reasoning is closed once it reaches N characters (--budget by
  default), which is what a static reasoning budget does;
- `governor`: the model is probed once the reasoning passes 300 characters and each
  time it doubles; the reasoning is closed as soon as the model is confident enough
  in the answer it would give, and at a cap if that never happens;
- `anchored`: the same, closing the reasoning with a sentence that commits to the
  answer the probe found;
- `tuned=PATH`: the operating point in a profile fragment written by
  compare_governors.py --operating-point-out, for the reasoning level of --effort.

The operating points are passed in, chosen beforehand on other problems: choosing
them here would make the result optimistic. The arms of a problem run back to back,
in an order rotated from one problem to the next, with prompt caching off, so that
neither drift nor a warm cache favours an arm.

Every reply is graded the same way, by the answer in its final box. The whole texts
are kept, and the summary grades them again, so a fix to the grader reaches runs
already made: the first pass of this experiment kept only the last 300 characters
of each reply, and lost answers to a grader that read `12{,}000` as `000`. Time is
the whole generation, from request to last token, probes included.

Resumable: runs already in the output file are skipped. With --stop-after-minutes
a run stops itself between two requests.

Usage:
    python tools/govern_live.py --cases reports/live_cases.json --effort medium \\
        --arms never,budget:2400,tuned=reports/op_medium.json \\
        --api-key-file ../localqwen/scripts/api-key.txt --out reports/live_tuned_runs.jsonl
    python tools/govern_live.py --summary reports/live_runs.jsonl --cases reports/live_cases.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from lab import api_key, client_for, run_completion
from measure_reasoning_cut import answer_in, reference_of

from flyball.governor import Governor, GovernorConfig
from flyball.profiles import all_profiles, load_profiles, profile_from_dict
from flyball.servers import adapter_for

KNOWN_ARMS = ("never", "budget", "governor", "anchored", "tuned")


def arm_config(spec: str, args) -> tuple[str, GovernorConfig | None]:
    """The name of an arm and its governor configuration (None: never governed)."""
    name, _, value = spec.partition("=") if "=" in spec else spec.partition(":")
    if name == "never":
        return name, None
    if name == "budget":
        return name, GovernorConfig(first_probe=None, cap=int(value or args.budget), anchor=False)
    if name in ("governor", "anchored"):
        return name, GovernorConfig(confidence=args.confidence, cap=args.cap, anchor=name == "anchored")
    if name == "tuned":
        fragment = json.loads(Path(value).read_text(encoding="utf-8"))
        entry = fragment["profiles"][0] if "profiles" in fragment else fragment
        known = {p.id: p for p in all_profiles()}
        point = profile_from_dict(entry, known).operating_point(args.effort)
        if point is None:
            raise SystemExit(f"{value} has no operating point for level {args.effort!r}")
        return name, point
    raise SystemExit(f"unknown arm {spec!r}; arms are " + ", ".join(KNOWN_ARMS))


def summarise(runs_path: Path, cases_path: Path, n_boot: int = 2000) -> None:
    cases = {c["name"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))}
    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by: dict = {}
    for run in runs:
        if "error" in run or run["case"] not in cases:
            continue
        if "content" in run:
            # graded again from the text, with the grader as it is now
            case = cases[run["case"]]
            run["answer"] = answer_in(run["content"], case.get("answer_type", "number"))
            run["correct"] = run["answer"] == reference_of(case)
        by.setdefault(run["case"], {})[run["arm"]] = run
    seen = {arm for a in by.values() for arm in a}
    arms = [arm for arm in KNOWN_ARMS if arm in seen] + sorted(seen - set(KNOWN_ARMS))
    governed = [arm for arm in arms if any(a.get(arm, {}).get("probes") for a in by.values())]
    complete = [name for name, a in by.items() if all(arm in a for arm in arms)]
    print("{} problems with every arm ({})".format(len(complete), ", ".join(arms)))

    def table(names: list[str], title: str) -> None:
        if not names:
            return
        print()
        print("{} ({} problems)".format(title, len(names)))
        baseline = "never" if "never" in arms else None
        print("  {:9} {:>9} {:>13} {:>12} {:>9} {:>10} {:>6} {:>13} {:>9} {:>6}".format(
            "arm", "accuracy", "mean seconds", "per request", "in total", "reasoning", "cuts",
            "reply after", "long", "empty"))
        print("  {:9} {:>9} {:>13} {:>12} {:>9} {:>10} {:>6} {:>13} {:>9} {:>6}".format(
            "", "", "", "(vs never)", "", "", "", "a cut (med.)", "replies", ""))
        base_t = np.array([by[n][baseline]["seconds"] for n in names]) if baseline else None
        stats = {}
        for arm in arms:
            r = [by[n][arm] for n in names]
            t = np.array([x["seconds"] for x in r])
            correct = np.array([x["correct"] for x in r], dtype=float)
            stats[arm] = (correct, t)
            cut = [x for x in r if x.get("cut")]
            per_request = float(np.mean((base_t - t) / base_t)) if baseline else float("nan")
            in_total = float((base_t.sum() - t.sum()) / base_t.sum()) if baseline else float("nan")
            print("  {:9} {:>8.1%} {:>12.1f}s {:>12} {:>9} {:>10.0f} {:>6} {:>13} {:>9} {:>6}".format(
                arm, correct.mean(), t.mean(),
                "{:.0%}".format(per_request) if baseline else "-", "{:.0%}".format(in_total) if baseline else "-",
                np.mean([x["reasoning_chars"] for x in r]), len(cut),
                int(np.median([x["content_chars"] for x in cut])) if cut else "-",
                sum(1 for x in cut if x["content_chars"] > 5000),
                sum(1 for x in cut if x["content_chars"] == 0)))

        rng = np.random.default_rng(0)
        samples = [rng.integers(0, len(names), len(names)) for _ in range(n_boot)]
        pairs = [(a, "budget") for a in arms if a not in ("never", "budget")]
        pairs += [("anchored", "governor"), ("tuned", "anchored")]
        for a, b in pairs:
            if a not in stats or b not in stats:
                continue
            a_c, a_t = stats[a]
            b_c, b_t = stats[b]
            diffs = np.array([(a_c[i].mean() - b_c[i].mean(), 1 - a_t[i].sum() / b_t[i].sum()) for i in samples])
            low, high = np.percentile(diffs, [2.5, 97.5], axis=0)
            print("  {} minus {}, 95% interval: accuracy [{:+.1%}, {:+.1%}]  time saved in total [{:+.0%}, {:+.0%}]".format(
                a, b, low[0], high[0], low[1], high[1]))

        for arm in governed:
            probes = [p for n in names for p in by[n][arm].get("probes", [])]
            ok = [p for p in probes if p["ok"]]
            reasons: dict = {}
            for n in names:
                cut = by[n][arm].get("cut")
                key = cut["reason"] if cut else "no cut"
                reasons[key] = reasons.get(key, 0) + 1
            print("  {}: {} probes ({} failed), median {:.0f} ms; cuts: {}".format(
                arm, len(probes), len(probes) - len(ok),
                float(np.median([p["ms"] for p in ok])) if ok else float("nan"),
                ", ".join("{} {}".format(k, v) for k, v in sorted(reasons.items()))))

    table(complete, "All problems")
    for prefix in ("math", "gsm8k", "mmlu-pro"):
        table([n for n in complete if n.startswith(prefix)], prefix)


async def run(args, arms: list[tuple[str, str, GovernorConfig | None]], cases: list[dict], done: set) -> int:
    profiles = all_profiles(load_profiles(args.profiles) if args.profiles else [])
    profile = next((p for p in profiles if p.id == args.profile), None)
    if profile is None or profile.level(args.effort) is None:
        raise SystemExit(f"profile {args.profile!r} has no level {args.effort!r}")
    adapter = adapter_for(args.server)
    deadline = time.time() + args.stop_after_minutes * 60 if args.stop_after_minutes else None
    total = len(cases) * len(arms)
    count = 0
    async with client_for(api_key(args.api_key_file)) as client:
        with args.out.open("a", encoding="utf-8") as handle:
            for index, case in enumerate(cases):
                reference = reference_of(case)
                # The order of the arms rotates from one problem to the next.
                shift = index % len(arms)
                for name, spec, config in arms[shift:] + arms[:shift]:
                    count += 1
                    if (case["name"], name) in done:
                        continue
                    if deadline and time.time() > deadline:
                        print("stopping: --stop-after-minutes reached; run again to resume", flush=True)
                        return 0
                    row = {"case": case["name"], "arm": name, "spec": spec, "effort": args.effort,
                           "config": config.as_dict() if config else None}
                    if name == "budget":
                        row["budget"] = config.cap
                    body = {"model": args.model, "stream": True, "max_tokens": args.max_tokens,
                            "messages": [{"role": "user", "content": case["prompt"]}]}
                    if adapter.kind == "llama.cpp":
                        body["cache_prompt"] = False  # a warm cache would favour whichever arm ran second
                    body = adapter.render_level(body, profile.level(args.effort), profile)
                    governor = None
                    if config is not None:
                        body = adapter.arm(body)
                        controller = adapter.controller(client, args.upstream.rstrip("/"), {}, args.model)
                        governor = Governor(config=config, control=controller, model=args.model,
                                            protocol=profile.probe)
                    try:
                        result = await run_completion(client, args.upstream, body, governor=governor,
                                                      max_seconds=args.max_seconds or None)
                        result["answer"] = answer_in(result["content"], case.get("answer_type", "number"))
                        result["correct"] = result["answer"] == reference
                        result.setdefault("cut", None)
                        result.setdefault("probes", [])
                        row.update(result)
                    except Exception as exc:  # noqa: BLE001 - a failed run is recorded and retried later
                        row["error"] = repr(exc)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print("[{:>3}/{}] {:<40} {:8} {:>6}s correct={} cut={}".format(
                        count, total, case["name"][:40], name, row.get("seconds", "-"), row.get("correct"),
                        (row.get("cut") or {}).get("reason")), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, help="only summarise an existing runs file")
    parser.add_argument("--only", help="run only the cases whose name starts with this")
    parser.add_argument("--arms", default="never,budget,governor",
                        help="comma-separated: never, budget[:N], governor, anchored, tuned=PATH")
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key-file", help="read the upstream key from this file (else FLYBALL_UPSTREAM_API_KEY)")
    parser.add_argument("--server", default="llama.cpp", choices=["llama.cpp"],
                        help="the governor needs a server that controls a live completion")
    parser.add_argument("--profile", default="bonsai-2")
    parser.add_argument("--profiles", help="a JSON file of extra profiles")
    parser.add_argument("--model", default="default")
    parser.add_argument("--effort", default="xhigh", help="the reasoning level, from the profile's ladder")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="cut a run longer than this; 0 = never")
    parser.add_argument("--budget", type=int, default=2400, help="characters, for the budget arm")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--cap", type=int, default=3200, help="characters, for the governed arms")
    parser.add_argument("--stop-after-minutes", type=float, default=105.0)
    args = parser.parse_args()

    if args.summary:
        summarise(args.summary, args.cases)
        return 0
    if not args.out:
        parser.error("--out is required to run")
    arms = []
    for spec in (a.strip() for a in args.arms.split(",") if a.strip()):
        name, config = arm_config(spec, args)
        arms.append((name, spec, config))
    if len({name for name, _, _ in arms}) != len(arms):
        parser.error("two arms share a name")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["name"].startswith(args.only)]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done.add((row["case"], row["arm"]))
    return asyncio.run(run(args, arms, cases, done))


if __name__ == "__main__":
    raise SystemExit(main())
