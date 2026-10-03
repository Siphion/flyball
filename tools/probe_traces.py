#!/usr/bin/env python3
"""The Gate B teacher: would cutting the reasoning here still give the right answer?

For each uncut reasoning trace, the reasoning is truncated at regular checkpoints,
the thinking block is closed by hand, and the model is asked to answer from that
point, greedily. If the forced answer is correct, cutting there was safe.

That is the exact question the governor has to answer, measured rather than
guessed: no regular expression over the reasoning, no judge model. It costs a short
generation on the GPU per label, which is why it is the teacher and not the
governor. The governor is a small model on the CPU that learns to predict these
labels from the partial reasoning alone.

Two protocols:

- `trial` (default): after closing the reasoning, the reply is started for the
  model with a final-answer box, and only the content of the box is generated. The
  label says whether the answer is already settled, and the probability the model
  gives to its own answer is recorded too: that confidence is what self-probing
  methods such as DEER use to decide when to stop, so it serves as a baseline. The
  box is decoded one token per request, because a server that decodes
  speculatively reports no probability for the tokens it accepts from its draft. A
  probe costs a few tokens and about a second.
- `free`: the model writes its whole reply after the cut. Closer to what happens
  when a governor really cuts, since the model may go on reasoning in its reply,
  but it costs a full answer per label.

Two details matter for the labels to mean anything:

- the prompt is rendered with the same reasoning effort used to produce the trace,
  because the chat template changes with it;
- checkpoints are moved back to the previous whitespace, so that the model is never
  asked to continue from half a word.

Each row also says whether the forced answer agrees with the one the uncut trace
gave. That second label needs no reference answer at all: "cutting here changes
nothing" can be measured on any prompt whose answers can be compared.

With `--final`, only the end of each answered trace is probed, into the same file.
"Never cut" is then graded the way every cut is, by the forced answer, instead of by
reading a number out of a free-form reply: a reply that ends "He earns $202 during
the 8-hour shift" is right, and its last number is 8.

Resumable: rows already present in the output file are skipped.

Usage:
    python tools/probe_traces.py --traces reports/math_traces.jsonl \\
        --cases reports/math_cases.json --out reports/math_probes.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

from measure_reasoning_cut import BOX, final_answer, last_number

CLOSE_REASONING = "\n</think>\n\n"
# The prefill stops before the opening brace, on purpose. Tokenisers merge the brace
# with what follows it ("{-" is one token in Qwen's vocabulary), so a prompt that ends
# in "{" leaves the model unable to write a minus sign: every negative answer came out
# positive, with low confidence, until this was found.
TRIAL_PREFILL = "**Final Answer**\n\n" + BOX.rstrip("{")


class Upstream:
    def __init__(self, base_url: str, api_key: str, effort: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.effort = effort
        self._rendered: dict[str, str] = {}

    def _post(self, path: str, body: dict, timeout: float = 600.0) -> dict:
        request = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), self.headers, method="POST"
        )
        return json.load(urllib.request.urlopen(request, timeout=timeout))

    def rendered(self, prompt: str) -> str:
        if prompt not in self._rendered:
            self._rendered[prompt] = self._post(
                "/apply-template",
                {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": self.effort},
            )["prompt"]
        return self._rendered[prompt]

    def forced_answer(self, prompt: str, reasoning_prefix: str, max_tokens: int) -> str:
        body = {
            "prompt": self.rendered(prompt) + reasoning_prefix + CLOSE_REASONING,
            "n_predict": max_tokens,
            "temperature": 0,
            # Successive probes on one trace share a growing prefix: the server only
            # has to process the characters added since the previous checkpoint.
            "cache_prompt": True,
        }
        return self._post("/completion", body)["content"]

    def trial_answer(self, prompt: str, reasoning_prefix: str, max_tokens: int = 16) -> dict:
        # One token per request. With speculative decoding (Bonsai's multi-token
        # prediction) the server reports a real probability only for the first token
        # of a reply: tokens accepted from the draft come back as 0.0, certain. Asking
        # for one token at a time makes every probability real; the cache keeps each
        # step down to a single token of work.
        head = self.rendered(prompt) + reasoning_prefix + CLOSE_REASONING + TRIAL_PREFILL
        ids: list[int] = []
        text = ""
        logprobs = []
        for _ in range(max_tokens):
            body = {
                "prompt": [head, *ids] if ids else head,
                "n_predict": 1,
                "temperature": 0,
                "cache_prompt": True,
                "n_probs": 1,
            }
            steps = self._post("/completion", body).get("completion_probabilities", [])
            if not steps:
                break
            step = steps[0]
            logprobs.append(step["logprob"])
            if "}" in step["token"]:
                # The closing brace counts: it is the model saying the number is complete.
                text += step["token"].split("}")[0]
                break
            ids.append(step["id"])
            text += step["token"]
        return {
            "text": text,
            # Probability of the whole greedy answer, and of its least certain token.
            "logprob": sum(logprobs),
            "min_prob": math.exp(min(logprobs)) if logprobs else 0.0,
        }


def checkpoints(reasoning: str, step: int, max_probes: int) -> list[int]:
    """Positions to probe, every `step` characters, snapped back to whitespace."""
    positions = []
    for target in range(step, len(reasoning), step):
        cut = reasoning.rfind(" ", 0, target)
        cut = max(cut, reasoning.rfind("\n", 0, target))
        if cut > 0 and (not positions or cut > positions[-1]):
            positions.append(cut)
    if len(positions) > max_probes:
        # Evenly thinned, always keeping the earliest and the latest.
        stride = (len(positions) - 1) / (max_probes - 1)
        positions = sorted({positions[round(i * stride)] for i in range(max_probes)})
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key", default=os.environ.get("FLYBALL_UPSTREAM_API_KEY", ""))
    parser.add_argument("--effort", default="xhigh", help="must match the effort of the traces")
    parser.add_argument("--protocol", choices=["trial", "free"], default="trial")
    parser.add_argument("--step", type=int, default=200, help="characters between checkpoints")
    parser.add_argument("--max-probes", type=int, default=12, help="per trace")
    parser.add_argument("--max-answer-tokens", type=int, default=512, help="free protocol only")
    parser.add_argument(
        "--final",
        action="store_true",
        help="probe only the end of each answered trace, so that 'never cut' is graded "
        "the same way as every cut",
    )
    args = parser.parse_args()

    cases = {c["name"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    traces = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [t for t in traces if not t.get("cut") and t["case"] in cases]

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["case"], row["repeat"], row["position"]))

    def positions(trace: dict) -> list[int]:
        if args.final:
            # A trace that ran out of tokens before answering has no end to probe:
            # never cutting it gives no answer at all.
            return [len(trace["reasoning"])] if trace["content"].strip() else []
        return checkpoints(trace["reasoning"], args.step, args.max_probes)

    upstream = Upstream(args.upstream, args.api_key, args.effort)
    total = sum(len(positions(t)) for t in traces)
    print("{} traces, {} checkpoints, {} already labelled, protocol {}".format(
        len(traces), total, len(done), args.protocol))

    count = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for trace in traces:
            case = cases[trace["case"]]
            reference = last_number(case["reference_answer"])
            uncut_answer = final_answer(trace["content"])
            for position in positions(trace):
                count += 1
                key = (trace["case"], trace["repeat"], position)
                if key in done:
                    continue
                prefix = trace["reasoning"][:position]
                started = time.time()
                try:
                    if args.protocol == "trial":
                        trial = upstream.trial_answer(case["prompt"], prefix)
                        given = last_number(trial["text"])
                        extra = {
                            "trial": trial["text"],
                            "logprob": round(trial["logprob"], 4),
                            "min_prob": round(trial["min_prob"], 4),
                        }
                    else:
                        answer = upstream.forced_answer(case["prompt"], prefix, args.max_answer_tokens)
                        given = final_answer(answer)
                        extra = {"answer_chars": len(answer), "answer_tail": answer[-200:]}
                except Exception as exc:  # noqa: BLE001 - one failed probe is one missing label
                    print("  [{}/{}] {} @{} FAILED: {!r}".format(count, total, trace["case"], position, exc))
                    continue
                safe = given == reference
                handle.write(json.dumps({
                    "case": trace["case"],
                    "repeat": trace["repeat"],
                    "position": position,
                    "reasoning_chars": len(trace["reasoning"]),
                    "fraction": round(position / len(trace["reasoning"]), 4),
                    "protocol": args.protocol,
                    "final": args.final,
                    "safe": safe,
                    "agrees_with_uncut": given is not None and given == uncut_answer,
                    "trace_correct": trace["correct"],
                    "probe_seconds": round(time.time() - started, 2),
                    **extra,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                if count % 25 == 0:
                    print("  [{}/{}] {} @{} safe={}".format(count, total, trace["case"], position, safe))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
