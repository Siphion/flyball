# Gate B research: can a governor tell when the model has thought enough?

Gate B is the mid-stream governor: it watches the reasoning while it is being written
and closes it when it has done its job. This note records the offline research that
has to come before any of that is built, because it decides whether the idea has a
thesis at all.

It needs no proxy and no live model, only full reasoning traces of runs that were left
alone. They were collected with:

```bash
python tools/measure_reasoning_cut.py --cases <maths, puzzle and code cases> \
    --cuts 0 --runs 5 --traces reports/m4_traces.jsonl
python tools/analyze_traces.py --traces reports/m4_traces.jsonl
```

Setup: Ternary Bonsai 2 27B, `reasoning_effort: xhigh`, RTX 3080, 11 cases, 5 runs
each. The eight maths and puzzle cases have a strict `answer` pattern in
`benchmarks/cases.json`, written to match the final answer and to reject intermediate
steps (for example `7^10 ≡ 1` when the question is about `7^100`).

## 1. The oracle: where does the correct answer first appear?

For each trace, the first character at which the correct final answer is already
written in the reasoning. Cutting there is the best any governor could do; everything
after it is reasoning the answer did not need.

| case | median reasoning (chars) | answer first appears at | reasoning after it | correct |
|---|---|---|---|---|
| puzzle-river-crossing | 4904 | 264 | **94%** | 5/5 |
| math-two-trains | 2558 | 513 | **68%** | 5/5 |
| puzzle-cryptarithm | 9741 | 3891 | **43%** | 4/5 |
| math-two-pipes | 192 | 109 | 39% | 5/5 |
| math-modular | 381 | 301 (found in 2/5) | 36% | 5/5 |
| math-same-colour | 309 | 218 | 33% | 5/5 |
| math-rectangle | 223 | 178 | 22% | 5/5 |
| puzzle-knights-knaves | 711 | 477 | 20% | 5/5 |

What this says:

- **The headroom is real and large where it matters.** On the problems with long
  reasoning, most of it comes after the answer: 94% for the river-crossing puzzle, 68%
  for the trains problem.
- **The cryptarithm is not an exception, it is the hardest case of the same rule.** The
  exploratory cut sweep found that every fixed cut between 150 and 1500 characters broke
  it, which looked like a problem that needs all its reasoning. It does not: it needs
  about 40% of it, and then overthinks like the others. Every threshold tried was simply
  far too early, because its answer only appears around character 3900.
- **The right cut point ranges over more than a factor of twenty**, from under 200
  characters to almost 4000, across problems that look alike. No fixed threshold can sit
  anywhere sensible on that range. This is the strongest argument for a governor that
  decides per generation, and against a static `--reasoning-budget`.

## 2. Can a generic signal find that point without knowing the answer?

A real governor does not know the answer. It sees the reasoning so far and nothing else.
Four cheap signals were evaluated every 150 characters, on the 37 traces where the answer
pattern was found. A signal is useful only if it fires after the answer exists: firing
before it means cutting reasoning the answer still needed.

| signal | premature | on time | never fired |
|---|---|---|---|
| a conclusion has been stated | 8 | 6 | 23 |
| a conclusion, then verification language | 4 | 6 | 27 |
| the latest stretch repeats earlier text | 0 | 1 | 36 |
| the latest stretch introduces no new number | 5 | 7 | 25 |

**None of them works.** The ones that fire often are wrong about as often as they are
right, and the only one that is never premature almost never fires. Thirty lines of
rules do not recognise "has thought enough".

## 3. What follows

- **Gate B needs a learned detector.** This is where a System-1 decision model earns its
  place: a calibrated answer to "has the reasoning reached its answer yet?" on a partial
  trace, which the heuristics above cannot give.
- **The oracle is also the training signal, and it is free.** For any problem with a
  checkable answer, every checkpoint of every trace can be labelled automatically: the
  answer is already there, or it is not. That needs no teacher model and no human, and it
  scales with however many verifiable problems can be generated or collected. It is the
  natural first dataset for the learning loop.
- **The labels need care.** The oracle detects the answer *written*, not the answer
  *settled*: a model can state an answer and then correct it. The cryptarithm's 4/5
  correct rate is a reminder that the first appearance is not always the final one. A
  label should probably require the answer to appear and not be contradicted afterwards.

## Caveats

- Eight cases with checkable answers, five runs each, written for this project. The
  direction of the oracle result is large enough to trust; the precise percentages are
  not. The standard benchmarks will be needed before any public claim.
- The answer pattern for `math-modular` matched in only 2 of 5 traces, where the model
  wrote the result in a form the pattern does not cover.
- Characters, not tokens, are the unit throughout, because that is what the stream
  exposes most directly.
