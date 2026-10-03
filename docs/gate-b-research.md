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

## 4. First learned governor, on GSM8K: a negative result

300 problems from the GSM8K training split, uncut traces, and 1,560 checkpoints labelled
by the teacher (`tools/probe_traces.py`: close the reasoning at the checkpoint, let the
model answer greedily, grade the answer). Problems were split by name into train,
calibration and test; configuration was chosen on the calibration split only.

| policy (61 held-out traces) | accuracy | reasoning cut |
|---|---|---|
| never cut | 93.4% | 0% |
| fixed budget, 400 characters | 93.4% | **19%** |
| learned, threshold 0.85 | 91.8% | 20% |
| learned, threshold 0.90 | 93.4% | 0% |
| oracle: first safe checkpoint | 98.4% | 62% |

**At equal accuracy a fixed budget beats the learned governor**, 19% against 0%. The
classifier does learn something (AUC 0.63 on held-out problems, after fixing a scaling
bug that had turned rare hashed words into noise), but not enough.

The reason is the data, not only the model: **GSM8K is too easy for this model to need
governing.** Ninety per cent of checkpoints are already safe before 200 characters of
reasoning, so almost any cut works and a fixed budget is enough. Risk tracks problem
difficulty rather than position: the latest checkpoints are the riskiest, because only
hard problems produce long reasoning. The regime where a governor can beat a fixed
budget, where the right cut point varies by a factor of twenty, is the one the puzzle
cases showed, and GSM8K does not contain it.

The oracle's 98.4% is not evidence that cutting improves answers: it gets up to twelve
attempts per trace and keeps the first correct one, so some of those are lucky guesses.

Next: the same pipeline on hard problems with checkable answers.

## 5. Second iteration: hard problems, and the model's own confidence

### Data

Hard problems come from MATH, levels 3 to 5, keeping only those with an integer
answer so that grading stays a comparison of numbers
(`tools/build_math_cases.py`). They are drawn from the training split of OpenAI's
PRM800K, which repartitions MATH so that its test split is exactly MATH-500: nothing
here overlaps the public benchmark that has to run before any release. The source
file is pinned by its SHA-256.

300 problems were sampled evenly across levels; the trace run hit a time limit at
233 of them (78, 78 and 77 per level), which was left as it was. Traces are uncut,
`reasoning_effort: xhigh`, capped at 8,192 tokens:

| level | traces | answered correctly | median reasoning |
|---|---|---|---|
| 3 | 78 | 99% | 488 chars |
| 4 | 78 | 95% | 1,074 chars |
| 5 | 77 | 75% | 3,746 chars |

This is the regime GSM8K lacked: the 90th percentile of reasoning is 17,765
characters, and 24 traces never answered at all, running into the token cap.
The 300 GSM8K traces of section 4 are reused, relabelled with the new teacher.

### A cheaper and stricter teacher

The teacher now closes the reasoning, starts the reply for the model with
`**Final Answer**` and a box, and lets it write only the content of the box
(`probe_traces.py --protocol trial`). The label is whether that answer is right, and
the model's probability for its own answer is recorded with it.

It is stricter than letting the model write a whole reply. On the 685 GSM8K
checkpoints labelled both ways, the two agree 78% of the time, and in 130 cases the
free reply was right where the boxed answer was not: the model was still working the
problem out in its reply. Cutting there looks safe but saves less time than it seems,
because the reasoning moves into the answer. The boxed label asks the question a
governor needs answered: is the answer already settled?

### Three measurement bugs, each of which reversed the verdict

All three were found before any conclusion was written down, and each one, left in,
would have produced a confident and wrong result.

1. **A token boundary ate the minus sign.** The prompt ended in `\boxed{`, and Qwen's
   vocabulary merges `{-` into one token: with the brace already in the prompt, the
   model could hardly write a minus sign, and every negative answer came out positive
   with low confidence (−3 became `3`, at p≈0.35; fixed, `−3` at p≈0.99). Seventeen
   of the 300 MATH problems have negative answers. The prefill now stops before the
   brace.
2. **Two graders.** "Never cut" was graded by reading the last number of the model's
   free reply, every cut by the boxed answer. Eighteen right GSM8K replies were read
   as wrong ("He earns **$202** during the 8-hour shift" reads as 8), which made never
   cutting look worse than it is. "Never cut" is now graded the way every cut is, by a
   probe at the very end of the reasoning (`probe_traces.py --final`).
3. **Speculative decoding hides probabilities.** With multi-token prediction on, the
   server returns a real probability only for the first token it generates; tokens
   accepted from the draft come back with log-probability 0, certain. The first
   "confidence" was therefore the probability of the first digit, and after fix 1 the
   probability of the brace, always 1, which made self-probing look worthless. The box
   is now decoded one token per request, and every probability is real.

In the first run, self-probing looked like it cut 42% of reasoning against a fixed
budget's 11%. With the first two bugs fixed and the third still in, it never
qualified at all. Neither was true.

### Results

Five-fold cross-validation over the 454 traces that have at least one checkpoint
(`tools/compare_governors.py --learned`). Each governor's operating point is chosen
on four folds, as the most reasoning cut within one point of never cutting, and
read on the fifth; the learned governor's scores are cross-fitted the same way.
Never cutting is right 91.2% of the time.

| governor | accuracy | reasoning cut | time saved per request | time saved in total | probes per request |
|---|---|---|---|---|---|
| fixed budget (2,400–4,800 chars) | 90.3% | 12% | 12% | 57% | – |
| learned, text only, on the CPU | 90.7% | 7% | 7% | 26% | – |
| self-probe at every checkpoint | 91.9% | **37%** | 0% | 31% | 3.5 |
| self-probe, doubling schedule | 91.6% | 27% | 12% | 42% | 1.5 |
| self-probe, doubling schedule + cap | 91.0% | 28% | 14% | 52% | 1.5 |

The doubling schedule probes once the reasoning passes 300 characters and again each
time it doubles; the cap is a fixed budget behind it, for traces that never settle.
Time saved is net of the probes' measured cost. Against the fixed budget, with 95%
bootstrap intervals over problems:

| governor | accuracy | reasoning cut | per request | in total |
|---|---|---|---|---|
| learned, text only | −1.3 to +2.2 | **−7 to −3** | −7 to −3 | −38 to −23 |
| self-probe, every checkpoint | −0.7 to +3.7 | **+22 to +27** | −15 to −9 | −35 to −16 |
| self-probe, doubling + cap | −1.3 to +2.4 | **+14 to +18** | 0 to +4 | −8 to −1 |

The model's confidence in its forced answer is well calibrated, over all 3,130
checkpoints:

| confidence | checkpoints | answer right |
|---|---|---|
| below 0.30 | 865 | 14% |
| 0.30–0.70 | 392 | 55% |
| 0.70–0.90 | 370 | 80% |
| 0.90–0.97 | 374 | 91% |
| 0.97 and above | 1,129 | 97% |

### What it means

1. **The text-only learned governor fails.** It learns something (AUC 0.71, ECE 0.04
   on held-out problems), but at equal accuracy it cuts less reasoning than a fixed
   budget. The words of the reasoning do not say when it is done, not well enough.
2. **The model knows when it is done.** Asked for its answer, its confidence is
   calibrated, and as a governor it cuts two to three times as much reasoning as the
   best fixed budget at the same accuracy. The bar set for Gate B is passed, by
   self-probing rather than by the classifier.
3. **In wall-clock time it is a draw, because probes are expensive here.** A probe
   takes 0.7 to 2 seconds, and almost all of it is the server reprocessing context:
   decoding the four-token answer takes about 60 ms. Bonsai 2 is built on Qwen3.5, a
   hybrid architecture: three layers in four are recurrent, one in four is full
   attention. llama.cpp can only resume a recurrent state from saved checkpoints, so
   a request that diverges from the cached sequence rereads 600 to 2,100 tokens even
   when it adds a hundred. Probing every checkpoint costs as much time as it saves.
4. **A cheap probe would turn the draw into a win.** Priced at what a forked
   sequence would cost, the suffix and the answer with nothing reprocessed, the
   doubling schedule with a cap beats the fixed budget on both time measures:

   | probe cost | per request | in total | accuracy |
   |---|---|---|---|
   | 0.1 s | +8 to +11 | +2 to +7 | −0.9 to +1.1 |
   | 0.3 s | +5 to +8 | +1 to +5 | −0.9 to +1.1 |

   (`compare_governors.py --probe-cost 0.1,0`, operating point chosen for time saved
   in total.)

### What follows

The System 1 that works is not a classifier reading the reasoning: it is a policy
that asks the model. What it needs is a mechanism llama.cpp does not have yet, a
probe on a live completion: a control action, next to `reasoning_end`, that forks
the running sequence, appends a suffix, decodes a few tokens with their real
probabilities, returns them and drops the fork. That is the next milestone, and the
natural contribution upstream. Live validation comes after it, because everything
here is simulated: a real cut lets the model write a free reply, not a boxed answer.

### Caveats

- Simulated, not live. The probe cost was measured with nothing else running; under
  a live generation it depends on how the server schedules it.
- One model, one reasoning effort, two datasets, 454 traces.
- Time saved is estimated from characters skipped at a constant generation rate.
- Traces with under 200 characters of reasoning have no checkpoint and are left out:
  nothing could be cut there, but they would lower every per-request mean.
- Runaway traces count as wrong when never cut, and a cap can score them right,
  which is part of why fixed budgets do so well on total time.
- A single train, calibration and test split was not enough to choose operating
  points: on the 84 calibration problems the fixed budget chose 800 characters, which
  then lost nine points of accuracy on the test split. Cross-validation is the
  reliable read, and the one reported above.

## 6. A probe inside the server

The probe that section 5 priced at "what a forked sequence would cost" now exists, as
a prototype: a third control action next to `reasoning_end`, on a local branch of the
llama.cpp fork that serves Bonsai. It is not published.

```
POST /v1/chat/completions/control
{"id": "<completion id>", "action": "probe",
 "suffix": "\n</think>\n\n**Final Answer**\n\n\\boxed", "n_predict": 12, "stop": "}"}

{"success": true,
 "probe": {"content": "{18", "logprob": -0.28, "tokens": [...], "timings": {...}}}
```

Between two generation steps, on the thread that owns the context, it saves the
slot's recurrent state, decodes the suffix and a few greedy tokens on the slot's own
sequence, reading their real probabilities, then restores the state and removes the
probe's tokens. It needs no second sequence and no extra GPU memory, and it rereads
nothing.

Measured on Bonsai 2 with `tools/probe_live.py`, generating greedily:

- **Invisible.** Each problem was generated with and without probes, and the two texts
  are identical byte for byte, five problems out of five, including 16,526 characters
  of reasoning probed 54 times. Speculative decoding was on throughout: a rejected
  draft leaves a pending rollback in the recurrent state, and saving the state
  honours it.
- **Cheap, and flat.** Over 70 probes on four problems, none failed: 181 ms median on
  the server (saving the state 44 ms, decoding 113 ms, restoring 25 ms, for 150 MiB of
  recurrent state), 213 ms as the client sees it. The cost grows by 0.6 ms per 1,000
  characters of reasoning, where a probe sent as a separate request grew by 310 ms.
  Each probe pauses the generation for about 0.2 seconds.

Priced at that (`compare_governors.py --probe-cost 0.2,0.0006`, operating point
chosen for time saved in total), self-probing on the doubling schedule with a cap
beats the fixed budget on every measure at the same accuracy:

| governor | accuracy | reasoning cut | time saved per request | time saved in total |
|---|---|---|---|---|
| fixed budget | 90.3% | 12% | 12% | 57% |
| self-probe, doubling + cap | 90.3% | 26% | 20% | 61% |
| difference, 95% interval | −1.1 to +1.1 | +12 to +15 | +6 to +10 | +2 to +6 |

What remains is the cut itself, which is still simulated: a probe's boxed answer
stands in for the reply the model writes after a real `reasoning_end`. The next step
is the governor live in flyball, probing on the doubling schedule, cutting with
`reasoning_end` when the model is confident, with a cap behind, measured end to end
against `--reasoning-budget`. The probe can also get cheaper: saving and restoring
the state goes through host memory, and a copy kept on the GPU would remove most of
those 70 ms.

## 7. Live: the governor cutting for real

Up to section 6 the cut was simulated. Here it is real (`tools/govern_live.py`), on
127 problems never used before: the 67 MATH problems the trace run of section 5 did
not reach, and 60 new GSM8K problems. Each was answered three times by Bonsai 2, at
its usual sampling (temperature 1), in an order rotated from problem to problem:

- **never cut**;
- **fixed budget**: the reasoning closed at 2,400 characters;
- **governor**: a probe at 300, 600, 1,200 and 2,400 characters, a cut as soon as
  the model's confidence in its answer reaches 0.8, and a cap at 3,200 characters.

Both operating points were chosen beforehand, on the 454 traces of section 5 and at
the probe's measured cost. Every prompt asks for a boxed answer, and every reply is
graded by it.

| arm | accuracy | mean time | time saved per request | time saved in total | reasoning |
|---|---|---|---|---|---|
| never cut | 92.1% | 14.4 s | – | – | 2,657 chars |
| fixed budget | 94.5% | 10.1 s | 4% | 30% | 1,013 chars |
| governor | **95.3%** | **8.9 s** | **9%** | **38%** | 886 chars |

Governor minus fixed budget, 95% intervals over problems: accuracy −2.4 to +4.7
points, time saved per request 0 to +11, in total −6 to +24. Time does not depend
on grading, so the first pass of the experiment (below) counts for it too: over
both passes, 254 runs per arm, the governor saves +4 to +13 points more per request,
and −3 to +21 in total.

What happened, in more detail:

- **Cutting made the answers better, not worse**, in both arms. Five times, never
  cutting meant never answering: the model reasoned until the token limit. A cut
  rescues those. Against never cutting, the governor lost two answers and won six.
- **Confidence is a good reason to cut**: 68 confident cuts, 97% of them right. The
  cap is a worse one, 10 cuts and 70% right, but it only fires on problems where
  the model never became confident.
- **The reasoning still moves into the reply, but less after the governor's cuts.**
  The median reply doubles after a governor cut (158 to 331 characters) and
  quadruples after a budget cut (202 to 824). A budget stops the model mid-thought;
  the governor stops one that has settled.
- **The probe held**: 210 calls, none failed, 221 ms median.
- **Two failure modes exist only live.** After a confident cut, the model can still
  go on reasoning in its reply for thousands of characters (three replies over
  5,000), and in the first pass it once wrote no reply at all. Both times the probe
  had the right answer. A governor that already knows the answer could act on that.

The first pass graded only the last 300 characters of each reply, and its grader
read `12{,}000` as `000`. Neither could be fixed after the fact, because only the
answers had been kept, so the whole experiment was run again with the full texts,
graded at the end. The accuracy above is the second pass's.

**Verdict.** Live, the governor clears the bar it was set: at least as accurate as
the fixed budget and as never cutting, and faster per request by a margin that the
intervals separate from zero once both passes count. In total time it is ahead, but
not separated: total time is dominated by the few runaway problems, which the budget
and the cap both cut.

Caveats: one model on one machine; 127 problems, all maths with a numeric answer;
sampling at temperature 1, so every run differs and the intervals are wide. The probe
is a local patch to llama.cpp. For code or prose, a probe has to ask the model a
different question than "what is your final answer?".

## 8. Anchoring the cut

Section 7 left two failures that only happen live: after a cut, the model sometimes
goes on reasoning in its reply, for thousands of characters, and once it wrote no
reply at all. Both times the probe already had the right answer. The cut threw that
knowledge away: `reasoning_end` closed the reasoning mid-sentence and left the model
to find its way back.

So the cut now says where the reasoning got to. `reasoning_end` takes an optional
`message`, forced into the reasoning just before it closes (a second small patch to
the local llama.cpp branch, next to the probe; llama.cpp already forces a fixed
message when a static budget runs out, this one is decided at the moment of the
cut). The governor closes a confident cut with "So the final answer is X.", X being
the probe's answer, and a cut at the cap with "I'll stop here and give my best final
answer."

Same 127 problems, the governor with and without the anchor, interleaved:

| arm | accuracy | mean time | median reply after a cut | replies over 5,000 chars after a cut |
|---|---|---|---|---|
| governor | 93.7% | 9.0 s | 368 chars | 2 |
| anchored | 94.5% | **7.0 s** | **268 chars** | **0** |

Anchored minus governor, 95% intervals: accuracy −3.1 to +4.7 points, time saved in
total +5 to +37%. The anchor costs nothing measurable in accuracy, and saves a fifth
of the total time, almost all of it from the replies that no longer run away. The
problems the two arms disagree on split four and four.

The anchor has a cost in principle: a confident probe that is wrong gets written into
the reasoning, where the model could otherwise have corrected it in its reply. It
happened three times in 67 confident cuts. In two of them the plain governor gave the
same wrong answer; in the third, its run never became confident at all and reasoned
its way to the right one, so a different sample rather than a correction.

Grading got one more fix on the way, applied to every run kept as text: a reply whose
box holds no number now reads as no answer, instead of falling back to the last
number elsewhere in the text. One flawed GSM8K problem (a "square" lake whose length
and width differ) had been credited because a reply boxed "No such lake exists"
after mentioning 100.

The anchor is on by default (`FLYBALL_GOVERNOR_ANCHOR`); a server whose
`reasoning_end` takes no message simply ignores it.

## 9. Beyond numbers: multiple choice

A probe asks for a final answer in a box, and an option letter fits in a box as well
as a number does. The governor now accepts a letter as an answer (`{B`, `{\text{B`,
`{(B)`), and governs questions that list at least three options.

Tested live on 75 MMLU-Pro questions (`tools/build_mmlu_pro_cases.py`, test split,
sampled at random with a fixed seed across its fourteen subjects, physics, law,
economics and psychology among them; MMLU-Pro is not one of the benchmarks kept for
the public results), with the operating point unchanged:

| arm | accuracy | mean time |
|---|---|---|
| never cut | 60.0% | 53.7 s |
| fixed budget | 70.7% | 10.9 s |
| anchored governor | 68.0% | **6.5 s** |

Anchored governor minus fixed budget, 95% intervals: accuracy −8.0 to +2.7 points,
time saved in total +14 to +59%. The governor is faster than the budget by a margin
the interval separates from zero, at an accuracy the interval cannot tell apart.

Two things differ from maths. With ten options the model is confident less often:
of 75 questions, 49 were cut on confidence and 23 at the cap. And never cutting is
far worse here than anywhere before: at the highest reasoning effort the model
reasons for 16,000 characters on average, often until the token limit, and gives no
answer at all. Any cut rescues those, which is why both cutting arms beat never
cutting by ten points.

Caveat for everyday use: every measurement so far runs at the highest reasoning
effort, where the model overthinks the most. A server's default is usually lower,
and so will the gains be; that has to be measured on its own.

## 10. Code: a probe that works on problems that do not need it

Code has no short final answer to ask for. The probe for code asks instead, inside
the reasoning, "Is my solution settled, so that I can write the final code now?",
and reads the probabilities of yes and no from the first token, where the server
reports real ones (`probe_traces.py --protocol code`). The teacher closes the
reasoning at each checkpoint, has the model write the code, and runs MBPP's tests on
it in a sandbox (`tools/sandbox.py`: one Docker container per problem, no network, a
read-only filesystem, bounded memory, CPU and processes, ten seconds per program).

150 problems from MBPP's training split (MBPP+ is built on its test split, kept for
the public results), 1,272 checkpoints. The readiness the model reports tracks how
often its code passes:

| readiness | checkpoints | code passes |
|---|---|---|
| 0.50–0.70 | 37 | 54% |
| 0.70–0.80 | 329 | 78% |
| 0.80–0.85 | 673 | 91% |
| 0.85–0.90 | 227 | 96% |

The signal is real but squeezed: the model says yes nearly always, and the useful
range is narrow.

And on MBPP it is not needed. Code written after 200 characters of reasoning passes
the tests 87% of the time; code written after the whole reasoning passes 80% of the
time, because 28 of the 150 reasonings, nearly one in five, never end before the
token limit. They are not loops: no sentence repeats in their last 8,000 characters.
The model has a working solution and goes on checking ever stranger cases. With the
best operating points chosen by cross-validation, a fixed budget of 200 characters
and the self-probing governor end up doing the same thing, cutting at once, and the
probe only adds its own cost.

MBPP is to code what GSM8K was to maths: too easy for this model to need governing.
Whether the readiness probe beats a fixed budget has to be measured on code that
needs its reasoning; until then flyball does not govern code requests. The broader
lesson of MBPP and MMLU-Pro together is that at the highest effort, the largest
saving comes from deciding before the call how much to think, which is Gate A's
question, not Gate B's.
