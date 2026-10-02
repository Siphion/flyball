# Measurements

Exploratory results from the custom case set in `benchmarks/cases.json`. These are
**internal evidence, not release numbers**: the cases were written by hand for this
project. Before any public release the standard suites (MATH-500, GSM8K, AIME, GPQA
Diamond, LiveCodeBench, HumanEval+/MBPP+) will be run in full.

Setup: Ternary Bonsai 2 27B (PTQ1_0 with an MTP head), RTX 3080 10GB, stock llama.cpp
server from the PrismML fork, `reasoning_effort: xhigh`. The cut is applied with
`reasoning_control` and `POST /v1/chat/completions/control` once the streamed reasoning
reaches the given number of characters. Runs are paired and interleaved.

## Stage 1 — where does cutting do anything at all?

25 cases, 8 classes, cuts at 0 / 300 / 800 characters, 3 repeats.

On trivial, formatting, lookup, short-code and tool-calling requests the cut almost
never fires: those requests produce fewer than 300 characters of reasoning on their own.
Any timing difference in those classes is noise. **A fixed reasoning threshold is inert on
easy prompts**, which is why it is not a substitute for a gate that decides.

## Stage 2 — reasoning-heavy classes, 8 repeats

Median wall clock and correct answers out of 8.

| case | no cut | 150 | 300 | 500 | 800 | 1500 |
|---|---|---|---|---|---|---|
| puzzle-river-crossing | 22.5 s 8/8 | **2.2 s 8/8** | 2.5 s 8/8 | 3.4 s 8/8 | 4.7 s 8/8 | 7.1 s 8/8 |
| puzzle-cryptarithm | **81.6 s 7/8** | 2.3 s 2/8 | 3.2 s 2/8 | 5.3 s 4/8 | 5.9 s 2/8 | 16.4 s 4/8 |
| puzzle-knights-knaves | 4.4 s 8/8 | 3.5 s 8/8 | 3.0 s 8/8 | 3.1 s 8/8 | 4.1 s 8/8 | 4.1 s 8/8 |
| math-two-trains | 18.1 s 8/8 | 6.6 s 6/8 | 5.9 s 5/8 | 8.0 s 7/8 | **9.9 s 8/8** | 14.9 s 7/8 |
| math-two-pipes | 2.7 s 8/8 | 1.9 s 8/8 | 2.7 s 8/8 | 2.4 s 8/8 | 2.1 s 7/8 | 1.7 s 8/8 |
| math-rectangle | 2.8 s 8/8 | 2.3 s 8/8 | 2.7 s 8/8 | 2.8 s 8/8 | 2.4 s 8/8 | 2.2 s 8/8 |
| reason-off-by-one | 8.0 s 7/8 | **3.4 s 8/8** | 3.5 s 7/8 | 3.9 s 7/8 | 5.1 s 8/8 | 8.4 s 8/8 |
| reason-complexity | 8.1 s 8/8 | **5.5 s 8/8** | 6.5 s 8/8 | 6.9 s 8/8 | 9.2 s 8/8 | 9.4 s 8/8 |
| reason-race-condition | 4.6 s 8/8 | 3.5 s 8/8 | 3.2 s 8/8 | 4.0 s 8/8 | 4.5 s 8/8 | 4.9 s 8/8 |

`math-same-colour` and `math-modular` are excluded: their grading regexes were broken
(`\frac` and `\boxed` were read as regex escapes), so the model's correct LaTeX answers
were scored as wrong. The regexes are fixed; those two cases need to be re-run.

## What the data says

1. **For many problems the long reasoning is pure overhead.** The river-crossing puzzle
   goes from 22.5 s to 2.2 s, ten times faster, with no answer lost. Code-reasoning cases
   lose nothing at any threshold.
2. **Some problems genuinely need it.** The cryptarithm is right 7 times out of 8 when
   left alone, and drops to 2-4 out of 8 under every cut. It looks like the river-crossing
   puzzle and behaves in the opposite way.
3. **The right cut point depends on the problem.** The two-trains problem tolerates a cut
   at 800 characters and loses answers at 150-300.

Together these are the case for the project. No single fixed threshold serves both the
river-crossing puzzle and the cryptarithm: the decision has to be made per request, and
ideally while watching the reasoning unfold.

## Caveats

- Eight repeats per cell on a single hand-written problem per row. Directions are
  reliable where the gap is large; small differences are not.
- Wall-clock times depend on this hardware. The relative saving and the correctness are
  what should transfer.
- Correctness is checked by regex. Grading bugs are possible, as the two excluded cases
  show; every surprising cell should be checked against the raw output.
