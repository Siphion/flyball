# Modes

flyball has three modes, from least to most invasive. The default is the least.

## `off`

Pure passthrough. Requests and responses are forwarded untouched; no decision is computed
and nothing is logged. This is the default, and it is also the fallback that every failure
path lands in.

Use it to put flyball in place without changing behaviour, and to confirm that your
clients keep working before anything else is switched on.

## `shadow`

Decisions are computed and recorded, but never applied. The upstream behaves exactly as it
would in `off`.

This is where the dataset comes from. For every eligible request flyball records the
compact state it built, the schema it asked about, the full probability distribution the
backend returned, its latency, and what happened afterwards. Nothing a client sees
changes, so a backend that is wrong is harmless.

A heuristic gate runs alongside in the same mode. It is the baseline: a learned backend
that cannot beat thirty lines of rules at equal coverage has not earned the right to act.

## `active`

Approved decisions are applied to live traffic.

Promotion is per question, not global, and it is gated on measurements taken in `shadow`:
accuracy and calibration error on held-out data, plus a risk-coverage curve above both the
majority-class and the heuristic baselines. A question whose threshold has not been fitted
stays inactive, and anything below its confidence threshold abstains to the upstream
default.

Decisions that are irreversible, or that end work rather than steer it, are not candidates
for automation on measured accuracy alone. The one exception is Gate B's cut, below, which
ends a model's reasoning: it was measured live, against a fixed reasoning budget, before
it was wired in, and it acts only on the kind of request it was measured on.

### Gate B: the governor

With `--governor` (or `FLYBALL_GOVERNOR=on`) in active mode, flyball closes the reasoning
of a streamed completion once the model has settled on its answer. While the reasoning
runs, it asks the model, through the server's `probe` control action, what its final
answer would be and how sure it is: once the reasoning passes 300 characters, and each
time it doubles. When the model is at least 0.8 confident in an answer that is a number,
or when the reasoning reaches 3,200 characters, flyball calls `reasoning_end` and the
model writes its reply. The research behind these numbers, and the live results, are in
[gate-b-research.md](gate-b-research.md), sections 5 to 7.

It governs a request only when all of these hold, and records why when it does not:

- active mode, the governor switched on, and no kill switch;
- a streamed request, with reasoning left on;
- a question that asks for a number or a short value, without tools, images or code;
- an upstream that offers the `probe` action, checked with a call that generates
  nothing, and checked again every few minutes;
- the conversation is in the treatment arm of the A/B split (`FLYBALL_AB_SHARE`).

The response stream is never touched: the governor reads a copy, and the client gets the
upstream's bytes unchanged. The only change to the request is `reasoning_control: true`,
which lets the reasoning be ended. Any failure, a refused probe, an unreachable control
endpoint, an exception, and the governor stops; the generation carries on as if it had
never been there. Each outcome row in the ledger carries the probes, their answers and
confidences, and the cut.

| setting | default | meaning |
|---|---|---|
| `FLYBALL_GOVERNOR` | `off` | `on` to enable it in active mode |
| `FLYBALL_GOVERNOR_CONFIDENCE` | `0.8` | cut once the model is this confident |
| `FLYBALL_GOVERNOR_CAP` | `3200` | cut here in any case, in characters of reasoning |
| `FLYBALL_AB_SHARE` | `0.5` | share of conversations governed; the rest are the control arm |

The `probe` action is not part of llama.cpp yet. `flyballd probe` reports whether an
upstream has it.
