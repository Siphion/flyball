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
for automation regardless of measured accuracy.
