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

### Servers and models

flyball is not tied to one server or one model. Two things are looked up for each
request, and both can be set by hand:

- **The server.** `FLYBALL_SERVER=auto` (the default) asks the upstream what it is.
  llama.cpp gets the full treatment: thinking switched on and off, effort levels, a
  per-request budget, and mid-stream control. Any other OpenAI-compatible server (vLLM,
  SGLang, Ollama, LM Studio, mlx-lm) is spoken to in the fields most of them accept,
  `chat_template_kwargs` and `reasoning_effort`, and is reported as unable to control
  a live completion; there only Gate A can act. `flyballd probe` shows what was found.
- **The model's profile.** A profile holds the model's ladder of reasoning levels,
  cheapest first, the probe that asks it for its answer, and the governor's operating
  point for each level. It is matched on the model id the server reports. Bonsai 2
  ships with one; an unknown model keeps the defaults measured on Bonsai 2. More
  profiles, or changes to a built-in one, go in a JSON file named by
  `FLYBALL_PROFILES`:

  ```json
  {"profiles": [{"id": "bonsai-2", "extends": "bonsai-2",
                 "governor": {"medium": {"first_probe": 1200, "confidence": 0.8, "cap": 3200}}}]}
  ```

  `tools/compare_governors.py --operating-point-out` writes exactly this, from
  measurements.

A server may start with a default level of its own (`llama-server --reasoning-effort
medium`) without reporting it. `FLYBALL_DEFAULT_LEVEL` says which, so that a request
that names no level is governed at the right operating point.

### Gate A, learned: how much to think

With `--gate-a` (or `FLYBALL_GATE_A=on`), a System 1 reads each request before it
leaves and picks a level of the model's ladder: on Bonsai 2, whether to step down from
`medium` to `low` or to no thinking at all. It estimates, level by level, the chance
that the model answers correctly, and takes the cheapest level whose estimate is close
enough to the reference level's. What the System 1 reads is swappable: flyball's own
request features, any embeddings endpoint, a Hugging Face or Laya encoder, or the
answers of any System One server (Laya's, Jev). The model is a JSON artifact trained
by `tools/gate_a_eval.py` on runs of `tools/effort_sweep.py`, and it is used only on
the profile and ladder it was trained for.

In shadow mode the choice is made after the request has left, so it costs nothing,
and it is recorded with the outcome. In active mode it is made before, within
`FLYBALL_GATE_A_TIMEOUT_MS`; a System 1 that is slower, or fails, abstains, and the
request goes out unchanged. A client that names its own level is left alone, unless
`FLYBALL_GATE_A_CLIENT_EFFORT=ceiling` lets Gate A lower it. Gate A has its own A/B
split, independent of the governor's. If the server rejects a request Gate A
rewrote, flyball sends the client's original instead.

### Gate B: the governor

With `--governor` (or `FLYBALL_GOVERNOR=on`) in active mode, flyball closes the reasoning
of a streamed completion once the model has settled on its answer. While the reasoning
runs, it asks the model, through the server's `probe` control action, what its final
answer would be and how sure it is: once the reasoning passes a first threshold, and
each time it grows by a factor. When the model is confident enough in an answer that is
a number or an option letter, or when the reasoning reaches a cap, flyball ends the
reasoning and the model writes its reply. The schedule, the confidence and the cap are
the operating point of the request's reasoning level, from the model's profile: on
Bonsai 2 the point measured live at `xhigh` (first probe at 300 characters, doubling,
0.8 confidence, cap at 3,200 characters), used at `medium` too until a point measured
there replaces it. A level without an operating point, such as Bonsai 2's `low`, is not
governed. The research behind these numbers, and the live results, are in
[gate-b-research.md](gate-b-research.md), sections 5 to 10.

It governs a request only when all of these hold, and records why when it does not:

- active mode, the governor switched on, and no kill switch;
- a streamed request, with reasoning left on;
- a question that asks for a number or a short value, without tools, images or code;
- an operating point for the request's reasoning level;
- an upstream that offers the `probe` action, checked with a call that generates
  nothing, and checked again every few minutes;
- the conversation is in the treatment arm of the A/B split (`FLYBALL_AB_SHARE`).

The response stream is never touched: the governor reads a copy, and the client gets the
upstream's bytes unchanged. The only change to the request is `reasoning_control: true`,
which lets the reasoning be ended. Any failure, a refused probe, an unreachable control
endpoint, an exception, and the governor stops; the generation carries on as if it had
never been there. Each outcome row in the ledger carries the probes, their answers and
confidences, and the cut.

### Settings

| setting | default | meaning |
|---|---|---|
| `FLYBALL_SERVER` | `auto` | `auto`, `llama.cpp` or `openai` (any other OpenAI-compatible server) |
| `FLYBALL_PROFILE` | | the id of the model profile to use, instead of matching the model id |
| `FLYBALL_PROFILES` | | a JSON file of extra or changed profiles |
| `FLYBALL_DEFAULT_LEVEL` | | the level of a request that names none, when the server sets its own default |
| `FLYBALL_GATE_A` | `off` | `on` lets a learned System 1 pick each request's level |
| `FLYBALL_GATE_A_ARTIFACT` | | the Gate A model; empty means the one in the model's profile |
| `FLYBALL_GATE_A_CLIENT_EFFORT` | `respect` | `ceiling` lets Gate A lower a level the client chose |
| `FLYBALL_GATE_A_TIMEOUT_MS` | `150` | a System 1 slower than this abstains |
| `FLYBALL_GATE_A_SHARE` | `FLYBALL_AB_SHARE` | share of conversations Gate A acts on |
| `FLYBALL_SYSTEMONE_URL`, `FLYBALL_SYSTEMONE_API_KEY` | | a System One server (Laya's, Jev), for backends that ask one |
| `FLYBALL_GOVERNOR` | `off` | `on` to enable the governor in active mode |
| `FLYBALL_GOVERNOR_CONFIDENCE` | profile | cut once the model is this confident |
| `FLYBALL_GOVERNOR_CAP` | profile | cut here in any case, in characters of reasoning |
| `FLYBALL_GOVERNOR_FIRST_PROBE` | profile | the first probe, in characters of reasoning |
| `FLYBALL_GOVERNOR_GROWTH` | profile | how much the reasoning grows between probes |
| `FLYBALL_GOVERNOR_ANCHOR` | profile | close the reasoning with a sentence that commits to the probe's answer |
| `FLYBALL_AB_SHARE` | `0.5` | share of conversations governed; the rest are the control arm |

The governor settings, when set, override the profile's operating point at every level.

The `probe` action is not part of llama.cpp yet. `flyballd probe` reports whether an
upstream has it.
