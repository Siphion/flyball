# flyball

**A System-1 governor for your local model's reasoning.**

A small, fast decision model watches your large, slow one *while it thinks*, and closes
the reasoning block when it has thought enough. flyball is an OpenAI-compatible proxy, so
every client you already use benefits at once.

```
chat UI  ─┐
agents  ──┼─►  flyball  ─►  your model (stock llama.cpp)
scripts ──┘      │
                 └─ closes the thinking mid-stream, when it is worth closing
```

## What this is not

flyball does **not** pick which model answers your request. That problem — model routing —
is solved by a dozen projects, and on a single consumer GPU it barely applies: swapping
models costs tens of seconds, and a 27B model leaves no room for a second one.

flyball answers a different question: **given the one model you can actually fit, how
should this request be served?**

## Why it is worth doing

Measured on an RTX 3080 10GB running a 27B model entirely in VRAM, on one reasoning
problem, four runs per arm:

| | wall clock | hidden reasoning | visible answer | correct |
|---|---|---|---|---|
| untouched | 28.1 s | 3621 chars | 926 chars | 3/4 |
| thinking closed at 300 chars | **11.3 s** | 304 chars | 1519 chars | 2/4 |
| thinking closed at 800 chars | 16.4 s | 808 chars | 1910 chars | 3/4 |

Two things that table says, and one it does not:

- **The lever is large.** Around 2.5x on wall clock.
- **Reasoning does not vanish, it migrates.** Cut the hidden thinking and the model
  redoes part of it in the visible answer, which grew from 926 to 1910 characters. So the
  metric that matters is *time to a complete answer*, never *how much reasoning was cut*.
- **It says nothing about quality.** Four runs on a single problem, with a 21-33 s spread
  at identical settings, cannot support a claim either way. Finding out when cutting is
  safe is the whole point of the project, and it needs a dataset, not an opinion.

## How it works

The mechanism is already in upstream llama.cpp: pass `reasoning_control: true` on a
completion, read the completion `id` from the SSE stream, then
`POST /v1/chat/completions/control` with `{"id": ..., "action": "reasoning_end"}` *while
tokens are still arriving*. It was added for a manual button in the web UI. flyball puts a
calibrated classifier behind that button.

Nothing is patched and nothing is forked: flyball runs against a stock server.

## Status

Early. The proxy and the capability probe work; the governor does not exist yet.

- [x] OpenAI-compatible passthrough proxy — streaming, tool calls, multimodal, verified
      against a real client suite
- [x] Capability probe: what does this upstream actually support?
- [ ] Measurement harness (`flyballd measure`)
- [ ] Heuristic gate, which every learned backend has to beat
- [ ] Decision backends behind one interface: heuristics, Laya, a small local LLM with
      constrained decoding, k-nearest-neighbour over your own labelled examples
- [ ] Calibration and threshold fitting, owned by flyball rather than the backend
- [ ] Shadow logging, outcome attribution, and a fine-tuning export

## Install

```bash
pip install -e ".[dev]"
```

## Use

```bash
# What can this server do?
flyballd probe --upstream http://127.0.0.1:8081

# Pure passthrough. flyball takes the port your clients already point at,
# the model moves to another one. Nothing else changes.
flyballd serve --upstream http://127.0.0.1:8081 --port 8080
```

Configuration is read from the environment; see `.env.example`. Modes are documented in
[docs/modes.md](docs/modes.md).

## Design rules

1. **Fail open.** flyball sits in the critical path. Any error, timeout, or low-confidence
   decision falls back to what the upstream would have done on its own. A bad decision
   backend degrades to stock behaviour, never to worse than stock.
2. **Byte-exact passthrough.** Bodies are streamed without being parsed, so tool calls,
   multimodal content and vendor-specific fields survive untouched.
3. **Measure, then decide.** No decision is enabled on live traffic before its accuracy
   and calibration have been measured on your own traffic, against a heuristic baseline.
4. **Your logs stay yours.** Decision logs contain raw prompts. They are local, excluded
   from version control, and never leave the machine unless you explicitly choose a remote
   teacher for labelling.

## License

Apache-2.0.
