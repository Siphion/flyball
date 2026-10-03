#!/usr/bin/env python3
"""Does a decision model's reading of the reasoning contain "it has its answer"?

Asked zero-shot, Laya could not tell (system1_reasoning.py): its answers were at
chance. That says little about what it reads, because its decision head was never
taught this question. This tool looks underneath: it runs the encoder alone over
every labelled checkpoint (the question and the end of the reasoning), keeps the
representation, and fits a small logistic regression on it, with problems split
into folds so that every score comes from a model that never saw that problem.

If the representation separates "already has the answer" from "not yet" well, a
fine-tuned decision model is worth training; if it does not, no head on top of it
will.

The encoder comes from Laya today; any encoder whose hidden states can be read
would do. With --features, the states are read from a file instead: those of the
language model itself, extracted at the same checkpoints by extract_hidden.py
(docs/gate-b-research.md, section 13). --derived-out then writes the checkpoints
with the head's out-of-fold score as their confidence, for compare_governors.py,
and --artifact-out the head refitted on everything, as a hidden-state probe
(flyball/hidden.py).

Usage:
    python tools/system1_representation.py \\
        --set math reports/math_traces.jsonl reports/math_cases.json reports/math_probes.jsonl \\
        --set gsm8k reports/gsm8k_traces.jsonl reports/gsm8k_cases.json reports/gsm8k_trial_probes.jsonl \\
        --out F:/ai/flyball/research/laya_repr.npz
    python tools/system1_representation.py --features F:/ai/flyball/research/hidden_prefix_xhigh.npz \
        --set math reports/math_traces.jsonl reports/math_cases.json reports/math_probes.jsonl \
        --derived-out reports --label hidden-prefix
"""

from __future__ import annotations

import os

# A long fit on every core overheats this machine's CPU; four threads are plenty.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import fit_logistic, predict  # noqa: E402
from system1_reasoning import auc, state_of  # noqa: E402

FOLDS = 5


def load_set(traces_path: Path, cases_path: Path, probes_path: Path) -> list[dict]:
    cases = {c["name"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))}
    traces = {(t["case"], t["repeat"]): t for t in map(json.loads, traces_path.read_text(encoding="utf-8").splitlines())
              if not t.get("cut")}
    rows = []
    for p in map(json.loads, probes_path.read_text(encoding="utf-8").splitlines()):
        if p.get("final") or (p["case"], p["repeat"]) not in traces:
            continue
        own = math.exp(p["logprob"]) if "logprob" in p else p.get("readiness")
        rows.append({"case": p["case"], "repeat": p["repeat"], "position": p["position"], "safe": bool(p["safe"]),
                     "own": own, "length": p["reasoning_chars"],
                     "prompt": cases[p["case"]]["prompt"],
                     "prefix": traces[(p["case"], p["repeat"])]["reasoning"][: p["position"]]})
    return rows


def encode(rows: list[dict], threads: int, question_chars: int, tail_chars: int, cache: Path,
           batch_size: int = 16) -> np.ndarray:
    """The encoder's reading of each checkpoint: the first token's state and the mean
    over all tokens. On a GPU when there is one, in batches; saved as it goes, so an
    interrupted run resumes where it stopped."""
    import torch
    import laya

    partial = cache.with_suffix(".partial.npz")
    done = np.zeros((0,), dtype=np.float32)
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)["features"]
        if len(stored) == len(rows):
            return stored
    if partial.exists():
        done = np.load(partial, allow_pickle=False)["features"]
    if threads:
        torch.set_num_threads(threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya.load("convaiinnovations/laya")
    encoder, tok = agent.model.encoder.eval().to(device), agent.tok
    if device == "cuda":
        encoder = encoder.half()
    features = list(done) if len(done) else []
    began, start = time.time(), len(features)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(start, len(rows), batch_size):
            texts = [state_of(r["prompt"], r["prefix"], question_chars, tail_chars) for r in rows[i:i + batch_size]]
            batch = tok(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            hidden = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            features.extend(torch.cat([hidden[:, 0], mean], dim=1).float().cpu().numpy())
            if len(features) % (batch_size * 25) < batch_size or len(features) == len(rows):
                np.savez_compressed(partial, features=np.stack(features))
                print("  encoded {}/{} on {} ({:.3f} s each)".format(
                    len(features), len(rows), device, (time.time() - began) / max(1, len(features) - start)), flush=True)
    features = np.stack(features)
    np.savez_compressed(cache, features=features)
    partial.unlink(missing_ok=True)
    return features


def load_features(path: Path, rows: list[dict]) -> tuple[np.ndarray, dict]:
    """States precomputed by extract_hidden.py, in the order of `rows`."""
    data = np.load(path, allow_pickle=False)
    index = {key: i for i, key in enumerate(data["keys"].tolist())}
    keys = ["{}|{}|{}".format(r["case"], r["repeat"], r["position"]) for r in rows]
    missing = [k for k in keys if k not in index]
    if missing:
        raise SystemExit("{} checkpoints have no state in {}, e.g. {}".format(len(missing), path, missing[0]))
    meta = {name: str(data[name]) for name in ("variant", "source", "effort") if name in data}
    return data["states"][[index[k] for k in keys]].astype(np.float32), meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs=4, action="append", required=True,
                        metavar=("NAME", "TRACES", "CASES", "PROBES"))
    parser.add_argument("--out", type=Path, help="where the encoded features are cached")
    parser.add_argument("--features", type=Path, help="read the states from this file (extract_hidden.py)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--question-chars", type=int, default=500)
    parser.add_argument("--tail-chars", type=int, default=1200)
    parser.add_argument("--derived-out", type=Path, help="a directory for the governor input, per set")
    parser.add_argument("--label", default="hidden", help="names the derived files")
    parser.add_argument("--artifact-out", type=Path, help="write the head refitted on everything")
    parser.add_argument("--model", default="", help="recorded in the artifact: the model the states come from")
    args = parser.parse_args()

    rows = []
    for name, traces, cases, probes in args.set:
        for row in load_set(Path(traces), Path(cases), Path(probes)):
            row["set"] = name
            rows.append(row)
    print("{} checkpoints from {}".format(len(rows), ", ".join(s[0] for s in args.set)))
    meta: dict = {"source": "laya-encoder"}
    if args.features:
        x, meta = load_features(args.features, rows)
    elif args.out:
        x = encode(rows, args.threads, args.question_chars, args.tail_chars, args.out)
    else:
        parser.error("give --features, or --out to encode with Laya")
    y = np.array([1.0 if r["safe"] else 0.0 for r in rows])
    fold = np.array([int(hashlib.sha256(r["case"].encode()).hexdigest(), 16) % FOLDS for r in rows])

    # Out-of-fold scores, the regularisation chosen on an inner split of the training folds.
    scores = np.zeros(len(rows))
    chosen = []
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        inner = (fold != f) & (fold != (f + 1) % FOLDS)
        held = fold == (f + 1) % FOLDS
        best = max((auc(predict(fit_logistic(x[inner], y[inner], l2), x[held]), y[held]), l2)
                   for l2 in (1e-4, 1e-3, 1e-2, 1e-1))
        chosen.append(best[1])
        scores[test] = predict(fit_logistic(x[train], y[train], best[1]), x[test])

    print()
    print("  {:44} {:>6}".format("out-of-fold AUC, by set", "AUC"))
    for name in [s[0] for s in args.set] + ["all"]:
        mask = np.ones(len(rows), bool) if name == "all" else np.array([r["set"] == name for r in rows])
        own = np.array([r["own"] if r["own"] is not None else np.nan for r in rows])[mask]
        line = "  {:12} encoder + logistic {:>6.3f}".format(name, auc(scores[mask], y[mask]))
        if not np.isnan(own).any():
            line += "   the model's own confidence {:.3f}".format(auc(own, y[mask]))
        print(line + "   ({} checkpoints, {:.0%} right)".format(int(mask.sum()), y[mask].mean()))

    if args.derived_out:
        # The checkpoints with the head's score as their confidence: a state read
        # while the model generates costs no probe, hence no probe time.
        args.derived_out.mkdir(parents=True, exist_ok=True)
        score_of = {(r["case"], r["repeat"], r["position"]): float(sc) for r, sc in zip(rows, scores)}
        for name, _, _, probes in args.set:
            derived = args.derived_out / "system1_{}_{}_probes.jsonl".format(args.label, name)
            with derived.open("w", encoding="utf-8") as handle:
                for line in Path(probes).read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    p = json.loads(line)
                    if p.get("final"):
                        handle.write(line + "\n")  # the end of each reasoning grades "never cut"
                        continue
                    key = (p["case"], p["repeat"], p["position"])
                    if key not in score_of:
                        continue
                    q = {k: v for k, v in p.items() if k not in ("logprob", "min_prob", "trial")}
                    q["readiness"] = score_of[key]
                    q["probe_seconds"] = 0.0
                    handle.write(json.dumps(q, ensure_ascii=False) + "\n")
            print("  governor input written to " + str(derived))

    if args.artifact_out:
        from flyball.calibration import fit_platt, logit
        from flyball.hidden import LinearProbe

        l2 = float(np.median(chosen))
        head = fit_logistic(x, y, l2)
        platt = fit_platt([logit(float(v)) for v in scores], y)
        meta.update({"model": args.model, "sets": [s[0] for s in args.set], "l2": l2,
                     "oof_auc": auc(scores, y), "width": int(x.shape[1])})
        LinearProbe(head, platt, meta).save(args.artifact_out)
        print("  probe written to {} ({})".format(args.artifact_out, ", ".join("{}={}".format(k, v) for k, v in meta.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
