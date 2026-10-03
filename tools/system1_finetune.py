#!/usr/bin/env python3
"""Fine-tune a decision model's encoder to read "it has its answer" from the reasoning.

system1_representation.py showed how much a frozen encoder carries; this tool trains
the whole encoder, with a small head on top, on the same labelled checkpoints. It is
the strongest test of the idea that a System 1 can read, from the text alone, when
the model has thought enough: if a fine-tuned encoder still falls well short of the
model's own confidence, reading the text is not the way.

Problems are split into five folds; each fold is scored by a model trained on the
other four, so every score comes from a model that never saw that problem. The
encoder is Laya's today (ModernBERT-large); any encoder would do.

Needs a GPU: stop the language model first, the training takes most of its memory.

Usage:
    python tools/system1_finetune.py \\
        --set math reports/math_traces.jsonl reports/math_cases.json reports/math_probes.jsonl \\
        --out F:/ai/flyball/research/laya_finetune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system1_reasoning import auc, state_of  # noqa: E402
from system1_representation import load_set  # noqa: E402

FOLDS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs=4, action="append", required=True,
                        metavar=("NAME", "TRACES", "CASES", "PROBES"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--question-chars", type=int, default=500)
    parser.add_argument("--tail-chars", type=int, default=1200)
    parser.add_argument("--folds", default="0,1,2,3,4", help="which folds to run, for resuming")
    args = parser.parse_args()

    import torch
    import laya

    rows = []
    for name, traces, cases, probes in args.set:
        for row in load_set(Path(traces), Path(cases), Path(probes)):
            row["set"] = name
            rows.append(row)
    texts = [state_of(r["prompt"], r["prefix"], args.question_chars, args.tail_chars) for r in rows]
    labels = np.array([1.0 if r["safe"] else 0.0 for r in rows], dtype=np.float32)
    fold = np.array([int(hashlib.sha256(r["case"].encode()).hexdigest(), 16) % FOLDS for r in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    print("{} checkpoints, {:.0%} with the answer in place; device {}".format(len(rows), labels.mean(), torch.cuda.get_device_name(0)))

    agent = laya.load("convaiinnovations/laya")
    tok = agent.tok
    # Only the encoder goes to the GPU; Laya's own decision head is not trained here.
    # The untouched weights every fold starts from stay in host memory: the GPU needs
    # all its memory for the weights, gradients and optimiser state being trained.
    agent.model.to("cpu")
    pristine = {k: v.detach().cpu().clone() for k, v in agent.model.encoder.state_dict().items()}
    encoded = tok(texts, truncation=True, max_length=args.max_len)

    def batches(indices, shuffle: bool, seed: int = 0):
        order = np.array(indices)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), args.batch):
            chunk = order[start:start + args.batch]
            padded = tok.pad({"input_ids": [encoded["input_ids"][i] for i in chunk],
                              "attention_mask": [encoded["attention_mask"][i] for i in chunk]},
                             return_tensors="pt")
            yield chunk, padded["input_ids"].to(device), padded["attention_mask"].to(device)

    class Reader(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            width = encoder.config.hidden_size
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(2 * width, 1))

        def forward(self, ids, mask):
            hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * m).sum(1) / m.sum(1)
            return self.head(torch.cat([hidden[:, 0], mean], dim=1)).squeeze(-1)

    scores_path = args.out / "oof_scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for f in [int(x) for x in args.folds.split(",")]:
        if str(f) in scores:
            print("fold {} already done".format(f))
            continue
        encoder = agent.model.encoder
        encoder.load_state_dict(pristine)  # every fold starts from the same encoder
        model = Reader(encoder).to(device)
        if hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        train = np.where(fold != f)[0]
        test = np.where(fold == f)[0]
        steps = args.epochs * math.ceil(len(train) / args.batch)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        began = time.time()
        model.train()
        step = 0
        for epoch in range(args.epochs):
            for chunk, ids, mask in batches(train, shuffle=True, seed=epoch + 10 * f):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                loss = loss_fn(logits.float(), torch.tensor(labels[chunk], device=device))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                schedule.step()
                step += 1
                if step % 100 == 0:
                    print("  fold {} step {}/{} loss {:.3f} ({:.0f} s)".format(f, step, steps, loss.item(), time.time() - began), flush=True)
        model.eval()
        out = {}
        with torch.no_grad():
            for chunk, ids, mask in batches(test, shuffle=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probabilities = torch.sigmoid(model(ids, mask).float()).cpu().numpy()
                for i, p in zip(chunk, probabilities):
                    out[int(i)] = float(p)
        scores[str(f)] = out
        scores_path.write_text(json.dumps(scores))
        fold_auc = auc(np.array([out[int(i)] for i in test]), labels[test])
        print("fold {}: AUC {:.3f} on {} held-out checkpoints, {:.0f} s".format(f, fold_auc, len(test), time.time() - began), flush=True)
        del model, optimiser
        torch.cuda.empty_cache()

    merged = {int(i): p for part in scores.values() for i, p in part.items()}
    covered = np.array(sorted(merged))
    oof = np.array([merged[i] for i in covered])
    print()
    print("  {:12} {:>18} {:>28}".format("set", "fine-tuned reader", "the model's own confidence"))
    for name in [s[0] for s in args.set] + ["all"]:
        mask = np.array([name == "all" or rows[i]["set"] == name for i in covered])
        own = np.array([rows[i]["own"] if rows[i]["own"] is not None else np.nan for i in covered])[mask]
        print("  {:12} {:>18.3f} {:>28}".format(name, auc(oof[mask], labels[covered][mask]),
                                                "{:.3f}".format(auc(own, labels[covered][mask])) if not np.isnan(own).any() else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
