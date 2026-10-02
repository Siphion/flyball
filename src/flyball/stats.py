"""Read the ledger back: join decisions to outcomes and summarise them.

The re-ask signal is computed here rather than at request time. A request whose
last user message is sent again shortly afterwards most likely got an answer the
user did not accept; it is the cheapest negative label there is, and it needs no
model and no human.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .ledger import Ledger

REASK_WINDOW_S = 120.0


def join(data_dir: Path) -> list[dict]:
    ledger = Ledger(data_dir)
    outcomes = {row["decision_id"]: row for row in ledger.outcomes.read() if "decision_id" in row}
    decisions = sorted(ledger.decisions.read(), key=lambda row: row.get("ts", 0))

    rows = []
    for index, decision in enumerate(decisions):
        row = dict(decision)
        row["outcome"] = outcomes.get(decision.get("decision_id"))
        row["reasked"] = any(
            later.get("last_user_hash") == decision.get("last_user_hash")
            and 0 < later.get("ts", 0) - decision.get("ts", 0) <= REASK_WINDOW_S
            for later in decisions[index + 1 : index + 20]
        )
        rows.append(row)
    return rows


def _median(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    return "{:.1f}".format(statistics.median(values)) if values else "-"


def report(data_dir: Path) -> int:
    rows = join(data_dir)
    if not rows:
        print("No decisions recorded in " + str(data_dir) + ". Run flyballd serve --mode shadow.")
        return 1

    with_outcome = [r for r in rows if r["outcome"]]
    print("decisions {}   with outcome {}   schema versions {}   state versions {}".format(
        len(rows),
        len(with_outcome),
        len({r.get("schema_sha256") for r in rows}),
        len({r.get("state_version") for r in rows}),
    ))
    print()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("values", {}).get("request_class", "?")].append(row)

    header = ("class", "n", "deliberate", "median s", "ttft", "reasoning ch", "aborted", "re-asked")
    print("{:10} {:>5} {:>11} {:>9} {:>6} {:>13} {:>8} {:>9}".format(*header))
    for name in sorted(groups):
        group = groups[name]
        done = [r["outcome"] for r in group if r["outcome"]]
        print("{:10} {:>5} {:>10.0%} {:>9} {:>6} {:>13} {:>8} {:>9}".format(
            name,
            len(group),
            sum(1 for r in group if r.get("values", {}).get("needs_deliberation")) / len(group),
            _median([o.get("seconds") for o in done]),
            _median([o.get("ttft_s") for o in done]),
            _median([o.get("reasoning_chars") for o in done]),
            sum(1 for o in done if o.get("aborted_by_client")),
            sum(1 for r in group if r["reasked"]),
        ))

    deliberate = [r for r in with_outcome if r.get("values", {}).get("needs_deliberation")]
    skip = [r for r in with_outcome if not r.get("values", {}).get("needs_deliberation")]
    if deliberate and skip:
        print()
        print("Requests the heuristic would let think:  median reasoning {} chars".format(
            _median([r["outcome"].get("reasoning_chars") for r in deliberate])))
        print("Requests it would tell not to think:      median reasoning {} chars".format(
            _median([r["outcome"].get("reasoning_chars") for r in skip])))
        print("A gate worth anything separates these two numbers.")
    return 0
