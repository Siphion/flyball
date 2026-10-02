"""The decision ledger: two append-only JSONL streams joined by decision_id.

decisions.jsonl  what flyball saw and what it would have decided
outcomes.jsonl   what actually happened to that request

They are written separately because an outcome is only known once the response has
finished streaming, possibly minutes later, and a crash in between must not lose
the decision.

These files contain prompt excerpts. They are local, excluded from version
control, and redacted on write: anything that looks like a credential is replaced
before it touches the disk.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

ROTATE_BYTES = 50 * 1024 * 1024

_SECRETS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S+"),
]


def redact(text: str) -> str:
    for pattern in _SECRETS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _redact_obj(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_obj(v) for v in value]
    return value


class JsonlStream:
    """Thread-safe append-only JSONL file with size-based rotation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        line = json.dumps(_redact_obj(record), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self.path.exists() and self.path.stat().st_size > ROTATE_BYTES:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self.path.rename(self.path.with_name(f"{self.path.stem}-{stamp}.jsonl"))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self) -> list[dict]:
        rows = []
        for path in sorted(self.path.parent.glob(f"{self.path.stem}*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue
        return rows


class Ledger:
    def __init__(self, data_dir: Path) -> None:
        self.decisions = JsonlStream(Path(data_dir) / "decisions.jsonl")
        self.outcomes = JsonlStream(Path(data_dir) / "outcomes.jsonl")
