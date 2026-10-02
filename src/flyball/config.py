"""Configuration, read from the environment with conservative defaults.

Deliberately plain: no settings framework, so that `pip install flyballd` pulls in
as little as possible. Every value can also be overridden on the command line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Modes, from least to most invasive. See docs/modes.md.
#   off    - pure passthrough. flyball touches nothing.
#   shadow - decisions are computed and logged, but never applied.
#   active - approved decisions are applied to live traffic.
MODES = ("off", "shadow", "active")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# FLYBALL_THRESHOLD_NEEDS_DELIBERATION=0.85 and so on, one variable per question.
_THRESHOLD_PREFIX = "FLYBALL_THRESHOLD_"


def _thresholds_from_env() -> dict:
    out = {}
    for name, raw in os.environ.items():
        if name.startswith(_THRESHOLD_PREFIX) and raw.strip():
            try:
                out[name[len(_THRESHOLD_PREFIX) :].lower()] = float(raw)
            except ValueError:
                continue
    return out


@dataclass
class Settings:
    """Runtime configuration for the flyball daemon."""

    upstream: str = field(default_factory=lambda: _env("FLYBALL_UPSTREAM", "http://127.0.0.1:8081"))
    host: str = field(default_factory=lambda: _env("FLYBALL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("FLYBALL_PORT", 8080))

    upstream_api_key: str = field(default_factory=lambda: _env("FLYBALL_UPSTREAM_API_KEY"))
    api_key: str = field(default_factory=lambda: _env("FLYBALL_API_KEY"))

    mode: str = field(default_factory=lambda: _env("FLYBALL_MODE", "off"))
    data_dir: Path = field(default_factory=lambda: Path(_env("FLYBALL_DATA_DIR", "./data")))

    # Confidence thresholds per question, fitted on held-out data. A question with
    # no threshold is never acted on, even in active mode: there is deliberately
    # no universal default to fall back to.
    thresholds: dict = field(default_factory=lambda: _thresholds_from_env())

    # Share of conversations that receive the decision in active mode. The rest
    # are the control arm, logged as "would have applied".
    ab_treatment_share: float = field(default_factory=lambda: _env_float("FLYBALL_AB_SHARE", 0.5))

    # Local models are slow. A short read timeout would abort perfectly healthy
    # generations, so only connect and write are bounded by default.
    connect_timeout: float = 5.0
    read_timeout: float | None = None

    def __post_init__(self) -> None:
        self.upstream = self.upstream.rstrip("/")
        if self.mode not in MODES:
            raise ValueError(f"FLYBALL_MODE must be one of {MODES}, got {self.mode!r}")

    def upstream_url(self, path: str) -> str:
        return f"{self.upstream}/{path.lstrip('/')}"

    @property
    def kill_switch(self) -> Path:
        """Create this file to stop every decision immediately, without a restart."""
        return self.data_dir / "KILL"
