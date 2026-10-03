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

    # Gate B, the mid-stream governor (docs/gate-b-research.md, sections 5 to 7).
    # It acts only in active mode, only when switched on here, and only with an
    # upstream that offers the `probe` control action. The defaults are the
    # operating point measured live: cut once the model is 0.8 confident in its
    # answer, and in any case at 3,200 characters of reasoning.
    governor: bool = field(
        default_factory=lambda: _env("FLYBALL_GOVERNOR", "off").lower() in ("1", "on", "true", "yes")
    )
    governor_confidence: float = field(default_factory=lambda: _env_float("FLYBALL_GOVERNOR_CONFIDENCE", 0.8))
    governor_cap: int = field(default_factory=lambda: _env_int("FLYBALL_GOVERNOR_CAP", 3200))
    # Close the reasoning with a sentence that commits to the answer the probe found.
    # On by default: measured live, it kept accuracy and cut total time by a fifth,
    # because the reply no longer picks the reasoning back up (section 8).
    governor_anchor: bool = field(
        default_factory=lambda: _env("FLYBALL_GOVERNOR_ANCHOR", "on").lower() in ("1", "on", "true", "yes")
    )

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
