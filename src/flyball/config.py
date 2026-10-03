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
SERVERS = ("auto", "llama.cpp", "openai")


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


def _env_opt_float(name: str) -> float | None:
    raw = _env(name)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _env_opt_int(name: str) -> int | None:
    raw = _env(name)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _env_opt_bool(name: str) -> bool | None:
    raw = _env(name).lower()
    if not raw:
        return None
    return raw in ("1", "on", "true", "yes")


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

    # The server and the model. The server's kind is discovered unless given
    # (auto, llama.cpp, openai); the model's profile is matched on the model id the
    # server reports (profiles.py), unless one is named here. Profiles can be added
    # or extended from a JSON file.
    server: str = field(default_factory=lambda: _env("FLYBALL_SERVER", "auto"))
    profile: str = field(default_factory=lambda: _env("FLYBALL_PROFILE"))
    profiles_path: str = field(default_factory=lambda: _env("FLYBALL_PROFILES"))
    # The reasoning level of a request that names none. A server started with its
    # own default (llama-server --reasoning-effort medium) does not report it, so
    # it is declared here; empty means the profile's default, the template's own.
    default_level: str = field(default_factory=lambda: _env("FLYBALL_DEFAULT_LEVEL"))

    # Gate B, the mid-stream governor (docs/gate-b-research.md, sections 5 to 8).
    # It acts only in active mode, only when switched on here, and only with an
    # upstream that can probe a live completion. The operating point comes from the
    # model's profile, per reasoning level: a level without one is not governed.
    # The variables below override the profile's point when they are set.
    governor: bool = field(
        default_factory=lambda: _env("FLYBALL_GOVERNOR", "off").lower() in ("1", "on", "true", "yes")
    )
    governor_confidence: float | None = field(default_factory=lambda: _env_opt_float("FLYBALL_GOVERNOR_CONFIDENCE"))
    governor_cap: int | None = field(default_factory=lambda: _env_opt_int("FLYBALL_GOVERNOR_CAP"))
    governor_first_probe: int | None = field(default_factory=lambda: _env_opt_int("FLYBALL_GOVERNOR_FIRST_PROBE"))
    governor_growth: float | None = field(default_factory=lambda: _env_opt_float("FLYBALL_GOVERNOR_GROWTH"))
    # Close the reasoning with a sentence that commits to the answer the probe found.
    # On in every measured profile: live, it kept accuracy and cut total time by a
    # fifth, because the reply no longer picks the reasoning back up (section 8).
    governor_anchor: bool | None = field(default_factory=lambda: _env_opt_bool("FLYBALL_GOVERNOR_ANCHOR"))

    # A System One server (Laya's, Jev, ...), for decision backends that use one.
    systemone_url: str = field(default_factory=lambda: _env("FLYBALL_SYSTEMONE_URL"))
    systemone_api_key: str = field(default_factory=lambda: _env("FLYBALL_SYSTEMONE_API_KEY"))

    # Local models are slow. A short read timeout would abort perfectly healthy
    # generations, so only connect and write are bounded by default.
    connect_timeout: float = 5.0
    read_timeout: float | None = None

    def __post_init__(self) -> None:
        self.upstream = self.upstream.rstrip("/")
        if self.mode not in MODES:
            raise ValueError(f"FLYBALL_MODE must be one of {MODES}, got {self.mode!r}")
        if self.server not in SERVERS:
            raise ValueError(f"FLYBALL_SERVER must be one of {SERVERS}, got {self.server!r}")

    def governor_overrides(self) -> dict:
        """The operating-point fields set in the environment, by GovernorConfig name."""
        values = {
            "confidence": self.governor_confidence,
            "cap": self.governor_cap,
            "first_probe": self.governor_first_probe,
            "growth": self.governor_growth,
            "anchor": self.governor_anchor,
        }
        return {name: value for name, value in values.items() if value is not None}

    def upstream_url(self, path: str) -> str:
        return f"{self.upstream}/{path.lstrip('/')}"

    @property
    def kill_switch(self) -> Path:
        """Create this file to stop every decision immediately, without a restart."""
        return self.data_dir / "KILL"
