"""Moved to servers.py, where every server-specific call now lives.

Kept so that code importing the first version's names still works.
"""

from __future__ import annotations

from .servers import (  # noqa: F401
    PROBE_CHECK_ID,
    Discovery,
    ServerCapabilities as Capabilities,
    probe,
    probe_action_supported,
    probe_sync,
)
