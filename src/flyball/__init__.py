"""flyball - a System-1 governor for a local model's reasoning.

flyball does not pick which model answers your request. It watches the one model
you can actually fit in VRAM while it thinks, and closes the reasoning block when
it has thought enough.
"""

__version__ = "0.0.1"
