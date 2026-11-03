"""Runtime-only optimizations of a loaded model (launch-overhead removal).

Imported only when `optimize != "none"`. Never edits files in the HF cache: it wraps or
replaces bound methods and generate() arguments on the loaded model instance.
"""

from __future__ import annotations

from typing import Any

from ..config import ProfileConfig
from ..counting import is_bolmo_like


class OptimizeError(RuntimeError):
    """The requested optimization cannot be applied or captured."""


class Handle:
    """What an applied optimization hands back to the runner."""

    def generate_kwargs(self) -> dict[str, Any]:
        return {}

    def describe(self) -> str:
        return ""

    def begin(self) -> None:
        """Called right before each measured generation (instrumentation hook)."""

    def end(self) -> dict[str, Any] | None:
        """Called right after it; returns instrumentation data or None."""
        return None

    def close(self) -> None:
        pass


def apply(model: Any, tok: Any, cfg: ProfileConfig) -> Handle:
    if is_bolmo_like(model):
        from . import bolmo

        return bolmo.apply(model, tok, cfg)
    from . import olmo

    return olmo.apply(model, tok, cfg)
