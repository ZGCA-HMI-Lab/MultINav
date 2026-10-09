"""Deferred access to simulator packages used by the ported simulation layer.

The scoring core and the episode contract import without MuJoCo or MolmoSpaces
installed, but the simulator-side modules ported from ``interactive-nav/sim``
still touch those packages once a rollout actually runs.  Resolving them on
first attribute access keeps an unused import path free of the simulator while
leaving the ported call sites readable.

Every module reachable through :func:`lazy_module` is a migration TODO: as each
one moves behind the platform seam in :mod:`multinav.platforms`, its proxy
disappears and the import becomes a plain one.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_INSTALL_HINT = (
    "install the simulator extra, for example: pip install -e '.[molmospaces]'"
)


class _LazyModule:
    """Module proxy that imports its target on the first attribute access."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError as exc:
                raise ImportError(
                    f"{self._name} is not importable; {_INSTALL_HINT}"
                ) from exc
        return self._module

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))

    def __repr__(self) -> str:
        state = "loaded" if self._module is not None else "deferred"
        return f"<lazy module {self._name!r} ({state})>"


def lazy_module(name: str) -> Any:
    """Return a proxy that imports ``name`` when its first attribute is used."""
    return _LazyModule(name)


def load(name: str) -> ModuleType:
    """Import ``name`` now, reporting the simulator extra when it is missing."""
    return _LazyModule(name)._load()


__all__ = ["lazy_module", "load"]
