"""Per-simulator adapters for the interactive navigation stack.

MolmoSpaces is the first implemented platform. An Isaac Sim adapter only has to
implement :class:`~multinav.platforms.base.PlatformInteractionInterface` and
register itself with :func:`register_platform`; the evaluation core does not
change.
"""

from __future__ import annotations

from collections.abc import Callable

from multinav.platforms.base import (
    InteractionJoint,
    InteractionObject,
    PlatformInteractionInterface,
)

PlatformFactory = Callable[..., PlatformInteractionInterface]

_PLATFORMS: dict[str, PlatformFactory] = {}


def register_platform(name: str, factory: PlatformFactory) -> None:
    """Register a simulator adapter under ``name`` (for example ``"molmospaces"``)."""
    if name in _PLATFORMS:
        raise ValueError(f"platform {name!r} is already registered")
    _PLATFORMS[name] = factory


def get_platform(name: str) -> PlatformFactory:
    """Return the factory registered for ``name``."""
    try:
        return _PLATFORMS[name]
    except KeyError:
        known = ", ".join(sorted(_PLATFORMS)) or "<none>"
        raise KeyError(f"unknown platform {name!r}; registered platforms: {known}") from None


def available_platforms() -> tuple[str, ...]:
    """Return the names of all registered platforms."""
    return tuple(sorted(_PLATFORMS))


__all__ = [
    "InteractionJoint",
    "InteractionObject",
    "PlatformFactory",
    "PlatformInteractionInterface",
    "available_platforms",
    "get_platform",
    "register_platform",
]
