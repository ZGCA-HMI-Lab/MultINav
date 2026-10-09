"""MolmoSpaces adapter for the interactive navigation stack."""

from __future__ import annotations

from multinav.platforms import register_platform
from multinav.platforms.molmospaces.interface import SimulatorInteractionInterface


def _factory(*args, **kwargs) -> SimulatorInteractionInterface:
    return SimulatorInteractionInterface(*args, **kwargs)


try:
    register_platform("molmospaces", _factory)
except ValueError:
    pass


__all__ = ["SimulatorInteractionInterface"]
