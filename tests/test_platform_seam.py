"""Contract tests for the platform seam maintained by this repository."""

from __future__ import annotations

import pytest

from multinav.platforms import (
    InteractionJoint,
    InteractionObject,
    PlatformInteractionInterface,
    available_platforms,
    get_platform,
    register_platform,
)


def test_dataclasses_are_frozen():
    joint = InteractionJoint(
        index=0,
        name="hinge",
        joint_type="hinge",
        position=0.0,
        closed_position=0.0,
        open_position=1.5,
        open_fraction=0.0,
    )
    obj = InteractionObject(name="door_0001", category="door", kind="door", joints=(joint,))
    with pytest.raises(Exception):
        joint.open_fraction = 1.0  # type: ignore[misc]
    with pytest.raises(Exception):
        obj.name = "other"  # type: ignore[misc]


def test_interface_requires_three_operations():
    class Incomplete(PlatformInteractionInterface):
        def scan(self):  # pragma: no cover - abstract surface only
            return ()

    with pytest.raises(TypeError):
        Incomplete()


def test_registry_roundtrip_and_duplicate_guard():
    name = "unit-test-platform"
    factory = lambda *a, **k: None  # noqa: E731
    register_platform(name, factory)
    assert get_platform(name) is factory
    assert name in available_platforms()
    with pytest.raises(ValueError):
        register_platform(name, factory)


def test_unknown_platform_reports_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_platform("does-not-exist")
    assert "registered platforms" in str(excinfo.value)


def test_molmospaces_adapter_satisfies_the_seam():
    molmo_spaces = pytest.importorskip("molmo_spaces")
    from multinav.platforms.molmospaces import SimulatorInteractionInterface

    assert issubclass(SimulatorInteractionInterface, PlatformInteractionInterface)
    assert "molmospaces" in available_platforms()
