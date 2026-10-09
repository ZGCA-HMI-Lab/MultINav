"""MolmoSpaces implementation of the :mod:`multinav.platforms` seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np

from molmo_spaces.env.data_views import Door, MlSpacesArticulationObject

from multinav.platforms.base import (
    InteractionKind,
    InteractionJoint,
    InteractionObject,
    PlatformInteractionInterface,
)

if TYPE_CHECKING:
    from molmo_spaces.env.env import BaseMujocoEnv


class SimulatorInteractionInterface(PlatformInteractionInterface):
    """Scan and actuate scene articulations without exposing MuJoCo indexing.

    Joint indices in this interface are local to an ``InteractionObject`` and
    remain the same indices accepted by ``MlSpacesArticulationObject``.
    ``open_fraction`` is normalized: zero is the joint position nearest zero,
    while one is the farther endpoint of the configured joint range.
    """

    _CONTAINER_TOKENS = ("cabinet", "drawer", "fridge", "refrigerator", "cupboard", "wardrobe")

    def __init__(
        self,
        env: "BaseMujocoEnv",
        batch_index: int | None = None,
        *,
        forward: Callable = mujoco.mj_forward,
        door_factory: Callable = Door,
    ) -> None:
        self._env = env
        self._batch_index = (
            int(env.current_batch_index) if batch_index is None else int(batch_index)
        )
        self._forward = forward
        self._door_factory = door_factory

    @property
    def _manager(self):
        return self._env.object_managers[self._batch_index]

    @property
    def _data(self):
        return self._env.mj_datas[self._batch_index]

    @staticmethod
    def _joint_type_name(joint_type) -> Literal["hinge", "slide"] | None:
        value = int(np.asarray(joint_type).reshape(-1)[0])
        if value == int(mujoco.mjtJoint.mjJNT_HINGE):
            return "hinge"
        if value == int(mujoco.mjtJoint.mjJNT_SLIDE):
            return "slide"
        return None

    @staticmethod
    def _closed_open_positions(joint_range) -> tuple[float, float]:
        lower, upper = (float(value) for value in joint_range)
        closed = 0.0 if lower <= 0.0 <= upper else min((lower, upper), key=abs)
        opened = lower if abs(lower - closed) >= abs(upper - closed) else upper
        return closed, opened

    def _door_names(self) -> set[str]:
        return set(self._manager.find_door_names())

    def _resolve(self, object_name: str, door_names: set[str] | None = None):
        known_doors = self._door_names() if door_names is None else door_names
        if object_name in known_doors:
            return self._door_factory(object_name, self._data), True
        obj = self._manager.get_object_by_name(object_name)
        if not isinstance(obj, MlSpacesArticulationObject):
            raise ValueError(f"Object {object_name!r} is not articulable")
        return obj, False

    def _summarize(
        self,
        obj: MlSpacesArticulationObject,
        *,
        is_door: bool,
    ) -> InteractionObject:
        metadata = self._manager.object_metadata(obj.name)
        category = str(metadata.get("category") or obj.name)
        normalized = f"{category} {obj.name}".lower()
        kind: InteractionKind
        if is_door:
            kind = "door"
        elif any(token in normalized for token in self._CONTAINER_TOKENS):
            kind = "container"
        else:
            kind = "articulation"

        joints = []
        for index in range(obj.njoints):
            joint_type = self._joint_type_name(obj.get_joint_type(index))
            if joint_type is None:
                continue
            closed, opened = self._closed_open_positions(obj.get_joint_range(index))
            position = float(obj.get_joint_position(index))
            span = opened - closed
            fraction = 0.0 if abs(span) <= 1e-12 else (position - closed) / span
            joints.append(
                InteractionJoint(
                    index=index,
                    name=str(obj.joint_names[index]),
                    joint_type=joint_type,
                    position=position,
                    closed_position=closed,
                    open_position=opened,
                    open_fraction=float(np.clip(fraction, 0.0, 1.0)),
                )
            )
        return InteractionObject(
            name=obj.name,
            category=category,
            kind=kind,
            joints=tuple(joints),
        )

    def scan(self) -> tuple[InteractionObject, ...]:
        """Return all doors, cabinets, drawers, and other scene articulations."""

        door_names = self._door_names()
        objects: dict[str, InteractionObject] = {}
        for door_name in sorted(door_names):
            door, _ = self._resolve(door_name, door_names)
            summary = self._summarize(door, is_door=True)
            if summary.joints:
                objects[summary.name] = summary
        for obj in self._manager.list_top_level_objects():
            if obj.name in objects or not isinstance(obj, MlSpacesArticulationObject):
                continue
            summary = self._summarize(obj, is_door=False)
            if summary.joints:
                objects[summary.name] = summary
        return tuple(objects[name] for name in sorted(objects))

    def set_joint_open_fraction(
        self,
        object_name: str,
        joint_index: int,
        open_fraction: float,
    ) -> InteractionObject:
        """Move exactly one door leaf, cabinet door, or drawer joint."""

        fraction = float(open_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("open_fraction must be finite and in [0, 1]")
        obj, is_door = self._resolve(object_name)
        index = int(joint_index)
        if index < 0 or index >= obj.njoints:
            raise IndexError(f"joint_index {index} is invalid for {object_name!r}")
        if self._joint_type_name(obj.get_joint_type(index)) is None:
            raise ValueError(f"Joint {index} on {object_name!r} is not a hinge or slider")
        closed, opened = self._closed_open_positions(obj.get_joint_range(index))
        obj.set_joint_position(index, closed + fraction * (opened - closed))
        self._forward(self._env.current_model, self._data)
        self._manager.invalidate_data_cache()
        return self._summarize(obj, is_door=is_door)

    def open_door(self, door_name: str, open_fraction: float = 1.0) -> InteractionObject:
        """Open one door through its hinge while leaving handle joints untouched."""

        door_names = self._door_names()
        if door_name not in door_names:
            raise ValueError(f"Object {door_name!r} is not a scene door")
        door, _ = self._resolve(door_name, door_names)
        return self.set_joint_open_fraction(
            door_name,
            door.get_hinge_joint_index(),
            open_fraction,
        )
