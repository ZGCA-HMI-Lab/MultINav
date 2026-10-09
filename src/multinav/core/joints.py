"""Articulation math shared by the simulation layer and the scorer.

These helpers are simulator-neutral: they operate on a MuJoCo ``model`` /
``data`` pair through the two named fields the evaluator reads, ``qpos`` and
``jnt_range``.  Sharing them keeps the semantic closed/open convention
identical between episode generation, force execution and scoring, which is
what makes the benchmark's ``SUCCESS_OPEN_FRACTION`` comparable across paths.

MuJoCo itself is resolved lazily so that importing the scoring core does not
require the simulator to be installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


from multinav.core.simulator import lazy_module

mujoco = lazy_module("mujoco")


def joint_name_to_id(model: Any, joint_name: str) -> int:
    """Return the joint id of ``joint_name``, raising when the model has none."""
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return joint_id


def joint_value_by_name(model: Any, data: Any, joint_name: str) -> float:
    """Return the current ``qpos`` value of ``joint_name``."""
    joint_id = joint_name_to_id(model, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    return float(data.qpos[qpos_address])


def joint_range_by_name(model: Any, joint_name: str) -> tuple[float, float]:
    """Return the ``(lower, upper)`` configured range of ``joint_name``."""
    joint_id = joint_name_to_id(model, joint_name)
    lower, upper = (float(value) for value in model.jnt_range[joint_id])
    return lower, upper


def joint_closed_open_values(joint_range: list[float]) -> tuple[float, float]:
    """Map an articulation range to semantic closed/open endpoints."""
    values = [float(value) for value in joint_range]
    if not values:
        raise ValueError("Joint range is empty")
    closed = min(values, key=lambda value: abs(value))
    open_value = max(values, key=lambda value: abs(value - closed))
    return float(closed), float(open_value)


def semantic_open_fraction(
    value: float, closed_value: float, open_value: float
) -> float:
    """Normalize a joint value using its semantic closed/open endpoints."""
    span = float(open_value) - float(closed_value)
    if abs(span) <= 1e-8:
        return 0.0
    return float(np.clip((float(value) - float(closed_value)) / span, 0.0, 1.0))


__all__ = [
    "joint_closed_open_values",
    "joint_name_to_id",
    "joint_range_by_name",
    "joint_value_by_name",
    "semantic_open_fraction",
]
