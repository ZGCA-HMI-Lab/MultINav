"""Platform seam for interactive navigation.

The evaluation core must not depend on MuJoCo, ROS, or MolmoSpaces. Everything a
policy or an evaluator is allowed to know about the simulated world is expressed
here, and every simulator must implement :class:`PlatformInteractionInterface`.

Two rules keep this seam stable:

* only *public* state crosses it -- object names, joint indices, positions and
  normalized open fractions; never joint axes, hinge geometry, task recipes, or
  evaluator-private identifiers;
* it exposes operations, not physics -- ``open_door`` and
  ``set_joint_open_fraction`` move exactly one articulation and report the
  resulting public state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

InteractionKind = Literal["door", "container", "articulation"]
JointType = Literal["hinge", "slide"]


@dataclass(frozen=True)
class InteractionJoint:
    """Public state of one controllable hinge or slider."""

    index: int
    name: str
    joint_type: JointType
    position: float
    closed_position: float
    open_position: float
    open_fraction: float


@dataclass(frozen=True)
class InteractionObject:
    """An articulable scene object exposed to an interactive policy."""

    name: str
    category: str
    kind: InteractionKind
    joints: tuple[InteractionJoint, ...]


class PlatformInteractionInterface(ABC):
    """Scan and actuate scene articulations without exposing physics indexing.

    Joint indices are local to an :class:`InteractionObject` and are the indices
    accepted by :meth:`set_joint_open_fraction`. ``open_fraction`` is normalized:
    zero is the joint position nearest zero and one is the farther endpoint of
    the configured joint range.
    """

    @abstractmethod
    def scan(self) -> tuple[InteractionObject, ...]:
        """Return every door, container, and other scene articulation."""

    @abstractmethod
    def open_door(self, door_name: str, open_fraction: float = 1.0) -> InteractionObject:
        """Open one door through its hinge, leaving handle joints untouched."""

    @abstractmethod
    def set_joint_open_fraction(
        self,
        object_name: str,
        joint_index: int,
        open_fraction: float,
    ) -> InteractionObject:
        """Move exactly one hinge or slider of ``object_name``."""
