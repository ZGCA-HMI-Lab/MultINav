"""Demonstration policy for the simulator interaction interface.

The policy is intentionally not a navigation algorithm. With no commands it
only scans and reports available doors/containers. To demonstrate execution,
pass explicit commands such as::

    commands = (
        InteractionDemoCommand.open_door("door_body_name"),
        InteractionDemoCommand.open_joint("cabinet_body_name", joint_index=2),
    )
    policy = InteractiveNavSimulatorPolicy(config, task, commands=commands)

The second command opens only joint 2, which is the single-drawer operation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from multinav.platforms.molmospaces.interface import (
    InteractionObject,
    SimulatorInteractionInterface,
)
from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask


@dataclass(frozen=True)
class InteractionDemoCommand:
    operation: Literal["open_door", "open_joint"]
    object_name: str
    joint_index: int | None = None
    open_fraction: float = 1.0

    @classmethod
    def open_door(
        cls, object_name: str, open_fraction: float = 1.0
    ) -> "InteractionDemoCommand":
        return cls("open_door", object_name, open_fraction=open_fraction)

    @classmethod
    def open_joint(
        cls,
        object_name: str,
        joint_index: int,
        open_fraction: float = 1.0,
    ) -> "InteractionDemoCommand":
        return cls("open_joint", object_name, joint_index, open_fraction)


class InteractiveNavSimulatorPolicy(BasePolicy):
    """Scan interaction capabilities and optionally replay explicit examples."""

    available_interfaces = (
        "scan doors, cabinets, and drawers",
        "read each hinge/slider state and normalized open fraction",
        "open a door hinge",
        "open or close one selected cabinet/drawer joint",
    )

    def __init__(
        self,
        config,
        task: BaseMujocoTask | None = None,
        *,
        commands: Sequence[InteractionDemoCommand] = (),
        interface_factory: Callable = SimulatorInteractionInterface,
    ) -> None:
        super().__init__(config, task)
        self.commands = tuple(commands)
        self._interface_factory = interface_factory
        self.interface: SimulatorInteractionInterface | None = None
        self.catalog: tuple[InteractionObject, ...] = ()
        self.results: list[InteractionObject] = []
        self._command_index = 0

    def reset(self) -> None:
        if self.task is None:
            raise RuntimeError("InteractiveNavSimulatorPolicy must be registered with a task")
        self.interface = self._interface_factory(self.task.env)
        self.catalog = self.interface.scan()
        self.results = []
        self._command_index = 0

    def get_action(self, observation):
        del observation
        if self.interface is None:
            self.reset()
        assert self.interface is not None
        assert self.task is not None
        if self._command_index < len(self.commands):
            command = self.commands[self._command_index]
            if command.operation == "open_door":
                result = self.interface.open_door(
                    command.object_name, command.open_fraction
                )
            else:
                assert command.joint_index is not None
                result = self.interface.set_joint_open_fraction(
                    command.object_name,
                    command.joint_index,
                    command.open_fraction,
                )
            self.results.append(result)
            self._command_index += 1

        action = self.task.env.current_robot.robot_view.get_noop_ctrl_dict()
        action["done"] = self._command_index >= len(self.commands)
        return action

    def get_info(self) -> dict:
        return {
            "available_interfaces": list(self.available_interfaces),
            "catalog": [asdict(item) for item in self.catalog],
            "executed_commands": [asdict(item) for item in self.results],
        }
