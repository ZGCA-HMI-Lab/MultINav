"""Evaluator-private articulation execution aligned with ordinary navigation.

The historical V3 path drove every joint independently for a fixed duration.
That made a multi-part container cost ``joint_count * duration`` and bypassed
the group postcondition used by ordinary interactive navigation.  This adapter
uses the same ``prepare_articulation_state_force`` /
``complete_articulation_force`` backend as the ordinary force bridge while
retaining the evaluator's private allow-list of joints.

This is deliberately the ordinary bridge's *fast* execution mode.  A future
smooth benchmark mode should call ``advance_articulation_force`` and
``finalize_articulation_force_transition`` from the rollout's before/after
task-step hooks; it must not reintroduce a blocking per-joint loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from multinav.sim.force_interaction_runtime import (
    ForceDriveConfig,
    complete_articulation_force,
    prepare_articulation_state_force,
)
from multinav.sim.force_interaction_bridge import (
    _refrigerator_open_sweep_preflight,
    _requires_refrigerator_open_sweep,
)

from .trusted_interaction_skill import JointOpenResult


@dataclass(frozen=True)
class BenchmarkArticulationExecution:
    success: bool
    joint_results: tuple[JointOpenResult, ...]
    simulated_seconds: float
    physics_substeps: int
    pre_state: str
    post_state: str
    private_metadata: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


def execute_open_articulation_group(
    env: Any,
    *,
    object_name: str,
    joints: Sequence[Any],
    max_physics_substeps: int,
    success_fraction: float,
    robot_lock_callback: Callable[[], None] | None = None,
    public_command: Mapping[str, Any] | None = None,
) -> BenchmarkArticulationExecution:
    """Open the evaluator-allowed joints as one ordinary force group."""

    ordered_joints = tuple(joints)
    joint_names = [str(joint.joint_name) for joint in ordered_joints]
    if not joint_names:
        raise ValueError("evaluator articulation group requires at least one joint")
    plan = prepare_articulation_state_force(
        env,
        str(object_name),
        open_joint_names=joint_names,
    )
    open_sweep_preflight: dict[str, Any] | None = None
    if _requires_refrigerator_open_sweep(dict(public_command or {})):
        open_sweep_preflight = dict(
            _refrigerator_open_sweep_preflight(env, plan)
        )
        if bool(open_sweep_preflight.get("checked")) and not bool(
            open_sweep_preflight.get("safe", True)
        ):
            before_by_name = {
                str(row.get("joint_name") or ""): row
                for row in plan.get("pre_joint_infos") or []
            }
            joint_results = tuple(
                JointOpenResult(
                    executor_succeeded=False,
                    open_fraction_before=(
                        None
                        if before_by_name.get(str(joint.joint_name), {}).get(
                            "open_fraction"
                        )
                        is None
                        else float(
                            before_by_name[str(joint.joint_name)]["open_fraction"]
                        )
                    ),
                    open_fraction_after=(
                        None
                        if before_by_name.get(str(joint.joint_name), {}).get(
                            "open_fraction"
                        )
                        is None
                        else float(
                            before_by_name[str(joint.joint_name)]["open_fraction"]
                        )
                    ),
                    error="unsafe_open_sweep",
                )
                for joint in ordered_joints
            )
            return BenchmarkArticulationExecution(
                success=False,
                joint_results=joint_results,
                simulated_seconds=0.0,
                physics_substeps=0,
                pre_state=str(plan.get("pre_state") or "unknown"),
                post_state=str(plan.get("pre_state") or "unknown"),
                private_metadata={
                    "execution_mode": "ordinary_force_group_fast",
                    "public_failure_reason": "unsafe_open_sweep",
                    "open_sweep_preflight": open_sweep_preflight,
                    "task_steps_consumed": 0,
                },
            )
    result = complete_articulation_force(
        env,
        plan,
        config=ForceDriveConfig(
            max_physics_substeps=max(1, int(max_physics_substeps)),
            open_fraction_threshold=float(success_fraction),
            assume_success=False,
        ),
        robot_lock_callback=robot_lock_callback,
    )
    before_by_name = {
        str(row.get("joint_name") or ""): row
        for row in result.get("pre_joint_infos") or []
    }
    after_by_name = {
        str(row.get("joint_name") or ""): row
        for row in result.get("joint_infos") or []
    }
    physics_substeps = int(result.get("physics_substeps", 0) or 0)
    simulated_seconds = physics_substeps * float(env.current_model.opt.timestep)
    joint_results: list[JointOpenResult] = []
    for index, joint in enumerate(ordered_joints):
        name = str(joint.joint_name)
        before = before_by_name.get(name) or {}
        after = after_by_name.get(name) or {}
        fraction_before = before.get("open_fraction")
        fraction_after = after.get("open_fraction")
        reached = (
            fraction_after is not None
            and float(fraction_after) >= float(success_fraction)
        )
        joint_results.append(
            JointOpenResult(
                executor_succeeded=bool(reached),
                open_fraction_before=(
                    None if fraction_before is None else float(fraction_before)
                ),
                open_fraction_after=(
                    None if fraction_after is None else float(fraction_after)
                ),
                # Preserve object-level simulated time exactly once.  Consumers
                # historically sum this field over joint results.
                simulated_seconds=simulated_seconds if index == 0 else 0.0,
                metadata={
                    "physics_substeps": physics_substeps if index == 0 else 0,
                    "atomic_fallback": bool(result.get("atomic_fallback", False)),
                },
            )
        )
    success = bool(result.get("success")) and all(
        item.executor_succeeded for item in joint_results
    )
    return BenchmarkArticulationExecution(
        success=success,
        joint_results=tuple(joint_results),
        simulated_seconds=simulated_seconds,
        physics_substeps=physics_substeps,
        pre_state=str(result.get("pre_state") or "unknown"),
        post_state=(
            str(result.get("post_state") or "open") if success else "blocked"
        ),
        private_metadata={
            "execution_mode": "ordinary_force_group_fast",
            "atomic_fallback": bool(result.get("atomic_fallback", False)),
            "physical_success": bool(result.get("physical_success", False)),
            "task_steps_consumed": int(result.get("task_steps_consumed", 1) or 1),
            "open_sweep_preflight": open_sweep_preflight,
        },
    )


__all__ = [
    "BenchmarkArticulationExecution",
    "execute_open_articulation_group",
]
