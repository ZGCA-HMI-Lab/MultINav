from __future__ import annotations

import json
import math
import queue
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

# This module is imported both from package-qualified benchmark code and from
# standalone InteractiveNav scripts/tests whose sys.path starts in this folder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multinav.core.simulator import lazy_module
from multinav.sim.force_interaction_runtime import (
    ForceDriveConfig,
    HeadViewController,
    advance_articulation_force,
    articulation_joint_infos,
    complete_articulation_force,
    collect_articulation_groups,
    finalize_articulation_force_transition,
    ground_drawer_open_regions,
    infer_articulation_front_axis_xy,
    prepare_articulation_force,
    prepare_articulation_state_force,
    open_door_root_with_force,
    set_all_articulations_closed,
    set_all_door_roots_closed,
)

mujoco = lazy_module("mujoco")


def drawer_sequence_task_step_budget(
    group_count: int,
    transition_steps: int,
    observation_steps: int,
    restore_settle_steps: int,
    *,
    preserve_open: bool = False,
) -> int:
    """Return the finite simulator-step budget of a grounded drawer macro.

    This deliberately describes task steps rather than host time.  A scan
    opens, observes, and closes each grounded drawer; an exploration open omits
    the close phase.  The value is public execution progress, not simulator
    topology or joint metadata.
    """

    groups = max(0, int(group_count))
    transition = max(1, int(transition_steps))
    observation = max(1, int(observation_steps))
    # Even a zero configured visual settle still needs one simulator step to
    # measure the restored head/torso state.
    settle = max(1, int(restore_settle_steps))
    per_group = transition + observation
    if not preserve_open:
        per_group += transition
    return groups * per_group + settle


def _capture_robot_lock(task_env) -> dict[str, Any] | None:
    """Snapshot the base and non-camera upper body during a drawer macro.

    Head and torso controls intentionally stay outside this lock. They own the
    drawer low-view target and must keep driving it while force substeps run;
    replaying their pre-view no-op controls would otherwise undo the requested
    camera posture.
    """

    try:
        robot_view = task_env.current_robot.robot_view
        base = robot_view.base
        groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
            try:
                group = robot_view.get_move_group(name)
            except (AttributeError, KeyError, ValueError):
                continue
            groups[name] = (
                np.asarray(group.joint_pos, dtype=float).copy(),
                np.zeros_like(np.asarray(group.joint_vel, dtype=float)),
                np.asarray(group.noop_ctrl, dtype=float).copy(),
            )
        return {
            "base_pose": np.asarray(base.pose, dtype=float).copy(),
            "base_ctrl": np.asarray(base.ctrl, dtype=float).copy(),
            "base_hold_target": np.asarray(
                [
                    float(base.pose[0, 3]),
                    float(base.pose[1, 3]),
                    math.atan2(float(base.pose[1, 0]), float(base.pose[0, 0])),
                ],
                dtype=float,
            ),
            "groups": groups,
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _apply_robot_lock(task_env, snapshot: dict[str, Any] | None) -> None:
    """Reassert the interaction pose before/after every force substep."""

    if snapshot is None:
        return
    try:
        robot_view = task_env.current_robot.robot_view
        base = robot_view.base
        base.pose = np.asarray(snapshot["base_pose"], dtype=float).copy()
        base.joint_vel = np.zeros_like(base.joint_vel)
        base_hold_target = np.asarray(snapshot["base_hold_target"], dtype=float)
        base_ctrl = np.asarray(snapshot["base_ctrl"], dtype=float)
        try:
            base.ctrl = (
                base_hold_target.copy()
                if np.asarray(base.ctrl).shape == base_hold_target.shape
                else base_ctrl.copy()
            )
        except (AttributeError, ValueError):
            pass
        for name, (qpos, qvel, ctrl) in dict(snapshot["groups"]).items():
            group = robot_view.get_move_group(name)
            group.joint_pos = qpos.copy()
            group.joint_vel = qvel.copy()
            try:
                group.ctrl = ctrl.copy()
            except (AttributeError, ValueError):
                pass
        mujoco.mj_forward(task_env.current_model, task_env.current_data)
        try:
            task_env.camera_manager.registry.update_all_cameras(task_env)
        except AttributeError:
            pass
    except (AttributeError, KeyError, TypeError, ValueError):
        # The bridge can also be unit-tested with a reduced mock environment;
        # in a real simulator a complete snapshot is always available.
        return


def _portal_aperture_feedback(
    command: dict[str, Any],
) -> tuple[str, str, dict[str, Any] | None]:
    """Resolve a fixed portal state from public geometry evidence only.

    A failed articulation lookup is ambiguous: both a fixed open doorway and
    a non-articulated door leaf have no MuJoCo joint.  The bridge therefore
    accepts an optional, *public* observation produced by mapping/perception.
    It must explicitly report both the door-leaf status and aperture
    connectivity before ``static_open`` can be emitted.  No simulator body,
    asset, or joint name is read or copied into the result.

    The compact wire contract is::

        {"door_leaf": "absent|present|unknown",
         "connectivity": "open|blocked|unknown",
         "confidence": 0.0..1.0}

    Boolean aliases are accepted for producers that already expose
    ``no_door_leaf``/``aperture_connected`` or ``traversable``.  Missing or
    contradictory evidence remains ``unavailable`` and is terminal.
    """

    node_type = str(command.get("node_type") or "").strip().casefold()
    if node_type != "portal":
        return "", "", None
    raw = command.get("portal_aperture_observation")
    if not isinstance(raw, dict):
        return "unavailable", "unavailable", None

    leaf = str(raw.get("door_leaf") or raw.get("leaf") or "unknown").strip().casefold()
    if raw.get("no_door_leaf") is True or raw.get("door_leaf_present") is False:
        leaf = "absent"
    elif raw.get("door_leaf_present") is True:
        leaf = "present"
    if leaf in {"none", "missing", "no_leaf", "no_door", "open_aperture"}:
        leaf = "absent"
    elif leaf in {"leaf", "door", "present_leaf", "closed_leaf"}:
        leaf = "present"
    elif leaf not in {"absent", "present", "unknown"}:
        leaf = "unknown"

    connectivity = str(
        raw.get("connectivity")
        or raw.get("aperture_state")
        or raw.get("passage_state")
        or "unknown"
    ).strip().casefold()
    if raw.get("aperture_connected") is True or raw.get("traversable") is True:
        connectivity = "open"
    elif raw.get("aperture_connected") is False or raw.get("traversable") is False:
        connectivity = "blocked"
    if connectivity in {"connected", "traversable", "free", "open_aperture"}:
        connectivity = "open"
    elif connectivity in {"closed", "blocked", "occluded", "not_traversable", "unconnected"}:
        connectivity = "blocked"
    elif connectivity not in {"open", "blocked", "unknown"}:
        connectivity = "unknown"

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 1.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = {
        "door_leaf": leaf,
        "connectivity": connectivity,
        "confidence": confidence,
    }
    # ``static_open`` is deliberately a two-factor result.  A single visual
    # class or a missing simulator joint is never enough to mark a passage
    # traversable.
    if confidence >= 0.5 and leaf == "absent" and connectivity == "open":
        return "static_open", "static", evidence
    if confidence >= 0.5 and leaf == "present" and connectivity == "blocked":
        return "static_closed", "static", evidence
    if confidence >= 0.5 and connectivity == "blocked":
        return "blocked", "blocked", evidence
    return "unavailable", "unavailable", evidence


def _refrigerator_open_sweep_preflight(
    task_env,
    plan: dict[str, Any],
    *,
    sample_count: int = 6,
    recommended_retreat_m: float = 0.25,
) -> dict[str, Any]:
    """Privately check whether opening a refrigerator leaf hits the robot.

    This is intentionally a small kinematic MuJoCo sweep, not an additional
    planner.  It snapshots the simulator, interpolates only the selected
    articulation joints, checks the robot collision predicate, and restores the
    exact preflight state before returning.  The result is deliberately public
    and compact: no joint, body, or asset identifiers can leave this boundary.

    Missing collision support is a fail-open ``checked=False`` outcome.  It
    preserves compatibility with lightweight test environments and cannot turn
    a capability probe into a long-running interaction timeout.
    """

    model = getattr(task_env, "current_model", None)
    data = getattr(task_env, "current_data", None)
    collision_check = getattr(task_env, "check_robot_collision_in_current_pose", None)
    if (
        model is None
        or data is None
        or not callable(collision_check)
        or not hasattr(data, "qpos")
        or not hasattr(data, "qvel")
        or not hasattr(model, "jnt_qposadr")
        or not hasattr(model, "jnt_dofadr")
    ):
        return {"checked": False, "safe": True}

    targets = dict(plan.get("targets") or {})
    group = dict(plan.get("group") or {})
    joints_by_name = {
        str(item.get("joint_name") or ""): dict(item)
        for item in group.get("joints") or []
        if isinstance(item, dict) and str(item.get("joint_name") or "")
    }
    sweep_joints: list[tuple[int, int, float]] = []
    try:
        for joint_name, target in targets.items():
            joint = joints_by_name.get(str(joint_name)) or {}
            joint_id = joint.get("joint_id")
            if joint_id is None:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
                )
            joint_id = int(joint_id)
            if joint_id < 0:
                return {"checked": False, "safe": True}
            qpos_addr = int(model.jnt_qposadr[joint_id])
            dof_addr = int(model.jnt_dofadr[joint_id])
            sweep_joints.append((qpos_addr, dof_addr, float(target)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"checked": False, "safe": True}
    if not sweep_joints:
        return {"checked": False, "safe": True}

    snapshots: dict[str, np.ndarray] = {}
    for field in ("qpos", "qvel", "act", "ctrl", "xfrc_applied"):
        value = getattr(data, field, None)
        if isinstance(value, np.ndarray):
            snapshots[field] = value.copy()
    if "qpos" not in snapshots or "qvel" not in snapshots:
        return {"checked": False, "safe": True}
    try:
        baseline_collision = bool(collision_check())
    except Exception:
        return {"checked": False, "safe": True}
    start_values = [float(data.qpos[qpos_addr]) for qpos_addr, _, _ in sweep_joints]
    try:
        # Exclude 0.0: baseline is checked separately, and a true starting
        # collision must not be attributed to a refrigerator opening sweep.
        for fraction in np.linspace(1.0 / max(1, int(sample_count)), 1.0, max(1, int(sample_count))):
            for (qpos_addr, dof_addr, target), start in zip(sweep_joints, start_values):
                data.qpos[qpos_addr] = start + (target - start) * float(fraction)
                data.qvel[dof_addr] = 0.0
            mujoco.mj_forward(model, data)
            if not baseline_collision and bool(collision_check()):
                return {
                    "checked": True,
                    "safe": False,
                    "reason": "unsafe_open_sweep",
                    "recommended_retreat_m": max(0.05, float(recommended_retreat_m)),
                }
    except Exception:
        # A preflight is advisory safety instrumentation.  Do not reject a
        # command merely because an optional simulator collision API is absent
        # or a reduced test environment cannot forward dynamics.
        return {"checked": False, "safe": True}
    finally:
        for field, snapshot in snapshots.items():
            try:
                getattr(data, field)[...] = snapshot
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            mujoco.mj_forward(model, data)
        except Exception:
            pass
    return {"checked": True, "safe": True}


def _requires_refrigerator_open_sweep(command: dict[str, Any]) -> bool:
    """Use only the public type tag emitted by the semantic executor."""

    return bool(
        str(command.get("node_type") or "").casefold() == "container"
        and str(command.get("action") or "open").casefold() == "open"
        and str(command.get("container_kind") or "").casefold()
        in {"fridge", "refrigerator"}
    )


class AtomicForceInteractionController:
    def __init__(
        self,
        command_topic: str = "/semantic_decision/interaction_command",
        result_topic: str = "/semantic_mapping/interaction_result",
        feedback_topic: str = "/semantic_decision/interaction_action_feedback",
        output_path: str | Path | None = None,
        force_config: ForceDriveConfig | None = None,
        close_all_doors_on_prepare: bool = True,
        close_all_containers_on_prepare: bool = False,
        interaction_execution_mode: str | None = None,
        interaction_transition_steps: int | None = None,
        drawer_execution_mode: str | None = None,
        drawer_transition_steps: int | None = None,
        drawer_observation_steps: int = 1,
        drawer_view_restore_settle_steps: int = 0,
        drawer_view_restore_max_steps: int = 24,
        bypass_unsafe_open_sweep: bool = False,
        object_id_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.command_topic = str(command_topic)
        self.result_topic = str(result_topic)
        self.feedback_topic = str(feedback_topic)
        self.output_path = Path(output_path).expanduser().resolve() if output_path else None
        self.force_config = force_config or ForceDriveConfig()
        self.close_all_doors_on_prepare = bool(close_all_doors_on_prepare)
        self.close_all_containers_on_prepare = bool(close_all_containers_on_prepare)
        requested_mode = interaction_execution_mode or drawer_execution_mode or "fast"
        self.interaction_execution_mode = str(requested_mode).strip().lower()
        if self.interaction_execution_mode not in {"fast", "smooth"}:
            raise ValueError(
                f"Unsupported interaction execution mode: {requested_mode}"
            )
        requested_steps = interaction_transition_steps or drawer_transition_steps or 5
        self.interaction_transition_steps = max(1, int(requested_steps))
        self.drawer_execution_mode = self.interaction_execution_mode
        self.drawer_transition_steps = self.interaction_transition_steps
        self.drawer_observation_steps = max(1, int(drawer_observation_steps))
        # This is a visual settle only; it does not add a semantic state.
        self.drawer_view_restore_settle_steps = max(
            0, int(drawer_view_restore_settle_steps)
        )
        # A restore is complete only after measured head/torso convergence. The
        # bound prevents a stalled actuator from holding navigation forever.
        self.drawer_view_restore_max_steps = max(1, int(drawer_view_restore_max_steps))
        # Temporary experiment switch.  This bypasses only the predictive
        # refrigerator-door sweep; the real articulation force and its
        # postcondition verification still run normally.
        self.bypass_unsafe_open_sweep = bool(bypass_unsafe_open_sweep)
        # The public semantic graph may use an opaque portal ID.  Only this
        # simulator-side controller resolves it to a MuJoCo body name; results
        # deliberately retain the public ID so private names never flow back
        # into the graph or MLLM context.
        self._object_id_resolver = object_id_resolver
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._seen_command_ids: set[str] = set()
        self._event_index = 1
        self._result_publisher = None
        self._feedback_publisher = None
        self._subscriber = None
        self._String = None
        self._initial_door_states: list[dict[str, Any]] = []
        self._initial_articulation_states: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._completed_steps = 0
        self._pending: dict[str, Any] | None = None
        self._force_observation_requested = False
        self._restore_view_pending = False
        self._head_view_controller = HeadViewController()
        self._last_view_restore_result: dict[str, Any] | None = None
        self._pause_navigation = False

    def prepare(self, task) -> None:
        self._ensure_ros()
        self._commands = queue.Queue()
        self._seen_command_ids.clear()
        self._event_index = 1
        self._events = []
        self._completed_steps = 0
        self._pending = None
        self._force_observation_requested = False
        self._restore_view_pending = False
        self._head_view_controller.reset()
        self._last_view_restore_result = None
        self._pause_navigation = False
        if self.close_all_containers_on_prepare:
            self._initial_articulation_states = set_all_articulations_closed(
                task.env,
                include_doors=self.close_all_doors_on_prepare,
            )
            self._initial_door_states = [
                row for row in self._initial_articulation_states if row.get("is_door")
            ]
        else:
            self._initial_articulation_states = []
            self._initial_door_states = (
                set_all_door_roots_closed(task.env) if self.close_all_doors_on_prepare else []
            )
        self._write_snapshot()

    def enqueue_command(self, payload: dict[str, Any]) -> bool:
        command = dict(payload or {})
        command_id = str(command.get("command_id") or "")
        if command_id and command_id in self._seen_command_ids:
            return False
        action = str(command.get("action") or "open").lower()
        sequence_type = str(command.get("sequence_type") or "")
        if action != "open" and not (
            sequence_type == "drawer_scan" and action == "scan"
        ):
            raise ValueError(f"Atomic force interaction currently supports open only: {action}")
        object_id = str(command.get("object_id") or "")
        if not object_id:
            raise ValueError("Interaction command requires object_id")
        if not command_id:
            command_id = f"interaction_command_{self._event_index:06d}"
            command["command_id"] = command_id
        self._seen_command_ids.add(command_id)
        command["action"] = action
        command["object_id"] = object_id
        execution_object_id = object_id
        if self._object_id_resolver is not None:
            resolved = self._object_id_resolver(object_id)
            if resolved:
                execution_object_id = str(resolved)
        command["_execution_object_id"] = execution_object_id
        for planner_forbidden_key in (
            "source_object_name",
            "target_root_name",
            "root_body_name",
            "interaction_groups",
            "joint_names",
            "close_other_joint_names",
            "close_other_joints",
            "open_fraction_threshold",
        ):
            command.pop(planner_forbidden_key, None)
        self._commands.put(command)
        return True

    @staticmethod
    def _execution_object_id(command: dict[str, Any]) -> str:
        """Return the simulator-private ID for physical articulation calls."""

        return str(
            command.get("_execution_object_id")
            or command.get("object_id")
            or ""
        )

    @classmethod
    def _public_error_detail(cls, command: dict[str, Any], exc: Exception) -> str:
        """Redact a resolver-only simulator name from public ROS feedback."""

        detail = str(exc)
        execution_id = cls._execution_object_id(command)
        public_id = str(command.get("object_id") or "")
        if execution_id and public_id and execution_id != public_id:
            detail = detail.replace(execution_id, public_id)
        return detail

    def before_step(self, task, step: int) -> dict[str, Any] | None:
        if self._pending is not None:
            if self._pending.get("kind") == "drawer_sequence":
                try:
                    self._advance_drawer_sequence_before_step(task)
                except Exception as exc:
                    return self._abort_drawer_sequence(task, step, exc)
            elif self._pending.get("kind") == "articulation_sequence":
                self._advance_articulation_sequence_before_step(task, step)
            return None
        if self._restore_view_pending:
            self._restore_view_pending = False
            self._pause_navigation = True
            self._last_view_restore_result = self._head_view_controller.restore(task.env)
            if self._events:
                self._events[-1]["result"]["view_restore_result"] = (
                    self._last_view_restore_result
                )
                self._write_snapshot()
            self._force_observation_requested = True
            return None
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return None
        try:
            # A refrigerator may already be open when the interaction command
            # reaches the bridge (for example after a previous action or a
            # simulator reset).  Treat the authoritative articulation state as
            # a terminal success before validating the robot pose or applying
            # any force.  This also prevents repeated "open" commands from
            # trying to drive an already-open hinge/slide assembly.
            already_open = self._already_open_refrigerator(task, command)
            if already_open is not None:
                return self._publish_already_open(
                    task,
                    command,
                    step,
                    already_open,
                )
            command["interaction_pose_validation"] = self._validate_interaction_pose(
                task, command
            )
            if str(command.get("sequence_type") or "").casefold() in {
                "drawer_scan",
                "drawer_open",
            }:
                self._start_drawer_sequence(task, command, step)
                try:
                    self._advance_drawer_sequence_before_step(task)
                except Exception as exc:
                    return self._abort_drawer_sequence(task, step, exc)
                return None
            view_profile = str(command.get("view_profile") or "default")
            try:
                view_result = self._head_view_controller.command(
                    task.env,
                    view_profile,
                    tilt_rad=float(command.get("view_tilt_rad", 0.55) or 0.55),
                    torso_pitch_rad=command.get("view_torso_pitch_rad"),
                )
            except AttributeError:
                view_result = {
                    "profile": view_profile,
                    "applied": False,
                    "reason": "environment_view_profile_unavailable",
                }
            plan = prepare_articulation_force(
                task.env,
                self._execution_object_id(command),
            )
            if not bool(plan.get("supported", True)):
                return self._publish_unsupported_interaction(
                    command,
                    step=step,
                    reason=str(plan.get("reason") or "interaction_unsupported"),
                    interaction_capability=str(
                        plan.get("interaction_capability") or "unsupported"
                    ),
                    view_result=view_result,
                )
            if _requires_refrigerator_open_sweep(command):
                if self.bypass_unsafe_open_sweep:
                    command["_open_sweep_preflight"] = {
                        "checked": False,
                        "safe": True,
                        "bypassed": True,
                        "reason": "configured_bypass",
                    }
                    command["_open_sweep_preflight_bypassed"] = True
                else:
                    preflight = _refrigerator_open_sweep_preflight(task.env, plan)
                    command["_open_sweep_preflight"] = preflight
                if (
                    not self.bypass_unsafe_open_sweep
                    and bool(preflight.get("checked"))
                    and not bool(preflight.get("safe", True))
                ):
                    # No force has been applied.  Keep the simulator-only
                    # sweep private and publish only the public retry contract.
                    command["_open_sweep_preflight"] = preflight
                    self._commands.task_done()
                    return self._publish_command_failure(
                        task,
                        command,
                        step,
                        ValueError("unsafe_open_sweep"),
                        view_result=view_result,
                    )
            execution_mode = str(
                command.get("interaction_execution_mode")
                or self.interaction_execution_mode
            ).strip().lower()
            if execution_mode not in {"fast", "smooth"}:
                raise ValueError(f"Unsupported interaction execution mode: {execution_mode}")
            self._pending = {
                "command": command,
                "plan": plan,
                "view_result": view_result,
                "step": int(step),
                "phase": "pre_interaction_hold",
                "remaining_view_hold_steps": max(
                    0, int(command.get("view_hold_task_steps", 0) or 0)
                ),
                "remaining_post_interaction_hold_steps": max(
                    0, int(command.get("post_interaction_hold_task_steps", 0) or 0)
                ),
                "execution_mode": execution_mode,
                "transition_steps": max(
                    1,
                    int(
                        command.get(
                            "interaction_transition_steps",
                            self.interaction_transition_steps,
                        )
                    ),
                ),
            }
            if execution_mode == "smooth":
                self._pending.update(
                    {
                        "kind": "articulation_sequence",
                        "phase": (
                            "pre_interaction_hold"
                            if self._pending["remaining_view_hold_steps"] > 0
                            else "interaction_transition"
                        ),
                        "transition_step": 0,
                        "transition_start_values": {
                            str(info["joint_name"]): float(info["joint_value"])
                            for info in plan["pre_joint_infos"]
                            if str(info.get("joint_name") or "") in plan["targets"]
                        },
                        "transition_log": [],
                        "physics_substeps": 0,
                        "atomic_fallback": False,
                    }
                )
                self._advance_articulation_sequence_before_step(task, step)
            self._pause_navigation = True
            if view_profile == "drawer_low_view":
                self._force_observation_requested = True
            return None
        except (KeyError, ValueError) as exc:
            self._commands.task_done()
            return self._publish_command_failure(
                task,
                command,
                step,
                exc,
                view_result=locals().get("view_result"),
            )
        except Exception:
            self._commands.task_done()
            raise

    def _already_open_refrigerator(
        self, task, command: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return live joint evidence when a refrigerator is already open.

        This check intentionally runs only for a normal refrigerator ``open``
        command.  Drawer scans and other containers have their own state
        machines.  ``articulation_joint_infos`` is simulator-owned and reports
        door hinge leaves (or slide leaves for slide-door-only models) in the
        articulation; internal refrigerator trays are not required to be open.
        Every selected door leaf must satisfy the same open-fraction threshold
        used by the force backend before the command can be completed without
        action.
        """

        if str(command.get("node_type") or "").strip().casefold() != "container":
            return None
        if str(command.get("container_kind") or "").strip().casefold() not in {
            "fridge",
            "refrigerator",
        }:
            return None
        if str(command.get("action") or "open").strip().casefold() != "open":
            return None
        if str(command.get("sequence_type") or "").strip().casefold() in {
            "drawer_scan",
            "drawer_open",
        }:
            return None
        object_id = self._execution_object_id(command)
        if not object_id:
            return None
        try:
            infos = articulation_joint_infos(task.env, object_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        articulated = [
            dict(info)
            for info in infos
            if str(info.get("joint_type") or "").strip().casefold()
            in {"hinge", "slide"}
        ]
        if not articulated:
            return None
        # Refrigerator roots often include internal slide-out drawers in
        # addition to the door hinge(s).  A visually/physically open appliance
        # is determined by its door leaf: prefer hinge joints when present and
        # use slide travel only for slide-door-only models.  Requiring every
        # internal tray to be open would make an already-open fridge issue a
        # redundant force command.
        hinge_infos = [
            info
            for info in articulated
            if str(info.get("joint_type") or "").strip().casefold() == "hinge"
        ]
        relevant = hinge_infos or [
            info
            for info in articulated
            if str(info.get("joint_type") or "").strip().casefold() == "slide"
        ]
        threshold = float(self.force_config.open_fraction_threshold)
        if not all(float(info.get("open_fraction", 0.0)) >= threshold for info in relevant):
            return None
        return {
            "checked": True,
            "state": "open",
            "threshold": threshold,
            # Keep the public result free of simulator joint/body names while
            # retaining enough evidence to audit the all-door-leaf check.
            "door_leaf_count": len(relevant),
            "door_leaf_joint_types": [
                str(info.get("joint_type") or "").casefold() for info in relevant
            ],
            "door_leaf_open_fractions": [
                float(info.get("open_fraction", 0.0)) for info in relevant
            ],
            "source": "simulator_articulation_open_fraction",
        }

    def _publish_already_open(
        self,
        task,
        command: dict[str, Any],
        step: int,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish a terminal no-op success for an already-open refrigerator."""

        event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
        self._event_index += 1
        stamp_sec = time.time()
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": str(command.get("node_id") or ""),
            "object_id": str(command.get("object_id") or ""),
            "node_type": "container",
            "container_kind": str(command.get("container_kind") or "refrigerator"),
            "action": "open",
            "interaction_mode": str(command.get("interaction_mode") or "open_close"),
            "interaction_capability": "articulated",
            "interactable": True,
            "retryable": False,
            "already_open": True,
            "action_executed": False,
            "observation_outcome": "finish_without_action",
            "state": "open",
            "pre_state": "open",
            "post_state": "open",
            "success": True,
            "status": "SUCCEEDED",
            "confidence": 1.0,
            "execution_cost": 0.0,
            "sim_steps_consumed": 0,
            "physics_substeps": 0,
            "task_steps_consumed": 0,
            "result_published_step": int(step),
            "source": "executor_already_open_precondition",
            "verification_source": str(evidence.get("source") or "simulator_articulation_state"),
            "already_open_evidence": evidence,
            "interaction_pose_validation": {
                "checked": False,
                "valid": True,
                "reason": "already_open_no_pose_required",
            },
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        feedback = {
            "command_id": result["command_id"],
            "candidate_id": result["candidate_id"],
            "decision_id": result["decision_id"],
            "event_id": event_id,
            "behavior_type": "INTERACT",
            "status": "SUCCEEDED",
            "success": True,
            "interaction_result": result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._pending = None
        self._restore_view_pending = False
        self._pause_navigation = False
        self._force_observation_requested = True
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        self._commands.task_done()
        return result

    def _publish_unsupported_interaction(
        self,
        command: dict[str, Any],
        *,
        step: int,
        reason: str,
        interaction_capability: str,
        view_result: dict[str, Any],
    ) -> dict[str, Any]:
        event_id = str(
            command.get("event_id") or f"interaction_{self._event_index:06d}"
        )
        self._event_index += 1
        stamp_sec = time.time()
        node_type = str(command.get("node_type") or "").casefold()
        capability = str(interaction_capability).casefold()
        aperture_state, aperture_capability, aperture_evidence = _portal_aperture_feedback(
            command
        )
        # ``static`` was the legacy spelling returned by the runtime whenever
        # articulation resolution failed.  It is not a passage observation.
        # For portals, downgrade it to unavailable unless the public aperture
        # observation proves a fixed opening (or a fixed/blocked closure).
        if node_type == "portal" and capability in {
            "static",
            "unavailable",
            "unsupported",
        }:
            resolved_capability = aperture_capability
            semantic_state = aperture_state
        else:
            resolved_capability = capability or "unavailable"
            semantic_state = (
                "static"
                if resolved_capability == "static"
                else "blocked"
                if resolved_capability in {"blocked", "unsupported", "locked"}
                else "unavailable"
            )
        static_open_portal = (
            node_type == "portal"
            and semantic_state == "static_open"
            and resolved_capability == "static"
        )
        reason_detail = ""
        if not static_open_portal:
            reason_detail = self._public_error_detail(command, ValueError(str(reason)))
        if semantic_state == "static_closed":
            reason_detail = "non_articulated_closed_portal"
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": str(command.get("node_id") or ""),
            "object_id": str(command["object_id"]),
            "node_type": node_type,
            "action": str(command.get("action") or "open"),
            "interaction_mode": "none",
            "interaction_group_id": str(
                command.get("interaction_group_id") or "all"
            ),
            "interaction_capability": resolved_capability,
            "interactable": False,
            "state": semantic_state,
            "pre_state": "unknown",
            "post_state": semantic_state,
            "success": static_open_portal,
            "status": "SUCCEEDED" if static_open_portal else "FAILED",
            "reason": reason_detail,
            "retryable": False,
            "confidence": float((aperture_evidence or {}).get("confidence", 1.0)),
            "execution_cost": 0.0,
            "sim_steps_consumed": 0,
            "physics_substeps": 0,
            "task_steps_consumed": 0,
            "result_published_step": int(step),
            "source": (
                "executor_static_portal"
                if static_open_portal
                else "force_interaction_capability_check"
            ),
            "verification_source": (
                "observed_aperture_geometry"
                if aperture_evidence is not None
                else "mujoco_articulation_registry"
            ),
            "view_profile": str(command.get("view_profile") or "default"),
            "view_profile_result": view_result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        if aperture_evidence is not None:
            result["portal_aperture_observation"] = aperture_evidence
        feedback = {
            "command_id": result["command_id"],
            "candidate_id": result["candidate_id"],
            "decision_id": result["decision_id"],
            "event_id": event_id,
            "behavior_type": "INTERACT",
            "status": result["status"],
            "success": result["success"],
            "reason": result["reason"],
            "interaction_result": result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._force_observation_requested = True
        self._pause_navigation = False
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        self._commands.task_done()
        return result

    def after_step(self, task, step: int) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None:
            return None
        if pending.get("kind") == "drawer_sequence":
            try:
                return self._advance_drawer_sequence_after_step(task, step)
            except Exception as exc:
                return self._abort_drawer_sequence(task, step, exc)
        if pending.get("phase") == "pre_interaction_hold" and int(
            pending.get("remaining_view_hold_steps", 0)
        ) > 0:
            pending["remaining_view_hold_steps"] -= 1
            self._force_observation_requested = True
            if (
                pending.get("kind") == "articulation_sequence"
                and int(pending["remaining_view_hold_steps"]) <= 0
            ):
                pending["phase"] = "interaction_transition"
            return None
        command = pending["command"]
        if (
            pending.get("kind") == "articulation_sequence"
            and pending.get("phase") == "interaction_transition"
        ):
            if int(pending["transition_step"]) < int(pending["transition_steps"]):
                return None
            force_result = finalize_articulation_force_transition(
                task.env,
                pending["plan"],
                physics_substeps=int(pending["physics_substeps"]),
                task_steps_consumed=int(pending["transition_steps"]),
                atomic_fallback=bool(pending["atomic_fallback"]),
                transition_log=list(pending["transition_log"]),
                config=self.force_config,
            )
            pending["force_result"] = force_result
            pending["phase"] = "post_interaction_hold"
            self._force_observation_requested = True
            if int(pending.get("remaining_post_interaction_hold_steps", 0)) > 0:
                return None
        elif pending.get("phase") == "pre_interaction_hold":
            force_result = complete_articulation_force(
                task.env,
                pending["plan"],
                config=self.force_config,
            )
            pending["force_result"] = force_result
            pending["force_applied_step"] = int(step)
            pending["phase"] = "post_interaction_hold"
            self._force_observation_requested = True
            if int(pending.get("remaining_post_interaction_hold_steps", 0)) > 0:
                return None
        if pending.get("phase") == "post_interaction_hold" and int(
            pending.get("remaining_post_interaction_hold_steps", 0)
        ) > 0:
            pending["remaining_post_interaction_hold_steps"] -= 1
            self._force_observation_requested = True
            if int(pending["remaining_post_interaction_hold_steps"]) > 0:
                return None
        self._pending = None
        try:
            force_result = pending["force_result"]
            event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
            self._event_index += 1
            stamp_sec = time.time()
            terminal_blocked = not bool(force_result["success"])
            semantic_state = "blocked" if terminal_blocked else str(force_result["post_state"])
            result = {
                "event_id": event_id,
                "command_id": str(command["command_id"]),
                "candidate_id": str(command.get("candidate_id") or ""),
                "decision_id": str(command.get("decision_id") or ""),
                "node_id": str(command.get("node_id") or ""),
                "object_id": str(command["object_id"]),
                "action": "open",
                "interaction_mode": str(command.get("interaction_mode") or "open_close"),
                "operation_method": str(command.get("operation_method") or "unknown"),
                "open_regions": list(command.get("open_regions") or []),
                "approach_goal_xyyaw": list(
                    command.get("approach_goal_xyyaw") or []
                ),
                "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
                "view_profile": str(command.get("view_profile") or "default"),
                "view_torso_pitch_rad": command.get("view_torso_pitch_rad"),
                "view_hold_task_steps": int(
                    command.get("view_hold_task_steps", 0) or 0
                ),
                "post_interaction_hold_task_steps": int(
                    command.get("post_interaction_hold_task_steps", 0) or 0
                ),
                "view_profile_result": pending["view_result"],
                "view_restore_result": self._last_view_restore_result,
                "method": "xfrc_applied_group_pd",
                "state": semantic_state,
                "pre_state": str(force_result["pre_state"]),
                "post_state": semantic_state,
                "success": bool(force_result["success"]),
                "status": "SUCCEEDED" if force_result["success"] else "FAILED",
                "interaction_capability": (
                    "blocked" if terminal_blocked else "articulated"
                ),
                "interactable": not terminal_blocked,
                "failure_reason": (
                    "force_target_not_reached" if terminal_blocked else ""
                ),
                "retryable": not terminal_blocked,
                "confidence": 1.0,
                "execution_cost": 1.0,
                "sim_steps_consumed": int(step) - int(pending["step"]) + 1,
                "physics_substeps": int(force_result["physics_substeps"]),
                "task_steps_consumed": int(force_result["task_steps_consumed"]),
                "force_applied_step": int(
                    pending.get("force_applied_step", pending["step"])
                ),
                "result_published_step": int(step),
                "interaction_execution_mode": str(
                    pending.get("execution_mode") or "fast"
                ),
                "interaction_transition_steps": int(
                    pending.get("transition_steps") or 1
                ),
                "source": (
                    "force_smooth_interaction"
                    if pending.get("execution_mode") == "smooth"
                    else "force_atomic_interaction"
                ),
                "verification_source": "executor_state_verification",
                "open_sweep_preflight_bypassed": bool(
                    command.get("_open_sweep_preflight_bypassed", False)
                ),
                "interaction_pose_validation": dict(
                    command.get("interaction_pose_validation") or {}
                ),
                "step": int(pending["step"]),
                "stamp_sec": stamp_sec,
            }
            feedback = {
                "command_id": result["command_id"],
                "candidate_id": result["candidate_id"],
                "decision_id": result["decision_id"],
                "event_id": event_id,
                "behavior_type": "INTERACT",
                "status": result["status"],
                "success": result["success"],
                "interaction_result": result,
                "step": int(pending["step"]),
                "stamp_sec": stamp_sec,
            }
            self._events.append({"result": result, "feedback": feedback})
            self._force_observation_requested = True
            self._restore_view_pending = bool(
                command.get("restore_view_after", False)
            )
            if self._restore_view_pending:
                self._last_view_restore_result = None
            self._pause_navigation = self._restore_view_pending
            self._publish(self._result_publisher, result)
            self._publish(self._feedback_publisher, feedback)
            self._write_snapshot()
            return result
        finally:
            self._commands.task_done()

    def _advance_articulation_sequence_before_step(self, task, step: int) -> None:
        pending = self._pending
        if (
            pending is None
            or pending.get("kind") != "articulation_sequence"
            or pending.get("phase") != "interaction_transition"
        ):
            return
        transition_steps = int(pending["transition_steps"])
        if int(pending["transition_step"]) >= transition_steps:
            return
        next_step = int(pending["transition_step"]) + 1
        transition = advance_articulation_force(
            task.env,
            pending["plan"],
            progress=float(next_step) / float(transition_steps),
            start_values=pending["transition_start_values"],
            transition_steps=transition_steps,
            config=self.force_config,
        )
        pending["transition_step"] = next_step
        pending["physics_substeps"] += int(transition.get("physics_substeps", 0))
        pending["atomic_fallback"] = bool(
            pending["atomic_fallback"] or transition.get("fallback", False)
        )
        pending["transition_log"].append(
            {
                "task_step": int(step),
                "task_step_index": next_step,
                "progress": float(transition.get("progress", next_step / transition_steps)),
                "fallback": bool(transition.get("fallback", False)),
                "physics_substeps": int(transition.get("physics_substeps", 0)),
            }
        )
        pending.setdefault("force_applied_step", int(step))
        self._force_observation_requested = True

    def _start_drawer_sequence(self, task, command: dict[str, Any], step: int) -> None:
        articulation_groups = collect_articulation_groups(task.env)
        articulation = articulation_groups.get(self._execution_object_id(command))
        if articulation is None:
            raise ValueError(
                "drawer interaction target is not an articulated simulator object: "
                f"{command['object_id']}"
            )
        body_heights = {}
        model = task.env.current_model
        data = task.env.current_data
        for joint in articulation.get("joints") or []:
            joint_name = str(joint.get("joint_name") or "")
            if not joint_name:
                continue
            body_id = None
            body_name = str(joint.get("body_name") or "")
            if body_name:
                try:
                    body_id = int(model.body(body_name).id)
                except (KeyError, ValueError):
                    body_id = None
            if body_id is None and joint.get("joint_id") is not None:
                body_id = int(model.jnt_bodyid[int(joint["joint_id"])])
            if body_id is not None:
                body_heights[joint_name] = float(data.xpos[body_id][2])
        sequence_type = str(command.get("sequence_type") or "drawer_scan").casefold()
        if sequence_type not in {"drawer_scan", "drawer_open"}:
            raise ValueError(f"Unsupported drawer interaction sequence: {sequence_type}")
        # A scan is a complete container audit: enumerate every physical slide
        # joint even when M1 returns an incomplete visible-region list.  A normal
        # drawer_open remains limited to the explicitly grounded visual regions.
        scan_all_drawers = sequence_type == "drawer_scan"
        groups = ground_drawer_open_regions(
            list(articulation.get("joints") or []),
            [] if scan_all_drawers else list(command.get("open_regions") or []),
            body_heights,
            fallback_to_all=scan_all_drawers,
        )
        if not groups:
            raise ValueError("drawer interaction requires at least one valid visible drawer region")
        drawer_joint_names = list(
            dict.fromkeys(
                str(joint.get("joint_name") or "")
                for joint in articulation.get("joints") or []
                if str(joint.get("joint_type") or "").casefold() == "slide"
                and str(joint.get("joint_name") or "")
            )
        )
        if not drawer_joint_names:
            raise ValueError("drawer interaction target has no slide joints")
        selected_joint_names = [
            name for group in groups for name in group["joint_names"]
        ]
        preserve_open = sequence_type == "drawer_open"
        transition_steps = max(
            1,
            int(command.get("drawer_transition_steps", self.drawer_transition_steps)),
        )
        observation_steps = max(
            1,
            int(command.get("drawer_observation_steps", self.drawer_observation_steps)),
        )
        restore_settle_steps = max(
            0,
            int(
                command.get(
                    "drawer_view_restore_settle_steps",
                    self.drawer_view_restore_settle_steps,
                )
                or 0
            ),
        )
        self._pending = {
            "kind": "drawer_sequence",
            "command": command,
            "step": int(step),
            "phase": "open",
            "phase_step": 0,
            "group_index": 0,
            "groups": groups,
            # ``drawer_scan`` is the sealed benchmark macro and deliberately
            # restores every selected drawer to closed.  ``drawer_open`` is
            # normal interactive exploration: selected, M1-grounded drawers
            # remain open so subsequent RGB/map updates can expose contents.
            "preserve_open": preserve_open,
            # ``all_joint_names`` remains the M1-grounded set used by the
            # exploration-open postcondition. A sealed scan additionally
            # closes and verifies every physical slide joint on the container.
            "all_joint_names": selected_joint_names,
            "drawer_joint_names": drawer_joint_names,
            "transition_steps": transition_steps,
            "observation_steps": observation_steps,
            "remaining_observation_steps": 0,
            "phase_plan": None,
            "phase_start_values": {},
            "group_results": [],
            "transition_log": [],
            "physics_substeps": 0,
            "view_result": None,
            "view_restore_result": None,
            "view_restore_settle_steps": restore_settle_steps,
            "expected_task_steps": drawer_sequence_task_step_budget(
                len(groups),
                transition_steps,
                observation_steps,
                restore_settle_steps,
                preserve_open=preserve_open,
            ),
            "remaining_view_restore_settle_steps": 0,
            "view_restore_max_steps": max(
                self.drawer_view_restore_max_steps, restore_settle_steps
            ),
            "view_restore_steps_elapsed": 0,
            "view_restore_convergence": None,
            "view_restore_required": False,
            "view_restore_failed": False,
            # Filled after the low view is applied, then reasserted around
            # every internal force substep for the whole drawer macro.
            "robot_lock_snapshot": None,
        }
        self._pause_navigation = True

    def _advance_drawer_sequence_before_step(self, task) -> None:
        pending = self._pending
        if pending is None or pending.get("kind") != "drawer_sequence":
            return
        phase = str(pending["phase"])
        if phase == "restore_settle":
            # The restore command has already set home targets. Reserve these
            # task steps for a visible camera settle and convergence check.
            self._force_observation_requested = True
            return
        if phase == "observe":
            # The drawer is physically compliant.  Keep the already-open
            # selected front at its target during the low-view dwell rather
            # than releasing it and sampling after it has sprung back.
            self._hold_drawer_observation(task)
            return
        groups = pending["groups"]
        group_index = int(pending["group_index"])
        current_group = groups[group_index]
        mode = self.interaction_execution_mode
        transition_steps = int(pending["transition_steps"]) if mode == "smooth" else 1
        if pending.get("phase_plan") is None:
            if phase == "open":
                close_joint_names = (
                    [
                        name
                        for name in pending["drawer_joint_names"]
                        if name not in current_group["joint_names"]
                    ]
                    if not bool(pending.get("preserve_open"))
                    else []
                )
                pending["phase_plan"] = prepare_articulation_state_force(
                    task.env,
                    self._execution_object_id(pending["command"]),
                    open_joint_names=current_group["joint_names"],
                    # A sealed scan never has more than one physical drawer
                    # open: close all sibling slide joints in the same force
                    # transition, including unseen/pre-opened drawers.
                    close_joint_names=close_joint_names,
                )
            else:
                pending["phase_plan"] = prepare_articulation_state_force(
                    task.env,
                    self._execution_object_id(pending["command"]),
                    # Close only for the explicit benchmark scan macro.  The
                    # normal exploration ``drawer_open`` path never enters a
                    # close phase, because the semantic postcondition must
                    # remain physically accessible after execution.
                    close_joint_names=current_group["joint_names"],
                )
            pending["phase_start_values"] = {
                str(info["joint_name"]): float(info["joint_value"])
                for info in pending["phase_plan"]["pre_joint_infos"]
                if str(info.get("joint_name") or "")
                in pending["phase_plan"]["targets"]
            }
        next_step = int(pending["phase_step"]) + 1
        progress = min(1.0, float(next_step) / float(transition_steps))
        command = pending["command"]
        if phase == "open" and group_index == 0 and pending["view_result"] is None:
            # Apply the full low view once before any drawer force is applied.
            # The camera then holds one stable pose for the whole macro rather
            # than ramping with the smooth drawer transition progress.
            view_result = self._head_view_controller.command(
                task.env,
                "drawer_low_view",
                tilt_rad=float(command.get("view_tilt_rad", 0.30) or 0.30),
                torso_pitch_rad=float(command.get("view_torso_pitch_rad", 0.35) or 0.35),
            )
            pending["view_result"] = view_result
        if pending.get("robot_lock_snapshot") is None:
            pending["robot_lock_snapshot"] = _capture_robot_lock(task.env)
        robot_lock = pending.get("robot_lock_snapshot")
        if robot_lock is not None:
            _apply_robot_lock(task.env, robot_lock)
        transition = advance_articulation_force(
            task.env,
            pending["phase_plan"],
            progress=progress,
            start_values=pending["phase_start_values"],
            transition_steps=transition_steps,
            config=self.force_config,
            robot_lock_callback=(
                None if robot_lock is None else lambda: _apply_robot_lock(task.env, robot_lock)
            ),
        )
        if robot_lock is not None:
            _apply_robot_lock(task.env, robot_lock)
        pending["phase_step"] = next_step
        pending["physics_substeps"] += int(transition.get("physics_substeps", 0))
        pending["transition_log"].append(
            {
                "phase": phase,
                "group_id": current_group["group_id"],
                "task_step_index": next_step,
                "progress": progress,
                "fallback": bool(transition.get("fallback", False)),
                "physics_substeps": int(transition.get("physics_substeps", 0)),
            }
        )

    def _advance_drawer_sequence_after_step(
        self, task, step: int
    ) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None:
            return None
        phase = str(pending["phase"])
        mode = self.interaction_execution_mode
        transition_steps = int(pending["transition_steps"]) if mode == "smooth" else 1
        if phase == "restore_settle":
            pending["remaining_view_restore_settle_steps"] = max(
                0, int(pending.get("remaining_view_restore_settle_steps", 0)) - 1
            )
            pending["view_restore_steps_elapsed"] = int(
                pending.get("view_restore_steps_elapsed", 0)
            ) + 1
            self._force_observation_requested = True
            try:
                convergence = self._head_view_controller.restore_convergence(task.env)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                convergence = {
                    "checked": False,
                    "converged": False,
                    "reason": f"restore_convergence_check_failed:{type(exc).__name__}",
                }
            pending["view_restore_convergence"] = convergence
            if bool(convergence.get("converged")) and int(
                pending["remaining_view_restore_settle_steps"]
            ) <= 0:
                return self._finish_drawer_sequence(task, step)
            if int(pending["view_restore_steps_elapsed"]) >= int(
                pending.get("view_restore_max_steps", self.drawer_view_restore_max_steps)
            ):
                pending["view_restore_failed"] = True
                return self._finish_drawer_sequence(task, step)
            # Keep stepping even after the fixed visual-settle budget expires;
            # physical convergence, not elapsed time, authorizes completion.
            if int(pending["remaining_view_restore_settle_steps"]) <= 0:
                pending["remaining_view_restore_settle_steps"] = 1
            return None
        if phase in {"open", "close"} and int(pending["phase_step"]) < transition_steps:
            return None
        if phase == "open":
            # Fast mode shortens only each force transition; it must not skip
            # the observation or merge closing this drawer with opening the
            # next one.  This keeps the physical sequence identical to the
            # evaluator-owned V3 drawer scan.
            pending["phase"] = "observe"
            pending["phase_step"] = 0
            # Keep the completed opening plan as the holding plan for the
            # observation dwell.  ``_hold_drawer_observation`` reasserts it
            # before every simulated observation step.
            pending["remaining_observation_steps"] = int(pending["observation_steps"])
            return None
        if phase == "observe":
            self._force_observation_requested = True
            pending["remaining_observation_steps"] -= 1
            if int(pending["remaining_observation_steps"]) > 0:
                return None
            self._record_drawer_observation(task, step)
            if bool(pending.get("preserve_open")):
                if int(pending["group_index"]) + 1 < len(pending["groups"]):
                    pending["group_index"] += 1
                    pending["phase"] = "open"
                    pending["phase_step"] = 0
                    pending["phase_plan"] = None
                    return None
                # The final selected drawer remains open.  Release the body
                # lock before restoring the camera; the result below verifies
                # that every selected public region is still open.
                pending["robot_lock_snapshot"] = None
                pending["view_restore_result"] = self._head_view_controller.restore(task.env)
                pending["view_restore_required"] = True
                pending["phase"] = "restore_settle"
                pending["phase_step"] = 0
                pending["remaining_view_restore_settle_steps"] = 1
                pending["view_restore_steps_elapsed"] = 0
                self._force_observation_requested = True
                return None
            pending["phase"] = "close"
            pending["phase_step"] = 0
            pending["phase_plan"] = None
            return None
        if phase == "close":
            if int(pending["group_index"]) + 1 < len(pending["groups"]):
                pending["group_index"] += 1
                pending["phase"] = "open"
                pending["phase_step"] = 0
                pending["phase_plan"] = None
                return None
            # Release the lock only after the final drawer has fully closed;
            # restoring the head/torso is a separate final action.
            pending["robot_lock_snapshot"] = None
            pending["view_restore_result"] = self._head_view_controller.restore(task.env)
            pending["view_restore_required"] = True
            settle_steps = int(pending.get("view_restore_settle_steps", 0) or 0)
            pending["phase"] = "restore_settle"
            pending["remaining_view_restore_settle_steps"] = max(1, settle_steps)
            pending["view_restore_steps_elapsed"] = 0
            self._force_observation_requested = True
            return None
        return None

    def _hold_drawer_observation(
        self, task, *, phase_label: str = "observe_hold"
    ) -> None:
        """Reassert the current drawer's open target during its view dwell."""

        pending = self._pending
        if pending is None or pending.get("kind") != "drawer_sequence":
            return
        plan = pending.get("phase_plan")
        if not isinstance(plan, dict):
            raise RuntimeError("drawer observation hold is missing its opening plan")
        robot_lock = pending.get("robot_lock_snapshot")
        if robot_lock is not None:
            _apply_robot_lock(task.env, robot_lock)
        transition_steps = (
            int(pending["transition_steps"])
            if self.interaction_execution_mode == "smooth"
            else 1
        )
        transition = advance_articulation_force(
            task.env,
            plan,
            progress=1.0,
            start_values=dict(pending.get("phase_start_values") or {}),
            transition_steps=transition_steps,
            config=self.force_config,
            robot_lock_callback=(
                None if robot_lock is None else lambda: _apply_robot_lock(task.env, robot_lock)
            ),
        )
        if robot_lock is not None:
            _apply_robot_lock(task.env, robot_lock)
        pending["physics_substeps"] += int(transition.get("physics_substeps", 0))
        observation_index = int(pending.get("observation_steps", 0)) - int(
            pending.get("remaining_observation_steps", 0)
        ) + 1
        pending["transition_log"].append(
            {
                "phase": str(phase_label),
                "group_id": pending["groups"][int(pending["group_index"])]["group_id"],
                "task_step_index": max(1, observation_index),
                "progress": 1.0,
                "fallback": bool(transition.get("fallback", False)),
                "physics_substeps": int(transition.get("physics_substeps", 0)),
            }
        )

    def _abort_drawer_sequence(
        self,
        task,
        step: int,
        exc: Exception,
    ) -> dict[str, Any]:
        """Best-effort close and restore if any drawer macro phase fails."""

        pending = self._pending
        if pending is None or pending.get("kind") != "drawer_sequence":
            raise exc
        command = dict(pending["command"])
        recovery_detail = "not_attempted"
        try:
            recovery_plan = prepare_articulation_state_force(
                task.env,
                self._execution_object_id(command),
                close_joint_names=list(
                    pending.get("drawer_joint_names")
                    or pending.get("all_joint_names")
                    or []
                ),
            )
            robot_lock = pending.get("robot_lock_snapshot") or _capture_robot_lock(task.env)
            if robot_lock is not None:
                _apply_robot_lock(task.env, robot_lock)
            recovery = complete_articulation_force(
                task.env,
                recovery_plan,
                config=self.force_config,
                robot_lock_callback=(
                    None if robot_lock is None else lambda: _apply_robot_lock(task.env, robot_lock)
                ),
            )
            recovery_detail = "succeeded" if bool(recovery.get("success")) else "not_confirmed"
        except Exception as recovery_exc:
            recovery_detail = f"failed:{type(recovery_exc).__name__}"
        finally:
            pending["robot_lock_snapshot"] = None
            self._commands.task_done()
        sequence_type = str(command.get("sequence_type") or "drawer_scan").casefold()
        failure = RuntimeError(
            f"{sequence_type}_execution_failed:{type(exc).__name__}; recovery={recovery_detail}"
        )
        return self._publish_command_failure(
            task,
            command,
            step,
            failure,
            view_result=pending.get("view_result"),
        )

    def _record_drawer_observation(self, task, step: int) -> None:
        pending = self._pending
        # The simulator advances once between the pre-step hold and this
        # callback. Reassert the open target immediately before sampling so a
        # compliant slide cannot spring closed during the camera capture.
        self._hold_drawer_observation(task, phase_label="observe_capture_hold")
        group = pending["groups"][int(pending["group_index"])]
        joint_infos = articulation_joint_infos(
            task.env, self._execution_object_id(pending["command"])
        )
        selected_infos = [
            info for info in joint_infos if info["joint_name"] in group["joint_names"]
        ]
        selected_open = bool(selected_infos) and all(
            float(info.get("open_fraction", 0.0))
            >= float(self.force_config.open_fraction_threshold)
            for info in selected_infos
        )
        if bool(pending.get("preserve_open")):
            # ``drawer_open`` deliberately leaves previously selected drawers
            # open for exploration, so sibling closure is a scan-only invariant.
            other_drawer_joint_names = []
            other_drawers_closed = True
        else:
            other_drawer_joint_names = [
                name
                for name in pending.get("drawer_joint_names") or []
                if name not in group["joint_names"]
            ]
            other_infos_by_name = {
                str(info.get("joint_name") or ""): info
                for info in joint_infos
                if str(info.get("joint_name") or "") in other_drawer_joint_names
            }
            other_drawers_closed = len(other_infos_by_name) == len(other_drawer_joint_names) and all(
                float(other_infos_by_name[name].get("open_fraction", 1.0))
                <= 1.0 - float(self.force_config.open_fraction_threshold)
                for name in other_drawer_joint_names
            )
        success = selected_open and other_drawers_closed
        pending["group_results"].append(
            {
                "region_id": group["group_id"],
                "open_region": group.get("open_region"),
                "grounding_source": group.get(
                    "grounding_source", "simulator_articulation_mapping"
                ),
                "success": success,
                "selected_drawer_open": selected_open,
                "other_drawer_count": len(other_drawer_joint_names),
                "other_drawers_closed": other_drawers_closed,
                "observed_open_fractions": {
                    str(info["joint_name"]): float(info.get("open_fraction", 0.0))
                    for info in selected_infos
                },
                "observation_step": int(step),
            }
        )

    def _finish_drawer_sequence(self, task, step: int) -> dict[str, Any]:
        pending = self._pending
        command = pending["command"]
        final_joint_infos = articulation_joint_infos(
            task.env, self._execution_object_id(command)
        )
        success = bool(pending["group_results"]) and all(
            bool(group.get("success")) for group in pending["group_results"]
        )
        if bool(pending.get("view_restore_required")):
            success = success and not bool(pending.get("view_restore_failed")) and bool(
                (pending.get("view_restore_convergence") or {}).get("converged")
            )
        sequence_type = str(command.get("sequence_type") or "drawer_scan").casefold()
        preserve_open = bool(pending.get("preserve_open"))
        final_state_settle: dict[str, Any] = {"attempted": False}
        final_state_joint_names = list(
            pending["all_joint_names"]
            if preserve_open
            else pending.get("drawer_joint_names") or pending["all_joint_names"]
        )
        final_infos_by_name = {
            str(info.get("joint_name") or ""): info
            for info in final_joint_infos
            if str(info.get("joint_name") or "") in final_state_joint_names
        }
        if preserve_open:
            final_state_success = len(final_infos_by_name) == len(final_state_joint_names) and all(
                float(final_infos_by_name[name].get("open_fraction", 0.0))
                >= float(self.force_config.open_fraction_threshold)
                for name in final_state_joint_names
            )
        else:
            final_state_success = len(final_infos_by_name) == len(final_state_joint_names) and all(
                float(final_infos_by_name[name].get("open_fraction", 1.0))
                <= 1.0 - float(self.force_config.open_fraction_threshold)
                for name in final_state_joint_names
            )
            if not final_state_success:
                # The final task step is intentionally a passive camera/nav
                # hold.  If a compliant drawer moves during it, give the
                # already-requested close one bounded physical settle before
                # reporting failure.  This is not a second interaction and
                # keeps the scan's closed postcondition truthful.
                final_state_settle = {"attempted": True}
                try:
                    settle_plan = prepare_articulation_state_force(
                        task.env,
                        self._execution_object_id(command),
                        close_joint_names=final_state_joint_names,
                    )
                    settle = complete_articulation_force(
                        task.env,
                        settle_plan,
                        config=self.force_config,
                    )
                    pending["physics_substeps"] += int(
                        settle.get("physics_substeps", 0)
                    )
                    final_state_settle["success"] = bool(settle.get("success"))
                    final_state_settle["physics_substeps"] = int(
                        settle.get("physics_substeps", 0)
                    )
                    final_joint_infos = articulation_joint_infos(
                        task.env, self._execution_object_id(command)
                    )
                    final_infos_by_name = {
                        str(info.get("joint_name") or ""): info
                        for info in final_joint_infos
                        if str(info.get("joint_name") or "") in final_state_joint_names
                    }
                    final_state_success = len(final_infos_by_name) == len(
                        final_state_joint_names
                    ) and all(
                        float(final_infos_by_name[name].get("open_fraction", 1.0))
                        <= 1.0 - float(self.force_config.open_fraction_threshold)
                        for name in final_state_joint_names
                    )
                except Exception as exc:
                    final_state_settle["error"] = type(exc).__name__
        success = success and final_state_success
        event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
        self._event_index += 1
        stamp_sec = time.time()
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": str(command.get("node_id") or ""),
            "object_id": str(command["object_id"]),
            "action": str(command.get("action") or ("open" if preserve_open else "scan")),
            "interaction_mode": str(
                command.get("interaction_mode") or sequence_type
            ),
            "sequence_type": sequence_type,
            "operation_method": str(command.get("operation_method") or "pull"),
            "open_regions": list(command.get("open_regions") or []),
            "approach_goal_xyyaw": list(command.get("approach_goal_xyyaw") or []),
            "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
            "grounded_regions": [
                {
                    "region_id": str(group.get("group_id") or ""),
                    "open_region": group.get("open_region"),
                    "grounding_source": str(
                        group.get("grounding_source")
                        or "simulator_articulation_mapping"
                    ),
                }
                for group in pending["groups"]
            ],
            "region_results": list(pending["group_results"]),
            "final_joint_open_fractions": {
                str(info["joint_name"]): float(info.get("open_fraction", 0.0))
                for info in final_joint_infos
                if info.get("joint_name") in final_state_joint_names
            },
            "final_close_success": final_state_success if not preserve_open else None,
            "final_open_success": final_state_success if preserve_open else None,
            "final_state_settle": final_state_settle,
            "view_profile": "drawer_low_view",
            "view_profile_result": pending["view_result"],
            "view_restore_result": pending["view_restore_result"],
            "view_restore_convergence": pending.get("view_restore_convergence"),
            "view_restore_max_steps": int(
                pending.get("view_restore_max_steps", self.drawer_view_restore_max_steps)
            ),
            "view_restore_steps_elapsed": int(
                pending.get("view_restore_steps_elapsed", 0)
            ),
            "interaction_execution_mode": self.interaction_execution_mode,
            "interaction_transition_steps": int(pending["transition_steps"]),
            "drawer_execution_mode": self.interaction_execution_mode,
            "drawer_transition_steps": int(pending["transition_steps"]),
            "drawer_observation_steps": int(pending["observation_steps"]),
            "drawer_view_restore_settle_steps": int(
                pending.get("view_restore_settle_steps", 0) or 0
            ),
            "expected_task_steps": int(pending.get("expected_task_steps", 0) or 0),
            "transition_log": list(pending["transition_log"]),
            "state": "open" if preserve_open else "closed",
            "pre_state": "closed",
            "post_state": "open" if preserve_open else "closed",
            "success": success,
            "status": "SUCCEEDED" if success else "FAILED",
            "failure_reason": (
                "drawer_view_restore_timeout"
                if bool(pending.get("view_restore_failed"))
                else None
            ),
            "confidence": 1.0,
            "execution_cost": 1.0,
            "sim_steps_consumed": int(step) - int(pending["step"]) + 1,
            "physics_substeps": int(pending["physics_substeps"]),
            "task_steps_consumed": int(step) - int(pending["step"]) + 1,
            "result_published_step": int(step),
            "source": "force_container_sequence",
            "verification_source": "executor_state_verification",
            "step": int(pending["step"]),
            "stamp_sec": stamp_sec,
        }
        feedback = {
            "command_id": result["command_id"],
            "candidate_id": result["candidate_id"],
            "decision_id": result["decision_id"],
            "event_id": event_id,
            "behavior_type": "INTERACT",
            "status": result["status"],
            "success": result["success"],
            "interaction_result": result,
            "step": int(pending["step"]),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._pending = None
        self._pause_navigation = False
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        self._commands.task_done()
        return result

    def consume_force_observation_request(self) -> bool:
        requested = self._force_observation_requested
        self._force_observation_requested = False
        return requested

    def should_pause_navigation(self) -> bool:
        return self._pending is not None or self._pause_navigation

    def view_torso_target(self) -> list[float] | None:
        return self._head_view_controller.torso_target()

    def after_task_step(self) -> None:
        if self._pending is None and not self._restore_view_pending:
            self._pause_navigation = False

    @staticmethod
    def _validate_interaction_pose(task, command: dict[str, Any]) -> dict[str, Any]:
        expected = list(command.get("interaction_approach_pose_xyyaw") or [])
        if len(expected) < 3:
            return {"checked": False, "reason": "no_expected_approach_pose"}
        base_pose = task.env.current_robot.robot_view.base.pose
        actual = [
            float(base_pose[0, 3]),
            float(base_pose[1, 3]),
            math.atan2(float(base_pose[1, 0]), float(base_pose[0, 0])),
        ]
        position_error_m = math.hypot(
            actual[0] - float(expected[0]), actual[1] - float(expected[1])
        )
        yaw_error_rad = abs(
            math.atan2(
                math.sin(actual[2] - float(expected[2])),
                math.cos(actual[2] - float(expected[2])),
            )
        )
        # The executor carries an explicit per-goal contract because
        # MoveBaseGoal itself has no tolerance fields.  Prefer that contract
        # here so the final physical precondition cannot silently use the
        # broader legacy interaction-ready envelope after navigation used a
        # stricter AABB/front constraint.
        distance_tolerance_m = max(
            0.05,
            float(
                command.get(
                    "navigation_goal_position_tolerance_m",
                    command.get("interaction_ready_distance_m", 0.45),
                )
                or 0.45
            ),
        )
        yaw_tolerance_rad = max(
            0.05,
            float(
                command.get(
                    "navigation_goal_yaw_tolerance_rad",
                    command.get("interaction_ready_yaw_tolerance_rad", 0.55),
                )
                or 0.55
            ),
        )
        comparison_epsilon = 1e-3
        pose_valid = (
            position_error_m <= distance_tolerance_m + comparison_epsilon
            and yaw_error_rad <= yaw_tolerance_rad + comparison_epsilon
        )
        face_required = bool(
            command.get("interaction_front_axis_validation_required", False)
        )
        axis_values = list(command.get("interaction_approach_axis_xy") or [])
        center_values = list(command.get("interaction_target_center_xy") or [])
        face_checked = False
        face_valid = not face_required
        face_position_error_rad: float | None = None
        face_yaw_error_rad: float | None = None
        face_position_tolerance_rad = max(
            0.05,
            float(command.get("interaction_front_position_tolerance_rad", 0.55) or 0.55),
        )
        face_yaw_tolerance_rad = max(
            0.05,
            float(command.get("interaction_front_yaw_tolerance_rad", 0.55) or 0.55),
        )
        normalized_axis: list[float] = []
        if len(axis_values) >= 2 and len(center_values) >= 2:
            try:
                axis_x = float(axis_values[0])
                axis_y = float(axis_values[1])
                center_x = float(center_values[0])
                center_y = float(center_values[1])
            except (TypeError, ValueError):
                axis_x = axis_y = center_x = center_y = float("nan")
            axis_norm = math.hypot(axis_x, axis_y)
            offset_x = actual[0] - center_x
            offset_y = actual[1] - center_y
            offset_norm = math.hypot(offset_x, offset_y)
            if (
                math.isfinite(axis_norm)
                and math.isfinite(offset_norm)
                and axis_norm > 1e-6
                and offset_norm > 1e-6
            ):
                axis_x /= axis_norm
                axis_y /= axis_norm
                normalized_axis = [axis_x, axis_y]
                dot = max(
                    -1.0,
                    min(1.0, (offset_x * axis_x + offset_y * axis_y) / offset_norm),
                )
                face_position_error_rad = math.acos(dot)
                expected_front_yaw = math.atan2(-axis_y, -axis_x)
                face_yaw_error_rad = abs(
                    math.atan2(
                        math.sin(actual[2] - expected_front_yaw),
                        math.cos(actual[2] - expected_front_yaw),
                    )
                )
                face_checked = True
                face_valid = (
                    face_position_error_rad <= face_position_tolerance_rad
                    and face_yaw_error_rad <= face_yaw_tolerance_rad
                )
        # Both containers and portals require an independent geometric front
        # check.  A portal's arrival yaw alone is not a front-side guarantee.
        # A portal's AABB face check above is the authoritative physical front
        # contract.  Articulation geometry is often unavailable for static or
        # hinge-less door assets, so requiring an inferred hinge/slide axis for
        # portals would reject a valid AABB-front arrival for the wrong reason.
        # Containers still require the independent articulation-front check.
        physical_front_required = bool(
            face_required
            and str(command.get("node_type") or "").strip().casefold()
            == "container"
        )
        physical_front_checked = False
        physical_front_valid = not physical_front_required
        physical_front_position_error_rad: float | None = None
        physical_front_yaw_error_rad: float | None = None
        physical_front_source = ""
        if physical_front_required:
            physical_front = infer_articulation_front_axis_xy(
                task.env, str(command.get("_execution_object_id") or command.get("object_id") or "")
            )
            physical_front_checked = bool(physical_front.get("checked"))
            physical_front_source = str(
                physical_front.get("source") or physical_front.get("reason") or ""
            )
            physical_axis = list(physical_front.get("axis_xy") or [])
            # Do not substitute the M1-selected axis when simulator geometry is
            # unavailable.  M1 is an image classifier, not an independent
            # physical-front oracle; accepting its axis here would allow a
            # side/rear view to authorize an action.  Fail closed and let the
            # executor advance to the next bounded anchor instead.
            if physical_front_checked and len(physical_axis) >= 2 and len(center_values) >= 2:
                offset_x = actual[0] - float(center_values[0])
                offset_y = actual[1] - float(center_values[1])
                offset_norm = math.hypot(offset_x, offset_y)
                if offset_norm > 1e-6:
                    physical_axis_x = float(physical_axis[0])
                    physical_axis_y = float(physical_axis[1])
                    dot = max(
                        -1.0,
                        min(
                            1.0,
                            (offset_x * physical_axis_x + offset_y * physical_axis_y)
                            / offset_norm,
                        ),
                    )
                    physical_front_position_error_rad = math.acos(dot)
                    physical_front_yaw = math.atan2(
                        -physical_axis_y, -physical_axis_x
                    )
                    physical_front_yaw_error_rad = abs(
                        math.atan2(
                            math.sin(actual[2] - physical_front_yaw),
                            math.cos(actual[2] - physical_front_yaw),
                        )
                    )
                    physical_front_valid = bool(
                        physical_front_position_error_rad
                        <= face_position_tolerance_rad
                        and physical_front_yaw_error_rad <= face_yaw_tolerance_rad
                    )
        valid = pose_valid and face_valid and physical_front_valid
        result = {
            "checked": True,
            "valid": valid,
            "expected_pose_xyyaw": [float(value) for value in expected[:3]],
            "actual_pose_xyyaw": actual,
            "position_error_m": position_error_m,
            "yaw_error_rad": yaw_error_rad,
            "distance_tolerance_m": distance_tolerance_m,
            "yaw_tolerance_rad": yaw_tolerance_rad,
            "comparison_epsilon": comparison_epsilon,
            "approach_axis_xy": normalized_axis or axis_values,
            "face_validation_required": face_required,
            "face_checked": face_checked,
            "face_valid": face_valid,
            "target_center_xy": center_values,
            "front_axis_source": str(
                command.get("interaction_front_axis_source") or ""
            ),
            "face_position_error_rad": face_position_error_rad,
            "face_yaw_error_rad": face_yaw_error_rad,
            "face_position_tolerance_rad": face_position_tolerance_rad,
            "face_yaw_tolerance_rad": face_yaw_tolerance_rad,
            "physical_front_required": physical_front_required,
            "physical_front_checked": physical_front_checked,
            "physical_front_valid": physical_front_valid,
            "physical_front_source": physical_front_source,
            "physical_front_position_error_rad": physical_front_position_error_rad,
            "physical_front_yaw_error_rad": physical_front_yaw_error_rad,
        }
        if not valid:
            command["interaction_pose_validation"] = result
            prefix = (
                "Interaction physical front invalid: "
                if physical_front_required and not physical_front_valid
                else "Interaction pose invalid: "
            )
            raise ValueError(
                prefix + f"position_error_m={position_error_m:.3f} "
                f"yaw_error_rad={yaw_error_rad:.3f} "
                f"face_checked={face_checked} "
                f"face_position_error_rad={face_position_error_rad} "
                f"face_yaw_error_rad={face_yaw_error_rad}"
            )
        return result

    def _publish_command_failure(
        self,
        task,
        command: dict[str, Any],
        step: int,
        exc: Exception,
        view_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
        self._event_index += 1
        stamp_sec = time.time()
        object_id = str(command.get("object_id") or "")
        node_id = str(command.get("node_id") or "")
        node_type = str(command.get("node_type") or "").casefold()
        missing_articulation = str(exc).startswith("Articulated object not found:")
        invalid_interaction_pose = str(exc).startswith("Interaction pose invalid:")
        invalid_physical_front = str(exc).startswith(
            "Interaction physical front invalid:"
        )
        unsafe_open_sweep = str(exc).strip() == "unsafe_open_sweep"
        drawer_sequence_type = str(command.get("sequence_type") or "").casefold()
        drawer_sequence_execution_failed = str(exc).startswith(
            "drawer_scan_execution_failed:"
        ) or str(exc).startswith("drawer_open_execution_failed:")
        portal_missing_articulation = missing_articulation and node_type == "portal"
        aperture_state, aperture_capability, aperture_evidence = _portal_aperture_feedback(
            command
        )
        if portal_missing_articulation:
            semantic_state = aperture_state
            resolved_capability = aperture_capability
        elif unsafe_open_sweep:
            # Articulation lookup succeeded; only the private simulated sweep
            # rejected this robot stance.  Do not leak the selected leaf/joint.
            semantic_state = "unknown"
            resolved_capability = "articulated"
        else:
            semantic_state = "unknown"
            resolved_capability = "unknown"
        static_open_portal = (
            portal_missing_articulation
            and semantic_state == "static_open"
            and resolved_capability == "static"
        )
        try:
            view_restore_result = self._head_view_controller.restore(task.env)
        except (AttributeError, KeyError, ValueError):
            view_restore_result = {
                "applied": False,
                "reason": "environment_view_restore_unavailable",
            }
        if static_open_portal:
            verification_source = (
                "observed_aperture_geometry"
                if aperture_evidence is not None
                else "simulator_no_articulation"
            )
            failure_reason = ""
        elif portal_missing_articulation:
            verification_source = (
                "observed_aperture_geometry"
                if aperture_evidence is not None
                else "simulator_no_articulation"
            )
            failure_reason = (
                "non_articulated_closed_portal"
                if semantic_state == "static_closed"
                else "non_articulated"
                if semantic_state == "unavailable"
                else "non_articulated_blocked_portal"
                if semantic_state == "blocked"
                else "non_articulated"
            )
        elif unsafe_open_sweep:
            verification_source = "executor_open_sweep_preflight"
            failure_reason = "unsafe_open_sweep"
        elif drawer_sequence_execution_failed:
            verification_source = "executor_drawer_sequence_failure"
            failure_reason = f"{drawer_sequence_type or 'drawer'}_execution_failed"
        elif invalid_physical_front:
            verification_source = "executor_pose_precondition"
            failure_reason = "interaction_wrong_face"
            resolved_capability = "articulated"
        elif invalid_interaction_pose:
            verification_source = "executor_pose_precondition"
            failure_reason = "interaction_pose_invalid"
        else:
            verification_source = "executor_resolution_failure"
            failure_reason = "articulation_resolution_failed"
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": node_id,
            "object_id": object_id,
            "node_type": node_type,
            "action": str(command.get("action") or "open"),
            "interaction_mode": str(command.get("interaction_mode") or "open_close"),
            "operation_method": str(command.get("operation_method") or "unknown"),
            "open_regions": list(command.get("open_regions") or []),
            "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
            "view_profile": str(command.get("view_profile") or "default"),
            "view_profile_result": view_result,
            "view_restore_result": view_restore_result,
            "interaction_capability": resolved_capability,
            "interactable": (
                False
                if portal_missing_articulation
                else True
                if unsafe_open_sweep or invalid_physical_front
                else None
            ),
            "retryable": (
                False
                if portal_missing_articulation
                else True
                if unsafe_open_sweep or invalid_physical_front
                else None
            ),
            "state": semantic_state,
            "pre_state": "unknown",
            "post_state": semantic_state,
            "success": static_open_portal,
            "status": "SUCCEEDED" if static_open_portal else "FAILED",
            "confidence": float((aperture_evidence or {}).get("confidence", 1.0)),
            "execution_cost": 0.0 if static_open_portal or unsafe_open_sweep else 1.0,
            "sim_steps_consumed": 0,
            "physics_substeps": 0,
            "task_steps_consumed": 0,
            "result_published_step": int(step),
            "interaction_execution_mode": self.interaction_execution_mode,
            "interaction_transition_steps": 0,
            "source": (
                "executor_static_portal"
                if static_open_portal
                else "force_interaction_open_sweep_preflight"
                if unsafe_open_sweep
                else "force_interaction_rejected"
            ),
            "verification_source": verification_source,
            "failure_reason": failure_reason,
            "interaction_pose_validation": dict(
                command.get("interaction_pose_validation") or {}
            ),
            "error_type": type(exc).__name__,
            "error": (
                failure_reason
                if portal_missing_articulation or unsafe_open_sweep
                else self._public_error_detail(command, exc)
            ),
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        if unsafe_open_sweep:
            result["recommended_retreat_m"] = float(
                (command.get("_open_sweep_preflight") or {}).get(
                    "recommended_retreat_m", 0.25
                )
                or 0.25
            )
        if aperture_evidence is not None:
            result["portal_aperture_observation"] = aperture_evidence
        feedback = {
            "command_id": result["command_id"],
            "candidate_id": result["candidate_id"],
            "decision_id": result["decision_id"],
            "event_id": event_id,
            "behavior_type": "INTERACT",
            "status": result["status"],
            "success": result["success"],
            "interaction_result": result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._pending = None
        self._restore_view_pending = False
        self._pause_navigation = False
        self._force_observation_requested = True
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        return result

    def finalize(self, completed_steps: int) -> None:
        self._completed_steps = int(completed_steps)
        self._write_snapshot()

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None

    def _command_callback(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("interaction command must be a JSON object")
            self.enqueue_command(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                import rospy

                rospy.logwarn("Atomic force interaction command ignored: %s", exc)
            except Exception:
                pass

    def _ensure_ros(self) -> None:
        if self._subscriber is not None:
            return
        import rospy
        from std_msgs.msg import String

        self._String = String
        self._result_publisher = rospy.Publisher(
            self.result_topic,
            String,
            queue_size=4,
            latch=True,
        )
        self._feedback_publisher = rospy.Publisher(
            self.feedback_topic,
            String,
            queue_size=4,
            latch=True,
        )
        self._subscriber = rospy.Subscriber(
            self.command_topic,
            String,
            self._command_callback,
            queue_size=8,
        )

    def _publish(self, publisher, payload: dict[str, Any]) -> None:
        if publisher is None or self._String is None:
            return
        publisher.publish(
            self._String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _write_snapshot(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "command_topic": self.command_topic,
                    "result_topic": self.result_topic,
                    "feedback_topic": self.feedback_topic,
                    "force_config": self.force_config.__dict__,
                    "initial_door_states": self._initial_door_states,
                    "initial_articulation_states": self._initial_articulation_states,
                    "events": self._events,
                    "completed_steps": self._completed_steps,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
