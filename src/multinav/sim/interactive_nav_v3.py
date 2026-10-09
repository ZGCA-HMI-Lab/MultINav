from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec


SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "dataset_definition/v3/interactive_nav_episode.schema.json"
)


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def build_interaction_id(container_name: str, joint_index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", container_name).strip("-").lower()
    return f"container::{slug}::{int(joint_index)}"


def build_container_target(
    *,
    object_record: dict[str, Any],
    category: str,
    container: dict[str, Any],
    matching_instance_count: int,
) -> dict[str, Any]:
    def json_value(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value

    object_name = str(object_record["name"])
    if matching_instance_count < 1:
        raise ValueError("Target grounding must match at least one scene instance")
    container_category = container.get("category")
    if not container_category:
        raise ValueError(f"Missing category for container {container.get('name')!r}")
    return {
        "selection_mode": "specific_instance",
        "category": category,
        "selected_instance": object_name,
        "instruction_consistent_candidates": [object_name],
        "container_name": container["name"],
        "container_category": str(container_category),
        "grounding": {
            "unique": matching_instance_count == 1,
            "matching_instance_count": int(matching_instance_count),
            "description": category,
            "attributes": {},
        },
        "object_aabb_center": json_value(object_record.get("aabb_center")),
        "object_aabb_size": json_value(object_record.get("aabb_size")),
        "container_aabb_center": json_value(container.get("aabb_center")),
        "container_aabb_size": json_value(container.get("aabb_size")),
    }


def build_nav_to_obj_success_criteria(succ_pos_threshold: float) -> dict[str, Any]:
    return {
        "type": "nav_to_obj",
        "target_selection": "specific_instance",
        "distance": {
            "metric": "planar_robot_base_to_object",
            "threshold_m": float(succ_pos_threshold),
            "comparison": "strictly_less",
        },
        "visibility": {
            "camera_name": "head_camera",
            "metric": "visibility_fraction",
            "threshold": 0.0,
            "comparison": "strictly_greater",
        },
        "combination": "all",
    }


def build_point_target(
    *,
    goal_point: list[float],
    goal_yaw: float | None,
    sampling_source: str,
    clearance_m: float,
    grounding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(goal_point) != 3:
        raise ValueError("PointGoal goal_point must contain exactly three values")
    if clearance_m < 0.0:
        raise ValueError("PointGoal clearance_m must be non-negative")
    return {
        "target_type": "point",
        "goal_point": [float(value) for value in goal_point],
        "goal_yaw": None if goal_yaw is None else float(goal_yaw),
        "sampling_source": str(sampling_source),
        "clearance_m": float(clearance_m),
        "grounding": dict(grounding or {"attributes": {}}),
    }


def build_nav_to_point_success_criteria(
    succ_pos_threshold: float,
    *,
    require_goal_yaw: bool = False,
    yaw_threshold_rad: float | None = None,
) -> dict[str, Any]:
    if succ_pos_threshold <= 0.0:
        raise ValueError("PointGoal success threshold must be positive")
    if require_goal_yaw and (yaw_threshold_rad is None or yaw_threshold_rad <= 0.0):
        raise ValueError("A positive yaw threshold is required when goal yaw is enforced")
    return {
        "type": "nav_to_point",
        "distance": {
            "metric": "planar_robot_base_to_point",
            "threshold_m": float(succ_pos_threshold),
            "comparison": "less_or_equal",
        },
        "require_goal_yaw": bool(require_goal_yaw),
        "yaw_threshold_rad": (
            float(yaw_threshold_rad) if require_goal_yaw else None
        ),
    }


def _joint_type(joint: dict[str, Any]) -> str:
    value = str(joint.get("mujoco_joint_type", joint.get("joint_type", ""))).lower()
    normalized = value.strip().strip("[]")
    if "hinge" in value or normalized == "3":
        return "container_hinged_door"
    if "slide" in value or normalized == "2":
        return "container_sliding_drawer"
    raise ValueError(f"Unsupported container joint type for v3: {value!r}")


def _initial_fraction_by_joint(
    articulation_states: list[dict[str, Any]],
) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for state in articulation_states:
        fraction = state.get("open_fraction")
        if fraction is None:
            raise ValueError(
                f"Missing open_fraction for initial joint {state.get('joint_name')!r}"
            )
        fractions[str(state["joint_name"])] = float(fraction)
    return fractions


def build_container_interactions(
    *,
    container: dict[str, Any],
    oracle_candidates: list[dict[str, Any]],
    articulation_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    container_category = container.get("category")
    if not container_category:
        raise ValueError(f"Missing category for container {container.get('name')!r}")
    joints_by_index = {int(joint["joint_index"]): joint for joint in container["joints"]}
    initial_fractions = _initial_fraction_by_joint(articulation_states)
    interaction_ids: dict[int, str] = {}
    effects: dict[int, set[str]] = {}
    prerequisites: dict[int, set[int]] = {}
    order: list[int] = []

    for candidate in oracle_candidates:
        sequence = [int(index) for index in candidate["joint_sequence"]]
        for sequence_index, joint_index in enumerate(sequence):
            if joint_index not in order:
                order.append(joint_index)
            interaction_ids.setdefault(
                joint_index, build_interaction_id(container["name"], joint_index)
            )
            effects.setdefault(joint_index, set()).add(
                "reveal_target_object"
                if sequence_index == len(sequence) - 1
                else "enable_interaction"
            )
            prerequisites.setdefault(joint_index, set()).update(sequence[:sequence_index])

    interactions = []
    for joint_index in order:
        joint = joints_by_index[joint_index]
        joint_name = str(joint["joint_name"])
        if joint_name not in initial_fractions:
            raise ValueError(f"No articulation initial state for joint {joint_name!r}")
        initial_fraction = initial_fractions[joint_name]
        if abs(initial_fraction) > 1e-6:
            raise ValueError(
                f"Container joint {joint_name!r} is not initially closed: {initial_fraction}"
            )
        interactions.append(
            {
                "interaction_id": interaction_ids[joint_index],
                "type": _joint_type(joint),
                "object_name": container["name"],
                "object_category": str(container_category),
                "joint_name": joint_name,
                "joint_index": joint_index,
                "effect_types": sorted(effects[joint_index]),
                "prerequisites": [
                    {
                        "interaction_id": interaction_ids[required_index],
                        "type": "mechanical",
                    }
                    for required_index in sorted(
                        prerequisites[joint_index], key=order.index
                    )
                ],
                "initial_state": {
                    "joint_fraction": initial_fraction,
                    "semantic_state": "closed",
                },
                "target_state": {
                    "joint_fraction": 1.0,
                    "semantic_state": "open",
                },
            }
        )
    return interactions, interaction_ids


def build_initial_state(
    interactions: list[dict[str, Any]],
    *,
    all_doors_open: bool,
    container_joints_closed: bool,
    target_visible: bool,
) -> dict[str, Any]:
    return {
        "interaction_states": [
            {
                "interaction_id": interaction["interaction_id"],
                **interaction["initial_state"],
            }
            for interaction in interactions
        ],
        "all_doors_open": bool(all_doors_open),
        "container_joints_closed": bool(container_joints_closed),
        "target_visible": bool(target_visible),
    }


def build_oracle_prefixes(
    *,
    plan: dict[str, Any],
    visibility_trace: list[dict[str, Any]],
    distance_passed: bool,
    reachable: bool,
) -> list[dict[str, Any]]:
    open_steps = [step for step in plan["steps"] if step["type"] == "open_joint"]
    first_open_step = next(
        (index for index, step in enumerate(plan["steps"]) if step["type"] == "open_joint"),
        len(plan["steps"]),
    )
    prefixes = []
    for trace_index, row in enumerate(visibility_trace):
        completed_open_steps = min(trace_index, len(open_steps))
        visibility = float(row.get("visibility_fraction", 0.0))
        pixels = int(row.get("visible_pixels", 0))
        prefixes.append(
            {
                "plan_id": plan["plan_id"],
                "completed_step_count": first_open_step + completed_open_steps,
                "robot_reachable_to_next_goal": bool(reachable),
                "target_distance_passed": bool(distance_passed),
                "target_visibility_fraction": visibility,
                "target_visible_pixels": pixels,
                "task_success": bool(distance_passed and visibility > 0.0),
                "opened_interaction_ids": [
                    step["interaction_id"]
                    for step in open_steps[:completed_open_steps]
                ],
            }
        )
    return prefixes


def _interaction_domain(interaction: dict[str, Any]) -> str:
    interaction_type = str(interaction.get("type", ""))
    if interaction_type.startswith("channel_"):
        return "channel"
    if interaction_type.startswith("container_"):
        return "container"
    raise ValueError(f"Unsupported interaction type: {interaction_type!r}")


def _validate_common_v3_episode(
    episode: dict[str, Any],
    *,
    expected_domains: list[str] | None = None,
) -> dict[str, Any]:
    payload = episode.get("interactive_nav", {})
    if payload.get("schema_version") != "interactive_nav_v3":
        raise ValueError("Unified validator only accepts interactive_nav_v3")
    domains = payload.get("interaction_domains")
    if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
        raise ValueError("interaction_domains must be a non-empty unique list")
    if expected_domains is not None and domains != expected_domains:
        raise ValueError(f"Expected interaction_domains={expected_domains}, got {domains}")

    task = episode.get("task", {})
    target = payload.get("target", {})
    task_type = task.get("task_type")
    is_point_goal = task_type == "nav_to_point"
    selected_instance = None if is_point_goal else target.get("selected_instance")
    candidate_instances: list[str] = []
    if is_point_goal:
        if target.get("target_type") != "point":
            raise ValueError("nav_to_point tasks require a point target")
        task_goal = task.get("goal_point")
        target_goal = target.get("goal_point")
        if task_goal != target_goal:
            raise ValueError("task.goal_point does not match target.goal_point")
        if len(task_goal or []) != 3:
            raise ValueError("PointGoal must contain a three-value goal_point")
        if bool(task.get("require_goal_yaw")) and task.get("goal_yaw") is None:
            raise ValueError("PointGoal requires goal_yaw when require_goal_yaw is true")
    else:
        if task_type != "nav_to_obj":
            raise ValueError(f"Unsupported V3 task_type: {task_type!r}")
        selection_mode = task.get("selection_mode")
        if selection_mode not in {"specific_instance", "any_candidate"}:
            raise ValueError(f"Unsupported object-goal selection_mode: {selection_mode!r}")
        if target.get("selection_mode") != selection_mode:
            raise ValueError("target.selection_mode does not mirror task.selection_mode")
        if payload.get("success_criteria", {}).get("target_selection") != selection_mode:
            raise ValueError("success_criteria.target_selection does not mirror task.selection_mode")
        raw_candidates = task.get("pickup_obj_candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("object-goal task requires non-empty pickup_obj_candidates")
        candidate_instances = [str(value) for value in raw_candidates]
        if any(not value for value in candidate_instances) or len(set(candidate_instances)) != len(
            candidate_instances
        ):
            raise ValueError("pickup_obj_candidates must contain unique non-empty names")
        target_candidates = [
            str(value) for value in target.get("instruction_consistent_candidates", [])
        ]
        if target_candidates != candidate_instances:
            raise ValueError(
                "target.instruction_consistent_candidates must mirror task.pickup_obj_candidates"
            )
        pickup_obj_name = task.get("pickup_obj_name")
        if pickup_obj_name not in candidate_instances:
            raise ValueError("task.pickup_obj_name must be one of pickup_obj_candidates")
        if selection_mode == "specific_instance":
            if pickup_obj_name != selected_instance:
                raise ValueError("task.pickup_obj_name does not match target.selected_instance")
            if candidate_instances != [selected_instance]:
                raise ValueError(
                    "specific_instance candidates must contain only the selected target"
                )
        elif selected_instance is not None and selected_instance not in candidate_instances:
            raise ValueError(
                "any_candidate target.selected_instance must be null or a runtime candidate"
            )

    if "container" in domains and not is_point_goal:
        grounding = target.get("grounding", {})
        matching_count = grounding.get("matching_instance_count")
        if not isinstance(matching_count, int) or matching_count < 1:
            raise ValueError("container target grounding count must be a measured positive integer")
        if grounding.get("unique") != (matching_count == 1):
            raise ValueError("target grounding unique flag disagrees with the measured count")
        if not target.get("container_name") or target.get("container_category") in {
            None,
            "",
            "unknown",
        }:
            raise ValueError("container target grounding must name a measured container")

    relevant_objects = episode.get("task_relevant_objects", [])
    required_objects = [] if is_point_goal else list(candidate_instances)
    if "container" in domains and not is_point_goal:
        required_objects.append(target.get("container_name"))
    for required_object in required_objects:
        if required_object not in relevant_objects:
            raise ValueError(f"task_relevant_objects is missing {required_object!r}")

    interactions = payload.get("interactions", [])
    requirement = payload.get("interaction_requirement")
    if requirement in {"required", "beneficial"} and not interactions:
        raise ValueError(
            f"{requirement.capitalize()} v3 episodes must contain interactions"
        )
    if requirement == "unnecessary" and interactions:
        raise ValueError("Unnecessary-interaction episodes must not contain interactions")
    interaction_ids = [interaction.get("interaction_id") for interaction in interactions]
    if None in interaction_ids or len(set(interaction_ids)) != len(interaction_ids):
        raise ValueError("Interaction IDs must be present and unique")
    interaction_by_id = {
        interaction["interaction_id"]: interaction for interaction in interactions
    }
    for interaction in interactions:
        if _interaction_domain(interaction) not in domains:
            raise ValueError("Interaction type is not declared in interaction_domains")
        if interaction.get("object_category") in {None, "", "unknown"}:
            raise ValueError("Interaction object_category must be measured, not a placeholder")
        for prerequisite in interaction.get("prerequisites", []):
            if prerequisite.get("interaction_id") not in interaction_by_id:
                raise ValueError("Interaction prerequisite references an unknown interaction")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(interaction_id: str) -> None:
        if interaction_id in visiting:
            raise ValueError("Interaction prerequisites contain a cycle")
        if interaction_id in visited:
            return
        visiting.add(interaction_id)
        for prerequisite in interaction_by_id[interaction_id].get("prerequisites", []):
            visit(prerequisite["interaction_id"])
        visiting.remove(interaction_id)
        visited.add(interaction_id)

    for interaction_id in interaction_by_id:
        visit(interaction_id)

    initial_rows = payload.get("initial_state", {}).get("interaction_states", [])
    initial_by_id = {state.get("interaction_id"): state for state in initial_rows}
    if set(initial_by_id) != set(interaction_by_id):
        raise ValueError("initial_state interaction IDs do not exactly match interactions")
    articulation_by_joint = {
        state["joint_name"]: state
        for state in episode.get("scene_modifications", {}).get("articulation_states", [])
    }
    for interaction in interactions:
        interaction_id = interaction["interaction_id"]
        if initial_by_id.get(interaction_id) != {
            "interaction_id": interaction_id,
            **interaction["initial_state"],
        }:
            raise ValueError("interactive_nav initial state does not mirror interactions")
        articulation = articulation_by_joint.get(interaction["joint_name"])
        if articulation is None:
            raise ValueError("Interaction joint is missing from scene articulation state")
        if abs(
            float(articulation.get("open_fraction", -1.0))
            - float(interaction["initial_state"]["joint_fraction"])
        ) > 1e-6:
            raise ValueError("Interaction and scene articulation initial fractions disagree")

    success_criteria = payload.get("success_criteria", {})
    task_threshold = float(task.get("succ_pos_threshold", -1.0))
    if float(success_criteria.get("distance", {}).get("threshold_m", -2.0)) != task_threshold:
        raise ValueError("success_criteria distance threshold does not mirror the task")
    expected_success_type = "nav_to_point" if is_point_goal else "nav_to_obj"
    if success_criteria.get("type") != expected_success_type:
        raise ValueError("success_criteria type does not mirror the task")
    if is_point_goal and bool(success_criteria.get("require_goal_yaw")) != bool(
        task.get("require_goal_yaw")
    ):
        raise ValueError("PointGoal yaw requirement does not mirror the task")

    plans = payload.get("oracle_plans", [])
    if not plans or plans[0] != payload.get("oracle_plan"):
        raise ValueError("oracle_plan must equal oracle_plans[0]")
    for plan in plans:
        required_ids = plan.get("required_interaction_ids", [])
        if any(interaction_id not in interaction_by_id for interaction_id in required_ids):
            raise ValueError(f"Plan {plan.get('plan_id')} references an unknown interaction")
        step_ids = [
            step["interaction_id"]
            for step in plan.get("steps", [])
            if step.get("type") == "open_joint"
        ]
        if step_ids != required_ids:
            raise ValueError(
                f"Plan {plan.get('plan_id')} required_interaction_ids do not match steps"
            )
        completed: list[str] = []
        for step in plan.get("steps", []):
            interaction_id = step.get("interaction_id")
            if interaction_id is not None and interaction_id not in interaction_by_id:
                raise ValueError("Oracle step references an unknown interaction")
            if step.get("type") != "open_joint":
                continue
            interaction = interaction_by_id[interaction_id]
            if (
                step.get("object_name") != interaction["object_name"]
                or step.get("joint_name") != interaction["joint_name"]
                or step.get("joint_index") != interaction["joint_index"]
            ):
                raise ValueError("Oracle open_joint step disagrees with its interaction")
            prerequisites = [
                prerequisite["interaction_id"]
                for prerequisite in interaction.get("prerequisites", [])
            ]
            if any(prerequisite not in completed for prerequisite in prerequisites):
                raise ValueError("Oracle plan opens an interaction before its prerequisite")
            completed.append(interaction_id)
        observe_steps = [
            step for step in plan.get("steps", []) if step.get("type") == "observe_target"
        ]
        if is_point_goal:
            if observe_steps:
                raise ValueError("PointGoal oracle plans must not contain observe_target")
            terminal_steps = [
                step
                for step in plan.get("steps", [])
                if step.get("type") == "navigate"
                and step.get("reason") == "satisfy_nav_to_point_success"
            ]
            if len(terminal_steps) != 1:
                raise ValueError("PointGoal oracle plan must contain one terminal navigate step")
            if terminal_steps[0].get("goal_point") != task.get("goal_point"):
                raise ValueError("PointGoal terminal oracle step does not match task.goal_point")
        elif (
            len(observe_steps) != 1
            or observe_steps[0].get("object_name") not in candidate_instances
        ):
            raise ValueError("Oracle plan must observe exactly one runtime target candidate")

    return {
        "payload": payload,
        "task": task,
        "target": target,
        "selected_instance": selected_instance,
        "candidate_instances": candidate_instances,
        "is_point_goal": is_point_goal,
        "domains": domains,
        "interactions": interactions,
        "interaction_by_id": interaction_by_id,
        "articulation_by_joint": articulation_by_joint,
        "generation_validation": payload.get("generation_validation", {}),
    }


def _validate_measured_container_state(context: dict[str, Any]) -> None:
    payload = context["payload"]
    generation_validation = context["generation_validation"]
    articulation_by_joint = context["articulation_by_joint"]
    container_validation = generation_validation.get("container_state_validation", {})
    container_rows = container_validation.get("joints", [])
    if container_validation.get("joint_count") != len(container_rows):
        raise ValueError("container_state_validation count does not match its measured rows")
    if container_validation.get("all_closed") != all(
        row.get("passed") is True for row in container_rows
    ):
        raise ValueError("container_state_validation all_closed disagrees with measured rows")
    for row in container_rows:
        articulation = articulation_by_joint.get(row["joint_name"])
        if articulation is None:
            raise ValueError("Validated container joint is missing from articulation states")
        if abs(float(articulation.get("position")) - float(row["joint_value"])) > 1e-6:
            raise ValueError("Container articulation position disagrees with measured validation")
        if abs(
            float(articulation.get("open_fraction")) - float(row["open_fraction"])
        ) > 1e-6:
            raise ValueError("Container articulation fraction disagrees with measured validation")
    if payload.get("initial_state", {}).get("container_joints_closed") is True and not (
        container_validation.get("all_closed", False)
    ):
        raise ValueError(
            "container_joints_closed is not supported by measured container state validation"
        )


def _validate_passing_success(context: dict[str, Any]) -> None:
    success_evidence = context["generation_validation"].get("success_evidence", {})
    if not (
        success_evidence.get("status") == "passed"
        and success_evidence.get("distance_passed") is True
        and success_evidence.get("visibility_passed") is True
        and success_evidence.get("expected_task_success") is True
    ):
        raise ValueError("Generated episode lacks passing measured NavToObj success evidence")
    task_threshold = float(context["task"].get("succ_pos_threshold", -1.0))
    if float(success_evidence.get("distance_threshold_m", -2.0)) != task_threshold:
        raise ValueError("success_evidence distance threshold does not mirror the task")
    if success_evidence.get("target_object_name") not in context["candidate_instances"]:
        raise ValueError("success_evidence target does not match a runtime candidate")


def _prefixes_by_plan(context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prefixes: dict[str, list[dict[str, Any]]] = {}
    for prefix in context["generation_validation"].get("oracle_prefixes", []):
        prefixes.setdefault(prefix["plan_id"], []).append(prefix)
    return prefixes


def _validate_container_scene(context: dict[str, Any]) -> None:
    payload = context["payload"]
    generation_validation = context["generation_validation"]
    articulation_by_joint = context["articulation_by_joint"]
    if payload.get("interaction_requirement") != "required":
        raise ValueError("Generated container episodes must require interaction")
    if any(_interaction_domain(row) != "container" for row in context["interactions"]):
        raise ValueError("Container episodes may only contain container interactions")

    door_validation = generation_validation.get("door_state_validation", {})
    door_rows = door_validation.get("doors", [])
    if door_validation.get("door_count") != len(door_rows):
        raise ValueError("door_state_validation count does not match its measured rows")
    if door_validation.get("all_open") != all(row.get("passed") is True for row in door_rows):
        raise ValueError("door_state_validation all_open does not match measured door rows")
    for row in door_rows:
        articulation = articulation_by_joint.get(row["joint_name"])
        if articulation is None:
            raise ValueError("Validated channel door is missing from articulation states")
        if abs(float(articulation.get("position")) - float(row["joint_value"])) > 1e-6:
            raise ValueError("Door articulation position does not match measured validation")
        if abs(
            float(articulation.get("open_fraction")) - float(row["open_fraction"])
        ) > 1e-6:
            raise ValueError("Door articulation fraction does not match measured validation")
    if payload.get("initial_state", {}).get("all_doors_open") is True and not door_validation.get(
        "all_open", False
    ):
        raise ValueError("all_doors_open is not supported by measured door state validation")

    _validate_measured_container_state(context)
    if payload.get("initial_state", {}).get("target_visible") is not False:
        raise ValueError("Required container interactions must start with a hidden target")
    navigation_validation = generation_validation.get("navigation_validation", {})
    if not navigation_validation.get("path_found", False):
        raise ValueError("Generated episode has no measured path to its interaction pose")
    if float(navigation_validation.get("start_visibility_fraction", 0.0)) > 0.0 or int(
        navigation_validation.get("start_visible_pixels", 0)
    ) > 0:
        raise ValueError("Generated episode target is visible at the source start")
    _validate_passing_success(context)

    prefixes_by_plan = _prefixes_by_plan(context)
    for plan in payload.get("oracle_plans", []):
        prefixes = prefixes_by_plan.get(plan["plan_id"], [])
        if not prefixes:
            raise ValueError("Oracle plan is missing prefix validation")
        if prefixes[0].get("task_success") is not False:
            raise ValueError("Oracle prefix validation does not prove initial failure")
        if prefixes[-1].get("task_success") is not True:
            raise ValueError("Oracle prefix validation does not prove terminal success")
    minimal_verified = generation_validation.get("minimal_plan_verified")
    minimal_validation = generation_validation.get("minimal_plan_validation", {})
    if minimal_verified is None and minimal_validation.get("status") != "not_executed":
        raise ValueError("Unverified minimal plan must explicitly record not_executed")


def _validate_channel_scene(context: dict[str, Any]) -> None:
    interactions = context["interactions"]
    if any(_interaction_domain(row) != "channel" for row in interactions):
        raise ValueError("Channel episodes may only contain channel interactions")
    if context["payload"].get("interaction_requirement") == "required" and not interactions:
        raise ValueError("Required channel episode has no channel interaction")


def _validate_mixed_scene(context: dict[str, Any]) -> None:
    payload = context["payload"]
    generation_validation = context["generation_validation"]
    interactions = context["interactions"]
    interaction_by_id = context["interaction_by_id"]
    articulation_by_joint = context["articulation_by_joint"]
    interaction_requirement = payload.get("interaction_requirement")
    if interaction_requirement not in {"required", "beneficial"}:
        raise ValueError(
            "Mixed production episodes must classify channel interaction as "
            "required or beneficial"
        )
    channel_ids = [
        row["interaction_id"] for row in interactions if _interaction_domain(row) == "channel"
    ]
    container_ids = [
        row["interaction_id"] for row in interactions if _interaction_domain(row) == "container"
    ]
    if not channel_ids or not container_ids:
        raise ValueError("Mixed episodes must contain both channel and container interactions")
    first_container = interaction_by_id[container_ids[0]]
    reachability_prerequisites = {
        row["interaction_id"]
        for row in first_container.get("prerequisites", [])
        if row.get("type") == "reachability"
    }
    if interaction_requirement == "required":
        if not set(channel_ids).issubset(reachability_prerequisites):
            raise ValueError(
                "First container interaction must depend on all required channel interactions"
            )
        if any(
            "restore_reachability" not in row.get("effect_types", [])
            for row in interactions
            if _interaction_domain(row) == "channel"
        ):
            raise ValueError(
                "Required mixed channel interactions must restore reachability"
            )
    else:
        if set(channel_ids) & reachability_prerequisites:
            raise ValueError(
                "Beneficial mixed channels must not be encoded as reachability prerequisites"
            )
        if any(
            "reduce_navigation_cost" not in row.get("effect_types", [])
            for row in interactions
            if _interaction_domain(row) == "channel"
        ):
            raise ValueError(
                "Beneficial mixed channel interactions must reduce navigation cost"
            )

    door_validation = generation_validation.get("door_state_validation", {})
    door_rows = door_validation.get("doors", [])
    if door_validation.get("door_count") != len(door_rows) or not door_rows:
        raise ValueError("Mixed door_state_validation must contain measured door rows")
    all_closed = door_validation.get(
        "all_closed", door_validation.get("all_required_closed")
    )
    if all_closed is not True:
        raise ValueError("Mixed initial state must prove all evaluated doors are closed")
    for row in door_rows:
        articulation = articulation_by_joint.get(row["joint_name"])
        if articulation is None:
            raise ValueError("Validated mixed door is missing from articulation states")
        if row.get("passed_closed") is not True:
            raise ValueError("A required mixed door was not measured closed")
        if abs(float(articulation.get("position")) - float(row["joint_value"])) > 1e-6:
            raise ValueError("Mixed door articulation position disagrees with readback")
        if abs(float(articulation.get("open_fraction")) - float(row["open_fraction"])) > 1e-6:
            raise ValueError("Mixed door articulation fraction disagrees with readback")

    _validate_measured_container_state(context)
    if payload.get("initial_state", {}).get("target_visible") is not False:
        raise ValueError("Mixed episodes must start with a hidden target")
    navigation = generation_validation.get("navigation_validation", {})
    closed_roots = door_validation.get(
        "closed_root_names",
        door_validation.get("required_closed_root_names", []),
    )
    crossed_roots = navigation.get("all_open_path_crossed_door_roots", [])
    if not navigation.get("all_open_path_found", False):
        raise ValueError("Mixed episode lacks an all-open GT path")
    if not closed_roots or any(root not in crossed_roots for root in closed_roots):
        raise ValueError(
            "All evaluated mixed doors must lie on the measured all-open GT path"
        )
    if interaction_requirement == "required":
        if navigation.get("initial_state_path_found") is not False:
            raise ValueError(
                "mixed_required initial state must make the container pose unreachable"
            )
    else:
        if navigation.get("initial_state_path_found") is not True:
            raise ValueError(
                "mixed_beneficial initial state must retain a measured alternate path"
            )
        if navigation.get("shortcut_verified") is not True:
            raise ValueError(
                "mixed_beneficial samples must verify a navigation-cost reduction"
            )
        initial_length = navigation.get("initial_state_path_length_m")
        restored_length = navigation.get("oracle_restored_path_length_m")
        delta = navigation.get("path_length_delta_m")
        ratio = navigation.get("path_length_ratio_delta")
        thresholds = navigation.get("shortcut_thresholds", {})
        if None in {initial_length, restored_length, delta, ratio}:
            raise ValueError(
                "mixed_beneficial navigation evidence is missing measured path lengths"
            )
        recomputed_delta = float(initial_length) - float(restored_length)
        recomputed_ratio = recomputed_delta / max(float(restored_length), 1e-6)
        if abs(float(delta) - recomputed_delta) > 1e-6:
            raise ValueError("Beneficial path-length delta disagrees with measured paths")
        if abs(float(ratio) - recomputed_ratio) > 1e-6:
            raise ValueError("Beneficial path-length ratio disagrees with measured paths")
        if recomputed_delta < float(thresholds.get("min_delta_m", float("inf"))):
            raise ValueError("Beneficial path-length delta does not pass its threshold")
        if recomputed_ratio < float(thresholds.get("min_ratio", float("inf"))):
            raise ValueError("Beneficial path-length ratio does not pass its threshold")
    if navigation.get("approach_path_found") is not True:
        raise ValueError("Mixed episode lacks a measured path to the closed-door approach pose")
    if navigation.get("oracle_restored_path_found") is not True:
        raise ValueError("Opening required doors must restore reachability")
    if float(navigation.get("start_visibility_fraction", 0.0)) > 0.0 or int(
        navigation.get("start_visible_pixels", 0)
    ) > 0:
        raise ValueError("Mixed target is visible at the source start")
    _validate_passing_success(context)

    prefixes_by_plan = _prefixes_by_plan(context)
    for plan in payload.get("oracle_plans", []):
        prefixes = prefixes_by_plan.get(plan["plan_id"], [])
        if len(prefixes) < 3:
            raise ValueError("Mixed oracle plan needs initial, post-door, and terminal prefixes")
        if prefixes[0].get("task_success") is not False:
            raise ValueError("Mixed prefix evidence does not prove initial failure")
        middle = prefixes[1:-1]
        if not any(
            row.get("robot_reachable_to_next_goal") is True
            and row.get("task_success") is False
            and float(row.get("target_visibility_fraction") or 0.0) == 0.0
            for row in middle
        ):
            raise ValueError("Mixed prefix evidence lacks a reachable-but-still-hidden post-door state")
        if prefixes[-1].get("task_success") is not True:
            raise ValueError("Mixed prefix evidence does not prove terminal success")

    minimal_validation = generation_validation.get("minimal_plan_validation", {})
    if generation_validation.get("minimal_plan_verified") is not True:
        raise ValueError("Mixed production samples must verify plan necessity/benefit")
    if minimal_validation.get("status") != "passed":
        raise ValueError("Mixed minimal-plan validation must pass")
    required_omission_ids = {
        row.get("omitted_interaction_id")
        for row in minimal_validation.get("omission_results", [])
        if row.get("required") is True
    }
    if interaction_requirement == "required":
        if required_omission_ids != set(channel_ids + container_ids):
            raise ValueError(
                "Mixed minimal-plan evidence must cover every required interaction"
            )
    else:
        beneficial_omission_ids = {
            row.get("omitted_interaction_id")
            for row in minimal_validation.get("omission_results", [])
            if row.get("beneficial") is True
        }
        if required_omission_ids != set(container_ids):
            raise ValueError(
                "Beneficial mixed evidence must keep container interactions required"
            )
        if beneficial_omission_ids != set(channel_ids):
            raise ValueError(
                "Beneficial mixed evidence must cover every cost-reducing channel"
            )


def _serialize_and_validate_v3(episode: dict[str, Any]) -> dict[str, Any]:
    minimal_value = episode.get("interactive_nav", {}).get(
        "generation_validation", {}
    ).get("minimal_plan_verified")
    cleaned_episode = _without_none(episode)
    cleaned_episode.setdefault("interactive_nav", {}).setdefault(
        "generation_validation", {}
    )["minimal_plan_verified"] = minimal_value
    validated_spec = EpisodeSpec.model_validate(cleaned_episode)
    validated = json.loads(validated_spec.model_dump_json(exclude_none=True))
    validated["interactive_nav"]["generation_validation"][
        "minimal_plan_verified"
    ] = minimal_value
    source_parent_index = episode.get("interactive_nav", {}).get(
        "parent_benchmark_episode_index"
    )
    if source_parent_index is None:
        validated["interactive_nav"]["parent_benchmark_episode_index"] = None
    source_prefixes = (
        episode.get("interactive_nav", {})
        .get("generation_validation", {})
        .get("oracle_prefixes", [])
    )
    validated_prefixes = (
        validated.get("interactive_nav", {})
        .get("generation_validation", {})
        .get("oracle_prefixes", [])
    )
    required_nullable_prefix_keys = {
        "robot_reachable_to_next_goal",
        "target_distance_passed",
        "target_visibility_fraction",
        "task_success",
    }
    for source, destination in zip(source_prefixes, validated_prefixes, strict=False):
        for key in required_nullable_prefix_keys:
            if key in source and source[key] is None:
                destination[key] = None
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator(schema).validate(validated)
    return validated


def validate_interactive_nav_v3_episode(
    episode: dict[str, Any],
    *,
    expected_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Unified V3 entry: common invariants followed by domain-specific checks."""
    context = _validate_common_v3_episode(episode, expected_domains=expected_domains)
    domains = context["domains"]
    if context["is_point_goal"] and "container" in domains:
        raise ValueError(
            "Pure PointGoal V3 episodes currently support channel interactions only"
        )
    if domains == ["container"]:
        _validate_container_scene(context)
    elif domains == ["channel"]:
        _validate_channel_scene(context)
    elif domains == ["channel", "container"]:
        _validate_mixed_scene(context)
    else:
        raise ValueError(f"Unsupported interaction domain combination: {domains}")
    return _serialize_and_validate_v3(episode)


def validate_channel_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return validate_interactive_nav_v3_episode(
        episode, expected_domains=["channel"]
    )


def validate_point_goal_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    if episode.get("task", {}).get("task_type") != "nav_to_point":
        raise ValueError("PointGoal validator requires task_type=nav_to_point")
    return validate_interactive_nav_v3_episode(episode, expected_domains=["channel"])


def validate_instruction_goal_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Validate an instruction overlay without repeating collection-time evidence checks."""

    language = episode.get("language", {})
    if language.get("task_input_mode") != "instruction":
        raise ValueError("InstructionGoal requires language.task_input_mode=instruction")
    if not str(language.get("task_description") or "").strip():
        raise ValueError("InstructionGoal requires a non-empty instruction")

    payload = episode.get("interactive_nav", {})
    plans = payload.get("oracle_plans") or []
    if not plans or plans[0] != payload.get("oracle_plan"):
        raise ValueError("InstructionGoal source must keep oracle_plan=oracle_plans[0]")
    valid_step_indices = set(range(len(plans[0].get("steps") or [])))
    grounded_steps = language.get("grounded_plan_step_indices") or []
    if not set(grounded_steps).issubset(valid_step_indices):
        raise ValueError("InstructionGoal references an unknown oracle plan step")

    valid_entities: set[str] = set()
    target = payload.get("target", {})
    for key in ("selected_instance", "container_name"):
        if target.get(key):
            valid_entities.add(str(target[key]))
    valid_entities.update(
        str(value)
        for value in target.get("instruction_consistent_candidates", [])
        if value
    )
    for interaction in payload.get("interactions") or []:
        for key in ("interaction_id", "object_name", "joint_name"):
            if interaction.get(key):
                valid_entities.add(str(interaction[key]))
    graph_context = (
        payload.get("instruction_generation", {}).get("graph_context") or {}
    )
    for node in graph_context.get("nodes") or []:
        for value in (
            node.get("id"),
            node.get("name"),
            (node.get("attributes") or {}).get("source_object_name"),
        ):
            if value:
                valid_entities.add(str(value))
    grounded_entities = {str(value) for value in language.get("grounded_entity_ids") or []}
    unknown_entities = grounded_entities - valid_entities
    if unknown_entities:
        raise ValueError(
            f"InstructionGoal references unknown V3 entities: {sorted(unknown_entities)}"
        )
    return _serialize_and_validate_v3(episode)


def validate_container_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return validate_interactive_nav_v3_episode(
        episode, expected_domains=["container"]
    )


def validate_mixed_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return validate_interactive_nav_v3_episode(
        episode, expected_domains=["channel", "container"]
    )
