"""V3 benchmark terminal checks and aggregate reporting metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any

import mujoco
import numpy as np

from multinav.sim import container_scene_probe as probe


SUCCESS_OPEN_FRACTION = 0.8
DEFAULT_PATH_BINS_M = (0.0, 3.0, 5.0, 8.0, 12.0, 20.0)

# These values are deliberately part of the evaluator protocol rather than a
# post-hoc plotting choice.  ``lambda`` matches the deployed semantic policy's
# interaction cost weight; the larger error surcharge and fixed failure penalty
# make Eq. (1) in the paper reproducible from saved episode results.  All three
# can be overridden by the evaluator CLI and are frozen into every manifest and
# episode result.
PAPER_METRIC_SCHEMA_VERSION = "interactive_nav_v3_paper_metrics_v1"
DEFAULT_PAPER_COST_INTERACTION_ATTEMPT = 0.30
DEFAULT_PAPER_COST_ERROR_SURCHARGE = 1.00
DEFAULT_PAPER_COST_FAILURE_PENALTY = 5.00


@dataclass(frozen=True)
class PaperMetricConfig:
    """Frozen coefficients for the paper's all-episode Total Cost metric."""

    interaction_attempt_cost: float = DEFAULT_PAPER_COST_INTERACTION_ATTEMPT
    error_interaction_surcharge: float = DEFAULT_PAPER_COST_ERROR_SURCHARGE
    failure_penalty: float = DEFAULT_PAPER_COST_FAILURE_PENALTY

    def validate(self) -> None:
        values = {
            "interaction_attempt_cost": self.interaction_attempt_cost,
            "error_interaction_surcharge": self.error_interaction_surcharge,
            "failure_penalty": self.failure_penalty,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.error_interaction_surcharge) <= float(self.interaction_attempt_cost):
            raise ValueError(
                "error_interaction_surcharge must be strictly greater than "
                "interaction_attempt_cost (mu > lambda)"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": PAPER_METRIC_SCHEMA_VERSION,
            "formula": "L_exec_m + lambda*A + mu*E + kappa*(1-S)",
            "interaction_attempt_cost": float(self.interaction_attempt_cost),
            "error_interaction_surcharge": float(self.error_interaction_surcharge),
            "failure_penalty": float(self.failure_penalty),
            "lambda_interaction_attempt_cost": float(self.interaction_attempt_cost),
            "mu_error_interaction_surcharge": float(self.error_interaction_surcharge),
            "kappa_failure_penalty": float(self.failure_penalty),
        }


@dataclass(frozen=True)
class PaperInteractionAttemptScore:
    """Per-episode interaction quantities needed by IP and Total Cost."""

    interaction_attempt_count: int
    valid_interaction_attempt_count: int
    error_interaction_attempt_count: int
    task_irrelevant_interaction_attempt_count: int
    failed_interaction_attempt_count: int
    repeated_interaction_attempt_count: int
    interaction_precision_episode: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def joint_open_fraction(env: Any, interaction: dict[str, Any]) -> float:
    """Read a joint in the benchmark's semantic closed-to-open direction."""

    joint_name = str(interaction["joint_name"])
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Interaction joint not found: {joint_name}")
    value = probe.joint_value_by_name(env, joint_name)
    lower, upper = [float(item) for item in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lower, upper])
    return float(np.clip(probe.semantic_open_fraction(value, closed, opened), 0.0, 1.0))


def target_candidate_names(episode: dict[str, Any]) -> list[str]:
    """Return the runtime-authoritative V3 NavToObj candidate instances."""

    nav = episode["interactive_nav"]
    target = nav["target"]
    task = episode.get("task", {})
    selection_mode = str(
        task.get("selection_mode")
        or target.get("selection_mode")
        or nav.get("success_criteria", {}).get("target_selection")
        or "specific_instance"
    )
    if selection_mode == "specific_instance":
        selected = target.get("selected_instance") or task.get("pickup_obj_name")
        if not selected:
            raise ValueError("specific_instance target is missing selected_instance")
        return [str(selected)]
    if selection_mode != "any_candidate":
        raise ValueError(f"Unsupported target selection mode: {selection_mode!r}")

    raw_candidates = task.get("pickup_obj_candidates") or target.get(
        "instruction_consistent_candidates"
    )
    candidates = list(dict.fromkeys(str(value) for value in (raw_candidates or []) if value))
    if not candidates:
        raise ValueError("any_candidate target has no runtime candidates")
    return candidates


def target_metric_rows(task: Any, episode: dict[str, Any]) -> list[dict[str, Any]]:
    objects = task.env.object_managers[task.env.current_batch_index]
    robot_xy = np.asarray(task.env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    criteria = episode["interactive_nav"]["success_criteria"]
    camera_name = str(criteria["visibility"].get("camera_name", "head_camera"))
    rows: list[dict[str, Any]] = []
    for target_name in target_candidate_names(episode):
        target = objects.get_object_by_name(target_name)
        distance = float(
            np.linalg.norm(np.asarray(target.position[:2], dtype=float) - robot_xy)
        )
        visibility = float(task.env.check_visibility(camera_name, target_name))
        rows.append(
            {
                "target_name": target_name,
                "distance_m": distance,
                "visibility_fraction": visibility,
            }
        )
    return rows


def target_metrics(task: Any, episode: dict[str, Any]) -> tuple[bool, float, float]:
    """Evaluate one frozen target or any instruction-consistent V3 candidate."""

    rows = target_metric_rows(task, episode)
    threshold = float(
        episode["interactive_nav"]["success_criteria"]["distance"]["threshold_m"]
    )
    # ``NavToObjTask`` is authoritative for the native benchmark: it selects
    # the nearest candidate first and checks visibility on that same object.
    # Keep this exact ordering for V3 as well; accepting ``any_candidate`` in
    # the schema must not silently introduce a different success definition.
    chosen = min(rows, key=lambda row: float(row["distance_m"]))
    return (
        bool(
            float(chosen["distance_m"]) < threshold
            and float(chosen["visibility_fraction"]) > 0.0
        ),
        float(chosen["distance_m"]),
        float(chosen["visibility_fraction"]),
    )


def oracle_terminal_goal_consistency(task: Any, episode: dict[str, Any]) -> dict[str, Any]:
    """Check that frozen terminal waypoints still describe the live target.

    V3 stores a selected *instance* and a frozen oracle plan separately.  A
    benchmark assembled from a stale source can accidentally pair the selected
    instance with a navigation goal for a different same-category object.  The
    evaluator must surface that before treating a policy score as a formal
    benchmark result.

    The check is intentionally conservative: it only rejects an episode when a
    plan explicitly labels a navigation step as ``satisfy_nav_to_obj_success``
    and none of those terminal goals could meet the recorded distance criterion
    under the live replayed scene.  Episodes without such a step remain
    ``checked=False`` rather than being guessed incompatible.
    """

    nav = episode["interactive_nav"]
    target_names = target_candidate_names(episode)
    objects = task.env.object_managers[task.env.current_batch_index]
    target_positions = {
        target_name: np.asarray(
            objects.get_object_by_name(target_name).position[:2], dtype=float
        )
        for target_name in target_names
    }
    threshold = float(nav["success_criteria"]["distance"]["threshold_m"])
    candidates: list[dict[str, Any]] = []
    for plan in _oracle_plans(nav):
        plan_id = str(plan.get("plan_id", "oracle"))
        for step in plan.get("steps", []):
            if step.get("type") != "navigate" or step.get("reason") != "satisfy_nav_to_obj_success":
                continue
            goal = np.asarray(step.get("goal_point", [])[:2], dtype=float)
            if goal.shape != (2,):
                continue
            tolerance = float(step.get("position_tolerance_m", 0.0))
            distances = {
                target_name: float(np.linalg.norm(goal - target_xy))
                for target_name, target_xy in target_positions.items()
            }
            matched_target_name = min(distances, key=distances.get)
            distance = distances[matched_target_name]
            allowed = threshold + max(0.0, tolerance)
            candidates.append(
                {
                    "plan_id": plan_id,
                    "goal_point_xy": goal.tolist(),
                    "position_tolerance_m": tolerance,
                    "distance_to_live_target_m": distance,
                    "allowed_distance_m": allowed,
                    "consistent": bool(distance < allowed),
                    "matched_target_name": matched_target_name,
                    "candidate_distances_m": distances,
                }
            )
    singular_target = target_names[0] if len(target_names) == 1 else None
    singular_position = (
        target_positions[singular_target].tolist() if singular_target is not None else None
    )
    return {
        "checked": bool(candidates),
        "consistent": True if not candidates else any(bool(item["consistent"]) for item in candidates),
        "target_name": singular_target,
        "target_position_xy": singular_position,
        "target_names": target_names,
        "target_positions_xy": {
            name: position.tolist() for name, position in target_positions.items()
        },
        "success_distance_threshold_m": threshold,
        "terminal_goal_candidates": candidates,
    }


@dataclass(frozen=True)
class InteractionTerminalScore:
    required_interaction_success: bool
    sequence_success: bool
    non_interaction_success: bool | None
    interaction_fractions: dict[str, float]
    valid_plan_id: str | None
    correct_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _oracle_plans(nav: dict[str, Any]) -> list[dict[str, Any]]:
    plans = list(nav.get("oracle_plans") or [])
    if not plans and nav.get("oracle_plan"):
        plans = [dict(nav["oracle_plan"])]
    return plans


def score_interactions(
    env: Any,
    episode: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> InteractionTerminalScore:
    """Score V3 interactions including failed/extra attempts.

    In particular, no-interaction episodes use *attempt count*, not successful
    interaction ids.  This prevents a policy from receiving credit after an
    invalid or failed door-open request.
    """

    nav = episode["interactive_nav"]
    requirement = str(nav["interaction_requirement"])
    interactions = list(nav.get("interactions", []))
    by_id = {str(row["interaction_id"]): row for row in interactions}
    fractions = {key: joint_open_fraction(env, row) for key, row in by_id.items()}
    successful_required: list[str] = []
    transient_satisfied: set[str] = set()
    for row in attempts:
        if row.get("classification") != "required_valid" or not bool(row.get("success")):
            continue
        # An evaluator-owned object-level skill may execute a private set of
        # joints for one public ``open(opaque_id)`` request.  Keep the legacy
        # singular key for older traces while accepting the private plural form
        # needed to score the resulting V3 postconditions correctly.
        resolved_ids = row.get("resolved_interaction_ids")
        if isinstance(resolved_ids, (list, tuple)) and resolved_ids:
            successful_required.extend(str(value) for value in resolved_ids if value is not None)
        elif row.get("resolved_interaction_id") is not None:
            successful_required.append(str(row["resolved_interaction_id"]))
        metadata = row.get("metadata") or {}
        transient_ids = metadata.get("transient_satisfied_interaction_ids", [])
        if isinstance(transient_ids, (list, tuple, set)):
            transient_satisfied.update(str(value) for value in transient_ids if value is not None)
    completed: set[str] = set()
    sequence_success = True
    correct_action_count = 0
    for interaction_id in successful_required:
        row = by_id.get(interaction_id)
        if row is None:
            sequence_success = False
            continue
        prereq = {str(item["interaction_id"]) for item in row.get("prerequisites", [])}
        order_ok = prereq.issubset(completed)
        if not order_ok:
            sequence_success = False
        elif interaction_id not in completed:
            correct_action_count += 1
        completed.add(interaction_id)

    plans = _oracle_plans(nav)
    plan_rows: list[tuple[str, set[str]]] = [
        (str(plan.get("plan_id", "oracle")), {str(value) for value in plan.get("required_interaction_ids", [])})
        for plan in plans
    ]
    if requirement == "unnecessary":
        required_ok = True
        valid_plan_id = None
        sequence_success = True
        non_interaction_success: bool | None = len(attempts) == 0
    else:
        valid_plan_id = next(
            (
                plan_id
                for plan_id, required_ids in plan_rows
                if required_ids.issubset(completed)
                and all(
                    fractions.get(item, 0.0) >= SUCCESS_OPEN_FRACTION
                    or item in transient_satisfied
                    for item in required_ids
                )
            ),
            None,
        )
        required_ok = valid_plan_id is not None
        if plan_rows and not required_ok:
            sequence_success = False
        non_interaction_success = None
    return InteractionTerminalScore(
        required_interaction_success=bool(required_ok),
        sequence_success=bool(sequence_success),
        non_interaction_success=non_interaction_success,
        interaction_fractions=fractions,
        valid_plan_id=valid_plan_id,
        correct_action_count=int(correct_action_count),
    )


def _attempt_resolved_interaction_ids(attempt: dict[str, Any]) -> list[str]:
    """Return every V3 interaction ID an evaluator associated with one attempt."""

    values = attempt.get("resolved_interaction_ids")
    ids = [str(value) for value in values if value is not None] if isinstance(values, (list, tuple, set)) else []
    value = attempt.get("resolved_interaction_id")
    if value is not None:
        ids.append(str(value))
    return list(dict.fromkeys(ids))


def _attempt_targeted_interaction_ids(attempt: dict[str, Any]) -> list[str]:
    """Return V3 IDs targeted by an attempt, including failed resolutions."""

    ids = _attempt_resolved_interaction_ids(attempt)
    metadata = attempt.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    values = metadata.get("requested_interaction_ids")
    if isinstance(values, (list, tuple, set)):
        ids.extend(str(value) for value in values if value is not None)
    return list(dict.fromkeys(ids))


def _attempt_effect_interaction_ids(
    attempt: dict[str, Any],
    required_ids: set[str],
) -> list[str]:
    """Return required effects actually produced by one private evaluator attempt.

    The ROS object skill can perform a physically successful sub-effect while
    returning a task-level acknowledgement of ``False`` because another
    required effect is still pending.  Its explicit effect list is therefore
    authoritative for process metrics.  The ordinary single-joint executor
    falls back to its successful resolved ID.
    """

    metadata = attempt.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    candidates: list[str] = []
    for key in (
        "effect_achieved_interaction_ids",
        "transient_satisfied_interaction_ids",
    ):
        values = metadata.get(key)
        if isinstance(values, (list, tuple, set)):
            candidates.extend(str(value) for value in values if value is not None)
    # ``resolved_interaction_ids`` is evaluator-private output from the
    # object-level skill and contains only joints whose expected open effect
    # was observed.  Keep this compatibility path for already generated V3
    # ROS traces while the explicit metadata is being introduced.
    plural = attempt.get("resolved_interaction_ids")
    if isinstance(plural, (list, tuple, set)):
        candidates.extend(str(value) for value in plural if value is not None)
    if not candidates and bool(attempt.get("success")):
        candidates.extend(_attempt_resolved_interaction_ids(attempt))
    return list(dict.fromkeys(value for value in candidates if value in required_ids))


def paper_interaction_attempt_score(
    episode: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> PaperInteractionAttemptScore:
    """Compute paper-IP numerator/denominator and the union-counted error set.

    ``E`` counts erroneous *attempts*, rather than adding category counts: an
    invalid request which also fails is one attempt in Eq. (1).  A repeated
    request only becomes an error once every required effect it produces was
    already complete; a multi-joint macro that completes a new effect is not
    penalised merely because it also touches an earlier joint.
    """

    nav = episode["interactive_nav"]
    requirement = str(nav.get("interaction_requirement", "required"))
    required_ids = {
        str(interaction["interaction_id"])
        for interaction in nav.get("interactions", [])
        if interaction.get("interaction_id") is not None
    }
    completed_ids: set[str] = set()
    valid_count = 0
    error_count = 0
    irrelevant_count = 0
    failed_count = 0
    repeated_count = 0

    for attempt in attempts:
        targeted_ids = _attempt_targeted_interaction_ids(attempt)
        classification = str(attempt.get("classification") or "")
        # A failed evaluator-side resolution may have no credited ID yet still
        # be a request for the annotated entity.  `required_valid` is assigned
        # only after matching that entity; do not turn such a failed relevant
        # request into an unrelated-object error in the saved breakdown.
        targets_required = bool(required_ids.intersection(targeted_ids)) or bool(
            required_ids and classification == "required_valid"
        )
        effect_ids = _attempt_effect_interaction_ids(attempt, required_ids)
        has_new_effect = bool(set(effect_ids) - completed_ids)
        repeated = bool(effect_ids) and not has_new_effect
        # A relevant request that cannot produce its expected effect is a
        # failed interaction even if an executor reported a benign low-level
        # completion.  Conversely, an effect-producing ROS macro is not
        # treated as failed just because its task-level acknowledgement waits
        # for a later prerequisite.
        failed = bool(
            not effect_ids
            and (
                targets_required
                or classification == "required_valid"
                or not bool(attempt.get("success"))
            )
        )
        irrelevant = not targets_required

        if has_new_effect:
            valid_count += 1
        if irrelevant:
            irrelevant_count += 1
        if failed:
            failed_count += 1
        if repeated:
            repeated_count += 1
        if irrelevant or failed or repeated:
            error_count += 1
        completed_ids.update(effect_ids)

    attempt_count = len(attempts)
    if attempt_count:
        precision = float(valid_count / attempt_count)
    else:
        precision = 1.0 if requirement == "unnecessary" else 0.0
    return PaperInteractionAttemptScore(
        interaction_attempt_count=int(attempt_count),
        valid_interaction_attempt_count=int(valid_count),
        error_interaction_attempt_count=int(error_count),
        task_irrelevant_interaction_attempt_count=int(irrelevant_count),
        failed_interaction_attempt_count=int(failed_count),
        repeated_interaction_attempt_count=int(repeated_count),
        interaction_precision_episode=float(precision),
    )


def paper_episode_total_cost(
    *,
    nav_success: bool,
    navigation_path_length_m: float,
    interaction_score: PaperInteractionAttemptScore,
    config: PaperMetricConfig,
) -> tuple[float, dict[str, float | int]]:
    """Evaluate and expose every term of the paper's Total Cost equation."""

    config.validate()
    path_length = max(0.0, float(navigation_path_length_m))
    attempt_cost = float(config.interaction_attempt_cost) * int(
        interaction_score.interaction_attempt_count
    )
    error_cost = float(config.error_interaction_surcharge) * int(
        interaction_score.error_interaction_attempt_count
    )
    failure_cost = float(config.failure_penalty) * int(not bool(nav_success))
    total = float(path_length + attempt_cost + error_cost + failure_cost)
    return total, {
        "navigation_path_length_m": path_length,
        "interaction_attempt_cost": attempt_cost,
        "error_interaction_surcharge": error_cost,
        "failure_penalty": failure_cost,
        "interaction_attempt_count": int(interaction_score.interaction_attempt_count),
        "error_interaction_attempt_count": int(interaction_score.error_interaction_attempt_count),
        "nav_success_indicator": int(bool(nav_success)),
        "total_cost": total,
    }


def reference_path_length_m(episode: dict[str, Any]) -> float | None:
    """Return the best frozen V3 path reference usable for SPL.

    Channel/mixed episodes normally carry ``oracle_restored_path_length_m``;
    container episodes store the validated start-to-interaction path as
    ``path_length_m``.  This ordered fallback covers all currently frozen V3
    episodes while retaining the provenance in the result trace.
    """

    nav = episode["interactive_nav"]
    validation = dict(nav.get("generation_validation", {}).get("navigation_validation", {}))
    if str(nav.get("interaction_requirement")) == "unnecessary":
        keys = (
            "initial_state_path_length_m",
            "oracle_path_length_m",
            "path_length_m",
            "all_open_path_length_m",
        )
    else:
        keys = (
            "oracle_restored_path_length_m",
            "oracle_path_length_m",
            "path_length_m",
            "interaction_pose_path_length_m",
            "all_open_path_length_m",
        )
    for key in keys:
        value = validation.get(key)
        if value is not None:
            return float(value)
    return None


def path_length_bin(length_m: float | None, bins: tuple[float, ...] = DEFAULT_PATH_BINS_M) -> str | None:
    if length_m is None:
        return None
    length = float(length_m)
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if lower <= length < upper:
            return f"[{lower:g},{upper:g})"
    return f"[{bins[-1]:g},inf)"


def spl(success: bool, reference_length_m: float | None, actual_length_m: float) -> float | None:
    if reference_length_m is None:
        return None
    reference = max(0.0, float(reference_length_m))
    actual = max(0.0, float(actual_length_m))
    if not success:
        return 0.0
    return float(reference / max(reference, actual, 1e-9))


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def _rate_with_fallback(rows: list[dict[str, Any]], key: str, fallback_key: str) -> float | None:
    values = [row.get(key, row.get(fallback_key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _strict_mean(values: list[float | None]) -> float | None:
    """Mean only when every episode in a formal denominator has the value."""

    if not values or any(value is None for value in values):
        return None
    return float(np.mean([float(value) for value in values if value is not None]))


def _strict_rate(values: list[bool | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(sum(bool(value) for value in values if value is not None) / len(values))


def _paper_nav_success(row: dict[str, Any]) -> bool | None:
    """Return the terminal NavToObj predicate, never legacy V3 success."""

    value = row.get("nav_success")
    if value is None:
        value = row.get("task_success")
    return None if value is None else bool(value)


def _paper_spl(row: dict[str, Any]) -> float | None:
    nav_success = _paper_nav_success(row)
    reference = _finite_float(row.get("reference_path_length_m"))
    executed = _finite_float(row.get("navigation_path_length_m"))
    if nav_success is None or reference is None or executed is None:
        return None
    # Recompute rather than trusting a stored `spl`: V3 versions before this
    # protocol incorrectly gated SPL by interaction-conditioned success.
    return spl(nav_success, reference, executed)


def _paper_interaction_precision(row: dict[str, Any]) -> float | None:
    value = _finite_float(row.get("interaction_precision_episode"))
    if value is not None:
        return value
    valid_count = _finite_float(row.get("valid_interaction_attempt_count"))
    attempt_count = _finite_float(row.get("interaction_action_count"))
    requirement = row.get("interaction_requirement")
    if valid_count is None or attempt_count is None or requirement is None:
        return None
    if attempt_count > 0.0:
        return float(valid_count / attempt_count)
    return 1.0 if str(requirement) == "unnecessary" else 0.0


def _paper_total_cost(row: dict[str, Any]) -> float | None:
    # Episode results store the evaluated equation and its full breakdown.  Do
    # not derive a cost from legacy correct-action counters: those cannot
    # distinguish effect-level credit from the paper's attempt-level V/E.
    return _finite_float(row.get("episode_total_cost"))


def _triggered_early_stops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        payload
        for row in rows
        if isinstance((payload := row.get("early_stop")), dict)
        and bool(payload.get("triggered"))
    ]


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interaction_attempts = sum(int(row.get("interaction_action_count", 0)) for row in rows)
    correct_actions = sum(int(row.get("correct_interaction_action_count", 0)) for row in rows)
    paper_nav_values = [_paper_nav_success(row) for row in rows]
    paper_spl_values = [_paper_spl(row) for row in rows]
    required_rows = [
        row for row in rows if str(row.get("interaction_requirement") or "") == "required"
    ]
    paper_isr_values = [
        None
        if row.get("required_interaction_success") is None
        else bool(row.get("required_interaction_success"))
        for row in required_rows
    ]
    paper_ip_values = [_paper_interaction_precision(row) for row in rows]
    paper_cost_values = [_paper_total_cost(row) for row in rows]
    paper_sr = _strict_rate(paper_nav_values)
    paper_spl = _strict_mean(paper_spl_values)
    paper_isr = _strict_rate(paper_isr_values)
    paper_ip = _strict_mean(paper_ip_values)
    paper_total_cost = _strict_mean(paper_cost_values)
    early_stops = _triggered_early_stops(rows)
    early_stop_reasons = Counter(
        str(payload.get("reason") or "unknown") for payload in early_stops
    )
    early_stop_failure_counts = [
        int(payload["failed_subgoal_count"])
        for payload in early_stops
        if payload.get("failed_subgoal_count") is not None
    ]
    return {
        "episode_count": len(rows),
        # The five primary fields below intentionally follow the paper rather
        # than the historical interaction-conditioned V3 `success` field.
        "success_rate": paper_sr,
        "task_success_rate": _rate_with_fallback(rows, "task_success", "nav_success"),
        "interaction_conditioned_success_rate": _rate_with_fallback(
            rows, "interaction_conditioned_success", "success"
        ),
        "nav_success_rate": paper_sr,
        "required_interaction_success_rate": paper_isr,
        "sequence_success_rate": _rate(rows, "sequence_success"),
        "non_interaction_success_rate": _rate(rows, "non_interaction_success"),
        "interaction_precision": paper_ip,
        "mean_total_cost": paper_total_cost,
        "paper_sr": paper_sr,
        "paper_spl": paper_spl,
        "paper_isr": paper_isr,
        "paper_ip": paper_ip,
        "paper_total_cost": paper_total_cost,
        "paper_metric_denominators": {
            "sr_episode_count": len(rows),
            "sr_missing_episode_count": sum(value is None for value in paper_nav_values),
            "spl_episode_count": len(rows),
            "spl_missing_episode_count": sum(value is None for value in paper_spl_values),
            "isr_required_episode_count": len(required_rows),
            "isr_missing_required_episode_count": sum(value is None for value in paper_isr_values),
            "ip_episode_count": len(rows),
            "ip_missing_episode_count": sum(value is None for value in paper_ip_values),
            "total_cost_episode_count": len(rows),
            "total_cost_missing_episode_count": sum(
                value is None for value in paper_cost_values
            ),
        },
        # Retain the former micro calculation only as a diagnostic.  It is not
        # paper IP because long traces would otherwise dominate the score.
        "interaction_precision_micro_legacy": (
            None if interaction_attempts == 0 else correct_actions / interaction_attempts
        ),
        "mean_step_count": _mean(rows, "step_count"),
        "mean_applied_action_step_count": _mean(
            rows, "applied_action_step_count"
        ),
        "mean_no_fresh_action_count": _mean(rows, "no_fresh_action_count"),
        "mean_navigation_path_length_m": _mean(rows, "navigation_path_length_m"),
        "mean_reference_path_length_m": _mean(rows, "reference_path_length_m"),
        "mean_spl": paper_spl,
        "mean_saved_spl_diagnostic": _mean(rows, "spl"),
        "spl_eligible_episode_count": sum(value is not None for value in paper_spl_values),
        "mean_total_simulated_seconds": _mean(rows, "total_simulated_seconds"),
        "mean_interaction_action_count": _mean(rows, "interaction_action_count"),
        "mean_extra_interaction_action_count": _mean(rows, "extra_interaction_action_count"),
        "mean_invalid_interaction_action_count": _mean(rows, "invalid_interaction_action_count"),
        "terminal_reason_counts": dict(Counter(str(row.get("terminal_reason")) for row in rows)),
        "early_stop_episode_count": len(early_stops),
        "early_stop_reason_counts": dict(sorted(early_stop_reasons.items())),
        "mean_early_stop_failed_subgoal_count": (
            float(np.mean(early_stop_failure_counts)) if early_stop_failure_counts else None
        ),
    }


def summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall, task-family and distribution-stratified metrics."""

    eligible_rows = [row for row in rows if bool(row.get("scoring_eligible", True))]
    exclusion_reasons = Counter(
        str(reason)
        for row in rows
        if not bool(row.get("scoring_eligible", True))
        for reason in row.get("scoring_exclusion_reasons", [])
    )
    early_stops = _triggered_early_stops(rows)
    early_stop_reasons = Counter(
        str(payload.get("reason") or "unknown") for payload in early_stops
    )
    groups: dict[str, list[dict[str, Any]]] = {"overall": list(eligible_rows)}
    for domain in ("channel", "container", "mixed"):
        expected = {"channel", "container"} if domain == "mixed" else {domain}
        groups[f"domain/{domain}"] = [row for row in eligible_rows if set(row.get("domains", [])) == expected]
    for requirement in ("required", "unnecessary", "beneficial"):
        groups[f"requirement/{requirement}"] = [
            row for row in eligible_rows if row.get("interaction_requirement") == requirement
        ]
    for recipe in sorted({str(row["recipe"]) for row in eligible_rows if row.get("recipe")}):
        groups[f"recipe/{recipe}"] = [row for row in eligible_rows if row.get("recipe") == recipe]
    for interaction_type in sorted({item for row in eligible_rows for item in row.get("interaction_types", [])}):
        groups[f"interaction_type/{interaction_type}"] = [
            row for row in eligible_rows if interaction_type in row.get("interaction_types", [])
        ]
    for label in sorted({str(row["path_length_bin"]) for row in eligible_rows if row.get("path_length_bin")}):
        groups[f"path_length/{label}"] = [row for row in eligible_rows if row.get("path_length_bin") == label]
    return {
        "schema_version": "interactive_nav_v3_benchmark_eval_summary_v4",
        "paper_metric_schema_version": PAPER_METRIC_SCHEMA_VERSION,
        "total_episode_count": len(rows),
        "scoring_eligible_episode_count": len(eligible_rows),
        "runtime_ineligible_episode_count": len(rows) - len(eligible_rows),
        "runtime_ineligible_reason_counts": dict(exclusion_reasons),
        "early_stop_episode_count": len(early_stops),
        "early_stop_reason_counts": dict(sorted(early_stop_reasons.items())),
        "groups": {name: _group_summary(values) for name, values in groups.items() if values},
    }
