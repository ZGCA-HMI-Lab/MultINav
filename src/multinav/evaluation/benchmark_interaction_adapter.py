"""Public interaction boundary shared by the V3 benchmark adapters.

The evaluated method may request an opaque object-level action, while the
evaluator is allowed to resolve that request to simulator-private objects and
joints.  This module owns the narrow seam between those two worlds:

* public command geometry is copied through an explicit allow-list;
* executor results are rebuilt from an explicit allow-list rather than
  redacting a private result in place; and
* a non-articulated portal is classified only from public aperture evidence.

The functions intentionally have no ROS or MuJoCo dependency, so the protocol
can be tested independently from a live rollout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


PUBLIC_FAILURE_REASONS = frozenset(
    {
        "articulation_resolution_failed",
        "capability_unavailable",
        "drawer_interaction_execution_unavailable",
        "drawer_open_execution_failed",
        "drawer_scan_execution_failed",
        "executor_failed",
        "force_target_not_reached",
        "interaction_approach_options_exhausted",
        "interaction_not_visible",
        "interaction_pose_invalid",
        "interaction_pose_poll_exhausted",
        "interaction_too_far",
        "interaction_unsupported",
        "non_articulated",
        "non_articulated_blocked_portal",
        "non_articulated_closed_portal",
        "postcondition_not_satisfied",
        "rejected",
        "unknown_instance_id",
        "unresolved_drawer_scan_target",
        "unsafe_open_sweep",
        "unsupported_action",
    }
)

PUBLIC_VERIFICATION_SOURCES = frozenset(
    {
        "backend_postcondition",
        "drawer_scan_backend",
        "evaluator_object_skill",
        "executor_capability_check",
        "executor_drawer_sequence_failure",
        "executor_open_sweep_preflight",
        "executor_pose_precondition",
        "executor_resolution_failure",
        "executor_state_verification",
        "observed_aperture_geometry",
        "successful_action_postcondition",
    }
)

PUBLIC_STATES = frozenset(
    {
        "ajar",
        "blocked",
        "closed",
        "open",
        "static_closed",
        "static_open",
        "unavailable",
        "unknown",
    }
)

PUBLIC_CAPABILITIES = frozenset(
    {
        "articulated",
        "blocked",
        "confirmed",
        "static",
        "unavailable",
        "unknown",
    }
)


def sanitize_public_failure_reason(value: Any) -> str:
    """Normalize one failure token without forwarding private diagnostics."""

    reason = str(value or "").strip().casefold()
    if not reason:
        return ""
    return reason if reason in PUBLIC_FAILURE_REASONS else "executor_failed"


def _finite_float(value: Any, *, minimum: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if minimum is not None and result < minimum:
        return None
    return result


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _finite_vector(value: Any, *, size: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    values: list[float] = []
    for item in list(value)[:size]:
        numeric = _finite_float(item)
        if numeric is None:
            return []
        values.append(numeric)
    return values if len(values) == size else []


def sanitize_public_aperture_observation(value: Any) -> dict[str, Any] | None:
    """Return the compact public portal-aperture contract, if supplied."""

    if not isinstance(value, Mapping):
        return None
    leaf = str(value.get("door_leaf") or "unknown").strip().casefold()
    connectivity = str(value.get("connectivity") or "unknown").strip().casefold()
    if leaf not in {"absent", "present", "unknown"}:
        leaf = "unknown"
    if connectivity not in {"open", "blocked", "unknown"}:
        connectivity = "unknown"
    confidence = _finite_float(value.get("confidence", 1.0), minimum=0.0)
    confidence = 0.0 if confidence is None else min(1.0, confidence)
    return {
        "door_leaf": leaf,
        "connectivity": connectivity,
        "confidence": confidence,
    }


def portal_capability_from_public_evidence(
    value: Any,
) -> dict[str, Any]:
    """Classify an unroutable portal without consulting simulator metadata."""

    evidence = sanitize_public_aperture_observation(value)
    if evidence is None:
        return {
            "state": "unavailable",
            "interaction_capability": "unavailable",
            "interactable": False,
            "retryable": False,
            "failure_reason": "capability_unavailable",
            "verification_source": "executor_capability_check",
        }
    leaf = str(evidence["door_leaf"])
    connectivity = str(evidence["connectivity"])
    confidence = float(evidence["confidence"])
    if confidence >= 0.5 and leaf == "absent" and connectivity == "open":
        result = {
            "state": "static_open",
            "interaction_capability": "static",
            "interactable": False,
            "retryable": False,
            "verification_source": "observed_aperture_geometry",
        }
    elif confidence >= 0.5 and leaf == "present" and connectivity == "blocked":
        result = {
            "state": "static_closed",
            "interaction_capability": "static",
            "interactable": False,
            "retryable": False,
            "failure_reason": "non_articulated_closed_portal",
            "verification_source": "observed_aperture_geometry",
        }
    elif confidence >= 0.5 and connectivity == "blocked":
        result = {
            "state": "blocked",
            "interaction_capability": "blocked",
            "interactable": False,
            "retryable": False,
            "failure_reason": "non_articulated_blocked_portal",
            "verification_source": "observed_aperture_geometry",
        }
    else:
        result = {
            "state": "unavailable",
            "interaction_capability": "unavailable",
            "interactable": False,
            "retryable": False,
            "failure_reason": "capability_unavailable",
            "verification_source": "observed_aperture_geometry",
        }
    result["portal_aperture_observation"] = evidence
    result["confidence"] = confidence
    return result


def sanitize_public_interaction_command(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only method-produced geometry needed by the trusted executor."""

    result: dict[str, Any] = {}
    for key in (
        "command_id",
        "decision_id",
        "candidate_id",
        "event_id",
        "node_id",
        "object_id",
        "node_type",
        "action",
        "interaction_mode",
        "container_kind",
        "expected_state",
        "sequence_type",
        "operation_method",
    ):
        value = payload.get(key)
        if value is not None:
            result[key] = str(value)
    for key, size in (
        ("approach_goal_xyyaw", 3),
        ("interaction_approach_pose_xyyaw", 3),
        ("interaction_approach_axis_xy", 2),
    ):
        vector = _finite_vector(payload.get(key), size=size)
        if vector:
            result[key] = vector
    for key in (
        "interaction_ready_distance_m",
        "interaction_ready_yaw_tolerance_rad",
    ):
        numeric = _finite_float(payload.get(key), minimum=0.0)
        if numeric is not None:
            result[key] = numeric
    aperture = sanitize_public_aperture_observation(
        payload.get("portal_aperture_observation")
    )
    if aperture is not None:
        result["portal_aperture_observation"] = aperture
    # Drawer regions are normalized image points produced by the method.  Joint
    # selectors and simulator part IDs are deliberately not accepted.
    regions: list[dict[str, list[float]]] = []
    raw_regions = payload.get("open_regions")
    if isinstance(raw_regions, list):
        for raw in raw_regions[:12]:
            if not isinstance(raw, Mapping):
                continue
            center = _finite_vector(raw.get("center"), size=2)
            if center and all(0.0 <= item <= 1.0 for item in center):
                candidate = {"center": center}
                if candidate not in regions:
                    regions.append(candidate)
    if regions:
        result["open_regions"] = regions
    return result


def validate_public_interaction_pose(
    command: Mapping[str, Any],
    *,
    actual_pose_xyyaw: Sequence[float],
) -> dict[str, Any]:
    """Apply the ordinary bridge's public approach-pose precondition.

    The expected pose and tolerances originate in the method-produced command;
    the evaluator supplies only the robot's current public navigation pose.  No
    object pose, simulator identity, or articulation geometry enters this check.
    """

    expected = _finite_vector(
        command.get("interaction_approach_pose_xyyaw"),
        size=3,
    )
    actual = _finite_vector(actual_pose_xyyaw, size=3)
    if not expected:
        return {"checked": False, "reason": "no_expected_approach_pose"}
    if not actual:
        return {"checked": True, "valid": False}
    position_error_m = math.hypot(
        actual[0] - expected[0],
        actual[1] - expected[1],
    )
    yaw_error_rad = abs(
        math.atan2(
            math.sin(actual[2] - expected[2]),
            math.cos(actual[2] - expected[2]),
        )
    )
    distance_tolerance_m = max(
        0.05,
        _finite_float(
            command.get("interaction_ready_distance_m", 0.45),
            minimum=0.0,
        )
        or 0.45,
    )
    yaw_tolerance_rad = max(
        0.05,
        _finite_float(
            command.get("interaction_ready_yaw_tolerance_rad", 0.55),
            minimum=0.0,
        )
        or 0.55,
    )
    result: dict[str, Any] = {
        "checked": True,
        "valid": bool(
            position_error_m <= distance_tolerance_m
            and yaw_error_rad <= yaw_tolerance_rad
        ),
        "expected_pose_xyyaw": expected,
        "actual_pose_xyyaw": actual,
        "position_error_m": position_error_m,
        "yaw_error_rad": yaw_error_rad,
        "distance_tolerance_m": distance_tolerance_m,
        "yaw_tolerance_rad": yaw_tolerance_rad,
    }
    approach_axis = _finite_vector(
        command.get("interaction_approach_axis_xy"),
        size=2,
    )
    if approach_axis:
        result["approach_axis_xy"] = approach_axis
    return result


def validate_public_interaction_observation(
    observation: Mapping[str, Any] | None,
    *,
    actual_pose_xyyaw: Sequence[float],
    max_distance_m: float,
) -> dict[str, Any]:
    """Validate proximity using only an already-published public 3D box.

    This closes the evaluator boundary against guessed opaque IDs and fabricated
    approach poses without consulting a simulator-only object pose or visibility
    test.  The evaluated method has already received the same world-frame box.
    """

    if not isinstance(observation, Mapping):
        return {"checked": False, "valid": False, "reason": "not_observed"}
    box = observation.get("box_3d")
    if not isinstance(box, Mapping):
        return {"checked": False, "valid": False, "reason": "missing_public_box"}
    center = _finite_vector(box.get("center"), size=3)
    size = _finite_vector(box.get("size"), size=3)
    actual = _finite_vector(actual_pose_xyyaw, size=3)
    limit = _finite_float(max_distance_m, minimum=0.0)
    if (
        not center
        or not size
        or any(value < 0.0 for value in size)
        or not actual
        or limit is None
        or limit <= 0.0
    ):
        return {"checked": False, "valid": False, "reason": "invalid_public_box"}
    dx = max(0.0, abs(actual[0] - center[0]) - 0.5 * size[0])
    dy = max(0.0, abs(actual[1] - center[1]) - 0.5 * size[1])
    distance = math.hypot(dx, dy)
    capture_step = _nonnegative_int(observation.get("capture_step"))
    age_seconds = _finite_float(observation.get("age_seconds"), minimum=0.0)
    result: dict[str, Any] = {
        "checked": True,
        "valid": bool(distance <= limit),
        "aabb_distance_m": distance,
        "maximum_distance_m": limit,
        "reason": "public_box_in_range" if distance <= limit else "public_box_too_far",
    }
    if capture_step is not None:
        result["capture_step"] = capture_step
    if age_seconds is not None:
        result["age_seconds"] = age_seconds
    return result


def _sanitize_pose_validation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in ("checked", "valid"):
        if isinstance(value.get(key), bool):
            result[key] = bool(value[key])
    for key, size in (
        ("expected_pose_xyyaw", 3),
        ("actual_pose_xyyaw", 3),
        ("approach_axis_xy", 2),
    ):
        vector = _finite_vector(value.get(key), size=size)
        if vector:
            result[key] = vector
    for key in (
        "position_error_m",
        "yaw_error_rad",
        "distance_tolerance_m",
        "yaw_tolerance_rad",
    ):
        numeric = _finite_float(value.get(key), minimum=0.0)
        if numeric is not None:
            result[key] = numeric
    reason = str(value.get("reason") or "").strip().casefold()
    if reason in {"no_expected_approach_pose"}:
        result["reason"] = reason
    return result or None


def sanitize_public_interaction_outcome(
    outcome: Mapping[str, Any] | None,
    *,
    success: bool,
) -> dict[str, Any]:
    """Project one private executor result onto a leak-safe public schema."""

    source = dict(outcome or {})
    result: dict[str, Any] = {}
    for key in ("state", "pre_state", "post_state"):
        value = str(source.get(key) or "").strip().casefold()
        if value in PUBLIC_STATES:
            result[key] = value
    capability = str(source.get("interaction_capability") or "").strip().casefold()
    if capability in PUBLIC_CAPABILITIES:
        result["interaction_capability"] = capability
    for key in ("interactable", "retryable"):
        if isinstance(source.get(key), bool):
            result[key] = bool(source[key])
    reason = sanitize_public_failure_reason(
        source.get("failure_reason") or source.get("reason")
    )
    if reason:
        result["failure_reason"] = reason
    verification = str(source.get("verification_source") or "").strip().casefold()
    if verification:
        result["verification_source"] = (
            verification
            if verification in PUBLIC_VERIFICATION_SOURCES
            else "evaluator_object_skill"
        )
    for key in (
        "sim_steps_consumed",
        "task_steps_consumed",
        "physics_substeps",
        "force_applied_step",
        "result_published_step",
        "step",
    ):
        numeric = _nonnegative_int(source.get(key))
        if numeric is not None:
            result[key] = numeric
    for key in ("confidence", "execution_cost"):
        numeric = _finite_float(source.get(key), minimum=0.0)
        if numeric is not None:
            result[key] = min(1.0, numeric) if key == "confidence" else numeric
    recommended_retreat_m = _finite_float(
        source.get("recommended_retreat_m"),
        minimum=0.0,
    )
    if recommended_retreat_m is not None:
        result["recommended_retreat_m"] = recommended_retreat_m
    validation = _sanitize_pose_validation(source.get("interaction_pose_validation"))
    if validation is not None:
        result["interaction_pose_validation"] = validation
    aperture = sanitize_public_aperture_observation(
        source.get("portal_aperture_observation")
    )
    if aperture is not None:
        result["portal_aperture_observation"] = aperture

    if success:
        result.setdefault("state", "open")
        result.setdefault("post_state", result["state"])
        result.setdefault("interaction_capability", "articulated")
        result.setdefault("interactable", True)
        result.setdefault("retryable", False)
        result.setdefault("verification_source", "executor_state_verification")
    else:
        result.setdefault("retryable", False)
        result.setdefault("verification_source", "evaluator_object_skill")
    return result


__all__ = [
    "portal_capability_from_public_evidence",
    "sanitize_public_aperture_observation",
    "sanitize_public_failure_reason",
    "sanitize_public_interaction_command",
    "sanitize_public_interaction_outcome",
    "validate_public_interaction_observation",
    "validate_public_interaction_pose",
]
