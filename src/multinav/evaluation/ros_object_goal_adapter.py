"""Evaluator-owned ROS adapter for restricted-GT object-goal episodes.

This module deliberately sits on the *evaluator* side of the V3 boundary.  It
does not import, configure, or modify ``Interactive-Nav-SG-nav``.  A running
ROS graph continues to use its established topics:

* ``/semantic_decision/target`` for a public object-goal context;
* ``/semantic_mapping/gt_observations`` for evaluator-published observations;
* ``/semantic_decision/interaction_command`` for an object-level request; and
* ``/semantic_mapping/interaction_result`` for its minimal completion result.

The perception contract is intentionally small: semantic name, opaque instance
ID, 2D bounding box, segmentation RLE, and 3D axis-aligned box.  The active
wire record is exactly ``{id, name, bbox_2d, mask_rle, box_3d}``, which is the
minimal-GT input understood by the semantic mapping stack.  It deliberately
does not include legacy aliases, derived visibility counters, articulation
metadata, hierarchy, state, or a simulator object name.

Interaction execution is an evaluator-private capability.  A caller registers
``opaque_id -> private_handle`` at reset, consumes an
:class:`EvaluatorInteractionRequest`, invokes its force skill internally, then
calls :meth:`RosObjectGoalEvaluatorAdapter.complete_interaction`.  Published
results expose only command routing plus the ordinary bridge's allow-listed
semantic state/capability/retry contract; they never contain a joint name,
joint value/fraction, simulator articulation record, or private handle.

``rospy`` and ``std_msgs`` are imported lazily by :meth:`start`, so this module
can be unit-tested without ROS installed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any

from .restricted_gt_perception import (
    audit_restricted_gt_payload,
    binary_mask_rle_stats,
)
from .benchmark_interaction_adapter import (
    portal_capability_from_public_evidence,
    sanitize_public_failure_reason,
    sanitize_public_interaction_command,
    sanitize_public_interaction_outcome,
)


DEFAULT_TARGET_TOPIC = "/semantic_decision/target"
DEFAULT_GT_OBSERVATIONS_TOPIC = "/semantic_mapping/gt_observations"
DEFAULT_INTERACTION_COMMAND_TOPIC = "/semantic_decision/interaction_command"
DEFAULT_INTERACTION_RESULT_TOPIC = "/semantic_mapping/interaction_result"

# A direct drawer scan is intentionally bound to a box that was already
# published on the public perception topic.  The threshold is deliberately
# high: this is identity routing for a sealed evaluator skill, not a detector
# association heuristic.
DIRECT_DRAWER_SCAN_BBOX_MIN_IOU = 0.85
# The semantic graph can lag the evaluator while the robot completes the
# approach to a drawer.  Retain a bounded public-frame window large enough for
# that normal navigation delay.  A command still has to name the exact public
# capture step and pass the high IoU/unique-match checks below, so this does
# not turn an old box into a free-form object selector.  These are wall-clock
# limits, intentionally independent of a simulator's logical ``stamp_sec``.
DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE = 128
DIRECT_DRAWER_SCAN_BBOX_TTL_S = 30.0


class RestrictedGTContractError(ValueError):
    """Raised when an evaluator-side payload violates the public contract."""


def _public_bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    """Normalize one finite, positive-area public pixel box.

    This helper deliberately accepts only the same geometry that the evaluator
    has already put on the ROS perception topic.  It never consults simulator
    geometry, segmentation IDs, or private object handles.
    """

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and item >= 0.0 for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _public_capture_step(value: Any) -> int | None:
    """Normalize one non-negative public perception capture step."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    capture_step = int(numeric)
    return capture_step if capture_step >= 0 else None


def _bbox_iou_xyxy(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return IoU for two validated public pixel boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


_PERCEPTION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "episode_id",
        "episode_reset",
        "capture_step",
        "stamp_sec",
        "observations",
    }
)
_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        # Canonical restricted-GT contract.
        "id",
        "name",
        "bbox_2d_xyxy",
        "segmentation_rle",
        "box3d_center",
        "box3d_size",
        # Compatibility aliases for the existing semantic-mapping input
        # normalizer.  They are exact copies of id/name, never private fields.
        "instance_id",
        "semantic_name",
        "source_object_name",
    }
)
_STRICT_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        "id",
        "name",
        "bbox_2d_xyxy",
        "segmentation_rle",
        "box3d_center",
        "box3d_size",
    }
)
_SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        "id",
        "name",
        "bbox_2d",
        "mask_rle",
        "box_3d",
    }
)
_SEMANTIC_MINIMAL_BOX_3D_KEYS = frozenset({"center", "size", "frame_id"})
_TARGET_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "episode_id",
        "enabled",
        "target_name",
        "object_labels",
        "instruction",
        # These configure public observation reliability, not task GT.  They
        # are needed because the legacy mapper receives mask-derived counts
        # instead of the old privileged visible-fraction estimate.
        "min_visible_pixels",
        "min_visible_fraction",
        "min_consecutive_observations",
        "completion_requires_visibility",
        "require_current_visibility",
    }
)
_FORBIDDEN_TARGET_CONTEXT_KEYS = frozenset(
    {
        "target_instance_id",
        "target_source_object_name",
        "target_container_instance_id",
        "target_container_source_object_name",
        "target_container_name",
        "target_container_labels",
        "require_interaction",
        "interaction_requirement",
        "oracle_plan",
        "selected_instance",
        "scene_modifications",
    }
)


def adapt_restricted_gt_frame_for_semantic_mapping(
    payload: Mapping[str, Any],
    *,
    stamp_sec: float,
    capture_step: int | None = None,
) -> dict[str, Any]:
    """Convert a V3 restricted-GT frame into the semantic minimal-GT wire form.

    The source frame is independently schema-audited before conversion.  The
    resulting observations retain only the five fields the semantic mapper
    needs: opaque ``id``, semantic ``name``, 2-D box, compact mask RLE and
    world-frame AABB.  In particular, this function must not add legacy
    aliases, decoded pixel coordinates, visibility statistics, object hierarchy
    or any simulator-side articulation metadata.
    """

    audit_restricted_gt_payload(payload)
    observations = [
        {
            "id": str(observation["instance_id"]),
            "name": str(observation["name"]),
            "bbox_2d": list(observation["bbox_2d_xyxy"]),
            "mask_rle": deepcopy(dict(observation["mask_rle"])),
            "box_3d": deepcopy(dict(observation["bbox_3d"])),
        }
        for observation in payload["observations"]
    ]
    adapted = {
        "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["frame_index"] if capture_step is None else capture_step),
        "stamp_sec": float(stamp_sec),
        "observations": observations,
    }
    validate_semantic_minimal_perception_payload(adapted)
    return adapted


def adapt_restricted_gt_frame_for_legacy_mapping(
    payload: Mapping[str, Any],
    *,
    stamp_sec: float,
    capture_step: int | None = None,
    consecutive_observations: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Losslessly bridge the canonical restricted-GT frame to the old mapper.

    The external semantic mapper predates the V3 public protocol and expects
    flattened geometry plus ``source_object_name``.  The latter is *only* an
    opaque-ID alias here.  Every extra field below is either a lossless alias
    of the public 3-D box/name/ID or a deterministic statistic of the public
    segmentation mask; no articulation, hierarchy, state, or source name is
    reintroduced.
    """

    audit_restricted_gt_payload(payload)
    streaks = consecutive_observations if consecutive_observations is not None else {}
    if bool(payload["episode_reset"]):
        streaks.clear()
    visible_now: set[str] = set()
    observations: list[dict[str, Any]] = []
    for observation in payload["observations"]:
        instance_id = str(observation["instance_id"])
        name = str(observation["name"])
        bbox_3d = dict(observation["bbox_3d"])
        mask_rle = dict(observation["mask_rle"])
        _height, _width, visible_pixels = binary_mask_rle_stats(mask_rle)
        visible_now.add(instance_id)
        streaks[instance_id] = int(streaks.get(instance_id, 0)) + 1
        observations.append(
            {
                "instance_id": instance_id,
                "name": name,
                "semantic_name": name,
                # Compatibility only: unchanged ROS code routes interaction
                # commands through this key.  Its value is never a MuJoCo name.
                "source_object_name": instance_id,
                "bbox_2d": list(observation["bbox_2d_xyxy"]),
                "mask_rle": mask_rle,
                "bbox_3d": bbox_3d,
                "position": list(bbox_3d["center"]),
                "aabb_center": list(bbox_3d["center"]),
                "aabb_size": list(bbox_3d["size"]),
                "visible_pixels": visible_pixels,
                "consecutive_observations": int(streaks[instance_id]),
            }
        )
    for instance_id in list(streaks):
        if instance_id not in visible_now:
            streaks.pop(instance_id, None)
    return {
        "schema_version": "interactive_nav_v3_restricted_gt_legacy_adapter_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["frame_index"] if capture_step is None else capture_step),
        "stamp_sec": float(stamp_sec),
        "observations": observations,
    }


def _float_vector(value: Sequence[float] | Any, *, size: int, field_name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RestrictedGTContractError(f"{field_name} must be a numeric {size}-vector") from exc
    if len(result) != size:
        raise RestrictedGTContractError(f"{field_name} must have exactly {size} values")
    return result


def _copy_json_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RestrictedGTContractError(f"{field_name} must be a mapping")
    result = deepcopy(dict(value))
    try:
        json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise RestrictedGTContractError(f"{field_name} must be JSON serialisable") from exc
    return result


@dataclass(frozen=True)
class RestrictedGTObservation:
    """One visible public instance in the restricted-GT perception contract.

    ``instance_id`` must be a stable, episode-local opaque token such as
    ``"obj_000017"``.  It must not be a MuJoCo body name, asset ID or source
    object name.  ``name`` is a semantic class label (for example ``"door"``
    or ``"refrigerator"``), not an instance identifier.
    """

    instance_id: str
    name: str
    bbox_2d_xyxy: Sequence[float]
    segmentation_rle: Mapping[str, Any] = field(default_factory=dict)
    box3d_center: Sequence[float] = (0.0, 0.0, 0.0)
    box3d_size: Sequence[float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        instance_id = str(self.instance_id).strip()
        name = str(self.name).strip()
        if not instance_id:
            raise RestrictedGTContractError("instance_id must be non-empty")
        if not name:
            raise RestrictedGTContractError("name must be non-empty")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "bbox_2d_xyxy",
            _float_vector(self.bbox_2d_xyxy, size=4, field_name="bbox_2d_xyxy"),
        )
        object.__setattr__(
            self,
            "box3d_center",
            _float_vector(self.box3d_center, size=3, field_name="box3d_center"),
        )
        object.__setattr__(
            self,
            "box3d_size",
            _float_vector(self.box3d_size, size=3, field_name="box3d_size"),
        )
        object.__setattr__(
            self,
            "segmentation_rle",
            _copy_json_mapping(self.segmentation_rle, field_name="segmentation_rle"),
        )

    def to_strict_payload(self) -> dict[str, Any]:
        """Return exactly the six canonical public fields."""

        return {
            "id": self.instance_id,
            "name": self.name,
            "bbox_2d_xyxy": list(self.bbox_2d_xyxy),
            "segmentation_rle": deepcopy(dict(self.segmentation_rle)),
            "box3d_center": list(self.box3d_center),
            "box3d_size": list(self.box3d_size),
        }

    def to_payload(self) -> dict[str, Any]:
        """Return a strict record adapted with lossless legacy aliases."""

        return adapt_strict_observation_for_legacy_mapping(self.to_strict_payload())

    def to_semantic_minimal_payload(self) -> dict[str, Any]:
        """Return the compact record consumed by the semantic mapping stack."""

        return {
            "id": self.instance_id,
            "name": self.name,
            "bbox_2d": list(self.bbox_2d_xyxy),
            "mask_rle": deepcopy(dict(self.segmentation_rle)),
            "box_3d": {
                "center": list(self.box3d_center),
                "size": list(self.box3d_size),
                "frame_id": "world",
            },
        }


def _validate_perception_header(payload: Mapping[str, Any]) -> list[Any]:
    if not isinstance(payload, Mapping):
        raise RestrictedGTContractError("perception payload must be a mapping")
    unexpected_top_level = set(payload) - _PERCEPTION_TOP_LEVEL_KEYS
    if unexpected_top_level:
        raise RestrictedGTContractError(
            "perception payload contains forbidden top-level field(s): "
            + ", ".join(sorted(unexpected_top_level))
        )
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise RestrictedGTContractError("perception payload observations must be a list")
    return observations


def _validate_canonical_observation(observation: Mapping[str, Any], index: int) -> None:
    canonical_id = str(observation.get("id") or "")
    canonical_name = str(observation.get("name") or "")
    if not canonical_id or not canonical_name:
        raise RestrictedGTContractError(f"observation {index} requires non-empty id and name")
    _float_vector(observation.get("bbox_2d_xyxy"), size=4, field_name="bbox_2d_xyxy")
    _float_vector(observation.get("box3d_center"), size=3, field_name="box3d_center")
    _float_vector(observation.get("box3d_size"), size=3, field_name="box3d_size")
    _copy_json_mapping(observation.get("segmentation_rle"), field_name="segmentation_rle")


def validate_strict_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate a frame with only canonical name/id/2D+3D fields.

    This is the transport-neutral contract a detector replacement must satisfy.
    It has no legacy aliases, articulation hints, visibility counters or state.
    """

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _STRICT_PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"strict observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        _validate_canonical_observation(observation, index)


def validate_semantic_minimal_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate the compact minimal-GT payload accepted by semantic mapping.

    This validator intentionally accepts no evaluator compatibility aliases.
    Geometry and masks are copied from the already-audited V3 restricted-GT
    frame, but it also validates direct callers such as a detector-backed
    transport implementation.
    """

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        required = _SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS - set(observation)
        if required:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} is missing required field(s): "
                + ", ".join(sorted(required))
            )
        instance_id = str(observation.get("id") or "").strip()
        name = str(observation.get("name") or "").strip()
        if not instance_id or not name:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} requires non-empty id and name"
            )
        _float_vector(observation.get("bbox_2d"), size=4, field_name="bbox_2d")
        try:
            binary_mask_rle_stats(observation["mask_rle"])
        except (TypeError, ValueError) as exc:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} has invalid mask_rle"
            ) from exc
        box_3d = observation.get("box_3d")
        if not isinstance(box_3d, Mapping):
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d must be a mapping"
            )
        unexpected_box_fields = set(box_3d) - _SEMANTIC_MINIMAL_BOX_3D_KEYS
        if unexpected_box_fields:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d contains forbidden field(s): "
                + ", ".join(sorted(unexpected_box_fields))
            )
        required_box_fields = _SEMANTIC_MINIMAL_BOX_3D_KEYS - set(box_3d)
        if required_box_fields:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d is missing required field(s): "
                + ", ".join(sorted(required_box_fields))
            )
        _float_vector(box_3d.get("center"), size=3, field_name="box_3d.center")
        _float_vector(box_3d.get("size"), size=3, field_name="box_3d.size")
        if not str(box_3d.get("frame_id") or "").strip():
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d.frame_id must be non-empty"
            )


def adapt_strict_observation_for_legacy_mapping(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly adapt one strict record for the existing mapping input.

    The current ROS semantic mapper reads ``semantic_name`` and
    ``instance_id``.  Its behavior planner also emits ``source_object_name``
    in interaction commands.  All three aliases below are mechanically copied
    from the public ``name``/opaque ``id`` fields, so this function cannot add
    private object identity, ``is_*`` affordances, joints or state.
    """

    validate_strict_perception_payload(
        {
            "schema_version": 1,
            "episode_id": "validation_only",
            "episode_reset": False,
            "capture_step": 0,
            "stamp_sec": 0.0,
            "observations": [dict(observation)],
        }
    )
    strict = deepcopy(dict(observation))
    strict["instance_id"] = strict["id"]
    strict["semantic_name"] = strict["name"]
    strict["source_object_name"] = strict["id"]
    return strict


def adapt_strict_perception_payload_for_legacy_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a legacy-compatible frame without adding any information."""

    validate_strict_perception_payload(payload)
    adapted = deepcopy(dict(payload))
    adapted["observations"] = [
        adapt_strict_observation_for_legacy_mapping(observation)
        for observation in payload["observations"]
    ]
    validate_restricted_perception_payload(adapted)
    return adapted


def adapt_strict_perception_payload_for_semantic_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert the strict detector-facing form to the active semantic wire form.

    The detector-facing schema intentionally remains independent of the ROS
    mapper.  This conversion is a mechanical field rename plus a fixed public
    world-frame label; it does not derive visibility, aliases or private state.
    """

    validate_strict_perception_payload(payload)
    adapted = {
        "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["capture_step"]),
        "stamp_sec": float(payload["stamp_sec"]),
        "observations": [
            {
                "id": str(observation["id"]),
                "name": str(observation["name"]),
                "bbox_2d": list(observation["bbox_2d_xyxy"]),
                "mask_rle": deepcopy(dict(observation["segmentation_rle"])),
                "box_3d": {
                    "center": list(observation["box3d_center"]),
                    "size": list(observation["box3d_size"]),
                    "frame_id": "world",
                },
            }
            for observation in payload["observations"]
        ],
    }
    validate_semantic_minimal_perception_payload(adapted)
    return adapted


def validate_restricted_perception_payload(payload: Mapping[str, Any]) -> None:
    """Assert that a legacy-compatible payload remains strictly lossless."""

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        _validate_canonical_observation(observation, index)
        canonical_id = str(observation.get("id") or "")
        canonical_name = str(observation.get("name") or "")
        if str(observation.get("instance_id") or "") != canonical_id:
            raise RestrictedGTContractError(f"observation {index} instance_id must alias id")
        if str(observation.get("source_object_name") or "") != canonical_id:
            raise RestrictedGTContractError(
                f"observation {index} source_object_name must alias the opaque id"
            )
        if str(observation.get("semantic_name") or "") != canonical_name:
            raise RestrictedGTContractError(f"observation {index} semantic_name must alias name")


@dataclass(frozen=True)
class EvaluatorInteractionRequest:
    """Evaluator-private object-level interaction request.

    ``private_handle`` is intentionally not serialisable and must only be used
    by the evaluator's trusted force-skill executor.  It is never copied into a
    ROS message or trace intended for the evaluated navigation method.
    """

    command_id: str
    episode_id: str
    instance_id: str
    action: str
    private_handle: Any = field(repr=False, compare=False)
    node_id: str = ""
    candidate_id: str = ""
    decision_id: str = ""
    # These are method-produced visual hints, never simulator selectors.  They
    # remain evaluator-private so the public completion event stays the compact
    # opaque-object contract.
    sequence_type: str = ""
    open_regions: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # True only when the evaluator uniquely routed a drawer scan from a box it
    # had already exposed in the latest public perception frame.  This is not
    # emitted in the ROS completion event.
    direct_bbox_drawer_scan: bool = False
    # A semantic portal can be present in the navigation graph while not
    # corresponding to an evaluator-registered articulated object.  Keep that
    # rejection queued so the evaluator can record it as an invalid *attempt*
    # rather than silently losing it or reporting a physical skill failure.
    rejection_reason: str = ""
    # Leak-safe method-produced command geometry.  This is suitable for the
    # evaluator-owned execution adapter, unlike the original ROS dictionary,
    # which may contain guessed joint/source selectors from legacy methods.
    public_command: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    # Recent geometry copied from a frame that already crossed the restricted
    # public ROS boundary.  It proves interaction provenance without consulting
    # a simulator-only visibility/AABB oracle.
    public_observation: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _PublicVisibleBoxFrame:
    """One evaluator-published frame eligible for direct bbox routing."""

    capture_step: int
    published_at_sec: float
    boxes: tuple[tuple[str, tuple[float, float, float, float]], ...]
    boxes_3d: tuple[
        tuple[str, tuple[float, float, float], tuple[float, float, float]], ...
    ] = ()


InteractionExecutor = Callable[[EvaluatorInteractionRequest], bool | Mapping[str, Any]]


def build_public_target_context(
    *,
    episode_id: str,
    target_name: str,
    instruction: str = "",
    object_labels: Sequence[str] | None = None,
    enabled: bool = True,
    min_visible_pixels: int = 16,
    min_visible_fraction: float = 0.0,
    min_consecutive_observations: int = 1,
    completion_requires_visibility: bool = True,
    require_current_visibility: bool = False,
) -> dict[str, Any]:
    """Build the only target context sent to the ROS navigation method.

    The target class/aliases must come from the public language task (or a
    fixed public language lexicon), never from V3's selected-instance or
    interaction annotations.
    """

    normalized_episode_id = str(episode_id).strip()
    normalized_target_name = str(target_name).strip()
    if not normalized_episode_id:
        raise RestrictedGTContractError("episode_id must be non-empty")
    if not normalized_target_name:
        raise RestrictedGTContractError("target_name must be non-empty")
    labels = [str(item).strip() for item in object_labels or [normalized_target_name]]
    labels = [item for item in labels if item]
    if not labels:
        labels = [normalized_target_name]
    # Preserve supplied order but avoid needless repeated target aliases.
    labels = list(dict.fromkeys(labels))
    context = {
        "schema_version": 1,
        "episode_id": normalized_episode_id,
        "enabled": bool(enabled),
        "target_name": normalized_target_name,
        "object_labels": labels,
        "instruction": str(instruction),
        "min_visible_pixels": max(1, int(min_visible_pixels)),
        "min_visible_fraction": max(0.0, min(1.0, float(min_visible_fraction))),
        "min_consecutive_observations": max(1, int(min_consecutive_observations)),
        "completion_requires_visibility": bool(completion_requires_visibility),
        "require_current_visibility": bool(require_current_visibility),
    }
    validate_public_target_context(context)
    return context


def validate_public_target_context(payload: Mapping[str, Any]) -> None:
    """Reject target contexts containing private V3 task annotations."""

    if not isinstance(payload, Mapping):
        raise RestrictedGTContractError("target context must be a mapping")
    keys = set(payload)
    forbidden = keys & _FORBIDDEN_TARGET_CONTEXT_KEYS
    if forbidden:
        raise RestrictedGTContractError(
            "target context contains private field(s): " + ", ".join(sorted(forbidden))
        )
    unexpected = keys - _TARGET_CONTEXT_KEYS
    if unexpected:
        raise RestrictedGTContractError(
            "target context contains unsupported field(s): " + ", ".join(sorted(unexpected))
        )


class RosObjectGoalEvaluatorAdapter:
    """Bridge a V3 evaluator episode to an unchanged external ROS graph.

    Usage from an evaluator-owned rollout loop::

        adapter.reset(
            episode_id="eval_000042",
            target_context=build_public_target_context(...),
            private_instances={"obj_000017": private_runtime_object},
        )
        adapter.publish_observations(visible_instances, capture_step=0)
        request = adapter.pop_next_interaction_request()
        if request is not None:
            # This call may use joints/force internally; the ROS graph cannot.
            completed = force_skill.open_object(request.private_handle)
            adapter.complete_interaction(request.command_id, success=completed)

    Passing ``interaction_executor`` is useful for a synchronous test harness.
    Production evaluators should normally leave it unset and run force control
    from their own simulation-step loop after consuming pending requests.
    """

    def __init__(
        self,
        *,
        rospy_module: Any | None = None,
        string_message_type: Any | None = None,
        target_topic: str = DEFAULT_TARGET_TOPIC,
        gt_observations_topic: str = DEFAULT_GT_OBSERVATIONS_TOPIC,
        interaction_command_topic: str = DEFAULT_INTERACTION_COMMAND_TOPIC,
        interaction_result_topic: str = DEFAULT_INTERACTION_RESULT_TOPIC,
        interaction_executor: InteractionExecutor | None = None,
        require_public_interaction_evidence: bool = False,
        queue_rejected_interactions: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.target_topic = str(target_topic)
        self.gt_observations_topic = str(gt_observations_topic)
        self.interaction_command_topic = str(interaction_command_topic)
        self.interaction_result_topic = str(interaction_result_topic)
        self._rospy = rospy_module
        self._String = string_message_type
        self._interaction_executor = interaction_executor
        self.require_public_interaction_evidence = bool(
            require_public_interaction_evidence
        )
        self.queue_rejected_interactions = bool(queue_rejected_interactions)
        self._clock = clock
        self._lock = threading.RLock()
        self._started = False
        self._target_publisher = None
        self._observations_publisher = None
        self._result_publisher = None
        self._command_subscriber = None
        self._episode_id = ""
        self._episode_generation = 0
        self._private_instances: dict[str, Any] = {}
        # Public aliases can be introduced by a compatibility layer that
        # deliberately normalizes an opaque object ID (for example an
        # ``obj_000017`` portal represented to the semantic graph as a generic
        # ``door_1234``).  They resolve only inside this evaluator bridge.
        self._canonical_instance_ids: dict[str, str] = {}
        self._pending_by_command_id: dict[str, EvaluatorInteractionRequest] = {}
        self._pending_order: deque[str] = deque()
        self._seen_command_ids: set[str] = set()
        self._command_sequence = 0
        self._event_sequence = 0
        self._published_result_events: list[dict[str, Any]] = []
        self._legacy_consecutive_observations: dict[str, int] = {}
        # These hold only ids and boxes that have already been published on
        # the public perception topic.  Do not add source names, simulator
        # handles, joints, visibility internals, or state.
        self._public_visible_box_frames: deque[_PublicVisibleBoxFrame] = deque(
            maxlen=DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE
        )

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def pending_interaction_count(self) -> int:
        with self._lock:
            return len(self._pending_by_command_id)

    @property
    def published_result_events(self) -> list[dict[str, Any]]:
        """Return public, redacted result events for evaluator diagnostics."""

        with self._lock:
            return deepcopy(self._published_result_events)

    def start(self) -> None:
        """Create ROS publishers/subscriber lazily.

        Supplying ``rospy_module`` and ``string_message_type`` is enough for
        tests; otherwise imports occur only when a live ROS evaluator starts.
        """

        with self._lock:
            if self._started:
                return
            if self._rospy is None or self._String is None:
                try:
                    import rospy  # type: ignore[import-not-found]
                    from std_msgs.msg import String  # type: ignore[import-not-found]
                except ImportError as exc:  # pragma: no cover - exercised only without ROS.
                    raise RuntimeError(
                        "ROS is unavailable; provide rospy_module and string_message_type for tests"
                    ) from exc
                self._rospy = rospy
                self._String = String
            self._target_publisher = self._rospy.Publisher(
                self.target_topic, self._String, queue_size=1, latch=True
            )
            self._observations_publisher = self._rospy.Publisher(
                self.gt_observations_topic, self._String, queue_size=2
            )
            self._result_publisher = self._rospy.Publisher(
                self.interaction_result_topic, self._String, queue_size=4
            )
            self._command_subscriber = self._rospy.Subscriber(
                self.interaction_command_topic,
                self._String,
                self._interaction_command_callback,
                queue_size=8,
            )
            self._started = True

    def close(self) -> None:
        """Detach the command subscriber without mutating the external ROS graph."""

        with self._lock:
            subscriber = self._command_subscriber
            self._command_subscriber = None
            self._started = False
        unregister = getattr(subscriber, "unregister", None)
        if callable(unregister):
            unregister()

    def reset(
        self,
        *,
        episode_id: str,
        target_context: Mapping[str, Any],
        private_instances: Mapping[str, Any],
        instance_aliases: Mapping[str, str] | None = None,
    ) -> None:
        """Reset public ROS state and replace the evaluator-private ID mapping.

        ``private_instances`` is never serialised.  Its keys must be the same
        opaque IDs that appear in :class:`RestrictedGTObservation` records.
        ``instance_aliases`` is an optional evaluator-private normalization
        map whose values must name one of those opaque IDs; aliases never reach
        the target or observation ROS topics.
        """

        self.start()
        normalized_episode_id = str(episode_id).strip()
        if not normalized_episode_id:
            raise RestrictedGTContractError("episode_id must be non-empty")
        normalized_target = dict(target_context)
        validate_public_target_context(normalized_target)
        if str(normalized_target.get("episode_id") or "") != normalized_episode_id:
            raise RestrictedGTContractError("target_context episode_id must match reset episode_id")
        normalized_instances: dict[str, Any] = {}
        for public_id, private_handle in private_instances.items():
            opaque_id = str(public_id).strip()
            if not opaque_id:
                raise RestrictedGTContractError("private_instances cannot contain an empty opaque id")
            normalized_instances[opaque_id] = private_handle
        normalized_aliases = {opaque_id: opaque_id for opaque_id in normalized_instances}
        for raw_alias, raw_canonical_id in dict(instance_aliases or {}).items():
            alias = str(raw_alias).strip()
            canonical_id = str(raw_canonical_id).strip()
            if not alias or not canonical_id:
                raise RestrictedGTContractError("instance_aliases cannot contain an empty ID")
            if canonical_id not in normalized_instances:
                raise RestrictedGTContractError(
                    "instance_aliases must resolve to a registered opaque ID"
                )
            existing = normalized_aliases.get(alias)
            if existing is not None and existing != canonical_id:
                raise RestrictedGTContractError(
                    "instance_aliases cannot map one public alias to multiple opaque IDs"
                )
            normalized_aliases[alias] = canonical_id
        with self._lock:
            self._episode_generation += 1
            episode_generation = self._episode_generation
            self._episode_id = normalized_episode_id
            self._private_instances = normalized_instances
            self._canonical_instance_ids = normalized_aliases
            self._pending_by_command_id.clear()
            self._pending_order.clear()
            self._seen_command_ids.clear()
            self._command_sequence = 0
            self._event_sequence = 0
            self._published_result_events = []
            self._legacy_consecutive_observations.clear()
            self._public_visible_box_frames.clear()
            # The mapping node resets on this empty episode marker.  Keeping
            # this publication under the episode lock prevents an in-flight
            # prior frame from repopulating the new episode's bbox cache.
            self._publish_semantic_minimal_observations_payload(
                {
                    "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
                    "episode_id": normalized_episode_id,
                    "episode_reset": True,
                    "capture_step": -1,
                    "stamp_sec": float(self._clock()),
                    "observations": [],
                },
                expected_generation=episode_generation,
            )
            self._publish(self._target_publisher, normalized_target)

    def publish_observations(
        self,
        observations: Sequence[RestrictedGTObservation],
        *,
        capture_step: int,
        stamp_sec: float | None = None,
    ) -> dict[str, Any]:
        """Publish a visible-instance frame and return its public payload."""

        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            episode_id = self._episode_id
            episode_generation = self._episode_generation
        semantic_payload = {
            "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
            "episode_id": episode_id,
            "episode_reset": False,
            "capture_step": int(capture_step),
            "stamp_sec": float(self._clock() if stamp_sec is None else stamp_sec),
            "observations": [
                observation.to_semantic_minimal_payload() for observation in observations
            ],
        }
        self._publish_semantic_minimal_observations_payload(
            semantic_payload,
            expected_generation=episode_generation,
        )
        return deepcopy(semantic_payload)

    def publish_semantic_minimal_perception_payload(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Publish one compact public frame to the semantic mapping stack."""

        validate_semantic_minimal_perception_payload(payload)
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(payload.get("episode_id") or "") != self._episode_id:
                raise RestrictedGTContractError(
                    "semantic minimal perception payload episode_id must match the active episode"
                )
            episode_generation = self._episode_generation
        self._publish_semantic_minimal_observations_payload(
            payload,
            expected_generation=episode_generation,
        )
        return deepcopy(dict(payload))

    def publish_strict_perception_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Receive a canonical strict frame and publish its semantic adapter.

        This is the entry point for future detector-backed perception.  The
        conversion adds only lossless field aliases required by the semantic
        minimal-GT input;
        :func:`validate_strict_perception_payload` rejects ``is_*``, joint,
        state, pose, relation and visibility fields before publication.
        """

        validate_strict_perception_payload(payload)
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(payload.get("episode_id") or "") != self._episode_id:
                raise RestrictedGTContractError(
                    "strict perception payload episode_id must match the active episode"
                )
            episode_generation = self._episode_generation
        adapted = adapt_strict_perception_payload_for_semantic_mapping(payload)
        self._publish_semantic_minimal_observations_payload(
            adapted,
            expected_generation=episode_generation,
        )
        return deepcopy(adapted)

    def publish_restricted_gt_frame(
        self,
        payload: Mapping[str, Any],
        *,
        capture_step: int | None = None,
        stamp_sec: float | None = None,
    ) -> dict[str, Any]:
        """Publish the canonical V3 restricted-GT payload to semantic mapping.

        The canonical payload is first audited by
        :mod:`restricted_gt_perception`; the wire message has exactly the
        semantic minimal-GT fields ``id/name/bbox_2d/mask_rle/box_3d`` per
        observation.  It never restores object source names, legacy aliases,
        decoded mask coordinates, articulation metadata, state, relations, or
        camera privilege.
        """

        # Convert and audit before acquiring the ROS-state lock, avoiding a
        # redundant outer payload audit on every frame.
        adapted = adapt_restricted_gt_frame_for_semantic_mapping(
            payload,
            capture_step=capture_step,
            stamp_sec=float(self._clock() if stamp_sec is None else stamp_sec),
        )
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(adapted["episode_id"]) != self._episode_id:
                raise RestrictedGTContractError(
                    "restricted-GT frame episode_id must match the active episode"
                )
            episode_generation = self._episode_generation
        self._publish_semantic_minimal_observations_payload(
            adapted,
            expected_generation=episode_generation,
        )
        return deepcopy(adapted)

    def pop_next_interaction_request(self) -> EvaluatorInteractionRequest | None:
        """Return one pending private request for evaluator-side force control."""

        with self._lock:
            while self._pending_order:
                command_id = self._pending_order.popleft()
                request = self._pending_by_command_id.get(command_id)
                if request is not None:
                    return request
        return None

    def complete_interaction(
        self,
        command_id: str,
        *,
        success: bool,
        status: str | None = None,
        reason: str | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a leak-safe high-level completion result for one request.

        ``outcome`` may be a full ordinary-interaction backend result.  It is
        projected through an explicit public allow-list: object/joint names,
        fractions, force traces, and evaluator recipe fields can never leave
        this adapter.  ``success`` must describe the physical public action
        postcondition; required-plan scoring remains evaluator-private.
        """

        with self._lock:
            request = self._pending_by_command_id.pop(str(command_id), None)
        if request is None:
            raise KeyError(f"No pending interaction command {command_id!r}")
        return self._emit_interaction_result(
            request,
            success=bool(success),
            status=status or ("SUCCEEDED" if success else "FAILED"),
            reason=reason,
            outcome=outcome,
        )

    def reject_interaction(
        self,
        command_id: str,
        *,
        reason: str = "rejected",
    ) -> dict[str, Any]:
        """Finish a pending request without exposing evaluator diagnostics."""

        with self._lock:
            request = self._pending_by_command_id.pop(str(command_id), None)
        if request is None:
            raise KeyError(f"No pending interaction command {command_id!r}")
        return self._emit_interaction_result(
            request,
            success=False,
            status="REJECTED",
            reason=str(reason),
            outcome=None,
        )

    def receive_interaction_command(
        self, payload: Mapping[str, Any]
    ) -> EvaluatorInteractionRequest | None:
        """Accept a ROS command without trusting its object/joint selectors.

        Only an opaque ID registered for the current episode can resolve to a
        private object.  Incoming ``joint_names``, force settings, or guessed
        source names are deliberately ignored and never reach the executor.
        """

        if not isinstance(payload, Mapping):
            raise RestrictedGTContractError("interaction command must be a mapping")
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before accepting interaction commands")
            self._command_sequence += 1
            command_id = str(payload.get("command_id") or "").strip()
            if not command_id:
                command_id = f"object_skill_{self._command_sequence:06d}"
            if command_id in self._seen_command_ids:
                return None
            self._seen_command_ids.add(command_id)
            instance_id = self._command_instance_id(payload)
            action = str(payload.get("action") or "open").strip().casefold()
            node_id = str(payload.get("node_id") or "")
            candidate_id = str(payload.get("candidate_id") or "")
            decision_id = str(payload.get("decision_id") or "")
            node_type = str(payload.get("node_type") or "").strip().casefold()
            (
                sequence_type,
                open_regions,
                drawer_container_bbox,
                drawer_container_capture_step,
                has_drawer_container_bbox,
            ) = self._drawer_scan_hint(payload)
            direct_bbox_drawer_scan = False
            unresolved_public_drawer_box = False
            public_command = sanitize_public_interaction_command(payload)
            if sequence_type in {"drawer_scan", "drawer_open"} and has_drawer_container_bbox:
                if (
                    drawer_container_bbox is None
                    or drawer_container_capture_step is None
                ):
                    unresolved_public_drawer_box = True
                else:
                    matched_instance_id = self._match_public_bbox_at_capture_step_locked(
                        drawer_container_bbox,
                        drawer_container_capture_step,
                    )
                    if matched_instance_id is None:
                        unresolved_public_drawer_box = True
                    else:
                        # The public box, not a method-provided object selector,
                        # determines the opaque object for this direct scan.
                        instance_id = matched_instance_id
                        direct_bbox_drawer_scan = True
            response_instance_id = instance_id
            # Commands may contain a semantic graph's public compatibility
            # alias.  Resolve it here, before the private-registry lookup, so
            # downstream evaluator scoring and force execution use the one
            # canonical opaque object identity.
            instance_id = self._canonical_instance_ids.get(instance_id, instance_id)
            private_handle = self._private_instances.get(instance_id)
            public_observation = self._latest_public_object_observation_locked(
                instance_id
            )
            episode_id = self._episode_id
            episode_generation = self._episode_generation
        # Match the ordinary force bridge's public action contract.  Generic
        # interactions and ``drawer_open`` use ``open``; the sealed
        # open-observe-close drawer macro may use the explicit ``scan`` action.
        if action != "open" and not (
            sequence_type == "drawer_scan" and action == "scan"
        ):
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                instance_id=instance_id,
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="unsupported_action",
                public_command=public_command,
            )
        if unresolved_public_drawer_box:
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                instance_id="",
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="unresolved_drawer_scan_target",
                public_command=public_command,
            )
        # Provenance is checked before capability lookup.  Otherwise a method
        # could enumerate opaque IDs and distinguish a hidden registered object
        # ("not visible") from an unregistered ID ("unknown instance").  Both
        # are deliberately indistinguishable until the evaluator has actually
        # published this object's public box in the current episode.
        if self.require_public_interaction_evidence and public_observation is None:
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                # Do not expose whether a guessed public alias canonicalized to
                # a private registered object before it had public provenance.
                instance_id=response_instance_id,
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="interaction_not_visible",
                public_command=public_command,
            )
        if private_handle is None:
            if instance_id and self._is_semantic_portal_command(
                node_type=node_type,
                node_id=node_id,
                candidate_id=candidate_id,
            ):
                # The semantic graph may infer a doorway from geometry even
                # though no public opaque object was registered for a force
                # interaction.  Do not send a false physical-failure signal:
                # queue an evaluator-only invalid attempt for the rollout to
                # score and trace on its next interaction poll.
                request = EvaluatorInteractionRequest(
                    command_id=command_id,
                    episode_id=episode_id,
                    instance_id=instance_id,
                    action=action,
                    private_handle=None,
                    node_id=node_id,
                    candidate_id=candidate_id,
                    decision_id=decision_id,
                    rejection_reason="unknown_instance_id",
                    public_command=public_command,
                )
                with self._lock:
                    if (
                        episode_generation != self._episode_generation
                        or episode_id != self._episode_id
                    ):
                        return None
                    self._pending_by_command_id[command_id] = request
                    self._pending_order.append(command_id)
                if self._interaction_executor is not None:
                    self.complete_interaction(
                        command_id,
                        success=False,
                        status="INVALID",
                        reason=request.rejection_reason,
                    )
                    return None
                return request
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                instance_id=instance_id,
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="unknown_instance_id",
                public_command=public_command,
            )
        request = EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id=episode_id,
            instance_id=instance_id,
            action=action,
            private_handle=private_handle,
            node_id=node_id,
            candidate_id=candidate_id,
            decision_id=decision_id,
            sequence_type=sequence_type,
            open_regions=open_regions,
            direct_bbox_drawer_scan=direct_bbox_drawer_scan,
            public_command=public_command,
            public_observation=public_observation or {},
        )
        with self._lock:
            if (
                episode_generation != self._episode_generation
                or episode_id != self._episode_id
            ):
                # A reset replaced the private registry after this command was
                # matched.  Do not allow an old request to enter the new
                # episode's evaluator queue.
                return None
            self._pending_by_command_id[command_id] = request
            self._pending_order.append(command_id)
        if self._interaction_executor is not None:
            try:
                outcome = self._interaction_executor(request)
                success = bool(outcome.get("success")) if isinstance(outcome, Mapping) else bool(outcome)
            except Exception:
                # The external method gets only a generic failure signal.  Full
                # exception details remain evaluator diagnostics.
                success = False
                outcome = None
            self.complete_interaction(
                command_id,
                success=success,
                outcome=outcome if isinstance(outcome, Mapping) else None,
            )
            return None
        return request

    def _interaction_command_callback(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise RestrictedGTContractError("interaction command must decode to an object")
            self.receive_interaction_command(payload)
        except (TypeError, ValueError, json.JSONDecodeError, RuntimeError):
            # A malformed command has no safe, routable command ID to respond
            # to.  Ignore it rather than leaking evaluator state on a log topic.
            return

    @staticmethod
    def _command_instance_id(payload: Mapping[str, Any]) -> str:
        # The semantic minimal-GT graph uses ``object_id`` for the same opaque
        # token that the evaluator registered at reset.  Legacy aliases remain
        # accepted for older standalone methods, but all values are resolved
        # only against the evaluator-private opaque-ID registry below.
        for key in ("object_id", "instance_id", "source_object_name", "target_instance_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _is_semantic_portal_command(
        *,
        node_type: str,
        node_id: str,
        candidate_id: str,
    ) -> bool:
        """Recognise a doorway request without treating arbitrary IDs as one."""

        if str(node_type).strip().casefold() == "portal":
            return True
        if str(node_id).strip().casefold().startswith("portal"):
            return True
        return str(candidate_id).strip().casefold().startswith("interaction:portal")

    def _match_public_bbox_at_capture_step_locked(
        self,
        requested_bbox: tuple[float, float, float, float],
        capture_step: int,
    ) -> str | None:
        """Route a box from one bounded evaluator-published public frame."""

        now = float(self._clock())
        fresh_frames = deque(
            (
                frame
                for frame in self._public_visible_box_frames
                if 0.0 <= now - frame.published_at_sec <= DIRECT_DRAWER_SCAN_BBOX_TTL_S
            ),
            maxlen=DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE,
        )
        self._public_visible_box_frames = fresh_frames
        matching_frames = [
            frame
            for frame in fresh_frames
            if int(frame.capture_step) == int(capture_step)
        ]
        if len(matching_frames) != 1:
            return None
        matches = {
            instance_id
            for instance_id, published_bbox in matching_frames[0].boxes
            if _bbox_iou_xyxy(requested_bbox, published_bbox)
            >= DIRECT_DRAWER_SCAN_BBOX_MIN_IOU
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def _latest_public_object_observation_locked(
        self,
        instance_id: str,
    ) -> dict[str, Any] | None:
        """Return recent geometry already exposed to the evaluated method."""

        now = float(self._clock())
        fresh_frames = deque(
            (
                frame
                for frame in self._public_visible_box_frames
                if 0.0
                <= now - frame.published_at_sec
                <= DIRECT_DRAWER_SCAN_BBOX_TTL_S
            ),
            maxlen=DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE,
        )
        self._public_visible_box_frames = fresh_frames
        for frame in reversed(fresh_frames):
            for object_id, center, size in frame.boxes_3d:
                if object_id == instance_id:
                    return {
                        "capture_step": int(frame.capture_step),
                        "age_seconds": max(
                            0.0,
                            now - float(frame.published_at_sec),
                        ),
                        "box_3d": {
                            "center": list(center),
                            "size": list(size),
                            "frame_id": "world",
                        },
                    }
        return None

    @staticmethod
    def _drawer_scan_hint(
        payload: Mapping[str, Any],
    ) -> tuple[
        str,
        tuple[tuple[float, float], ...],
        tuple[float, float, float, float] | None,
        int | None,
        bool,
    ]:
        """Keep only public visual drawer-sequence hints from a ROS command.

        A V3 method may request ``drawer_scan/scan`` or ``drawer_open/open``.
        The evaluator may use normalized image centers to choose private slide
        joints.  Guessed joint names, force settings, part IDs and other
        simulator metadata are intentionally not accepted here.  A
        ``drawer_container_bbox_2d`` plus its public capture step binds a direct
        drawer sequence to an object visible in a recent public frame.
        """

        sequence_type = str(payload.get("sequence_type") or "").strip().casefold()
        if sequence_type not in {"drawer_scan", "drawer_open"}:
            return "", (), None, None, False
        has_drawer_container_bbox = "drawer_container_bbox_2d" in payload
        drawer_container_bbox = _public_bbox_xyxy(
            payload.get("drawer_container_bbox_2d")
        )
        drawer_container_capture_step = _public_capture_step(
            payload.get("drawer_container_capture_step")
        )
        raw_regions = payload.get("open_regions")
        if not isinstance(raw_regions, list):
            return (
                sequence_type,
                (),
                drawer_container_bbox,
                drawer_container_capture_step,
                has_drawer_container_bbox,
            )
        regions: list[tuple[float, float]] = []
        for raw_region in raw_regions[:12]:
            if not isinstance(raw_region, Mapping):
                continue
            center = raw_region.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            try:
                x, y = float(center[0]), float(center[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            point = (x, y)
            if point not in regions:
                regions.append(point)
        regions.sort(key=lambda point: (point[1], point[0]))
        return (
            sequence_type,
            tuple(regions),
            drawer_container_bbox,
            drawer_container_capture_step,
            has_drawer_container_bbox,
        )

    def _reject_unresolved_command(
        self,
        *,
        command_id: str,
        episode_id: str,
        instance_id: str,
        action: str,
        node_id: str,
        candidate_id: str,
        decision_id: str,
        reason: str,
        public_command: Mapping[str, Any] | None = None,
    ) -> EvaluatorInteractionRequest | None:
        safe_command = dict(public_command or {})
        if not safe_command:
            safe_command = sanitize_public_interaction_command(
                {
                    "command_id": command_id,
                    "node_id": node_id,
                    "candidate_id": candidate_id,
                    "decision_id": decision_id,
                    "object_id": instance_id,
                    "action": action,
                }
            )
        request = EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id=episode_id,
            instance_id=instance_id,
            action=action,
            private_handle=None,
            node_id=node_id,
            candidate_id=candidate_id,
            decision_id=decision_id,
            rejection_reason=str(reason),
            public_command=safe_command,
        )
        if self.queue_rejected_interactions:
            with self._lock:
                if episode_id != self._episode_id:
                    return None
                self._pending_by_command_id[command_id] = request
                self._pending_order.append(command_id)
            return request
        provenance_failure = str(reason) == "interaction_not_visible"
        self._emit_interaction_result(
            request,
            success=False,
            status="REJECTED",
            reason=reason,
            outcome={
                "state": "unknown" if provenance_failure else "unavailable",
                "post_state": "unknown" if provenance_failure else "unavailable",
                "interaction_capability": (
                    "unknown" if provenance_failure else "unavailable"
                ),
                **({} if provenance_failure else {"interactable": False}),
                "retryable": provenance_failure,
                "failure_reason": reason,
                "verification_source": "executor_capability_check",
            },
        )
        return None

    def _emit_interaction_result(
        self,
        request: EvaluatorInteractionRequest,
        *,
        success: bool,
        status: str,
        reason: str | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._event_sequence += 1
            event_id = f"object_skill_{self._event_sequence:06d}"
        # Rebuild a compact ordinary-bridge-style outcome from a public
        # allow-list.  State/capability/retryability may cross this boundary;
        # simulator object names, joint names/values, force traces and hidden
        # oracle relevance never do.
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_id": event_id,
            "episode_id": request.episode_id,
            "command_id": request.command_id,
            "candidate_id": request.candidate_id,
            "decision_id": request.decision_id,
            "node_id": request.node_id,
            # ``object_id`` is the semantic graph's public opaque identifier;
            # it is copied from the command and never resolved back to a
            # simulator name on this ROS topic.
            "object_id": request.instance_id,
            "instance_id": request.instance_id,
            "action": request.action,
            "success": bool(success),
            "status": str(status),
            "source": "evaluator_object_skill",
            "stamp_sec": float(self._clock()),
        }
        public_outcome = sanitize_public_interaction_outcome(
            outcome,
            success=bool(success),
        )
        payload.update(public_outcome)
        normalized_reason = sanitize_public_failure_reason(reason)
        if normalized_reason:
            payload["reason"] = normalized_reason
            payload.setdefault("failure_reason", normalized_reason)
        public_command = dict(request.public_command or {})
        node_type = str(public_command.get("node_type") or "").strip().casefold()
        if node_type:
            payload["node_type"] = node_type
        if (
            str(status).strip().upper() == "INVALID"
            and normalized_reason == "unknown_instance_id"
            and self._is_semantic_portal_command(
                node_type=node_type,
                node_id=request.node_id,
                candidate_id=request.candidate_id,
            )
        ):
            # A missing articulation is ambiguous.  Reuse the ordinary bridge's
            # public two-factor aperture contract; never promote an unknown
            # semantic doorway to ``static`` from evaluator registration alone.
            portal_outcome = portal_capability_from_public_evidence(
                public_command.get("portal_aperture_observation")
            )
            payload.update(portal_outcome)
            if portal_outcome["state"] == "static_open":
                payload["success"] = True
                payload["status"] = "SUCCEEDED"
                payload.pop("reason", None)
                payload.pop("failure_reason", None)
            else:
                payload["success"] = False
                payload["status"] = "FAILED"
                payload["reason"] = str(portal_outcome["failure_reason"])
        self._publish(self._result_publisher, payload)
        with self._lock:
            self._published_result_events.append(deepcopy(payload))
        return payload

    def _publish_observations_payload(self, payload: Mapping[str, Any]) -> None:
        validate_restricted_perception_payload(payload)
        self._publish(self._observations_publisher, dict(payload))

    def _publish_semantic_minimal_observations_payload(
        self,
        payload: Mapping[str, Any],
        *,
        expected_generation: int | None = None,
    ) -> None:
        """Publish and atomically cache one public perception frame.

        The cache becomes eligible only after the ROS publication succeeds.
        Holding the episode lock across both operations means an interaction
        callback cannot route against a frame from a reset or half-published
        episode.
        """

        validate_semantic_minimal_perception_payload(payload)
        payload_episode_id = str(payload.get("episode_id") or "")
        is_reset = bool(payload.get("episode_reset", False))
        capture_step = _public_capture_step(payload.get("capture_step"))
        visible_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        visible_boxes_3d: list[
            tuple[str, tuple[float, float, float], tuple[float, float, float]]
        ] = []
        if not is_reset and capture_step is not None:
            for observation in payload.get("observations") or []:
                if not isinstance(observation, Mapping):
                    continue
                instance_id = str(observation.get("id") or "").strip()
                bbox = _public_bbox_xyxy(observation.get("bbox_2d"))
                if instance_id and bbox is not None:
                    visible_boxes.append((instance_id, bbox))
                box_3d = observation.get("box_3d")
                if instance_id and isinstance(box_3d, Mapping):
                    try:
                        center = tuple(
                            float(value)
                            for value in list(box_3d.get("center") or [])[:3]
                        )
                        size = tuple(
                            float(value)
                            for value in list(box_3d.get("size") or [])[:3]
                        )
                    except (TypeError, ValueError):
                        center = ()
                        size = ()
                    if (
                        len(center) == 3
                        and len(size) == 3
                        and all(math.isfinite(value) for value in (*center, *size))
                        and all(value >= 0.0 for value in size)
                    ):
                        visible_boxes_3d.append((instance_id, center, size))
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if payload_episode_id != self._episode_id:
                raise RestrictedGTContractError(
                    "semantic minimal perception payload episode_id must match the active episode"
                )
            if (
                expected_generation is not None
                and int(expected_generation) != self._episode_generation
            ):
                raise RestrictedGTContractError(
                    "active episode changed before publishing perception payload"
                )
            self._publish(self._observations_publisher, dict(payload))
            if is_reset:
                self._public_visible_box_frames.clear()
                return
            if capture_step is None:
                # The mapper may still consume a detector frame with a malformed
                # step, but that frame cannot serve as a direct-scan selector.
                return
            # A drawer macro can publish several physical views at the same
            # decision step.  Keep only the latest one for that public step so
            # duplicate boxes cannot make the route spuriously ambiguous.
            retained = [
                frame
                for frame in self._public_visible_box_frames
                if int(frame.capture_step) != capture_step
            ]
            retained.append(
                _PublicVisibleBoxFrame(
                    capture_step=capture_step,
                    published_at_sec=float(self._clock()),
                    boxes=tuple(visible_boxes),
                    boxes_3d=tuple(visible_boxes_3d),
                )
            )
            self._public_visible_box_frames = deque(
                retained[-DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE:],
                maxlen=DIRECT_DRAWER_SCAN_BBOX_HISTORY_SIZE,
            )

    def _publish(self, publisher: Any, payload: Mapping[str, Any]) -> None:
        if publisher is None or self._String is None:
            raise RuntimeError("start() must be called before publishing ROS messages")
        encoded = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        publisher.publish(self._String(data=encoded))
