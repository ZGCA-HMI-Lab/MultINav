"""Evaluator-owned restricted GT perception for InteractiveNav V3.

The simulator needs rich state to render masks and to execute an articulated
interaction.  A benchmark policy must *not* receive that state.  This module is
the narrow boundary between the two: it publishes only opaque instance IDs,
semantic class names, 2-D boxes/masks, and 3-D axis-aligned boxes.

``OpaqueEpisodeRegistry`` deliberately keeps the source-name mapping private to
the evaluator.  An interaction bridge can resolve an opaque ID through that
registry, while a ROS policy can never recover a MuJoCo body name, joint, range,
or state from a payload produced here.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


RESTRICTED_GT_PROTOCOL_VERSION = "interactive_nav_v3_restricted_gt_v1"

# Keep this list explicit.  Unknown fields are also rejected by the schema
# audit, but these names make accidental additions particularly easy to spot in
# a review or test failure.
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "source_object_name",
        "object_name",
        "raw_name",
        "object_id",
        "asset_id",
        "parent",
        "parent_source_name",
        "parent_chain",
        "children",
        "room",
        "support_below",
        "is_door",
        "is_movable_door",
        "is_receptacle",
        "is_articulable",
        "is_pickup_candidate",
        "interaction_approach_axis_xy",
        "interaction_reference_aabb_center",
        "joint_infos",
        "joint_name",
        "joint_names",
        "joint_index",
        "joint_type",
        "joint_range",
        "joint_value",
        "primary_joint_name",
        "open_fraction",
        "closed_value",
        "open_value",
        "position",
        "orientation",
        "quat",
        "camera_pose_world",
        "camera_position",
        "camera_forward",
        "camera_up",
        "fov_deg",
        "distance_m",
        "visible_pixels",
        "visible_fraction",
        "projected_bbox_2d",
        "confidence",
        "metadata",
        "performance",
    }
)

_ROOT_FIELDS = frozenset(
    {
        "protocol_version",
        "episode_id",
        "episode_reset",
        "frame_index",
        "observations",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"instance_id", "name", "bbox_2d_xyxy", "mask_rle", "bbox_3d"}
)
_MASK_RLE_FIELDS = frozenset({"size", "counts"})
_BOX_3D_FIELDS = frozenset({"center", "size", "frame_id"})


class ForbiddenField(ValueError):
    """The public restricted-GT protocol contains an illegal field or value."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = str(path)
        self.reason = str(reason)
        super().__init__(f"Restricted GT payload violation at {self.path}: {self.reason}")


def _json_number(value: Any, *, path: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ForbiddenField(path, "expected a finite number, got boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForbiddenField(path, "expected a finite number") from exc
    if not math.isfinite(number):
        raise ForbiddenField(path, "number must be finite")
    return number


def _integer(value: Any, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ForbiddenField(path, "expected integer, got boolean")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ForbiddenField(path, "expected integer") from exc
    if isinstance(value, float) and not float(value).is_integer():
        raise ForbiddenField(path, "expected integer")
    if minimum is not None and integer < minimum:
        raise ForbiddenField(path, f"must be >= {minimum}")
    return integer


def _triplet(values: Sequence[Any], *, path: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 3:
        raise ForbiddenField(path, "expected exactly three finite coordinates")
    return tuple(_json_number(value, path=f"{path}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


@dataclass(frozen=True)
class MaskRLE:
    """COCO-order binary-mask run-length encoding without an external dependency.

    The mask is flattened in column-major (Fortran/COCO) order.  ``counts``
    starts with a background run and always sums to ``height * width``.
    """

    size: tuple[int, int]  # (height, width)
    counts: tuple[int, ...]

    @classmethod
    def from_mask(cls, mask: np.ndarray | Sequence[Sequence[bool]]) -> "MaskRLE":
        array = np.asarray(mask, dtype=bool)
        if array.ndim != 2:
            raise ValueError("MaskRLE requires a two-dimensional binary mask")
        height, width = (int(array.shape[0]), int(array.shape[1]))
        flat = array.reshape(-1, order="F").astype(np.uint8, copy=False)
        if flat.size == 0:
            counts: tuple[int, ...] = (0,)
        else:
            transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
            starts = np.concatenate((np.asarray([0]), transitions))
            ends = np.concatenate((transitions, np.asarray([flat.size])))
            values = [int(end - start) for start, end in zip(starts, ends, strict=True)]
            if int(flat[0]) == 1:
                values.insert(0, 0)
            counts = tuple(values)
        return cls(size=(height, width), counts=counts)

    def decode(self) -> np.ndarray:
        height, width = self.size
        total = int(height) * int(width)
        flat = np.zeros(total, dtype=bool)
        cursor = 0
        value = False
        for index, count in enumerate(self.counts):
            run = _integer(count, path=f"mask_rle.counts[{index}]", minimum=0)
            if cursor + run > total:
                raise ForbiddenField("mask_rle.counts", "runs exceed mask area")
            if value and run:
                flat[cursor : cursor + run] = True
            cursor += run
            value = not value
        if cursor != total:
            raise ForbiddenField("mask_rle.counts", "runs do not cover mask area")
        return flat.reshape((height, width), order="F")

    def to_dict(self) -> dict[str, Any]:
        return {"size": [int(self.size[0]), int(self.size[1])], "counts": [int(value) for value in self.counts]}


@dataclass(frozen=True)
class BoundingBox3D:
    """Public world-frame AABB.  It intentionally omits orientation."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    frame_id: str = "world"

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": [float(value) for value in self.center],
            "size": [float(value) for value in self.size],
            "frame_id": str(self.frame_id),
        }


@dataclass(frozen=True)
class RestrictedObservation:
    """One policy-visible observation under the restricted GT contract."""

    instance_id: str
    name: str
    bbox_2d_xyxy: tuple[int, int, int, int]
    mask_rle: MaskRLE
    bbox_3d: BoundingBox3D

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "bbox_2d_xyxy": [int(value) for value in self.bbox_2d_xyxy],
            "mask_rle": self.mask_rle.to_dict(),
            "bbox_3d": self.bbox_3d.to_dict(),
        }


@dataclass(frozen=True)
class RestrictedPerceptionFrame:
    """The complete public payload for one perception frame."""

    episode_id: str
    episode_reset: bool
    frame_index: int
    observations: tuple[RestrictedObservation, ...]
    protocol_version: str = RESTRICTED_GT_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "episode_reset": bool(self.episode_reset),
            "frame_index": int(self.frame_index),
            "observations": [item.to_dict() for item in self.observations],
        }
        audit_restricted_gt_payload(payload)
        return payload


@dataclass(frozen=True)
class PrivateObjectSpec:
    """Evaluator-private object descriptor used to project a segmentation mask.

    ``source_name`` is intentionally private and must never be copied into a
    :class:`RestrictedObservation`.  Callers may provide dynamic AABB values for
    unit tests or precomputed candidates; live MuJoCo AABBs take precedence when
    a model/data/body triple is available.
    """

    source_name: str
    body_id: int | None = None
    semantic_category: str | None = None
    geom_ids: tuple[int, ...] = ()
    aabb_center: tuple[float, float, float] | None = None
    aabb_size: tuple[float, float, float] | None = None


def _bbox_area_xyxy(bbox: Sequence[int] | Sequence[float] | None) -> float:
    if bbox is None or len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0]) + 1.0) * max(
        0.0, float(bbox[3]) - float(bbox[1]) + 1.0
    )


def _bbox_short_side_pixels(bbox: Sequence[int] | Sequence[float] | None) -> int:
    """Return the shorter inclusive 2-D extent, or zero for an invalid box.

    A long one-pixel segmentation sliver is not a visually grounded object
    observation.  This is especially important for doors: a wall-edge sliver
    otherwise acquires a public portal identity and changes the navigation
    graph despite not being recognizable in the policy-visible RGB frame.
    """

    if bbox is None or len(bbox) < 4:
        return 0
    width = max(0, int(round(float(bbox[2]) - float(bbox[0]) + 1.0)))
    height = max(0, int(round(float(bbox[3]) - float(bbox[1]) + 1.0)))
    return min(width, height)


def _largest_connected_component(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, int, tuple[int, int, int, int]] | None:
    """Return one visually grounded 8-connected component for an object.

    A body/geom association can contain several disconnected image islands.
    Publishing their union as one target box makes the policy associate an
    object with unrelated furniture between those islands.  Retain only the
    dominant island, matching the realtime-GT observation contract.
    """

    if xs.size == 0 or ys.size == 0:
        return None
    min_x = int(np.min(xs))
    min_y = int(np.min(ys))
    max_x = int(np.max(xs))
    max_y = int(np.max(ys))
    cropped = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    cropped[ys - min_y, xs - min_x] = 1
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cropped, connectivity=8
    )
    if component_count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    # OpenCV assigns labels in raster order; np.argmax gives a stable tie-break.
    component_index = 1 + int(np.argmax(areas))
    component_pixels = int(stats[component_index, cv2.CC_STAT_AREA])
    left = int(stats[component_index, cv2.CC_STAT_LEFT])
    top = int(stats[component_index, cv2.CC_STAT_TOP])
    component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    component_mask = np.zeros((height, width), dtype=bool)
    local_ys, local_xs = np.nonzero(labels == component_index)
    component_mask[min_y + local_ys, min_x + local_xs] = True
    return (
        component_mask,
        component_pixels,
        (
            min_x + left,
            min_y + top,
            min_x + left + component_width - 1,
            min_y + top + component_height - 1,
        ),
    )


def _project_aabb_bbox(
    *,
    camera_position: Sequence[float],
    camera_forward: Sequence[float],
    camera_up: Sequence[float],
    fov_deg: float,
    image_width: int,
    image_height: int,
    center: Sequence[float],
    size: Sequence[float],
) -> tuple[float, float, float, float] | None:
    """Project a world AABB into the camera image for an extent sanity check."""

    if image_width <= 1 or image_height <= 1:
        return None
    forward = np.asarray(camera_forward, dtype=np.float64).copy()
    up = np.asarray(camera_up, dtype=np.float64).copy()
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up))
    if forward_norm <= 1e-8 or up_norm <= 1e-8:
        return None
    forward /= forward_norm
    up /= up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        return None
    right /= right_norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    half = 0.5 * np.abs(np.asarray(size, dtype=np.float64))
    offsets = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    relative = (
        np.asarray(center, dtype=np.float64)[None, :]
        + offsets
        - np.asarray(camera_position, dtype=np.float64)[None, :]
    )
    depth = relative @ forward
    valid = depth > 1e-3
    if int(np.count_nonzero(valid)) < 2:
        return None
    relative = relative[valid]
    depth = depth[valid]
    fov_rad = np.deg2rad(min(179.0, max(1.0, float(fov_deg))))
    focal = 0.5 * float(image_height) / max(np.tan(0.5 * fov_rad), 1e-6)
    xs = 0.5 * float(image_width - 1) + (relative @ right) / depth * focal
    ys = 0.5 * float(image_height - 1) - (relative @ up) / depth * focal
    min_x = max(0.0, float(np.min(xs)))
    min_y = max(0.0, float(np.min(ys)))
    max_x = min(float(image_width - 1), float(np.max(xs)))
    max_y = min(float(image_height - 1), float(np.max(ys)))
    if max_x < min_x or max_y < min_y:
        return None
    return min_x, min_y, max_x, max_y


class OpaqueEpisodeRegistry:
    """Private bidirectional mapping between source names and opaque IDs.

    Only evaluator-owned code should call :meth:`resolve_private_source_name`.
    The returned IDs are stable within one episode and never encode the raw
    source name, asset ID, or MuJoCo body index.
    """

    def __init__(
        self,
        *,
        instance_prefix: str = "obj",
        initial_episode_index: int = 1,
    ) -> None:
        prefix = re.sub(r"[^a-z0-9]", "", str(instance_prefix).casefold())
        if re.fullmatch(r"[a-z][a-z0-9]*", prefix) is None:
            raise ValueError("instance_prefix must start with an ASCII letter")
        if not 1 <= int(initial_episode_index) <= 999999:
            raise ValueError("initial_episode_index must be in [1, 999999]")
        self._instance_prefix = prefix
        self._episode_index = int(initial_episode_index) - 1
        self._episode_id = ""
        self._next_instance_index = 1
        self._source_to_id: dict[str, str] = {}
        self._id_to_source: dict[str, str] = {}
        self.reset()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def reset(self) -> str:
        """Begin a new opaque episode namespace and return its public ID."""

        self._episode_index += 1
        self._episode_id = f"episode_{self._episode_index:06d}"
        self._next_instance_index = 1
        self._source_to_id.clear()
        self._id_to_source.clear()
        return self._episode_id

    def public_id_for(self, source_name: str) -> str:
        """Return a stable episode-local opaque ID for a private source name."""

        source = str(source_name)
        if not source:
            raise ValueError("A private source name is required for an opaque ID")
        existing = self._source_to_id.get(source)
        if existing is not None:
            return existing
        instance_id = f"{self._instance_prefix}_{self._next_instance_index:06d}"
        self._next_instance_index += 1
        self._source_to_id[source] = instance_id
        self._id_to_source[instance_id] = source
        return instance_id

    def resolve_private_source_name(self, instance_id: str) -> str | None:
        """Resolve an opaque ID for the evaluator-side interaction executor."""

        return self._id_to_source.get(str(instance_id))

    def contains_public_id(self, instance_id: str) -> bool:
        return str(instance_id) in self._id_to_source


_CATEGORY_ALIASES = {
    "armchair": "chair",
    "bookcase": "bookshelf",
    "cabinet": "cabinet",
    "couch": "sofa",
    "cupboard": "cabinet",
    "dish washer": "dishwasher",
    "fridge": "refrigerator",
    "gate": "door",
    "microwave oven": "microwave",
    "night stand": "nightstand",
    "refrigerator": "refrigerator",
    "sliding door": "door",
    "television": "tv",
    "tv": "tv",
    "wardrobe": "closet",
}
_SOURCE_TOKEN_LABELS = (
    ("refrigerator", "refrigerator"),
    ("fridge", "refrigerator"),
    ("sliding door", "door"),
    ("door", "door"),
    ("drawer", "drawer"),
    ("dresser", "dresser"),
    ("cabinet", "cabinet"),
    ("closet", "closet"),
    ("wardrobe", "closet"),
    ("microwave", "microwave"),
    ("dishwasher", "dishwasher"),
    ("toilet", "toilet"),
    ("sink", "sink"),
    ("table", "table"),
    ("chair", "chair"),
    ("sofa", "sofa"),
    ("bed", "bed"),
)


def _clean_semantic_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null", "unknown", "n/a"}:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_|/\\-]+", " ", text.casefold())
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    return " ".join(text.split())


def normalize_semantic_category(category: Any, *, fallback_source_name: str | None = None) -> str:
    """Return a public semantic class without falling back to a raw source ID.

    A provided annotation category is public semantic metadata, so a cleaned
    generic label is retained.  If an annotation category is absent, the private
    source name is used only to recognize a small fixed ontology; arbitrary raw
    suffixes never escape and become ``"object"`` instead.
    """

    normalized = _clean_semantic_text(category)
    if normalized:
        return _CATEGORY_ALIASES.get(normalized, normalized)
    source = _clean_semantic_text(fallback_source_name)
    for token, label in _SOURCE_TOKEN_LABELS:
        if token in source:
            return label
    return "object"


def encode_binary_mask_rle(mask: np.ndarray | Sequence[Sequence[bool]]) -> dict[str, Any]:
    """Encode a two-dimensional binary segmentation mask as JSON-safe RLE."""

    return MaskRLE.from_mask(mask).to_dict()


def _coerce_binary_mask_rle(payload: Mapping[str, Any]) -> MaskRLE:
    """Validate an uncompressed public RLE without materialising its mask."""

    if not isinstance(payload, Mapping):
        raise ForbiddenField("mask_rle", "must be an object")
    _assert_fields(payload, _MASK_RLE_FIELDS, path="mask_rle")
    size = payload.get("size")
    counts = payload.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ForbiddenField("mask_rle.size", "expected [height, width]")
    if not isinstance(counts, (list, tuple)):
        raise ForbiddenField("mask_rle.counts", "expected an integer list")
    height = _integer(size[0], path="mask_rle.size[0]", minimum=0)
    width = _integer(size[1], path="mask_rle.size[1]", minimum=0)
    rle = MaskRLE(
        size=(height, width),
        counts=tuple(_integer(value, path=f"mask_rle.counts[{index}]", minimum=0) for index, value in enumerate(counts)),
    )
    if sum(rle.counts) != height * width:
        raise ForbiddenField("mask_rle.counts", "runs do not cover mask area")
    return rle


def binary_mask_rle_stats(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return ``(height, width, foreground_pixels)`` without decoding an RLE."""

    rle = _coerce_binary_mask_rle(payload)
    foreground_pixels = sum(rle.counts[1::2])
    return int(rle.size[0]), int(rle.size[1]), int(foreground_pixels)


def decode_binary_mask_rle(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode a :func:`encode_binary_mask_rle` payload (useful for tests/tools)."""

    return _coerce_binary_mask_rle(payload).decode()


def _assert_fields(value: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    for key in value:
        key_text = str(key)
        field_path = f"{path}.{key_text}" if path else key_text
        if key_text in FORBIDDEN_FIELD_NAMES:
            raise ForbiddenField(field_path, "field is prohibited by the restricted GT protocol")
        if key_text not in allowed:
            raise ForbiddenField(field_path, "field is not in the restricted GT allow-list")


def _audit_string(value: Any, *, path: str, pattern: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ForbiddenField(path, "expected a non-empty string")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise ForbiddenField(path, "does not use the required opaque identifier format")
    return value


def audit_restricted_gt_payload(
    payload: Mapping[str, Any],
    *,
    known_private_identifiers: Iterable[str] = (),
) -> None:
    """Validate that a JSON payload is exactly the restricted public schema.

    ``known_private_identifiers`` is optional test-time defence in depth.  It is
    intentionally an exact-value check (not substring matching) because labels
    such as ``door`` can legitimately equal a simulator source name.
    """

    if not isinstance(payload, Mapping):
        raise ForbiddenField("$", "payload must be an object")
    _assert_fields(payload, _ROOT_FIELDS, path="$")
    missing = _ROOT_FIELDS - set(payload)
    if missing:
        raise ForbiddenField("$", "missing required fields: " + ", ".join(sorted(missing)))
    if payload.get("protocol_version") != RESTRICTED_GT_PROTOCOL_VERSION:
        raise ForbiddenField("$.protocol_version", "unexpected protocol version")
    _audit_string(payload.get("episode_id"), path="$.episode_id", pattern=r"episode_[0-9]{6}")
    if not isinstance(payload.get("episode_reset"), (bool, np.bool_)):
        raise ForbiddenField("$.episode_reset", "expected boolean")
    _integer(payload.get("frame_index"), path="$.frame_index", minimum=0)
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ForbiddenField("$.observations", "expected a list")
    private_values = {str(value) for value in known_private_identifiers if str(value)}
    for index, observation in enumerate(observations):
        path = f"$.observations[{index}]"
        if not isinstance(observation, Mapping):
            raise ForbiddenField(path, "observation must be an object")
        _assert_fields(observation, _OBSERVATION_FIELDS, path=path)
        missing_observation = _OBSERVATION_FIELDS - set(observation)
        if missing_observation:
            raise ForbiddenField(path, "missing required fields: " + ", ".join(sorted(missing_observation)))
        instance_id = _audit_string(
            observation.get("instance_id"),
            path=f"{path}.instance_id",
            pattern=r"[a-z][a-z0-9]*_[0-9]{6}",
        )
        if instance_id in private_values:
            raise ForbiddenField(f"{path}.instance_id", "contains a private identifier")
        name = _audit_string(observation.get("name"), path=f"{path}.name")
        if name in private_values:
            raise ForbiddenField(f"{path}.name", "contains a private identifier")
        bbox = observation.get("bbox_2d_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ForbiddenField(f"{path}.bbox_2d_xyxy", "expected [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (
            _integer(value, path=f"{path}.bbox_2d_xyxy[{axis}]", minimum=0)
            for axis, value in enumerate(bbox)
        )
        if x1 < x0 or y1 < y0:
            raise ForbiddenField(f"{path}.bbox_2d_xyxy", "must have non-negative extent")
        mask = observation.get("mask_rle")
        if not isinstance(mask, Mapping):
            raise ForbiddenField(f"{path}.mask_rle", "must be an object")
        height, width, foreground_pixels = binary_mask_rle_stats(mask)
        if foreground_pixels <= 0:
            raise ForbiddenField(f"{path}.mask_rle", "must contain at least one foreground pixel")
        if x1 >= width or y1 >= height:
            raise ForbiddenField(f"{path}.bbox_2d_xyxy", "lies outside mask dimensions")
        box_3d = observation.get("bbox_3d")
        if not isinstance(box_3d, Mapping):
            raise ForbiddenField(f"{path}.bbox_3d", "must be an object")
        _assert_fields(box_3d, _BOX_3D_FIELDS, path=f"{path}.bbox_3d")
        if _BOX_3D_FIELDS - set(box_3d):
            raise ForbiddenField(f"{path}.bbox_3d", "missing required fields")
        _triplet(box_3d.get("center"), path=f"{path}.bbox_3d.center")
        size = _triplet(box_3d.get("size"), path=f"{path}.bbox_3d.size")
        if any(value < 0.0 for value in size):
            raise ForbiddenField(f"{path}.bbox_3d.size", "sizes must be non-negative")
        _audit_string(box_3d.get("frame_id"), path=f"{path}.bbox_3d.frame_id")


def _candidate_value(candidate: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(candidate, Mapping) and key in candidate:
            value = candidate[key]
        else:
            value = getattr(candidate, key, None)
        if value is not None and value != "":
            return value
    return None


def coerce_private_object_spec(candidate: PrivateObjectSpec | Mapping[str, Any] | Any) -> PrivateObjectSpec:
    """Normalize a private object/candidate record without serialising it."""

    if isinstance(candidate, PrivateObjectSpec):
        return candidate
    source = _candidate_value(candidate, "source_name", "source_object_name", "object_name", "name")
    if source is None:
        raise ValueError("Private object candidate requires source_name/object_name/name")
    body_id_value = _candidate_value(candidate, "body_id")
    geom_ids_value = _candidate_value(candidate, "geom_ids", "geometry_ids") or ()
    center = _candidate_value(candidate, "aabb_center", "bbox_3d_center", "center")
    size = _candidate_value(candidate, "aabb_size", "bbox_3d_size", "size")
    category = _candidate_value(candidate, "semantic_category", "semantic_name", "category", "object_category")
    return PrivateObjectSpec(
        source_name=str(source),
        body_id=None if body_id_value is None else int(body_id_value),
        semantic_category=None if category is None else str(category),
        geom_ids=tuple(int(value) for value in geom_ids_value),
        aabb_center=None if center is None else _triplet(center, path="private_candidate.aabb_center"),
        aabb_size=None if size is None else _triplet(size, path="private_candidate.aabb_size"),
    )


def _deduplicate_specs(candidates: Iterable[PrivateObjectSpec | Mapping[str, Any] | Any]) -> list[PrivateObjectSpec]:
    merged: dict[str, PrivateObjectSpec] = {}
    for candidate in candidates:
        spec = coerce_private_object_spec(candidate)
        existing = merged.get(spec.source_name)
        if existing is None:
            merged[spec.source_name] = spec
            continue
        # A catalog can contain one record per joint.  Keep the first body and
        # semantic metadata, while retaining any explicit geometry IDs.
        merged[spec.source_name] = PrivateObjectSpec(
            source_name=existing.source_name,
            body_id=existing.body_id if existing.body_id is not None else spec.body_id,
            semantic_category=existing.semantic_category or spec.semantic_category,
            geom_ids=tuple(sorted(set(existing.geom_ids) | set(spec.geom_ids))),
            aabb_center=existing.aabb_center or spec.aabb_center,
            aabb_size=existing.aabb_size or spec.aabb_size,
        )
    return [merged[key] for key in sorted(merged)]


def _model_body_id(model: Any, source_name: str) -> int | None:
    try:
        return int(model.body(source_name).id)
    except Exception:
        return None


def build_private_object_specs_from_env(env: Any) -> list[PrivateObjectSpec]:
    """Create evaluator-private render candidates from the current scene.

    Categories come from scene metadata/annotations.  Source names are used only
    to locate bodies and as a last-resort fixed-ontology hint in
    :func:`normalize_semantic_category`; they are never emitted.
    """

    model = getattr(env, "current_model", None)
    if model is None:
        return []
    metadata_by_name = dict((getattr(env, "current_scene_metadata", None) or {}).get("objects", {}) or {})
    object_manager = None
    try:
        object_manager = env.object_managers[env.current_batch_index]
    except Exception:
        pass
    names = {str(name) for name in metadata_by_name}
    if object_manager is not None:
        try:
            names.update(str(name) for name in object_manager.find_door_names())
        except Exception:
            pass
        try:
            names.update(str(getattr(item, "name", "")) for item in object_manager.list_top_level_objects())
        except Exception:
            pass
    specs: list[PrivateObjectSpec] = []
    for source_name in sorted(name for name in names if name):
        body_id = _model_body_id(model, source_name)
        if body_id is None:
            continue
        metadata = dict(metadata_by_name.get(source_name) or {})
        category = metadata.get("category")
        if not category and object_manager is not None:
            try:
                category = object_manager.get_annotation_category(source_name)
            except Exception:
                pass
        specs.append(
            PrivateObjectSpec(
                source_name=source_name,
                body_id=body_id,
                semantic_category=None if category is None else str(category),
            )
        )
    return specs


def _mujoco_geom_object_type() -> int:
    try:
        import mujoco

        return int(mujoco.mjtObj.mjOBJ_GEOM)
    except Exception as exc:  # pragma: no cover - only useful for dependency-light tooling
        raise RuntimeError("Pass geom_object_type when MuJoCo is unavailable") from exc


def _geom_to_spec_mapping(model: Any, specs: Sequence[PrivateObjectSpec]) -> np.ndarray:
    max_explicit_geom = max((geom_id for spec in specs for geom_id in spec.geom_ids), default=-1)
    n_geom = int(getattr(model, "ngeom", max_explicit_geom + 1)) if model is not None else max_explicit_geom + 1
    n_geom = max(n_geom, max_explicit_geom + 1, 0)
    mapping = np.full(n_geom, -1, dtype=np.int32)
    for spec_index, spec in enumerate(specs):
        for geom_id in spec.geom_ids:
            if 0 <= int(geom_id) < n_geom:
                mapping[int(geom_id)] = spec_index
    if model is None or not hasattr(model, "geom_bodyid") or not hasattr(model, "body_parentid"):
        return mapping
    body_to_spec = {int(spec.body_id): index for index, spec in enumerate(specs) if spec.body_id is not None}
    for geom_id in range(n_geom):
        if mapping[geom_id] >= 0:
            continue
        try:
            body_id = int(model.geom_bodyid[geom_id])
        except Exception:
            continue
        visited: set[int] = set()
        while body_id >= 0 and body_id not in visited:
            spec_index = body_to_spec.get(body_id)
            if spec_index is not None:
                mapping[geom_id] = int(spec_index)
                break
            visited.add(body_id)
            try:
                parent_id = int(model.body_parentid[body_id])
            except Exception:
                break
            if parent_id == body_id:
                break
            body_id = parent_id
    return mapping


def _runtime_aabb(spec: PrivateObjectSpec, model: Any, data: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if model is not None and data is not None and spec.body_id is not None:
        try:
            from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

            center, size = body_aabb(model, data, int(spec.body_id), visible_only=True)
            return _triplet(center, path="private_runtime.aabb_center"), _triplet(size, path="private_runtime.aabb_size")
        except Exception:
            try:
                center = _triplet(data.xpos[int(spec.body_id)], path="private_runtime.xpos")
                return center, (0.0, 0.0, 0.0)
            except Exception:
                pass
    return spec.aabb_center or (0.0, 0.0, 0.0), spec.aabb_size or (0.0, 0.0, 0.0)


def build_restricted_gt_frame(
    *,
    segmentation: np.ndarray,
    registry: OpaqueEpisodeRegistry,
    candidates: Iterable[PrivateObjectSpec | Mapping[str, Any] | Any] | None = None,
    specs: Iterable[PrivateObjectSpec | Mapping[str, Any] | Any] | None = None,
    model: Any = None,
    data: Any = None,
    frame_index: int = 0,
    episode_reset: bool = False,
    min_visible_pixels: int = 1,
    min_bbox_area_pixels: int = 1,
    min_bbox_short_side_pixels: int = 1,
    min_portal_bbox_short_side_pixels: int = 8,
    min_visible_fraction: float = 0.0,
    max_distance_m: float = 0.0,
    camera_position: Sequence[float] | None = None,
    camera_forward: Sequence[float] | None = None,
    camera_up: Sequence[float] | None = None,
    camera_fov_deg: float | None = None,
    frame_id: str = "world",
    geom_object_type: int | None = None,
    geom_to_spec: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one schema-audited public frame from a segmentation render.

    ``candidates`` and ``specs`` are aliases; the latter is convenient for a
    cached environment-derived list.  Both are evaluator-private input records.
    The returned dictionary is safe to JSON-publish to the navigation process.
    """

    if candidates is not None and specs is not None:
        raise ValueError("Pass only one of candidates or specs")
    private_specs = _deduplicate_specs(candidates if candidates is not None else (specs or ()))
    array = np.asarray(segmentation)
    if array.ndim < 3 or array.shape[-1] < 2:
        raise ValueError("Segmentation must have shape [height, width, >=2]")
    if int(min_visible_pixels) < 1:
        raise ValueError("min_visible_pixels must be >= 1")
    if int(min_bbox_area_pixels) < 1:
        raise ValueError("min_bbox_area_pixels must be >= 1")
    if int(min_bbox_short_side_pixels) < 1:
        raise ValueError("min_bbox_short_side_pixels must be >= 1")
    if int(min_portal_bbox_short_side_pixels) < int(min_bbox_short_side_pixels):
        raise ValueError(
            "min_portal_bbox_short_side_pixels must be >= min_bbox_short_side_pixels"
        )
    if (
        not math.isfinite(float(min_visible_fraction))
        or not 0.0 <= float(min_visible_fraction) <= 1.0
    ):
        raise ValueError("min_visible_fraction must be finite and in [0, 1]")
    if not math.isfinite(float(max_distance_m)) or float(max_distance_m) < 0.0:
        raise ValueError("max_distance_m must be finite and non-negative")
    if float(max_distance_m) > 0.0 and camera_position is None:
        raise ValueError("camera_position is required when max_distance_m is enabled")
    camera_xyz = (
        _triplet(camera_position, path="private_runtime.camera_position")
        if camera_position is not None
        else None
    )
    object_type = _mujoco_geom_object_type() if geom_object_type is None else int(geom_object_type)
    mapping = _geom_to_spec_mapping(model, private_specs) if geom_to_spec is None else np.asarray(geom_to_spec, dtype=np.int32)
    height, width = int(array.shape[0]), int(array.shape[1])
    if not private_specs or mapping.size == 0:
        return RestrictedPerceptionFrame(
            episode_id=registry.episode_id,
            episode_reset=bool(episode_reset),
            frame_index=_integer(frame_index, path="frame_index", minimum=0),
            observations=(),
        ).to_dict()

    geom_mask = array[..., 1] == object_type
    ys, xs = np.nonzero(geom_mask)
    observations: list[RestrictedObservation] = []
    if ys.size:
        geom_ids = array[..., 0][geom_mask].astype(np.int64, copy=False)
        valid = (geom_ids >= 0) & (geom_ids < mapping.size)
        ys, xs, geom_ids = ys[valid], xs[valid], geom_ids[valid]
        spec_indices = mapping[geom_ids] if geom_ids.size else np.empty(0, dtype=np.int32)
        valid_specs = spec_indices >= 0
        ys, xs, spec_indices = ys[valid_specs], xs[valid_specs], spec_indices[valid_specs]
        for spec_index, spec in enumerate(private_specs):
            selection = spec_indices == spec_index
            visible_count = int(np.count_nonzero(selection))
            if visible_count < int(min_visible_pixels):
                continue
            component = _largest_connected_component(
                xs[selection],
                ys[selection],
                height=height,
                width=width,
            )
            if component is None:
                continue
            mask, visible_count, bbox_2d = component
            if visible_count < int(min_visible_pixels):
                continue
            center, size = _runtime_aabb(spec, model, data)
            if _bbox_area_xyxy(bbox_2d) < int(min_bbox_area_pixels):
                continue
            semantic_name = normalize_semantic_category(
                spec.semantic_category,
                fallback_source_name=spec.source_name,
            )
            minimum_short_side = (
                int(min_portal_bbox_short_side_pixels)
                if semantic_name == "door"
                else int(min_bbox_short_side_pixels)
            )
            if _bbox_short_side_pixels(bbox_2d) < minimum_short_side:
                continue
            if (
                camera_xyz is not None
                and float(max_distance_m) > 0.0
                and math.dist(center, camera_xyz) > float(max_distance_m)
            ):
                continue
            if (
                float(min_visible_fraction) > 0.0
                and camera_xyz is not None
                and camera_forward is not None
                and camera_up is not None
                and camera_fov_deg is not None
            ):
                projected_bbox = _project_aabb_bbox(
                    camera_position=camera_xyz,
                    camera_forward=camera_forward,
                    camera_up=camera_up,
                    fov_deg=float(camera_fov_deg),
                    image_width=width,
                    image_height=height,
                    center=center,
                    size=size,
                )
                projected_area = _bbox_area_xyxy(projected_bbox)
                if projected_area > 1e-6:
                    observed_extent_fraction = min(
                        1.0,
                        _bbox_area_xyxy(bbox_2d) / projected_area,
                    )
                    if observed_extent_fraction < float(min_visible_fraction):
                        continue
            observations.append(
                RestrictedObservation(
                    instance_id=registry.public_id_for(spec.source_name),
                    name=semantic_name,
                    bbox_2d_xyxy=bbox_2d,
                    mask_rle=MaskRLE.from_mask(mask),
                    bbox_3d=BoundingBox3D(center=center, size=size, frame_id=str(frame_id)),
                )
            )
    return RestrictedPerceptionFrame(
        episode_id=registry.episode_id,
        episode_reset=bool(episode_reset),
        frame_index=_integer(frame_index, path="frame_index", minimum=0),
        observations=tuple(observations),
    ).to_dict()


class RestrictedGTPerceptionPublisher:
    """Build/publish restricted frames from the current evaluator environment.

    Passing ``rospy_module`` and ``string_message_type`` is optional.  Without
    them this remains a synchronous, dependency-light frame builder suitable for
    the benchmark runner and unit tests.
    """

    def __init__(
        self,
        *,
        camera_name: str = "head_camera",
        topic: str = "/semantic_mapping/gt_observations",
        min_visible_pixels: int = 16,
        min_bbox_area_pixels: int = 512,
        min_bbox_short_side_pixels: int = 1,
        min_portal_bbox_short_side_pixels: int = 8,
        min_visible_fraction: float = 0.2,
        max_distance_m: float = 4.0,
        step_interval: int = 1,
        frame_id: str = "world",
        rospy_module: Any | None = None,
        string_message_type: Any | None = None,
        queue_size: int = 1,
        initial_episode_index: int = 1,
    ) -> None:
        if int(min_visible_pixels) < 1:
            raise ValueError("min_visible_pixels must be >= 1")
        if int(min_bbox_area_pixels) < 1:
            raise ValueError("min_bbox_area_pixels must be >= 1")
        if int(min_bbox_short_side_pixels) < 1:
            raise ValueError("min_bbox_short_side_pixels must be >= 1")
        if int(min_portal_bbox_short_side_pixels) < int(min_bbox_short_side_pixels):
            raise ValueError(
                "min_portal_bbox_short_side_pixels must be >= min_bbox_short_side_pixels"
            )
        if (
            not math.isfinite(float(min_visible_fraction))
            or not 0.0 <= float(min_visible_fraction) <= 1.0
        ):
            raise ValueError("min_visible_fraction must be finite and in [0, 1]")
        if not math.isfinite(float(max_distance_m)) or float(max_distance_m) < 0.0:
            raise ValueError("max_distance_m must be finite and non-negative")
        if int(step_interval) < 1:
            raise ValueError("step_interval must be >= 1")
        if (rospy_module is None) != (string_message_type is None):
            raise ValueError("rospy_module and string_message_type must be supplied together")
        self.camera_name = str(camera_name)
        self.topic = str(topic)
        self.min_visible_pixels = int(min_visible_pixels)
        self.min_bbox_area_pixels = int(min_bbox_area_pixels)
        self.min_bbox_short_side_pixels = int(min_bbox_short_side_pixels)
        self.min_portal_bbox_short_side_pixels = int(
            min_portal_bbox_short_side_pixels
        )
        self.min_visible_fraction = float(min_visible_fraction)
        self.max_distance_m = float(max_distance_m)
        self.step_interval = int(step_interval)
        self.frame_id = str(frame_id)
        self.registry = OpaqueEpisodeRegistry(initial_episode_index=initial_episode_index)
        self.frame_index = 0
        self._episode_reset_pending = True
        self._cached_model_identity: int | None = None
        self._cached_specs: list[PrivateObjectSpec] = []
        self._cached_geom_to_spec = np.empty(0, dtype=np.int32)
        self._rospy = rospy_module
        self._String = string_message_type
        self.publisher = (
            rospy_module.Publisher(self.topic, string_message_type, queue_size=queue_size)
            if rospy_module is not None
            else None
        )

    @property
    def episode_id(self) -> str:
        return self.registry.episode_id

    def reset(self) -> str:
        """Forget all private mappings and begin an opaque episode namespace."""

        episode_id = self.registry.reset()
        self.frame_index = 0
        self._episode_reset_pending = True
        self._cached_model_identity = None
        self._cached_specs = []
        self._cached_geom_to_spec = np.empty(0, dtype=np.int32)
        return episode_id

    def resolve_private_source_name(self, instance_id: str) -> str | None:
        """Evaluator-only convenience for an opaque interaction command."""

        return self.registry.resolve_private_source_name(instance_id)

    def _ensure_cache(self, env: Any) -> None:
        model = getattr(env, "current_model", None)
        if model is None:
            raise ValueError("Environment has no current_model")
        if self._cached_model_identity == id(model):
            return
        self._cached_specs = build_private_object_specs_from_env(env)
        self._cached_geom_to_spec = _geom_to_spec_mapping(model, self._cached_specs)
        self._cached_model_identity = id(model)

    @staticmethod
    def _environment(value: Any) -> Any:
        return getattr(value, "env", value)

    def build(
        self,
        env_or_task: Any,
        *,
        segmentation: np.ndarray | None = None,
        candidates: Iterable[PrivateObjectSpec | Mapping[str, Any] | Any] | None = None,
        step_index: int | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Build a public frame, rendering segmentation only when required."""

        env = self._environment(env_or_task)
        capture_step = self.frame_index if step_index is None else int(step_index)
        if not force and not self._episode_reset_pending and capture_step % self.step_interval != 0:
            return None
        if segmentation is None:
            try:
                segmentation = np.asarray(env.render_segmentation_frame(self.camera_name))
            except Exception as exc:
                raise RuntimeError(f"Restricted GT segmentation render failed: {type(exc).__name__}: {exc}") from exc
        camera_position = None
        camera_forward = None
        camera_up = None
        camera_fov_deg = None
        if self.max_distance_m > 0.0 or self.min_visible_fraction > 0.0:
            try:
                camera = env.camera_manager.registry[self.camera_name]
                camera_position = _triplet(
                    camera.pos,
                    path="private_runtime.camera_position",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Restricted GT camera position is required for visibility filtering"
                ) from exc
            if self.min_visible_fraction > 0.0:
                try:
                    camera_forward = _triplet(
                        camera.forward,
                        path="private_runtime.camera_forward",
                    )
                    camera_up = _triplet(
                        camera.up,
                        path="private_runtime.camera_up",
                    )
                    camera_fov_deg = _json_number(
                        camera.fov,
                        path="private_runtime.camera_fov_deg",
                    )
                except (AttributeError, TypeError, ValueError):
                    # Some custom camera adapters expose a position but not a
                    # full projection contract.  Preserve their existing
                    # restricted-GT behavior rather than discarding all boxes.
                    camera_forward = None
                    camera_up = None
                    camera_fov_deg = None
        model = getattr(env, "current_model", None)
        data = getattr(env, "current_data", None)
        if candidates is None:
            self._ensure_cache(env)
            specs = self._cached_specs
            mapping = self._cached_geom_to_spec
        else:
            specs = _deduplicate_specs(candidates)
            mapping = _geom_to_spec_mapping(model, specs)
        payload = build_restricted_gt_frame(
            segmentation=segmentation,
            registry=self.registry,
            specs=specs,
            model=model,
            data=data,
            frame_index=self.frame_index,
            episode_reset=self._episode_reset_pending,
            min_visible_pixels=self.min_visible_pixels,
            min_bbox_area_pixels=self.min_bbox_area_pixels,
            min_bbox_short_side_pixels=self.min_bbox_short_side_pixels,
            min_portal_bbox_short_side_pixels=(
                self.min_portal_bbox_short_side_pixels
            ),
            min_visible_fraction=self.min_visible_fraction,
            max_distance_m=self.max_distance_m,
            camera_position=camera_position,
            camera_forward=camera_forward,
            camera_up=camera_up,
            camera_fov_deg=camera_fov_deg,
            frame_id=self.frame_id,
            geom_to_spec=mapping,
        )
        self.frame_index += 1
        self._episode_reset_pending = False
        return payload

    def publish(self, env_or_task: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Build a frame and publish it if a ROS publisher was configured."""

        payload = self.build(env_or_task, **kwargs)
        if payload is not None and self.publisher is not None:
            self.publisher.publish(self._String(data=json.dumps(payload, separators=(",", ":"))))
        return payload


__all__ = [
    "BoundingBox3D",
    "FORBIDDEN_FIELD_NAMES",
    "ForbiddenField",
    "MaskRLE",
    "OpaqueEpisodeRegistry",
    "PrivateObjectSpec",
    "RESTRICTED_GT_PROTOCOL_VERSION",
    "RestrictedGTPerceptionPublisher",
    "RestrictedObservation",
    "RestrictedPerceptionFrame",
    "audit_restricted_gt_payload",
    "binary_mask_rle_stats",
    "build_private_object_specs_from_env",
    "build_restricted_gt_frame",
    "coerce_private_object_spec",
    "decode_binary_mask_rle",
    "encode_binary_mask_rle",
    "normalize_semantic_category",
]
