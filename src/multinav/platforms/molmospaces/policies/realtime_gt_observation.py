from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mujoco
import numpy as np


def gather_joint_info(*args, **kwargs):
    from molmo_spaces.utils.articulation_utils import gather_joint_info as implementation

    return implementation(*args, **kwargs)


def body_aabb(*args, **kwargs):
    from molmo_spaces.utils.mj_model_and_data_utils import body_aabb as implementation

    return implementation(*args, **kwargs)


def _joint_type_name(joint_type: Any) -> str:
    text = str(joint_type).lower()
    if "hinge" in text:
        return "hinge"
    if "slide" in text:
        return "slide"
    try:
        numeric_type = int(np.asarray(joint_type).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return "none"
    if numeric_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        return "hinge"
    if numeric_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return "slide"
    return "none"


def _safe_body_aabb(model, data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a world-space AABB without allowing one bad filter to erase it.

    ``body_aabb`` uses ``visible_only`` (the utility API was renamed from the
    older ``visual_only`` spelling).  A visible-only body can legitimately have
    no qualifying geoms, so retry with the complete body hierarchy before
    reporting a zero-sized geometry.
    """

    def _valid(size: np.ndarray) -> bool:
        values = np.asarray(size, dtype=np.float64).reshape(-1)
        return values.size >= 3 and bool(np.all(np.isfinite(values[:3]))) and bool(
            np.all(values[:3] > 1e-6)
        )

    try:
        center, size = body_aabb(model, data, body_id, visible_only=True)
        if _valid(size):
            return center, size
        # Some assets put their collision geometry in groups excluded from the
        # visual filter.  Their full body AABB is still valid interaction
        # geometry and is preferable to silently publishing [0, 0, 0].
        full_center, full_size = body_aabb(model, data, body_id, visible_only=False)
        if _valid(full_size):
            return full_center, full_size
        return center, size
    except Exception:
        try:
            center, size = body_aabb(model, data, body_id, visible_only=False)
            if _valid(size):
                return center, size
        except Exception:
            pass
        return data.xpos[body_id].copy(), np.zeros(3, dtype=np.float64)


def _quat_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    return [float(qx), float(qy), float(qz), float(qw)]


def _joint_closed_open_values(joint_range: list[float]) -> tuple[float, float]:
    lower, upper = float(joint_range[0]), float(joint_range[1])
    closed = 0.0 if lower <= 0.0 <= upper else min((lower, upper), key=abs)
    opened = lower if abs(lower - closed) >= abs(upper - closed) else upper
    return closed, opened


def _joint_axis_world(model, data, joint_id: int) -> np.ndarray | None:
    if hasattr(data, "xaxis"):
        axis = np.asarray(data.xaxis[joint_id], dtype=np.float64)
    else:
        body_id = int(model.jnt_bodyid[joint_id])
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        axis = rotation @ np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return None
    return axis / norm


def _aligned_mean_axis(axes: list[np.ndarray]) -> list[float] | None:
    if not axes:
        return None
    reference = axes[0]
    aligned = [axis if float(np.dot(axis, reference)) >= 0.0 else -axis for axis in axes]
    mean_axis = np.mean(aligned, axis=0)
    norm = float(np.linalg.norm(mean_axis))
    if norm <= 1e-8:
        return [float(reference[0]), float(reference[1])]
    mean_axis /= norm
    return [float(mean_axis[0]), float(mean_axis[1])]


def _interaction_approach_axis_xy(model, data, joint_infos: list[dict[str, Any]]) -> list[float] | None:
    slide_axes: list[np.ndarray] = []
    hinge_axes: list[np.ndarray] = []
    for joint_info in joint_infos:
        joint_name = str(joint_info.get("joint_name") or "")
        joint_type = str(joint_info.get("joint_type") or "none")
        joint_range = list(joint_info.get("joint_range") or [0.0, 0.0])
        if not joint_name or joint_type not in {"hinge", "slide"} or len(joint_range) < 2:
            continue
        try:
            joint_id = int(model.joint(joint_name).id)
        except (KeyError, TypeError, ValueError):
            continue
        axis_world = _joint_axis_world(model, data, joint_id)
        if axis_world is None:
            continue
        closed_value, open_value = _joint_closed_open_values(joint_range)
        opening_delta = float(open_value - closed_value)
        if abs(opening_delta) <= 1e-8:
            continue
        opening_sign = 1.0 if opening_delta > 0.0 else -1.0
        if joint_type == "slide":
            axis_xy = opening_sign * axis_world[:2]
            norm = float(np.linalg.norm(axis_xy))
            if norm > 1e-6:
                slide_axes.append(axis_xy / norm)
            continue
        if not hasattr(data, "xanchor"):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_center, _ = _safe_body_aabb(model, data, body_id)
        radial = np.asarray(body_center, dtype=np.float64) - np.asarray(
            data.xanchor[joint_id], dtype=np.float64
        )
        current_value = float(joint_info.get("joint_value", closed_value) or 0.0)
        angle_to_closed = -(current_value - closed_value)
        cosine = float(np.cos(angle_to_closed))
        sine = float(np.sin(angle_to_closed))
        closed_radial = (
            radial * cosine
            + np.cross(axis_world, radial) * sine
            + axis_world * float(np.dot(axis_world, radial)) * (1.0 - cosine)
        )
        tangent_xy = np.cross(opening_sign * axis_world, closed_radial)[:2]
        norm = float(np.linalg.norm(tangent_xy))
        if norm > 1e-6:
            hinge_axes.append(tangent_xy / norm)
    # For appliances with both a hinged door and internal slide rails, the
    # hinged door motion is the operational front.  Using the slide direction
    # here makes the graph publish a tray/rail axis, which later produces an
    # apparently reachable but physically wrong refrigerator anchor.  Drawers
    # and slide-only containers retain their slide direction.
    return _aligned_mean_axis(hinge_axes or slide_axes)


def _bbox_area(bbox: list[float] | list[int]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0]) + 1.0) * max(
        0.0, float(bbox[3]) - float(bbox[1]) + 1.0
    )


def _bbox_short_side_pixels(bbox: list[float] | list[int]) -> int:
    """Return the shorter inclusive pixel extent of a 2-D bounding box.

    A one-pixel-wide segmentation sliver can have many visible pixels while
    still being visually unidentifiable.  In particular, treating such a
    sliver as a doorway leaks a simulator object behind an occluding wall into
    the public geometry-observation stream.  Area alone cannot catch that
    failure mode, so retain this small shape check next to the segmentation
    component logic.
    """

    if len(bbox) < 4:
        return 0
    width = max(0, int(round(float(bbox[2]) - float(bbox[0]) + 1.0)))
    height = max(0, int(round(float(bbox[3]) - float(bbox[1]) + 1.0)))
    return min(width, height)


def _largest_connected_component(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[int, list[int], np.ndarray, np.ndarray] | None:
    """Return the dominant 8-connected visible component for one object.

    MuJoCo's segmentation image is exact at the geom level, but an object can
    contribute several disconnected pixel islands after occlusion or an overly
    broad body-to-object association.  A union bounding box over those islands
    can cover unrelated furniture.  The public 2-D observation must represent
    one visually grounded target, so retain its largest visible component.
    """

    if xs.size == 0 or ys.size == 0:
        return None
    min_x = int(np.min(xs))
    min_y = int(np.min(ys))
    max_x = int(np.max(xs))
    max_y = int(np.max(ys))
    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    mask[ys - min_y, xs - min_x] = 1
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Stable tie-break: OpenCV labels components in raster order, so argmax
    # deterministically retains the upper-left component when areas tie.
    component_index = 1 + int(np.argmax(areas))
    left = int(stats[component_index, cv2.CC_STAT_LEFT])
    top = int(stats[component_index, cv2.CC_STAT_TOP])
    width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    component_ys, component_xs = np.nonzero(labels == component_index)
    return (
        int(stats[component_index, cv2.CC_STAT_AREA]),
        [min_x + left, min_y + top, min_x + left + width - 1, min_y + top + height - 1],
        component_xs.astype(np.int32, copy=False) + min_x,
        component_ys.astype(np.int32, copy=False) + min_y,
    )


def _largest_connected_component_bbox(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[int, list[int]] | None:
    component = _largest_connected_component(xs, ys)
    if component is None:
        return None
    return component[0], component[1]


def _encode_component_mask_rle(
    height: int,
    width: int,
    xs: np.ndarray,
    ys: np.ndarray,
) -> dict[str, list[int]]:
    """Encode public target pixels in compact COCO/Fortran run order."""

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    mask[ys, xs] = 1
    flat = mask.reshape(-1, order="F")
    if flat.size == 0:
        counts = [0]
    else:
        transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), transitions))
        ends = np.concatenate((transitions, np.asarray([flat.size])))
        counts = [int(end - start) for start, end in zip(starts, ends, strict=True)]
        if int(flat[0]) == 1:
            counts.insert(0, 0)
    return {"size": [int(height), int(width)], "counts": counts}


def _project_aabb_bbox(
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    image_size: list[int],
    center: np.ndarray,
    size: np.ndarray,
) -> list[float] | None:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 1 or height <= 1:
        return None
    forward = np.asarray(camera_forward, dtype=np.float64)
    up = np.asarray(camera_up, dtype=np.float64)
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
    relative = np.asarray(center, dtype=np.float64)[None, :] + offsets - np.asarray(
        camera_position, dtype=np.float64
    )[None, :]
    depth = relative @ forward
    valid = depth > 1e-3
    if int(np.count_nonzero(valid)) < 2:
        return None
    relative = relative[valid]
    depth = depth[valid]
    fov_rad = np.deg2rad(min(179.0, max(1.0, float(fov_deg))))
    focal = 0.5 * float(height) / max(np.tan(0.5 * fov_rad), 1e-6)
    xs = 0.5 * float(width - 1) + (relative @ right) / depth * focal
    ys = 0.5 * float(height - 1) - (relative @ up) / depth * focal
    min_x = max(0.0, float(np.min(xs)))
    min_y = max(0.0, float(np.min(ys)))
    max_x = min(float(width - 1), float(np.max(xs)))
    max_y = min(float(height - 1), float(np.max(ys)))
    if max_x < min_x or max_y < min_y:
        return None
    return [min_x, min_y, max_x, max_y]


def _visible_fraction(
    bbox_2d: list[int],
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    image_size: list[int],
    center: np.ndarray,
    size: np.ndarray,
    *,
    visible_pixels: int | None = None,
) -> tuple[float, list[float] | None]:
    projected_bbox = _project_aabb_bbox(
        camera_position,
        camera_forward,
        camera_up,
        fov_deg,
        image_size,
        center,
        size,
    )
    projected_area = _bbox_area(projected_bbox or [])
    if projected_area <= 1e-6:
        return 0.0, projected_bbox
    overlap = [
        max(float(bbox_2d[0]), projected_bbox[0]),
        max(float(bbox_2d[1]), projected_bbox[1]),
        min(float(bbox_2d[2]), projected_bbox[2]),
        min(float(bbox_2d[3]), projected_bbox[3]),
    ]
    visible_area = _bbox_area(overlap)
    if visible_pixels is not None:
        visible_area = min(visible_area, max(0, int(visible_pixels)))
    return min(1.0, visible_area / projected_area), projected_bbox


@dataclass
class _ObjectSpec:
    source_name: str
    metadata: dict[str, Any]
    body_id: int
    joint_names: tuple[str, ...]
    is_door: bool
    is_receptacle: bool
    is_articulable: bool
    is_pickup_candidate: bool
    parent_source_name: str = ""


class RealtimeGTObservationPublisher:
    def __init__(
        self,
        rospy_module,
        string_message_type,
        topic: str = "/semantic_mapping/gt_observations",
        camera_name: str = "head_camera",
        min_visible_pixels: int = 16,
        min_visible_bbox_short_side_px: int = 2,
        min_portal_bbox_short_side_px: int = 8,
        min_visible_fraction: float = 0.2,
        required_consecutive_observations: int = 2,
        max_distance_m: float = 4.0,
        step_interval: int = 3,
        emit_interaction_approach_axis: bool = False,
        queue_size: int = 1,
        async_processing: bool = True,
    ) -> None:
        self._rospy = rospy_module
        self._String = string_message_type
        self.topic = str(topic)
        self.camera_name = str(camera_name)
        self.min_visible_pixels = max(1, int(min_visible_pixels))
        self.min_visible_bbox_short_side_px = max(
            2, int(min_visible_bbox_short_side_px)
        )
        # Door/portal observations affect topology and planning, so require a
        # visibly two-dimensional component rather than accepting a long wall
        # edge.  Keep this stricter than the generic object gate by default.
        self.min_portal_bbox_short_side_px = max(
            self.min_visible_bbox_short_side_px,
            int(min_portal_bbox_short_side_px),
        )
        self.min_visible_fraction = min(1.0, max(0.0, float(min_visible_fraction)))
        self.required_consecutive_observations = max(
            1, int(required_consecutive_observations)
        )
        self.max_distance_m = max(0.0, float(max_distance_m))
        self.step_interval = max(1, int(step_interval))
        # Explicit rule-oracle mode for evaluating a dynamically derived joint
        # frame.  Detector and MLLM lanes must keep this disabled.
        self.emit_interaction_approach_axis = bool(emit_interaction_approach_axis)
        self.publisher = self._rospy.Publisher(self.topic, self._String, queue_size=queue_size)
        self.episode_index = 0
        self.episode_id = ""
        self.frame_index = 0
        self.next_instance_index = 1
        # Door body/source names are simulator-private.  The restricted GT
        # stream exposes a stable opaque portal reference instead, while the
        # simulator-side interaction bridge can resolve it through
        # ``resolve_public_object_id`` without feeding that identity back into
        # mapping, planning, rendering, or an MLLM prompt.
        self.instance_ids: dict[str, str] = {}
        self._public_to_source_ids: dict[str, str] = {}
        self._episode_reset_pending = True
        self._cache_model_identity: int | None = None
        self._specs: list[_ObjectSpec] = []
        self._geom_to_spec = np.empty(0, dtype=np.int32)
        # Per-publish diagnostic consumed by the bridge timing record.  It is
        # intentionally scalar-only; the raw segmentation never leaves this
        # simulator-local path.
        self.last_snapshot_used = False
        self._async_processing = bool(async_processing)
        self._publish_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None
        self.dropped_payload_count = 0
        if self._async_processing:
            self._worker = threading.Thread(target=self._publish_worker, name="realtime-gt-publisher", daemon=True)
            self._worker.start()

    def reset(self) -> None:
        self.episode_index += 1
        self.episode_id = f"episode_{self.episode_index:06d}"
        self.frame_index = 0
        self.next_instance_index = 1
        self.instance_ids.clear()
        self._public_to_source_ids.clear()
        self._episode_reset_pending = True
        self._cache_model_identity = None
        self._specs = []
        self._geom_to_spec = np.empty(0, dtype=np.int32)
        self.last_snapshot_used = False
        self._clear_queue()

    def should_publish_step(self, step_index: int) -> bool:
        """Whether a normal (non-force) publish is due for this policy step."""
        return bool(
            self._episode_reset_pending
            or int(step_index) % self.step_interval == 0
        )

    def close(self) -> None:
        if self._worker is None:
            return
        self._stop_event.set()
        self._clear_queue()
        try:
            self._publish_queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)
        self._worker = None

    def publish(self, task, stamp=None, step_index: int | None = None, force: bool = False) -> dict[str, Any] | None:
        self.last_snapshot_used = False
        if task is None or getattr(task, "env", None) is None:
            return None
        capture_step = int(self.frame_index if step_index is None else step_index)
        if not force and not self.should_publish_step(capture_step):
            return None
        env = task.env
        if self.camera_name not in env.camera_manager.registry:
            self._rospy.logwarn_throttle(2.0, "RealtimeGTObservationPublisher: camera %s not found", self.camera_name)
            return None
        self._ensure_cache(env)
        segmentation = None
        if force:
            # Interaction paths may have changed qpos outside task.step.  A
            # forced post-interaction observation must always be a new render,
            # never a task snapshot from the pre-interaction state.
            invalidate_snapshot = getattr(
                task, "invalidate_private_realtime_gt_segmentation_snapshot", None
            )
            if callable(invalidate_snapshot):
                invalidate_snapshot()
        else:
            get_snapshot = getattr(task, "get_private_realtime_gt_segmentation_snapshot", None)
            if callable(get_snapshot):
                try:
                    segmentation = get_snapshot(self.camera_name)
                    self.last_snapshot_used = segmentation is not None
                except Exception:
                    # Snapshot reuse is an optional local optimization.  Never
                    # let it change the GT publisher's existing render path.
                    segmentation = None
                    self.last_snapshot_used = False
        if segmentation is None:
            try:
                segmentation = np.asarray(env.render_segmentation_frame(self.camera_name))[..., :2]
            except Exception as exc:
                self._rospy.logwarn_throttle(2.0, "Realtime GT segmentation failed: %s", exc)
                return None
        visible = self._visible_instances(segmentation)
        model = env.current_model
        data = env.current_data
        camera = env.camera_manager.registry[self.camera_name]
        camera_position = np.asarray(camera.pos, dtype=np.float64).copy()
        camera_forward = np.asarray(camera.forward, dtype=np.float64).copy()
        image_size = [int(segmentation.shape[1]), int(segmentation.shape[0])]
        observations = []
        for spec_index, visible_pixels, bbox_2d, mask_rle in visible:
            spec = self._specs[spec_index]
            position = np.asarray(data.xpos[spec.body_id], dtype=np.float64).copy()
            distance_m = float(np.linalg.norm(position - camera_position))
            if self.max_distance_m > 0.0 and distance_m > self.max_distance_m:
                continue
            center, size = _safe_body_aabb(model, data, spec.body_id)
            if self.min_visible_fraction > 0.0:
                try:
                    observed_extent_fraction, projected_bbox = _visible_fraction(
                        bbox_2d,
                        camera_position,
                        np.asarray(camera.forward, dtype=np.float64),
                        np.asarray(camera.up, dtype=np.float64),
                        float(camera.fov),
                        image_size,
                        center,
                        size,
                        # Door frames surround empty apertures; mask fill is
                        # not their observed extent. Resolved-component pixel
                        # and width checks above still reject tiny fragments.
                        visible_pixels=None if spec.is_door else visible_pixels,
                    )
                except (AttributeError, TypeError, ValueError):
                    # Keep the geometry observation usable for custom cameras
                    # that cannot provide an AABB projection contract.
                    projected_bbox = None
                    observed_extent_fraction = 1.0
                if (
                    projected_bbox is not None
                    and observed_extent_fraction < self.min_visible_fraction
                ):
                    continue
            interaction_approach_axis_xy = None
            if (
                self.emit_interaction_approach_axis
                and spec.is_articulable
                and not spec.is_door
            ):
                try:
                    interaction_approach_axis_xy = _interaction_approach_axis_xy(
                        model,
                        data,
                        self._joint_infos(model, data, spec.joint_names),
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    interaction_approach_axis_xy = None
            observations.append(
                self._build_observation(
                    spec,
                    bbox_2d,
                    visible_pixels,
                    center,
                    size,
                    mask_rle=mask_rle,
                    interaction_approach_axis_xy=interaction_approach_axis_xy,
                )
            )
        capture_stamp_sec = (
            float(stamp.to_sec()) if stamp is not None and hasattr(stamp, "to_sec") else time.time()
        )
        payload = {
            "episode_id": self.episode_id,
            "episode_reset": bool(self._episode_reset_pending),
            "frame_index": int(self.frame_index),
            "capture_step": capture_step,
            "camera_name": self.camera_name,
            "stamp_sec": capture_stamp_sec,
            "capture_stamp_sec": capture_stamp_sec,
            "observation_pose_xyyaw": [
                float(camera_position[0]),
                float(camera_position[1]),
                float(np.arctan2(camera_forward[1], camera_forward[0])),
            ],
            # Public telemetry names the observation contract rather than the
            # evaluator implementation.  The mapping callback already knows
            # this arrives on its dedicated realtime-GT subscription.
            "source_mode": "geometry_observation",
            "observation_performed": True,
            "image_size": image_size,
            "observations": observations,
        }
        # Preserve the exact source ROS identity; capture_stamp_sec remains
        # the float wall clock used by latency accounting.
        if stamp is not None and hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
            payload["stamp_sec"] = int(stamp.secs)
            payload["stamp_nsec"] = int(stamp.nsecs)
        self._submit_payload(payload)
        self._episode_reset_pending = False
        self.frame_index += 1
        return payload

    def _ensure_cache(self, env) -> None:
        model = env.current_model
        if self._cache_model_identity == id(model):
            return
        object_manager = env.object_managers[env.current_batch_index]
        objects_meta = dict((env.current_scene_metadata or {}).get("objects", {}) or {})
        entries: list[tuple[str, dict[str, Any], bool]] = []
        seen = set()
        for source_name, metadata in objects_meta.items():
            try:
                model.body(source_name)
            except KeyError:
                continue
            entries.append((str(source_name), dict(metadata or {}), False))
            seen.add(str(source_name))
        try:
            door_names = object_manager.find_door_names()
        except Exception:
            door_names = []
        for door_name in door_names:
            if str(door_name) not in seen:
                entries.append((str(door_name), {"category": "Door", "object_id": door_name}, True))

        specs = []
        body_to_spec = {}
        for source_name, metadata, force_door in entries:
            body_id = int(model.body(source_name).id)
            category = "Door" if force_door else metadata.get("category") or source_name
            joint_names = self._joint_names(model, body_id, metadata)
            is_door = bool(force_door or "door" in str(category).lower() or "door" in source_name.lower())
            try:
                is_receptacle = bool(object_manager.has_receptacle_site(source_name))
            except Exception:
                is_receptacle = False
            try:
                is_pickup_candidate = bool(object_manager.has_free_joint(source_name))
            except Exception:
                is_pickup_candidate = False
            try:
                is_articulable = bool(object_manager.is_object_articulable(source_name))
            except Exception:
                is_articulable = bool(joint_names)
            body_to_spec[body_id] = len(specs)
            specs.append(
                _ObjectSpec(
                    source_name=source_name,
                    metadata=metadata,
                    body_id=body_id,
                    joint_names=joint_names,
                    is_door=is_door,
                    is_receptacle=is_receptacle,
                    is_articulable=is_articulable,
                    is_pickup_candidate=is_pickup_candidate,
                )
            )

        geom_to_spec = np.full(int(model.ngeom), -1, dtype=np.int32)
        canonical_door_specs = self._canonical_door_root_specs(model, specs)
        door_body_to_spec = {}
        for door_name in door_names:
            try:
                door_body_id = int(model.body(door_name).id)
            except KeyError:
                continue
            root_id = int(model.body_rootid[door_body_id])
            spec_index = canonical_door_specs.get(root_id)
            if spec_index is None:
                spec_index = body_to_spec.get(door_body_id)
            if spec_index is not None:
                door_body_to_spec[door_body_id] = int(spec_index)
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            door_spec = self._door_spec_for_body(
                model, body_id, door_body_to_spec
            )
            if door_spec is not None:
                geom_to_spec[geom_id] = door_spec
                continue
            while body_id >= 0:
                spec_index = body_to_spec.get(body_id)
                if spec_index is not None:
                    geom_to_spec[geom_id] = spec_index
                    break
                parent_id = int(model.body_parentid[body_id])
                if parent_id == body_id:
                    break
                body_id = parent_id
        self._specs = specs
        body_to_source = {spec.body_id: spec.source_name for spec in specs}
        for spec in self._specs:
            parent_body_id = int(model.body_parentid[spec.body_id])
            visited = {spec.body_id}
            while parent_body_id >= 0 and parent_body_id not in visited:
                visited.add(parent_body_id)
                parent_source_name = body_to_source.get(parent_body_id)
                if parent_source_name:
                    spec.parent_source_name = parent_source_name
                    break
                next_parent_body_id = int(model.body_parentid[parent_body_id])
                if next_parent_body_id == parent_body_id:
                    break
                parent_body_id = next_parent_body_id
        self._geom_to_spec = geom_to_spec
        self._cache_model_identity = id(model)

    @staticmethod
    def _canonical_door_root_specs(model, specs: list[_ObjectSpec]) -> dict[int, int]:
        """Map an articulated doorway root to its single root-level GT spec."""
        result = {}
        for spec_index, spec in enumerate(specs):
            root_id = int(model.body_rootid[spec.body_id])
            if (
                spec.is_door
                and spec.is_articulable
                and int(spec.body_id) == root_id
            ):
                result[root_id] = int(spec_index)
        return result

    @staticmethod
    def _door_spec_for_body(model, body_id: int, door_body_to_spec: dict[int, int]):
        """Return the door spec only for an explicit door body hierarchy.

        A MuJoCo articulated root can contain non-door siblings.  Mapping every
        geom sharing that root to the door inflated door masks across rooms.
        """
        current = int(body_id)
        visited = set()
        while current >= 0 and current not in visited:
            if current in door_body_to_spec:
                return int(door_body_to_spec[current])
            visited.add(current)
            parent = int(model.body_parentid[current])
            if parent == current:
                break
            current = parent
        return None

    def _visible_instances(
        self, segmentation: np.ndarray
    ) -> list[tuple[int, int, list[int], dict[str, list[int]]]]:
        if not self._specs or self._geom_to_spec.size == 0:
            return []
        geom_mask = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ys, xs = np.nonzero(geom_mask)
        if ys.size == 0:
            return []
        geom_ids = segmentation[..., 0][geom_mask].astype(np.int64, copy=False)
        valid_geom = (geom_ids >= 0) & (geom_ids < self._geom_to_spec.size)
        ys = ys[valid_geom]
        xs = xs[valid_geom]
        spec_indices = self._geom_to_spec[geom_ids[valid_geom]]
        valid_spec = spec_indices >= 0
        ys = ys[valid_spec]
        xs = xs[valid_spec]
        spec_indices = spec_indices[valid_spec]
        if spec_indices.size == 0:
            return []
        counts = np.bincount(spec_indices, minlength=len(self._specs))
        result = []
        for spec_index in np.flatnonzero(counts >= self.min_visible_pixels):
            component = _largest_connected_component(
                xs[spec_indices == spec_index],
                ys[spec_indices == spec_index],
            )
            if component is None:
                continue
            component_pixels, bbox_2d, component_xs, component_ys = component
            # Do not combine two individually invisible islands merely because
            # they share a GT object ID.  This prevents a tiny pair of door
            # fragments from becoming a large box over unrelated objects.
            if component_pixels < self.min_visible_pixels:
                continue
            spec = self._specs[int(spec_index)]
            minimum_short_side = (
                self.min_portal_bbox_short_side_px
                if bool(getattr(spec, "is_door", False))
                else self.min_visible_bbox_short_side_px
            )
            if _bbox_short_side_pixels(bbox_2d) < minimum_short_side:
                continue
            long_side = max(
                bbox_2d[2] - bbox_2d[0] + 1,
                bbox_2d[3] - bbox_2d[1] + 1,
            )
            # A diagonal or sparse edge can have a large bbox but no resolved
            # surface. Require actual mask area to support its short extent.
            if component_pixels < max(2, minimum_short_side) * long_side:
                continue
            result.append(
                (
                    int(spec_index),
                    int(component_pixels),
                    bbox_2d,
                    _encode_component_mask_rle(
                        int(segmentation.shape[0]),
                        int(segmentation.shape[1]),
                        component_xs,
                        component_ys,
                    ),
                )
            )
        return result

    def _build_observation(
        self,
        spec: _ObjectSpec,
        bbox_2d: list[int],
        visible_pixels: int,
        center: np.ndarray,
        size: np.ndarray,
        mask_rle: dict[str, list[int]] | None = None,
        interaction_approach_axis_xy: list[float] | None = None,
    ) -> dict[str, Any]:
        metadata = spec.metadata
        if spec.is_door:
            # ``doorframe``, ``doorway`` and door-leaf body names are private
            # simulator annotations.  Publish one generic door reference for
            # the whole canonical doorway; the mapping side keeps its internal
            # ``portal`` topology type without exposing a subtype.
            public_id = (
                self._public_instance_id(spec.source_name, prefix="door")
                if self is not None
                else "door_0001"
            )
            category = public_id
        else:
            category = metadata.get("category") or spec.source_name
            public_id = spec.source_name
        observation = {
            "id": public_id,
            "name": str(category),
            "bbox_2d": list(bbox_2d),
            "visible_pixels": max(0, int(visible_pixels)),
            "visible_fraction": min(
                1.0,
                float(max(0, int(visible_pixels)))
                / max(
                    1.0,
                    float(bbox_2d[2] - bbox_2d[0] + 1)
                    * float(bbox_2d[3] - bbox_2d[1] + 1),
                ),
            ),
            "box_3d": {
                "center": [float(value) for value in center],
                "size": [float(value) for value in size],
                "frame_id": "world",
            },
        }
        if mask_rle is not None:
            observation["mask_rle"] = {
                "size": [int(value) for value in mask_rle.get("size", [])],
                "counts": [int(value) for value in mask_rle.get("counts", [])],
            }
        if interaction_approach_axis_xy is not None:
            observation["interaction_approach_axis_xy"] = [
                float(value) for value in interaction_approach_axis_xy[:2]
            ]
            observation["oracle_rule_gt_interaction_axis"] = True
            observation["interaction_approach_axis_source"] = (
                "rule_oracle_gt_joint_geometry"
            )
        return observation

    def _public_instance_id(self, source_name: str, *, prefix: str) -> str:
        """Return a stable episode-local public identifier for a private body.

        The mapping/MLLM side only needs identity continuity.  It must not
        receive the simulator object's descriptive name in order to recognize
        a portal or determine whether it can be opened.
        """

        source_name = str(source_name or "")
        existing = self.instance_ids.get(source_name)
        if existing:
            return existing
        public_id = f"{prefix}_{self.next_instance_index:04d}"
        self.next_instance_index += 1
        self.instance_ids[source_name] = public_id
        self._public_to_source_ids[public_id] = source_name
        return public_id

    def resolve_public_object_id(self, object_id: str) -> str:
        """Resolve an opaque portal ID only inside the simulator process."""

        public_id = str(object_id or "")
        return self._public_to_source_ids.get(public_id, public_id)

    @staticmethod
    def _joint_names(model, body_id: int, metadata: dict[str, Any]) -> tuple[str, ...]:
        names = set((metadata.get("name_map", {}).get("joints", {}) or {}).keys())
        root_id = int(model.body_rootid[body_id])
        for joint_id in range(int(model.njnt)):
            joint_body_id = int(model.jnt_bodyid[joint_id])
            if int(model.body_rootid[joint_body_id]) == root_id:
                joint_name = model.joint(joint_id).name
                if joint_name:
                    names.add(str(joint_name))
        return tuple(sorted(names))

    @staticmethod
    def _joint_infos(model, data, joint_names: tuple[str, ...]) -> list[dict[str, Any]]:
        result = []
        for joint_name in joint_names:
            try:
                info = gather_joint_info(model, data, joint_name)
            except Exception:
                continue
            joint_type = _joint_type_name(info.get("joint_type"))
            if joint_type not in {"hinge", "slide"}:
                continue
            result.append(
                {
                    "joint_name": joint_name,
                    "joint_type": joint_type,
                    "joint_range": [float(value) for value in info.get("joint_range", [0.0, 0.0])],
                    "joint_value": float(info.get("joint_pos", 0.0)),
                }
            )
        return result

    def _submit_payload(self, payload: dict[str, Any]) -> None:
        if not self._async_processing:
            self._publish_payload(payload)
            return
        try:
            self._publish_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._publish_queue.get_nowait()
                self.dropped_payload_count += 1
            except queue.Empty:
                pass
            self._publish_queue.put_nowait(payload)

    def _publish_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._publish_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            self._publish_payload(payload)

    def _publish_payload(self, payload: dict[str, Any]) -> None:
        publish_t0 = time.perf_counter()
        payload["publish_stamp_sec"] = time.time()
        payload["processing_latency_ms"] = max(
            0.0, (payload["publish_stamp_sec"] - float(payload["capture_stamp_sec"])) * 1000.0
        )
        self.publisher.publish(self._String(data=json.dumps(payload, separators=(",", ":"))))

    def _clear_queue(self) -> None:
        while True:
            try:
                self._publish_queue.get_nowait()
            except queue.Empty:
                return
