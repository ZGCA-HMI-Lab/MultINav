"""Offline top-down reporting for a completed InteractiveNav V3 episode.

The renderer deliberately consumes evaluator artifacts only after an episode has
finished.  It never publishes its GT annotations to the policy process.  When a
private ``episode_visualization.json`` sidecar is available, it uses the live
evaluator geometry for literal target and articulated-object locations.  Older
results remain renderable with clearly labelled benchmark-plan fallbacks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from .benchmark_io import load_benchmark_episodes
except ImportError:  # Allow direct ``python episode_topdown.py`` use.
    from benchmark_io import load_benchmark_episodes


TOPDOWN_SCHEMA_VERSION = "interactive_nav_v3_episode_topdown_v4"
_UNKNOWN_MIN = 50
_UNKNOWN_MAX = 250


class _RosOccupancyScene:
    """Minimal scene-map adapter used when the frozen static scene is absent.

    A completed ROS run already contains a world-frame occupancy raster.  It is
    sufficient for a truthful trajectory report even when the optional ProcTHOR
    LMDB mirror is unavailable (for example on a clean evaluation machine).
    """

    def __init__(self, image: np.ndarray, resolution: float, origin_xy: np.ndarray, origin_yaw: float) -> None:
        self._image = np.asarray(image)
        self._resolution = float(resolution)
        self._origin_xy = np.asarray(origin_xy, dtype=float)
        self._origin_yaw = float(origin_yaw)
        self.px_per_m = int(round(1.0 / self._resolution))
        # Keep the plotted extent to observed cells.  Unknown ROS cells are
        # normally encoded as 205; including the whole unknown canvas makes a
        # small explored room unreadable in the report.
        observed = self._image != 205
        self.occupancy = observed if np.any(observed) else (self._image > _UNKNOWN_MIN)
        self.room_map = None
        self.room_ids_to_name: dict[int, str] = {}

    def pos_px_to_m(self, row_col: np.ndarray) -> np.ndarray:
        values = np.asarray(row_col, dtype=float).reshape((-1, 2))
        local_x = values[:, 1] * self._resolution
        local_y = (self._image.shape[0] - 1.0 - values[:, 0]) * self._resolution
        cosine, sine = math.cos(self._origin_yaw), math.sin(self._origin_yaw)
        world_x = self._origin_xy[0] + cosine * local_x - sine * local_y
        world_y = self._origin_xy[1] + sine * local_x + cosine * local_y
        return np.column_stack((world_x, world_y, np.zeros(len(values), dtype=float)))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _as_xy(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        xy = np.asarray(value[:2], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(xy)):
        return None
    return xy


def _load_result_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"Episode result must be a JSON object: {path}")
    result = document.get("result", document)
    trace = document.get("trace", [])
    if not isinstance(result, dict):
        raise ValueError(f"Episode result payload is invalid: {path}")
    if not isinstance(trace, list):
        trace = []
    return result, [row for row in trace if isinstance(row, dict)]


def _load_benchmark_episode(path: Path, *, episode_index: int, case_id: str | None) -> dict[str, Any]:
    benchmark_path, episodes = load_benchmark_episodes(path)
    if 0 <= episode_index < len(episodes) and isinstance(episodes[episode_index], dict):
        candidate = episodes[episode_index]
        candidate_case_id = candidate.get("interactive_nav", {}).get("case_id")
        if case_id is None or str(candidate_case_id) == case_id:
            return candidate
    if case_id is not None:
        for candidate in episodes:
            if isinstance(candidate, dict) and str(candidate.get("interactive_nav", {}).get("case_id")) == case_id:
                return candidate
    raise ValueError(f"Could not find episode index={episode_index} case_id={case_id!r} in {benchmark_path}")


def load_ros_map(debug_dir: Path) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Load the recorder's final occupancy grid in its world-coordinate frame."""

    yaml_path = debug_dir / "final_occ_map.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing ROS occupancy metadata: {yaml_path}")
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid occupancy metadata: {yaml_path}")
    image_path = Path(str(metadata["image"]))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read occupancy image: {image_path}")
    origin = np.asarray(metadata.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape[0] < 3:
        raise ValueError(f"Occupancy origin must contain x, y, yaw: {yaml_path}")
    return image, float(metadata["resolution"]), origin[:2], float(origin[2])


def load_trajectory(debug_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return recorder trajectory as world ``xy`` and yaw arrays."""

    path = debug_dir / "trajectory.csv"
    if not path.is_file():
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    points: list[tuple[float, float]] = []
    yaws: list[float] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                points.append((float(row["x"]), float(row["y"])))
                yaws.append(float(row.get("yaw", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
    if not points:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    return np.asarray(points, dtype=float), np.asarray(yaws, dtype=float)


def load_trace_trajectory(trace: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Recover executed base poses from an evaluator trace when no recorder ran."""

    points: list[tuple[float, float]] = []
    yaws: list[float] = []
    for row in trace:
        # New traces keep the pose in the public ``base`` block; older/basic
        # policy traces put it under the navigation detail block.  Accept all
        # of these shapes so a non-ROS evaluator still gets a driven path.
        pose = None
        for block_name in ("base", "oracle_navigation", "python_navigation", "navigation"):
            block = row.get(block_name, {})
            if isinstance(block, dict) and block.get("base_pose_xyyaw") is not None:
                pose = block.get("base_pose_xyyaw")
                break
        if pose is None:
            pose = row.get("base_pose_xyyaw")
        xy = _as_xy(pose)
        if xy is None:
            continue
        yaw = 0.0
        if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) >= 3:
            try:
                candidate = float(pose[2])
                if math.isfinite(candidate):
                    yaw = candidate
            except (TypeError, ValueError):
                pass
        points.append((float(xy[0]), float(xy[1])))
        yaws.append(yaw)
    if not points:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    return np.asarray(points, dtype=float), np.asarray(yaws, dtype=float)


def world_to_map_pixels(
    world_xy: np.ndarray,
    *,
    image_shape: tuple[int, int],
    resolution: float,
    origin_xy: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    """Convert world XY positions to recorder image ``row, column`` pixels."""

    points = np.asarray(world_xy, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points.reshape((-1, 2))
    delta = points - origin_xy[None, :]
    cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    cols = local_x / resolution
    rows = image_shape[0] - 1 - local_y / resolution
    return np.column_stack((rows, cols))


def _in_image(points_rc: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if not len(points_rc):
        return np.zeros((0,), dtype=bool)
    return (
        np.isfinite(points_rc).all(axis=1)
        & (points_rc[:, 0] >= 0)
        & (points_rc[:, 0] < image_shape[0])
        & (points_rc[:, 1] >= 0)
        & (points_rc[:, 1] < image_shape[1])
    )


def _oracle_navigation_point(
    episode: dict[str, Any],
    *,
    interaction_id: str | None = None,
    reason: str | None = None,
) -> np.ndarray | None:
    nav = episode.get("interactive_nav", {})
    plans = list(nav.get("oracle_plans") or [])
    if not plans and isinstance(nav.get("oracle_plan"), dict):
        plans = [nav["oracle_plan"]]
    for plan in plans:
        for step in plan.get("steps", []):
            if step.get("type") != "navigate":
                continue
            if interaction_id is not None and str(step.get("interaction_id")) != interaction_id:
                continue
            if reason is not None and str(step.get("reason")) != reason:
                continue
            point = _as_xy(step.get("goal_point"))
            if point is not None:
                return point
    return None


def _benchmark_start_pose(episode: dict[str, Any]) -> dict[str, Any] | None:
    """Return the frozen episode's initial base pose without using recorder state."""

    robot = episode.get("robot", {})
    init_qpos = robot.get("init_qpos", {}) if isinstance(robot, dict) else {}
    base = init_qpos.get("base") if isinstance(init_qpos, dict) else None
    xy = _as_xy(base)
    if xy is not None:
        yaw = None
        if isinstance(base, (list, tuple, np.ndarray)) and len(base) >= 3:
            try:
                candidate = float(base[2])
                yaw = candidate if math.isfinite(candidate) else None
            except (TypeError, ValueError):
                pass
        return {"xy": xy, "yaw": yaw, "source": "frozen_robot_init_qpos"}

    task = episode.get("task", {})
    pose = task.get("robot_base_pose") if isinstance(task, dict) else None
    xy = _as_xy(pose)
    if xy is None:
        return None
    yaw = None
    # V3 task poses use [x, y, z, qw, qx, qy, qz].  They are normally yaw-only.
    if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) >= 7:
        try:
            qw, qz = float(pose[3]), float(pose[6])
            candidate = 2.0 * math.atan2(qz, qw)
            yaw = candidate if math.isfinite(candidate) else None
        except (TypeError, ValueError):
            pass
    return {"xy": xy, "yaw": yaw, "source": "frozen_task_robot_base_pose"}


def _oracle_navigation_stages(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Return frozen V3 oracle navigation phase endpoints in execution order."""

    nav = episode.get("interactive_nav", {})
    plans = list(nav.get("oracle_plans") or [])
    if not plans and isinstance(nav.get("oracle_plan"), dict):
        plans = [nav["oracle_plan"]]
    stages: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        for step in plan.get("steps", []):
            if not isinstance(step, dict) or step.get("type") != "navigate":
                continue
            xy = _as_xy(step.get("goal_point"))
            if xy is None:
                continue
            if stages and np.allclose(stages[-1]["xy"], xy, atol=1e-5):
                continue
            stages.append(
                {
                    "xy": xy,
                    "reason": str(step.get("reason", "navigate")),
                    "interaction_id": step.get("interaction_id"),
                }
            )
    return stages


def _polyline_length_m(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())


def _frozen_oracle_stage_path(episode: dict[str, Any]) -> dict[str, Any] | None:
    """Return the benchmark's frozen oracle navigation stages as a polyline.

    Rendering must not import the research-time planner that originally created
    a benchmark.  The benchmark already freezes every navigation phase endpoint;
    joining those points gives a stable, auditable reference without making the
    standalone evaluator depend on method code.
    """

    start = _benchmark_start_pose(episode)
    stages = _oracle_navigation_stages(episode)
    if start is None or not stages:
        return None
    path_points = [np.asarray(start["xy"], dtype=float)]
    path_points.extend(np.asarray(stage["xy"], dtype=float) for stage in stages)
    path_xy = np.asarray(path_points, dtype=float)
    return {
        "source": "frozen_oracle_plan_stages",
        "complete": False,
        "xy": path_xy,
        "start": start,
        "stages": stages,
        "planner": None,
        "length_m": _polyline_length_m(path_xy),
    }


def _private_context_markers(context: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    value = context.get(key, [])
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _frozen_target_name(episode: dict[str, Any]) -> str | None:
    """Return the frozen task instance name without querying evaluator-only state."""

    nav_target = episode.get("interactive_nav", {}).get("target", {})
    if isinstance(nav_target, dict) and nav_target.get("selected_instance"):
        return str(nav_target["selected_instance"])
    task = episode.get("task", {})
    if isinstance(task, dict) and task.get("pickup_obj_name"):
        return str(task["pickup_obj_name"])
    return None


def _extract_gt_markers(episode: dict[str, Any], context: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build target and required-object markers, with plan-point fallbacks."""

    target: dict[str, Any] | None = None
    if isinstance(context, dict) and isinstance(context.get("target"), dict):
        point = _as_xy(context["target"].get("xy") or context["target"].get("aabb_center"))
        if point is not None:
            target = {
                "xy": point,
                "label": "GT target",
                "source": "evaluator_private_geometry",
            }
    if target is None:
        point = _oracle_navigation_point(episode, reason="satisfy_nav_to_obj_success")
        if point is not None:
            target = {
                "xy": point,
                "label": "GT target success point",
                "source": "frozen_oracle_plan_fallback",
            }
    if target is not None:
        target_name = _frozen_target_name(episode)
        if target_name is not None:
            target["object_name"] = target_name

    context_by_id = {
        str(row.get("interaction_id")): row
        for row in _private_context_markers(context, "gt_interactions")
        if row.get("interaction_id") is not None
    }
    markers: list[dict[str, Any]] = []
    for interaction in episode.get("interactive_nav", {}).get("interactions", []):
        if not isinstance(interaction, dict):
            continue
        interaction_id = str(interaction.get("interaction_id", ""))
        private = context_by_id.get(interaction_id)
        point = None if private is None else _as_xy(private.get("xy") or private.get("aabb_center"))
        if point is not None:
            source = "evaluator_private_geometry"
            label = "GT interaction object"
        else:
            point = _oracle_navigation_point(episode, interaction_id=interaction_id)
            source = "frozen_oracle_plan_fallback"
            label = "GT interaction approach"
        if point is None:
            continue
        markers.append(
            {
                "xy": point,
                "label": label,
                "source": source,
                "interaction_id": interaction_id,
                "interaction_type": str(interaction.get("type", "interaction")),
            }
        )
    return target, markers


def _command_node_id(request_id: Any) -> str | None:
    if not isinstance(request_id, str):
        return None
    match = re.search(r":interaction:([^:]+):", request_id)
    return None if match is None else str(match.group(1))


def _load_events(debug_dir: Path) -> list[dict[str, Any]]:
    path = debug_dir / "events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_graph_nodes(debug_dir: Path) -> list[dict[str, Any]]:
    candidates = [debug_dir / "graph" / "graph_latest.json", debug_dir / "graph_latest.json"]
    for path in candidates:
        if not path.is_file():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
            return [row for row in payload["nodes"] if isinstance(row, dict)]
    return []


def _graph_point(node: dict[str, Any]) -> np.ndarray | None:
    attributes = node.get("attributes", {})
    attributes = attributes if isinstance(attributes, dict) else {}
    for key in ("interaction_reference_aabb_center", "viz_aabb_center"):
        point = _as_xy(attributes.get(key))
        if point is not None:
            return point
    for key in ("aabb_center", "centroid"):
        point = _as_xy(node.get(key))
        if point is not None:
            return point
    return None


def _event_approach_point(event: dict[str, Any]) -> np.ndarray | None:
    payload = event.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    command = payload.get("interaction_command", payload)
    command = command if isinstance(command, dict) else {}
    for key in ("interaction_approach_pose_xyyaw", "approach_pose_xyyaw"):
        point = _as_xy(command.get(key))
        if point is not None:
            return point
    return None


def _extract_actual_markers(
    result: dict[str, Any],
    context: dict[str, Any] | None,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    """Map public attempts to evaluator-private sidecar, graph, or command pose."""

    private_rows = _private_context_markers(context, "actual_interactions")
    private_by_request = {
        str(row.get("request_id")): row
        for row in private_rows
        if row.get("request_id") is not None
    }
    nodes = _load_graph_nodes(debug_dir)
    nodes_by_id = {str(row.get("id")): row for row in nodes if row.get("id") is not None}
    nodes_by_instance = {
        str(row.get("attributes", {}).get("instance_id")): row
        for row in nodes
        if isinstance(row.get("attributes"), dict) and row["attributes"].get("instance_id") is not None
    }
    events = _load_events(debug_dir)
    events_by_request: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = event.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        command = payload.get("interaction_command", payload)
        command = command if isinstance(command, dict) else {}
        request_id = command.get("command_id") or command.get("request_id")
        if request_id is not None:
            events_by_request[str(request_id)] = event

    markers: list[dict[str, Any]] = []
    attempts = result.get("interaction_attempts", [])
    if not isinstance(attempts, list):
        return markers
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        request_id = attempt.get("request_id")
        private = private_by_request.get(str(request_id))
        point = None if private is None else _as_xy(private.get("xy") or private.get("aabb_center"))
        source = "evaluator_private_geometry" if point is not None else ""
        if point is None:
            node_id = _command_node_id(request_id)
            node = nodes_by_id.get(str(node_id)) if node_id is not None else None
            if node is None and attempt.get("instance_id") is not None:
                node = nodes_by_instance.get(str(attempt.get("instance_id")))
            if node is not None:
                point = _graph_point(node)
                source = "debug_semantic_graph"
        if point is None and request_id is not None:
            point = _event_approach_point(events_by_request.get(str(request_id), {}))
            source = "debug_interaction_command"
        if point is None:
            continue
        status = str(attempt.get("result_status") or attempt.get("status") or "unknown")
        succeeded = status.upper() in {"SUCCEEDED", "SUCCESS", "COMPLETED"}
        markers.append(
            {
                "xy": point,
                "label": f"Actual interaction {index} ({'success' if succeeded else 'failed'})",
                "source": source,
                "success": succeeded,
                "request_id": request_id,
                "instance_id": attempt.get("instance_id"),
            }
        )
    return markers


def _occupancy_rgb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour recorder cells while preserving observed-vs-unknown distinction."""

    unknown = (image > _UNKNOWN_MIN) & (image < _UNKNOWN_MAX)
    free = image >= _UNKNOWN_MAX
    occupied = image <= _UNKNOWN_MIN
    rgb = np.empty((*image.shape, 3), dtype=np.uint8)
    rgb[unknown] = (45, 55, 72)  # unobserved
    rgb[free] = (209, 237, 255)  # observed free
    rgb[occupied] = (100, 116, 139)  # observed occupied
    return rgb, ~unknown


def _crop_bounds(image_shape: tuple[int, int], known: np.ndarray, points_rc: np.ndarray, margin_px: int) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(known)
    valid_points = points_rc[_in_image(points_rc, image_shape)] if len(points_rc) else points_rc
    if len(valid_points):
        rows = np.concatenate((rows, np.rint(valid_points[:, 0]).astype(int)))
        cols = np.concatenate((cols, np.rint(valid_points[:, 1]).astype(int)))
    if not len(rows):
        return 0, image_shape[0], 0, image_shape[1]
    row0 = max(0, int(rows.min()) - margin_px)
    row1 = min(image_shape[0], int(rows.max()) + margin_px + 1)
    col0 = max(0, int(cols.min()) - margin_px)
    col1 = min(image_shape[1], int(cols.max()) + margin_px + 1)
    return row0, row1, col0, col1


def _load_coverage_payload(debug_dir: Path, coverage_path: Path | None) -> dict[str, Any]:
    candidates = [coverage_path] if coverage_path is not None else [debug_dir / "exploration_coverage.json"]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            return {"source": str(path), **payload}
    return {}


def _coverage_summary(
    debug_dir: Path,
    *,
    recomputed_metrics: dict[str, Any] | None,
    coverage_path: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Return whole-scene coverage text and auditable numeric metadata.

    Coverage is meaningful only when the ROS raster is projected onto the full
    GT navigable map.  A ROS-only raster has no GT denominator, so it must not be
    reported as 100% merely because all pixels in its cropped canvas are known.
    """

    payload = _load_coverage_payload(debug_dir, coverage_path)
    if recomputed_metrics is not None:
        coverage_metadata = {
            "source": "static_scene_map_recomputed",
            **recomputed_metrics,
        }
        if payload:
            # Keep an older standalone coverage result for comparison, but use
            # the exact map/radius represented by this image for its title and
            # primary metrics.
            coverage_metadata["reported_artifact"] = payload
    elif isinstance(payload.get("exploration_coverage_ratio"), (int, float)):
        coverage_metadata = dict(payload)
    else:
        coverage_metadata = {
            "source": "ros_occupancy_without_gt_denominator",
            "exploration_coverage_ratio": None,
            "mapped_free_coverage_ratio": None,
            "mapped_occupied_on_gt_free_ratio": None,
        }

    observed_ratio = coverage_metadata.get("exploration_coverage_ratio")
    mapped_free_ratio = coverage_metadata.get("mapped_free_coverage_ratio")
    false_occupied_ratio = coverage_metadata.get("mapped_occupied_on_gt_free_ratio")
    if not isinstance(observed_ratio, (int, float)):
        return "Whole-scene GT coverage: unavailable (full GT scene not loaded)", coverage_metadata

    segments = [f"observed {float(observed_ratio):.1%}"]
    if isinstance(mapped_free_ratio, (int, float)):
        segments.append(f"mapped free {float(mapped_free_ratio):.1%}")
    if isinstance(false_occupied_ratio, (int, float)):
        segments.append(f"false occupied {float(false_occupied_ratio):.1%}")
    return f"Whole-scene GT coverage: {' | '.join(segments)}", coverage_metadata


def _resolve_scene_model_path(
    episode: dict[str, Any],
    context: dict[str, Any] | None,
    explicit_path: Path | None,
) -> Path:
    """Resolve the frozen scene XML used for a static all-rooms background."""

    if explicit_path is not None:
        path = Path(explicit_path).expanduser().absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Requested scene model does not exist: {path}")
        return path
    if isinstance(context, dict) and context.get("scene_model_path"):
        candidate = Path(str(context["scene_model_path"]))
        if candidate.is_file():
            return candidate.absolute()
    scene_dataset = str(episode.get("scene_dataset", "procthor-10k"))
    data_split = str(episode.get("data_split", "val"))
    house_index = int(episode["house_index"])
    filename = f"{data_split}_{house_index}.xml"
    repo_root = Path(__file__).resolve().parents[3]
    scenes_roots = [repo_root / "assets" / "scenes"]
    registry_error: Exception | None = None
    try:
        # This follows the same asset-root selection as the evaluator, including
        # an MLSPACES_SCENES_ROOT supplied by the local environment.
        from molmo_spaces.molmo_spaces_constants import get_scenes, get_scenes_root
        from molmo_spaces.utils.lazy_loading_utils import install_scene_from_path

        scenes_roots.append(Path(get_scenes_root()))
        scene_entry = get_scenes(scene_dataset, data_split)[data_split][house_index]
        registry_path_value = (
            scene_entry.get("base") if isinstance(scene_entry, dict) else scene_entry
        )
        if registry_path_value is not None:
            registry_path = Path(str(registry_path_value)).expanduser()
            if not registry_path.is_file():
                install_scene_from_path(registry_path)
            if registry_path.is_file():
                return registry_path.absolute()
    except Exception as exc:
        # Explicit/local candidates below still make the report usable when a
        # resource registry is unavailable or intentionally disabled.
        registry_error = exc

    candidates: list[Path] = []
    for scenes_root in dict.fromkeys(scenes_roots):
        candidates.extend(
            [
                scenes_root / f"{scene_dataset}-{data_split}" / filename,
                scenes_root / scene_dataset / filename,
            ]
        )
        candidates.extend(sorted(scenes_root.glob(f"*{data_split}*/{filename}")))
    for candidate in candidates:
        if candidate.is_file():
            # Preserve a local symlink path.  ``prepare_writable_scene_path``
            # needs its assets/scenes-relative form to repair XML includes.
            return candidate.absolute()
    raise FileNotFoundError(
        "Could not resolve the static scene XML for "
        f"dataset={scene_dataset!r}, split={data_split!r}, house={house_index}. "
        "Pass --scene-model-path explicitly."
        + (
            f" Resource registry lookup failed with {type(registry_error).__name__}: {registry_error}"
            if registry_error is not None
            else ""
        )
    )


def _resolve_precomputed_scene_map_path(
    *,
    episode: dict[str, Any],
    source_model_path: Path | None,
    allow_registry_lookup: bool,
) -> Path | None:
    """Resolve and, when cached, link the complete precomputed scene map."""

    candidates: list[Path] = []
    if source_model_path is not None:
        stem = source_model_path.stem.removesuffix("_ceiling")
        sibling_map = source_model_path.with_name(f"{stem}_map.png")
        if sibling_map.is_file():
            # Resolve the global assets symlink to the persistent resource-cache
            # target.  Concurrent scene installs may refresh the symlink tree,
            # while the cache target itself remains stable for this render.
            return sibling_map.resolve()
        candidates.append(sibling_map)

    # A resource map can exist in the local cache without having been linked
    # into assets/scenes yet.  Query the authoritative scene registry so all
    # benchmark houses get the fast path, not only houses used earlier in this
    # process.  An explicit XML override deliberately skips this lookup because
    # its sibling map is the only map guaranteed to describe that exact model.
    if allow_registry_lookup:
        try:
            from molmo_spaces.molmo_spaces_constants import get_scenes

            dataset = str(episode.get("scene_dataset", "procthor-10k"))
            split = str(episode.get("data_split", "val"))
            house_index = int(episode["house_index"])
            scene_entry = get_scenes(dataset, split)[split][house_index]
            if isinstance(scene_entry, dict) and scene_entry.get("map"):
                candidates.append(Path(str(scene_entry["map"])))
        except Exception:
            # Registry access is an optimization only; rendering can still use
            # the resolved XML and the live-scene fallback below.
            pass

    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() and allow_registry_lookup:
            try:
                from molmo_spaces.utils.lazy_loading_utils import install_scene_from_path

                install_scene_from_path(candidate)
            except Exception:
                # The live XML path below remains the authoritative fallback.
                continue
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_static_scene_map(
    *,
    episode: dict[str, Any],
    context: dict[str, Any] | None,
    coverage_metadata: dict[str, Any],
    scene_model_path: Path | None,
) -> tuple[Any, Path, float, int, str, Path | None]:
    """Load the complete scene map, preferring its precomputed occupancy PNG."""

    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    # Prefer a stable XML when available, but allow a registry-provided
    # precomputed map to render a completed episode without one.
    try:
        source_model_path: Path | None = _resolve_scene_model_path(episode, context, scene_model_path)
    except FileNotFoundError:
        if scene_model_path is not None:
            raise
        source_model_path = None
    from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig

    default_agent_radius_m = float(NavToObjTaskSamplerConfig().robot_safety_radius)
    agent_radius_m = float(
        coverage_metadata.get("gt_agent_radius_m", default_agent_radius_m)
    )
    dataset_name = str(episode.get("scene_dataset", "")).lower()
    is_ithor = "ithor" in dataset_name
    # The frozen V3 map contract uses approximately 200 px/m.  Keep that
    # resolution for the static background and coverage projection.
    px_per_m = int(round(float(coverage_metadata.get("oracle_route_px_per_m", 200))))
    map_cls = iTHORMap if is_ithor else ProcTHORMap
    precomputed_map_path = _resolve_precomputed_scene_map_path(
        episode=episode,
        source_model_path=source_model_path,
        allow_registry_lookup=scene_model_path is None,
    )
    if precomputed_map_path is not None:
        scene_map = map_cls.load(path=str(precomputed_map_path), agent_radius=agent_radius_m)
        # The V3 benchmark source map is defined at approximately 200 px/m.
        # Saved maps preserve the exact fitted scale (for example 200.014), while
        # ``ProcTHORMap.load`` historically rounds it upward to 201.  Accept that
        # one-pixel metadata rounding but reject genuinely different resolutions.
        if abs(float(scene_map.px_per_m) - px_per_m) <= 1.0:
            reported_model_path = source_model_path
            if reported_model_path is None:
                split = str(episode.get("data_split", "val"))
                house_index = int(episode["house_index"])
                inferred_model_path = precomputed_map_path.with_name(f"{split}_{house_index}.xml")
                reported_model_path = inferred_model_path if inferred_model_path.is_file() else Path(
                    "<precomputed-scene-map>"
                )
            return (
                scene_map,
                reported_model_path,
                agent_radius_m,
                int(scene_map.px_per_m),
                "precomputed_complete_scene_map",
                precomputed_map_path,
            )
    if source_model_path is None:
        raise FileNotFoundError(
            "The scene has neither a precomputed map nor a resolvable base XML."
        )
    # Generate the fallback through the tracked simulator map module.  The core
    # ProcTHOR implementation clears traversable doorway regions and avoids any
    # dependency on benchmark-generation or exploration scripts.
    scene_map = map_cls.from_mj_model_path(
        model_path=str(source_model_path),
        agent_radius=agent_radius_m,
        px_per_m=px_per_m,
        device_id=None,
    )
    return (
        scene_map,
        source_model_path,
        agent_radius_m,
        px_per_m,
        "generated_core_scene_map",
        None,
    )


def _sample_ros_map(
    image: np.ndarray,
    resolution: float,
    origin_xy: np.ndarray,
    origin_yaw: float,
    world_xy: np.ndarray,
) -> np.ndarray:
    """Sample the recorder map with the same transform used by coverage eval."""

    delta = world_xy - origin_xy[None, :]
    cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    cell_x = np.floor(local_x / resolution).astype(np.int64)
    cell_y = np.floor(local_y / resolution).astype(np.int64)
    rows = image.shape[0] - 1 - cell_y
    valid = (cell_x >= 0) & (cell_x < image.shape[1]) & (rows >= 0) & (rows < image.shape[0])
    sampled = np.full(len(world_xy), 205, dtype=np.uint8)
    sampled[valid] = image[rows[valid], cell_x[valid]]
    return sampled


def _scene_coverage_classes(
    scene_map: Any,
    ros_image: np.ndarray,
    ros_resolution: float,
    ros_origin_xy: np.ndarray,
    ros_origin_yaw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify static scene pixels by their final recorder observation state."""

    free_mask = np.asarray(scene_map.occupancy, dtype=bool)
    free_rc = np.argwhere(free_mask)
    free_world = scene_map.pos_px_to_m(free_rc)[:, :2]
    sampled = _sample_ros_map(ros_image, ros_resolution, ros_origin_xy, ros_origin_yaw, free_world)
    observed = sampled != 205
    mapped_free = sampled >= 250
    mapped_occupied = sampled <= 50
    classes = np.zeros(free_mask.shape, dtype=np.uint8)
    classes[free_mask] = 1  # static navigable but not observed
    classes[free_rc[observed, 0], free_rc[observed, 1]] = 2
    classes[free_rc[mapped_free, 0], free_rc[mapped_free, 1]] = 3
    classes[free_rc[mapped_occupied, 0], free_rc[mapped_occupied, 1]] = 4
    gt_count = int(len(free_rc))
    observed_count = int(np.count_nonzero(observed))
    mapped_free_count = int(np.count_nonzero(mapped_free))
    mapped_occupied_count = int(np.count_nonzero(mapped_occupied))
    denominator = float(gt_count) if gt_count else 1.0
    return classes, {
        "gt_navigable_cells": gt_count,
        "observed_gt_cells": observed_count,
        "mapped_free_gt_cells": mapped_free_count,
        "mapped_occupied_on_gt_free_cells": mapped_occupied_count,
        "exploration_coverage_ratio": observed_count / denominator if gt_count else 0.0,
        "mapped_free_coverage_ratio": mapped_free_count / denominator if gt_count else 0.0,
        "mapped_occupied_on_gt_free_ratio": mapped_occupied_count / denominator if gt_count else 0.0,
    }


def _unobserved_scene_classes(scene_map: Any) -> np.ndarray:
    """Render a truthful static-layout fallback when the ROS recorder is absent."""

    free_mask = np.asarray(scene_map.occupancy, dtype=bool)
    classes = np.zeros(free_mask.shape, dtype=np.uint8)
    classes[free_mask] = 1  # static navigable, observation state unavailable
    return classes


def _scene_mesh(scene_map: Any, *, max_mesh_side: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world-coordinate mesh edges/centres following the reference plot."""

    height, width = np.asarray(scene_map.occupancy).shape[:2]
    stride = max(1, int(math.ceil(max(height, width) / max_mesh_side)))
    row_edges = np.arange(0, height + 1, stride)
    col_edges = np.arange(0, width + 1, stride)
    if row_edges[-1] != height:
        row_edges = np.append(row_edges, height)
    if col_edges[-1] != width:
        col_edges = np.append(col_edges, width)
    rows, cols = np.meshgrid(row_edges, col_edges, indexing="ij")
    edge_rc = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    edge_world = scene_map.pos_px_to_m(edge_rc).reshape(rows.shape + (3,))
    centre_rows = (row_edges[:-1] + row_edges[1:]) / 2.0
    centre_cols = (col_edges[:-1] + col_edges[1:]) / 2.0
    center_rr, center_cc = np.meshgrid(centre_rows, centre_cols, indexing="ij")
    center_rc = np.stack([center_rr.reshape(-1), center_cc.reshape(-1)], axis=1)
    center_world = scene_map.pos_px_to_m(center_rc).reshape(center_rr.shape + (3,))
    return edge_world, center_world, row_edges, col_edges


def _draw_scene_background(ax: Any, scene_map: Any, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Draw all rooms using the same occupancy + room-contour style as the reference."""

    from matplotlib.colors import ListedColormap

    edge_world, center_world, row_edges, col_edges = _scene_mesh(scene_map)
    sampled_classes = classes[row_edges[:-1][:, None], col_edges[None, :-1]]
    cmap = ListedColormap(["#2f3437", "#eeeeee", "#f6bd60", "#76b85a", "#d32f2f"])
    ax.pcolormesh(
        edge_world[..., 0],
        edge_world[..., 1],
        sampled_classes,
        cmap=cmap,
        vmin=-0.5,
        vmax=4.5,
        shading="flat",
        alpha=1.0,
        zorder=0,
        antialiased=False,
    )
    room_map = getattr(scene_map, "room_map", None)
    if room_map is not None:
        sampled_room = np.asarray(room_map)[row_edges[:-1][:, None], col_edges[None, :-1]]
        for room_id in sorted(int(value) for value in np.unique(sampled_room).tolist() if int(value) > 0):
            mask = (sampled_room == room_id).astype(float)
            if np.count_nonzero(mask) < 4:
                continue
            ax.contour(
                center_world[..., 0],
                center_world[..., 1],
                mask,
                levels=[0.5],
                colors=["#111827"],
                linewidths=1.15,
                alpha=0.85,
                zorder=1,
            )
            room_points = center_world[mask.astype(bool), :2]
            room_name = getattr(scene_map, "room_ids_to_name", {}).get(room_id, f"room_{room_id}")
            ax.text(
                float(np.mean(room_points[:, 0])),
                float(np.mean(room_points[:, 1])),
                str(room_name),
                fontsize=8.5,
                color="#111827",
                ha="center",
                va="center",
                weight="bold",
                zorder=2,
            )
    world_xy = edge_world[..., :2].reshape(-1, 2)
    return np.min(world_xy, axis=0), np.max(world_xy, axis=0)


def render_episode_topdown(
    *,
    episode_result_path: Path,
    benchmark_path: Path,
    debug_dir: Path,
    output_path: Path,
    private_context_path: Path | None = None,
    coverage_path: Path | None = None,
    scene_model_path: Path | None = None,
    ros_only: bool = False,
    require_full_scene: bool = False,
) -> dict[str, Any]:
    """Render one completed V3 evaluation episode as a top-down PNG.

    ``private_context_path`` points to the evaluator-only sidecar written by new
    V3 runs.  It is optional so the function also supports historic results.
    The returned dictionary is also persisted next to ``output_path``.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    episode_result_path = Path(episode_result_path).resolve()
    benchmark_path = Path(benchmark_path).resolve()
    debug_dir = Path(debug_dir).resolve()
    output_path = Path(output_path).resolve()
    result, trace = _load_result_document(episode_result_path)
    episode_index = int(result.get("episode_index", 0))
    case_id = None if result.get("case_id") is None else str(result.get("case_id"))
    episode = _load_benchmark_episode(benchmark_path, episode_index=episode_index, case_id=case_id)
    if private_context_path is None:
        candidate = episode_result_path.with_name("episode_visualization.json")
        private_context_path = candidate if candidate.is_file() else None
    context = _load_json(private_context_path) if private_context_path is not None and private_context_path.is_file() else None
    if context is not None and not isinstance(context, dict):
        raise ValueError(f"Private visualization context must be an object: {private_context_path}")

    ros_map_available = (debug_dir / "final_occ_map.yaml").is_file()
    ros_fallback_error: str | None = None
    ros_image: np.ndarray | None = None
    ros_resolution: float | None = None
    ros_origin_xy: np.ndarray | None = None
    ros_origin_yaw: float | None = None
    if ros_map_available:
        ros_image, ros_resolution, ros_origin_xy, ros_origin_yaw = load_ros_map(debug_dir)
    coverage_hint = _load_coverage_payload(debug_dir, coverage_path)
    static_scene_loaded = False
    try:
        if ros_only:
            raise RuntimeError("ros_only requested")
        (
            scene_map,
            resolved_scene_model,
            scene_agent_radius_m,
            scene_px_per_m,
            static_scene_map_source,
            precomputed_scene_map_path,
        ) = _load_static_scene_map(
            episode=episode,
            context=context,
            coverage_metadata=coverage_hint,
            scene_model_path=scene_model_path,
        )
        static_scene_loaded = True
    except Exception as exc:
        if require_full_scene:
            raise RuntimeError(
                "The complete GT scene was required for this top-down report, "
                f"but it could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        if not ros_map_available or ros_image is None or ros_resolution is None or ros_origin_xy is None or ros_origin_yaw is None:
            raise
        # Keep the report useful when the optional static scene/LMDB is missing.
        # GT and actual markers remain in the benchmark world frame, so the ROS
        # occupancy raster is a valid coordinate-aligned background.
        ros_fallback_error = f"{type(exc).__name__}: {exc}"
        scene_map = _RosOccupancyScene(ros_image, ros_resolution, ros_origin_xy, ros_origin_yaw)
        resolved_scene_model = Path("<ros-occupancy-fallback>")
        scene_agent_radius_m = float(coverage_hint.get("gt_agent_radius_m", 0.1))
        scene_px_per_m = int(round(1.0 / ros_resolution))
        static_scene_map_source = "ros_recorder_occupancy_fallback"
        precomputed_scene_map_path = None
    if ros_map_available:
        assert ros_image is not None and ros_resolution is not None and ros_origin_xy is not None and ros_origin_yaw is not None
        scene_classes, candidate_coverage_metrics = _scene_coverage_classes(
            scene_map,
            ros_image,
            ros_resolution,
            ros_origin_xy,
            ros_origin_yaw,
        )
        # A ROS-only canvas can colour its known cells, but it cannot provide the
        # denominator for whole-scene GT coverage.  Only retain these metrics when
        # the full static scene was successfully loaded.
        recomputed_coverage_metrics = candidate_coverage_metrics if static_scene_loaded else None
    else:
        scene_classes = _unobserved_scene_classes(scene_map)
        recomputed_coverage_metrics = None
    trajectory_xy, trajectory_yaw = load_trajectory(debug_dir)
    trajectory_source = "ros_recorder"
    if not len(trajectory_xy):
        trajectory_xy, trajectory_yaw = load_trace_trajectory(trace)
        trajectory_source = "evaluator_trace" if len(trajectory_xy) else "unavailable"
    target, gt_markers = _extract_gt_markers(episode, context)
    actual_markers = _extract_actual_markers(result, context, debug_dir)
    benchmark_start = _benchmark_start_pose(episode)
    if benchmark_start is None and len(trajectory_xy):
        benchmark_start = {
            "xy": trajectory_xy[0],
            "yaw": float(trajectory_yaw[0]) if len(trajectory_yaw) else None,
            "source": "recorder_trajectory_fallback",
        }
    gt_oracle_path = _frozen_oracle_stage_path(episode)
    gt_oracle_path_error: str | None = None
    if ros_map_available:
        coverage_label, coverage_metadata = _coverage_summary(
            debug_dir,
            recomputed_metrics=recomputed_coverage_metrics,
            coverage_path=coverage_path,
        )
    else:
        coverage_label = "ROS recorder map unavailable — static scene + evaluator trace"
        coverage_metadata = {
            "source": "evaluator_trace_fallback",
            "exploration_coverage_ratio": None,
        }
    fig, ax = plt.subplots(figsize=(13.0, 10.0), dpi=160)
    world_min, world_max = _draw_scene_background(ax, scene_map, scene_classes)
    world_margin = max(0.35, 0.04 * float(np.max(world_max - world_min)))
    ax.set_xlim(float(world_min[0] - world_margin), float(world_max[0] + world_margin))
    ax.set_ylim(float(world_min[1] - world_margin), float(world_max[1] + world_margin))
    ax.set_aspect("equal", adjustable="box")
    if gt_oracle_path is not None and len(gt_oracle_path["xy"]) >= 2:
        gt_path_xy = np.asarray(gt_oracle_path["xy"], dtype=float)
        ax.plot(
            gt_path_xy[:, 0],
            gt_path_xy[:, 1],
            color="#7c3aed",
            linewidth=2.2,
            linestyle=(0, (5, 2.4)),
            alpha=0.94,
            zorder=4,
        )
        for stage in gt_oracle_path["stages"]:
            stage_xy = np.asarray(stage["xy"], dtype=float)
            ax.scatter(
                stage_xy[0],
                stage_xy[1],
                s=28,
                marker="o",
                color="#7c3aed",
                edgecolors="white",
                linewidths=0.65,
                zorder=8,
            )
    if len(trajectory_xy) >= 2:
        ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color="#06b6d4", linewidth=2.0, alpha=0.96, zorder=5)
    if len(trajectory_xy):
        end = trajectory_xy[-1]
        ax.scatter(end[0], end[1], s=82, marker="o", color="#ef4444", edgecolors="white", linewidths=0.8, zorder=10)
    if benchmark_start is not None:
        start_xy = np.asarray(benchmark_start["xy"], dtype=float)
        ax.scatter(
            start_xy[0],
            start_xy[1],
            s=115,
            marker="^",
            color="#2563eb",
            edgecolors="white",
            linewidths=0.95,
            zorder=11,
        )
        yaw = benchmark_start.get("yaw")
        if isinstance(yaw, (int, float)) and math.isfinite(float(yaw)):
            heading = start_xy + 0.55 * np.asarray([math.cos(float(yaw)), math.sin(float(yaw))])
            ax.annotate(
                "",
                xy=(heading[0], heading[1]),
                xytext=(start_xy[0], start_xy[1]),
                arrowprops={"arrowstyle": "-|>", "color": "#2563eb", "lw": 1.9},
                zorder=12,
            )
        ax.annotate(
            "Benchmark start",
            xy=(start_xy[0], start_xy[1]),
            xytext=(8, -15),
            textcoords="offset points",
            fontsize=7.4,
            color="white",
            zorder=13,
            bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.84, "pad": 1.8},
        )

    def draw_marker(
        marker: dict[str, Any], *, color: str, marker_style: str, size: float, text: str, font_size: float = 7.4
    ) -> None:
        xy = np.asarray(marker["xy"], dtype=float)
        ax.scatter(xy[0], xy[1], s=size, marker=marker_style, color=color, edgecolors="white", linewidths=0.85, zorder=12)
        ax.annotate(
            text,
            xy=(xy[0], xy[1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=font_size,
            color="white",
            zorder=13,
            bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.84, "pad": 1.8},
        )

    if target is not None:
        target_suffix = "" if target["source"] == "evaluator_private_geometry" else " (plan point)"
        target_text = f"{target['label']}{target_suffix}"
        if target.get("object_name"):
            wrapped_name = "\n".join(textwrap.wrap(str(target["object_name"]), width=32, break_long_words=True))
            target_text = f"{target_text}\n{wrapped_name}"
        draw_marker(
            target,
            color="#fbbf24",
            marker_style="*",
            size=210,
            text=target_text,
            font_size=6.5 if target.get("object_name") else 7.4,
        )
    for index, marker in enumerate(gt_markers, start=1):
        suffix = "" if marker["source"] == "evaluator_private_geometry" else " (plan point)"
        draw_marker(marker, color="#d946ef", marker_style="D", size=68, text=f"GT interaction {index}{suffix}")
    actual_interaction_paths: list[dict[str, Any]] = []
    for index, marker in enumerate(actual_markers, start=1):
        # Make the physical interaction approach visible as a path, not only a
        # point.  The nearest recorded base pose is the last executed walking
        # pose before the evaluator's interaction marker.
        if len(trajectory_xy):
            marker_xy = np.asarray(marker["xy"], dtype=float)
            nearest_index = int(np.argmin(np.linalg.norm(trajectory_xy - marker_xy[None, :], axis=1)))
            approach_xy = trajectory_xy[nearest_index]
            if np.linalg.norm(approach_xy - marker_xy) > 1e-4:
                ax.plot(
                    [approach_xy[0], marker_xy[0]],
                    [approach_xy[1], marker_xy[1]],
                    color="#f97316",
                    linewidth=1.8,
                    linestyle=(0, (3, 2)),
                    alpha=0.9,
                    zorder=11,
                )
                actual_interaction_paths.append(
                    {
                        "from_xy": approach_xy.astype(float).tolist(),
                        "to_xy": marker_xy.astype(float).tolist(),
                        "request_id": marker.get("request_id"),
                    }
                )
        color = "#22c55e" if marker.get("success") else "#ef4444"
        draw_marker(marker, color=color, marker_style="X", size=88, text=f"Actual interaction {index}")

    case_label = str(result.get("case_id", episode_index))
    status_label = str(result.get("terminal_reason", result.get("status", "unknown")))
    path_length = result.get("navigation_path_length_m")
    path_text = "unknown" if not isinstance(path_length, (int, float)) else f"{float(path_length):.2f} m"
    gt_path_text = "unavailable"
    if gt_oracle_path is not None and isinstance(gt_oracle_path.get("length_m"), (int, float)):
        qualifier = "reconstructed" if gt_oracle_path.get("complete") else "frozen stages"
        gt_path_text = f"{float(gt_oracle_path['length_m']):.2f} m ({qualifier})"
    title = f"InteractiveNav V3 eval top-down — {case_label}"
    subtitle = f"{coverage_label}  |  GT oracle route: {gt_path_text}  |  driven path: {path_text}  |  terminal: {status_label}"
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11.5, pad=12)
    ax.set_axis_off()
    legend_handles = [
        Patch(facecolor="#2f3437", label="static obstacle / wall"),
        Patch(facecolor="#eeeeee", label="unobserved GT navigable"),
        Patch(facecolor="#f6bd60", label="observed / uncertain"),
        Patch(facecolor="#76b85a", label="mapped free"),
        Patch(facecolor="#d32f2f", label="false occupied"),
        Line2D([], [], color="#7c3aed", lw=2.2, linestyle=(0, (5, 2.4)), label="GT oracle plan (frozen stages)"),
        Line2D([], [], color="#06b6d4", lw=2, label="robot trajectory"),
        Line2D([], [], marker="^", color="w", markerfacecolor="#2563eb", markeredgecolor="white", markersize=8, label="benchmark start + heading"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#ef4444", markeredgecolor="white", markersize=7, label="final pose"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#fbbf24", markeredgecolor="white", markersize=11, label="GT target"),
        Line2D([], [], marker="D", color="w", markerfacecolor="#d946ef", markeredgecolor="white", markersize=7, label="GT required interaction"),
        Line2D([], [], marker="X", color="w", markerfacecolor="#ef4444", markeredgecolor="white", markersize=7, label="actual interaction (red=failed)"),
        Line2D([], [], color="#f97316", lw=1.8, linestyle=(0, (3, 2)), label="actual interaction approach path"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.2,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#94a3b8",
        ncol=2,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)

    metadata = {
        "schema_version": TOPDOWN_SCHEMA_VERSION,
        "episode_result": str(episode_result_path),
        "benchmark": str(benchmark_path),
        "debug_dir": str(debug_dir),
        "private_context": None if private_context_path is None else str(private_context_path),
        "output": str(output_path),
        "case_id": result.get("case_id"),
        "episode_index": episode_index,
        "coverage": coverage_metadata,
        "scene_background": {
            "mode": (
                "ros_recorder_occupancy_fallback"
                if not static_scene_loaded
                else (
                    "full_gt_scene_with_ros_coverage_overlay"
                    if ros_map_available
                    else "full_gt_scene_without_ros_overlay"
                )
            ),
            "model_path": str(resolved_scene_model),
            "map_source": static_scene_map_source,
            "precomputed_map_path": (
                str(precomputed_scene_map_path) if precomputed_scene_map_path is not None else None
            ),
            "agent_radius_m": scene_agent_radius_m,
            "px_per_m": scene_px_per_m,
            "complete_gt_scene_loaded": static_scene_loaded,
            "complete_gt_scene_required": require_full_scene,
            "coverage_overlay": (
                "ros_observed_mapped_free_false_occupied"
                if ros_map_available
                else "unavailable"
            ),
            "recomputed_coverage": recomputed_coverage_metrics,
            "ros_map_available": ros_map_available,
            "ros_fallback_error": ros_fallback_error,
        },
        "trajectory_samples": int(len(trajectory_xy)),
        "trajectory_source": trajectory_source,
        "benchmark_start": None
        if benchmark_start is None
        else {
            **benchmark_start,
            "xy": np.asarray(benchmark_start["xy"], dtype=float).tolist(),
        },
        "gt_oracle_path": None
        if gt_oracle_path is None
        else {
            **{key: value for key, value in gt_oracle_path.items() if key not in {"xy", "start", "stages"}},
            "xy": np.asarray(gt_oracle_path["xy"], dtype=float).tolist(),
            "start": {
                **gt_oracle_path["start"],
                "xy": np.asarray(gt_oracle_path["start"]["xy"], dtype=float).tolist(),
            },
            "stages": [
                {**stage, "xy": np.asarray(stage["xy"], dtype=float).tolist()}
                for stage in gt_oracle_path["stages"]
            ],
        },
        "gt_oracle_path_error": gt_oracle_path_error,
        "gt_target": None if target is None else {**target, "xy": np.asarray(target["xy"], dtype=float).tolist()},
        "gt_interactions": [{**row, "xy": np.asarray(row["xy"], dtype=float).tolist()} for row in gt_markers],
        "actual_interactions": [{**row, "xy": np.asarray(row["xy"], dtype=float).tolist()} for row in actual_markers],
        "actual_interaction_paths": actual_interaction_paths,
        "fallbacks_used": {
            "target": target is not None and target["source"] != "evaluator_private_geometry",
            "gt_interaction_count": sum(row["source"] != "evaluator_private_geometry" for row in gt_markers),
            "actual_interaction_count": sum(row["source"] != "evaluator_private_geometry" for row in actual_markers),
        },
    }
    _atomic_json(output_path.with_suffix(".json"), metadata)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-result", type=Path, required=True, help="Evaluator episode_result.json")
    parser.add_argument("--benchmark", type=Path, required=True, help="Frozen V3 benchmark JSON or its directory")
    parser.add_argument("--debug-dir", type=Path, required=True, help="ROS recorder debug directory for the same episode")
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG")
    parser.add_argument("--private-context", type=Path, help="Optional evaluator-private episode_visualization.json")
    parser.add_argument("--coverage-json", type=Path, help="Optional exact GT coverage JSON from evaluate_exploration_coverage.py")
    parser.add_argument(
        "--scene-model-path",
        type=Path,
        help="Optional static scene XML override; otherwise resolve it from the frozen benchmark episode.",
    )
    parser.add_argument(
        "--ros-only",
        action="store_true",
        help="Use the recorded ROS occupancy raster without loading the ProcTHOR/LMDB scene.",
    )
    parser.add_argument(
        "--require-full-scene",
        action="store_true",
        help="Fail instead of emitting a ROS-only fallback when the complete GT scene cannot be loaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata = render_episode_topdown(
        episode_result_path=args.episode_result,
        benchmark_path=args.benchmark,
        debug_dir=args.debug_dir,
        output_path=args.output,
        private_context_path=args.private_context,
        coverage_path=args.coverage_json,
        scene_model_path=args.scene_model_path,
        ros_only=args.ros_only,
        require_full_scene=args.require_full_scene,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
