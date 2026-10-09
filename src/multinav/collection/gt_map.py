"""Private simulator GT occupancy and route helpers for collection.

Extracted geometry helpers only: no online policy, ROS, MLLM or interaction graph.
"""
from __future__ import annotations
import gc
import logging
from pathlib import Path
from typing import Any
import cv2
import mujoco
import networkx as nx
import numpy as np
import molmo_spaces.utils.distance_transform_utils as dtutils
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.linalg_utils import inverse_homogeneous_matrix
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
from molmo_spaces.utils.scene_maps import ProcTHORMap, circular_kernel

log = logging.getLogger(__name__)
_WALL_COLLISION_SLICE_CACHE = {}
_WALL_COLLISION_SLICE_CACHE_MAX_SCENES = 8

def _collect_open_door_root_ids(model: mujoco.MjModel, data: mujoco.MjData, open_threshold: float) -> tuple[set[int], set[int]]:
    parent_to_child: dict[int, list[int]] = {}
    for body_id in range(model.nbody):
        root_body = model.body(model.body(body_id).rootid.item())
        root_body_id = int(root_body.id)
        root_body_name = root_body.name
        if root_body_name and (root_body_name.startswith("door_") or root_body_name.startswith("doorway_")):
            parent_to_child.setdefault(root_body_id, []).append(body_id)

    open_door_ids: set[int] = set()
    doorway_ids: set[int] = set()
    for root_body_id, children in parent_to_child.items():
        for door_body_id in children:
            door_body = model.body(door_body_id)
            jntadr = door_body.jntadr.item()
            if jntadr >= 0 and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE:
                qposadr = model.joint(jntadr).qposadr.item()
                if abs(float(data.qpos[qposadr])) > open_threshold:
                    open_door_ids.add(door_body_id)
                    doorway_ids.update(children)
            elif jntadr < 0 and len(children) == 2:
                doorway_ids.add(door_body_id)
    return open_door_ids, doorway_ids

def _collect_doorway_analysis(
    model: mujoco.MjModel, data: mujoco.MjData, open_threshold: float
) -> dict[str, Any]:
    parent_to_child: dict[int, list[int]] = {}
    for body_id in range(model.nbody):
        root_body = model.body(model.body(body_id).rootid.item())
        root_body_id = int(root_body.id)
        root_body_name = root_body.name
        if root_body_name and (
            root_body_name.startswith("door_")
            or root_body_name.startswith("doorway_")
            or root_body_name.startswith("doorframe_")
        ):
            parent_to_child.setdefault(root_body_id, []).append(body_id)

    open_door_ids: set[int] = set()
    doorway_root_ids: set[int] = set()
    non_interactive_root_ids: set[int] = set()
    interactive_door_body_ids: set[int] = set()
    fixed_opening_root_ids: set[int] = set()
    root_records: list[dict[str, Any]] = []

    for root_body_id, children in sorted(parent_to_child.items()):
        root_body = model.body(root_body_id)
        root_body_name = root_body.name
        if root_body_name.startswith("doorframe_"):
            root_kind = "doorframe"
        elif root_body_name.startswith("doorway_"):
            root_kind = "doorway"
        else:
            root_kind = "door"

        hinge_body_ids: list[int] = []
        open_hinge_body_ids: list[int] = []
        no_joint_body_ids: list[int] = []

        for body_id in children:
            body = model.body(body_id)
            jntadr = int(body.jntadr.item())
            if jntadr >= 0 and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE:
                hinge_body_ids.append(body_id)
                qposadr = int(model.joint(jntadr).qposadr.item())
                if abs(float(data.qpos[qposadr])) > open_threshold:
                    open_hinge_body_ids.append(body_id)
            elif jntadr < 0:
                no_joint_body_ids.append(body_id)

        interactive = len(hinge_body_ids) > 0
        pipeline_static_passage = (root_kind == "doorway") and (len(children) == 2) and (not interactive)
        fixed_opening = pipeline_static_passage or (
            root_kind == "doorframe" and len(children) == 2 and not interactive
        )
        if interactive:
            interactive_door_body_ids.update(hinge_body_ids)
            if open_hinge_body_ids:
                open_door_ids.update(open_hinge_body_ids)
                doorway_root_ids.add(root_body_id)
        else:
            non_interactive_root_ids.add(root_body_id)
            if fixed_opening:
                doorway_root_ids.add(root_body_id)
                fixed_opening_root_ids.add(root_body_id)

        root_records.append(
            {
                "root_body_id": root_body_id,
                "root_body_name": root_body_name,
                "root_kind": root_kind,
                "interactive": interactive,
                "child_body_names": [model.body(body_id).name for body_id in children],
                "hinge_body_names": [model.body(body_id).name for body_id in hinge_body_ids],
                "open_hinge_body_names": [model.body(body_id).name for body_id in open_hinge_body_ids],
                "no_joint_body_names": [model.body(body_id).name for body_id in no_joint_body_ids],
                "pipeline_static_passage": pipeline_static_passage,
                "fixed_opening": fixed_opening,
            }
        )

    return {
        "open_door_ids": open_door_ids,
        "doorway_root_ids": doorway_root_ids,
        "non_interactive_root_ids": non_interactive_root_ids,
        "interactive_door_body_ids": interactive_door_body_ids,
        "fixed_opening_root_ids": fixed_opening_root_ids,
        "root_records": root_records,
    }

def _compile_model_without_ceiling_geoms(model_path: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(model_path)

    ceiling_geoms = []

    def collect_ceiling_geoms_recursively(body_spec: mujoco.MjsBody) -> None:
        for geom in body_spec.geoms:
            geom_name = geom.name
            if geom_name and "ceiling" in geom_name.lower():
                ceiling_geoms.append(geom)
        for child_body in body_spec.bodies:
            collect_ceiling_geoms_recursively(child_body)

    collect_ceiling_geoms_recursively(spec.worldbody)
    for geom in ceiling_geoms:
        spec.delete(geom)

    try:
        return spec.compile()
    finally:
        del spec

def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1

def _copy_joint_positions_by_name(
    src_model: mujoco.MjModel,
    src_data: mujoco.MjData,
    dst_model: mujoco.MjModel,
    dst_data: mujoco.MjData,
) -> None:
    for src_joint_id in range(src_model.njnt):
        joint_name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, src_joint_id)
        if not joint_name:
            continue

        dst_joint_id = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if dst_joint_id < 0:
            continue

        src_width = _joint_qpos_width(int(src_model.jnt_type[src_joint_id]))
        dst_width = _joint_qpos_width(int(dst_model.jnt_type[dst_joint_id]))
        width = min(src_width, dst_width)
        src_adr = int(src_model.jnt_qposadr[src_joint_id])
        dst_adr = int(dst_model.jnt_qposadr[dst_joint_id])
        dst_data.qpos[dst_adr : dst_adr + width] = src_data.qpos[src_adr : src_adr + width]

def _move_root_free_joint_far_away(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_name: str,
    translation_xyz: np.ndarray | None = None,
) -> bool:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if body_id < 0:
        return False

    if translation_xyz is None:
        translation_xyz = np.array([1000.0, 1000.0, -100.0], dtype=float)

    body = model.body(body_id)
    jntadr = int(body.jntadr.item())
    if jntadr < 0:
        return False
    joint = model.joint(jntadr)
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:
        return False

    qposadr = int(joint.qposadr.item())
    data.qpos[qposadr : qposadr + 3] = np.asarray(translation_xyz, dtype=float)
    data.qpos[qposadr + 3 : qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return True

def _triangle_horizontal_slice_segment(
    triangle_xyz: np.ndarray,
    height_m: float,
    epsilon: float = 1e-6,
) -> np.ndarray | None:
    """Return the longest XY segment where a triangle intersects a Z plane."""
    triangle = np.asarray(triangle_xyz, dtype=float)
    if triangle.shape != (3, 3):
        raise ValueError(f"Expected a (3, 3) triangle, got {triangle.shape}")

    signed = triangle[:, 2] - float(height_m)
    points: list[np.ndarray] = []
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = triangle[start_index]
        end = triangle[end_index]
        start_signed = float(signed[start_index])
        end_signed = float(signed[end_index])
        start_on_plane = abs(start_signed) <= epsilon
        end_on_plane = abs(end_signed) <= epsilon

        if start_on_plane:
            points.append(start[:2].copy())
        if end_on_plane:
            points.append(end[:2].copy())
        if start_signed * end_signed < -(epsilon * epsilon):
            ratio = start_signed / (start_signed - end_signed)
            points.append((start + ratio * (end - start))[:2])

    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= epsilon for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None

    best_pair = None
    best_distance = 0.0
    for first_index in range(len(unique) - 1):
        for second_index in range(first_index + 1, len(unique)):
            distance = float(np.linalg.norm(unique[first_index] - unique[second_index]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (unique[first_index], unique[second_index])
    if best_pair is None or best_distance <= epsilon:
        return None
    return np.asarray(best_pair, dtype=float)

def _mesh_geom_world_vertices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None

    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    local_vertices = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
    )
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count], dtype=np.int64
    )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    translation = np.asarray(data.geom_xpos[geom_id], dtype=float)
    world_vertices = local_vertices @ rotation.T + translation
    return world_vertices, faces

def collect_wall_collision_slice_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height_m: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Slice thin ProcTHOR wall collision meshes without filling their door holes."""
    segments: list[np.ndarray] = []
    wall_geom_count = 0
    sliced_triangle_count = 0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name or ""
        if not body_name.startswith("wall_") or "collision" not in geom_name.lower():
            continue
        mesh_geometry = _mesh_geom_world_vertices(model, data, geom_id)
        if mesh_geometry is None:
            continue
        wall_geom_count += 1
        world_vertices, faces = mesh_geometry
        for face in faces:
            segment = _triangle_horizontal_slice_segment(
                world_vertices[np.asarray(face, dtype=np.int64)], height_m
            )
            if segment is None:
                continue
            segments.append(segment)
            sliced_triangle_count += 1

    segment_array = (
        np.asarray(segments, dtype=float)
        if segments
        else np.empty((0, 2, 2), dtype=float)
    )
    return segment_array, {
        "height_m": float(height_m),
        "wall_collision_geom_count": int(wall_geom_count),
        "slice_segment_count": int(len(segment_array)),
        "sliced_triangle_count": int(sliced_triangle_count),
    }

def cached_wall_collision_slice_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    model_path: str | None,
    height_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = None
    if model_path is not None:
        cache_key = (str(Path(model_path).resolve()), round(float(height_m), 4))
        cached = _WALL_COLLISION_SLICE_CACHE.get(cache_key)
        if cached is not None:
            segments, stats = cached
            return segments, {**stats, "cache_hit": True}

    segments, stats = collect_wall_collision_slice_segments(
        model, data, height_m=height_m
    )
    if cache_key is not None:
        if len(_WALL_COLLISION_SLICE_CACHE) >= _WALL_COLLISION_SLICE_CACHE_MAX_SCENES:
            oldest_key = next(iter(_WALL_COLLISION_SLICE_CACHE))
            _WALL_COLLISION_SLICE_CACHE.pop(oldest_key)
        _WALL_COLLISION_SLICE_CACHE[cache_key] = (segments, dict(stats))
    return segments, {**stats, "cache_hit": False}

def rasterize_world_xy_segments(
    segments_xy: np.ndarray,
    world_to_map: np.ndarray,
    shape: tuple[int, int],
    *,
    height_m: float = 0.0,
    thickness_px: int = 1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for segment in np.asarray(segments_xy, dtype=float):
        homogeneous = np.column_stack(
            [segment, np.full(2, float(height_m)), np.ones(2, dtype=float)]
        )
        pixels = homogeneous @ np.asarray(world_to_map, dtype=float).T
        start = (int(round(float(pixels[0, 1]))), int(round(float(pixels[0, 0]))))
        end = (int(round(float(pixels[1, 1]))), int(round(float(pixels[1, 0]))))
        cv2.line(
            mask,
            start,
            end,
            color=1,
            thickness=max(1, int(thickness_px)),
            lineType=cv2.LINE_8,
        )
    return mask.astype(bool)

def build_live_procthor_map(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_path: str | None = None,
    px_per_m: int = 200,
    agent_radius: float | None = None,
    open_threshold: float = 1e-3,
    device_id: int | None = None,
    treat_all_non_interactive_doorways_as_open: bool = False,
    return_doorway_analysis: bool = False,
    ignored_root_body_names: set[str] | None = None,
    include_wall_collision_slices: bool = True,
    wall_slice_height_m: float = 0.45,
    wall_slice_thickness_px: int = 1,
    doorway_clearance_m: float = 0.30,
) -> ProcTHORMap | tuple[ProcTHORMap, dict[str, Any] | None]:
    work_model = model
    work_data = data
    owns_work_model = False

    if model_path is not None:
        work_model = _compile_model_without_ceiling_geoms(model_path)
        work_data = mujoco.MjData(work_model)
        _copy_joint_positions_by_name(model, data, work_model, work_data)
        if ignored_root_body_names:
            ignored_moved = []
            for root_body_name in ignored_root_body_names:
                if _move_root_free_joint_far_away(work_model, work_data, root_body_name):
                    ignored_moved.append(root_body_name)
            if ignored_moved:
                log.info("Moved %d ignored movable roots out of scene for occupancy: %s", len(ignored_moved), ignored_moved[:8])
        mujoco.mj_forward(work_model, work_data)
        owns_work_model = True

    floor_ids = []
    room_ids_to_name = {}
    for geom_id in range(work_model.ngeom):
        geom_name = mujoco.mj_id2name(work_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
            floor_ids.append(geom_id)
            room_body_id = work_model.geom(geom_id).bodyid.item()
            room_ids_to_name[geom_id + 1] = work_model.body(room_body_id).name

    if not floor_ids:
        raise ValueError("No floors found in the live model.")

    doorway_analysis: dict[str, Any] | None = None
    if treat_all_non_interactive_doorways_as_open:
        doorway_analysis = _collect_doorway_analysis(work_model, work_data, open_threshold)
        open_door_ids = doorway_analysis["open_door_ids"]
        doorway_ids = doorway_analysis["doorway_root_ids"]
    else:
        open_door_ids, doorway_ids = _collect_open_door_root_ids(
            work_model, work_data, open_threshold
        )

    doorframe_geom_ids = []
    door_geom_ids = []
    for geom_id in range(work_model.ngeom):
        body_id = work_model.geom(geom_id).bodyid.item()
        parent_body_id = work_model.body(body_id).parentid.item()
        if body_id in open_door_ids or parent_body_id in open_door_ids:
            door_geom_ids.append(geom_id)
        root_body_id = work_model.body(body_id).rootid.item()
        if root_body_id in doorway_ids:
            doorframe_geom_ids.append(geom_id)

    aabb_center, aabb_size = geom_aabb(work_model, work_data, floor_ids, tight_mesh=False)
    aabb_size += np.array([2, 2, 0])

    def render_occupancy(cam_distance: float):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = aabb_center
        cam.distance = cam_distance
        cam.azimuth = 0
        cam.elevation = -90
        cam.orthographic = 1

        h = round(px_per_m * aabb_size[0])
        w = round(px_per_m * aabb_size[1])
        effective_px = h / aabb_size[0]

        renderer = MjOpenGLRenderer(model=work_model, height=h, width=w, device_id=device_id)
        renderer.update(work_data, cam)
        for camera in renderer.scene.camera:
            camera.orthographic = 1
            camera.frustum_bottom = -aabb_size[0] / 2
            camera.frustum_top = aabb_size[0] / 2

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        seg_geom = seg[..., 0]
        cam_to_world = None
        if cam_distance == 5.0:
            cam_to_world = np.eye(4)
            cam_to_world[:3, 3] = renderer.scene.camera[0].pos
            camera_x_ax = np.cross(renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
            cam_to_world[:3, :3] = np.column_stack(
                (camera_x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
            )
        renderer.close()

        occ_room_floor = np.zeros_like(seg_geom, dtype=int)
        for fid in floor_ids:
            occ_room_floor[seg_geom == fid] = fid + 1

        occ_floor = np.ones_like(seg_geom, dtype=bool)
        for fid in floor_ids:
            occ_floor &= seg_geom != fid

        occ_door = np.zeros_like(seg_geom, dtype=bool)
        for did in door_geom_ids:
            occ_door[seg_geom == did] = True

        occ_doorframe = np.zeros_like(seg_geom, dtype=bool)
        for did in doorframe_geom_ids:
            occ_doorframe[seg_geom == did] = True

        occ_door_path = occ_doorframe & ~occ_door
        doorway_clearance_px = max(1, int(round(doorway_clearance_m * effective_px)))
        occ_door_path = cv2.dilate(
            occ_door_path.astype(np.uint8),
            circular_kernel(doorway_clearance_px),
        ).astype(bool)

        occ = occ_floor.copy()
        occ[occ_door_path == 1] = False
        # The carve removes the top-down door-frame/lintel projection. Keep the
        # actually swung-open leaf as a collision obstacle if it overlaps the
        # carved portal region.
        occ[occ_door] = True

        if cam_distance == 5.0:
            return occ, occ_room_floor, effective_px, (h, w), cam_to_world
        return occ, occ_room_floor, effective_px, (h, w)

    occ_map_5, room_map_5, effective_px, (h, w), cam_to_world = render_occupancy(5.0)
    occ_final = occ_map_5.copy()
    room_map_final = room_map_5.copy()

    cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
    world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)

    map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
    centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
    cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
    cam_to_world_floor[2, 2] = 0
    map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered

    wall_slice_mask = np.zeros_like(occ_final, dtype=bool)
    wall_slice_stats = {
        "enabled": bool(include_wall_collision_slices),
        "height_m": float(wall_slice_height_m),
        "wall_collision_geom_count": 0,
        "slice_segment_count": 0,
        "sliced_triangle_count": 0,
        "rasterized_pixel_count": 0,
    }
    if include_wall_collision_slices:
        wall_segments, collected_stats = cached_wall_collision_slice_segments(
            work_model,
            work_data,
            model_path=model_path,
            height_m=wall_slice_height_m,
        )
        wall_slice_mask = rasterize_world_xy_segments(
            wall_segments,
            world_to_map,
            occ_final.shape,
            height_m=wall_slice_height_m,
            thickness_px=wall_slice_thickness_px,
        )
        occ_final |= wall_slice_mask
        room_map_final[wall_slice_mask] = 0
        wall_slice_stats.update(collected_stats)
        wall_slice_stats["rasterized_pixel_count"] = int(wall_slice_mask.sum())

    if agent_radius is not None:
        rad_px = int(agent_radius * effective_px)
        kernel = circular_kernel(rad_px)
        occ_final = cv2.dilate(occ_final.astype(np.uint8), kernel).astype(bool)
        room_map_final[occ_final] = 0

    occ_final = ~occ_final
    if not np.any(occ_final) or np.all(occ_final):
        raise RuntimeError(
            "build_live_procthor_map produced a degenerate occupancy map "
            f"(all_free={bool(np.all(occ_final))}, all_blocked={bool(not np.any(occ_final))})."
        )

    instance = ProcTHORMap(
        occupancy=occ_final,
        room_map=room_map_final,
        room_ids_to_name=room_ids_to_name,
        world_to_map=world_to_map,
        map_to_world=map_to_world,
        px_per_m=effective_px,
    )
    instance.occupancy_rendered_base = occ_map_5
    instance.occupancy_wall_slice_mask = wall_slice_mask
    instance.occupancy_base = occ_map_5 | wall_slice_mask
    instance.wall_slice_stats = wall_slice_stats

    if owns_work_model:
        del work_data
        del work_model

    gc.collect()
    if return_doorway_analysis:
        return instance, doorway_analysis
    return instance

def compute_path_from_map(
    scene_map: ProcTHORMap,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    downscale_factor: int = 5,
    max_start_goal_distance: int = 40,
) -> np.ndarray | None:
    grid = scene_map.occupancy.astype(bool)
    padded = np.zeros(
        (
            grid.shape[0] + (downscale_factor - grid.shape[0] % downscale_factor),
            grid.shape[1] + (downscale_factor - grid.shape[1] % downscale_factor),
        ),
        dtype=bool,
    )
    padded[: grid.shape[0], : grid.shape[1]] = grid
    downscaled_grid = (
        padded.reshape(
            padded.shape[0] // downscale_factor,
            downscale_factor,
            padded.shape[1] // downscale_factor,
            downscale_factor,
        )
        .min(axis=1)
        .min(axis=-1)
    )

    grid_spacing = downscale_factor / scene_map.px_per_m
    dt = dtutils.make_distance_transform(downscaled_grid, grid_spacing)
    graph = dtutils.make_grid_graph(downscaled_grid, dt, weight_exp=2)

    def discretize(location_xy: np.ndarray) -> tuple[int, int]:
        px = scene_map.pos_m_to_px(np.array([location_xy[0], location_xy[1], 0.0]))
        rc = np.floor(px / downscale_factor).astype(np.int32)
        return int(rc[0]), int(rc[1])

    def find_close(missing: tuple[int, int]) -> tuple[int, int] | None:
        for search_range in range(1, max_start_goal_distance + 1):
            for shiftr in range(-search_range, search_range + 1):
                for shiftc in range(-search_range, search_range + 1):
                    if shiftr != search_range and shiftc != search_range:
                        continue
                    candidate = (missing[0] + shiftr, missing[1] + shiftc)
                    if candidate in graph:
                        return candidate
        return None

    start = discretize(start_xy)
    goal = discretize(goal_xy)
    if start not in graph:
        start = find_close(start)
    if goal not in graph:
        goal = find_close(goal)
    if start is None or goal is None:
        return None

    try:
        waypoints, _, _ = dtutils.make_discrete_path(
            graph, start[0], start[1], goal[0], goal[1], dt, 3, grid_spacing, 0.6
        )
    except nx.NetworkXUnfeasible:
        return None

    pixel_waypoints = np.array(waypoints) * downscale_factor
    return scene_map.pos_px_to_m(pixel_waypoints)[:, :2]
