from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Mapping

import mujoco
import numpy as np

from molmo_spaces.env.data_views import Door


@dataclass(frozen=True)
class ForceDriveConfig:
    max_physics_substeps: int = 1500
    stable_substeps: int = 8
    position_tolerance: float = 0.01
    velocity_tolerance: float = 0.01
    hinge_kp: float = 90.0
    hinge_kd: float = 12.0
    hinge_max_effort: float = 45.0
    slide_kp: float = 600.0
    slide_kd: float = 80.0
    slide_max_effort: float = 160.0
    open_fraction_threshold: float = 0.67
    assume_success: bool = True


class ForceInteractionError(RuntimeError):
    pass


def joint_closed_open_values(joint_range) -> tuple[float, float]:
    lower, upper = (float(joint_range[0]), float(joint_range[1]))
    if lower <= 0.0 <= upper:
        closed = 0.0
    else:
        closed = lower if abs(lower) <= abs(upper) else upper
    opened = lower if abs(lower - closed) >= abs(upper - closed) else upper
    return closed, opened


def joint_open_fraction(value: float, joint_range) -> float:
    closed, opened = joint_closed_open_values(joint_range)
    span = abs(opened - closed)
    if span <= 1e-8:
        return 0.0
    return float(np.clip(abs(float(value) - closed) / span, 0.0, 1.0))


def ground_drawer_open_regions(
    joints: list[dict[str, Any]],
    open_regions: list[dict[str, Any]],
    body_heights: dict[str, float],
    *,
    fallback_to_all: bool = True,
) -> list[dict[str, Any]]:
    """Privately map public crop-space drawer regions to slide joints.

    The visual plan contains only normalized image centers.  Simulator joint
    names and body heights remain on the trusted execution side.  Regions and
    joints are both ordered top-to-bottom before matching, so a plan never
    needs to expose a private joint selector.
    """

    drawers = [
        dict(joint)
        for joint in joints
        if str(joint.get("joint_type") or "").casefold() == "slide"
        and str(joint.get("joint_name") or "")
    ]
    drawers.sort(
        key=lambda joint: (
            -float(body_heights.get(str(joint.get("joint_name") or ""), 0.0)),
            str(joint.get("joint_name") or ""),
        )
    )
    if not drawers:
        return []
    regions = [
        dict(region)
        for region in open_regions
        if isinstance(region, dict)
        and isinstance(region.get("center"), (list, tuple))
        and len(region["center"]) >= 2
    ]
    regions.sort(key=lambda region: (float(region["center"][1]), float(region["center"][0])))
    if not regions:
        if not fallback_to_all:
            return []
        return [
            {
                "group_id": f"sim_drawer_{index:02d}",
                "joint_names": [str(joint["joint_name"])],
                "open_region": None,
                "grounding_source": "simulator_all_slide_joints",
            }
            for index, joint in enumerate(drawers)
        ]

    available = list(range(len(drawers)))
    grounded = []
    for region_index, region in enumerate(regions[: len(drawers)]):
        y = max(0.0, min(1.0, float(region["center"][1])))
        target_rank = y * max(0, len(drawers) - 1)
        drawer_index = min(available, key=lambda index: (abs(index - target_rank), index))
        available.remove(drawer_index)
        grounded.append(
            {
                "group_id": f"visual_drawer_{region_index:02d}",
                "joint_names": [str(drawers[drawer_index]["joint_name"])],
                "open_region": region,
                "grounding_source": "visual_region_vertical_order",
                "simulator_rank": drawer_index,
            }
        )
    grounded.sort(key=lambda group: float((group["open_region"] or {})["center"][1]))
    return grounded


def merge_door_leaf_joint_records(
    discovered_leaves: list[dict[str, Any]],
    root_joints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    leaves = [dict(leaf) for leaf in discovered_leaves]
    known_joint_names = {str(leaf["hinge_joint_name"]) for leaf in leaves}
    for joint in root_joints:
        joint_name = str(joint["joint_name"])
        searchable = f"{joint_name} {joint.get('body_name', '')}".casefold()
        if (
            joint_name in known_joint_names
            or "door" not in searchable
            or "handle" in searchable
            or "knob" in searchable
        ):
            continue
        leaves.append(
            {
                "leaf_body_name": str(joint.get("body_name") or ""),
                "hinge_joint_index": None,
                "hinge_joint_name": joint_name,
                "joint_range": list(joint["joint_range"]),
                "joint_id": int(joint["joint_id"]),
            }
        )
        known_joint_names.add(joint_name)
    return leaves


def collect_door_root_groups(env) -> dict[str, dict[str, Any]]:
    model = env.current_model
    data = env.current_data
    qpos_before = np.array(data.qpos, copy=True)
    qvel_before = np.array(data.qvel, copy=True)
    ctrl_before_values = np.array(data.ctrl, copy=True)
    object_manager = env.object_managers[env.current_batch_index]
    groups: dict[int, dict[str, Any]] = {}
    for door_name in object_manager.find_door_names():
        try:
            door = Door(door_name, data)
            hinge_index = int(door.get_hinge_joint_index())
            joint_name = str(door.joint_names[hinge_index])
            joint_range = [float(value) for value in door.get_joint_range(hinge_index)]
        except (KeyError, ValueError):
            continue
        body_id = int(model.body(door_name).id)
        root_id = int(model.body_rootid[body_id])
        group = groups.setdefault(
            root_id,
            {
                "root_body_id": root_id,
                "root_body_name": str(model.body(root_id).name or f"door_root_{root_id}"),
                "leaves": [],
            },
        )
        group["leaves"].append(
            {
                "leaf_body_name": str(door_name),
                "hinge_joint_index": hinge_index,
                "hinge_joint_name": joint_name,
                "joint_range": joint_range,
                "joint_id": int(
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                ),
            }
        )
    for group in groups.values():
        group["leaves"] = merge_door_leaf_joint_records(
            group["leaves"],
            _root_joint_records(model, int(group["root_body_id"])),
        )
    return {
        str(group["root_body_name"]): group
        for group in sorted(groups.values(), key=lambda item: int(item["root_body_id"]))
    }


def _set_all_door_roots_state(env, state: str) -> list[dict[str, Any]]:
    normalized_state = str(state).strip().lower()
    if normalized_state not in {"closed", "open"}:
        raise ValueError(f"Unsupported initial door state: {state}")
    groups = collect_door_root_groups(env)
    transitions = []
    for root_name, group in groups.items():
        leaf_transitions = []
        for leaf in group["leaves"]:
            joint_id = int(leaf["joint_id"])
            qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
            dof_addr = int(env.current_model.jnt_dofadr[joint_id])
            before = float(env.current_data.qpos[qpos_addr])
            closed, opened = joint_closed_open_values(leaf["joint_range"])
            target = closed if normalized_state == "closed" else opened
            env.current_data.qpos[qpos_addr] = target
            env.current_data.qvel[dof_addr] = 0.0
            leaf_transitions.append(
                {
                    **leaf,
                    "before": before,
                    "target": target,
                }
            )
        transitions.append(
            {
                "root_body_name": root_name,
                "state": normalized_state,
                "leaf_transitions": leaf_transitions,
            }
        )
    mujoco.mj_forward(env.current_model, env.current_data)
    for transition in transitions:
        for leaf in transition["leaf_transitions"]:
            joint_id = int(leaf["joint_id"])
            qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
            value = float(env.current_data.qpos[qpos_addr])
            leaf["after"] = value
            leaf["open_fraction"] = joint_open_fraction(
                value, leaf["joint_range"]
            )
        transition["state_verified"] = all(
            (
                float(leaf["open_fraction"]) <= 0.05
                if normalized_state == "closed"
                else float(leaf["open_fraction"]) >= 0.95
            )
            for leaf in transition["leaf_transitions"]
        )
    return transitions


def set_all_door_roots_closed(env) -> list[dict[str, Any]]:
    return _set_all_door_roots_state(env, "closed")


def set_all_door_roots_open(env) -> list[dict[str, Any]]:
    return _set_all_door_roots_state(env, "open")


def build_articulation_targets(
    joints: list[dict[str, Any]],
    selected_joint_names: list[str] | None = None,
    close_other_joint_names: list[str] | None = None,
    close_other_joints: bool = False,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Build open/close targets for a container articulation command."""
    available = {
        str(joint.get("joint_name")): joint
        for joint in joints
        if joint.get("joint_name")
    }
    if not available:
        raise ValueError("Articulation has no hinge or slide joints")
    selected = list(selected_joint_names or available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown articulation joints: {unknown}")
    close_names = list(close_other_joint_names or [])
    if close_other_joints and not close_names:
        close_names = [
            name
            for name, joint in available.items()
            if name not in selected and str(joint.get("joint_type")) == "slide"
        ]
    unknown_close = sorted(set(close_names) - set(available))
    if unknown_close:
        raise ValueError(f"Unknown close-other joints: {unknown_close}")
    targets = {}
    for name in selected:
        _closed, opened = joint_closed_open_values(available[name]["joint_range"])
        targets[name] = opened
    for name in close_names:
        closed, _opened = joint_closed_open_values(available[name]["joint_range"])
        targets[name] = closed
    return targets, selected, close_names


def _root_joint_records(model, root_body_id: int) -> list[dict[str, Any]]:
    records = []
    for joint_id in range(int(model.njnt)):
        body_id = int(model.jnt_bodyid[joint_id])
        if int(model.body_rootid[body_id]) != int(root_body_id):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        records.append(
            {
                "joint_name": str(model.joint(joint_id).name),
                "joint_type": "slide" if joint_type == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                "joint_range": [float(value) for value in model.jnt_range[joint_id]],
                "joint_id": int(joint_id),
                "body_name": str(model.body(body_id).name or ""),
            }
        )
    return records


def collect_articulation_groups(env) -> dict[str, dict[str, Any]]:
    """Collect top-level articulated scene objects, including non-door containers."""
    model = env.current_model
    object_manager = env.object_managers[env.current_batch_index]
    metadata_by_name = dict((env.current_scene_metadata or {}).get("objects", {}) or {})
    groups: dict[int, dict[str, Any]] = {}
    try:
        objects = object_manager.list_top_level_objects()
    except Exception:
        objects = []
    for obj in objects:
        name = str(getattr(obj, "name", "") or "")
        if not name:
            continue
        try:
            body_id = int(model.body(name).id)
            root_id = int(model.body_rootid[body_id])
        except (KeyError, ValueError):
            continue
        joints = _root_joint_records(model, root_id)
        if not joints:
            continue
        metadata = dict(metadata_by_name.get(name, {}) or {})
        group = groups.setdefault(
            root_id,
            {
                "root_body_id": root_id,
                "root_body_name": str(model.body(root_id).name or name),
                "object_name": name,
                "aliases": set(),
                "category": str(metadata.get("category") or name),
                "is_door": False,
                "joints": joints,
            },
        )
        group["aliases"].update({name, str(group["root_body_name"])})
    try:
        door_groups = collect_door_root_groups(env)
    except Exception:
        door_groups = {}
    for root_name, door_group in door_groups.items():
        root_id = int(door_group["root_body_id"])
        group = groups.setdefault(
            root_id,
            {
                "root_body_id": root_id,
                "root_body_name": str(root_name),
                "object_name": str(root_name),
                "aliases": set(),
                "category": "Door",
                "is_door": True,
                "joints": [],
            },
        )
        group["is_door"] = True
        group["category"] = "Door"
        group["aliases"].update(
            {
                str(root_name),
                *(str(leaf["leaf_body_name"]) for leaf in door_group["leaves"]),
            }
        )
        discovered_door_joints = [
            {
                "joint_name": str(leaf["hinge_joint_name"]),
                "joint_type": "hinge",
                "joint_range": list(leaf["joint_range"]),
                "joint_id": int(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        leaf["hinge_joint_name"],
                    )
                ),
            }
            for leaf in door_group["leaves"]
        ]
        existing_joint_names = {
            str(joint["joint_name"]) for joint in group["joints"]
        }
        group["joints"].extend(
            joint
            for joint in discovered_door_joints
            if str(joint["joint_name"]) not in existing_joint_names
        )
        group["joints"] = [
            joint
            for joint in group["joints"]
            if "handle" not in str(joint["joint_name"]).casefold()
            and "knob" not in str(joint["joint_name"]).casefold()
        ]
    result = {}
    for group in groups.values():
        group["aliases"] = sorted(str(alias) for alias in group["aliases"])
        for alias in group["aliases"]:
            result[alias] = group
    return result


def _joint_infos_for_group(model, data, joints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    infos = []
    for joint in joints:
        joint_id = int(joint["joint_id"])
        qpos_addr = int(model.jnt_qposadr[joint_id])
        value = float(data.qpos[qpos_addr])
        infos.append(
            {
                "joint_name": str(joint["joint_name"]),
                "joint_type": str(joint["joint_type"]),
                "joint_range": list(joint["joint_range"]),
                "joint_value": value,
                "open_fraction": joint_open_fraction(value, joint["joint_range"]),
            }
        )
    return infos


def open_articulation_with_force(
    env,
    object_name: str,
    config: ForceDriveConfig | None = None,
    selected_joint_names: list[str] | None = None,
    close_other_joint_names: list[str] | None = None,
    close_other_joints: bool = False,
    interaction_group_id: str = "all",
) -> dict[str, Any]:
    """Open a door or container articulation using the force PD backend."""
    config = config or ForceDriveConfig()
    groups = collect_articulation_groups(env)
    group = groups.get(str(object_name))
    if group is None:
        return {
            "supported": False,
            "reason": "non_articulated",
            # Missing a simulator articulation only proves that this command
            # cannot be executed by the force backend.  It does *not* prove
            # that the portal is an already-open passage: a fixed door leaf
            # and a fixed opening have the same articulation lookup result.
            # Keep the capability explicitly unavailable until a public
            # aperture/occupancy observation establishes the passage state.
            "interaction_capability": "unavailable",
            "object_name": str(object_name),
            "available_object_names": sorted(groups),
        }
    model = env.current_model
    data = env.current_data
    joints = list(group["joints"])
    targets, selected, close_names = build_articulation_targets(
        joints,
        selected_joint_names=selected_joint_names,
        close_other_joint_names=close_other_joint_names,
        close_other_joints=close_other_joints,
    )
    pre_infos = _joint_infos_for_group(model, data, joints)
    drive = drive_joint_group_to_targets(model, data, targets, config=config)
    post_infos = _joint_infos_for_group(model, data, joints)
    selected_set = set(selected)
    open_success = bool(selected) and all(
        float(info["open_fraction"]) >= float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in selected_set
    )
    close_success = all(
        float(info["open_fraction"]) <= 1.0 - float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in set(close_names)
    )
    success = bool(open_success and close_success)
    if config.assume_success and not success:
        raise ForceInteractionError(
            f"Force interaction did not reach targets for {object_name}: "
            f"selected={selected} close={close_names} infos={post_infos}"
        )
    pre_max = max((float(info["open_fraction"]) for info in pre_infos), default=0.0)
    post_selected = [info for info in post_infos if info["joint_name"] in selected_set]
    post_min = min((float(info["open_fraction"]) for info in post_selected), default=0.0)
    return {
        "root_body_name": str(group["root_body_name"]),
        "object_name": str(group.get("object_name") or object_name),
        "category": str(group.get("category") or "articulation"),
        "is_door": bool(group.get("is_door")),
        "action": "open",
        "method": "force_pd",
        "success": success,
        "pre_state": "closed" if pre_max <= 0.10 else "ajar",
        "post_state": "open" if post_min >= float(config.open_fraction_threshold) else "ajar",
        "interaction_group_id": str(interaction_group_id or "all"),
        "selected_joint_names": selected,
        "closed_joint_names": close_names,
        "pre_joint_infos": pre_infos,
        "joint_infos": post_infos,
        "physics_substeps": int(drive["physics_substeps"]),
        "drive": drive,
    }


def prepare_articulation_force(
    env,
    object_name: str,
    selected_joint_names: list[str] | None = None,
    close_other_joint_names: list[str] | None = None,
    close_other_joints: bool = False,
) -> dict[str, Any]:
    groups = collect_articulation_groups(env)
    group = groups.get(str(object_name))
    if group is None:
        return {
            "supported": False,
            "reason": "non_articulated",
            "interaction_capability": "unavailable",
            "object_name": str(object_name),
            "available_object_names": sorted(groups),
        }
    targets, selected, close_names = build_articulation_targets(
        list(group["joints"]),
        selected_joint_names=selected_joint_names,
        close_other_joint_names=close_other_joint_names,
        close_other_joints=close_other_joints,
    )
    return {
        "supported": True,
        "group": group,
        "targets": targets,
        "selected_joint_names": selected,
        "closed_joint_names": close_names,
        "pre_joint_infos": _joint_infos_for_group(
            env.current_model, env.current_data, list(group["joints"])
        ),
    }


def prepare_articulation_state_force(
    env,
    object_name: str,
    open_joint_names: list[str] | None = None,
    close_joint_names: list[str] | None = None,
) -> dict[str, Any]:
    groups = collect_articulation_groups(env)
    group = groups.get(str(object_name))
    if group is None:
        raise ValueError(
            f"Articulated object not found: {object_name}; available={sorted(groups)}"
        )
    available = {
        str(joint["joint_name"]): joint for joint in list(group["joints"])
    }
    selected = [str(name) for name in (open_joint_names or [])]
    closed = [str(name) for name in (close_joint_names or [])]
    unknown = sorted((set(selected) | set(closed)) - set(available))
    if unknown:
        raise ValueError(f"Unknown articulation joints: {unknown}")
    targets = {}
    for name in selected:
        _closed, opened = joint_closed_open_values(available[name]["joint_range"])
        targets[name] = opened
    for name in closed:
        closed_value, _opened = joint_closed_open_values(available[name]["joint_range"])
        targets[name] = closed_value
    if not targets:
        raise ValueError("Articulation state transition requires at least one joint")
    return {
        "group": group,
        "targets": targets,
        "selected_joint_names": selected,
        "closed_joint_names": closed,
        "pre_joint_infos": _joint_infos_for_group(
            env.current_model, env.current_data, list(group["joints"])
        ),
    }


def articulation_joint_infos(env, object_name: str) -> list[dict[str, Any]]:
    groups = collect_articulation_groups(env)
    group = groups.get(str(object_name))
    if group is None:
        raise ValueError(
            f"Articulated object not found: {object_name}; available={sorted(groups)}"
        )
    return _joint_infos_for_group(env.current_model, env.current_data, list(group["joints"]))


def _body_subtree_geometry_centroid(model, data, root_body_id: int) -> np.ndarray | None:
    """Return a size-weighted live geom centroid for one articulated leaf."""

    root_body_id = int(root_body_id)
    geom_positions: list[np.ndarray] = []
    geom_weights: list[float] = []
    for geom_id in range(int(getattr(model, "ngeom", 0) or 0)):
        body_id = int(model.geom_bodyid[geom_id])
        cursor = body_id
        belongs = cursor == root_body_id
        while not belongs and cursor > 0:
            cursor = int(model.body_parentid[cursor])
            belongs = cursor == root_body_id
        if not belongs:
            continue
        position = np.asarray(data.geom_xpos[geom_id], dtype=float)
        if position.shape[0] < 3 or not np.all(np.isfinite(position[:3])):
            continue
        try:
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            weight = float(np.prod(np.maximum(np.abs(size[:3]), 1e-3)))
        except (AttributeError, IndexError, TypeError, ValueError):
            weight = 1.0
        geom_positions.append(position[:3])
        geom_weights.append(max(1e-9, weight))
    if not geom_positions:
        return None
    return np.average(
        np.asarray(geom_positions, dtype=float),
        axis=0,
        weights=np.asarray(geom_weights, dtype=float),
    )


def infer_articulation_front_axis_xy(env, object_name: str) -> dict[str, Any]:
    """Infer a private physical front normal from live articulation geometry.

    Slide travel directly defines the outward normal. For a hinge, the initial
    opening velocity of the articulated body (axis cross hinge-to-body radius)
    defines the side from which an outward-opening leaf is operated. No joint
    names or axis values leave the force bridge; callers receive only a compact
    validation source and the normalized axis for an immediate private check.
    """

    groups = collect_articulation_groups(env)
    group = groups.get(str(object_name))
    if group is None:
        return {"checked": False, "reason": "articulation_group_unavailable"}
    model = env.current_model
    data = env.current_data
    axes: list[np.ndarray] = []
    sources: list[str] = []
    # A refrigerator can expose both a hinge for the door leaf and slide joints
    # for internal trays/rails.  Averaging those motion vectors can produce a
    # diagonal (or even cancelling) normal that is not an operable front.  The
    # articulated door/leaf hinge is the stronger operational-front signal;
    # use slide travel only when no hinge geometry is available.
    hinge_axes: list[np.ndarray] = []
    hinge_sources: list[str] = []
    slide_axes: list[np.ndarray] = []
    slide_sources: list[str] = []
    for joint in list(group.get("joints") or []):
        joint_id = int(joint.get("joint_id", -1))
        if joint_id < 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        local_axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
        body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        world_axis = body_rotation @ local_axis
        closed_value, open_value = joint_closed_open_values(joint.get("joint_range") or [])
        direction_sign = 1.0 if open_value >= closed_value else -1.0
        joint_type = str(joint.get("joint_type") or "").casefold()
        if joint_type == "slide":
            axis_xy = direction_sign * world_axis[:2]
            source = "slide_open_travel"
        elif joint_type == "hinge":
            anchor = np.asarray(data.xanchor[joint_id], dtype=float)
            panel_centroid = _body_subtree_geometry_centroid(model, data, body_id)
            radial_source = "body_origin"
            if panel_centroid is not None:
                radial = np.asarray(panel_centroid, dtype=float) - anchor
                radial_source = "panel_geometry"
            else:
                radial = np.asarray(data.xpos[body_id], dtype=float) - anchor
            axis_xy = direction_sign * np.cross(world_axis, radial)[:2]
            source = f"hinge_{radial_source}_initial_open_motion"
        else:
            continue
        norm = float(np.linalg.norm(axis_xy))
        if math.isfinite(norm) and norm > 1e-6:
            normalized_axis = np.asarray(axis_xy, dtype=float) / norm
            axes.append(normalized_axis)
            sources.append(source)
            if joint_type == "hinge":
                hinge_axes.append(normalized_axis)
                hinge_sources.append(source)
            elif joint_type == "slide":
                slide_axes.append(normalized_axis)
                slide_sources.append(source)
    if not axes:
        return {"checked": False, "reason": "front_axis_geometry_unavailable"}
    # Prefer hinge-derived panel motion when mixed hinge/slide articulations
    # exist.  Slide-only appliances (or drawer-like containers) still use their
    # direct travel axis.
    selected_axes = hinge_axes or slide_axes or axes
    selected_sources = hinge_sources or slide_sources or sources
    reference = selected_axes[0]
    aligned = [
        axis if float(np.dot(axis, reference)) >= 0.0 else -axis
        for axis in selected_axes
    ]
    mean_axis = np.mean(aligned, axis=0)
    norm = float(np.linalg.norm(mean_axis))
    if not math.isfinite(norm) or norm <= 1e-6:
        mean_axis = reference
        norm = float(np.linalg.norm(mean_axis))
    return {
        "checked": True,
        "axis_xy": [float(value) for value in mean_axis / norm],
        "source": "+".join(sorted(set(selected_sources))),
        "joint_count": len(selected_axes),
        "candidate_joint_count": len(axes),
    }


def _build_force_specs(model, joint_targets: Mapping[str, float]) -> list[dict[str, Any]]:
    specs = []
    for joint_name, target_value in joint_targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Force drive only supports hinge/slide joints, got type={joint_type}")
        joint_range = [float(value) for value in model.jnt_range[joint_id]]
        lower, upper = min(joint_range), max(joint_range)
        specs.append(
            {
                "joint_name": str(joint_name),
                "joint_id": int(joint_id),
                "joint_type": joint_type,
                "joint_range": joint_range,
                "qpos_addr": int(model.jnt_qposadr[joint_id]),
                "dof_addr": int(model.jnt_dofadr[joint_id]),
                "body_id": int(model.jnt_bodyid[joint_id]),
                "local_axis": np.asarray(model.jnt_axis[joint_id], dtype=float),
                "target_value": float(np.clip(float(target_value), lower, upper)),
            }
        )
    if not specs:
        raise ValueError("Force drive requires at least one joint target")
    return specs


def apply_articulation_force_once(
    env,
    plan: dict[str, Any],
    config: ForceDriveConfig | None = None,
) -> None:
    config = config or ForceDriveConfig()
    model = env.current_model
    data = env.current_data
    specs = _build_force_specs(model, plan["targets"])
    data.xfrc_applied[:, :] = 0.0
    for spec in specs:
        current = float(data.qpos[spec["qpos_addr"]])
        velocity = float(data.qvel[spec["dof_addr"]])
        error = float(spec["target_value"] - current)
        is_slide = spec["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE
        kp = config.slide_kp if is_slide else config.hinge_kp
        kd = config.slide_kd if is_slide else config.hinge_kd
        max_effort = config.slide_max_effort if is_slide else config.hinge_max_effort
        effort = float(np.clip(kp * error - kd * velocity, -max_effort, max_effort))
        world_axis = data.xmat[spec["body_id"]].reshape(3, 3) @ spec["local_axis"]
        if is_slide:
            data.xfrc_applied[spec["body_id"], :3] += world_axis * effort
        else:
            data.xfrc_applied[spec["body_id"], 3:] += world_axis * effort
    plan["force_specs"] = specs


def _robot_articulation_contact_stats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
) -> dict[str, Any]:
    count = 0
    minimum_distance = None
    pairs = set()
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        body_ids = [
            int(model.geom_bodyid[int(contact.geom1)]),
            int(model.geom_bodyid[int(contact.geom2)]),
        ]
        root_matches = [
            body_id
            for body_id in body_ids
            if int(model.body_rootid[body_id]) == int(root_body_id)
        ]
        robot_matches = [
            body_id
            for body_id in body_ids
            if str(model.body(body_id).name or "").startswith("robot")
        ]
        if not root_matches or not robot_matches:
            continue
        count += 1
        distance = float(contact.dist)
        minimum_distance = distance if minimum_distance is None else min(minimum_distance, distance)
        pairs.add(
            tuple(sorted(
                str(model.body(body_id).name or body_id)
                for body_id in body_ids
            ))
        )
    return {
        "count": int(count),
        "minimum_distance": minimum_distance,
        "body_pairs": sorted(pairs),
    }


def complete_articulation_force(
    env,
    plan: dict[str, Any],
    config: ForceDriveConfig | None = None,
    *,
    robot_lock_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = config or ForceDriveConfig()
    model = env.current_model
    data = env.current_data
    drive_result = drive_joint_group_to_targets(
        model,
        data,
        plan["targets"],
        config=config,
        robot_lock_callback=robot_lock_callback,
    )
    plan["drive"] = drive_result
    plan["force_specs"] = _build_force_specs(model, plan["targets"])
    post_infos = _joint_infos_for_group(model, data, list(plan["group"]["joints"]))
    selected = set(plan["selected_joint_names"])
    closed = set(plan["closed_joint_names"])
    open_success = bool(selected) and all(
        float(info["open_fraction"]) >= float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in selected
    )
    close_success = all(
        float(info["open_fraction"]) <= 1.0 - float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in closed
    )
    pre_max = max(
        (float(info["open_fraction"]) for info in plan["pre_joint_infos"]),
        default=0.0,
    )
    post_selected = [
        info for info in post_infos if info["joint_name"] in selected
    ]
    post_min = min(
        (float(info["open_fraction"]) for info in post_selected),
        default=0.0,
    )
    physical_success = bool(open_success and close_success)
    atomic_fallback = False
    if not physical_success:
        atomic_fallback = True
        for spec in plan["force_specs"]:
            data.qpos[spec["qpos_addr"]] = spec["target_value"]
            data.qvel[spec["dof_addr"]] = 0.0
        mujoco.mj_forward(model, data)
        post_infos = _joint_infos_for_group(model, data, list(plan["group"]["joints"]))
        open_success = bool(selected) and all(
            float(info["open_fraction"]) >= float(config.open_fraction_threshold)
            for info in post_infos
            if info["joint_name"] in selected
        )
        close_success = all(
            float(info["open_fraction"]) <= 1.0 - float(config.open_fraction_threshold)
            for info in post_infos
            if info["joint_name"] in closed
        )
        post_selected = [
            info for info in post_infos if info["joint_name"] in selected
        ]
        post_min = min(
            (float(info["open_fraction"]) for info in post_selected),
            default=0.0,
        )
    success = bool(open_success and close_success)
    joints = []
    for spec in plan.get("force_specs", []):
        info = next(item for item in post_infos if item["joint_name"] == spec["joint_name"])
        joints.append(
            {
                **info,
                "target_value": float(spec["target_value"]),
                "final_value": float(info["joint_value"]),
                "final_error": float(spec["target_value"] - info["joint_value"]),
                "reached_target": abs(float(spec["target_value"] - info["joint_value"]))
                <= config.position_tolerance,
            }
        )
    return {
        "method": "xfrc_applied_group_pd",
        "success": success,
        "physical_success": physical_success,
        "atomic_fallback": atomic_fallback,
        "physics_substeps": int(drive_result.get("physics_substeps", 0)),
        "task_steps_consumed": 1,
        "stable_substeps": int(drive_result.get("stable_substeps", 0)),
        "drive": drive_result,
        "joints": joints,
        "pre_joint_infos": plan["pre_joint_infos"],
        "joint_infos": post_infos,
        "selected_joint_names": list(plan["selected_joint_names"]),
        "closed_joint_names": list(plan["closed_joint_names"]),
        "pre_state": "closed" if pre_max <= 0.10 else "ajar",
        "post_state": "open" if post_min >= float(config.open_fraction_threshold) else "ajar",
        "config": asdict(config),
    }


def finalize_articulation_force_transition(
    env,
    plan: dict[str, Any],
    *,
    physics_substeps: int,
    task_steps_consumed: int,
    atomic_fallback: bool,
    transition_log: list[dict[str, Any]] | None = None,
    config: ForceDriveConfig | None = None,
) -> dict[str, Any]:
    """Summarize a task-step articulation transition without another force pulse."""
    config = config or ForceDriveConfig()
    model = env.current_model
    data = env.current_data
    force_specs = _build_force_specs(model, plan["targets"])
    post_infos = _joint_infos_for_group(model, data, list(plan["group"]["joints"]))
    selected = set(plan["selected_joint_names"])
    closed = set(plan["closed_joint_names"])
    open_success = bool(selected) and all(
        float(info["open_fraction"]) >= float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in selected
    )
    close_success = all(
        float(info["open_fraction"]) <= 1.0 - float(config.open_fraction_threshold)
        for info in post_infos
        if info["joint_name"] in closed
    )
    pre_max = max(
        (float(info["open_fraction"]) for info in plan["pre_joint_infos"]),
        default=0.0,
    )
    post_selected = [info for info in post_infos if info["joint_name"] in selected]
    post_min = min(
        (float(info["open_fraction"]) for info in post_selected),
        default=0.0,
    )
    joints = []
    for spec in force_specs:
        info = next(item for item in post_infos if item["joint_name"] == spec["joint_name"])
        joints.append(
            {
                **info,
                "target_value": float(spec["target_value"]),
                "final_value": float(info["joint_value"]),
                "final_error": float(spec["target_value"] - info["joint_value"]),
                "reached_target": abs(float(spec["target_value"] - info["joint_value"]))
                <= config.position_tolerance,
            }
        )
    success = bool(open_success and close_success)
    return {
        "method": "xfrc_applied_group_pd_task_steps",
        "success": success,
        "physical_success": bool(success and not atomic_fallback),
        "atomic_fallback": bool(atomic_fallback),
        "physics_substeps": int(physics_substeps),
        "task_steps_consumed": max(1, int(task_steps_consumed)),
        "stable_substeps": 0,
        "drive": {
            "mode": "task_step_transition",
            "transitions": list(transition_log or []),
        },
        "joints": joints,
        "pre_joint_infos": plan["pre_joint_infos"],
        "joint_infos": post_infos,
        "selected_joint_names": list(plan["selected_joint_names"]),
        "closed_joint_names": list(plan["closed_joint_names"]),
        "pre_state": "closed" if pre_max <= 0.10 else "ajar",
        "post_state": "open" if post_min >= float(config.open_fraction_threshold) else "ajar",
        "config": asdict(config),
    }


def advance_articulation_force(
    env,
    plan: dict[str, Any],
    progress: float,
    start_values: Mapping[str, float],
    transition_steps: int,
    config: ForceDriveConfig | None = None,
    *,
    robot_lock_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Advance an articulation toward a target over one task step."""
    config = config or ForceDriveConfig()
    alpha = float(np.clip(float(progress), 0.0, 1.0))
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    targets = {
        name: float(start_values.get(name, value))
        + (float(value) - float(start_values.get(name, value))) * eased
        for name, value in plan["targets"].items()
    }
    step_config = replace(
        config,
        max_physics_substeps=max(
            1,
            int(math.ceil(float(config.max_physics_substeps) / max(1, int(transition_steps)))),
        ),
        assume_success=False,
    )
    drive = drive_joint_group_to_targets(
        env.current_model,
        env.current_data,
        targets,
        config=step_config,
        robot_lock_callback=robot_lock_callback,
    )
    fallback = False
    if not drive.get("success"):
        specs = _build_force_specs(env.current_model, targets)
        for spec in specs:
            env.current_data.qpos[spec["qpos_addr"]] = spec["target_value"]
            env.current_data.qvel[spec["dof_addr"]] = 0.0
        mujoco.mj_forward(env.current_model, env.current_data)
        fallback = True
    return {
        "success": True,
        "progress": alpha,
        "eased_progress": eased,
        "targets": targets,
        "physics_substeps": int(drive.get("physics_substeps", 0)),
        "fallback": fallback,
        "joint_infos": _joint_infos_for_group(
            env.current_model, env.current_data, list(plan["group"]["joints"])
        ),
    }


def set_all_articulations_closed(env, include_doors: bool = True) -> list[dict[str, Any]]:
    """Reset all discovered hinge/slide articulations to their closed references."""
    groups = collect_articulation_groups(env)
    seen_roots = set()
    transitions = []
    model = env.current_model
    data = env.current_data
    for group in groups.values():
        root_id = int(group["root_body_id"])
        if root_id in seen_roots or (group.get("is_door") and not include_doors):
            continue
        seen_roots.add(root_id)
        before = _joint_infos_for_group(model, data, group["joints"])
        for joint in group["joints"]:
            joint_id = int(joint["joint_id"])
            qpos_addr = int(model.jnt_qposadr[joint_id])
            dof_addr = int(model.jnt_dofadr[joint_id])
            closed, _opened = joint_closed_open_values(joint["joint_range"])
            data.qpos[qpos_addr] = closed
            data.qvel[dof_addr] = 0.0
        transitions.append(
            {
                "root_body_name": str(group["root_body_name"]),
                "object_name": str(group.get("object_name") or group["root_body_name"]),
                "is_door": bool(group.get("is_door")),
                "before_joint_infos": before,
                "joint_names": [str(joint["joint_name"]) for joint in group["joints"]],
            }
        )
    mujoco.mj_forward(model, data)
    groups_by_root = {
        int(group["root_body_id"]): group for group in groups.values()
    }
    for transition in transitions:
        root_name = str(transition["root_body_name"])
        group = next(
            (
                candidate
                for candidate in groups_by_root.values()
                if str(candidate["root_body_name"]) == root_name
            ),
            None,
        )
        after = (
            _joint_infos_for_group(model, data, group["joints"])
            if group is not None
            else []
        )
        transition["after_joint_infos"] = after
        transition["closed_state_verified"] = bool(after) and all(
            float(info["open_fraction"]) <= 0.05 for info in after
        )
    return transitions


def apply_view_profile(
    env,
    profile: str,
    tilt_rad: float = 0.55,
    restore_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Command the head position actuators without teleporting joint state."""
    normalized = str(profile or "default").strip().lower()
    desired_tilt = 0.0 if normalized in {"", "default", "level"} else float(tilt_rad)
    model = env.current_model
    data = env.current_data
    qpos_before = np.array(data.qpos, copy=True)
    qvel_before = np.array(data.qvel, copy=True)
    ctrl_before_values = np.array(data.ctrl, copy=True)
    head_joint_ids = []
    for joint_id in range(int(model.njnt)):
        name = str(model.joint(joint_id).name or "")
        if name.endswith("head_0") or name.endswith("head_1"):
            head_joint_ids.append((name, int(joint_id)))
    head_joint_ids.sort(key=lambda item: item[0])
    before = []
    after = []
    restore_values = {
        str(item.get("joint_name")): item
        for item in (restore_state or {}).get("joints", [])
        if item.get("joint_name")
    }
    for index, (name, joint_id) in enumerate(head_joint_ids[:2]):
        qpos_addr = int(model.jnt_qposadr[joint_id])
        dof_addr = int(model.jnt_dofadr[joint_id])
        old_value = float(data.qpos[qpos_addr])
        old_velocity = float(data.qvel[dof_addr])
        restored = restore_values.get(name)
        target = 0.0 if index == 0 else desired_tilt
        lower, upper = [float(value) for value in model.jnt_range[joint_id]]
        target = float(np.clip(target, min(lower, upper), max(lower, upper)))
        actuator_name = f"{name}_act"
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        ctrl_previous = None
        ctrl_target = None
        if actuator_id >= 0:
            ctrl_previous = float(data.ctrl[actuator_id])
            ctrl_target = (
                float(restored.get("ctrl"))
                if restored is not None and restored.get("ctrl") is not None
                else target
            )
            data.ctrl[actuator_id] = ctrl_target
        before.append(
            {
                "joint_name": name,
                "qpos": old_value,
                "qvel": old_velocity,
                "ctrl": ctrl_previous,
            }
        )
        after.append(
            {
                "joint_name": name,
                "qpos": float(data.qpos[qpos_addr]),
                "qvel": float(data.qvel[dof_addr]),
                "ctrl": ctrl_target,
            }
        )
    head_qpos_addrs = {int(model.jnt_qposadr[joint_id]) for _, joint_id in head_joint_ids[:2]}
    head_dof_addrs = {int(model.jnt_dofadr[joint_id]) for _, joint_id in head_joint_ids[:2]}
    changed_qpos = [
        int(index)
        for index, value in enumerate(data.qpos)
        if index not in head_qpos_addrs and not np.isclose(value, qpos_before[index])
    ]
    changed_qvel = [
        int(index)
        for index, value in enumerate(data.qvel)
        if index not in head_dof_addrs and not np.isclose(value, qvel_before[index])
    ]
    head_actuator_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act"))
        for name, _ in head_joint_ids[:2]
    }
    changed_ctrl = [
        int(index)
        for index, value in enumerate(data.ctrl)
        if index not in head_actuator_ids and not np.isclose(value, ctrl_before_values[index])
    ]
    return {
        "profile": normalized or "default",
        "control_mode": "head_position_actuator",
        "applied": bool(head_joint_ids),
        "before": before,
        "after": after,
        "snapshot": {"joints": before},
        "non_head_state_changed": bool(changed_qpos or changed_qvel or changed_ctrl),
        "non_head_changed_qpos_indices": changed_qpos,
        "non_head_changed_qvel_indices": changed_qvel,
        "non_head_changed_ctrl_indices": changed_ctrl,
    }


class HeadViewController:
    def __init__(self) -> None:
        self._restore_state: dict[str, Any] | None = None
        self._restore_head_targets: dict[str, float] = {}
        self._torso_target: np.ndarray | None = None
        self._torso_restore_target: np.ndarray | None = None
        self._torso_home_target: np.ndarray | None = None

    def reset(self) -> None:
        self._restore_state = None
        self._restore_head_targets = {}
        self._torso_target = None
        self._torso_restore_target = None
        self._torso_home_target = None

    def command(
        self,
        env,
        profile: str,
        tilt_rad: float = 0.55,
        torso_pitch_rad: float | None = None,
    ) -> dict[str, Any]:
        torso_controller = getattr(env.current_robot, "controllers", {}).get("torso")
        torso_target = getattr(torso_controller, "target", None)
        torso_target_values = (
            None
            if torso_target is None
            else np.asarray(torso_target, dtype=float).reshape(-1).copy()
        )
        if self._torso_home_target is None and torso_target_values is not None:
            self._torso_home_target = torso_target_values.copy()
        self._torso_restore_target = (
            None if self._torso_home_target is None else self._torso_home_target.copy()
        )
        result = apply_view_profile(env, profile, tilt_rad=tilt_rad)
        normalized = str(profile or "default").strip().lower()
        if normalized in {"", "default", "level"} or self._restore_state is None:
            self._restore_state = dict(result.get("snapshot") or {})
        if (
            normalized == "drawer_low_view"
            and torso_pitch_rad is not None
            and self._torso_home_target is not None
            and self._torso_home_target.size > 0
        ):
            self._torso_target = self._torso_home_target.copy()
            pitch_index = 1 if self._torso_target.size > 1 else 0
            self._torso_target[pitch_index] += float(torso_pitch_rad)
        else:
            self._torso_target = None
        result["torso_control_mode"] = (
            "joint_position_controller" if self._torso_target is not None else None
        )
        result["torso_target"] = (
            None if self._torso_target is None else self._torso_target.tolist()
        )
        result["torso_restore_target"] = (
            None
            if self._torso_restore_target is None
            else self._torso_restore_target.tolist()
        )
        return result

    def restore(self, env) -> dict[str, Any]:
        result = apply_view_profile(
            env,
            "default",
            restore_state=self._restore_state,
        )
        self._restore_head_targets = {
            str(row.get("joint_name")): float(
                row.get("ctrl") if row.get("ctrl") is not None else row.get("qpos", 0.0)
            )
            for row in result.get("after", [])
            if row.get("joint_name")
        }
        self._torso_target = self._torso_restore_target
        self._restore_state = None
        result["restore_head_targets"] = dict(self._restore_head_targets)
        result["restore_torso_target"] = (
            None if self._torso_target is None else self._torso_target.tolist()
        )
        return result

    def restore_convergence(
        self,
        env,
        *,
        head_tolerance_rad: float = 0.03,
        torso_tolerance: float = 0.03,
        velocity_tolerance: float = 0.08,
    ) -> dict[str, Any]:
        """Check that the commanded home view has physically settled.

        The drawer macro must not release navigation merely because actuator
        targets were changed.  This check is deliberately fail-closed when a
        real simulator exposes no measurable head/torso state; tests and
        lightweight adapters may explicitly return a verified result instead.
        """
        model = getattr(env, "current_model", None)
        data = getattr(env, "current_data", None)
        head_rows: list[dict[str, Any]] = []
        if model is not None and data is not None:
            for joint_id in range(int(getattr(model, "njnt", 0))):
                name = str(model.joint(joint_id).name or "")
                if name not in self._restore_head_targets:
                    continue
                qpos_addr = int(model.jnt_qposadr[joint_id])
                dof_addr = int(model.jnt_dofadr[joint_id])
                qpos = float(data.qpos[qpos_addr])
                qvel = float(data.qvel[dof_addr])
                target = float(self._restore_head_targets[name])
                head_rows.append(
                    {
                        "joint_name": name,
                        "qpos": qpos,
                        "target": target,
                        "qvel": qvel,
                        "position_error": abs(qpos - target),
                        "converged": abs(qpos - target) <= float(head_tolerance_rad)
                        and abs(qvel) <= float(velocity_tolerance),
                    }
                )

        torso_row: dict[str, Any] | None = None
        torso_target = self._torso_target
        try:
            torso_group = env.current_robot.robot_view.get_move_group("torso")
            torso_qpos = np.asarray(torso_group.joint_pos, dtype=float).reshape(-1)
            if torso_target is not None:
                target = np.asarray(torso_target, dtype=float).reshape(-1)
                if torso_qpos.shape == target.shape:
                    error = float(np.max(np.abs(torso_qpos - target))) if target.size else 0.0
                    torso_row = {
                        "qpos": torso_qpos.tolist(),
                        "target": target.tolist(),
                        "max_position_error": error,
                        "converged": error <= float(torso_tolerance),
                    }
        except (AttributeError, KeyError, TypeError, ValueError):
            torso_row = None

        checked = bool(head_rows or torso_row is not None)
        converged = checked and bool(head_rows) and all(
            bool(row["converged"]) for row in head_rows
        ) and (torso_row is None or bool(torso_row["converged"]))
        return {
            "checked": checked,
            "converged": converged,
            "head": head_rows,
            "torso": torso_row,
            "head_tolerance_rad": float(head_tolerance_rad),
            "torso_tolerance": float(torso_tolerance),
            "velocity_tolerance": float(velocity_tolerance),
        }

    def torso_target(self) -> list[float] | None:
        return None if self._torso_target is None else self._torso_target.tolist()


def drive_joint_group_to_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: Mapping[str, float],
    config: ForceDriveConfig | None = None,
    *,
    robot_lock_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = config or ForceDriveConfig()
    specs = []
    for joint_name, target_value in joint_targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Force drive only supports hinge/slide joints, got type={joint_type}")
        joint_range = [float(value) for value in model.jnt_range[joint_id]]
        lower, upper = min(joint_range), max(joint_range)
        specs.append(
            {
                "joint_name": str(joint_name),
                "joint_id": int(joint_id),
                "joint_type": joint_type,
                "joint_range": joint_range,
                "qpos_addr": int(model.jnt_qposadr[joint_id]),
                "dof_addr": int(model.jnt_dofadr[joint_id]),
                "body_id": int(model.jnt_bodyid[joint_id]),
                "local_axis": np.asarray(model.jnt_axis[joint_id], dtype=float),
                "target_value": float(np.clip(float(target_value), lower, upper)),
            }
        )
    if not specs:
        raise ValueError("Force drive requires at least one joint target")

    stable_substeps = 0
    completed_substeps = 0
    reached = False
    root_body_id = int(model.body_rootid[specs[0]["body_id"]])
    initial_contacts = _robot_articulation_contact_stats(model, data, root_body_id)
    max_contact_count = int(initial_contacts["count"])
    minimum_contact_distance = initial_contacts["minimum_distance"]
    try:
        for substep in range(max(1, int(config.max_physics_substeps))):
            if robot_lock_callback is not None:
                robot_lock_callback()
            data.xfrc_applied[:, :] = 0.0
            all_stable = True
            for spec in specs:
                current = float(data.qpos[spec["qpos_addr"]])
                velocity = float(data.qvel[spec["dof_addr"]])
                error = float(spec["target_value"] - current)
                is_slide = spec["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE
                kp = config.slide_kp if is_slide else config.hinge_kp
                kd = config.slide_kd if is_slide else config.hinge_kd
                max_effort = config.slide_max_effort if is_slide else config.hinge_max_effort
                effort = float(np.clip(kp * error - kd * velocity, -max_effort, max_effort))
                world_axis = data.xmat[spec["body_id"]].reshape(3, 3) @ spec["local_axis"]
                if is_slide:
                    data.xfrc_applied[spec["body_id"], :3] += world_axis * effort
                else:
                    data.xfrc_applied[spec["body_id"], 3:] += world_axis * effort
                all_stable = all_stable and (
                    abs(error) <= config.position_tolerance
                    and abs(velocity) <= config.velocity_tolerance
                )
            mujoco.mj_step(model, data)
            if robot_lock_callback is not None:
                robot_lock_callback()
            completed_substeps = substep + 1
            contact_stats = _robot_articulation_contact_stats(model, data, root_body_id)
            max_contact_count = max(max_contact_count, int(contact_stats["count"]))
            contact_distance = contact_stats["minimum_distance"]
            if contact_distance is not None:
                minimum_contact_distance = (
                    contact_distance
                    if minimum_contact_distance is None
                    else min(minimum_contact_distance, contact_distance)
                )
            if all_stable:
                stable_substeps += 1
                if stable_substeps >= max(1, int(config.stable_substeps)):
                    reached = True
                    break
            else:
                stable_substeps = 0
    finally:
        data.xfrc_applied[:, :] = 0.0
        for spec in specs:
            data.qvel[spec["dof_addr"]] = 0.0
        mujoco.mj_forward(model, data)

    joints = []
    for spec in specs:
        final_value = float(data.qpos[spec["qpos_addr"]])
        final_error = float(spec["target_value"] - final_value)
        joints.append(
            {
                "joint_name": spec["joint_name"],
                "joint_type": (
                    "slide" if spec["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge"
                ),
                "joint_range": list(spec["joint_range"]),
                "target_value": float(spec["target_value"]),
                "final_value": final_value,
                "final_error": final_error,
                "open_fraction": joint_open_fraction(final_value, spec["joint_range"]),
                "reached_target": abs(final_error) <= config.position_tolerance,
            }
        )
    reached = reached or all(joint["reached_target"] for joint in joints)
    final_contacts = _robot_articulation_contact_stats(model, data, root_body_id)
    return {
        "method": "xfrc_applied_group_pd",
        "success": bool(reached),
        "physics_substeps": int(completed_substeps),
        "stable_substeps": int(stable_substeps),
        "robot_target_contacts_before": initial_contacts,
        "robot_target_contacts_after": final_contacts,
        "robot_target_max_contact_count": int(max_contact_count),
        "robot_target_min_contact_distance": minimum_contact_distance,
        "joints": joints,
        "config": asdict(config),
    }


def open_door_root_with_force(
    env,
    root_body_name: str,
    config: ForceDriveConfig | None = None,
) -> dict[str, Any]:
    config = config or ForceDriveConfig()
    groups = collect_door_root_groups(env)
    group = groups.get(str(root_body_name))
    if group is None:
        raise ValueError(
            f"Door root not found: {root_body_name}; available={sorted(groups)}"
        )
    pre_joint_infos = []
    targets = {}
    for leaf in group["leaves"]:
        joint_name = str(leaf["hinge_joint_name"])
        joint_id = mujoco.mj_name2id(
            env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
        before = float(env.current_data.qpos[qpos_addr])
        _closed, opened = joint_closed_open_values(leaf["joint_range"])
        targets[joint_name] = opened
        pre_joint_infos.append(
            {
                "joint_name": joint_name,
                "joint_type": "hinge",
                "joint_range": list(leaf["joint_range"]),
                "joint_value": before,
                "open_fraction": joint_open_fraction(before, leaf["joint_range"]),
            }
        )

    drive = drive_joint_group_to_targets(
        env.current_model,
        env.current_data,
        targets,
        config=config,
    )
    joint_infos = [
        {
            "joint_name": joint["joint_name"],
            "joint_type": joint["joint_type"],
            "joint_range": list(joint["joint_range"]),
            "joint_value": float(joint["final_value"]),
            "open_fraction": float(joint["open_fraction"]),
        }
        for joint in drive["joints"]
    ]
    open_success = bool(joint_infos) and all(
        float(joint["open_fraction"]) >= float(config.open_fraction_threshold)
        for joint in joint_infos
    )
    if config.assume_success and not open_success:
        fractions = [round(float(joint["open_fraction"]), 4) for joint in joint_infos]
        raise ForceInteractionError(
            f"Force interaction did not reach the configured open threshold for "
            f"{root_body_name}: fractions={fractions} threshold={config.open_fraction_threshold}"
        )
    return {
        "root_body_name": str(root_body_name),
        "action": "open",
        "method": "force_pd",
        "success": bool(open_success),
        "pre_state": (
            "closed"
            if pre_joint_infos and max(info["open_fraction"] for info in pre_joint_infos) <= 0.10
            else "ajar"
        ),
        "post_state": "open" if open_success else "ajar",
        "pre_joint_infos": pre_joint_infos,
        "joint_infos": joint_infos,
        "physics_substeps": int(drive["physics_substeps"]),
        "drive": drive,
    }


def yaw_from_pose_matrix(pose) -> float:
    return math.atan2(float(pose[1, 0]), float(pose[0, 0]))
