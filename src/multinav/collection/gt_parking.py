"""Conservative planar clearance against an articulation's sampled sweep."""
import itertools

import mujoco
import numpy as np

from .gt_operation import descendants


def sweep_boxes(model, data, joint_id, target):
    """World AABBs of moving geometry at <= 2 degree / 1 cm intervals.

    This is a conservative footprint check, not an arm feasibility certificate.
    Restore all simulator state after this read-only preview.
    """
    bodies = set(descendants(model, int(model.jnt_bodyid[joint_id])))
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
    address = model.jnt_qposadr[joint_id]
    original = float(data.qpos[address])
    spacing = .01 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else np.deg2rad(2)
    values = np.linspace(original, target, max(2, int(np.ceil(abs(target-original)/spacing))+1))
    boxes = []
    try:
        for value in values:
            data.qpos[address] = value
            mujoco.mj_forward(model, data)
            for geom in geoms:
                rotation = data.geom_xmat[geom].reshape(3, 3)
                center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
                half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
                if center[2] + half[2] < .05 or center[2] - half[2] > 1.9:
                    continue
                boxes.append([center[:2] - half[:2], center[:2] + half[:2]])
    finally:
        data.qpos[address] = original
        mujoco.mj_forward(model, data)
    if not boxes:
        raise ValueError("No moving geometry for parking clearance")
    return np.asarray(boxes), len(values)


def sweep_clearance(xy, boxes):
    delta = np.maximum(np.maximum(boxes[:, 0] - xy, xy - boxes[:, 1]), 0)
    return float(np.linalg.norm(delta, axis=1).min())


def sweep_sequence(model, data, motions):
    """Union all openings at one stop, respecting earlier joint transitions."""
    original = data.qpos.copy()
    boxes, counts = [], []
    try:
        for joint_id, target in motions:
            part, count = sweep_boxes(model, data, joint_id, target)
            boxes.append(part); counts.append(count)
            data.qpos[model.jnt_qposadr[joint_id]] = target
            mujoco.mj_forward(model, data)
    finally:
        data.qpos[:] = original
        mujoco.mj_forward(model, data)
    return np.concatenate(boxes), counts


def parking_candidates(original_xy, target_xy, backoff):
    direction = np.asarray(target_xy) - original_xy
    direction /= np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    offsets = sorted(itertools.product(np.arange(backoff, 2.01, .2), [0, -.3, .3, -.6, .6, -1., 1.]),
                     key=lambda t: t[0] + abs(t[1]) * 1.5)
    for retreat, side in offsets:
        xy = original_xy - direction * retreat + lateral * side
        heading = np.arctan2(target_xy[1] - xy[1], target_xy[0] - xy[0])
        yield np.array([*xy, heading]), float(retreat), float(side)
