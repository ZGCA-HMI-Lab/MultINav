"""Bind a material contact frame to the moving link of a real MuJoCo asset."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .gt_trajectory import pose_matrix


def body_pose(data, body_id):
    return pose_matrix(data.xpos[body_id], data.xmat[body_id].reshape(3, 3))


def descendants(model, parent):
    result = []
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != parent:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == parent:
            result.append(body)
    return result


def mesh_handle_components(model, data, bodies, base_xyz):
    """Find slender disconnected handle bars inside a combined visual mesh.

    Preserve OBJ seams: welding connects the handle's mounting screws to the
    panel, losing the separate bar. Large faces and tiny screws are rejected.
    """
    import trimesh

    candidates = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body not in bodies or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            continue
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        mesh_obj = trimesh.Trimesh(vertices=model.mesh_vert[va:va + vn],
                                  faces=model.mesh_face[fa:fa + fn], process=False)
        rotation = data.geom_xmat[geom].reshape(3, 3)
        for component in mesh_obj.split(only_watertight=False):
            vertices = np.asarray(component.vertices)
            if len(vertices) < 8:
                continue
            centered = vertices - vertices.mean(axis=0)
            _, axes = np.linalg.eigh(centered.T @ centered)
            projected = centered @ axes
            extents = np.ptp(projected, axis=0)
            ordered = np.sort(extents)
            if not (0.06 < ordered[2] < 0.7 and 0.001 < ordered[0] < 0.06 and ordered[1] < 0.06):
                continue
            center_local = vertices.mean(axis=0) + axes @ ((projected.min(axis=0) + projected.max(axis=0)) / 2)
            center = data.geom_xpos[geom] + rotation @ center_local
            tangent = rotation @ axes[:, np.argmax(extents)]
            candidates.append((float(np.linalg.norm(center - base_xyz)), body, geom, center,
                               "mesh_handle_component", tangent))
    return candidates


@dataclass
class OperationPoint:
    body_id: int
    local_frame: np.ndarray
    joint_id: int
    handle_joint_id: int | None
    metadata: dict = field(default_factory=dict)

    def world_pose(self, data):
        return body_pose(data, self.body_id) @ self.local_frame


def bind_operation(env, interaction, base_xyz):
    model, data = env.current_model, env.current_data
    joint = model.joint(interaction["joint_name"]).id
    root = int(model.jnt_bodyid[joint])
    bodies = descendants(model, root)
    # Named handle bodies/geoms preserve the moving handle joint, unlike a
    # point attached to the door hinge or to the static container root.
    candidates = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body not in bodies:
            continue
        name = (model.body(body).name + " " + model.geom(geom).name).lower()
        # THOR anonymizes body names. Its articulation handle is commonly a
        # small visual leaf mesh attached to the moving door/drawer link.
        leaf = body != root and not np.any(model.body_parentid[1:] == body)
        half = np.sort(model.geom_aabb[geom, 3:])
        handle_shape = leaf and half[0] < 0.04 and half[1] < 0.09 and half[2] < 0.5
        visual = model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
        if "handle" in name or (handle_shape and visual):
            rotation = data.geom_xmat[geom].reshape(3, 3)
            center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
            source = "named_handle_geometry" if "handle" in name else "leaf_handle_geometry"
            tangent = rotation[:, np.argmax(model.geom_aabb[geom, 3:])]
            candidates.append((float(np.linalg.norm(center - base_xyz)), body, geom, center, source, tangent))
    if not candidates:
        candidates = mesh_handle_components(model, data, bodies, base_xyz)
    handle_joint = None
    if candidates:
        _, body, geom, center, source, tangent = min(candidates, key=lambda c: c[0])
        # z points from the robot into the contact; x is the handle's longest
        # extent projected onto its tangent plane. Store this convention.
        z = center - base_xyz
        z[2] = 0
        z /= max(np.linalg.norm(z), 1e-9)
        x = tangent
        x = x - z * np.dot(x, z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        world = pose_matrix(center, np.column_stack([x, y, z]))
        ancestor = body
        while ancestor != root and ancestor:
            for j in range(model.njnt):
                if model.jnt_bodyid[j] == ancestor and "handle" in model.joint(j).name.lower():
                    handle_joint = j
                    break
            ancestor = int(model.body_parentid[ancestor])
        metadata = {"source": source, "geom": model.geom(geom).name,
                    "physical_handle_center_verified": source == "named_handle_geometry"}
    else:
        # Many THOR containers have one combined mesh. Use the asset's
        # measured per-joint grasp library, never the whole object's centroid.
        from molmo_spaces.env.data_views import MlSpacesArticulationObject
        from molmo_spaces.utils.grasps import get_joint_grasps

        obj = MlSpacesArticulationObject(data=data, object_name=interaction["object_name"])
        index = obj.joint_names.index(interaction["joint_name"])
        poses, _ = get_joint_grasps(env, obj, index, grasp_libraries=["droid"])
        selected = int(np.argmin(np.linalg.norm(poses[:, :3, 3] - base_xyz, axis=1)))
        world, body = poses[selected], root
        metadata = {"source": "asset_joint_grasp", "candidate_index": selected,
                    "candidate_count": len(poses), "physical_handle_center_verified": False}
    metadata.update({"body_name": model.body(body).name,
                     "joint_name": interaction["joint_name"],
                     "handle_joint_name": model.joint(handle_joint).name if handle_joint is not None else None,
                     "latch_simulated": False,
                     "grasp_frame_convention": "local z approach; local x handle tangent"})
    return OperationPoint(body, np.linalg.inv(body_pose(data, body)) @ world,
                          joint, handle_joint, metadata)
