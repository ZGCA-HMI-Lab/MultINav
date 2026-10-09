"""Collect fixed-clock, constant-speed GT mixed navigation demonstrations.

This is explicitly a kinematic oracle: MuJoCo forward kinematics, real scene
geometry and rendering, not a claim of physically executed arm manipulation.
The robot arm stays parked; ee_target is a material grasp frame, not ee_actual.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from multinav.collection.gt_trajectory import (
    Sampling, navigation_samples, pose_matrix, pose_record, rule_instruction,
    uniform_values, wrap,
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def select_samples(episodes, count, seed):
    """Round-robin interaction types, preferring distinct houses; fixed seed."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i in rng.permutation(len(episodes)).tolist():
        e = episodes[i]
        if set(e["interactive_nav"]["interaction_domains"]) != {"channel", "container"}:
            continue
        key = tuple(sorted(j["type"] for j in e["interactive_nav"]["interactions"]
                           if j["type"].startswith("container_")))
        groups.setdefault(key, []).append(i)
    selected, houses = [], set()
    while len(selected) < count:
        added = False
        for key in sorted(groups):
            candidates = groups[key]
            if not candidates:
                continue
            index = next((i for i in candidates if episodes[i]["house_index"] not in houses), candidates[0])
            candidates.remove(index)
            selected.append(index)
            houses.add(episodes[index]["house_index"])
            added = True
            if len(selected) == count:
                break
        if not added:
            raise ValueError(f"Only {len(selected)} mixed episodes available")
    return selected


def restore_head_camera(model, episode):
    """Restore the frozen benchmark mount, not the newer MJCF camera defaults."""
    spec = next(c for c in episode["cameras"] if c["name"] == "head_camera")
    cid = model.camera("robot_0/head_camera").id
    body = next((model.body(name).id for name in spec["reference_body_names"]
                 if any(model.body(i).name == name for i in range(model.nbody))), None)
    if body is None:
        raise ValueError("Benchmark head camera reference body missing")
    model.cam_bodyid[cid] = body
    model.cam_pos[cid] = spec["camera_offset"]
    quat = np.asarray(spec["camera_quaternion"], dtype=float)
    model.cam_quat[cid] = quat / np.linalg.norm(quat)
    model.cam_fovy[cid] = spec["fov"]


class FullGTCollector:
    def __init__(self, episode, output, args):
        import mujoco
        from multinav.sim import container_scene_probe as probe
        from multinav.collection.full_rollout_recorder import H5StepRolloutRecorder

        self.mj, self.probe = mujoco, probe
        self.episode, self.output, self.args = episode, output, args
        self.sampling = Sampling(args.hz, args.speed, args.yaw_speed,
                                 args.hinge_speed, args.slide_speed, args.handle_speed)
        self.frames, self.segments, self.operations, self.paths = [], [], [], []
        self.operation = None
        self.interaction = None
        self.previous_qpos = None
        self.attachments = []
        self.parking = []
        self.view_overrides = []
        self.ctx = None
        self.renderer = None
        self.recorder = None
        scene_args = argparse.Namespace(seed=args.seed, scene_dataset=episode["scene_dataset"],
                                        data_split=episode["data_split"], robot="rby1", variant="base",
                                        image_width=args.width, image_height=args.height)
        self.ctx = probe.load_scene_context(scene_args, episode["house_index"])
        try:
            self.env = self.ctx.env
            self.model, self.data = self.env.current_model, self.env.current_data
            self.robot = self.env.current_robot.robot_view
            probe.apply_episode_scene_state(self.env, episode)
            probe.apply_robot_state(self.env, {"base_pose": episode["task"]["robot_base_pose"],
                                   "move_groups": episode["robot"]["init_qpos"]})
            from multinav.sim.navigation_posture import ROS_NAVIGATION_ARM_QPOS

            self.initial_posture = {k: list(v) for k, v in ROS_NAVIGATION_ARM_QPOS.items()}
            self.initial_posture.update(head=[0., 0.], torso=[0.] * 6)
            for group, qpos in self.initial_posture.items():
                self.robot.get_move_group(group).joint_pos = np.asarray(qpos)
            restore_head_camera(self.model, episode)
            self.env.camera_manager.setup_cameras(self.env, self.ctx.cfg.camera_config)
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            self.renderer = mujoco.Renderer(self.model, height=args.height, width=args.width)
            self.view = mujoco.MjvOption()
            self.view.geomgroup[5] = 1
            self.external_view = mujoco.MjvOption()
            self.external_view.geomgroup[5] = 0
            # A cutaway external view avoids ceilings/walls hiding the robot.
            # The head view still renders the full scene.
            for geom in range(self.model.ngeom):
                name = self.model.geom(geom).name.lower()
                if "ceiling" in name or "wall" in name:
                    self.model.geom_group[geom] = 5
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.distance = 3.5
            self.camera.elevation = -40
            for name in ("head", "external"):
                (output / "images" / name).mkdir(parents=True, exist_ok=True)
            self.recorder = H5StepRolloutRecorder(output / "trajectory.h5",
                episode_id=episode["interactive_nav"]["case_id"], camera_names=["head", "external"],
                metadata={"execution_mode": "gt_kinematic", "sampling": vars(self.sampling),
                          "frame": "world", "length_unit": "meter", "angle_unit": "radian",
                          "rotation_convention": "extrinsic xyz roll pitch yaw; quaternion wxyz",
                          "ee_target_is_executed": False,
                          "grasp_offset_m": args.grasp_offset})
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.ctx is not None:
            self.probe.close_context(self.ctx)

    def base(self):
        pose = self.robot.base.pose
        return np.array([*pose[:2, 3], math.atan2(pose[1, 0], pose[0, 0])])

    def set_base(self, xy_yaw):
        self.robot.base.pose = pose_matrix([*xy_yaw[:2], 0],
                                           Rotation.from_euler("z", xy_yaw[2]).as_matrix())
        self.mj.mj_forward(self.model, self.data)

    def sample(self, phase, *, endpoint=False, terminal=False, joint_id=None, expected_speed=None):
        index = len(self.frames)
        self.data.time = index / self.sampling.hz
        if self.previous_qpos is not None:
            self.mj.mj_differentiatePos(self.model, self.data.qvel, 1 / self.sampling.hz,
                                       self.previous_qpos, self.data.qpos)
        self.previous_qpos = self.data.qpos.copy()
        self.mj.mj_forward(self.model, self.data)
        base = self.base()
        images = {}
        self.renderer.update_scene(self.data, camera="robot_0/head_camera", scene_option=self.view)
        images["head"] = self.renderer.render().copy()
        self.camera.lookat[:] = [base[0], base[1], 0.8]
        # MuJoCo azimuth is the viewing direction, not eye bearing.
        self.camera.azimuth = math.degrees(base[2])
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.external_view)
        images["external"] = self.renderer.render().copy()
        eye = np.mean([c.pos for c in self.renderer.scene.camera], axis=0)
        camera_poses = {"external": {"lookat": self.camera.lookat.tolist(), "distance": self.camera.distance,
                                     "azimuth_deg": self.camera.azimuth, "elevation_deg": self.camera.elevation,
                                     "fovy_deg": float(self.model.vis.global_.fovy),
                                     "eye_xyz": eye.tolist(),
                                     "cutaway_hidden_geom_group": 5}}
        cid = self.model.camera("robot_0/head_camera").id
        camera_poses["head"] = {"xyz": self.data.cam_xpos[cid].tolist(),
                                "rotation_matrix": self.data.cam_xmat[cid].reshape(3, 3).tolist(),
                                "fovy_deg": float(self.model.cam_fovy[cid])}
        row = {"frame": index, "time": index / self.sampling.hz, "phase": phase,
               "base_xy_yaw": base.tolist(), "endpoint_interval": bool(endpoint),
               "expected_speed": expected_speed, "interaction_id": None,
               "operation_point": None, "ee_target": None, "joint_value": None,
               "joint_name": None, "cameras": camera_poses}
        row['robot_posture'] = {g: self.robot.get_move_group(g).joint_pos.tolist() for g in self.initial_posture}
        if self.operation is not None:
            op = self.operation.world_pose(self.data)
            ee = op.copy()
            ee[:3, 3] -= self.args.grasp_offset * ee[:3, 2]
            row.update(operation_point=pose_record(op), ee_target=pose_record(ee),
                       interaction_id=self.interaction["interaction_id"])
            jid = self.operation.joint_id if joint_id is None else joint_id
            row["joint_value"] = float(self.data.qpos[self.model.jnt_qposadr[jid]])
            row["joint_name"] = self.model.joint(jid).name
        self.recorder.record_step(images=images, action={"type": phase, "vector": []},
            state={"qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(), **row},
            phase=phase, segment=str(len(self.segments)), timestamp_seconds=row["time"],
            dt_seconds=1 / self.sampling.hz, terminal=terminal)
        for name, image in images.items():
            Image.fromarray(image).save(self.output / "images" / name / f"{index:06d}.png")
        self.frames.append(row)

    def navigate(self, step):
        from multinav.collection import gt_map as emi

        self.operation, self.interaction = None, None
        start = self.base()
        goal = np.asarray(step["goal_point"][:2], dtype=float)
        scene_map = emi.build_live_procthor_map(self.model, self.data,
            model_path=str(self.env.current_model_path), px_per_m=50,
            agent_radius=self.ctx.cfg.task_sampler_config.robot_safety_radius,
            treat_all_non_interactive_doorways_as_open=True)
        original_step = copy.deepcopy(step)
        step = copy.deepcopy(step)
        if step.get('interaction_id'):
            step = self.adjust_parking(step, scene_map, start)
            goal = np.asarray(step['goal_point'][:2])
        path = emi.compute_path_from_map(scene_map, start[:2], goal, downscale_factor=1)
        if path is None:
            raise ValueError(f"No collision-inflated GT path {start[:2]} -> {goal}")
        path = np.asarray(path)[:, :2]
        if max(np.linalg.norm(path[0] - start[:2]), np.linalg.norm(path[-1] - goal)) > 0.15:
            raise ValueError("GT path endpoint snapped more than 0.15 m")
        # Replace tiny snapped endpoints instead of inserting centimetre-scale
        # backwards segments (which would cause large spurious turn-in-place).
        path[0], path[-1] = start[:2], goal
        self.paths.append({"plan_step": step, "original_plan_step": original_step, "points": path.tolist()})
        previous = start
        active = None
        for base, phase, endpoint in navigation_samples(path, start[2], step.get("goal_yaw", start[2]), self.sampling):
            if active is None or active["phase"] != phase:
                active = {"phase": phase, "start_frame": len(self.frames), "end_frame": len(self.frames),
                          "distance_m": 0.0, "yaw_delta": 0.0, "interaction_id": step.get("interaction_id")}
                active.update(start_xy=previous[:2].tolist(), start_yaw=float(previous[2]))
                self.segments.append(active)
            self.set_base(base)
            self.sample(phase, endpoint=endpoint,
                        expected_speed=self.sampling.base_speed if phase == "navigate" else self.sampling.yaw_speed)
            active["end_frame"] = len(self.frames) - 1
            active["distance_m"] += float(np.linalg.norm(base[:2] - previous[:2]))
            active["yaw_delta"] += float(wrap(base[2] - previous[2]))
            active['end_xy'] = base[:2].tolist()
            previous = base

    def adjust_parking(self, step, scene_map, start):
        from multinav.collection import gt_map as emi
        from multinav.collection.gt_operation import bind_operation
        from multinav.collection.gt_parking import parking_candidates, sweep_sequence, sweep_clearance

        interaction = next(i for i in self.episode['interactive_nav']['interactions']
                           if i['interaction_id'] == step['interaction_id'])
        original = np.asarray(step['goal_point'][:2], dtype=float)
        op = bind_operation(self.env, interaction, np.array([*original, .8]))
        target_xy = op.world_pose(self.data)[:2, 3].copy()
        steps = self.episode['interactive_nav']['oracle_plan']['steps']
        step_index = next(k for k, s in enumerate(steps) if s == step)
        motions, sweep_ids = [], []
        for upcoming in steps[step_index+1:]:
            if upcoming['type'] == 'navigate':
                break
            if upcoming['type'] != 'open_joint':
                continue
            joint_id = self.model.joint(upcoming['joint_name']).id
            closed, opened = self.probe.joint_closed_open_values(self.model.jnt_range[joint_id].tolist())
            motions.append((joint_id, closed + upcoming.get('target_fraction', 1.)*(opened-closed)))
            sweep_ids.append(upcoming['interaction_id'])
        boxes, sample_counts = sweep_sequence(self.model, self.data, motions)
        radius = max(.4, float(self.ctx.cfg.task_sampler_config.robot_safety_radius))
        required = radius + .1
        for pose, retreat, lateral in parking_candidates(original, target_xy, self.args.parking_backoff):
            clearance = sweep_clearance(pose[:2], boxes)
            if clearance < required:
                continue
            px = np.floor(scene_map.pos_m_to_px(np.array([*pose[:2], 0]))).astype(int)
            if not (0 <= px[0] < scene_map.occupancy.shape[0] and 0 <= px[1] < scene_map.occupancy.shape[1]
                    and scene_map.occupancy[tuple(px)]):
                continue
            path = emi.compute_path_from_map(scene_map, start[:2], pose[:2], downscale_factor=1)
            if path is None or np.linalg.norm(np.asarray(path)[-1, :2]-pose[:2]) > .05:
                continue
            step['goal_point'] = [*pose[:2].tolist(), 0.]
            step['goal_yaw'] = float(pose[2])
            self.parking.append({'interaction_id': step['interaction_id'],
                'original_xy': original.tolist(), 'adjusted_xy_yaw': pose.tolist(),
                'facing_point_xy': target_xy.tolist(), 'backoff_m': retreat, 'lateral_m': lateral,
                'sweep_samples': sum(sample_counts), 'sweep_sample_counts': sample_counts,
                'sweep_interaction_ids': sweep_ids, 'robot_footprint_radius_m': radius,
                'required_sweep_clearance_m': required, 'sweep_clearance_m': clearance,
                'original_sweep_clearance_m': sweep_clearance(original, boxes),
                'criterion': 'sampled_moving_geometry_world_AABB_planar_clearance'})
            return step
        raise ValueError(f"No reachable sweep-clear parking pose for {step['interaction_id']}")

    def articulate(self, step):
        from multinav.collection.gt_operation import bind_operation

        self.interaction = next(i for i in self.episode["interactive_nav"]["interactions"]
                                if i["interaction_id"] == step["interaction_id"])
        base_xyz = np.array([*self.base()[:2], 0.8])
        self.operation = bind_operation(self.env, self.interaction, base_xyz)
        op = self.operation
        record = {**op.metadata, "interaction_id": step["interaction_id"],
                  "local_frame": op.local_frame.tolist(), "pull": None}
        self.operations.append(record)
        # No invented lock: if the asset has a moving handle, turn it first.
        if op.handle_joint_id is not None:
            j = op.handle_joint_id
            lo, hi = self.model.jnt_range[j]
            q = self.data.qpos[self.model.jnt_qposadr[j]]
            target = float(hi if abs(hi - q) >= abs(lo - q) else lo)
            self.joint_motion(j, target, "rotate_handle", self.sampling.handle_speed)
        jid = op.joint_id
        closed, opened = self.probe.joint_closed_open_values(self.model.jnt_range[jid].tolist())
        target = closed + float(step.get("target_fraction", 1)) * (opened - closed)
        qadr = self.model.jnt_qposadr[jid]
        current = float(self.data.qpos[qadr])
        p0 = op.world_pose(self.data)[:3, 3].copy()
        self.data.qpos[qadr] = current + math.copysign(1e-4, target - current)
        self.mj.mj_forward(self.model, self.data)
        dp = op.world_pose(self.data)[:3, 3] - p0
        record["pull"] = bool(np.dot(dp, base_xyz - p0) > 0)
        self.data.qpos[qadr] = current
        self.mj.mj_forward(self.model, self.data)
        speed = self.sampling.slide_speed if self.model.jnt_type[jid] == self.mj.mjtJoint.mjJNT_SLIDE else self.sampling.hinge_speed
        self.joint_motion(jid, target, "open", speed, pull=record["pull"])
        record["final_value"] = float(self.data.qpos[qadr])
        record["target_value"] = float(target)
        record["endpoint_error"] = abs(record["final_value"] - target)
        self.operation, self.interaction = None, None

    def joint_motion(self, jid, target, phase, speed, pull=None):
        from multinav.collection.gt_operation import body_pose

        qadr = self.model.jnt_qposadr[jid]
        start = float(self.data.qpos[qadr])
        attachments = []
        # Drawer contents move with their support in a kinematic GT replay.
        # Preserve the benchmark target's relative pose; this is explicitly
        # an oracle attachment rather than simulated friction/contact.
        if self.model.jnt_type[jid] == self.mj.mjtJoint.mjJNT_SLIDE:
            target_name = self.episode["interactive_nav"]["target"].get("selected_instance")
            if target_name:
                target_pose = self.probe.free_joint_pose(self.env, target_name)
                body = int(self.model.jnt_bodyid[jid])
                if target_pose is not None:
                    local = np.linalg.inv(body_pose(self.data, body)) @ target_pose
                    attachments.append((target_name, body, local))
                    self.attachments.append({"object_name": target_name, "support_joint": self.model.joint(jid).name,
                                             "mode": "oracle_rigid_support", "local_pose": local.tolist()})
        segment = {"phase": phase, "start_frame": len(self.frames),
                   "interaction_id": self.interaction["interaction_id"],
                   "object_category": self.interaction["object_category"], "pull": pull}
        self.segments.append(segment)
        # Include the closed contact state on the same global sample clock.
        self.sample("contact", joint_id=jid)
        for value, endpoint in uniform_values(start, target, speed, self.sampling.hz):
            self.data.qpos[qadr] = value
            self.mj.mj_forward(self.model, self.data)
            for name, body, local in attachments:
                self.probe.set_free_joint_pose(self.env, name, body_pose(self.data, body) @ local)
            self.sample(phase, endpoint=endpoint, joint_id=jid, expected_speed=speed)
        segment["end_frame"] = len(self.frames) - 1

    def run(self):
        write_json(self.output / "episode.json", self.episode)
        self.sample("initial")
        steps = self.episode["interactive_nav"]["oracle_plan"]["steps"]
        for step in steps:
            if step["type"] == "navigate":
                self.navigate(step)
            elif step["type"] == "open_joint":
                self.articulate(step)
            elif step["type"] == "set_view":
                segment = {"phase": "set_view", "start_frame": len(self.frames)}
                targets = {g: np.asarray(step[g + "_qpos"]) for g in ("head", "torso") if g + "_qpos" in step}
                interaction_id = step.get('interaction_id') or (self.paths[-1]['plan_step'].get('interaction_id') if self.paths else None)
                interaction = next((i for i in self.episode['interactive_nav']['interactions']
                                    if i['interaction_id'] == interaction_id), None)
                if interaction and interaction['object_category'].lower() in {'refrigerator', 'fridge'}:
                    targets = {g: np.asarray(self.initial_posture[g]) for g in ('head', 'torso')}
                    self.view_overrides.append({'original_step': step, 'reason': 'keep_fridge_view_level',
                                                'head_qpos': targets['head'].tolist(), 'torso_qpos': targets['torso'].tolist()})
                starts = {g: self.robot.get_move_group(g).joint_pos.copy() for g in targets}
                n = max(1, math.ceil(max(np.max(abs(targets[g] - starts[g])) for g in targets) * self.sampling.hz / self.sampling.yaw_speed)) if targets else 1
                for k in range(1, n + 1):
                    for group in targets:
                        self.robot.get_move_group(group).joint_pos = starts[group] + (targets[group] - starts[group]) * k / n
                    self.sample("set_view")
                segment["end_frame"] = len(self.frames) - 1
                self.segments.append(segment)
            elif step["type"] == "observe_target":
                self.segments.append({"phase": "observe_target", "start_frame": len(self.frames), "end_frame": len(self.frames)})
                self.sample("observe_target")
            else:
                raise ValueError(f"Unsupported oracle step: {step['type']}")
        self.sample("terminal", terminal=True)
        target = self.episode["interactive_nav"]["target"].get("selected_instance")
        terminal_evidence = {}
        if target:
            target_pos = self.data.xpos[self.model.body(target).id]
            terminal_evidence = {"target": target,
                "planar_distance_m": float(np.linalg.norm(self.base()[:2] - target_pos[:2])),
                "visibility_fraction": float(self.env.check_visibility("head_camera", target))}
        instruction = rule_instruction(self.segments, self.episode["interactive_nav"]["target"]["category"])
        self.recorder.finalize(success=True, terminal_reason="oracle_plan_complete")
        # Numeric, named pose arrays make the training contract usable without
        # parsing JSON or reverse engineering robot-dependent qpos indices.
        with h5py.File(self.output / "trajectory.h5", "a") as handle:
            gt = handle.create_group("gt")
            gt.attrs["frame"] = "world"
            gt.attrs["rotation"] = "extrinsic xyz radians; quaternion wxyz"
            gt.create_dataset("base_xy_yaw", data=[f["base_xy_yaw"] for f in self.frames])
            gt.create_dataset("operation_valid", data=[f["operation_point"] is not None for f in self.frames])
            for name in ("operation_point", "ee_target"):
                for key, size in (("xyz_rpy", 6), ("xyz_quat_wxyz", 7)):
                    gt.create_dataset(f"{name}_{key}", data=[f[name][key] if f[name] else [np.nan] * size for f in self.frames])
            gt.create_dataset("joint_value", data=[f["joint_value"] if f["joint_value"] is not None else np.nan for f in self.frames])
        write_json(self.output / "frames.json", self.frames)
        write_json(self.output / "instruction.json", instruction)
        write_json(self.output / "plan.json", {"paths": self.paths, "segments": self.segments,
                   "operations": self.operations, "attachments": self.attachments,
                   "parking_adjustments": self.parking, "view_overrides": self.view_overrides,
                   "initial_posture": self.initial_posture})
        from multinav.collection.gt_topdown import render_topdown

        render_topdown(self.model, self.data, self.external_view, self.frames, self.output)
        with h5py.File(self.output / 'trajectory.h5', 'a') as handle:
            overview = handle.create_group('overview')
            for name in ('topdown_scene', 'topdown_trajectory'):
                with Image.open(self.output / f'{name}.png') as image:
                    overview.create_dataset(name, data=np.asarray(image), compression='lzf')
            overview.attrs['projection_json'] = (self.output/'topdown.json').read_text()
        write_json(self.output / "model_layout.json", {
            "joints": [{"name": self.model.joint(j).name, "type": int(self.model.jnt_type[j]),
                        "qpos_address": int(self.model.jnt_qposadr[j]), "dof_address": int(self.model.jnt_dofadr[j])}
                       for j in range(self.model.njnt)],
            "scene_path": str(self.env.current_model_path),
            "head_fovy_degrees": float(self.model.cam_fovy[self.model.camera("robot_0/head_camera").id])})
        audit = validate_run(self.output, self.sampling)
        audit["terminal_target"] = terminal_evidence
        write_json(self.output / "quality.json", audit)
        return audit


def validate_run(output, sampling):
    frames = json.loads((output / "frames.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    episode = json.loads((output / "episode.json").read_text())
    errors, speed_errors = [], []
    with h5py.File(output / "trajectory.h5") as h:
        n = len(frames)
        if h.attrs["step_count"] != n:
            errors.append("H5 frame count mismatch")
        for name in ("head", "external"):
            if len(h[f"steps/images/{name}"]) != n or len(list((output / "images" / name).glob("*.png"))) != n:
                errors.append(f"Missing {name} frames")
            else:
                for i in range(n):
                    with Image.open(output / "images" / name / f"{i:06d}.png") as image:
                        if not np.array_equal(np.asarray(image), h[f"steps/images/{name}"][i]):
                            errors.append(f"PNG / H5 image mismatch {name} {i}")
        if not np.allclose(np.diff(h["steps/timestamp_seconds"][:]), 1 / sampling.hz, atol=1e-9):
            errors.append("Nonuniform time")
        if not np.allclose(h['gt/base_xy_yaw'][:], [f['base_xy_yaw'] for f in frames], atol=1e-8):
            errors.append('Numeric base poses differ from frame records')
        for name in ('operation_point', 'ee_target'):
            valid = h['gt/operation_valid'][:]
            quaternion = h[f'gt/{name}_xyz_quat_wxyz'][:][valid, 3:]
            if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-7):
                errors.append(f'Nonunit {name} quaternion')
            if not np.isfinite(h[f'gt/{name}_xyz_rpy'][:][valid]).all():
                errors.append(f'Invalid {name} numeric poses')
        if not bool(h['steps/terminal'][-1]) or np.count_nonzero(h['steps/terminal'][:]) != 1:
            errors.append('Missing unique terminal frame')
    for prev, row in zip(frames[:-1], frames[1:]):
        base, old = np.array(row["base_xy_yaw"]), np.array(prev["base_xy_yaw"])
        phase = row["phase"]
        if phase == "navigate":
            actual = np.linalg.norm(base[:2] - old[:2]) * sampling.hz
        elif phase == "turn":
            actual = abs(wrap(base[2] - old[2])) * sampling.hz
        elif phase in {"open", "rotate_handle"} and row["joint_name"] == prev["joint_name"]:
            actual = abs(row["joint_value"] - prev["joint_value"]) * sampling.hz
        else:
            continue
        expected = row["expected_speed"]
        if not row["endpoint_interval"]:
            speed_errors.append(abs(actual - expected))
        elif actual > expected + 1e-5:
            errors.append("Endpoint speed exceeded")
    if speed_errors and max(speed_errors) > 1e-4:
        errors.append("Uniform-speed tolerance exceeded")
    for row in frames:
        if not np.isfinite(row["base_xy_yaw"]).all():
            errors.append("Invalid base pose")
        if row["phase"] in {"contact", "open", "rotate_handle"}:
            if row["operation_point"] is None or not np.isfinite(row["ee_target"]["xyz_rpy"]).all():
                errors.append("Missing operation pose")
        eye_delta = np.asarray(row['cameras']['external']['eye_xyz']) - np.array([*row['base_xy_yaw'][:2], 0])
        yaw = row['base_xy_yaw'][2]
        if np.dot(eye_delta[:2], [np.cos(yaw), np.sin(yaw)]) >= 0 or eye_delta[2] <= 0:
            errors.append('External camera not rear-above')
        for group in ('left_arm', 'right_arm'):
            if not np.allclose(row['robot_posture'][group], plan['initial_posture'][group]):
                errors.append('Navigation arm posture drift')
        interaction = next((i for i in episode['interactive_nav']['interactions'] if i['interaction_id'] == row['interaction_id']), None)
        if interaction and interaction['object_category'].lower() in {'refrigerator', 'fridge'}:
            if not np.allclose(row['robot_posture']['head'], [0, 0]) or not np.allclose(row['robot_posture']['torso'], 0):
                errors.append('Fridge interaction view not level')
    if any(p['sweep_clearance_m'] < p['required_sweep_clearance_m'] for p in plan['parking_adjustments']):
        errors.append('Unsafe parked footprint in articulation sweep')
    if not (output/'topdown_trajectory.png').exists():
        errors.append('Missing top-down trajectory image')
    else:
        with h5py.File(output/'trajectory.h5') as h:
            with Image.open(output/'topdown_trajectory.png') as image:
                if not np.array_equal(np.asarray(image), h['overview/topdown_trajectory'][:]):
                    errors.append('Top-down PNG / H5 mismatch')
    start = episode['task']['robot_base_pose']
    if not np.allclose(frames[0]['base_xy_yaw'][:2], start[:2], atol=1e-6):
        errors.append('Initial position mismatch')
    expected_ids = [s['interaction_id'] for s in episode['interactive_nav']['oracle_plan']['steps'] if s['type'] == 'open_joint']
    if [i for p in plan['parking_adjustments'] for i in p['sweep_interaction_ids']] != expected_ids:
        errors.append('Parking sweep does not cover every planned interaction')
    if [op['interaction_id'] for op in plan['operations']] != expected_ids:
        errors.append('Interaction sequence incomplete')
    if any(op['endpoint_error'] > 1e-6 for op in plan['operations']):
        errors.append('Incomplete opening')
    if plan['paths']:
        final = plan['paths'][-1]['plan_step']
        if not np.allclose(frames[-1]['base_xy_yaw'][:2], final['goal_point'][:2], atol=1e-6):
            errors.append('Final position mismatch')
        if abs(wrap(frames[-1]['base_xy_yaw'][2] - final['goal_yaw'])) > 1e-6:
            errors.append('Final yaw mismatch')
    instruction = json.loads((output/'instruction.json').read_text())
    if not instruction['clauses'] or any(not 0 <= c['start_frame'] <= c['end_frame'] < len(frames) for c in instruction['clauses']):
        errors.append('Instruction frame alignment invalid')
    return {"passed": not errors, "errors": errors, "frames": len(frames),
            "duration_seconds": frames[-1]["time"], "max_uniform_speed_error": max(speed_errors, default=0),
            "endpoint_interval_count": sum(f["endpoint_interval"] for f in frames),
            "interaction_count": len(plan['operations']),
            "movable_handle_count": sum(op['handle_joint_name'] is not None for op in plan['operations']),
            "physical_unlock_verified": False,
            "png_h5_exact_match": not any('image' in e or 'frames' in e for e in errors),
            "execution_mode": "gt_kinematic", "arm_execution_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--hz", type=float, default=5)
    parser.add_argument("--yaw-speed", type=float, default=0.8)
    parser.add_argument("--hinge-speed", type=float, default=0.5)
    parser.add_argument("--slide-speed", type=float, default=0.1)
    parser.add_argument("--handle-speed", type=float, default=0.8)
    parser.add_argument("--grasp-offset", type=float, default=0.03)
    parser.add_argument("--parking-backoff", type=float, default=0.45,
                        help="Minimum retreat from benchmark interaction stop, metres")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--videos", action="store_true", help="Also encode both aligned streams to MP4")
    args = parser.parse_args()
    if not 0 < args.parking_backoff <= 2:
        parser.error('--parking-backoff must be in (0, 2] metres')
    Sampling(args.hz, args.speed, args.yaw_speed, args.hinge_speed, args.slide_speed, args.handle_speed)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty: keep each run independently reproducible")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR", str(args.output.resolve() / "scene_cache"))
    from multinav.evaluation.benchmark_io import load_benchmark_episodes

    benchmark_file, episodes = load_benchmark_episodes(args.benchmark)
    archive = benchmark_file.read_bytes()
    raw = gzip.decompress(archive) if benchmark_file.suffix == '.gz' else archive
    indices = args.indices or select_samples(episodes, args.count, args.seed)
    source_files = [Path(__file__), Path(__file__).with_name('navigation_posture.py')] + [
        Path(__file__).with_name('collection') / name for name in
        ('gt_operation.py', 'gt_trajectory.py', 'gt_parking.py', 'gt_topdown.py', 'gt_map.py', 'full_rollout_recorder.py')]
    write_json(args.output / "manifest.json", {"benchmark": str(benchmark_file.resolve()),
        "benchmark_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed,
        "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "samples": [{"source_index": i, "house_index": episodes[i]["house_index"],
                      "case_id": episodes[i]["interactive_nav"]["case_id"]} for i in indices]})
    results = []
    for index in indices:
        output = args.output / f"episode_{index:04d}"
        output.mkdir()
        collector = None
        start = time.monotonic()
        row = {"source_index": index, "house_index": episodes[index]["house_index"], "directory": output.name}
        print(f"START {index} house={row['house_index']}", flush=True)
        try:
            collector = FullGTCollector(episodes[index], output, args)
            row.update(collector.run())
            if args.videos:
                import imageio.v2 as imageio

                for name in ('head', 'external'):
                    with imageio.get_writer(str(output / f'{name}.mp4'), fps=args.hz,
                                            codec='libx264', macro_block_size=1) as writer:
                        for image_path in sorted((output / 'images' / name).glob('*.png')):
                            writer.append_data(imageio.imread(image_path))
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
            write_json(output / "failure.json", row)
            print(row["error"], flush=True)
        finally:
            if collector is not None:
                collector.close()
        row["wall_seconds"] = time.monotonic() - start
        results.append(row)
        write_json(args.output / "summary.json", {"requested": len(indices), "completed": len(results),
                    "passed": sum(bool(r["passed"]) for r in results), "results": results})
        print(f"END {index} passed={row['passed']} seconds={row['wall_seconds']:.1f}", flush=True)
    from multinav.sim.render_full_gt import build_viewer

    print(build_viewer(args.output.resolve()), flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
