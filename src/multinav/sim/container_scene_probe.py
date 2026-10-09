from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import Door, MlSpacesArticulationObject, MlSpacesFreeJointBody
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.constants.object_constants import RECEPTACLE_TYPES_THOR
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat
from molmo_spaces.utils.rendering_utils import get_geom_seg_mask
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

if TYPE_CHECKING:
    from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts/InteractiveNav/output/container_scene_probe"
DEFAULT_NAV_BENCHMARK_JSON = (
    REPO_ROOT
    / "assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
)
# Writable scene mirrors default to the system temporary directory. Callers may
# override this per worker with INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR.
WRITABLE_ASSET_MIRROR = Path(
    os.environ.get(
        "INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR",
        os.path.join(tempfile.gettempdir(), "container_scene_probe_assets"),
    )
)
CONTAINER_TOKENS = (
    "drawer",
    "cabinet",
    "fridge",
    "refrigerator",
    "microwave",
    "oven",
    "dishwasher",
    "box",
    "chestofdrawers",
)
PORTABLE_PREFERRED_TOKENS = (
    "apple",
    "mug",
    "bowl",
    "atomizer",
    "alarmclock",
    "vase",
    "laptop",
    "basketball",
)
DEFAULT_LEFT_ARM_QPOS = np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0], dtype=np.float32)
DEFAULT_RIGHT_ARM_QPOS = np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0], dtype=np.float32)
# Keep the head camera level, matching the navigation ROS simulator.  The
# second RBY1 head joint is the pitch/tilt joint (radians); zero is the
# navigation sensor pose (0 degrees pitch).
DEFAULT_HEAD_QPOS = np.array([0.0, 0.0], dtype=np.float32)
FORCE_DRIVE_TOLERANCE = 0.01
DEFAULT_RBY1_INTERACTION_CAMERAS = ("head_camera", "wrist_camera_l", "wrist_camera_r", "camera_follower")


def default_rby1_episode_camera_specs() -> list[dict[str, Any]]:
    """Return deterministic RBY1 camera mounts used by official frozen episodes."""
    return [
        {
            "name": "head_camera",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/link_head_2"],
            "camera_offset": [0.05, 0.0, 0.05],
            "lookat_offset": [0.0, 0.0, 0.08],
            "camera_quaternion": [0.5, 0.5, -0.5, -0.5],
            "fov": 139.0,
            "record_depth": False,
        },
        {
            "name": "wrist_camera_l",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/link_left_arm_6"],
            "camera_offset": [0.0, -0.1, -0.15],
            "lookat_offset": [0.0, 0.0, 0.08],
            "camera_quaternion": [0.0, 0.0, -0.258819, 0.965926],
            "fov": 58.0,
            "record_depth": False,
        },
        {
            "name": "wrist_camera_r",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/link_right_arm_6"],
            "camera_offset": [0.0, 0.1, -0.15],
            "lookat_offset": [0.0, 0.0, 0.08],
            "camera_quaternion": [0.965926, -0.258819, 0.0, 0.0],
            "fov": 58.0,
            "record_depth": False,
        },
        {
            "name": "camera_follower",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/base"],
            "camera_offset": [-1.3, 0.0, 2.7],
            "lookat_offset": [0.0, 0.0, 0.08],
            "camera_quaternion": [0.653288, 0.270582, -0.270582, -0.653288],
            "fov": 58.0,
            "record_depth": False,
        },
    ]


def default_rby1_episode_qpos() -> dict[str, list[float]]:
    """Return a complete RBY1M state compatible with JSON episode replay."""
    return {
        "base": [0.0, 0.0, 0.0],
        "torso": [0.0] * 6,
        "left_arm": DEFAULT_LEFT_ARM_QPOS.astype(float).tolist(),
        "right_arm": DEFAULT_RIGHT_ARM_QPOS.astype(float).tolist(),
        "left_gripper": [-0.05, 0.05],
        "right_gripper": [-0.05, 0.05],
        "head": DEFAULT_HEAD_QPOS.astype(float).tolist(),
    }


@dataclass
class LoadedContext:
    cfg: Any
    sampler: Any
    task: Any | None
    initial_head_qpos: np.ndarray | None = None
    initial_torso_qpos: np.ndarray | None = None

    @property
    def env(self):
        return self.sampler.env


@dataclass
class RBY1InteractionRequest:
    """Algorithm-facing request for opening one articulated scene object."""

    house_ind: int
    interaction_kind: str
    target_name: str
    joint_index: int = 0
    robot_pose_mode: str = "current_or_adjust"
    robot_base_pose: list[float] | np.ndarray | None = None
    door_arm: str = "auto"
    container_arm: str = "auto"
    approach_distance: float = 0.5
    min_base_clearance: float = 0.15
    max_approach_distance: float = 1.2
    max_base_adjustment_distance: float = 0.75
    max_base_adjustment_steps: int = 120
    allow_back_approach: bool = False
    door_tcp_offset: float = 0.03
    door_max_steps_per_waypoint: int = 35
    door_max_planning_reattempts: int = 5
    door_joint_position_tolerance: float = 0.10
    door_articulation_delta_deg: float = 7.0
    allow_force_fallback: bool = False
    force_fallback_target_fraction: float = 1.0
    # RBY1 scenes currently use a 2 ms MuJoCo timestep. 1000 force steps
    # therefore represent the intended ~2 s continuous interaction window.
    force_fallback_max_steps: int = 1000
    force_duration_seconds: float = 2.0
    success_threshold: float = 0.8
    max_steps: int = 400
    container_max_steps_per_waypoint: int = 30
    container_max_batch_plan_attempts: int = 8
    container_max_planning_reattempts: int = 8
    container_pregrasp_tolerance: float = 0.04
    video_fps: float | None = None
    camera_names: tuple[str, ...] = DEFAULT_RBY1_INTERACTION_CAMERAS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    scene_dataset: str = "procthor-10k"
    data_split: str = "train"
    variant: str = "base"
    seed: int = 0
    curobo_server_url: str | None = None


class SceneOnlyTaskSampler(BaseMujocoTaskSampler):
    """Load a MolmoSpaces scene without sampling a concrete task."""

    def init_scene(self, env) -> None:
        return None

    def randomize_scene(self, env, robot_view) -> None:
        return None

    def _sample_task(self, env) -> BaseMujocoTask:
        raise NotImplementedError("SceneOnlyTaskSampler only loads scenes.")


class RBY1InteractionJsonTaskSampler(JsonEvalTaskSampler):
    """Replay JSON episodes with policy-specific pre-compile helper bodies."""

    def add_auxiliary_objects(self, spec) -> None:
        super().add_auxiliary_objects(spec)
        policy_cls = getattr(self.config.policy_config, "policy_cls", None)
        if policy_cls is not None and hasattr(policy_cls, "add_auxiliary_objects"):
            policy_cls.add_auxiliary_objects(self.config, spec)


def build_scene_config(args: argparse.Namespace) -> NavToObjBaseConfig:
    from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
    from molmo_spaces.configs.camera_configs import (
        FrankaDroidCameraSystem,
        RBY1GoProD455CameraSystem,
    )
    from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
    from molmo_spaces.configs.robot_configs import (
        FloatingRUMRobotConfig,
        FrankaRobotConfig,
        RBY1MOpenCloseConfig,
    )

    cfg = NavToObjBaseConfig()
    cfg.seed = args.seed
    cfg.task_type = "nav_to_obj"
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.num_workers = 1
    cfg.num_threads = 1
    cfg.use_passive_viewer = False
    if "use_filament" in type(cfg).model_fields:
        cfg.use_filament = False
    cfg.record_videos = False
    cfg.task_sampler_config.task_sampler_class = SceneOnlyTaskSampler
    cfg.task_sampler_config.house_inds = []
    cfg.task_sampler_config.house_variant = args.variant
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.randomize_lighting = False
    cfg.task_sampler_config.randomize_textures = False
    cfg.task_sampler_config.randomize_dynamics = False
    cfg.policy_config = AStarNavToObjPolicyConfig()

    if args.robot == "droid":
        cfg.robot_config = FrankaRobotConfig()
        cfg.camera_config = FrankaDroidCameraSystem()
        cfg.camera_config.img_resolution = (320, 240)
    elif args.robot == "rby1":
        cfg.robot_config = RBY1MOpenCloseConfig()
        # Navigation and full-rollout observations use a level head camera.
        # Disable initialization noise as well so every replay starts at 0°
        # pitch rather than inheriting the robot config's 0.6-rad tilt.
        cfg.robot_config.init_qpos["head"] = DEFAULT_HEAD_QPOS.copy()
        cfg.robot_config.init_qpos_noise_range["head"] = np.zeros(2, dtype=float)
        cfg.camera_config = RBY1GoProD455CameraSystem()
    elif args.robot == "rum":
        cfg.robot_config = FloatingRUMRobotConfig()
    else:
        raise ValueError(f"Unsupported robot: {args.robot}")

    # For debug / visibility analysis we want deterministic camera poses across
    # repeated setup_cameras() calls, so disable all camera randomization.
    for cam in cfg.camera_config.cameras:
        if hasattr(cam, "pos_noise_range"):
            cam.pos_noise_range = None
        if hasattr(cam, "orientation_noise_degrees"):
            cam.orientation_noise_degrees = None
        if hasattr(cam, "fov_noise_degrees"):
            cam.fov_noise_degrees = None

    if hasattr(args, "image_width"):
        cfg.camera_config.img_resolution = (args.image_width, args.image_height)
    return cfg


def load_scene_context(args: argparse.Namespace, house_ind: int) -> LoadedContext:
    import molmo_spaces.tasks.task_sampler as task_sampler_module

    manager = get_resource_manager()
    manager.symlink_lock = False
    manager.cache_lock = False
    task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: {}
    cfg = build_scene_config(args)
    cfg.task_sampler_config.house_inds = [house_ind]
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=house_ind)
        original_scene_path = Path(sampler._current_house_scene_path())
        scene_path = prepare_writable_scene_path(original_scene_path)
        sampler.update_scene(scene_path=scene_path)
        env = sampler.env
        if env.current_scene_metadata is None:
            env._scene_metadata = get_scene_metadata(scene_path) or get_scene_metadata(original_scene_path)
        # Enforce the navigation reference pose after scene reset too.  This
        # guards against samplers/models that restore the robot's class-level
        # default (0.6 rad pitch) during initialization.
        initial_head_qpos = set_head_joint_position(env, DEFAULT_HEAD_QPOS)
        initial_torso_qpos = get_torso_joint_position(env)
        apply_default_torso_pose(env, initial_torso_qpos)
        apply_default_arm_pose(env)
        env.camera_manager.setup_cameras(env, cfg.camera_config)
        return LoadedContext(
            cfg=cfg,
            sampler=sampler,
            task=None,
            initial_head_qpos=None if initial_head_qpos is None else initial_head_qpos.copy(),
            initial_torso_qpos=None if initial_torso_qpos is None else initial_torso_qpos.copy(),
        )
    except Exception:
        sampler.close()
        raise


def prepare_writable_scene_path(scene_path: Path) -> str:
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    assets_root = Path(ASSETS_DIR)
    cache_root = Path(
        os.environ.get("MLSPACES_CACHE_DIR", str(Path.home() / ".cache/molmo-spaces-resources"))
    )
    temp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    local_assets_root = REPO_ROOT / "assets"
    tmp_asset_roots = [
        root
        for root in (
            [temp_root / "molmo-spaces-assets-proxy"]
            + sorted(temp_root.glob("container_probe_assets_*"))
        )
        if (root / "scenes").exists()
    ]
    resource_roots = [
        cache_root,
        *[
            root
            for root in sorted(temp_root.glob("container_probe_resources_*"))
            if (root / "scenes").exists()
        ],
    ]
    asset_roots = [
        assets_root,
        local_assets_root,
    ] + tmp_asset_roots
    # Every MuJoCo worker must have an isolated writable mirror.  The old fixed
    # path let concurrent workers replace each other's scene symlinks, causing
    # intermittent ``val_*.xml not found`` failures.  Callers may provide a
    # stable per-worker path; otherwise derive one from the process ID.
    mirror_root = Path(
        os.environ.get(
            "INTERACTIVE_NAV_SCENE_MIRROR",
            str(WRITABLE_ASSET_MIRROR.with_name(f"{WRITABLE_ASSET_MIRROR.name}_{os.getpid()}")),
        )
    )
    mirror_root.mkdir(parents=True, exist_ok=True)

    def usable_source(src: Path) -> Path:
        default_cache = Path.home() / ".cache/molmo-spaces-resources"

        def cache_scene_match(path: Path) -> Path | None:
            for root in asset_roots:
                try:
                    rel = path.relative_to(root / "scenes")
                except ValueError:
                    continue
                for candidate_root in asset_roots:
                    candidate = candidate_root / "scenes" / rel
                    if candidate.exists():
                        return candidate
                parts = rel.parts
                if len(parts) < 2:
                    continue
                dataset = parts[0]
                scene_rel = Path(*parts[1:])
                for resource_root in resource_roots:
                    for candidate in (resource_root / "scenes" / dataset).glob(
                        "*/" + str(scene_rel)
                    ):
                        if candidate.exists():
                            return candidate
            return None

        matched = cache_scene_match(src)
        if matched is not None:
            return matched
        cur = src
        seen: set[Path] = set()
        for _ in range(8):
            matched = cache_scene_match(cur)
            if matched is not None:
                return matched
            if cur.exists() and not cur.is_symlink():
                return cur
            if not cur.is_symlink() or cur in seen:
                break
            seen.add(cur)
            target = Path(os.readlink(cur))
            if not target.is_absolute():
                target = cur.parent / target
            try:
                rel = target.relative_to(default_cache)
            except ValueError:
                rel = None
            if rel is not None:
                redirected = cache_root / rel
                if redirected.exists():
                    return redirected
            cur = target
        if src.exists():
            return src
        return src

    def ensure_symlink(dst: Path, src: Path) -> None:
        src = usable_source(src)
        if dst.is_symlink() and (not dst.exists() or Path(os.readlink(dst)) != src):
            dst.unlink()
        if src.exists() and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)

    for top_level in ("objects", "robots", "grasps"):
        src = assets_root / top_level
        dst = mirror_root / top_level
        ensure_symlink(dst, src)

    refs_src = assets_root / "scenes" / "refs"
    refs_dst = mirror_root / "scenes" / "refs"
    refs_dst.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(refs_dst, refs_src)

    rel_scene = None
    for root in asset_roots:
        try:
            rel_scene = scene_path.relative_to(root / "scenes")
            break
        except ValueError:
            continue
    if rel_scene is None:
        raise ValueError(f"Scene path is not under a known assets/scenes root: {scene_path}")
    dst_scene = mirror_root / "scenes" / rel_scene
    dst_scene.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(dst_scene, scene_path)

    resolved_scene_path = usable_source(scene_path)
    asset_stem = resolved_scene_path.stem.removesuffix("_ceiling").removesuffix("_map")
    for sibling in resolved_scene_path.parent.glob(f"{asset_stem}*"):
        if sibling == resolved_scene_path:
            continue
        if sibling.is_dir() and not sibling.name.endswith("_assets"):
            continue
        sibling_dst = dst_scene.parent / sibling.name
        ensure_symlink(sibling_dst, sibling)

    scene_assets_dir = resolved_scene_path.parent / f"{asset_stem}_assets"
    if scene_assets_dir.exists():
        dst_assets_dir = dst_scene.parent / scene_assets_dir.name
        ensure_symlink(dst_assets_dir, scene_assets_dir)
    return str(dst_scene)


def close_context(ctx: LoadedContext) -> None:
    ctx.sampler.close()


def apply_default_arm_pose(env) -> None:
    robot_view = env.current_robot.robot_view
    if "left_arm" in robot_view.move_group_ids():
        left = robot_view.get_move_group("left_arm")
        if np.asarray(left.joint_pos).shape == DEFAULT_LEFT_ARM_QPOS.shape:
            left.joint_pos = DEFAULT_LEFT_ARM_QPOS.copy()
    if "right_arm" in robot_view.move_group_ids():
        right = robot_view.get_move_group("right_arm")
        if np.asarray(right.joint_pos).shape == DEFAULT_RIGHT_ARM_QPOS.shape:
            right.joint_pos = DEFAULT_RIGHT_ARM_QPOS.copy()
    for gripper_name in ("left_gripper", "right_gripper"):
        if gripper_name not in robot_view.move_group_ids():
            continue
        gripper = robot_view.get_move_group(gripper_name)
        target = np.asarray(default_rby1_episode_qpos()[gripper_name], dtype=float)
        if np.asarray(gripper.joint_pos).shape == target.shape:
            gripper.joint_pos = target
    mujoco.mj_forward(env.current_model, env.current_data)


def get_torso_joint_position(env) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "torso" not in robot_view.move_group_ids():
        return None
    return np.asarray(robot_view.get_move_group("torso").joint_pos, dtype=np.float32).copy()


def get_default_torso_qpos(env, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
        current = get_torso_joint_position(env)
        if current is not None and fallback.shape == current.shape:
            return fallback.copy()
    try:
        torso_qpos = env.current_robot.exp_config.robot_config.init_qpos.get("torso")
    except Exception:
        torso_qpos = None
    if torso_qpos is None:
        current = get_torso_joint_position(env)
        return None if current is None else np.zeros_like(current, dtype=np.float32)
    torso_qpos = np.asarray(torso_qpos, dtype=np.float32).reshape(-1)
    current = get_torso_joint_position(env)
    if current is not None and torso_qpos.shape != current.shape:
        return np.zeros_like(current, dtype=np.float32)
    return torso_qpos.copy()


def set_torso_joint_position(env, torso_qpos: np.ndarray | list[float]) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "torso" not in robot_view.move_group_ids():
        return None
    torso_group = robot_view.get_move_group("torso")
    torso_qpos = np.asarray(torso_qpos, dtype=np.float32).reshape(-1)
    limits = np.asarray(torso_group.joint_pos_limits, dtype=np.float32)
    if torso_qpos.shape[0] != limits.shape[0]:
        raise ValueError(f"Expected torso qpos shape {(limits.shape[0],)}, got {torso_qpos.shape}")
    clipped = np.clip(torso_qpos, limits[:, 0], limits[:, 1]).astype(np.float32)
    torso_group.joint_pos = clipped.copy()
    torso_group.ctrl = torso_group.noop_ctrl
    mujoco.mj_forward(env.current_model, env.current_data)
    return clipped


def apply_default_torso_pose(env, default_qpos: np.ndarray | None = None) -> np.ndarray | None:
    default_qpos = get_default_torso_qpos(env, fallback=default_qpos)
    if default_qpos is None:
        return None
    return set_torso_joint_position(env, default_qpos)


def lean_torso_for_drawer_view(
    env,
    default_qpos: np.ndarray | None = None,
    pitch_delta: float = 0.35,
) -> np.ndarray | None:
    base = get_default_torso_qpos(env, fallback=default_qpos)
    if base is None:
        return None
    target = base.copy()
    if target.shape[0] < 2:
        return set_torso_joint_position(env, target)
    # RBY1 torso_1 is the lower pitch joint; positive values move the head
    # forward in the robot frame, giving a "lean over the drawer" viewpoint.
    target[1] = target[1] + float(pitch_delta)
    return set_torso_joint_position(env, target)


def get_default_head_qpos(env, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if fallback.shape == DEFAULT_HEAD_QPOS.shape:
            return fallback.copy()
    current = get_head_joint_position(env)
    if current is not None and current.shape == DEFAULT_HEAD_QPOS.shape:
        return current.copy()
    try:
        head_qpos = env.current_robot.exp_config.robot_config.init_qpos.get("head")
    except Exception:
        head_qpos = None
    if head_qpos is None:
        return DEFAULT_HEAD_QPOS.copy()
    head_qpos = np.asarray(head_qpos, dtype=np.float32).reshape(-1)
    if head_qpos.shape != DEFAULT_HEAD_QPOS.shape:
        return DEFAULT_HEAD_QPOS.copy()
    return head_qpos.copy()


def get_head_joint_position(env) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "head" not in robot_view.move_group_ids():
        return None
    return np.asarray(robot_view.get_move_group("head").joint_pos, dtype=np.float32).copy()


def set_head_joint_position(env, head_qpos: np.ndarray | list[float]) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "head" not in robot_view.move_group_ids():
        return None
    head_qpos = np.asarray(head_qpos, dtype=np.float32).reshape(-1)
    head_group = robot_view.get_move_group("head")
    limits = np.asarray(head_group.joint_pos_limits, dtype=np.float32)
    if head_qpos.shape[0] != limits.shape[0]:
        raise ValueError(f"Expected head qpos shape {(limits.shape[0],)}, got {head_qpos.shape}")
    clipped = np.clip(head_qpos, limits[:, 0], limits[:, 1]).astype(np.float32)
    head_group.joint_pos = clipped.copy()
    head_group.ctrl = head_group.noop_ctrl
    mujoco.mj_forward(env.current_model, env.current_data)
    return clipped


def apply_default_head_pose(env, default_qpos: np.ndarray | None = None) -> np.ndarray | None:
    default_qpos = get_default_head_qpos(env, fallback=default_qpos)
    return set_head_joint_position(env, default_qpos)


def lower_head_for_drawer_view(env, tilt_delta: float = 0.35, pan: float | None = None) -> np.ndarray | None:
    current = get_head_joint_position(env)
    if current is None:
        return None
    target = current.copy()
    if pan is not None:
        target[0] = float(pan)
    target[1] = current[1] + float(tilt_delta)
    return set_head_joint_position(env, target)


def raise_head_to_pose(env, head_qpos: np.ndarray | list[float] | None = None) -> np.ndarray | None:
    target = get_default_head_qpos(env) if head_qpos is None else np.asarray(head_qpos, dtype=np.float32)
    return set_head_joint_position(env, target)


def token_match(value: str, tokens: tuple[str, ...]) -> bool:
    lower = value.lower()
    return any(token in lower for token in tokens)


def safe_body_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return body_aabb(model, data, body_id, visible_only=True)
    except Exception:
        return data.xpos[body_id].copy(), np.zeros(3)


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,):
        return np.eye(3)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.eye(3)
    quat = quat / norm
    out = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def body_pose_mat(env, body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = env.current_data.xmat[body_id].reshape(3, 3).copy()
    pose[:3, 3] = env.current_data.xpos[body_id].copy()
    return pose


def pos_quat_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R.from_quat(np.asarray(quat, dtype=float), scalar_first=True).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=float)
    return pose


def mat_to_pos_quat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=float)
    pos = pose[:3, 3].copy()
    quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    return pos, quat


def aabb_bounds(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=float) / 2.0
    center = np.asarray(center, dtype=float)
    return center - half, center + half


def aabb_overlap_metrics(
    a_center: np.ndarray,
    a_size: np.ndarray,
    b_center: np.ndarray,
    b_size: np.ndarray,
) -> dict[str, Any]:
    a_min, a_max = aabb_bounds(np.asarray(a_center, dtype=float), np.asarray(a_size, dtype=float))
    b_min, b_max = aabb_bounds(np.asarray(b_center, dtype=float), np.asarray(b_size, dtype=float))
    inter_size = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    inter_vol = float(np.prod(inter_size))
    a_vol = max(float(np.prod(np.maximum(np.asarray(a_size, dtype=float), 1e-6))), 1e-9)
    b_vol = max(float(np.prod(np.maximum(np.asarray(b_size, dtype=float), 1e-6))), 1e-9)
    return {
        "inter_size": inter_size,
        "inter_vol": inter_vol,
        "ratio_of_a": inter_vol / a_vol,
        "ratio_of_b": inter_vol / b_vol,
    }


def compute_box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    offsets = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    return center + offsets * half


def add_box_to_ax(ax, center: np.ndarray, size: np.ndarray, color: str, label: str, alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    corners = compute_box_corners(center, size)
    faces = [
        [corners[idx] for idx in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (4, 7, 3, 0),
        )
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidths=0.8)
    ax.add_collection3d(poly)
    ax.text(center[0], center[1], center[2], label, color=color, fontsize=7)


def set_axes_equal_3d(ax) -> None:
    x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
    y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
    z_limits = np.asarray(ax.get_zlim3d(), dtype=float)
    centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()], dtype=float)
    radius = 0.5 * max(
        float(np.ptp(x_limits)),
        float(np.ptp(y_limits)),
        float(np.ptp(z_limits)),
        1e-3,
    )
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(max(0.0, centers[2] - radius), centers[2] + radius)


def save_relation_plot(
    output_path: Path,
    container_rec: dict[str, Any],
    object_rec: dict[str, Any],
    relation: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    add_box_to_ax(
        ax,
        np.asarray(container_rec["aabb_center"], dtype=float),
        np.asarray(container_rec["aabb_size"], dtype=float),
        "tab:blue",
        f"container:{container_rec['name']}",
        0.15,
    )
    add_box_to_ax(
        ax,
        np.asarray(object_rec["aabb_center"], dtype=float),
        np.asarray(object_rec["aabb_size"], dtype=float),
        "tab:red",
        f"object:{object_rec['name']}",
        0.45,
    )

    c_center = np.asarray(container_rec["aabb_center"], dtype=float)
    o_center = np.asarray(object_rec["aabb_center"], dtype=float)
    ax.plot(
        [c_center[0], o_center[0]],
        [c_center[1], o_center[1]],
        [c_center[2], o_center[2]],
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_title(
        f"{relation['label']} | score={relation['score']:.2f}\n"
        f"{container_rec['name']} -> {object_rec['name']}"
    )
    all_pts = np.vstack(
        [
            compute_box_corners(container_rec["aabb_center"], container_rec["aabb_size"]),
            compute_box_corners(object_rec["aabb_center"], object_rec["aabb_size"]),
        ]
    )
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    spans = np.maximum(maxs - mins, 0.1)
    max_span = float(np.max(spans)) * 0.6
    mid = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - max_span, mid[0] + max_span)
    ax.set_ylim(mid[1] - max_span, mid[1] + max_span)
    ax.set_zlim(max(0.0, mid[2] - max_span), mid[2] + max_span)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_segmentation_preview(path: Path, seg_frame: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    seg = np.asarray(seg_frame[..., :2], dtype=np.int32)
    geom_ids = seg[..., 0]
    obj_types = seg[..., 1]
    preview = np.zeros((*geom_ids.shape, 3), dtype=np.uint8)
    preview[..., 0] = (geom_ids * 37) % 255
    preview[..., 1] = (geom_ids * 67) % 255
    preview[..., 2] = (obj_types * 97) % 255
    Image.fromarray(preview).save(path)


def save_object_mask_preview(path: Path, mask: np.ndarray | None) -> None:
    from PIL import Image

    if mask is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def scene_category(name: str, category: str | None) -> str:
    lower_name = name.lower()
    lower_cat = (category or "").lower()
    if "door" in lower_name or "door" in lower_cat or "gate" in lower_name:
        return "portal"
    if token_match(lower_name, CONTAINER_TOKENS) or token_match(lower_cat, CONTAINER_TOKENS):
        return "container"
    return "other"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def joint_closed_open_values(joint_range: list[float]) -> tuple[float, float]:
    """Map an articulation range to semantic closed/open endpoints."""
    values = [float(value) for value in joint_range]
    if not values:
        raise ValueError("Joint range is empty")
    closed = min(values, key=lambda value: abs(value))
    open_value = max(values, key=lambda value: abs(value - closed))
    return float(closed), float(open_value)


def collect_scene_records(ctx: LoadedContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    objects_meta = env.current_scene_metadata.get("objects", {})
    records: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []

    for object_name, meta in objects_meta.items():
        try:
            body_id = om.get_object_body_id(object_name)
        except Exception:
            continue

        center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
        category = meta.get("category") or om.get_annotation_category(object_name)
        support = om.get_support_below(object_name, RECEPTACLE_TYPES_THOR)
        room = om.infer_room_name(object_name, RECEPTACLE_TYPES_THOR)
        parent_chain = om.get_parent_chain_names(object_name)
        is_articulable = om.is_object_articulable(object_name)
        is_structural = om.is_structural(object_name)
        has_free_joint = om.has_free_joint(object_name)

        rec = {
            "name": object_name,
            "body_id": int(body_id),
            "category": category,
            "position": env.current_data.xpos[body_id].copy(),
            "quat": env.current_data.xquat[body_id].copy(),
            "aabb_center": center.copy(),
            "aabb_size": size.copy(),
            "parent": meta.get("parent"),
            "parent_chain": parent_chain,
            "support_below": support,
            "room": room,
            "is_structural": is_structural,
            "is_articulable": is_articulable,
            "has_free_joint": has_free_joint,
            "interaction_group": scene_category(object_name, category),
            "is_receptacle": om.has_receptacle_site(object_name),
            "asset_id": meta.get("asset_id"),
        }
        records.append(rec)

        if rec["interaction_group"] == "container" and is_articulable:
            art_obj = om.get_object_by_name(object_name)
            if not isinstance(art_obj, MlSpacesArticulationObject):
                continue
            joints = []
            for joint_index, joint_name in enumerate(art_obj.joint_names):
                joint_range = [float(v) for v in art_obj.get_joint_range(joint_index)]
                closed_value, open_value = joint_closed_open_values(joint_range)
                joints.append(
                    {
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "joint_type": str(art_obj.get_joint_type(joint_index)).split(".")[-1],
                        "joint_range": joint_range,
                        "current_value": float(art_obj.get_joint_position(joint_index)),
                        "closed_value": closed_value,
                        "open_value": open_value,
                    }
                )
            container_rec = copy.deepcopy(rec)
            container_rec["joints"] = joints
            containers.append(container_rec)

    return records, containers


def collect_door_records(ctx: LoadedContext) -> list[dict[str, Any]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    door_records: list[dict[str, Any]] = []
    for door_name in om.find_door_names():
        try:
            door = Door(door_name, env.current_data)
            hinge_idx = door.get_hinge_joint_index()
            joint_range = [float(v) for v in door.get_joint_range(hinge_idx)]
            closed_value, open_value = joint_closed_open_values(joint_range)
            center, size = safe_body_aabb(env.current_model, env.current_data, door.body_id)
            door_records.append(
                {
                    "name": door_name,
                    "interaction_group": "portal",
                    "body_id": int(door.body_id),
                    "category": "Door",
                    "position": env.current_data.xpos[door.body_id].copy(),
                    "quat": env.current_data.xquat[door.body_id].copy(),
                    "aabb_center": center.copy(),
                    "aabb_size": size.copy(),
                    "hinge_joint_index": hinge_idx,
                    "hinge_joint_name": door.joint_names[hinge_idx],
                    "hinge_joint_range": joint_range,
                    "closed_value": closed_value,
                    "open_value": open_value,
                }
            )
        except Exception:
            continue
    return door_records


def set_articulation_state_by_record(
    env,
    rec: dict[str, Any],
    joint_index: int,
    joint_value: float,
) -> None:
    if rec.get("interaction_group") == "portal":
        door = Door(rec["name"], env.current_data)
        door.set_joint_position(joint_index, float(joint_value))
    else:
        om = env.object_managers[env.current_batch_index]
        obj = om.get_object_by_name(rec["name"])
        if not isinstance(obj, MlSpacesArticulationObject):
            raise ValueError(f"{rec['name']} is not an articulable container.")
        obj.set_joint_position(joint_index, float(joint_value))
    mujoco.mj_forward(env.current_model, env.current_data)


def joint_name_for_record(env, rec: dict[str, Any], joint_index: int) -> str:
    if rec.get("interaction_group") == "portal":
        door = Door(rec["name"], env.current_data)
        return door.joint_names[joint_index]
    om = env.object_managers[env.current_batch_index]
    obj = om.get_object_by_name(rec["name"])
    if not isinstance(obj, MlSpacesArticulationObject):
        raise ValueError(f"{rec['name']} is not an articulable container.")
    return obj.joint_names[joint_index]


def joint_value_by_name(env, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
    return float(env.current_data.qpos[qpos_addr])


def snapshot_robot_state(env) -> dict[str, Any]:
    """Serialize the semantic RBY1 state needed for a cross-task handoff.

    Raw ``data.qpos`` arrays are not stable across the door and container task
    models because each task may compile different auxiliary bodies.  Handoffs
    therefore use named move groups plus the base pose instead of raw indices.
    """
    robot_view = env.current_robot.robot_view
    return {
        "base_pose": pose_mat_to_7d(robot_view.base.pose).astype(float).tolist(),
        "move_groups": {
            group_name: np.asarray(robot_view.get_move_group(group_name).joint_pos, dtype=float).tolist()
            for group_name in robot_view.move_group_ids()
            if group_name != "base"
        },
    }


def apply_robot_state(env, state: dict[str, Any]) -> dict[str, int]:
    """Apply a semantic robot-state handoff before a policy or navigation phase."""
    robot_view = env.current_robot.robot_view
    applied_groups = 0
    base_pose = state.get("base_pose")
    if base_pose is not None:
        robot_view.base.pose = pos_quat_to_pose_mat(np.asarray(base_pose, dtype=float))
    for group_name, values in state.get("move_groups", {}).items():
        if group_name == "base" or group_name not in robot_view.move_group_ids():
            continue
        group = robot_view.get_move_group(group_name)
        values_array = np.asarray(values, dtype=float)
        if values_array.shape != np.asarray(group.joint_pos).shape:
            raise ValueError(
                f"Robot state shape mismatch for {group_name}: "
                f"expected {np.asarray(group.joint_pos).shape}, got {values_array.shape}"
            )
        group.joint_pos = values_array
        applied_groups += 1
    mujoco.mj_forward(env.current_model, env.current_data)
    return {"move_group_count": applied_groups, "base_pose_applied": int(base_pose is not None)}


def apply_episode_scene_state(
    env,
    episode: dict[str, Any] | None,
    *,
    articulation_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay V3 object/articulation state into a freshly reset RBY1 task.

    This is initialization/state handoff only; it deliberately writes named
    qpos values before the policy starts.  The actual opening motion remains
    the RBY1 contact policy and is recorded through ``task.step``.
    """
    modifications = (episode or {}).get("scene_modifications", {})
    applied_poses = 0
    for object_name, pose_7d in modifications.get("object_poses", {}).items():
        pose = pos_quat_to_pose_mat(np.asarray(pose_7d, dtype=float))
        if not set_free_joint_pose(env, object_name, pose):
            raise ValueError(f"Episode free-joint object was not found: {object_name}")
        applied_poses += 1

    applied_joints = 0
    for state in modifications.get("articulation_states", []):
        joint_name = state["joint_name"]
        joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Episode articulation joint was not found: {joint_name}")
        qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
        env.current_data.qpos[qpos_addr] = float(state["position"])
        applied_joints += 1

    overrides_applied = 0
    for joint_name, value in (articulation_overrides or {}).items():
        joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Override articulation joint was not found: {joint_name}")
        qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
        env.current_data.qpos[qpos_addr] = float(value)
        overrides_applied += 1
    mujoco.mj_forward(env.current_model, env.current_data)
    return {
        "object_pose_count": applied_poses,
        "articulation_state_count": applied_joints,
        "articulation_override_count": overrides_applied,
    }


def drive_joint_to_value_with_force(
    env,
    joint_name: str,
    target_value: float,
    *,
    max_steps: int = 1500,
    tolerance: float = FORCE_DRIVE_TOLERANCE,
) -> dict[str, Any]:
    """Move a 1-DoF articulation joint by external body force/torque instead of writing qpos."""
    controller = ForceJointController(
        env,
        joint_name,
        target_value,
        tolerance=tolerance,
    )
    try:
        for _ in range(int(max_steps)):
            controller.step()
            if controller.reached:
                break
    finally:
        controller.finish()
    result = controller.result()
    final_value = result["final_value"]
    final_error = result["final_error"]
    reached = result["reached"]
    if not reached:
        log.warning(
            "Force drive did not fully converge for %s: target=%.4f final=%.4f error=%.4f",
            joint_name,
            result["target_value"],
            final_value,
            final_error,
        )
    return result


class ForceJointController:
    """Incremental external-force joint controller used by light and full collection."""

    def __init__(
        self,
        env,
        joint_name: str,
        target_value: float,
        *,
        tolerance: float = FORCE_DRIVE_TOLERANCE,
    ) -> None:
        self.env = env
        self.model = env.current_model
        self.data = env.current_data
        self.joint_name = joint_name
        self.tolerance = float(tolerance)
        self.joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if self.joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        self.joint_type_id = int(self.model.jnt_type[self.joint_id])
        if self.joint_type_id not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(
                "Force drive only supports hinge/slide joints, "
                f"got type={self.joint_type_id}"
            )
        self.qpos_addr = int(self.model.jnt_qposadr[self.joint_id])
        self.dof_addr = int(self.model.jnt_dofadr[self.joint_id])
        self.body_id = int(self.model.jnt_bodyid[self.joint_id])
        self.local_axis = np.asarray(self.model.jnt_axis[self.joint_id], dtype=float)
        self.joint_range = [float(value) for value in self.model.jnt_range[self.joint_id]]
        lo, hi = min(self.joint_range), max(self.joint_range)
        self.target_value = float(np.clip(target_value, lo, hi))
        self.is_slide = self.joint_type_id == mujoco.mjtJoint.mjJNT_SLIDE
        self.kp = 600.0 if self.is_slide else 300.0
        self.kd = 80.0 if self.is_slide else 30.0
        self.max_force = 160.0 if self.is_slide else 1000.0
        self.stable_steps = 0
        self.steps = 0
        self.reached = False
        self.finished = False
        self.last_effort = 0.0
        # Generalized joint force follows the joint coordinate. Body-force
        # slides use the same positive convention; the first readback still
        # self-corrects either case when an asset uses a different convention.
        self.direction_sign = 1.0
        self.use_generalized_force = not self.is_slide

    def step(self) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError("ForceJointController has already been finished")
        current = float(self.data.qpos[self.qpos_addr])
        velocity = float(self.data.qvel[self.dof_addr])
        error = self.target_value - current
        effort = float(
            self.direction_sign
            * np.clip(self.kp * error - self.kd * velocity, -self.max_force, self.max_force)
        )
        world_axis = self.data.xmat[self.body_id].reshape(3, 3) @ self.local_axis
        self.data.xfrc_applied[self.body_id, :] = 0.0
        self.data.qfrc_applied[self.dof_addr] = 0.0
        if self.use_generalized_force:
            self.data.qfrc_applied[self.dof_addr] = effort
        elif self.is_slide:
            self.data.xfrc_applied[self.body_id, :3] = world_axis * effort
        else:
            self.data.xfrc_applied[self.body_id, 3:] = world_axis * effort
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.last_effort = effort
        current_after = float(self.data.qpos[self.qpos_addr])
        velocity_after = float(self.data.qvel[self.dof_addr])
        if self.steps <= 10 and abs(current_after - current) > 1e-7:
            requested_direction = np.sign(error)
            measured_direction = np.sign(current_after - current)
            if requested_direction != 0 and measured_direction != requested_direction:
                self.direction_sign = -1.0
                log.info(
                    "Reversing force direction for %s after readback sign check: "
                    "requested=%+.3f measured=%+.3f",
                    self.joint_name,
                    requested_direction,
                    measured_direction,
                )
        error_after = self.target_value - current_after
        if abs(error_after) <= self.tolerance and abs(velocity_after) <= self.tolerance:
            self.stable_steps += 1
            self.reached = self.stable_steps >= 8
        else:
            self.stable_steps = 0
        return {
            "step": self.steps,
            "joint_value": current_after,
            "joint_velocity": velocity_after,
            "target_value": self.target_value,
            "direction_sign": self.direction_sign,
            "error": error_after,
            "effort": effort,
            "world_axis": world_axis.astype(float).tolist(),
            "reached": self.reached,
        }

    def finish(self) -> None:
        if self.finished:
            return
        self.data.xfrc_applied[self.body_id, :] = 0.0
        self.data.qfrc_applied[self.dof_addr] = 0.0
        self.data.qvel[self.dof_addr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.finished = True

    def result(self) -> dict[str, Any]:
        final_value = float(self.data.qpos[self.qpos_addr])
        final_error = float(self.target_value - final_value)
        reached = bool(self.reached or abs(final_error) <= self.tolerance)
        return {
            "method": "qfrc_applied_pd" if self.use_generalized_force else "xfrc_applied_pd",
            "force_application": (
                "qfrc_applied_generalized_force"
                if self.use_generalized_force
                else "xfrc_applied_body_force_or_torque"
            ),
            "joint_name": self.joint_name,
            "joint_type": "slide" if self.is_slide else "hinge",
            "body_id": self.body_id,
            "joint_range": self.joint_range,
            "target_value": self.target_value,
            "final_value": final_value,
            "final_error": final_error,
            "steps": self.steps,
            "reached": reached,
            "tolerance": self.tolerance,
        }


def drive_articulation_state_by_record(
    env,
    rec: dict[str, Any],
    joint_index: int,
    joint_value: float,
) -> dict[str, Any]:
    joint_name = joint_name_for_record(env, rec, joint_index)
    return drive_joint_to_value_with_force(env, joint_name, joint_value)


def set_container_joint_fraction(
    env,
    container_name: str,
    joint_index: int,
    open_fraction: float,
) -> dict[str, Any]:
    om = env.object_managers[env.current_batch_index]
    container_obj = om.get_object_by_name(container_name)
    if not isinstance(container_obj, MlSpacesArticulationObject):
        raise ValueError(f"{container_name} is not an articulable container.")
    joint_range = [float(v) for v in container_obj.get_joint_range(joint_index)]
    closed_value, open_value = joint_closed_open_values(joint_range)
    target = closed_value + float(open_fraction) * (open_value - closed_value)
    drive_meta = drive_joint_to_value_with_force(env, container_obj.joint_names[joint_index], target)
    return {
        "container_name": container_name,
        "joint_index": joint_index,
        "joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
        "drive": drive_meta,
    }


def open_container_joint(env, container_name: str, joint_index: int) -> dict[str, Any]:
    return set_container_joint_fraction(env, container_name, joint_index, 1.0)


def close_container_joint(env, container_name: str, joint_index: int) -> dict[str, Any]:
    return set_container_joint_fraction(env, container_name, joint_index, 0.0)


def set_door_open_fraction(env, door_name: str, open_fraction: float) -> dict[str, Any]:
    door = Door(door_name, env.current_data)
    hinge_idx = door.get_hinge_joint_index()
    joint_range = [float(v) for v in door.get_joint_range(hinge_idx)]
    closed_value, open_value = joint_closed_open_values(joint_range)
    target = closed_value + float(open_fraction) * (open_value - closed_value)
    drive_meta = drive_joint_to_value_with_force(env, door.joint_names[hinge_idx], target)
    return {
        "door_name": door_name,
        "hinge_joint_index": hinge_idx,
        "hinge_joint_name": door.joint_names[hinge_idx],
        "hinge_joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
        "drive": drive_meta,
    }


def open_door_space(env, door_name: str) -> dict[str, Any]:
    return set_door_open_fraction(env, door_name, 1.0)


def close_door_space(env, door_name: str) -> dict[str, Any]:
    return set_door_open_fraction(env, door_name, 0.0)


def is_target_like(rec: dict[str, Any]) -> bool:
    if rec["is_structural"] or rec["interaction_group"] in {"container", "portal"}:
        return False
    if rec["has_free_joint"]:
        return True
    category = str(rec["category"] or "").lower()
    name = rec["name"].lower()
    return token_match(category, PORTABLE_PREFERRED_TOKENS) or token_match(name, PORTABLE_PREFERRED_TOKENS)


def compute_relation(container_rec: dict[str, Any], object_rec: dict[str, Any]) -> dict[str, Any]:
    c_name = container_rec["name"]
    c_min, c_max = aabb_bounds(container_rec["aabb_center"], container_rec["aabb_size"])
    o_min, o_max = aabb_bounds(object_rec["aabb_center"], object_rec["aabb_size"])
    inside = bool(np.all(o_min >= c_min) and np.all(o_max <= c_max))
    overlap = np.maximum(0.0, np.minimum(o_max, c_max) - np.maximum(o_min, c_min))
    overlap_vol = float(np.prod(overlap))
    o_size = np.asarray(object_rec["aabb_size"], dtype=float)
    o_vol = max(float(np.prod(np.maximum(o_size, 1e-4))), 1e-6)
    overlap_ratio = overlap_vol / o_vol
    support = object_rec.get("support_below")
    parent = object_rec.get("parent")
    parent_chain = object_rec.get("parent_chain") or []
    parent_chain_hit = c_name in parent_chain
    support_hit = bool(support and (support == c_name or support.startswith(c_name)))
    parent_hit = bool(parent and (parent == c_name or parent.startswith(c_name)))
    distance = float(
        np.linalg.norm(np.asarray(object_rec["aabb_center"], dtype=float) - np.asarray(container_rec["aabb_center"], dtype=float))
    )
    score = 0.0
    if inside:
        score += 3.0
    score += 2.0 * overlap_ratio
    if parent_chain_hit:
        score += 1.5
    if parent_hit:
        score += 1.0
    if support_hit:
        score += 1.0
    score += max(0.0, 1.0 - distance / 2.0)

    # Keep "inside" strict: require strong geometric evidence instead of
    # structural metadata such as parent/support, which often marks objects
    # resting on top of furniture rather than inside a cavity.
    if inside or overlap_ratio > 0.8:
        label = "inside"
    elif parent_hit or support_hit:
        label = "attached_to_container"
    elif parent_chain_hit or overlap_ratio > 0.25:
        label = "likely_inside"
    elif distance < 0.75:
        label = "near_container"
    else:
        label = "unrelated"

    return {
        "container_name": c_name,
        "object_name": object_rec["name"],
        "inside_aabb": inside,
        "overlap_ratio": overlap_ratio,
        "parent_chain_hit": parent_chain_hit,
        "parent_hit": parent_hit,
        "support_hit": support_hit,
        "distance_to_container_center": distance,
        "label": label,
        "score": score,
    }


def yaw_to_face(source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(source_xy, dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def nearest_free_point(free_points: np.ndarray, target_xy: np.ndarray) -> np.ndarray | None:
    if free_points.size == 0:
        return None
    dists = np.linalg.norm(free_points[:, :2] - target_xy[None, :2], axis=1)
    return free_points[int(np.argmin(dists))]


def container_front_axis(container_rec: dict[str, Any]) -> np.ndarray:
    size = np.asarray(container_rec["aabb_size"], dtype=float)
    quat = np.asarray(container_rec.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    rot = quat_to_rotmat(quat)
    horizontal_extents = [
        (abs(rot[0, 0]) * size[0] + abs(rot[1, 0]) * size[1], rot[:2, 0]),
        (abs(rot[0, 1]) * size[0] + abs(rot[1, 1]) * size[1], rot[:2, 1]),
    ]
    horizontal_extents.sort(key=lambda item: item[0])
    front_axis_xy = horizontal_extents[0][1]
    norm = np.linalg.norm(front_axis_xy)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return front_axis_xy / norm


def container_approach_axis(env, container_rec: dict[str, Any]) -> np.ndarray:
    """Infer the physical front normal from door hinges or drawer travel."""
    joints = articulation_joint_records(container_rec)
    qpos_before = env.current_data.qpos.copy()
    axes: list[np.ndarray] = []
    try:
        set_all_articulation_joints_closed(env, container_rec, joints)
        for joint in joints:
            joint_type = joint_mujoco_type_name(env, joint)
            if joint_type == "slide":
                closed_center, _ = joint_target_geometry(env, container_rec, joint)
                set_articulation_state_by_record(
                    env,
                    container_rec,
                    int(joint["joint_index"]),
                    float(joint["open_value"]),
                )
                open_center, _ = joint_target_geometry(env, container_rec, joint)
                axis = np.asarray(open_center, dtype=float)[:2] - np.asarray(
                    closed_center, dtype=float
                )[:2]
                set_articulation_state_by_record(
                    env,
                    container_rec,
                    int(joint["joint_index"]),
                    float(joint["closed_value"]),
                )
            elif joint_type == "hinge":
                center, _ = joint_target_geometry(env, container_rec, joint)
                joint_id = env.current_model.joint(joint["joint_name"]).id
                anchor = np.asarray(env.current_data.xanchor[joint_id], dtype=float)
                radial = np.asarray(center, dtype=float)[:2] - anchor[:2]
                axis = np.array([-radial[1], radial[0]], dtype=float)
            else:
                continue
            norm = float(np.linalg.norm(axis))
            if norm > 1e-4:
                axes.append(axis / norm)
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)
    if not axes:
        return container_front_axis(container_rec)
    reference = axes[0]
    aligned = [axis if float(np.dot(axis, reference)) >= 0.0 else -axis for axis in axes]
    mean_axis = np.mean(aligned, axis=0)
    norm = float(np.linalg.norm(mean_axis))
    if norm <= 1e-6:
        return reference
    return mean_axis / norm


def make_robot_pose_from_xy(robot_view, xy: np.ndarray, yaw: float) -> np.ndarray:
    pose = robot_view.base.pose.copy()
    pose[:3, 3] = np.array([xy[0], xy[1], 0.0], dtype=float)
    pose[:3, :3] = quat_to_rotmat(yaw_to_quat(yaw))
    return pose


def choose_collision_free_pose(
    env,
    robot_view,
    center_xy: np.ndarray,
    free_points: np.ndarray,
    candidate_xy: np.ndarray,
    penalty: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    free_pt = nearest_free_point(free_points, candidate_xy)
    if free_pt is None:
        return None
    yaw = yaw_to_face(free_pt[:2], center_xy)
    pose = make_robot_pose_from_xy(robot_view, free_pt[:2], yaw)
    if env.check_if_robot_collision_at_base_pose(robot_view, pose):
        return None
    score = float(np.linalg.norm(free_pt[:2] - candidate_xy[:2]) + penalty)
    return pose, score


def choose_pose_valid_for_joint_states(
    ctx: LoadedContext,
    articulation_rec: dict[str, Any],
    joint: dict[str, Any],
    closed_val: float,
    open_val: float,
    desired_dist: float = 0.8,
    torso_heights: tuple[float, ...] = (0.0,),
    min_clearance_m: float = 0.15,
    max_center_distance_m: float = 1.2,
    allow_back_approach: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, articulation_rec, joint)
    if articulation_rec.get("interaction_group") == "container":
        front_axis_xy = container_approach_axis(env, articulation_rec)
    else:
        front_axis_xy = container_front_axis(articulation_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    clearance_map = distance_transform_edt(thormap.occupancy) / float(thormap.px_per_m)
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)

    # Search progressively farther away and laterally around the target. The
    # score keeps the closest safe pose, but narrow layouts can still fall back
    # to a farther or off-center stance instead of spawning in collision.
    candidate_specs: list[tuple[np.ndarray, float, str]] = []
    for direction, side_penalty, side_name in (
        (front_axis_xy, 0.0, "front"),
        (-front_axis_xy, 1.5, "back"),
    ):
        for retreat in (0.0, 0.15, 0.30, 0.50, 0.70, 0.90):
            base_xy = center[:2] + direction * (desired_dist + retreat)
            for lateral_offset in (0.0, 0.20, -0.20, 0.40, -0.40):
                if lateral_offset > 0.0:
                    label = f"{side_name}_left"
                elif lateral_offset < 0.0:
                    label = f"{side_name}_right"
                else:
                    label = side_name
                candidate_specs.append(
                    (
                        base_xy + lateral_xy * lateral_offset,
                        side_penalty + retreat + 0.5 * abs(lateral_offset),
                        label,
                    )
                )

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = robot_view.base.pose.copy()
    best_pose = None
    best_score = float("inf")
    best_meta: dict[str, Any] = {}

    try:
        for candidate_xy, penalty, label in candidate_specs:
            if (
                articulation_rec.get("interaction_group") == "container"
                and label.startswith("back")
                and not allow_back_approach
            ):
                continue
            free_pt = nearest_free_point(free_points, candidate_xy)
            if free_pt is None:
                continue
            center_dist = float(np.linalg.norm(free_pt[:2] - center[:2]))
            if center_dist > max_center_distance_m:
                continue
            free_px = np.asarray(thormap.pos_m_to_px(free_pt), dtype=int)
            if not np.all((free_px >= 0) & (free_px < clearance_map.shape)):
                continue
            clearance_m = float(clearance_map[free_px[0], free_px[1]])
            if clearance_m < min_clearance_m:
                continue
            yaw = yaw_to_face(free_pt[:2], center[:2])
            pose = make_robot_pose_from_xy(robot_view, free_pt[:2], yaw)

            collision_free = True
            for articulation_value in (closed_val, open_val):
                for torso_height in torso_heights:
                    env.current_data.qpos[:] = qpos_before
                    torso_group = robot_view.get_move_group("torso")
                    torso_group.joint_pos = np.array(
                        [0.0, torso_height, -2.0 * torso_height, torso_height, 0.0, 0.0]
                    )
                    set_articulation_state_by_record(
                        env,
                        articulation_rec,
                        joint["joint_index"],
                        articulation_value,
                    )
                    mujoco.mj_forward(env.current_model, env.current_data)
                    if env.check_if_robot_collision_at_base_pose(robot_view, pose):
                        collision_free = False
                        break
                if not collision_free:
                    break
            if not collision_free:
                continue

            score = center_dist + penalty - 0.25 * min(clearance_m, 0.5)
            if score < best_score:
                best_score = score
                best_pose = pose.copy()
                best_meta = {
                    "candidate_label": label,
                    "candidate_target_xy": candidate_xy.tolist(),
                    "free_point_xy": free_pt[:2].tolist(),
                    "center_distance": center_dist,
                    "base_clearance_m": clearance_m,
                    "score": score,
                    "validated_torso_heights": list(torso_heights),
                }
    finally:
        env.current_data.qpos[:] = qpos_before
        robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)

    return best_pose, best_meta


def validate_pose_for_joint_states(
    ctx: LoadedContext,
    articulation_rec: dict[str, Any],
    joint: dict[str, Any],
    robot_pose: np.ndarray,
    closed_val: float,
    open_val: float,
    *,
    torso_heights: tuple[float, ...] = (0.0,),
    max_center_distance_m: float = 1.2,
    max_direct_operation_distance_m: float | None = None,
    max_facing_error_rad: float = float(np.deg2rad(25.0)),
) -> tuple[bool, dict[str, Any]]:
    """Check whether a navigation endpoint is safe throughout an interaction."""
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, articulation_rec, joint)
    robot_xy = np.asarray(robot_pose, dtype=float)[:2, 3]
    center_distance = float(np.linalg.norm(robot_xy - center[:2]))
    meta: dict[str, Any] = {
        "source": "caller_supplied_robot_base_pose",
        "center_distance": center_distance,
        "max_center_distance": float(max_center_distance_m),
        "validated_torso_heights": list(torso_heights),
        "collision_states": [],
    }
    if center_distance > max_center_distance_m:
        meta["valid"] = False
        meta["rejection_reason"] = "target_too_far"
        return False, meta
    if (
        max_direct_operation_distance_m is not None
        and center_distance > max_direct_operation_distance_m
    ):
        meta["valid"] = False
        meta["max_direct_operation_distance"] = float(max_direct_operation_distance_m)
        meta["rejection_reason"] = "outside_direct_manipulation_workspace"
        return False, meta

    robot_forward = np.asarray(robot_pose, dtype=float)[:2, 0]
    target_direction = center[:2] - robot_xy
    target_norm = float(np.linalg.norm(target_direction))
    if target_norm > 1e-8:
        target_direction /= target_norm
        facing_error = float(
            np.arccos(np.clip(np.dot(robot_forward, target_direction), -1.0, 1.0))
        )
    else:
        facing_error = 0.0
    meta["facing_error_degrees"] = float(np.rad2deg(facing_error))
    if facing_error > max_facing_error_rad:
        meta["valid"] = False
        meta["max_facing_error_degrees"] = float(np.rad2deg(max_facing_error_rad))
        meta["rejection_reason"] = "not_facing_target"
        return False, meta

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = robot_view.base.pose.copy()
    try:
        for state_name, articulation_value in (
            ("closed", closed_val),
            ("open", open_val),
        ):
            for torso_height in torso_heights:
                env.current_data.qpos[:] = qpos_before
                torso_group = robot_view.get_move_group("torso")
                torso_group.joint_pos = np.array(
                    [0.0, torso_height, -2.0 * torso_height, torso_height, 0.0, 0.0]
                )
                set_articulation_state_by_record(
                    env,
                    articulation_rec,
                    joint["joint_index"],
                    articulation_value,
                )
                mujoco.mj_forward(env.current_model, env.current_data)
                colliding = bool(
                    env.check_if_robot_collision_at_base_pose(robot_view, robot_pose)
                )
                meta["collision_states"].append(
                    {
                        "articulation_state": state_name,
                        "torso_height": float(torso_height),
                        "colliding": colliding,
                    }
                )
                if colliding:
                    meta["valid"] = False
                    meta["rejection_reason"] = (
                        f"collision_{state_name}_torso_{torso_height:.3f}"
                    )
                    return False, meta
    finally:
        env.current_data.qpos[:] = qpos_before
        robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)

    meta["valid"] = True
    return True, meta


def resolve_rby1_interaction_pose(
    ctx: LoadedContext,
    articulation_rec: dict[str, Any],
    joint: dict[str, Any],
    closed_val: float,
    open_val: float,
    *,
    pose_mode: str,
    supplied_robot_pose: np.ndarray | None,
    desired_dist: float,
    torso_heights: tuple[float, ...],
    min_clearance_m: float,
    max_center_distance_m: float,
    max_base_adjustment_distance_m: float,
    allow_back_approach: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Prefer a safe navigation endpoint and otherwise find a nearby stance."""
    if pose_mode not in {"auto", "current", "current_or_adjust"}:
        raise ValueError(f"Unsupported robot_pose_mode: {pose_mode}")
    if pose_mode in {"current", "current_or_adjust"} and supplied_robot_pose is None:
        raise ValueError(f"robot_pose_mode={pose_mode} requires robot_base_pose")

    supplied_meta = None
    if supplied_robot_pose is not None and pose_mode != "auto":
        valid, supplied_meta = validate_pose_for_joint_states(
            ctx,
            articulation_rec,
            joint,
            supplied_robot_pose,
            closed_val,
            open_val,
            torso_heights=torso_heights,
            max_center_distance_m=max_center_distance_m,
            max_direct_operation_distance_m=desired_dist + 0.30,
        )
        if valid:
            supplied_meta["pose_mode"] = pose_mode
            supplied_meta["adjusted"] = False
            return supplied_robot_pose, supplied_meta
        if pose_mode == "current":
            return None, supplied_meta

    robot_pose, auto_meta = choose_pose_valid_for_joint_states(
        ctx,
        articulation_rec,
        joint,
        closed_val,
        open_val,
        desired_dist=desired_dist,
        torso_heights=torso_heights,
        min_clearance_m=min_clearance_m,
        max_center_distance_m=max_center_distance_m,
        allow_back_approach=allow_back_approach,
    )
    auto_meta["source"] = "automatic_safe_pose_search"
    auto_meta["pose_mode"] = pose_mode
    auto_meta["adjusted"] = supplied_robot_pose is not None
    if supplied_meta is not None:
        auto_meta["supplied_pose_validation"] = supplied_meta
    if supplied_robot_pose is None or pose_mode == "auto" or robot_pose is None:
        return robot_pose, auto_meta

    closed_collisions = [
        state["colliding"]
        for state in supplied_meta.get("collision_states", [])
        if state["articulation_state"] == "closed"
    ]
    if any(closed_collisions):
        auto_meta["rejection_reason"] = "navigation_endpoint_collides_while_closed"
        return None, auto_meta

    adjustment_distance = float(
        np.linalg.norm(robot_pose[:2, 3] - supplied_robot_pose[:2, 3])
    )
    auto_meta["adjustment_distance"] = adjustment_distance
    auto_meta["max_base_adjustment_distance"] = float(max_base_adjustment_distance_m)
    if adjustment_distance > max_base_adjustment_distance_m:
        auto_meta["rejection_reason"] = "adjustment_exceeds_local_limit"
        return None, auto_meta

    auto_meta["adjustment_target_pose"] = pose_mat_to_7d(robot_pose).tolist()
    auto_meta["episode_start_pose"] = pose_mat_to_7d(supplied_robot_pose).tolist()
    auto_meta["source"] = "continuous_base_adjustment"
    return supplied_robot_pose, auto_meta


def joint_target_geometry(env, container_rec: dict[str, Any], joint: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = env.current_model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        return np.asarray(container_rec["aabb_center"], dtype=float), np.asarray(
            container_rec["aabb_size"], dtype=float
        )
    body_id = int(model.jnt_bodyid[joint_id])
    center, size = safe_body_aabb(model, env.current_data, body_id)
    return np.asarray(center, dtype=float), np.asarray(size, dtype=float)


def joint_body_id(env, joint: dict[str, Any]) -> int:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint['joint_name']}")
    return int(env.current_model.jnt_bodyid[joint_id])


def joint_mujoco_type_name(env, joint: dict[str, Any]) -> str:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        return str(joint.get("joint_type"))
    joint_type = int(env.current_model.jnt_type[joint_id])
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return "slide"
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return "hinge"
    return str(joint_type)


def articulation_joint_records(rec: dict[str, Any]) -> list[dict[str, Any]]:
    if rec.get("joints"):
        return list(rec["joints"])
    if rec.get("interaction_group") == "portal":
        return [
            {
                "joint_index": int(rec["hinge_joint_index"]),
                "joint_name": rec["hinge_joint_name"],
                "joint_type": "hinge",
                "joint_range": rec["hinge_joint_range"],
                "current_value": None,
                "closed_value": float(rec["closed_value"]),
                "open_value": float(rec["open_value"]),
            }
        ]
    return []


def set_all_articulation_joints_closed(
    env,
    articulation_rec: dict[str, Any],
    joints: list[dict[str, Any]],
) -> None:
    for joint in joints:
        set_articulation_state_by_record(
            env,
            articulation_rec,
            int(joint["joint_index"]),
            float(joint["closed_value"]),
        )


def collect_joint_box_state_records(
    env,
    articulation_rec: dict[str, Any],
    joints: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    qpos_before = env.current_data.qpos.copy()
    records: dict[int, dict[str, Any]] = {}
    try:
        set_all_articulation_joints_closed(env, articulation_rec, joints)
        for joint in joints:
            joint_index = int(joint["joint_index"])
            center, size = joint_target_geometry(env, articulation_rec, joint)
            records[joint_index] = {
                "joint_index": joint_index,
                "joint_name": joint["joint_name"],
                "joint_type": joint.get("joint_type"),
                "mujoco_joint_type": joint_mujoco_type_name(env, joint),
                "body_id": joint_body_id(env, joint),
                "closed_value": float(joint["closed_value"]),
                "open_value": float(joint["open_value"]),
                "closed_box": {"center": center, "size": size},
            }

        for joint in joints:
            joint_index = int(joint["joint_index"])
            set_all_articulation_joints_closed(env, articulation_rec, joints)
            set_articulation_state_by_record(
                env,
                articulation_rec,
                joint_index,
                float(joint["open_value"]),
            )
            center, size = joint_target_geometry(env, articulation_rec, joint)
            closed_center = np.asarray(records[joint_index]["closed_box"]["center"], dtype=float)
            records[joint_index]["open_box"] = {"center": center, "size": size}
            records[joint_index]["open_delta"] = np.asarray(center, dtype=float) - closed_center
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)
    return records


def infer_front_axis_from_joint_motion(
    container_rec: dict[str, Any],
    joint_box_records: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, str]:
    slide_dirs = []
    for rec in joint_box_records.values():
        if rec.get("mujoco_joint_type") != "slide":
            continue
        delta_xy = np.asarray(rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
        norm = float(np.linalg.norm(delta_xy))
        if norm > 1e-3:
            slide_dirs.append(delta_xy / norm)

    if slide_dirs:
        front_axis = np.mean(np.stack(slide_dirs, axis=0), axis=0)
        norm = float(np.linalg.norm(front_axis))
        if norm > 1e-6:
            return front_axis / norm, "slide_open_delta"

    front_axis = container_front_axis(container_rec)
    motion_votes = []
    for rec in joint_box_records.values():
        delta_xy = np.asarray(rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
        if np.linalg.norm(delta_xy) > 1e-3:
            motion_votes.append(float(np.dot(delta_xy, front_axis)))
    if motion_votes and sum(motion_votes) < 0.0:
        return -front_axis, "container_front_axis_flipped_by_open_delta"
    return front_axis, "container_front_axis"


def box_front_projection(
    center: np.ndarray,
    size: np.ndarray,
    front_axis_xy: np.ndarray,
) -> dict[str, Any]:
    front_axis_xy = np.asarray(front_axis_xy, dtype=float)
    front_axis_xy = front_axis_xy / max(float(np.linalg.norm(front_axis_xy)), 1e-9)
    lateral_axis_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
    corners = compute_box_corners(np.asarray(center, dtype=float), np.asarray(size, dtype=float))
    xy = corners[:, :2]
    depth = xy @ front_axis_xy
    lateral = xy @ lateral_axis_xy
    z_vals = corners[:, 2]
    return {
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_center": float(np.asarray(center, dtype=float)[:2] @ front_axis_xy),
        "lateral_min": float(lateral.min()),
        "lateral_max": float(lateral.max()),
        "z_min": float(z_vals.min()),
        "z_max": float(z_vals.max()),
    }


def front_projection_overlap_metrics(
    target_proj: dict[str, Any],
    candidate_proj: dict[str, Any],
) -> dict[str, Any]:
    lateral_overlap = max(
        0.0,
        min(target_proj["lateral_max"], candidate_proj["lateral_max"])
        - max(target_proj["lateral_min"], candidate_proj["lateral_min"]),
    )
    z_overlap = max(
        0.0,
        min(target_proj["z_max"], candidate_proj["z_max"])
        - max(target_proj["z_min"], candidate_proj["z_min"]),
    )
    overlap_area = float(lateral_overlap * z_overlap)
    target_area = max(
        float(
            (target_proj["lateral_max"] - target_proj["lateral_min"])
            * (target_proj["z_max"] - target_proj["z_min"])
        ),
        1e-9,
    )
    candidate_area = max(
        float(
            (candidate_proj["lateral_max"] - candidate_proj["lateral_min"])
            * (candidate_proj["z_max"] - candidate_proj["z_min"])
        ),
        1e-9,
    )
    return {
        "overlap_lateral": lateral_overlap,
        "overlap_z": z_overlap,
        "overlap_area": overlap_area,
        "ratio_of_target_projection": overlap_area / target_area,
        "ratio_of_candidate_projection": overlap_area / candidate_area,
    }


def union_aabb(
    a_center: np.ndarray,
    a_size: np.ndarray,
    b_center: np.ndarray,
    b_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_min, a_max = aabb_bounds(np.asarray(a_center, dtype=float), np.asarray(a_size, dtype=float))
    b_min, b_max = aabb_bounds(np.asarray(b_center, dtype=float), np.asarray(b_size, dtype=float))
    out_min = np.minimum(a_min, b_min)
    out_max = np.maximum(a_max, b_max)
    return (out_min + out_max) / 2.0, out_max - out_min


def box_projection_along_axis(
    center: np.ndarray,
    size: np.ndarray,
    axis_xy: np.ndarray,
) -> dict[str, Any]:
    axis_xy = np.asarray(axis_xy, dtype=float)
    axis_xy = axis_xy / max(float(np.linalg.norm(axis_xy)), 1e-9)
    lateral_axis_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=float)
    corners = compute_box_corners(np.asarray(center, dtype=float), np.asarray(size, dtype=float))
    xy = corners[:, :2]
    depth = xy @ axis_xy
    lateral = xy @ lateral_axis_xy
    z_vals = corners[:, 2]
    return {
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "lateral_min": float(lateral.min()),
        "lateral_max": float(lateral.max()),
        "z_min": float(z_vals.min()),
        "z_max": float(z_vals.max()),
    }


def interval_overlap_and_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> tuple[float, float]:
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    gap = max(0.0, max(a_min, b_min) - min(a_max, b_max))
    return overlap, gap


def slide_hinge_blocker_candidates(
    target_rec: dict[str, Any],
    joint_box_records: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    delta_xy = np.asarray(target_rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
    delta_norm = float(np.linalg.norm(delta_xy))
    if delta_norm <= 1e-4:
        return []
    slide_axis_xy = delta_xy / delta_norm
    swept_center, swept_size = union_aabb(
        target_rec["closed_box"]["center"],
        target_rec["closed_box"]["size"],
        target_rec["open_box"]["center"],
        target_rec["open_box"]["size"],
    )
    swept_proj = box_projection_along_axis(swept_center, swept_size, slide_axis_xy)
    swept_depth = max(swept_proj["depth_max"] - swept_proj["depth_min"], 1e-9)
    swept_lateral = max(swept_proj["lateral_max"] - swept_proj["lateral_min"], 1e-9)
    swept_z = max(swept_proj["z_max"] - swept_proj["z_min"], 1e-9)
    lateral_gap_allowance = max(0.12, 1.5 * swept_lateral)

    candidates = []
    for candidate_rec in joint_box_records.values():
        if candidate_rec["joint_index"] == target_rec["joint_index"]:
            continue
        if candidate_rec.get("mujoco_joint_type") != "hinge":
            continue
        candidate_proj = box_projection_along_axis(
            candidate_rec["closed_box"]["center"],
            candidate_rec["closed_box"]["size"],
            slide_axis_xy,
        )
        depth_overlap, depth_gap = interval_overlap_and_gap(
            swept_proj["depth_min"],
            swept_proj["depth_max"],
            candidate_proj["depth_min"],
            candidate_proj["depth_max"],
        )
        lateral_overlap, lateral_gap = interval_overlap_and_gap(
            swept_proj["lateral_min"],
            swept_proj["lateral_max"],
            candidate_proj["lateral_min"],
            candidate_proj["lateral_max"],
        )
        z_overlap, z_gap = interval_overlap_and_gap(
            swept_proj["z_min"],
            swept_proj["z_max"],
            candidate_proj["z_min"],
            candidate_proj["z_max"],
        )
        depth_ratio = depth_overlap / swept_depth
        lateral_ratio = lateral_overlap / swept_lateral
        z_ratio = z_overlap / swept_z
        lateral_near = lateral_gap <= lateral_gap_allowance
        valid = z_ratio >= 0.5 and depth_ratio >= 0.1 and (lateral_overlap > 0.0 or lateral_near)
        if not valid:
            continue

        volume_overlap = aabb_overlap_metrics(
            swept_center,
            swept_size,
            candidate_rec["closed_box"]["center"],
            candidate_rec["closed_box"]["size"],
        )
        lateral_score = lateral_ratio if lateral_overlap > 0.0 else 0.2 * (1.0 - lateral_gap / lateral_gap_allowance)
        score = 3.0 * z_ratio + depth_ratio + max(0.0, lateral_score)
        candidates.append(
            {
                "joint_index": int(candidate_rec["joint_index"]),
                "score": float(score),
                "has_lateral_overlap": bool(lateral_overlap > 0.0),
                "depth_overlap": float(depth_overlap),
                "depth_gap": float(depth_gap),
                "depth_ratio": float(depth_ratio),
                "lateral_overlap": float(lateral_overlap),
                "lateral_gap": float(lateral_gap),
                "lateral_ratio": float(lateral_ratio),
                "lateral_gap_allowance": float(lateral_gap_allowance),
                "z_overlap": float(z_overlap),
                "z_gap": float(z_gap),
                "z_ratio": float(z_ratio),
                "slide_axis_xy": slide_axis_xy,
                "swept_box": {"center": swept_center, "size": swept_size},
                "swept_projection": swept_proj,
                "candidate_projection_in_slide_frame": candidate_proj,
                "swept_overlap_with_candidate_closed_box": volume_overlap,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["has_lateral_overlap"],
            item["score"],
            item["depth_overlap"],
            -item["lateral_gap"],
        ),
        reverse=True,
    )
    return candidates


def infer_front_axis_from_dependencies(
    dependencies_by_target: dict[int, list[dict[str, Any]]],
    joint_box_records: dict[int, dict[str, Any]],
) -> np.ndarray | None:
    directions = []
    for target_index, candidates in dependencies_by_target.items():
        target_rec = joint_box_records.get(target_index)
        if target_rec is None:
            continue
        target_center = np.asarray(target_rec["closed_box"]["center"], dtype=float)[:2]
        for candidate in candidates:
            prereq_rec = joint_box_records.get(int(candidate["joint_index"]))
            if prereq_rec is None:
                continue
            prereq_center = np.asarray(prereq_rec["closed_box"]["center"], dtype=float)[:2]
            delta = prereq_center - target_center
            norm = float(np.linalg.norm(delta))
            if norm > 1e-3:
                directions.append(delta / norm)
    if not directions:
        return None
    axis = np.mean(np.stack(directions, axis=0), axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return None
    return axis / norm


def infer_joint_open_dependencies(
    env,
    articulation_rec: dict[str, Any],
    *,
    method: str = "front_occlusion",
    min_overlap_volume: float = 1e-6,
    min_overlap_ratio: float = 0.0,
    min_projection_ratio: float = 0.15,
    min_depth_separation: float = 0.01,
) -> list[dict[str, Any]]:
    """Infer prerequisite joints from articulation geometry.

    The default front-occlusion mode treats a candidate joint as a prerequisite
    only when its closed box is in front of the target joint's closed box and
    substantially overlaps the target in the front-view projection.
    """
    joints = articulation_joint_records(articulation_rec)
    joint_box_records = collect_joint_box_state_records(env, articulation_rec, joints)
    blocker_candidates_by_target: dict[int, list[dict[str, Any]]] = {}
    if method == "front_occlusion":
        for rec in joint_box_records.values():
            if rec.get("mujoco_joint_type") != "slide":
                continue
            candidates = slide_hinge_blocker_candidates(rec, joint_box_records)
            if candidates:
                blocker_candidates_by_target[int(rec["joint_index"])] = [candidates[0]]

    dependency_front_axis = infer_front_axis_from_dependencies(
        blocker_candidates_by_target, joint_box_records
    )
    if dependency_front_axis is not None:
        front_axis_xy = dependency_front_axis
        front_axis_source = "slide_hinge_blocker_geometry"
    else:
        front_axis_xy, front_axis_source = infer_front_axis_from_joint_motion(
            articulation_rec, joint_box_records
        )

    if method not in {"front_occlusion", "open_aabb_overlap"}:
        raise ValueError(f"Unsupported dependency inference method: {method}")

    results: list[dict[str, Any]] = []

    for target_joint in joints:
        target_index = int(target_joint["joint_index"])
        target_rec = joint_box_records[target_index]
        target_closed_box = target_rec["closed_box"]
        target_open_box = target_rec["open_box"]
        target_proj = box_front_projection(
            target_closed_box["center"], target_closed_box["size"], front_axis_xy
        )
        prerequisites = []

        for candidate_joint in joints:
            candidate_index = int(candidate_joint["joint_index"])
            if candidate_index == target_index:
                continue
            candidate_rec = joint_box_records[candidate_index]
            candidate_box = candidate_rec["closed_box"]

            if method == "open_aabb_overlap":
                overlap = aabb_overlap_metrics(
                    target_open_box["center"],
                    target_open_box["size"],
                    candidate_box["center"],
                    candidate_box["size"],
                )
                is_prerequisite = (
                    overlap["inter_vol"] > float(min_overlap_volume)
                    and max(overlap["ratio_of_a"], overlap["ratio_of_b"]) >= float(min_overlap_ratio)
                )
                evidence = {"overlap_with_target_open_box": overlap}
            else:
                selected = next(
                    (
                        item
                        for item in blocker_candidates_by_target.get(target_index, [])
                        if int(item["joint_index"]) == candidate_index
                    ),
                    None,
                )
                is_prerequisite = selected is not None
                candidate_proj = box_front_projection(candidate_box["center"], candidate_box["size"], front_axis_xy)
                projection = front_projection_overlap_metrics(target_proj, candidate_proj)
                depth_separation = candidate_proj["depth_center"] - target_proj["depth_center"]
                evidence = {
                    "dependency_source": "slide_hinge_blocker_geometry",
                    "front_projection_overlap": projection,
                    "target_front_projection": target_proj,
                    "candidate_front_projection": candidate_proj,
                    "depth_separation": depth_separation,
                    "is_in_front": depth_separation >= float(min_depth_separation),
                    "is_projected_occluder": projection["overlap_area"] > 0.0
                    and projection["ratio_of_target_projection"] >= float(min_projection_ratio),
                    "slide_hinge_blocker_evidence": selected,
                }

            if not is_prerequisite:
                continue

            prerequisites.append(
                {
                    "joint_index": candidate_index,
                    "joint_name": candidate_joint["joint_name"],
                    "joint_type": candidate_joint.get("joint_type"),
                    "mujoco_joint_type": candidate_rec.get("mujoco_joint_type"),
                    "body_id": candidate_rec["body_id"],
                    "closed_box": candidate_box,
                    **evidence,
                }
            )

        results.append(
            {
                "joint_index": target_index,
                "joint_name": target_joint["joint_name"],
                "joint_type": target_joint.get("joint_type"),
                "mujoco_joint_type": target_rec.get("mujoco_joint_type"),
                "body_id": target_rec["body_id"],
                "closed_value": float(target_joint["closed_value"]),
                "open_value": float(target_joint["open_value"]),
                "closed_box": target_closed_box,
                "open_box": target_open_box,
                "open_delta": target_rec.get("open_delta"),
                "front_axis_xy": front_axis_xy,
                "front_axis_source": front_axis_source,
                "dependency_method": method,
                "prerequisite_joint_indices": [item["joint_index"] for item in prerequisites],
                "prerequisite_joints": prerequisites,
            }
        )
    return results


def save_joint_dependency_plot(
    output_path: Path,
    articulation_rec: dict[str, Any],
    joint_dependencies: list[dict[str, Any]],
    object_rec: dict[str, Any] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    front_axis = np.array([1.0, 0.0], dtype=float)
    if joint_dependencies:
        front_axis = np.asarray(joint_dependencies[0].get("front_axis_xy", [1.0, 0.0]), dtype=float)
    front_axis = front_axis / max(float(np.linalg.norm(front_axis)), 1e-9)
    lateral_axis = np.array([-front_axis[1], front_axis[0]], dtype=float)
    origin_xy = np.asarray(articulation_rec["aabb_center"], dtype=float)[:2]

    def local_box(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corners = compute_box_corners(center, size)
        xy = corners[:, :2] - origin_xy[None, :]
        local_corners = np.column_stack(
            [
                xy @ lateral_axis,
                xy @ front_axis,
                corners[:, 2],
            ]
        )
        bmin = local_corners.min(axis=0)
        bmax = local_corners.max(axis=0)
        return (bmin + bmax) / 2.0, bmax - bmin

    def local_point(point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        xy = point[:2] - origin_xy
        return np.array([float(xy @ lateral_axis), float(xy @ front_axis), float(point[2])], dtype=float)

    all_centers = []
    all_sizes = []
    for dep in joint_dependencies:
        color = colors[int(dep["joint_index"]) % len(colors)]
        closed_box = dep["closed_box"]
        open_box = dep["open_box"]
        closed_center_world = np.asarray(closed_box["center"], dtype=float)
        closed_size = np.asarray(closed_box["size"], dtype=float)
        open_center_world = np.asarray(open_box["center"], dtype=float)
        open_size = np.asarray(open_box["size"], dtype=float)
        closed_center, closed_size_local = local_box(closed_center_world, closed_size)
        open_center, open_size_local = local_box(open_center_world, open_size)
        all_centers.extend([closed_center, open_center])
        all_sizes.extend([closed_size_local, open_size_local])
        add_box_to_ax(
            ax,
            closed_center,
            closed_size_local,
            color,
            f"j{dep['joint_index']} closed",
            0.20,
        )
        add_box_to_ax(
            ax,
            open_center,
            open_size_local,
            color,
            f"j{dep['joint_index']} open",
            0.06,
        )
        ax.plot(
            [closed_center[0], open_center[0]],
            [closed_center[1], open_center[1]],
            [closed_center[2], open_center[2]],
            color=color,
            linestyle="--",
            linewidth=1.0,
        )

    if object_rec is not None:
        object_center_world = np.asarray(object_rec["aabb_center"], dtype=float)
        object_size_world = np.asarray(object_rec["aabb_size"], dtype=float)
        object_center, object_size = local_box(object_center_world, object_size_world)
        all_centers.append(object_center)
        all_sizes.append(object_size)
        add_box_to_ax(
            ax,
            object_center,
            object_size,
            "black",
            f"object:{object_rec['name']}",
            0.45,
        )

    dep_by_index = {int(dep["joint_index"]): dep for dep in joint_dependencies}
    for dep in joint_dependencies:
        target_center = local_point(np.asarray(dep["closed_box"]["center"], dtype=float))
        for prereq in dep["prerequisite_joints"]:
            prereq_dep = dep_by_index.get(int(prereq["joint_index"]))
            if prereq_dep is None:
                continue
            prereq_center = local_point(np.asarray(prereq_dep["closed_box"]["center"], dtype=float))
            delta = target_center - prereq_center
            ax.quiver(
                prereq_center[0],
                prereq_center[1],
                prereq_center[2],
                delta[0],
                delta[1],
                delta[2],
                color="black",
                arrow_length_ratio=0.12,
                linewidth=1.5,
            )

    if joint_dependencies:
        container_center = local_point(np.asarray(articulation_rec["aabb_center"], dtype=float))
        arrow_start = container_center.copy()
        arrow_start[2] = max(arrow_start[2], 0.05)
        ax.quiver(
            arrow_start[0],
            arrow_start[1],
            arrow_start[2],
            0.0,
            0.6,
            0.0,
            color="magenta",
            arrow_length_ratio=0.18,
            linewidth=2.0,
        )
        ax.text(
            arrow_start[0],
            arrow_start[1] + 0.65,
            arrow_start[2],
            "front",
            color="magenta",
            fontsize=10,
        )

    if all_centers:
        mins = []
        maxs = []
        for center, size in zip(all_centers, all_sizes):
            bmin, bmax = aabb_bounds(center, size)
            mins.append(bmin)
            maxs.append(bmax)
        mins_arr = np.min(np.stack(mins, axis=0), axis=0)
        maxs_arr = np.max(np.stack(maxs, axis=0), axis=0)
        margin = 0.25
        ax.set_xlim(float(mins_arr[0] - margin), float(maxs_arr[0] + margin))
        ax.set_ylim(float(mins_arr[1] - margin), float(maxs_arr[1] + margin))
        ax.set_zlim(max(0.0, float(mins_arr[2] - margin)), float(maxs_arr[2] + margin))
        set_axes_equal_3d(ax)

    ax.set_xlabel("lateral")
    ax.set_ylabel("front_depth")
    ax.set_zlabel("z")
    title = f"{articulation_rec.get('asset_id') or articulation_rec.get('category')} joint boxes"
    if object_rec is not None:
        title += f"\nobject: {object_rec['name']}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_slide_force_transition_plot(
    output_path: Path,
    articulation_rec: dict[str, Any],
    joint_index: int,
    binding: dict[str, Any],
    front_axis_xy: np.ndarray,
) -> None:
    """Plot force-driven drawer and object AABBs before and after opening."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    front_axis = np.asarray(front_axis_xy, dtype=float)
    front_axis /= max(float(np.linalg.norm(front_axis)), 1e-9)
    lateral_axis = np.array([-front_axis[1], front_axis[0]], dtype=float)
    origin_xy = np.asarray(articulation_rec["aabb_center"], dtype=float)[:2]

    def local_box(box: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        corners = compute_box_corners(box["center"], box["size"])
        xy = corners[:, :2] - origin_xy[None, :]
        local_corners = np.column_stack(
            [xy @ lateral_axis, xy @ front_axis, corners[:, 2]]
        )
        bmin = local_corners.min(axis=0)
        bmax = local_corners.max(axis=0)
        return (bmin + bmax) / 2.0, bmax - bmin

    specs = [
        ("closed_joint_box", "tab:blue", f"j{joint_index} closed", 0.20),
        ("open_joint_box", "tab:cyan", f"j{joint_index} force-open", 0.16),
        ("closed_object_box", "tab:red", "object closed", 0.45),
        ("open_object_box", "tab:orange", "object force-open", 0.40),
    ]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    centers = []
    sizes = []
    local_by_key = {}
    for key, color, label, alpha in specs:
        center, size = local_box(binding[key])
        centers.append(center)
        sizes.append(size)
        local_by_key[key] = center
        add_box_to_ax(ax, center, size, color, label, alpha)

    for closed_key, open_key, color in (
        ("closed_joint_box", "open_joint_box", "tab:blue"),
        ("closed_object_box", "open_object_box", "tab:red"),
    ):
        start = local_by_key[closed_key]
        end = local_by_key[open_key]
        delta = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            color=color,
            arrow_length_ratio=0.15,
            linewidth=1.8,
        )

    mins = []
    maxs = []
    for center, size in zip(centers, sizes):
        bmin, bmax = aabb_bounds(center, size)
        mins.append(bmin)
        maxs.append(bmax)
    mins_arr = np.min(np.stack(mins), axis=0)
    maxs_arr = np.max(np.stack(maxs), axis=0)
    margin = 0.20
    ax.set_xlim(float(mins_arr[0] - margin), float(maxs_arr[0] + margin))
    ax.set_ylim(float(mins_arr[1] - margin), float(maxs_arr[1] + margin))
    ax.set_zlim(max(0.0, float(mins_arr[2] - margin)), float(maxs_arr[2] + margin))
    set_axes_equal_3d(ax)
    ax.set_xlabel("lateral")
    ax.set_ylabel("front_depth")
    ax.set_zlabel("z")
    ax.set_title(
        f"Force-driven slide j{joint_index}\n"
        f"motion_ratio={binding.get('motion_ratio', 0.0):.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aabb_contains_object(
    box_center: np.ndarray,
    box_size: np.ndarray,
    object_rec: dict[str, Any],
    padding: float = 0.0,
) -> bool:
    padded_size = np.asarray(box_size, dtype=float) + 2.0 * float(padding)
    bmin, bmax = aabb_bounds(np.asarray(box_center, dtype=float), padded_size)
    omin, omax = aabb_bounds(
        np.asarray(object_rec["aabb_center"], dtype=float),
        np.asarray(object_rec["aabb_size"], dtype=float),
    )
    return bool(np.all(omin >= bmin) and np.all(omax <= bmax))


def free_joint_pose(env, object_name: str) -> np.ndarray | None:
    try:
        body = MlSpacesFreeJointBody(env.current_data, object_name)
    except AssertionError:
        return None
    return pos_quat_to_mat(body.position, body.quat)


def set_free_joint_pose(env, object_name: str, pose: np.ndarray) -> bool:
    try:
        body = MlSpacesFreeJointBody(env.current_data, object_name)
    except AssertionError:
        return False
    pos, quat = mat_to_pos_quat(pose)
    body.position = pos
    body.quat = quat
    mujoco.mj_forward(env.current_model, env.current_data)
    return True


def drawer_joint_contained_objects(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    object_records: list[dict[str, Any]],
    padding: float = 0.0,
) -> list[dict[str, Any]]:
    env = ctx.env
    qpos_before = env.current_data.qpos.copy()
    try:
        set_articulation_state_by_record(
            env, container_rec, joint["joint_index"], joint["closed_value"]
        )
        center, size = joint_target_geometry(env, container_rec, joint)
        out = []
        for object_rec in object_records:
            if not object_rec.get("has_free_joint"):
                continue
            if object_rec["name"] == container_rec["name"]:
                continue
            body_id = int(object_rec["body_id"])
            cur_center, cur_size = safe_body_aabb(env.current_model, env.current_data, body_id)
            cur_rec = {**object_rec, "aabb_center": cur_center, "aabb_size": cur_size}
            if aabb_contains_object(center, size, cur_rec, padding=padding):
                out.append(cur_rec)
        return out
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)


def object_in_closed_joint_box(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    object_rec: dict[str, Any],
    *,
    padding: float = 0.05,
) -> dict[str, Any]:
    """Check whether a free object is geometrically contained by a closed joint body."""
    env = ctx.env
    qpos_before = env.current_data.qpos.copy()
    try:
        set_all_articulation_joints_closed(env, container_rec, container_rec["joints"])
        joint_center, joint_size = joint_target_geometry(env, container_rec, joint)
        object_center, object_size = safe_body_aabb(
            env.current_model,
            env.current_data,
            int(object_rec["body_id"]),
        )
        current_object = {
            **object_rec,
            "aabb_center": object_center,
            "aabb_size": object_size,
        }
        return {
            "contained": aabb_contains_object(
                joint_center,
                joint_size,
                current_object,
                padding=padding,
            ),
            "padding": float(padding),
            "joint_box": {
                "center": np.asarray(joint_center, dtype=float).tolist(),
                "size": np.asarray(joint_size, dtype=float).tolist(),
            },
            "object_box": {
                "center": np.asarray(object_center, dtype=float).tolist(),
                "size": np.asarray(object_size, dtype=float).tolist(),
            },
        }
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)


def place_robot_in_front_of_container(ctx: LoadedContext, container_rec: dict[str, Any]) -> bool:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center = np.asarray(container_rec["aabb_center"], dtype=float)
    front_axis_xy = container_front_axis(container_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    desired_dist = 0.8
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
    candidate_specs: list[tuple[np.ndarray, float]] = []
    for sign in (1.0, -1.0):
        front_xy = center[:2] + sign * front_axis_xy * desired_dist
        candidate_specs.append((front_xy, 0.0))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25, 0.25))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.50, 0.50))
        candidate_specs.append((front_xy + lateral_xy * 0.30, 0.30))
        candidate_specs.append((front_xy - lateral_xy * 0.30, 0.30))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25 + lateral_xy * 0.30, 0.55))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25 - lateral_xy * 0.30, 0.55))

    best_pose = None
    best_rank = float("inf")
    for candidate_xy, penalty in candidate_specs:
        result = choose_collision_free_pose(
            env, robot_view, center[:2], free_points, candidate_xy, penalty=penalty
        )
        if result is None:
            continue
        pose, score = result
        if score < best_rank:
            best_rank = score
            best_pose = pose.copy()

    if best_pose is None:
        return False

    robot_view.base.pose = best_pose
    mujoco.mj_forward(env.current_model, env.current_data)
    apply_default_arm_pose(env)
    apply_default_head_pose(env, ctx.initial_head_qpos)
    env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
    return True


def place_robot_for_container_joint(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    desired_dist: float = 0.8,
) -> tuple[bool, np.ndarray | None]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, container_rec, joint)
    front_axis_xy = container_front_axis(container_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)

    primary_front = center[:2] + front_axis_xy * desired_dist
    fallback_front = center[:2] - front_axis_xy * desired_dist
    candidate_specs: list[tuple[np.ndarray, float]] = []
    for base_xy, base_penalty in ((primary_front, 0.0), (fallback_front, 1.5)):
        candidate_specs.extend(
            [
                (base_xy, base_penalty),
                (base_xy + front_axis_xy * 0.25, base_penalty + 0.25),
                (base_xy + front_axis_xy * 0.50, base_penalty + 0.50),
                (base_xy + lateral_xy * 0.30, base_penalty + 0.30),
                (base_xy - lateral_xy * 0.30, base_penalty + 0.30),
                (base_xy + front_axis_xy * 0.25 + lateral_xy * 0.30, base_penalty + 0.55),
                (base_xy + front_axis_xy * 0.25 - lateral_xy * 0.30, base_penalty + 0.55),
                (base_xy + front_axis_xy * 0.50 + lateral_xy * 0.30, base_penalty + 0.80),
                (base_xy + front_axis_xy * 0.50 - lateral_xy * 0.30, base_penalty + 0.80),
            ]
        )

    best_pose = None
    best_rank = float("inf")
    for candidate_xy, penalty in candidate_specs:
        result = choose_collision_free_pose(
            env, robot_view, center[:2], free_points, candidate_xy, penalty=penalty
        )
        if result is None:
            continue
        pose, score = result
        if score < best_rank:
            best_pose = pose.copy()
            best_rank = score

    if best_pose is None:
        return False, None

    robot_view.base.pose = best_pose
    mujoco.mj_forward(env.current_model, env.current_data)
    apply_default_arm_pose(env)
    apply_default_head_pose(env, ctx.initial_head_qpos)
    env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
    return True, best_pose.copy()


def save_debug_head_image(env, path: Path) -> None:
    rgb = env.render_rgb_frame("head_camera")
    save_rgb_image(path, rgb)


def command_debug_container_view(args: argparse.Namespace) -> int:
    house_dir = args.output_dir / f"debug_container_view_{args.house_ind}"
    house_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(ctx)
        container_rec = next(
            (rec for rec in containers if rec["name"] == args.container_name),
            None,
        )
        if container_rec is None:
            raise RuntimeError(f"Container not found: {args.container_name}")

        env = ctx.env
        om = env.object_managers[env.current_batch_index]
        container_obj = om.get_object_by_name(container_rec["name"])
        if not isinstance(container_obj, MlSpacesArticulationObject):
            raise RuntimeError("Selected container is not articulable")

        thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
        free_points = thormap.get_free_points()
        center = np.asarray(container_rec["aabb_center"], dtype=float)
        front_axis_xy = container_front_axis(container_rec)
        lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
        desired_dist = 0.8
        base_front = center[:2] + front_axis_xy * desired_dist
        viewpoint_targets = {
            "front": base_front,
            "left": base_front + lateral_xy * 0.35,
            "right": base_front - lateral_xy * 0.35,
        }

        qpos_before = env.current_data.qpos.copy()
        robot_before = env.current_robot.robot_view.base.pose.copy()
        head_before = get_head_joint_position(env)
        images = {}
        pose_rows = []
        head_pose_rows = []
        try:
            container_obj.set_joint_position(args.joint_index, float(args.joint_value))
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

            for name, candidate_xy in viewpoint_targets.items():
                pose = None
                # try direct, back, and lateral retreats for this requested viewpoint
                candidates = [
                    (candidate_xy, 0.0),
                    (candidate_xy + front_axis_xy * 0.25, 0.25),
                    (candidate_xy + front_axis_xy * 0.50, 0.50),
                    (candidate_xy + lateral_xy * 0.20, 0.20),
                    (candidate_xy - lateral_xy * 0.20, 0.20),
                    (candidate_xy + front_axis_xy * 0.25 + lateral_xy * 0.20, 0.45),
                    (candidate_xy + front_axis_xy * 0.25 - lateral_xy * 0.20, 0.45),
                ]
                best = None
                best_score = float("inf")
                for cand_xy, penalty in candidates:
                    result = choose_collision_free_pose(
                        env, env.current_robot.robot_view, center[:2], free_points, cand_xy, penalty
                    )
                    if result is None:
                        continue
                    cur_pose, cur_score = result
                    if cur_score < best_score:
                        best = cur_pose
                        best_score = cur_score
                if best is None:
                    pose_rows.append({"view": name, "success": False})
                    continue
                env.current_robot.robot_view.base.pose = best
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                apply_default_head_pose(env, ctx.initial_head_qpos)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                img_path = house_dir / f"{name}_head_rgb.png"
                save_debug_head_image(env, img_path)
                images[name] = str(img_path)
                pose_rows.append(
                    {
                        "view": name,
                        "success": True,
                        "robot_pose": best.tolist(),
                        "image_path": str(img_path),
                    }
                )
                if name == "front":
                    default_head = (
                        None if ctx.initial_head_qpos is None else ctx.initial_head_qpos.copy()
                    )
                    if default_head is None:
                        default_head = get_head_joint_position(env)
                    if default_head is not None:
                        head_states = [
                            ("front_head_default", default_head.copy()),
                            (
                                "front_head_look_down",
                                np.asarray(
                                    [default_head[0], default_head[1] + args.head_tilt_delta],
                                    dtype=np.float32,
                                ),
                            ),
                            ("front_head_restored", default_head.copy()),
                        ]
                    else:
                        head_states = []
                    for tag, target_head_qpos in head_states:
                        head_qpos = set_head_joint_position(env, target_head_qpos)
                        if head_qpos is None:
                            continue
                        env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                        head_img_path = house_dir / f"{tag}_rgb.png"
                        save_debug_head_image(env, head_img_path)
                        images[tag] = str(head_img_path)
                        head_pose_rows.append(
                            {
                                "tag": tag,
                                "head_qpos": np.asarray(head_qpos, dtype=float).tolist(),
                                "image_path": str(head_img_path),
                            }
                        )
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            if head_before is not None:
                set_head_joint_position(env, head_before)
            else:
                apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        payload = {
            "house_ind": args.house_ind,
            "container_name": args.container_name,
            "joint_index": args.joint_index,
            "joint_value": args.joint_value,
            "images": images,
            "poses": pose_rows,
            "head_pose_images": head_pose_rows,
            "head_tilt_delta": args.head_tilt_delta,
        }
        write_json(house_dir / "debug_view_result.json", payload)
        return 0
    finally:
        close_context(ctx)


def command_debug_drawer_bound_object(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"debug_drawer_bound_object_house_{args.house_ind}_joint_{args.joint_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(ctx)
        container_rec = next((rec for rec in containers if rec["name"] == args.container_name), None)
        if container_rec is None:
            raise RuntimeError(f"Container not found: {args.container_name}")
        joint = next((item for item in container_rec["joints"] if item["joint_index"] == args.joint_index), None)
        if joint is None:
            raise RuntimeError(f"Joint index {args.joint_index} not found on {args.container_name}")

        env = ctx.env
        object_records = [rec for rec in records if is_target_like(rec)]
        contained = drawer_joint_contained_objects(
            ctx, container_rec, joint, object_records, padding=args.box_padding
        )
        if args.object_name:
            object_rec = next((rec for rec in records if rec["name"] == args.object_name), None)
            if object_rec is None:
                raise RuntimeError(f"Object not found: {args.object_name}")
        elif contained:
            contained.sort(key=lambda rec: (str(rec["category"]), rec["name"]))
            object_rec = contained[0]
        else:
            raise RuntimeError(
                f"No free-joint target-like object found inside joint {args.joint_index} "
                f"AABB with padding={args.box_padding}."
            )
        object_name = object_rec["name"]

        qpos_before = env.current_data.qpos.copy()
        robot_pose_before = env.current_robot.robot_view.base.pose.copy()
        head_before = get_head_joint_position(env)
        torso_before = get_torso_joint_position(env)
        try:
            set_articulation_state_by_record(
                env, container_rec, joint["joint_index"], joint["closed_value"]
            )
            closed_joint_body_pose = body_pose_mat(env, joint_body_id(env, joint))
            closed_object_pose = free_joint_pose(env, object_name)
            if closed_object_pose is None:
                raise RuntimeError(f"{object_name} does not have a free joint pose.")
            closed_object_pos = env.current_data.xpos[int(object_rec["body_id"])].copy()

            set_articulation_state_by_record(
                env, container_rec, joint["joint_index"], joint["open_value"]
            )
            open_joint_body_pose = body_pose_mat(env, joint_body_id(env, joint))
            drawer_delta = open_joint_body_pose @ np.linalg.inv(closed_joint_body_pose)
            moved_object_pose = drawer_delta @ closed_object_pose

            shared_pose, shared_meta = choose_pose_valid_for_joint_states(
                ctx,
                container_rec,
                joint,
                joint["closed_value"],
                joint["open_value"],
                desired_dist=args.view_distance,
            )
            if shared_pose is None:
                ok, placed_pose = place_robot_for_container_joint(
                    ctx, container_rec, joint, desired_dist=args.view_distance
                )
                if not ok or placed_pose is None:
                    raise RuntimeError("Could not place robot for drawer view.")
                shared_pose = placed_pose
                shared_meta = {"fallback": "place_robot_for_container_joint"}

            def render_case(
                tag: str,
                joint_value: float,
                object_pose: np.ndarray,
                *,
                force_drive: bool = False,
                known_issue: str | None = None,
            ) -> dict[str, Any]:
                env.current_data.qpos[:] = qpos_before
                mujoco.mj_forward(env.current_model, env.current_data)
                if force_drive:
                    set_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint["closed_value"]
                    )
                    set_free_joint_pose(env, object_name, object_pose)
                    drive_meta = drive_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint_value
                    )
                else:
                    set_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint_value
                    )
                    set_free_joint_pose(env, object_name, object_pose)
                    drive_meta = {"method": "direct_set_joint_position"}
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_torso_pose(env, ctx.initial_torso_qpos)
                torso_qpos = lean_torso_for_drawer_view(
                    env, ctx.initial_torso_qpos, pitch_delta=args.torso_lean_delta
                )
                apply_default_arm_pose(env)
                apply_default_head_pose(env, ctx.initial_head_qpos)
                lower_head_for_drawer_view(env, tilt_delta=args.head_tilt_delta)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                rgb_path = out_dir / f"{tag}_rgb.png"
                seg_path = out_dir / f"{tag}_seg.png"
                save_debug_head_image(env, rgb_path)
                save_segmentation_preview(seg_path, env.render_segmentation_frame("head_camera"))
                visibility = env.check_visibility("head_camera", object_name)
                if isinstance(visibility, dict):
                    vis_value = float(visibility.get(object_name, 0.0))
                else:
                    vis_value = float(visibility)
                body_id = int(object_rec["body_id"])
                center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
                return {
                    "tag": tag,
                    "joint_value": float(joint_value),
                    "actual_joint_value": joint_value_by_name(env, joint["joint_name"]),
                    "drive": drive_meta,
                    "known_issue": known_issue,
                    "rgb": str(rgb_path),
                    "seg": str(seg_path),
                    "visibility": vis_value,
                    "object_xpos": env.current_data.xpos[body_id].copy().tolist(),
                    "object_aabb_center": center.tolist(),
                    "object_aabb_size": size.tolist(),
                    "head_qpos": get_head_joint_position(env).tolist()
                    if get_head_joint_position(env) is not None
                    else None,
                    "torso_qpos": None if torso_qpos is None else torso_qpos.tolist(),
                }

            cases = [
                render_case("closed_look_down", joint["closed_value"], closed_object_pose),
                render_case("open_without_move_look_down", joint["open_value"], closed_object_pose),
                render_case(
                    "open_with_moved_object_look_down",
                    joint["open_value"],
                    moved_object_pose,
                    known_issue=(
                        "Known-bad workaround: the object is moved by the joint body's global "
                        "delta, but this can mismatch drawer compartments and make an object "
                        "from one drawer appear to fly out with another drawer."
                    ),
                ),
                render_case(
                    "open_with_force_look_down",
                    joint["open_value"],
                    closed_object_pose,
                    force_drive=True,
                ),
            ]
            moved_pos = np.asarray(cases[2]["object_xpos"], dtype=float)
            force_pos = np.asarray(cases[-1]["object_xpos"], dtype=float)
            payload = {
                "house_ind": args.house_ind,
                "container_name": container_rec["name"],
                "joint": joint,
                "box_padding": args.box_padding,
                "view_distance": args.view_distance,
                "head_tilt_delta": args.head_tilt_delta,
                "torso_lean_delta": args.torso_lean_delta,
                "contained_candidates": [
                    {
                        "name": rec["name"],
                        "category": rec["category"],
                        "aabb_center": np.asarray(rec["aabb_center"], dtype=float).tolist(),
                        "aabb_size": np.asarray(rec["aabb_size"], dtype=float).tolist(),
                    }
                    for rec in contained
                ],
                "selected_object": {
                    "name": object_name,
                    "category": object_rec["category"],
                    "body_id": int(object_rec["body_id"]),
                },
                "shared_robot_pose": shared_pose.tolist(),
                "shared_pose_meta": shared_meta,
                "closed_object_xpos": closed_object_pos.tolist(),
                "moved_object_xpos_delta": (moved_pos - closed_object_pos).tolist(),
                "force_drive_object_xpos_delta": (force_pos - closed_object_pos).tolist(),
                "manual_move_known_issue": (
                    "Do not treat open_with_moved_object_look_down as a valid interaction result. "
                    "It was kept only as a diagnostic for the old workaround; it can attach an "
                    "object to the wrong drawer space."
                ),
                "drawer_delta": drawer_delta.tolist(),
                "cases": cases,
            }
            write_json(out_dir / "drawer_bound_object_result.json", payload)
            return 0
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            if torso_before is not None:
                set_torso_joint_position(env, torso_before)
            else:
                apply_default_torso_pose(env, ctx.initial_torso_qpos)
            apply_default_arm_pose(env)
            if head_before is not None:
                set_head_joint_position(env, head_before)
            else:
                apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
    finally:
        close_context(ctx)


def command_debug_door_view(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"debug_door_view_{args.house_ind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        door_records = collect_door_records(ctx)
        if args.door_name:
            door_rec = next((rec for rec in door_records if rec["name"] == args.door_name), None)
            if door_rec is None:
                raise RuntimeError(f"Door not found: {args.door_name}")
        else:
            if not door_records:
                raise RuntimeError(f"No interactive doors found in house {args.house_ind}")
            door_rec = door_records[0]

        env = ctx.env
        qpos_before = env.current_data.qpos.copy()
        robot_pose_before = env.current_robot.robot_view.base.pose.copy()
        images = {}
        try:
            close_meta = close_door_space(env, door_rec["name"])
            shared_pose, shared_meta = choose_pose_valid_for_joint_states(
                ctx,
                {
                    "name": door_rec["name"],
                    "interaction_group": "portal",
                    "aabb_center": door_rec["aabb_center"],
                    "aabb_size": door_rec["aabb_size"],
                    "quat": door_rec["quat"],
                },
                {
                    "joint_index": door_rec["hinge_joint_index"],
                    "joint_name": door_rec["hinge_joint_name"],
                },
                close_meta["target_value"],
                door_rec["open_value"],
            )
            if shared_pose is None:
                raise RuntimeError("Could not find a shared robot pose for door closed/open states.")

            for tag, target_value in (
                ("closed", close_meta["target_value"]),
                ("open", door_rec["open_value"]),
            ):
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                set_articulation_state_by_record(
                    env, door_rec, door_rec["hinge_joint_index"], target_value
                )
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                img_path = out_dir / f"{tag}_head_rgb.png"
                seg_path = out_dir / f"{tag}_head_seg.png"
                save_debug_head_image(env, img_path)
                save_segmentation_preview(seg_path, env.render_segmentation_frame("head_camera"))
                images[tag] = {"rgb": str(img_path), "seg": str(seg_path)}
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        payload = {
            "house_ind": args.house_ind,
            "door_name": door_rec["name"],
            "hinge_joint_index": door_rec["hinge_joint_index"],
            "hinge_joint_range": door_rec["hinge_joint_range"],
            "shared_robot_pose": shared_pose.tolist(),
            "shared_pose_meta": shared_meta,
            "images": images,
        }
        write_json(out_dir / "debug_door_result.json", payload)
        return 0
    finally:
        close_context(ctx)


def measure_container_visibility(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    object_names: list[str],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    container_obj = om.get_object_by_name(container_rec["name"])
    if not isinstance(container_obj, MlSpacesArticulationObject):
        return []

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = env.current_robot.robot_view.base.pose.copy()
    results = []
    for joint in container_rec["joints"]:
        closed_val = joint["closed_value"]
        open_val = joint["open_value"]
        shared_pose, shared_pose_meta = choose_pose_valid_for_joint_states(
            ctx, container_rec, joint, closed_val, open_val
        )
        closed_robot_pose_actual = None
        open_robot_pose_actual = None
        try:
            if shared_pose is None:
                continue
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
            container_obj.set_joint_position(joint["joint_index"], closed_val)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            closed_robot_pose_actual = env.current_robot.robot_view.base.pose.copy()
            closed_vis = env.check_visibility("head_camera", *object_names)
            closed_rgb = env.render_rgb_frame("head_camera")
            closed_seg = env.render_segmentation_frame("head_camera")
            container_obj.set_joint_position(joint["joint_index"], open_val)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            open_robot_pose_actual = env.current_robot.robot_view.base.pose.copy()
            open_vis = env.check_visibility("head_camera", *object_names)
            open_rgb = env.render_rgb_frame("head_camera")
            open_seg = env.render_segmentation_frame("head_camera")
        finally:
            env.current_data.qpos[:] = qpos_before
            mujoco.mj_forward(env.current_model, env.current_data)
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        if isinstance(closed_vis, float):
            closed_map = {object_names[0]: float(closed_vis)}
        else:
            closed_map = {k: float(v) for k, v in closed_vis.items()}
        if isinstance(open_vis, float):
            open_map = {object_names[0]: float(open_vis)}
        else:
            open_map = {k: float(v) for k, v in open_vis.items()}

        for object_name in object_names:
            cval = closed_map.get(object_name, 0.0)
            oval = open_map.get(object_name, 0.0)
            image_paths = {}
            if output_dir is not None:
                img_dir = output_dir / "visibility_images"
                base = (
                    f"{sanitize_name(container_rec['name'])}__joint{joint['joint_index']}__"
                    f"{sanitize_name(object_name)}"
                )
                closed_rgb_path = img_dir / f"{base}__closed_rgb.png"
                open_rgb_path = img_dir / f"{base}__open_rgb.png"
                closed_seg_path = img_dir / f"{base}__closed_seg.png"
                open_seg_path = img_dir / f"{base}__open_seg.png"
                closed_mask_path = img_dir / f"{base}__closed_mask.png"
                open_mask_path = img_dir / f"{base}__open_mask.png"
                save_rgb_image(closed_rgb_path, closed_rgb)
                save_rgb_image(open_rgb_path, open_rgb)
                save_segmentation_preview(closed_seg_path, closed_seg)
                save_segmentation_preview(open_seg_path, open_seg)
                try:
                    body_id = env.current_model.body(object_name).id
                    closed_mask = get_geom_seg_mask(env.current_model, closed_seg[..., :2], body_id)
                    open_mask = get_geom_seg_mask(env.current_model, open_seg[..., :2], body_id)
                except KeyError:
                    closed_mask = None
                    open_mask = None
                save_object_mask_preview(closed_mask_path, closed_mask)
                save_object_mask_preview(open_mask_path, open_mask)
                image_paths = {
                    "closed_rgb": str(closed_rgb_path),
                    "open_rgb": str(open_rgb_path),
                    "closed_seg": str(closed_seg_path),
                    "open_seg": str(open_seg_path),
                    "closed_mask": str(closed_mask_path),
                    "open_mask": str(open_mask_path),
                }
            results.append(
                {
                    "container_name": container_rec["name"],
                    "joint_index": joint["joint_index"],
                    "joint_name": joint["joint_name"],
                    "object_name": object_name,
                    "closed_visibility": cval,
                    "open_visibility": oval,
                    "delta_visibility": oval - cval,
                    "became_visible": cval <= 1e-4 and oval > 1e-4,
                    "shared_pose_ok": shared_pose is not None,
                    "shared_robot_pose": shared_pose.tolist() if shared_pose is not None else None,
                    "closed_robot_pose_actual": closed_robot_pose_actual.tolist()
                    if shared_pose is not None
                    else None,
                    "open_robot_pose_actual": open_robot_pose_actual.tolist()
                    if shared_pose is not None
                    else None,
                    "shared_pose_meta": shared_pose_meta,
                    "image_paths": image_paths,
                }
            )
    return results


def articulation_dependency_order(
    target_joint_index: int,
    dependency_rows: list[dict[str, Any]],
) -> list[int]:
    """Return prerequisite joints in topological order followed by the target."""
    prerequisites = {
        int(row["joint_index"]): [int(value) for value in row["prerequisite_joint_indices"]]
        for row in dependency_rows
    }
    visiting: set[int] = set()
    visited: set[int] = set()
    order: list[int] = []

    def visit(joint_index: int) -> None:
        if joint_index in visited:
            return
        if joint_index in visiting:
            raise ValueError(f"Cyclic articulation dependency at joint {joint_index}")
        visiting.add(joint_index)
        for prerequisite in prerequisites.get(joint_index, []):
            visit(prerequisite)
        visiting.remove(joint_index)
        visited.add(joint_index)
        order.append(joint_index)

    visit(int(target_joint_index))
    return order


def apply_container_view_profile(
    ctx: LoadedContext,
    view_profile: str,
) -> dict[str, Any]:
    """Apply a deterministic head/torso profile used for GT visibility checks."""
    env = ctx.env
    torso_qpos = apply_default_torso_pose(env, ctx.initial_torso_qpos)
    head_qpos = apply_default_head_pose(env, ctx.initial_head_qpos)
    if view_profile == "drawer_low_view":
        torso_qpos = lean_torso_for_drawer_view(
            env,
            ctx.initial_torso_qpos,
            pitch_delta=0.35,
        )
        apply_default_head_pose(env, ctx.initial_head_qpos)
        head_qpos = lower_head_for_drawer_view(env, tilt_delta=0.30)
    elif view_profile != "default":
        raise ValueError(f"Unsupported container view profile: {view_profile}")
    apply_default_arm_pose(env)
    env.camera_manager.registry.update_all_cameras(env)
    return {
        "view_profile": view_profile,
        "head_qpos": None if head_qpos is None else np.asarray(head_qpos, dtype=float).tolist(),
        "torso_qpos": None
        if torso_qpos is None
        else np.asarray(torso_qpos, dtype=float).tolist(),
    }


def object_visibility_measurement(
    env,
    object_name: str,
    camera_name: str = "head_camera",
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Measure GT visibility as target segmentation pixels and image fraction."""
    segmentation = env.render_segmentation_frame(camera_name)
    body_id = env.current_model.body(object_name).id
    mask = get_geom_seg_mask(env.current_model, segmentation[..., :2], body_id)
    pixels = int(mask.sum())
    fraction = float(mask.mean())
    return fraction, pixels, segmentation, mask


def visibility_measurements_from_segmentation(
    env,
    segmentation: np.ndarray,
    object_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract several body masks from one already-rendered segmentation frame."""
    measurements: dict[str, dict[str, Any]] = {}
    for object_name in dict.fromkeys(object_names):
        body_id = env.current_model.body(object_name).id
        mask = get_geom_seg_mask(env.current_model, segmentation[..., :2], body_id)
        measurements[object_name] = {
            "visibility_fraction": float(mask.mean()),
            "visible_pixels": int(mask.sum()),
            "mask": mask,
        }
    return measurements


def _aabb_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    offsets = np.asarray(
        [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)],
        dtype=float,
    )
    return np.asarray(center, dtype=float)[None, :] + offsets * np.asarray(size, dtype=float)[None, :]


def object_may_be_in_camera_frustum(
    env,
    object_name: str,
    camera_name: str = "head_camera",
    *,
    fov_margin: float = 1.5,
) -> dict[str, Any]:
    """Conservatively reject objects wholly outside a camera frustum without rendering.

    This is only a prefilter: occlusion is deliberately not inferred here.  A
    positive result still requires segmentation validation; a negative result
    means every target AABB corner is behind or well outside the wide frustum.
    """
    model = env.current_model
    data = env.current_data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return {"possibly_visible": True, "reason": "camera_not_found"}
    try:
        body_id = model.body(object_name).id
        center, size = safe_body_aabb(model, data, body_id)
    except Exception as exc:
        return {"possibly_visible": True, "reason": f"aabb_unavailable:{exc}"}

    camera_position = np.asarray(data.cam_xpos[camera_id], dtype=float)
    camera_rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    # MuJoCo camera coordinates look along local -Z.  xmat has local axes as
    # world-space columns, so a row-vector world offset is transformed by xmat.
    local = (_aabb_corners(center, size) - camera_position[None, :]) @ camera_rotation
    depth = -local[:, 2]
    half_vertical_fov = np.deg2rad(float(model.cam_fovy[camera_id])) * 0.5
    tangent = np.tan(half_vertical_fov) * max(float(fov_margin), 1.0)
    # The head cameras are near-square in the collection configs.  Reusing the
    # widened vertical FOV horizontally makes this deliberately conservative.
    inside = (
        (depth > 1e-5)
        & (np.abs(local[:, 0]) <= depth * tangent)
        & (np.abs(local[:, 1]) <= depth * tangent)
    )
    return {
        "possibly_visible": bool(np.any(inside)),
        "reason": "aabb_intersects_wide_frustum" if np.any(inside) else "aabb_outside_frustum",
        "camera_name": camera_name,
    }


def valid_robot_poses_for_joint_sequence(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint_sequence: list[int],
    *,
    desired_distance: float = 0.8,
    max_poses: int = 12,
    front_axis_xy: np.ndarray | None = None,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Find robot poses collision-free for every prefix of a joint-open sequence."""
    env = ctx.env
    robot_view = env.current_robot.robot_view
    joints_by_index = {int(joint["joint_index"]): joint for joint in container_rec["joints"]}
    if not joint_sequence or any(index not in joints_by_index for index in joint_sequence):
        return []
    target_joint = joints_by_index[joint_sequence[-1]]
    target_center, _ = joint_target_geometry(env, container_rec, target_joint)
    if front_axis_xy is None:
        front_axis_xy = container_front_axis(container_rec)
    else:
        front_axis_xy = np.asarray(front_axis_xy, dtype=float)
        front_axis_xy /= max(float(np.linalg.norm(front_axis_xy)), 1e-9)
    lateral_axis_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    if free_points.size == 0:
        return []

    candidate_specs: list[tuple[np.ndarray, float, str]] = []
    for side_sign, side_name, side_penalty in (
        (1.0, "front", 0.0),
        (-1.0, "opposite", 0.0),
    ):
        for distance in (0.5, 0.65, 0.8, 0.95):
            base_xy = target_center[:2] + side_sign * front_axis_xy * distance
            for lateral_offset in (0.0, 0.2, -0.2):
                candidate_xy = base_xy + lateral_axis_xy * lateral_offset
                score = (
                    abs(distance - desired_distance)
                    + abs(lateral_offset)
                    + side_penalty
                )
                candidate_specs.append(
                    (candidate_xy, score, f"{side_name}_d{distance:.2f}_l{lateral_offset:+.2f}")
                )

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = robot_view.base.pose.copy()
    valid: list[tuple[np.ndarray, dict[str, Any]]] = []
    seen_free_points: set[tuple[float, float]] = set()
    try:
        for candidate_xy, base_score, label in sorted(candidate_specs, key=lambda item: item[1]):
            free_point = nearest_free_point(free_points, candidate_xy)
            if free_point is None:
                continue
            free_key = tuple(np.round(np.asarray(free_point[:2], dtype=float), 3))
            if free_key in seen_free_points:
                continue
            seen_free_points.add(free_key)
            yaw = yaw_to_face(free_point[:2], target_center[:2])
            pose = make_robot_pose_from_xy(robot_view, free_point[:2], yaw)
            collision_prefix = None
            for prefix_length in range(len(joint_sequence) + 1):
                env.current_data.qpos[:] = qpos_before
                mujoco.mj_forward(env.current_model, env.current_data)
                set_all_articulation_joints_closed(env, container_rec, container_rec["joints"])
                for joint_index in joint_sequence[:prefix_length]:
                    joint = joints_by_index[joint_index]
                    set_articulation_state_by_record(
                        env,
                        container_rec,
                        joint_index,
                        float(joint["open_value"]),
                    )
                if env.check_if_robot_collision_at_base_pose(robot_view, pose):
                    collision_prefix = prefix_length
                    break
            if collision_prefix is not None:
                continue
            valid.append(
                (
                    pose.copy(),
                    {
                        "candidate_label": label,
                        "candidate_target_xy": np.asarray(candidate_xy, dtype=float).tolist(),
                        "free_point_xy": np.asarray(free_point[:2], dtype=float).tolist(),
                        "target_center": np.asarray(target_center, dtype=float).tolist(),
                        "score": float(base_score + np.linalg.norm(free_point[:2] - candidate_xy)),
                        "validated_prefix_count": len(joint_sequence) + 1,
                    },
                )
            )
            if len(valid) >= max_poses:
                break
    finally:
        env.current_data.qpos[:] = qpos_before
        robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)
    valid.sort(key=lambda item: item[1]["score"])
    return valid


def container_visibility_trace(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    object_name: str,
    joint_sequence: list[int],
    robot_pose: np.ndarray,
    *,
    view_profile: str,
    force_slide_joints: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Render visibility after every prefix of a deterministic joint sequence."""
    env = ctx.env
    joints_by_index = {int(joint["joint_index"]): joint for joint in container_rec["joints"]}
    qpos_before = env.current_data.qpos.copy()
    qvel_before = env.current_data.qvel.copy()
    robot_pose_before = env.current_robot.robot_view.base.pose.copy()
    head_before = get_head_joint_position(env)
    torso_before = get_torso_joint_position(env)
    trace: list[dict[str, Any]] = []
    view_state: dict[str, Any] = {}
    target_joint = joints_by_index[joint_sequence[-1]] if joint_sequence else None
    try:
        set_all_articulation_joints_closed(env, container_rec, container_rec["joints"])
        env.current_robot.robot_view.base.pose = robot_pose.copy()
        mujoco.mj_forward(env.current_model, env.current_data)
        view_state = apply_container_view_profile(ctx, view_profile)
        for prefix_length in range(len(joint_sequence) + 1):
            drive = None
            if prefix_length > 0:
                joint_index = joint_sequence[prefix_length - 1]
                joint = joints_by_index[joint_index]
                if force_slide_joints and joint_mujoco_type_name(env, joint) == "slide":
                    drive = drive_articulation_state_by_record(
                        env,
                        container_rec,
                        joint_index,
                        float(joint["open_value"]),
                    )
                else:
                    set_articulation_state_by_record(
                        env,
                        container_rec,
                        joint_index,
                        float(joint["open_value"]),
                    )
            env.current_robot.robot_view.base.pose = robot_pose.copy()
            mujoco.mj_forward(env.current_model, env.current_data)
            view_state = apply_container_view_profile(ctx, view_profile)
            env.camera_manager.registry.update_all_cameras(env)
            frustum = object_may_be_in_camera_frustum(env, object_name)
            segmentation = None
            mask = None
            if frustum["possibly_visible"] or output_dir is not None:
                segmentation = env.render_segmentation_frame("head_camera")
                measurements = visibility_measurements_from_segmentation(
                    env, segmentation, [object_name, container_rec["name"]]
                )
                fraction = float(measurements[object_name]["visibility_fraction"])
                pixels = int(measurements[object_name]["visible_pixels"])
                mask = measurements[object_name]["mask"]
                container_fraction = float(
                    measurements[container_rec["name"]]["visibility_fraction"]
                )
                container_pixels = int(measurements[container_rec["name"]]["visible_pixels"])
            else:
                fraction = 0.0
                pixels = 0
                container_fraction = 0.0
                container_pixels = 0
            object_body_id = env.current_model.body(object_name).id
            object_center, object_size = safe_body_aabb(
                env.current_model, env.current_data, object_body_id
            )
            row = {
                "prefix_length": prefix_length,
                "opened_joint_indices": list(joint_sequence[:prefix_length]),
                "last_opened_joint_index": None
                if prefix_length == 0
                else int(joint_sequence[prefix_length - 1]),
                "visibility_fraction": fraction,
                "visible_pixels": pixels,
                "container_visibility_fraction": container_fraction,
                "container_visible_pixels": container_pixels,
                "geometric_visibility_prefilter": frustum,
                "object_position": env.current_data.xpos[object_body_id].copy().tolist(),
                "object_aabb": {
                    "center": object_center.tolist(),
                    "size": object_size.tolist(),
                },
                "drive": drive,
            }
            if target_joint is not None:
                joint_center, joint_size = joint_target_geometry(
                    env, container_rec, target_joint
                )
                row["target_joint_aabb"] = {
                    "center": np.asarray(joint_center, dtype=float).tolist(),
                    "size": np.asarray(joint_size, dtype=float).tolist(),
                }
            if (
                output_dir is not None
                and segmentation is not None
                and mask is not None
                and prefix_length in {0, len(joint_sequence)}
            ):
                tag = "closed" if prefix_length == 0 else "final"
                rgb_path = output_dir / f"{tag}_rgb.png"
                seg_path = output_dir / f"{tag}_seg.png"
                mask_path = output_dir / f"{tag}_mask.png"
                save_rgb_image(rgb_path, env.render_rgb_frame("head_camera"))
                save_segmentation_preview(seg_path, segmentation)
                save_object_mask_preview(mask_path, mask)
                row["image_paths"] = {
                    "rgb": str(rgb_path),
                    "segmentation": str(seg_path),
                    "mask": str(mask_path),
                }
            trace.append(row)
        return {"view_state": view_state, "trace": trace}
    finally:
        env.current_data.qpos[:] = qpos_before
        env.current_data.qvel[:] = qvel_before
        env.current_robot.robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)
        if torso_before is not None:
            set_torso_joint_position(env, torso_before)
        if head_before is not None:
            set_head_joint_position(env, head_before)
        apply_default_arm_pose(env)
        env.camera_manager.registry.update_all_cameras(env)


def slide_compartment_object_binding(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    object_name: str,
    target_joint_index: int,
    prerequisite_joint_indices: list[int],
    *,
    min_joint_motion_m: float = 0.05,
    min_motion_ratio: float = 0.5,
) -> dict[str, Any]:
    """Check whether an object follows a translating compartment during force opening."""
    env = ctx.env
    joints_by_index = {int(joint["joint_index"]): joint for joint in container_rec["joints"]}
    target_joint = joints_by_index[target_joint_index]
    if joint_mujoco_type_name(env, target_joint) != "slide":
        return {"applicable": False, "consistent": True, "reason": "target_joint_not_slide"}

    qpos_before = env.current_data.qpos.copy()
    qvel_before = env.current_data.qvel.copy()
    object_body_id = env.current_model.body(object_name).id
    try:
        set_all_articulation_joints_closed(env, container_rec, container_rec["joints"])
        for prerequisite_index in prerequisite_joint_indices:
            prerequisite = joints_by_index[prerequisite_index]
            set_articulation_state_by_record(
                env,
                container_rec,
                prerequisite_index,
                float(prerequisite["open_value"]),
            )
        closed_object_position = env.current_data.xpos[object_body_id].copy()
        closed_joint_center, closed_joint_size = joint_target_geometry(
            env, container_rec, target_joint
        )
        closed_object_center, closed_object_size = safe_body_aabb(
            env.current_model, env.current_data, object_body_id
        )
        drive = drive_articulation_state_by_record(
            env,
            container_rec,
            target_joint_index,
            float(target_joint["open_value"]),
        )
        open_joint_center, open_joint_size = joint_target_geometry(
            env, container_rec, target_joint
        )
        open_object_position = env.current_data.xpos[object_body_id].copy()
        open_object_center, open_object_size = safe_body_aabb(
            env.current_model, env.current_data, object_body_id
        )
        joint_delta = np.asarray(open_joint_center) - np.asarray(closed_joint_center)
        object_delta = np.asarray(open_object_position) - np.asarray(closed_object_position)
        joint_motion = float(np.linalg.norm(joint_delta))
        if joint_motion < min_joint_motion_m:
            return {
                "applicable": False,
                "consistent": True,
                "reason": "slide_joint_motion_below_threshold",
                "joint_motion_m": joint_motion,
                "drive": drive,
                "closed_joint_box": {
                    "center": np.asarray(closed_joint_center, dtype=float).tolist(),
                    "size": np.asarray(closed_joint_size, dtype=float).tolist(),
                },
                "open_joint_box": {
                    "center": np.asarray(open_joint_center, dtype=float).tolist(),
                    "size": np.asarray(open_joint_size, dtype=float).tolist(),
                },
                "closed_object_box": {
                    "center": np.asarray(closed_object_center, dtype=float).tolist(),
                    "size": np.asarray(closed_object_size, dtype=float).tolist(),
                },
                "open_object_box": {
                    "center": np.asarray(open_object_center, dtype=float).tolist(),
                    "size": np.asarray(open_object_size, dtype=float).tolist(),
                },
            }
        axis = joint_delta / joint_motion
        object_motion_along_axis = float(abs(np.dot(object_delta, axis)))
        motion_ratio = object_motion_along_axis / joint_motion
        return {
            "applicable": True,
            "consistent": bool(drive.get("reached", False) and motion_ratio >= min_motion_ratio),
            "joint_motion_m": joint_motion,
            "object_motion_m": float(np.linalg.norm(object_delta)),
            "object_motion_along_axis_m": object_motion_along_axis,
            "motion_ratio": motion_ratio,
            "minimum_motion_ratio": min_motion_ratio,
            "drive": drive,
            "closed_joint_box": {
                "center": np.asarray(closed_joint_center, dtype=float).tolist(),
                "size": np.asarray(closed_joint_size, dtype=float).tolist(),
            },
            "open_joint_box": {
                "center": np.asarray(open_joint_center, dtype=float).tolist(),
                "size": np.asarray(open_joint_size, dtype=float).tolist(),
            },
            "closed_object_box": {
                "center": np.asarray(closed_object_center, dtype=float).tolist(),
                "size": np.asarray(closed_object_size, dtype=float).tolist(),
            },
            "open_object_box": {
                "center": np.asarray(open_object_center, dtype=float).tolist(),
                "size": np.asarray(open_object_size, dtype=float).tolist(),
            },
        }
    finally:
        env.current_data.qpos[:] = qpos_before
        env.current_data.qvel[:] = qvel_before
        mujoco.mj_forward(env.current_model, env.current_data)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(frame.max()) if frame.size else 0.0
        if max_value <= 1.0 + 1e-6:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _phase_name(policy: Any) -> str:
    phase = getattr(policy, "current_phase", None)
    return getattr(phase, "name", str(phase))


def densify_pose_path(
    start_pose: np.ndarray,
    target_poses: list[np.ndarray],
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> list[np.ndarray]:
    """Interpolate an SE(3) path so articulated motion remains contact-friendly."""
    dense: list[np.ndarray] = []
    previous = np.asarray(start_pose, dtype=float)
    for target in target_poses:
        target = np.asarray(target, dtype=float)
        translation = float(np.linalg.norm(target[:3, 3] - previous[:3, 3]))
        relative_rotation = R.from_matrix(previous[:3, :3]).inv() * R.from_matrix(
            target[:3, :3]
        )
        rotation = float(relative_rotation.magnitude())
        segments = max(
            1,
            int(np.ceil(translation / max_translation_m)),
            int(np.ceil(rotation / max_rotation_rad)),
        )
        rotations = R.from_matrix(np.stack([previous[:3, :3], target[:3, :3]]))
        slerp = Slerp([0.0, 1.0], rotations)
        for alpha in np.linspace(0.0, 1.0, segments + 1)[1:]:
            pose = np.eye(4)
            pose[:3, 3] = (1.0 - alpha) * previous[:3, 3] + alpha * target[:3, 3]
            pose[:3, :3] = slerp([alpha]).as_matrix()[0]
            dense.append(pose)
        previous = target
    return dense


def _read_interaction_joint(task: BaseMujocoTask, interaction_kind: str) -> float:
    if interaction_kind == "door":
        door = task.door_object
        return float(door.get_joint_position(door.get_hinge_joint_index()))
    target = task.articulation_objects[0][0]
    return float(target.get_joint_position(task.config.task_config.joint_index))


def _read_interaction_joint_range(
    task: BaseMujocoTask, interaction_kind: str
) -> list[float]:
    if interaction_kind == "door":
        door = task.door_object
        values = door.get_joint_range(door.get_hinge_joint_index())
    else:
        target = task.articulation_objects[0][0]
        values = target.get_joint_range(task.config.task_config.joint_index)
    return [float(values[0]), float(values[1])]


def semantic_open_fraction(value: float, closed_value: float, open_value: float) -> float:
    """Normalize a joint value using its semantic closed/open endpoints."""
    span = float(open_value) - float(closed_value)
    if abs(span) <= 1e-8:
        return 0.0
    return float(np.clip((float(value) - float(closed_value)) / span, 0.0, 1.0))


def stable_rby1_container_policy_cls(
    arm_preference: str = "auto",
    max_steps_per_waypoint: int = 30,
    max_batch_plan_attempts: int = 8,
    max_planning_reattempts: int = 8,
    pregrasp_tolerance: float = 0.04,
):
    """Build a local policy variant with target-aware torso and retry controls."""
    from molmo_spaces.controllers.torso_height import TorsoHeightJointPosController
    from molmo_spaces.policy.solvers.object_manipulation.curobo_open_close_planner_policy import (
        CuroboOpenClosePlannerPolicy,
        OpenClosePhase,
    )

    class StableRBY1ContainerOpenPolicy(CuroboOpenClosePlannerPolicy):
        def __init__(self, config, task) -> None:
            config.policy_config.max_steps_per_waypoint = int(max_steps_per_waypoint)
            config.policy_config.max_batch_plan_attempts = int(max_batch_plan_attempts)
            config.policy_config.max_planning_reattempts = int(max_planning_reattempts)
            super().__init__(config, task)
            self._torso_height_candidates: list[float] | None = None
            self._torso_height_index = 0
            self.torso_height_attempts: list[float] = []
            self.articulation_grasp_losses = 0

        def select_arm(self) -> None:
            if arm_preference not in {"auto", "left", "right"}:
                raise ValueError(f"Unsupported container arm preference: {arm_preference}")
            task_config = self.config.task_config
            om = self.task.env.object_managers[self.task.env.current_batch_index]
            pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
            target = self.task.articulation_objects[0][0]
            joint_id = target.joint_ids[int(task_config.joint_index)]
            body_id = int(self.task.env.current_model.jnt_bodyid[joint_id])
            target_center, _target_size = body_aabb(
                self.task.env.current_model,
                self.task.env.current_data,
                body_id,
                visible_only=False,
            )
            left_tcp = self.task.env.current_robot.robot_view.get_move_group(
                "left_gripper"
            ).leaf_frame_to_world[:3, 3]
            right_tcp = self.task.env.current_robot.robot_view.get_move_group(
                "right_gripper"
            ).leaf_frame_to_world[:3, 3]
            # The joint body can be much larger than its actual handle.  For
            # asymmetric containers (e.g. a two-door refrigerator), selecting
            # from the body center can therefore choose the wrong arm.  Use the
            # measured per-joint grasp poses whenever available and retain the
            # body center as a conservative fallback for unusual assets.
            left_distance = float(np.linalg.norm(left_tcp - target_center))
            right_distance = float(np.linalg.norm(right_tcp - target_center))
            selection_reason = "joint_body_center_fallback"
            grasp_candidate_count = 0
            try:
                from molmo_spaces.utils.grasp_sample import get_all_grasp_poses

                grasp_poses, _gripper, _object_pose = get_all_grasp_poses(
                    self, pickup_obj
                )
                grasp_positions = np.asarray(grasp_poses[:, :3, 3], dtype=float)
                if grasp_positions.size:
                    grasp_candidate_count = int(len(grasp_positions))
                    left_distance = float(
                        np.min(np.linalg.norm(grasp_positions - left_tcp, axis=1))
                    )
                    right_distance = float(
                        np.min(np.linalg.norm(grasp_positions - right_tcp, axis=1))
                    )
                    selection_reason = "joint_grasp_pose"
            except Exception as exc:
                log.warning(
                    "Could not load joint-specific grasp poses for %s joint %d; "
                    "falling back to joint body center: %s",
                    pickup_obj.name,
                    int(task_config.joint_index),
                    exc,
                )
            selected_arm = (
                "left" if left_distance <= right_distance else "right"
            ) if arm_preference == "auto" else arm_preference
            self.arm_side = selected_arm
            self.arm_selection_reason = (
                "caller_override" if arm_preference != "auto" else selection_reason
            )
            self.container_grasp_candidate_count = grasp_candidate_count
            if self._use_local_planner:
                self._select_arm_local(selected_arm)
            else:
                server_url = random.choice(self.config.policy_config.server_urls)
                log.info(
                    "Connecting to CuroboPlanner server at %s (arm=%s)",
                    server_url,
                    selected_arm,
                )
                server_timeout = getattr(
                    self.config.policy_config, "server_timeout", 120.0
                )
                self.client = CuroboClient(
                    base_url=server_url, arm=selected_arm, timeout=server_timeout
                )
                self._base_lock_joints = self.client.get_lock_joints()
            if selected_arm == "left":
                self.planner_joint_ranges = self.config.policy_config.left_planner_joint_ranges
            else:
                self.planner_joint_ranges = self.config.policy_config.right_planner_joint_ranges
            self.arm_start_idx = self.planner_joint_ranges[f"{selected_arm}_arm"][0]
            self.arm_end_idx = self.planner_joint_ranges[f"{selected_arm}_arm"][1]
            log.info(
                "Selected %s arm for %s joint %d "
                "(left_dist=%.3f right_dist=%.3f preference=%s reason=%s "
                "grasp_candidates=%d)",
                selected_arm,
                pickup_obj.name,
                int(task_config.joint_index),
                left_distance,
                right_distance,
                arm_preference,
                self.arm_selection_reason,
                grasp_candidate_count,
            )

        def reset(self, reset_retries: bool = True) -> None:
            super().reset(reset_retries=reset_retries)
            self._torso_height_candidates = None
            self._torso_height_index = 0
            self.torso_height_attempts = []
            self.articulation_grasp_losses = 0

        def _target_joint_center_z(self) -> float:
            target = self.task.articulation_objects[0][0]
            joint_id = target.joint_ids[self.config.task_config.joint_index]
            body_id = int(self.task.env.current_model.jnt_bodyid[joint_id])
            center, _size = body_aabb(
                self.task.env.current_model,
                self.task.env.current_data,
                body_id,
                visible_only=False,
            )
            return float(center[2])

        def _build_torso_height_candidates(self) -> list[float]:
            env = self.task.env
            torso = env.current_robot.robot_view.get_move_group("torso")
            gripper = env.current_robot.robot_view.get_move_group(f"{self.arm_side}_gripper")
            torso_ctrl = env.current_robot.controllers["torso"]
            target_z = self._target_joint_center_z()
            qpos_before = env.current_data.qpos.copy()
            samples = np.linspace(0.0, float(torso_ctrl.max_height), 4)
            scored: list[tuple[float, float]] = []
            try:
                for height in samples:
                    torso.joint_pos = TorsoHeightJointPosController.height_to_joints(float(height))
                    mujoco.mj_forward(env.current_model, env.current_data)
                    gripper_z = float(gripper.leaf_frame_to_world[2, 3])
                    scored.append((abs(gripper_z - target_z), float(height)))
            finally:
                env.current_data.qpos[:] = qpos_before
                mujoco.mj_forward(env.current_model, env.current_data)

            candidates = [height for _score, height in sorted(scored)]
            log.info(
                "Target joint center z=%.3f; torso candidates=%s",
                target_z,
                [round(value, 3) for value in candidates],
            )
            return candidates

        def _select_height(self) -> float:
            if self._torso_height_candidates is None:
                self._torso_height_candidates = self._build_torso_height_candidates()
            height = self._torso_height_candidates[self._torso_height_index]
            self.torso_height_attempts.append(height)
            log.info(
                "Trying target-aware torso height %.3f (%d/%d)",
                height,
                self._torso_height_index + 1,
                len(self._torso_height_candidates),
            )
            return height

        def _is_waypoint_reached(self, waypoint, tolerance: float = 0.04) -> bool:
            current = np.asarray(self._get_current_joint_positions(), dtype=float)
            target = np.asarray(waypoint, dtype=float)
            diff = np.abs(current - target)
            self.last_waypoint_max_diff = float(np.max(diff)) if diff.size else 0.0
            self.last_waypoint_diff = diff.astype(float).tolist()
            if self.steps_spent_in_waypoint in {
                max(0, int(self.config.policy_config.max_steps_per_waypoint) - 1)
            }:
                log.warning(
                    "Waypoint convergence diagnostics: max_diff=%.4f diffs=%s",
                    self.last_waypoint_max_diff,
                    [round(float(value), 4) for value in diff],
                )
            effective_tolerance = float(tolerance)
            if getattr(self, "current_phase", None) == OpenClosePhase.PREGRASP:
                effective_tolerance = max(effective_tolerance, float(pregrasp_tolerance))
            return super()._is_waypoint_reached(waypoint, tolerance=effective_tolerance)

        def _stage_torso_height(self, height: float) -> None:
            """Keep planning and simulation on the same staged torso posture."""
            torso_group = self.task.env.current_robot.robot_view.get_move_group("torso")
            torso_group.joint_pos = TorsoHeightJointPosController.height_to_joints(height)
            torso_group.joint_vel = np.zeros_like(torso_group.joint_vel)
            mujoco.mj_forward(self.task.env.current_model, self.task.env.current_data)

        def _sync_local_planner_torso_lock(self) -> None:
            if not self._use_local_planner:
                return
            torso_joint_pos = self.task.env.current_robot.robot_view.get_move_group(
                "torso"
            ).joint_pos
            lock_joints = dict(
                self.planner.curobo_robot_config_dict["kinematics"].get(
                    "lock_joints", {}
                )
            )
            lock_joints.update(
                {
                    "torso_1": float(torso_joint_pos[1]),
                    "torso_2": float(torso_joint_pos[2]),
                    "torso_3": float(torso_joint_pos[3]),
                }
            )
            self.planner.motion_gen.update_locked_joints(
                lock_joints, self.planner.curobo_robot_config_dict
            )
            log.info(
                "Synchronized local CuRobo torso locks: torso_1=%.3f torso_2=%.3f torso_3=%.3f",
                lock_joints["torso_1"],
                lock_joints["torso_2"],
                lock_joints["torso_3"],
            )

        def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
            self._stage_torso_height(self.current_height)
            action = super().get_action(info)
            self._stage_torso_height(self.current_height)
            return action

        def _execute_height_selection_phase(self) -> dict[str, Any]:
            interpolation_steps = int(
                getattr(self.config.policy_config, "max_height_adjustment_steps", 40)
            )
            settle_limit = interpolation_steps * 2
            settle_tolerance = 0.025

            if self._height_target is None:
                log.info("ENTERING HEIGHT SELECTION PHASE")
                self._height_target = self._select_height()
                torso_joint_pos = self.task.env.current_robot.robot_view.get_move_group(
                    "torso"
                ).joint_pos
                self._height_initial = float(torso_joint_pos[1])
                log.info(
                    "Interpolating torso height %.3f -> %.3f over %d steps",
                    self._height_initial,
                    self._height_target,
                    interpolation_steps,
                )

            self.height_adjustment_steps += 1
            alpha = min(1.0, self.height_adjustment_steps / interpolation_steps)
            self.current_height = self._height_initial + alpha * (
                self._height_target - self._height_initial
            )
            torso_group = self.task.env.current_robot.robot_view.get_move_group("torso")
            torso_controller = self.task.env.current_robot.controllers["torso"]
            torso_controller.set_target(np.array([self.current_height]))
            self._stage_torso_height(self.current_height)
            actual_height = float(torso_group.joint_pos[1])
            if self.height_adjustment_steps == 1 or self.height_adjustment_steps % 10 == 0:
                log.info(
                    "Torso adjustment step=%d command=%.3f controller_target=%.3f actual=%.3f",
                    self.height_adjustment_steps,
                    self.current_height,
                    float(torso_controller.target[0]),
                    actual_height,
                )
            settled = abs(actual_height - self._height_target) <= settle_tolerance
            timed_out = self.height_adjustment_steps >= settle_limit
            if self.height_adjustment_steps < interpolation_steps or not (settled or timed_out):
                return {}

            if timed_out and not settled:
                log.warning(
                    "Torso settling timed out: target=%.3f actual=%.3f error=%.3f",
                    self._height_target,
                    actual_height,
                    abs(actual_height - self._height_target),
                )
            else:
                log.info(
                    "Torso settled: target=%.3f actual=%.3f after %d steps",
                    self._height_target,
                    actual_height,
                    self.height_adjustment_steps,
                )

            self.current_height = self._height_target
            torso_controller.hold_at_height(self._height_target)
            self._sync_local_planner_torso_lock()
            self.pre_grasp_poses = self._get_pregrasp_poses()
            self.current_phase = OpenClosePhase.PREGRASP
            self.height_adjustment_steps = 0
            self._height_initial = None
            self._height_target = None
            return {}

        def _execute_pre_grasp_phase(self) -> dict[str, Any]:
            if self.planned_trajectory is not None:
                return super()._execute_pre_grasp_phase()

            log.info("ENTERING PREGRASP PHASE")
            self.batch_plan_trajectory()
            if self.planned_trajectory:
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

            assert self._torso_height_candidates is not None
            if self._torso_height_index + 1 >= len(self._torso_height_candidates):
                raise ValueError(
                    "No pregrasp trajectory found for any torso height: "
                    f"{self.torso_height_attempts}"
                )

            self._torso_height_index += 1
            self.current_phase = OpenClosePhase.HEIGHT_SELECTION
            self.height_adjustment_steps = 0
            self._height_initial = None
            self._height_target = None
            self.pre_grasp_poses = None
            self.trajectory_index = 0
            self.steps_spent_in_waypoint = 0
            log.warning("Pregrasp planning failed; retrying with the next torso height")
            return {}

        def _execute_articulate_phase(self) -> dict[str, Any]:
            trajectory_exhausted = (
                self.planned_trajectory is not None
                and self.trajectory_index >= len(self.planned_trajectory)
            )
            if trajectory_exhausted and not self._grasping_something():
                self.articulation_grasp_losses += 1
                if self.articulation_grasp_losses >= 3:
                    raise ValueError(
                        "Lost the drawer grasp during articulation 3 times; "
                        "aborting instead of repeating indefinitely."
                    )
            return super()._execute_articulate_phase()

        def _get_articulation_poses(self) -> list[np.ndarray]:
            coarse_poses = super()._get_articulation_poses()
            start_pose = self.task.env.current_robot.robot_view.get_move_group(
                f"{self.arm_side}_gripper"
            ).leaf_frame_to_world.copy()
            dense_poses = densify_pose_path(
                start_pose,
                coarse_poses,
                max_translation_m=0.03,
                max_rotation_rad=float(np.deg2rad(10.0)),
            )
            log.info(
                "Densified articulation path from %d to %d poses",
                len(coarse_poses),
                len(dense_poses),
            )
            self.articulation_pose_index = 0
            return dense_poses

    return StableRBY1ContainerOpenPolicy


def stable_rby1_door_policy_cls(arm_preference: str = "auto"):
    """Build a door policy with logged official arm selection or an override."""
    from molmo_spaces.policy.solvers.opening_solver import DoorOpeningPlannerPolicy

    class StableRBY1DoorOpeningPolicy(DoorOpeningPlannerPolicy):
        def reset(self) -> None:
            super().reset()
            base_offset = float(self.config.task_config.additional_tcp_offset_distance)
            self.grasp_tcp_offset_candidates = [
                base_offset,
                base_offset - 0.015,
                base_offset + 0.015,
                base_offset - 0.030,
                base_offset + 0.030,
            ]
            self.grasp_tcp_offset_index = 0
            self.grasp_tcp_offsets_tried = [base_offset]

        def select_arm_for_opening(self) -> str:
            if arm_preference in {"left", "right"}:
                selected_arm = arm_preference
                selection_reason = "caller_override"
                handle_y = float("nan")
                hinge_y = float("nan")
            else:
                robot_pose = self.task.env.robots[0].get_world_pose_tf_mat()
                handle_world = self.task.get_door_handle_position()
                hinge_world = self.task.get_door_joint_position()
                handle_robot = np.linalg.inv(robot_pose) @ np.array([*handle_world, 1.0])
                hinge_robot = np.linalg.inv(robot_pose) @ np.array([*hinge_world, 1.0])
                handle_y = float(handle_robot[1])
                hinge_y = float(hinge_robot[1])
                selected_arm = "left" if hinge_y > handle_y else "right"
                selection_reason = "official_hinge_side_heuristic"

            self.arm_side = selected_arm
            self.planner_joint_ranges = self._get_planner_joint_ranges()
            self.arm_selection_reason = selection_reason
            self.handle_lateral_position_robot = handle_y
            self.hinge_lateral_position_robot = hinge_y
            log.info(
                "[ARM] Selected %s arm reason=%s hinge_y_robot=%.3f handle_y_robot=%.3f",
                selected_arm,
                selection_reason,
                hinge_y,
                handle_y,
            )
            return selected_arm

        def _execute_grasp_handle_phase(self) -> dict[str, Any]:
            about_to_fail = (
                self.grasping_timesteps >= self.config.policy_config.max_grasping_timesteps
                and not self._grasping_something()
            )
            if about_to_fail and self.grasp_tcp_offset_index + 1 < len(
                self.grasp_tcp_offset_candidates
            ):
                self.grasp_tcp_offset_index += 1
                next_offset = self.grasp_tcp_offset_candidates[self.grasp_tcp_offset_index]
                self.config.task_config.additional_tcp_offset_distance = next_offset
                self.grasp_tcp_offsets_tried.append(next_offset)
                log.warning(
                    "[GRASP] Retrying with door TCP offset %.3fm",
                    next_offset,
                )
            return super()._execute_grasp_handle_phase()

        def _is_waypoint_reached(self, waypoint, tolerance: float | None = None) -> bool:
            configured_tolerance = float(self.config.policy_config.joint_position_tolerance)
            current = np.asarray(self._get_current_joint_positions(), dtype=float)
            target = np.asarray(waypoint, dtype=float)
            diff = np.abs(current - target)
            reached = bool(np.all(diff < configured_tolerance))
            if not reached and self.steps_spent_in_waypoint + 1 >= self.max_steps_per_waypoint:
                log.warning(
                    "Door waypoint tracking max_error=%.4f tolerance=%.4f errors=%s",
                    float(diff.max()),
                    configured_tolerance,
                    np.round(diff, 4).tolist(),
                )
            return reached

    return StableRBY1DoorOpeningPolicy


def execute_rby1_whole_body_interaction(
    cfg: Any,
    episode_spec: EpisodeSpec,
    *,
    interaction_kind: str,
    variant: str,
    output_dir: Path,
    camera_names: tuple[str, ...] = DEFAULT_RBY1_INTERACTION_CAMERAS,
    max_steps: int | None = None,
    collection_hz: float | None = None,
    video_fps: float | None = None,
    navigation_speed_mps: float = 0.84,
    base_adjustment_target_pose: np.ndarray | None = None,
    base_adjustment_path: list[np.ndarray] | None = None,
    max_base_adjustment_steps: int = 120,
    post_interaction_path: list[np.ndarray] | None = None,
    max_post_interaction_steps: int = 600,
    lock_base_during_force: bool = True,
    initial_state_episode: dict[str, Any] | None = None,
    initial_articulation_overrides: dict[str, float] | None = None,
    initial_robot_state: dict[str, Any] | None = None,
    frame_callback: Any | None = None,
    step_callback: Any | None = None,
    interaction_executor: str = "policy",
    hold_base_during_policy: bool | None = None,
    allow_force_fallback: bool = False,
    force_fallback_target_fraction: float = 1.0,
    force_fallback_max_steps: int = 1000,
    force_duration_seconds: float = 2.0,
) -> dict[str, Any]:
    """Execute an official RBY1 manipulation policy and record its camera streams.

    This is the algorithm-facing primitive. The caller supplies a frozen episode
    containing the current robot base pose and a concrete articulation target.
    """
    if interaction_kind not in {"container", "door"}:
        raise ValueError(f"Unsupported interaction kind: {interaction_kind}")
    if interaction_executor not in {"policy", "force"}:
        raise ValueError(
            f"Unsupported interaction executor {interaction_executor!r}; expected policy/force"
        )
    resolved_collection_hz = float(
        collection_hz
        if collection_hz is not None
        else (video_fps if video_fps is not None else 5.0)
    )
    if resolved_collection_hz <= 0.0:
        raise ValueError("collection_hz must be positive")
    if video_fps is not None and not np.isclose(
        float(video_fps), resolved_collection_hz, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Full rollout video_fps must equal collection_hz")
    if force_duration_seconds <= 0.0:
        raise ValueError("force_duration_seconds must be positive")
    if navigation_speed_mps <= 0.0:
        raise ValueError("navigation_speed_mps must be positive")
    if hold_base_during_policy is None:
        # Container articulation should not let contact dynamics push the
        # navigation base away from the planned grasp stance.  Door opening
        # retains the historical free-base behavior unless the caller opts in.
        hold_base_during_policy = interaction_kind == "container"

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_spec.to_json_file(output_dir / "episode_spec.json")
    sampler = RBY1InteractionJsonTaskSampler(cfg, episode_spec)
    task = None
    try:
        mapping = sampler._get_dataset_index_map()
        house_variants = mapping[episode_spec.data_split][episode_spec.house_index]
        original_scene_path = Path(house_variants[variant])
        house_variants[variant] = prepare_writable_scene_path(original_scene_path)
        task = sampler.sample_task(house_index=episode_spec.house_index, variant=variant)
        if task is None:
            raise RuntimeError("The RBY1 interaction task sampler returned no task.")
        if (
            interaction_executor == "policy"
            and (cfg.policy_config is None or cfg.policy_config.policy_cls is None)
        ):
            raise RuntimeError("The RBY1 interaction policy config was not initialized.")

        policy = None
        if interaction_executor == "policy":
            policy = cfg.policy_config.policy_cls(cfg, task)
            task.register_policy(policy)
        observation, _info = task.reset()
        initial_joint_position_before_enforcement = _read_interaction_joint(
            task, interaction_kind
        )
        if interaction_kind == "door":
            reset_value = float(task.exp_config.task_config.articulated_joint_reset_state[0])
            task.door_object.set_joint_position(
                task.door_object.get_hinge_joint_index(), reset_value
            )
            task.current_door_joint_state = np.array([reset_value], dtype=float)
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            observation = task.get_observations()
        state_application = apply_episode_scene_state(
            task.env,
            initial_state_episode,
            articulation_overrides=initial_articulation_overrides,
        )
        robot_state_application = None
        if initial_robot_state is not None:
            robot_state_application = apply_robot_state(task.env, initial_robot_state)
        if initial_state_episode is not None or initial_robot_state is not None:
            observation = task.get_observations()
        initial_joint_position = _read_interaction_joint(task, interaction_kind)
        robot_base = task.env.current_robot.robot_view.base
        initial_base_position = robot_base.pose[:3, 3].copy()
        initial_robot_collision = task.env.check_robot_collision_in_current_pose()
        if initial_robot_collision:
            raise RuntimeError(
                "Selected RBY1 interaction pose is colliding after JSON task initialization."
            )

        robot_view = task.env.current_robot.robot_view
        upper_body_hold_groups = tuple(
            group_name
            for group_name in (
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
                "torso",
                "head",
            )
            if group_name in robot_view.move_group_ids()
        )
        upper_body_hold_qpos = {
            group_name: np.asarray(
                robot_view.get_move_group(group_name).joint_pos, dtype=float
            ).copy()
            for group_name in upper_body_hold_groups
        }
        upper_body_hold_ctrl = {
            group_name: np.asarray(
                robot_view.get_move_group(group_name).noop_ctrl, dtype=float
            ).copy()
            for group_name in upper_body_hold_groups
        }

        def restore_upper_body_pose() -> None:
            for group_name in upper_body_hold_groups:
                group = robot_view.get_move_group(group_name)
                group.joint_pos = upper_body_hold_qpos[group_name].copy()
                group.joint_vel = np.zeros_like(group.joint_vel)
                group.ctrl = upper_body_hold_ctrl[group_name].copy()

        def execute_navigation_step(
            target_pose: np.ndarray,
            target_action: np.ndarray,
        ) -> tuple[list[dict[str, Any]], float, bool, bool, Any]:
            # Reset residual holonomic-joint velocity before each low-level
            # update so fixed-rate position targets do not accumulate the
            # oscillation seen with sparse absolute waypoints.
            robot_base.joint_vel = np.zeros_like(robot_base.joint_vel)
            action_cmd = {
                group_name: target.copy()
                for group_name, target in upper_body_hold_ctrl.items()
            }
            action_cmd["base"] = target_action
            observation, reward, terminal, truncated, infos = task.step(action_cmd)
            restore_upper_body_pose()
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            task.env.camera_manager.registry.update_all_cameras(task.env)
            observation = task.get_observations()
            return (
                observation,
                float(np.asarray(reward).reshape(-1)[0]),
                bool(np.asarray(terminal).reshape(-1)[0]),
                bool(np.asarray(truncated).reshape(-1)[0]),
                infos,
            )

        selected_cameras = tuple(
            name for name in camera_names if name in task.env.camera_manager.registry
        )
        if not selected_cameras:
            raise RuntimeError(f"None of the requested cameras are available: {camera_names}")
        frames: dict[str, list[np.ndarray]] = {name: [] for name in selected_cameras}
        capture_video_fps = resolved_collection_hz
        collection_dt_seconds = 1.0 / resolved_collection_hz
        sim_dt_seconds = float(task.env.current_model.opt.timestep)
        task_step_hz = float(getattr(cfg, "fps", 10.0))
        navigation_internal_steps_per_sample = max(
            1, int(round(task_step_hz / resolved_collection_hz))
        )
        navigation_physics_steps_per_sample = max(
            1, int(round(collection_dt_seconds / sim_dt_seconds))
        )
        recorded_sample_count = 0

        def execute_fixed_rate_navigation_sample(
            target_pose: np.ndarray,
            target_action: np.ndarray,
            *,
            linear_speed_mps: float = navigation_speed_mps,
            angular_speed_rad_s: float = 1.5,
        ) -> tuple[list[dict[str, Any]], float, bool, bool, Any]:
            robot_base.joint_vel = np.zeros_like(robot_base.joint_vel)
            for _physics_step in range(navigation_physics_steps_per_sample):
                restore_upper_body_pose()
                robot_base.ctrl = target_action.copy()
                velocity = np.asarray(robot_base.joint_vel, dtype=float).copy()
                current_xy = robot_base.pose[:2, 3]
                remaining = target_pose[:2, 3] - current_xy
                remaining_norm = float(np.linalg.norm(remaining))
                if remaining_norm > 1e-8:
                    direction = remaining / remaining_norm
                    along = float(np.dot(velocity[:2], direction))
                    velocity[:2] = direction * np.clip(
                        along, 0.0, float(linear_speed_mps)
                    )
                else:
                    velocity[:2] = 0.0
                velocity[2] = float(
                    np.clip(
                        velocity[2],
                        -float(angular_speed_rad_s),
                        float(angular_speed_rad_s),
                    )
                )
                robot_base.joint_vel = velocity
                mujoco.mj_step(task.env.current_model, task.env.current_data)
            restore_upper_body_pose()
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            task.env.camera_manager.registry.update_all_cameras(task.env)
            return task.get_observations(), 0.0, False, False, {
                "physics_steps": navigation_physics_steps_per_sample,
                "linear_speed_mps": float(linear_speed_mps),
                "angular_speed_rad_s": float(angular_speed_rad_s),
            }

        def capture(
            obs: list[dict[str, Any]],
            phase: str | None = None,
            *,
            segment: str | None = None,
            action: dict[str, Any] | None = None,
            reward: float = 0.0,
            terminal: bool = False,
            truncated: bool = False,
            infos: Any | None = None,
            record_video_frame: bool = True,
            record_step: bool = True,
        ) -> None:
            nonlocal recorded_sample_count
            resolved_segment = segment
            if resolved_segment is None:
                phase_name = str(phase or _phase_name(policy))
                resolved_segment = {
                    "INITIAL": "initial",
                    "BASE_ADJUSTMENT": f"nav_to_{interaction_kind}",
                    "BASE_ADJUSTMENT_COMPLETE": f"nav_to_{interaction_kind}",
                    "FORCE_OPEN": f"force_open_{interaction_kind}",
                    "FORCE_FALLBACK": f"force_open_{interaction_kind}",
                }.get(phase_name, "interaction")
            synchronized_frames: dict[str, np.ndarray] = {}
            for camera_name in selected_cameras:
                frame = obs[0].get(camera_name)
                if frame is None:
                    frame = task.env.render_rgb_frame(camera_name)
                frame_uint8 = _frame_to_uint8(frame)
                synchronized_frames[camera_name] = frame_uint8.copy()
                if record_video_frame:
                    frames[camera_name].append(frame_uint8)
                if record_video_frame and frame_callback is not None:
                    frame_callback(
                        camera_name,
                        frame_uint8.copy(),
                        {
                            "phase": phase or _phase_name(policy),
                            "segment": resolved_segment,
                            "task_step": int(task.episode_step_count),
                            "interaction_kind": interaction_kind,
                            "timestamp_seconds": float(
                                recorded_sample_count * collection_dt_seconds
                            ),
                            "dt_seconds": float(collection_dt_seconds),
                            "collection_hz": float(resolved_collection_hz),
                        },
                    )
            if record_step and step_callback is not None:
                components = {}
                vector_parts = []
                action_type = str((action or {}).get("__action_type__", "robot_control"))
                for key, value in (action or {}).items():
                    if str(key).startswith("__"):
                        continue
                    try:
                        values = np.asarray(value, dtype=np.float32).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    components[str(key)] = values.astype(float).tolist()
                    vector_parts.append(values)
                action_payload = {
                    "type": "initial" if action is None else action_type,
                    "vector": (
                        np.concatenate(vector_parts).astype(float).tolist()
                        if vector_parts
                        else []
                    ),
                    "components": components,
                }
                step_callback(
                    images=synchronized_frames,
                    action=action_payload,
                    state={
                        "qpos": np.asarray(task.env.current_data.qpos, dtype=float).tolist(),
                        "qvel": np.asarray(task.env.current_data.qvel, dtype=float).tolist(),
                    },
                    metadata={
                        "phase": phase or _phase_name(policy),
                        "segment": resolved_segment,
                        "task_step": int(task.episode_step_count),
                        "interaction_kind": interaction_kind,
                        "reward": float(reward),
                        "terminal": bool(terminal),
                        "truncated": bool(truncated),
                        "infos": to_jsonable(infos),
                        "timestamp_seconds": float(
                            recorded_sample_count * collection_dt_seconds
                        ),
                        "dt_seconds": float(collection_dt_seconds),
                        "collection_hz": float(resolved_collection_hz),
                    },
                )
            if record_step:
                recorded_sample_count += 1

        capture(observation, phase="INITIAL")
        phase_trace: list[str] = []
        base_adjustment_steps = 0
        base_targets = [
            np.asarray(pose, dtype=float)
            for pose in (base_adjustment_path or [])
        ]
        if not base_targets and base_adjustment_target_pose is not None:
            base_targets = [np.asarray(base_adjustment_target_pose, dtype=float)]
        base_adjustment_completed = not base_targets
        base_adjustment_waypoint_index = 0
        base_adjustment_error = None
        base_adjustment_collision_steps = 0
        base_position_after_adjustment = robot_base.pose[:3, 3].copy()
        max_base_adjustment_translation_per_step = 0.0
        if base_targets:
            phase_trace.append("BASE_ADJUSTMENT")
            if len(base_targets) > int(max_base_adjustment_steps):
                base_adjustment_error = (
                    "Fixed-rate navigation requires "
                    f"{len(base_targets)} samples, exceeding max_base_adjustment_steps="
                    f"{max_base_adjustment_steps}."
                )
            for adjustment_step, target_pose in enumerate(base_targets):
                if base_adjustment_error is not None:
                    break
                target_yaw = float(R.from_matrix(target_pose[:3, :3]).as_euler("XYZ")[2])
                target_action = np.array(
                    [target_pose[0, 3], target_pose[1, 3], target_yaw], dtype=float
                )
                action_cmd = {"base": target_action}
                position_before_step = robot_base.pose[:3, 3].copy()
                observation, _reward, _terminal, _truncated, _infos = (
                    execute_fixed_rate_navigation_sample(target_pose, target_action)
                )
                base_adjustment_steps = adjustment_step + 1
                base_adjustment_waypoint_index = adjustment_step + 1
                adjustment_translation = float(
                    np.linalg.norm(
                        robot_base.pose[:2, 3] - position_before_step[:2]
                    )
                )
                max_base_adjustment_translation_per_step = max(
                    max_base_adjustment_translation_per_step,
                    adjustment_translation,
                )
                capture(
                    observation,
                    phase="BASE_ADJUSTMENT",
                    action=action_cmd,
                    reward=float(np.asarray(_reward).reshape(-1)[0]),
                    terminal=bool(np.asarray(_terminal).reshape(-1)[0]),
                    truncated=bool(np.asarray(_truncated).reshape(-1)[0]),
                    infos={
                        "env_infos": _infos,
                        "collection_hz": resolved_collection_hz,
                        "dt_seconds": collection_dt_seconds,
                        "internal_physics_steps": navigation_physics_steps_per_sample,
                    },
                )
                if task.env.check_robot_collision_in_current_pose():
                    # Occupancy-map paths can end in deliberate contact with the
                    # interaction object. Keep recording the continuous path and
                    # let the interaction success gate decide whether the rollout
                    # is usable; retain the count for audit/debugging.
                    base_adjustment_collision_steps += 1

            if base_adjustment_error is None and base_targets:
                final_target = base_targets[-1]
                final_target_yaw = float(
                    R.from_matrix(final_target[:3, :3]).as_euler("XYZ")[2]
                )
                final_pose = robot_base.pose.copy()
                final_yaw = float(R.from_matrix(final_pose[:3, :3]).as_euler("XYZ")[2])
                final_position_error = float(
                    np.linalg.norm(final_pose[:2, 3] - final_target[:2, 3])
                )
                final_yaw_error = float(
                    abs(
                        np.arctan2(
                            np.sin(final_yaw - final_target_yaw),
                            np.cos(final_yaw - final_target_yaw),
                        )
                    )
                )
                base_adjustment_completed = (
                    final_position_error <= 0.15 and final_yaw_error <= 0.35
                )

            if not base_adjustment_completed and base_adjustment_error is None:
                base_adjustment_error = (
                    "Base adjustment did not reach the interaction stance within "
                    f"{max_base_adjustment_steps} steps."
                )
            if base_adjustment_error is not None:
                log.error(base_adjustment_error)
            else:
                try:
                    if policy is not None:
                        policy.reset()
                except TypeError:
                    policy.reset(reset_retries=True)
                observation = task.get_observations()
            base_position_after_adjustment = robot_base.pose[:3, 3].copy()

        phase_trace.append(
            "FORCE_OPEN" if interaction_executor == "force" else _phase_name(policy)
        )
        step_limit = int(max_steps if max_steps is not None else (cfg.task_horizon or 400))
        error_message = base_adjustment_error
        previous_base_position = robot_base.pose[:3, 3].copy()
        hold_base_target = np.array(
            [
                float(robot_base.pose[0, 3]),
                float(robot_base.pose[1, 3]),
                float(R.from_matrix(robot_base.pose[:3, :3]).as_euler("XYZ")[2]),
            ],
            dtype=float,
        )
        max_base_translation_per_step = 0.0
        max_base_translation_step = 0
        for _step in range(
            step_limit if base_adjustment_completed and interaction_executor == "policy" else 0
        ):
            if bool(np.all(task.is_done())):
                break
            try:
                action_cmd = policy.get_action(observation)
                if action_cmd is None:
                    raise RuntimeError(f"Policy returned no action in phase {_phase_name(policy)}")
                if hold_base_during_policy and "base" not in action_cmd:
                    action_cmd["base"] = hold_base_target.copy()
                observation, _reward, _terminal, _truncated, _infos = task.step(action_cmd)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                log.exception("RBY1 interaction failed in phase %s", _phase_name(policy))
                break
            current_base_position = robot_base.pose[:3, 3].copy()
            base_translation = float(
                np.linalg.norm(current_base_position[:2] - previous_base_position[:2])
            )
            if base_translation > max_base_translation_per_step:
                max_base_translation_per_step = base_translation
                max_base_translation_step = int(task.episode_step_count)
            previous_base_position = current_base_position
            if base_translation > 0.25:
                error_message = (
                    "Robot base position jumped "
                    f"{base_translation:.3f}m at step {task.episode_step_count}; "
                    "aborting an invalid collision-driven rollout."
                )
                log.error(error_message)
                capture(
                    observation,
                    phase=_phase_name(policy),
                    action=action_cmd,
                    reward=float(np.asarray(_reward).reshape(-1)[0]),
                    terminal=bool(np.asarray(_terminal).reshape(-1)[0]),
                    truncated=bool(np.asarray(_truncated).reshape(-1)[0]),
                    infos=_infos,
                )
                break
            current_phase = _phase_name(policy)
            should_capture_policy_sample = (
                (_step + 1) % navigation_internal_steps_per_sample == 0
                or bool(np.asarray(_terminal).reshape(-1)[0])
                or bool(np.asarray(_truncated).reshape(-1)[0])
                or bool(np.all(task.is_done()))
            )
            if should_capture_policy_sample:
                capture(
                    observation,
                    phase=current_phase,
                    action=action_cmd,
                    reward=float(np.asarray(_reward).reshape(-1)[0]),
                    terminal=bool(np.asarray(_terminal).reshape(-1)[0]),
                    truncated=bool(np.asarray(_truncated).reshape(-1)[0]),
                    infos={
                        "env_infos": _infos,
                        "internal_task_steps": navigation_internal_steps_per_sample,
                    },
                )
            if current_phase != phase_trace[-1]:
                phase_trace.append(current_phase)
            if (_step + 1) % 50 == 0:
                live_joint_position = _read_interaction_joint(task, interaction_kind)
                live_joint_range = _read_interaction_joint_range(task, interaction_kind)
                live_span = abs(live_joint_range[1] - live_joint_range[0])
                live_fraction = (
                    abs(live_joint_position) / live_span if live_span > 1e-8 else 0.0
                )
                log.info(
                    "RBY1 progress: step=%d phase=%s joint=%.4f fraction=%.3f",
                    int(task.episode_step_count),
                    current_phase,
                    float(live_joint_position),
                    float(live_fraction),
                )

        force_executor_meta: dict[str, Any] | None = None
        force_success_threshold = float(
            getattr(getattr(cfg, "task_config", None), "task_success_threshold", 0.67)
        )
        if base_adjustment_completed and interaction_executor == "force":
            if interaction_kind == "door":
                joint_obj = task.door_object
                force_joint_index = joint_obj.get_hinge_joint_index()
                force_joint_name = joint_obj.joint_names[force_joint_index]
                force_closed_value = float(
                    task.exp_config.task_config.articulated_joint_reset_state[0]
                )
                force_open_value = float(
                    max(task.exp_config.task_config.articulated_joint_range)
                )
            else:
                joint_obj = task.articulation_objects[0][0]
                force_joint_index = int(task.config.task_config.joint_index)
                force_joint_name = joint_obj.joint_names[force_joint_index]
                force_closed_value = float(
                    task.exp_config.task_config.joint_start_position[0]
                )
                force_open_value = float(task.exp_config.task_config.joint_goal_position)
            force_target = force_closed_value + float(force_fallback_target_fraction) * (
                force_open_value - force_closed_value
            )
            controller = ForceJointController(task.env, force_joint_name, force_target)
            force_base_hold_target = np.array(
                [
                    float(robot_base.pose[0, 3]),
                    float(robot_base.pose[1, 3]),
                    float(R.from_matrix(robot_base.pose[:3, :3]).as_euler("XYZ")[2]),
                ],
                dtype=float,
            )
            force_base_hold_pose = robot_base.pose.copy()
            force_base_max_drift = 0.0
            force_upper_body_targets: dict[str, np.ndarray] = {}
            force_upper_body_hold_qpos: dict[str, np.ndarray] = {}
            force_upper_body_hold_qvel: dict[str, np.ndarray] = {}
            force_task_completion_reached = False
            force_termination_reason = "scheduled_duration_complete"
            force_dataset_steps = max(
                1, int(round(float(force_duration_seconds) * resolved_collection_hz))
            )
            force_internal_steps_per_sample = max(
                1, int(round(collection_dt_seconds / sim_dt_seconds))
            )
            required_force_internal_steps = (
                force_dataset_steps * force_internal_steps_per_sample
            )
            if required_force_internal_steps > int(force_fallback_max_steps):
                raise ValueError(
                    "Scheduled force rollout requires "
                    f"{required_force_internal_steps} internal steps for "
                    f"{force_duration_seconds:.3f}s at {resolved_collection_hz:.3f}Hz, "
                    f"exceeding force_fallback_max_steps={force_fallback_max_steps}."
                )
            for group_name in (
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
                "torso",
                "head",
            ):
                if group_name in robot_view.move_group_ids():
                    group = robot_view.get_move_group(group_name)
                    force_upper_body_targets[group_name] = np.asarray(
                        group.noop_ctrl, dtype=float
                    ).copy()
                    force_upper_body_hold_qpos[group_name] = np.asarray(
                        group.joint_pos, dtype=float
                    ).copy()
                    force_upper_body_hold_qvel[group_name] = np.zeros_like(
                        np.asarray(group.joint_vel, dtype=float)
                    )

            def apply_force_upper_body_lock() -> None:
                if not lock_base_during_force:
                    return
                for group_name, target in force_upper_body_targets.items():
                    group = robot_view.get_move_group(group_name)
                    # The position actuators alone are not strong enough to
                    # counter reaction forces from a force-driven articulation.
                    # Restore both state and actuator target so the arm/torso
                    # cannot sag during an oracle force interaction.
                    group.joint_pos = force_upper_body_hold_qpos[group_name].copy()
                    group.joint_vel = force_upper_body_hold_qvel[group_name].copy()
                    group.ctrl = target.copy()

            try:
                force_action: dict[str, Any] | None = None
                for force_sample_index in range(force_dataset_steps):
                    sample_efforts: list[float] = []
                    for internal_index in range(force_internal_steps_per_sample):
                        global_progress = (
                            force_sample_index
                            + (internal_index + 1) / force_internal_steps_per_sample
                        ) / force_dataset_steps
                        controller.target_value = force_closed_value + (
                            float(force_fallback_target_fraction)
                            * global_progress
                            * (force_open_value - force_closed_value)
                        )
                        if lock_base_during_force:
                            robot_base.pose = force_base_hold_pose
                            robot_base.joint_vel = np.zeros_like(robot_base.joint_vel)
                            robot_base.ctrl = force_base_hold_target.copy()
                            apply_force_upper_body_lock()
                            mujoco.mj_forward(task.env.current_model, task.env.current_data)
                        force_action = controller.step()
                        sample_efforts.append(float(force_action["effort"]))
                        current_base = robot_base.pose
                        force_base_max_drift = max(
                            force_base_max_drift,
                            float(
                                np.linalg.norm(
                                    current_base[:2, 3] - force_base_hold_target[:2]
                                )
                            ),
                        )
                        if lock_base_during_force:
                            robot_base.pose = force_base_hold_pose
                            robot_base.joint_vel = np.zeros_like(robot_base.joint_vel)
                            apply_force_upper_body_lock()
                            mujoco.mj_forward(task.env.current_model, task.env.current_data)
                    assert force_action is not None
                    force_fraction = semantic_open_fraction(
                        float(force_action["joint_value"]),
                        force_closed_value,
                        force_open_value,
                    )
                    force_task_completion_reached = bool(
                        force_task_completion_reached
                        or force_fraction >= force_success_threshold
                    )
                    target_fraction = float(force_fallback_target_fraction) * (
                        force_sample_index + 1
                    ) / force_dataset_steps
                    task.env.camera_manager.registry.update_all_cameras(task.env)
                    observation = task.get_observations()
                    force_action_payload = {
                        "__action_type__": "force_joint",
                        "effort": [force_action["effort"]],
                        "mean_effort": [float(np.mean(sample_efforts))],
                        "target_value": [controller.target_value],
                        "joint_value": [force_action["joint_value"]],
                        "target_fraction": [target_fraction],
                        "measured_fraction": [force_fraction],
                        "duration_seconds": [collection_dt_seconds],
                    }
                    if lock_base_during_force:
                        force_action_payload["base_hold"] = force_base_hold_target.tolist()
                    capture(
                        observation,
                        phase="FORCE_OPEN",
                        action=force_action_payload,
                        terminal=False,
                        infos={
                            **force_action,
                            "dataset_step": force_sample_index + 1,
                            "dataset_step_count": force_dataset_steps,
                            "target_fraction": target_fraction,
                            "measured_fraction": force_fraction,
                            "success_threshold_reached": force_task_completion_reached,
                            "internal_control_steps": force_internal_steps_per_sample,
                        },
                    )
            finally:
                controller.finish()
            force_executor_meta = controller.result()
            force_executor_meta["base_lock_enabled"] = bool(lock_base_during_force)
            force_executor_meta["base_hold_target"] = force_base_hold_target.tolist()
            force_executor_meta["base_max_xy_drift"] = float(force_base_max_drift)
            force_executor_meta["upper_body_lock_enabled"] = bool(lock_base_during_force)
            force_executor_meta["upper_body_lock_groups"] = sorted(force_upper_body_targets)
            force_executor_meta["task_success_threshold"] = float(force_success_threshold)
            force_executor_meta["task_success_threshold_reached"] = bool(
                force_task_completion_reached
            )
            force_executor_meta["termination_reason"] = force_termination_reason
            force_executor_meta["dataset_steps"] = int(force_dataset_steps)
            force_executor_meta["duration_seconds"] = float(force_duration_seconds)
            force_executor_meta["collection_hz"] = float(resolved_collection_hz)
            force_executor_meta["dt_seconds"] = float(collection_dt_seconds)
            force_executor_meta["internal_steps_per_dataset_step"] = int(
                force_internal_steps_per_sample
            )
            policy_success = bool(force_task_completion_reached)
        else:
            policy_success = bool(task.judge_success())
        force_fallback_meta: dict[str, Any] | None = None
        current_fraction = semantic_open_fraction(
            _read_interaction_joint(task, interaction_kind),
            float(
                task.exp_config.task_config.articulated_joint_reset_state[0]
                if interaction_kind == "door"
                else task.exp_config.task_config.joint_start_position[0]
            ),
            float(
                max(task.exp_config.task_config.articulated_joint_range)
                if interaction_kind == "door"
                else task.exp_config.task_config.joint_goal_position
            ),
        )
        has_contact_phase = (
            ("OPEN_DOOR" in phase_trace or "GRASP_HANDLE" in phase_trace)
            if interaction_kind == "door"
            else ("ARTICULATE" in phase_trace or "GRASP" in phase_trace)
        )
        if (
            allow_force_fallback
            and has_contact_phase
            and interaction_executor == "policy"
            and current_fraction < float(force_fallback_target_fraction)
        ):
            if interaction_kind == "door":
                joint_obj = task.door_object
                joint_index = joint_obj.get_hinge_joint_index()
                joint_name = joint_obj.joint_names[joint_index]
                closed_value = float(task.exp_config.task_config.articulated_joint_reset_state[0])
                open_value = float(max(task.exp_config.task_config.articulated_joint_range))
            else:
                joint_obj = task.articulation_objects[0][0]
                joint_index = int(task.config.task_config.joint_index)
                joint_name = joint_obj.joint_names[joint_index]
                closed_value = float(task.exp_config.task_config.joint_start_position[0])
                open_value = float(task.exp_config.task_config.joint_goal_position)
            fallback_target = closed_value + float(force_fallback_target_fraction) * (
                open_value - closed_value
            )
            log.warning(
                "Policy phase ended at fraction %.3f; applying explicit force fallback to %.3f. "
                "This is recorded separately from policy success.",
                current_fraction,
                float(force_fallback_target_fraction),
            )
            drive_meta = drive_joint_to_value_with_force(
                task.env,
                joint_name,
                fallback_target,
                max_steps=int(force_fallback_max_steps),
            )
            observation = task.get_observations()
            capture(observation, phase="FORCE_FALLBACK")
            phase_trace.append("FORCE_FALLBACK")
            force_fallback_meta = {
                "used": True,
                "target_fraction": float(force_fallback_target_fraction),
                "target_value": float(fallback_target),
                "joint_name": joint_name,
                "joint_index": int(joint_index),
                "drive": drive_meta,
                "policy_success_before_fallback": policy_success,
            }

        interaction_success = bool(
            policy_success
            or (
                force_fallback_meta is not None
                and bool(force_fallback_meta["drive"].get("reached", False))
            )
            or current_fraction >= float(force_fallback_target_fraction)
            or (
                interaction_executor == "force"
                and current_fraction >= force_success_threshold
            )
        )
        post_targets = [
            np.asarray(pose, dtype=float)
            for pose in (post_interaction_path or [])
        ]
        post_navigation_completed = not post_targets
        post_navigation_error: str | None = None
        post_navigation_steps = 0
        post_navigation_waypoint_index = 0
        post_navigation_collision_steps = 0
        if post_targets and interaction_success:
            phase_trace.append("POST_NAVIGATION")
            if len(post_targets) > int(max_post_interaction_steps):
                post_navigation_error = (
                    "Fixed-rate post navigation requires "
                    f"{len(post_targets)} samples, exceeding max_post_interaction_steps="
                    f"{max_post_interaction_steps}."
                )
            for post_step, target_pose in enumerate(post_targets):
                if post_navigation_error is not None:
                    break
                target_yaw = float(R.from_matrix(target_pose[:3, :3]).as_euler("XYZ")[2])
                target_action = np.array(
                    [target_pose[0, 3], target_pose[1, 3], target_yaw], dtype=float
                )
                action_cmd = {"base": target_action}
                observation, _reward, _terminal, _truncated, _infos = (
                    execute_fixed_rate_navigation_sample(target_pose, target_action)
                )
                post_navigation_steps = post_step + 1
                post_navigation_waypoint_index = post_step + 1
                capture(
                    observation,
                    phase="POST_NAVIGATION",
                    segment="nav_to_target",
                    action=action_cmd,
                    reward=float(np.asarray(_reward).reshape(-1)[0]),
                    terminal=bool(np.asarray(_terminal).reshape(-1)[0]),
                    truncated=bool(np.asarray(_truncated).reshape(-1)[0]),
                    infos={
                        "env_infos": _infos,
                        "collection_hz": resolved_collection_hz,
                        "dt_seconds": collection_dt_seconds,
                        "internal_physics_steps": navigation_physics_steps_per_sample,
                    },
                )
                if task.env.check_robot_collision_in_current_pose():
                    post_navigation_collision_steps += 1
            if post_navigation_error is None and post_targets:
                final_target = post_targets[-1]
                final_target_yaw = float(
                    R.from_matrix(final_target[:3, :3]).as_euler("XYZ")[2]
                )
                final_pose = robot_base.pose.copy()
                final_yaw = float(R.from_matrix(final_pose[:3, :3]).as_euler("XYZ")[2])
                final_position_error = float(
                    np.linalg.norm(final_pose[:2, 3] - final_target[:2, 3])
                )
                final_yaw_error = float(
                    abs(
                        np.arctan2(
                            np.sin(final_yaw - final_target_yaw),
                            np.cos(final_yaw - final_target_yaw),
                        )
                    )
                )
                post_navigation_completed = (
                    final_position_error <= 0.15 and final_yaw_error <= 0.35
                )
            if not post_navigation_completed and post_navigation_error is None:
                post_navigation_error = (
                    "Post-interaction navigation did not reach the target within "
                    f"{max_post_interaction_steps} steps."
                )
        elif post_targets:
            post_navigation_error = "Post-interaction navigation skipped because interaction failed."

        if post_navigation_error is not None:
            error_message = post_navigation_error if error_message is None else (
                f"{error_message}; {post_navigation_error}"
            )
        phase_trace.append("TERMINAL_OBSERVATION")
        terminal_observation = task.get_observations()
        capture(
            terminal_observation,
            phase="TERMINAL_OBSERVATION",
            segment="terminal_observation",
            action={"__action_type__": "observe"},
            terminal=bool(interaction_success and post_navigation_completed),
            truncated=False,
            infos={
                "interaction_success": interaction_success,
                "post_navigation_completed": post_navigation_completed,
            },
        )

        fps = capture_video_fps
        video_paths: dict[str, str] = {}
        from molmo_spaces.utils.save_utils import save_frames_to_mp4

        for camera_name, camera_frames in frames.items():
            path = output_dir / f"{camera_name}.mp4"
            save_frames_to_mp4(camera_frames, str(path), fps=fps)
            video_paths[camera_name] = str(path)

        final_joint_position = _read_interaction_joint(task, interaction_kind)
        final_joint_range = _read_interaction_joint_range(task, interaction_kind)
        joint_span = abs(final_joint_range[1] - final_joint_range[0])
        joint_open_fraction = (
            abs(final_joint_position) / joint_span if joint_span > 1e-8 else 0.0
        )
        if interaction_kind == "door":
            semantic_closed_value = float(
                task.exp_config.task_config.articulated_joint_reset_state[0]
            )
            semantic_open_value = float(
                max(task.exp_config.task_config.articulated_joint_range)
            )
        else:
            semantic_closed_value = float(task.exp_config.task_config.joint_start_position[0])
            semantic_open_value = float(task.exp_config.task_config.joint_goal_position)
        semantic_fraction = semantic_open_fraction(
            final_joint_position,
            semantic_closed_value,
            semantic_open_value,
        )
        result = {
            "success": bool(interaction_success and post_navigation_completed),
            "policy_success": policy_success,
            "success_source": "policy_plus_force_fallback"
            if force_fallback_meta is not None
            else interaction_executor,
            "interaction_executor": interaction_executor,
            "lock_base_during_force": bool(lock_base_during_force),
            "force_executor": force_executor_meta,
            "force_fallback": force_fallback_meta
            if force_fallback_meta is not None
            else {"used": False},
            "error": error_message,
            "interaction_kind": interaction_kind,
            "state_application": state_application,
            "robot_state_application": robot_state_application,
            "selected_arm": getattr(policy, "arm_side", None),
            "arm_selection_reason": getattr(policy, "arm_selection_reason", None),
            "handle_lateral_position_robot": getattr(
                policy, "handle_lateral_position_robot", None
            ),
            "hinge_lateral_position_robot": getattr(
                policy, "hinge_lateral_position_robot", None
            ),
            "steps": int(task.episode_step_count),
            "policy_dt_ms": float(cfg.policy_dt_ms),
            "simulated_seconds": float(task.episode_step_count * cfg.policy_dt_ms / 1000.0),
            "phase_trace": phase_trace,
            "final_phase": (
                "FORCE_OPEN" if interaction_executor == "force" else _phase_name(policy)
            ),
            "final_joint_position": final_joint_position,
            "initial_joint_position_before_enforcement": float(
                initial_joint_position_before_enforcement
            ),
            "initial_joint_position": float(initial_joint_position),
            "final_joint_range": final_joint_range,
            "joint_open_fraction": float(joint_open_fraction),
            "semantic_open_fraction": float(semantic_fraction),
            "force_success_threshold": float(force_success_threshold),
            "force_success_threshold_met": bool(
                interaction_executor == "force"
                and semantic_fraction >= force_success_threshold
            ),
            "semantic_closed_value": semantic_closed_value,
            "semantic_open_value": semantic_open_value,
            "policy_done": bool(getattr(policy, "is_done", False)),
            "torso_height_attempts": [
                float(value) for value in getattr(policy, "torso_height_attempts", [])
            ],
            "articulation_grasp_losses": int(
                getattr(policy, "articulation_grasp_losses", 0)
            ),
            "container_grasp_candidate_count": int(
                getattr(policy, "container_grasp_candidate_count", 0)
            ),
            "last_waypoint_max_diff": getattr(
                policy, "last_waypoint_max_diff", None
            ),
            "last_waypoint_diff": getattr(policy, "last_waypoint_diff", None),
            "grasp_tcp_offsets_tried": [
                float(value) for value in getattr(policy, "grasp_tcp_offsets_tried", [])
            ],
            "initial_robot_collision": bool(initial_robot_collision),
            "initial_base_position": initial_base_position.tolist(),
            "final_base_position": robot_base.pose[:3, 3].astype(float).tolist(),
            "final_robot_state": snapshot_robot_state(task.env),
            "max_base_translation_per_step": float(max_base_translation_per_step),
            "max_base_translation_step": int(max_base_translation_step),
            "base_adjustment_requested": bool(base_targets),
            "base_adjustment_waypoint_count": len(base_targets),
            "base_adjustment_waypoint_index": int(base_adjustment_waypoint_index),
            "base_adjustment_completed": bool(base_adjustment_completed),
            "base_adjustment_steps": int(base_adjustment_steps),
            "base_adjustment_error": base_adjustment_error,
            "base_adjustment_collision_steps": int(base_adjustment_collision_steps),
            "base_position_after_adjustment": base_position_after_adjustment.tolist(),
            "post_navigation_requested": bool(post_targets),
            "post_navigation_completed": bool(post_navigation_completed),
            "post_navigation_waypoint_count": len(post_targets),
            "post_navigation_waypoint_index": int(post_navigation_waypoint_index),
            "post_navigation_steps": int(post_navigation_steps),
            "post_navigation_error": post_navigation_error,
            "post_navigation_collision_steps": int(post_navigation_collision_steps),
            "hold_base_during_policy": bool(hold_base_during_policy),
            "hold_base_target": hold_base_target.tolist(),
            "max_base_adjustment_translation_per_step": float(
                max_base_adjustment_translation_per_step
            ),
            "video_fps": fps,
            "collection_hz": float(resolved_collection_hz),
            "dt_seconds": float(collection_dt_seconds),
            "task_step_hz": float(task_step_hz),
            "navigation_internal_steps_per_sample": int(
                navigation_internal_steps_per_sample
            ),
            "navigation_physics_steps_per_sample": int(
                navigation_physics_steps_per_sample
            ),
            "sim_dt_seconds": float(sim_dt_seconds),
            "video_paths": video_paths,
        }
        write_json(output_dir / "result.json", result)
        return result
    finally:
        sampler.close()


def _build_rby1_container_episode(
    args: argparse.Namespace,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    robot_pose: np.ndarray,
    object_pose: np.ndarray,
) -> EpisodeSpec:
    start_value = joint["closed_value"] if args.action == "open" else joint["open_value"]
    goal_value = joint["open_value"] if args.action == "open" else joint["closed_value"]
    return EpisodeSpec.model_validate(
        {
            "house_index": args.house_ind,
            "scene_dataset": args.scene_dataset,
            "data_split": args.data_split,
            "seed": args.seed,
            "robot": {"robot_name": "rby1m", "init_qpos": default_rby1_episode_qpos()},
            "img_resolution": [1024, 576],
            "cameras": default_rby1_episode_camera_specs(),
            "scene_modifications": {},
            "task": {
                "task_cls": "molmo_spaces.tasks.opening_tasks.OpeningTask",
                "task_type": args.action,
                "robot_base_pose": pose_mat_to_7d(robot_pose).tolist(),
                "pickup_obj_name": container_rec["name"],
                "pickup_obj_start_pose": pose_mat_to_7d(object_pose).tolist(),
                "articulation_object_name": container_rec["name"],
                "joint_name": joint["joint_name"],
                "joint_index": joint["joint_index"],
                "joint_start_position": [start_value],
                "joint_goal_position": goal_value,
                "task_success_threshold": args.success_threshold,
                "any_inst_of_category": False,
            },
            "task_relevant_objects": [container_rec["name"]],
            "language": {
                "task_description": f"{args.action.title()} {container_rec['name']} joint {joint['joint_index']}",
                "referral_expressions": {"pickup_obj_name": container_rec.get("category", "container")},
                "referral_expressions_priority": {},
            },
        }
    )


def _build_rby1_door_episode(
    args: argparse.Namespace,
    door_rec: dict[str, Any],
    robot_pose: np.ndarray,
) -> EpisodeSpec:
    if args.action != "open":
        raise ValueError("The official RBY1 door planner currently supports opening, not closing.")
    return EpisodeSpec.model_validate(
        {
            "house_index": args.house_ind,
            "scene_dataset": args.scene_dataset,
            "data_split": args.data_split,
            "seed": args.seed,
            "robot": {"robot_name": "rby1m", "init_qpos": default_rby1_episode_qpos()},
            "img_resolution": [1024, 576],
            "cameras": default_rby1_episode_camera_specs(),
            "scene_modifications": {},
            "task": {
                "task_cls": "molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
                "task_type": "open_door",
                "robot_base_pose": pose_mat_to_7d(robot_pose).tolist(),
                "door_body_name": door_rec["name"],
                "articulated_joint_range": door_rec["hinge_joint_range"],
                "articulated_joint_reset_state": [door_rec["closed_value"]],
                "door_openness_threshold": args.success_threshold,
            },
            "task_relevant_objects": [door_rec["name"]],
            "language": {
                "task_description": f"Open door {door_rec['name']}",
                "referral_expressions": {},
                "referral_expressions_priority": {},
            },
        }
    )


def prepare_rby1_interaction_episode(args: argparse.Namespace) -> tuple[EpisodeSpec, dict[str, Any]]:
    """Resolve the concrete scene target and freeze an interaction episode."""
    ctx = load_scene_context(args, args.house_ind)
    try:
        supplied_robot_pose = (
            pos_quat_to_pose_mat(args.robot_base_pose) if args.robot_base_pose is not None else None
        )
        if args.interaction_kind == "container":
            _records, containers = collect_scene_records(ctx)
            target = next((rec for rec in containers if rec["name"] == args.target_name), None)
            if target is None:
                raise RuntimeError(f"Container not found: {args.target_name}")
            if not 0 <= args.joint_index < len(target["joints"]):
                raise ValueError(
                    f"joint_index={args.joint_index} outside [0, {len(target['joints']) - 1}]"
                )
            joint = target["joints"][args.joint_index]
            robot_pose, pose_meta = resolve_rby1_interaction_pose(
                ctx,
                target,
                joint,
                joint["closed_value"],
                joint["open_value"],
                pose_mode=args.robot_pose_mode,
                supplied_robot_pose=supplied_robot_pose,
                desired_dist=args.approach_distance,
                torso_heights=(0.0, 0.246, 0.492, 0.738),
                min_clearance_m=args.min_base_clearance,
                max_center_distance_m=args.max_approach_distance,
                max_base_adjustment_distance_m=args.max_base_adjustment_distance,
                allow_back_approach=args.allow_back_approach,
            )
            if robot_pose is None:
                raise RuntimeError("Could not find a collision-free robot pose for both joint states.")
            obj = ctx.env.object_managers[0].get_object_by_name(target["name"])
            episode = _build_rby1_container_episode(args, target, joint, robot_pose, obj.pose)
            target_meta = {
                "target_name": target["name"],
                "joint_index": joint["joint_index"],
                "joint_name": joint["joint_name"],
                "joint_range": joint["joint_range"],
                "robot_pose_meta": pose_meta,
            }
        else:
            doors = collect_door_records(ctx)
            target = next((rec for rec in doors if rec["name"] == args.target_name), None)
            if target is None:
                raise RuntimeError(f"Door not found: {args.target_name}")
            joint = {
                "joint_index": target["hinge_joint_index"],
                "joint_name": target["hinge_joint_name"],
            }
            door_articulation = {
                "name": target["name"],
                "interaction_group": "portal",
                "aabb_center": target["aabb_center"],
                "aabb_size": target["aabb_size"],
                "quat": target["quat"],
            }
            robot_pose, pose_meta = resolve_rby1_interaction_pose(
                ctx,
                door_articulation,
                joint,
                target["closed_value"],
                target["open_value"],
                pose_mode=args.robot_pose_mode,
                supplied_robot_pose=supplied_robot_pose,
                desired_dist=args.approach_distance,
                torso_heights=(0.0,),
                min_clearance_m=args.min_base_clearance,
                max_center_distance_m=args.max_approach_distance,
                max_base_adjustment_distance_m=args.max_base_adjustment_distance,
                allow_back_approach=True,
            )
            if robot_pose is None:
                raise RuntimeError("Could not find a collision-free robot pose for the door swing.")
            episode = _build_rby1_door_episode(args, target, robot_pose)
            target_meta = {
                "target_name": target["name"],
                "joint_index": target["hinge_joint_index"],
                "joint_name": target["hinge_joint_name"],
                "joint_range": target["hinge_joint_range"],
                "robot_pose_meta": pose_meta,
            }
        return episode, target_meta
    finally:
        close_context(ctx)


def plan_rby1_interaction_stance(request: RBY1InteractionRequest) -> dict[str, Any]:
    """Return a collision-checked operation stance without moving the robot."""
    planning_request = copy.copy(request)
    planning_request.robot_pose_mode = "auto"
    planning_request.robot_base_pose = None
    args = _rby1_request_to_args(planning_request)
    episode_spec, target_meta = prepare_rby1_interaction_episode(args)
    episode_payload = episode_spec.model_dump(mode="json")
    return {
        "base_pose": episode_payload["task"]["robot_base_pose"],
        "interaction_kind": request.interaction_kind,
        "target_name": request.target_name,
        "joint_index": request.joint_index,
        **target_meta["robot_pose_meta"],
    }


def build_rby1_interaction_config(args: argparse.Namespace) -> Any:
    """Build the official planner config shared by CLI and algorithm callers."""
    if args.interaction_kind == "container":
        from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
            RBY1OpenDataGenConfig,
        )

        cfg = RBY1OpenDataGenConfig()
        cfg.robot_config.command_mode = {
            "arm": "joint_position",
            "gripper": "joint_position",
            "base": "holo_joint_planar_position",
            "head": None,
            "torso": "height",
        }
        if cfg.policy_config is not None:
            cfg.policy_config.server_urls = (
                [args.curobo_server_url] if args.curobo_server_url else []
            )
        if (
            getattr(args, "interaction_executor", "policy") == "policy"
            and cfg.policy_config is not None
        ):
            cfg.policy_config.policy_cls = stable_rby1_container_policy_cls(
                args.container_arm,
                max_steps_per_waypoint=args.container_max_steps_per_waypoint,
                max_batch_plan_attempts=args.container_max_batch_plan_attempts,
                max_planning_reattempts=args.container_max_planning_reattempts,
                pregrasp_tolerance=getattr(args, "container_pregrasp_tolerance", 0.04),
            )
        if cfg.policy_config is not None:
            cfg.policy_config.batch_size = 8
            cfg.policy_config.max_batch_plan_attempts = args.container_max_batch_plan_attempts
            cfg.policy_config.max_planning_reattempts = args.container_max_planning_reattempts
            cfg.policy_config.max_height_adjustment_steps = 40
            cfg.policy_config.max_steps_per_waypoint = args.container_max_steps_per_waypoint
    else:
        from molmo_spaces.data_generation.config.door_opening_configs import (
            DoorOpeningDataGenConfig,
        )

        cfg = DoorOpeningDataGenConfig()
        cfg.task_config.additional_tcp_offset_distance = args.door_tcp_offset
        if (
            getattr(args, "interaction_executor", "policy") == "policy"
            and cfg.policy_config is not None
        ):
            cfg.policy_config.policy_cls = stable_rby1_door_policy_cls(args.door_arm)
        if cfg.policy_config is not None:
            cfg.policy_config.max_steps_per_waypoint = args.door_max_steps_per_waypoint
            cfg.policy_config.joint_position_tolerance = args.door_joint_position_tolerance
            cfg.policy_config.articulation_deltas = [
                float(np.deg2rad(args.door_articulation_delta_deg))
            ]

    cfg.seed = args.seed
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.num_workers = 1
    cfg.use_passive_viewer = False
    cfg.use_filament = False
    cfg.task_horizon = args.max_steps + args.max_base_adjustment_steps
    cfg.task_sampler_config.randomize_lighting = False
    cfg.task_sampler_config.randomize_textures = False
    cfg.task_sampler_config.randomize_dynamics = False
    return cfg


def _rby1_request_to_args(request: RBY1InteractionRequest) -> argparse.Namespace:
    robot_base_pose = (
        None
        if request.robot_base_pose is None
        else np.asarray(request.robot_base_pose, dtype=float).reshape(7).tolist()
    )
    return argparse.Namespace(
        house_ind=int(request.house_ind),
        interaction_kind=request.interaction_kind,
        target_name=request.target_name,
        joint_index=int(request.joint_index),
        action="open",
        robot_pose_mode=request.robot_pose_mode,
        robot_base_pose=robot_base_pose,
        door_arm=request.door_arm,
        approach_distance=float(request.approach_distance),
        min_base_clearance=float(request.min_base_clearance),
        max_approach_distance=float(request.max_approach_distance),
        max_base_adjustment_distance=float(request.max_base_adjustment_distance),
        max_base_adjustment_steps=int(request.max_base_adjustment_steps),
        allow_back_approach=bool(request.allow_back_approach),
        door_tcp_offset=float(request.door_tcp_offset),
        door_max_steps_per_waypoint=int(request.door_max_steps_per_waypoint),
        door_max_planning_reattempts=int(request.door_max_planning_reattempts),
        door_joint_position_tolerance=float(request.door_joint_position_tolerance),
        door_articulation_delta_deg=float(request.door_articulation_delta_deg),
        allow_force_fallback=bool(request.allow_force_fallback),
        force_fallback_target_fraction=float(request.force_fallback_target_fraction),
        force_fallback_max_steps=int(request.force_fallback_max_steps),
        force_duration_seconds=float(request.force_duration_seconds),
        success_threshold=float(request.success_threshold),
        max_steps=int(request.max_steps),
        container_arm=request.container_arm,
        container_max_steps_per_waypoint=int(request.container_max_steps_per_waypoint),
        container_max_batch_plan_attempts=int(request.container_max_batch_plan_attempts),
        container_max_planning_reattempts=int(request.container_max_planning_reattempts),
        container_pregrasp_tolerance=float(request.container_pregrasp_tolerance),
        video_fps=request.video_fps,
        camera_names=list(request.camera_names),
        output_dir=Path(request.output_dir),
        scene_dataset=request.scene_dataset,
        data_split=request.data_split,
        variant=request.variant,
        seed=int(request.seed),
        robot="rby1",
        curobo_server_url=request.curobo_server_url,
        interaction_executor=getattr(request, "interaction_executor", "policy"),
    )


def open_articulation_with_rby1(request: RBY1InteractionRequest) -> dict[str, Any]:
    """Open a door/container from a navigation endpoint and record the rollout.

    ``current_or_adjust`` preserves a collision-free supplied base pose. If the
    navigation endpoint is unsafe or too far away, it falls back to a nearby
    pose that is valid with the articulation both closed and open.
    """
    if request.interaction_kind not in {"container", "door"}:
        raise ValueError(f"Unsupported interaction kind: {request.interaction_kind}")
    args = _rby1_request_to_args(request)
    episode_spec, target_meta = prepare_rby1_interaction_episode(args)
    run_name = (
        f"rby1_{args.interaction_kind}_{args.action}_house_{args.house_ind}_"
        f"{sanitize_name(args.target_name)}_joint_{args.joint_index}"
    )
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target.json", target_meta)
    cfg = build_rby1_interaction_config(args)
    result = execute_rby1_whole_body_interaction(
        cfg,
        episode_spec,
        interaction_kind=args.interaction_kind,
        variant=args.variant,
        output_dir=out_dir,
        camera_names=tuple(args.camera_names),
        max_steps=args.max_steps,
        video_fps=args.video_fps,
        base_adjustment_target_pose=(
            pos_quat_to_pose_mat(target_meta["robot_pose_meta"]["adjustment_target_pose"])
            if "adjustment_target_pose" in target_meta["robot_pose_meta"]
            else None
        ),
        max_base_adjustment_steps=args.max_base_adjustment_steps,
        allow_force_fallback=args.allow_force_fallback,
        force_fallback_target_fraction=args.force_fallback_target_fraction,
        force_fallback_max_steps=args.force_fallback_max_steps,
        force_duration_seconds=args.force_duration_seconds,
    )
    result["robot_pose_mode"] = request.robot_pose_mode
    result["robot_pose_meta"] = target_meta["robot_pose_meta"]
    write_json(out_dir / "result.json", result)
    log.info("RBY1 interaction result: %s", result)
    return result


def command_run_rby1_interaction(args: argparse.Namespace) -> int:
    if args.action != "open":
        raise ValueError("This RBY1 wrapper currently supports opening doors and containers only.")
    request = RBY1InteractionRequest(
        house_ind=args.house_ind,
        interaction_kind=args.interaction_kind,
        target_name=args.target_name,
        joint_index=args.joint_index,
        robot_pose_mode=args.robot_pose_mode,
        robot_base_pose=args.robot_base_pose,
        door_arm=args.door_arm,
        container_arm=args.container_arm,
        approach_distance=args.approach_distance,
        min_base_clearance=args.min_base_clearance,
        max_approach_distance=args.max_approach_distance,
        max_base_adjustment_distance=args.max_base_adjustment_distance,
        max_base_adjustment_steps=args.max_base_adjustment_steps,
        allow_back_approach=args.allow_back_approach,
        door_tcp_offset=args.door_tcp_offset,
        door_max_steps_per_waypoint=args.door_max_steps_per_waypoint,
        door_max_planning_reattempts=args.door_max_planning_reattempts,
        door_joint_position_tolerance=args.door_joint_position_tolerance,
        door_articulation_delta_deg=args.door_articulation_delta_deg,
        allow_force_fallback=args.allow_force_fallback,
        force_fallback_target_fraction=args.force_fallback_target_fraction,
        force_fallback_max_steps=args.force_fallback_max_steps,
        force_duration_seconds=args.force_duration_seconds,
        success_threshold=args.success_threshold,
        max_steps=args.max_steps,
        container_max_steps_per_waypoint=args.container_max_steps_per_waypoint,
        container_max_batch_plan_attempts=args.container_max_batch_plan_attempts,
        container_max_planning_reattempts=args.container_max_planning_reattempts,
        container_pregrasp_tolerance=args.container_pregrasp_tolerance,
        video_fps=args.video_fps,
        camera_names=tuple(args.camera_names),
        output_dir=args.output_dir,
        scene_dataset=args.scene_dataset,
        data_split=args.data_split,
        variant=args.variant,
        seed=args.seed,
        curobo_server_url=args.curobo_server_url,
    )
    result = open_articulation_with_rby1(request)
    return 0 if result["success"] else 2


def build_scan_output_dir(args: argparse.Namespace) -> Path:
    house_tag = f"{min(args.house_inds)}_{max(args.house_inds)}" if args.house_inds else "empty"
    return args.output_dir / f"scan_{args.scene_dataset}_{args.data_split}_{house_tag}"


def benchmark_entries_by_house(path: Path, max_episodes: int) -> dict[int, list[dict[str, Any]]]:
    with open(path) as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    out: dict[int, list[dict[str, Any]]] = {}
    for episode_index, episode in enumerate(episodes[:max_episodes]):
        house_ind = int(episode["house_index"])
        task = episode.get("task", {})
        candidates = task.get("pickup_obj_candidates") or []
        target = task.get("pickup_obj_name")
        out.setdefault(house_ind, []).append(
            {
                "episode_index": episode_index,
                "pickup_obj_name": target,
                "pickup_obj_candidates": candidates,
                "task_description": episode.get("language", {}).get("task_description"),
            }
        )
    return out


def resolve_benchmark_json(path: Path) -> Path:
    if path.exists():
        return path
    rel = Path(
        "benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
        "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
    )
    candidates = [
        Path(os.environ.get("MLSPACES_ASSETS_DIR", str(REPO_ROOT / "assets"))) / rel,
        Path(
            os.environ.get(
                "MLSPACES_CACHE_DIR", str(Path.home() / ".cache/molmo-spaces-resources")
            )
        )
        / "benchmarks/molmospaces-bench-v2/20240407/procthor-10k/NavToObjDataGenConfig/"
        "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Benchmark JSON not found: {path}")


def command_scan_container_target_overlap(args: argparse.Namespace) -> int:
    benchmark_json = resolve_benchmark_json(args.benchmark_json)
    benchmark_by_house = benchmark_entries_by_house(benchmark_json, args.max_episodes)
    house_inds = args.house_inds or sorted(benchmark_by_house)
    out_dir = args.output_dir / f"container_target_overlap_first_{args.max_episodes}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "benchmark_json": str(benchmark_json),
        "max_episodes": args.max_episodes,
        "num_houses": len(house_inds),
        "num_inside_objects": 0,
        "num_exact_target_inside_objects": 0,
        "num_candidate_target_inside_objects": 0,
        "houses_with_inside_objects": 0,
        "houses_with_exact_target_inside": 0,
        "houses_with_candidate_target_inside": 0,
        "per_house": [],
    }

    for house_ind in house_inds:
        log.info("Scanning container-target overlap for house %s", house_ind)
        ctx = load_scene_context(args, house_ind)
        try:
            records, containers = collect_scene_records(ctx)
            target_records = [rec for rec in records if is_target_like(rec)]
            house_episodes = benchmark_by_house.get(house_ind, [])
            exact_targets = {
                ep["pickup_obj_name"] for ep in house_episodes if ep.get("pickup_obj_name")
            }
            candidate_targets = {
                name
                for ep in house_episodes
                for name in (ep.get("pickup_obj_candidates") or [])
            }
            house_rows = []
            for container_rec in containers:
                for object_rec in target_records:
                    rel = compute_relation(container_rec, object_rec)
                    if rel["label"] != "inside":
                        continue
                    row = {
                        "house_ind": house_ind,
                        "container_name": container_rec["name"],
                        "container_category": container_rec["category"],
                        "object_name": object_rec["name"],
                        "object_category": object_rec["category"],
                        "is_exact_benchmark_target": object_rec["name"] in exact_targets,
                        "is_candidate_benchmark_target": object_rec["name"] in candidate_targets,
                        "benchmark_episode_indices": [
                            ep["episode_index"]
                            for ep in house_episodes
                            if object_rec["name"] == ep.get("pickup_obj_name")
                            or object_rec["name"] in (ep.get("pickup_obj_candidates") or [])
                        ],
                        "relation": rel,
                    }
                    rows.append(row)
                    house_rows.append(row)

            exact_count = sum(1 for row in house_rows if row["is_exact_benchmark_target"])
            candidate_count = sum(1 for row in house_rows if row["is_candidate_benchmark_target"])
            summary["per_house"].append(
                {
                    "house_ind": house_ind,
                    "num_benchmark_episodes": len(house_episodes),
                    "exact_targets": sorted(exact_targets),
                    "candidate_targets": sorted(candidate_targets),
                    "num_inside_objects": len(house_rows),
                    "num_exact_target_inside_objects": exact_count,
                    "num_candidate_target_inside_objects": candidate_count,
                }
            )
            summary["num_inside_objects"] += len(house_rows)
            summary["num_exact_target_inside_objects"] += exact_count
            summary["num_candidate_target_inside_objects"] += candidate_count
            summary["houses_with_inside_objects"] += int(bool(house_rows))
            summary["houses_with_exact_target_inside"] += int(exact_count > 0)
            summary["houses_with_candidate_target_inside"] += int(candidate_count > 0)
            write_json(out_dir / "inside_objects.partial.json", rows)
            write_json(out_dir / "summary.partial.json", summary)
        finally:
            close_context(ctx)

    write_json(out_dir / "inside_objects.json", rows)
    write_json(out_dir / "summary.json", summary)
    return 0


def command_scan_houses(args: argparse.Namespace) -> int:
    out_dir = build_scan_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_summary = []

    for house_ind in args.house_inds:
        house_dir = out_dir / f"house_{house_ind}"
        house_dir.mkdir(parents=True, exist_ok=True)
        ctx = load_scene_context(args, house_ind)
        try:
            records, containers = collect_scene_records(ctx)
            target_records = [rec for rec in records if is_target_like(rec)]
            write_json(house_dir / "all_objects.json", records)
            write_json(house_dir / "containers.json", containers)
            write_json(house_dir / "target_objects.json", target_records)

            relations = []
            visibility_results = []
            top_candidates = []
            for container_rec in containers:
                per_container_rel = [
                    compute_relation(container_rec, object_rec) for object_rec in target_records
                ]
                per_container_rel.sort(key=lambda item: (-item["score"], item["object_name"]))
                relations.extend(per_container_rel)

                related_names = [
                    rel["object_name"]
                    for rel in per_container_rel
                    if rel["label"] in {"inside", "likely_inside"} and rel["score"] >= 1.5
                ][: args.max_visibility_objects_per_container]
                if related_names:
                    vis_rows = measure_container_visibility(
                        ctx, container_rec, related_names, output_dir=house_dir
                    )
                    visibility_results.extend(vis_rows)

                for rel in per_container_rel[: args.max_plots_per_container]:
                    if rel["label"] == "unrelated":
                        continue
                    object_rec = next(rec for rec in target_records if rec["name"] == rel["object_name"])
                    plot_name = (
                        f"relation__{container_rec['name'].replace('/', '_')}__"
                        f"{object_rec['name'].replace('/', '_')}.png"
                    )
                    save_relation_plot(house_dir / plot_name, container_rec, object_rec, rel)

            vis_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in visibility_results:
                vis_lookup.setdefault((row["container_name"], row["object_name"]), []).append(row)

            for rel in relations:
                rows = vis_lookup.get((rel["container_name"], rel["object_name"]), [])
                best_delta = max((row["delta_visibility"] for row in rows), default=0.0)
                became_visible = any(row["became_visible"] for row in rows)
                candidate = {
                    **rel,
                    "best_delta_visibility": best_delta,
                    "became_visible": became_visible,
                    "interaction_value_score": rel["score"] + 5.0 * max(best_delta, 0.0),
                }
                if rel["label"] != "unrelated":
                    top_candidates.append(candidate)
            top_candidates.sort(
                key=lambda item: (
                    -item["interaction_value_score"],
                    -item["best_delta_visibility"],
                    -item["score"],
                    item["container_name"],
                    item["object_name"],
                )
            )

            write_json(house_dir / "relations.json", relations)
            write_json(house_dir / "visibility.json", visibility_results)
            write_json(house_dir / "top_candidates.json", top_candidates[:50])

            overall_summary.append(
                {
                    "house_ind": house_ind,
                    "scene_path": str(ctx.env.current_model_path),
                    "num_objects": len(records),
                    "num_target_objects": len(target_records),
                    "num_containers": len(containers),
                    "num_relations": len(relations),
                    "num_visibility_rows": len(visibility_results),
                    "top_candidates": top_candidates[:10],
                }
            )
        finally:
            close_context(ctx)

    write_json(out_dir / "scan_report_all_houses.json", overall_summary)
    log.info("Wrote scan report to %s", out_dir)
    return 0


def command_scan_joint_dependencies(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"joint_dependencies_house_{args.house_ind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        _records, containers = collect_scene_records(ctx)
        doors = collect_door_records(ctx)
        articulation_records = containers + doors
        if args.articulation_name:
            articulation_records = [
                rec for rec in articulation_records if rec["name"] == args.articulation_name
            ]
            if not articulation_records:
                raise RuntimeError(f"Articulation not found: {args.articulation_name}")

        rows = []
        for rec in articulation_records:
            joint_dependencies = infer_joint_open_dependencies(
                ctx.env,
                rec,
                method=args.dependency_method,
                min_overlap_volume=args.min_overlap_volume,
                min_overlap_ratio=args.min_overlap_ratio,
                min_projection_ratio=args.min_projection_ratio,
                min_depth_separation=args.min_depth_separation,
            )
            plot_path = None
            if args.save_plots:
                safe_name = rec["name"].replace("/", "_")
                plot_path = out_dir / "joint_dependency_plots" / f"{safe_name}.png"
                save_joint_dependency_plot(plot_path, rec, joint_dependencies)
            rows.append(
                {
                    "house_ind": args.house_ind,
                    "name": rec["name"],
                    "category": rec.get("category"),
                    "asset_id": rec.get("asset_id"),
                    "interaction_group": rec.get("interaction_group"),
                    "num_joints": len(articulation_joint_records(rec)),
                    "plot_path": None if plot_path is None else str(plot_path),
                    "joint_dependencies": joint_dependencies,
                }
            )

        payload = {
            "house_ind": args.house_ind,
            "dependency_method": args.dependency_method,
            "min_overlap_volume": args.min_overlap_volume,
            "min_overlap_ratio": args.min_overlap_ratio,
            "min_projection_ratio": args.min_projection_ratio,
            "min_depth_separation": args.min_depth_separation,
            "articulations": rows,
        }
        out_path = out_dir / "joint_dependencies.json"
        write_json(out_path, payload)
        log.info("Wrote joint dependency report to %s", out_path)
        return 0
    finally:
        close_context(ctx)


def choose_existing_asset(ctx: LoadedContext) -> tuple[str, str, str]:
    env = ctx.env
    objects_meta = env.current_scene_metadata.get("objects", {})
    for object_name, meta in objects_meta.items():
        category = str(meta.get("category", "")).lower()
        asset_id = meta.get("asset_id")
        if not asset_id:
            continue
        if token_match(category, PORTABLE_PREFERRED_TOKENS):
            object_xml = install_uid(asset_id)
            try:
                object_ref = str(object_xml.relative_to(REPO_ROOT / "assets"))
            except ValueError:
                object_ref = str(object_xml)
            return object_ref, asset_id, category
    raise RuntimeError("Could not auto-select an existing portable asset from the scene.")


def build_added_object_pose(container_rec: dict[str, Any], current_name: str) -> list[float]:
    center = np.asarray(container_rec["aabb_center"], dtype=float).copy()
    size = np.asarray(container_rec["aabb_size"], dtype=float)
    center[2] = max(center[2], 0.05)
    if "drawer" in container_rec["name"].lower() or "drawer" in str(container_rec["category"]).lower():
        center[2] = center[2] - 0.1 * size[2]
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return np.concatenate([center, quat]).tolist()


def build_added_episode_spec(
    base_ctx: LoadedContext,
    args: argparse.Namespace,
    object_name: str,
    object_relpath: str,
    object_pose: list[float],
) -> EpisodeSpec:
    env = base_ctx.env
    robot_init_qpos = {
        group_name: np.asarray(qpos, dtype=float).tolist()
        for group_name, qpos in base_ctx.cfg.robot_config.init_qpos.items()
    }
    camera_specs = [
        {
            "name": "probe_exocentric_camera",
            "type": "exocentric",
            "pos": [0.0, -6.0, 6.0],
            "up": [0.0, 0.0, 1.0],
            "forward": [0.0, 0.7, -0.7],
            "fov": 60.0,
            "record_depth": False,
        }
    ]
    episode = {
        "source": {
            "h5_file": "container_scene_probe",
            "traj_key": "synthetic",
            "episode_length": 0,
            "camera_system_class": type(base_ctx.cfg.camera_config).__name__,
            "source_data_date": None,
            "benchmark_created_date": None,
        },
        "house_index": args.house_ind,
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "seed": args.seed,
        "robot": {
            "robot_name": base_ctx.cfg.robot_config.name,
            "init_qpos": robot_init_qpos,
        },
        "img_resolution": list(base_ctx.cfg.camera_config.img_resolution),
        "cameras": camera_specs,
        "scene_modifications": {
            "added_objects": {object_name: object_relpath},
            "object_poses": {object_name: object_pose},
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": object_name,
            "pickup_obj_candidates": [object_name],
            "succ_pos_threshold": 1.5,
        },
        "language": {
            "task_description": f"Navigate to the added object {object_name.split('/')[-1]}",
            "referral_expressions": {"object_name": object_name.split("/")[-1]},
            "referral_expressions_priority": {},
        },
    }
    return EpisodeSpec.model_validate(episode)


def build_added_object_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / f"add_object_{args.scene_dataset}_{args.data_split}_{args.house_ind}"


def command_add_object_test(args: argparse.Namespace) -> int:
    out_dir = build_added_object_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(base_ctx)
        if not containers:
            raise RuntimeError(f"No containers found in house {args.house_ind}")
        if args.container_name:
            container_rec = next(
                (rec for rec in containers if rec["name"] == args.container_name),
                None,
            )
            if container_rec is None:
                raise RuntimeError(f"Container '{args.container_name}' not found in house {args.house_ind}")
        else:
            container_rec = max(containers, key=lambda rec: len(rec["joints"]))

        if args.object_relpath:
            object_relpath = args.object_relpath
            object_uid = Path(object_relpath).stem
            object_label = object_uid
        else:
            object_relpath, object_uid, object_label = choose_existing_asset(base_ctx)

        object_body_name = args.object_name or f"custom_probe_{Path(object_relpath).stem}"
        object_name = f"custom_object/{object_body_name}"
        object_pose = build_added_object_pose(container_rec, object_name)
        episode_spec = build_added_episode_spec(
            base_ctx,
            args,
            object_name=object_name,
            object_relpath=object_relpath,
            object_pose=object_pose,
        )
        write_json(out_dir / "synthetic_episode.json", episode_spec.model_dump())
    finally:
        close_context(base_ctx)

    import molmo_spaces.tasks.task_sampler as task_sampler_module

    manager = get_resource_manager()
    manager.symlink_lock = False
    manager.cache_lock = False
    task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: {}

    cfg = build_scene_config(args)
    task_sampler = JsonEvalTaskSampler(cfg, episode_spec)
    try:
        task_sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=args.house_ind)
        original_scene_path = Path(task_sampler._current_house_scene_path(variant=args.variant))
        scene_path = prepare_writable_scene_path(original_scene_path)
        task_sampler.update_scene(scene_path=scene_path, variant=args.variant)
        env = task_sampler.env
        if env.current_scene_metadata is None:
            env._scene_metadata = get_scene_metadata(original_scene_path)
        env.camera_manager.setup_cameras(env, cfg.camera_config)
        om = env.object_managers[env.current_batch_index]
        added_obj = om.get_object_by_name(object_name)
        center, size = safe_body_aabb(env.current_model, env.current_data, added_obj.body_id)
        container_obj = om.get_object_by_name(container_rec["name"])
        if isinstance(container_obj, MlSpacesArticulationObject) and container_rec["joints"]:
            first_joint = container_rec["joints"][0]
            container_obj.set_joint_position(first_joint["joint_index"], first_joint["open_value"])
            mujoco.mj_forward(env.current_model, env.current_data)
            env.camera_manager.setup_cameras(env, cfg.camera_config)
        result = {
            "house_ind": args.house_ind,
            "scene_path": str(env.current_model_path),
            "container_name": container_rec["name"],
            "container_category": container_rec["category"],
            "added_object_name": object_name,
            "asset_relpath": object_relpath,
            "asset_uid": object_uid,
            "asset_label": object_label,
            "requested_pose": object_pose,
            "readback_aabb_center": center.tolist(),
            "readback_aabb_size": size.tolist(),
            "success": True,
        }
        write_json(out_dir / "add_object_result.json", result)
        added_rec = {
            "name": object_name,
            "aabb_center": center.tolist(),
            "aabb_size": size.tolist(),
        }
        save_relation_plot(
            out_dir / "add_object_debug.png",
            container_rec,
            added_rec,
            {"label": "added_object", "score": 1.0},
        )
    finally:
        task_sampler.close()

    log.info("Added object test output written to %s", out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe container geometry and visibility in MolmoSpaces scenes.")
    parser.add_argument("--scene_dataset", default="procthor-10k")
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan-houses")
    scan_parser.add_argument("--house_inds", nargs="+", type=int, required=True)
    scan_parser.add_argument("--max_plots_per_container", type=int, default=4)
    scan_parser.add_argument("--max_visibility_objects_per_container", type=int, default=8)
    scan_parser.set_defaults(func=command_scan_houses)

    overlap_parser = subparsers.add_parser("scan-container-target-overlap")
    overlap_parser.add_argument("--benchmark_json", type=Path, default=DEFAULT_NAV_BENCHMARK_JSON)
    overlap_parser.add_argument("--max_episodes", type=int, default=100)
    overlap_parser.add_argument("--house_inds", nargs="+", type=int)
    overlap_parser.set_defaults(func=command_scan_container_target_overlap)

    dep_parser = subparsers.add_parser("scan-joint-dependencies")
    dep_parser.add_argument("--house_ind", type=int, required=True)
    dep_parser.add_argument("--articulation_name", type=str)
    dep_parser.add_argument(
        "--dependency_method",
        choices=["front_occlusion", "open_aabb_overlap"],
        default="front_occlusion",
    )
    dep_parser.add_argument("--min_overlap_volume", type=float, default=1e-6)
    dep_parser.add_argument("--min_overlap_ratio", type=float, default=0.0)
    dep_parser.add_argument("--min_projection_ratio", type=float, default=0.15)
    dep_parser.add_argument("--min_depth_separation", type=float, default=0.01)
    dep_parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    dep_parser.set_defaults(func=command_scan_joint_dependencies)

    add_parser = subparsers.add_parser("add-object-test")
    add_parser.add_argument("--house_ind", type=int, required=True)
    add_parser.add_argument("--container_name", type=str)
    add_parser.add_argument("--object_relpath", type=str)
    add_parser.add_argument("--object_name", type=str)
    add_parser.set_defaults(func=command_add_object_test)

    debug_parser = subparsers.add_parser("debug-container-view")
    debug_parser.add_argument("--house_ind", type=int, required=True)
    debug_parser.add_argument("--container_name", type=str, required=True)
    debug_parser.add_argument("--joint_index", type=int, required=True)
    debug_parser.add_argument("--joint_value", type=float, required=True)
    debug_parser.add_argument("--head_tilt_delta", type=float, default=0.35)
    debug_parser.set_defaults(func=command_debug_container_view)

    drawer_bound_parser = subparsers.add_parser("debug-drawer-bound-object")
    drawer_bound_parser.add_argument("--house_ind", type=int, required=True)
    drawer_bound_parser.add_argument("--container_name", type=str, required=True)
    drawer_bound_parser.add_argument("--joint_index", type=int, required=True)
    drawer_bound_parser.add_argument("--object_name", type=str)
    drawer_bound_parser.add_argument("--box_padding", type=float, default=0.05)
    drawer_bound_parser.add_argument("--view_distance", type=float, default=0.45)
    drawer_bound_parser.add_argument("--head_tilt_delta", type=float, default=0.25)
    drawer_bound_parser.add_argument("--torso_lean_delta", type=float, default=0.45)
    drawer_bound_parser.set_defaults(func=command_debug_drawer_bound_object)

    debug_door_parser = subparsers.add_parser("debug-door-view")
    debug_door_parser.add_argument("--house_ind", type=int, required=True)
    debug_door_parser.add_argument("--door_name", type=str)
    debug_door_parser.set_defaults(func=command_debug_door_view)

    rby1_parser = subparsers.add_parser(
        "run-rby1-interaction",
        help="Execute an official RBY1 whole-body door/container manipulation policy.",
    )
    rby1_parser.add_argument("--house_ind", type=int, required=True)
    rby1_parser.add_argument(
        "--interaction_kind",
        choices=["container", "door"],
        required=True,
    )
    rby1_parser.add_argument("--target_name", type=str, required=True)
    rby1_parser.add_argument("--joint_index", type=int, default=0)
    rby1_parser.add_argument("--action", choices=["open"], default="open")
    rby1_parser.add_argument(
        "--door_arm",
        choices=["auto", "left", "right"],
        default="auto",
        help="Use the official hinge-side heuristic or force the left/right arm.",
    )
    rby1_parser.add_argument(
        "--container_arm",
        choices=["auto", "left", "right"],
        default="auto",
        help="Use nearest-arm selection or force the left/right arm for a container.",
    )
    rby1_parser.add_argument("--approach_distance", type=float, default=0.5)
    rby1_parser.add_argument("--min_base_clearance", type=float, default=0.15)
    rby1_parser.add_argument("--max_approach_distance", type=float, default=1.2)
    rby1_parser.add_argument("--max_base_adjustment_distance", type=float, default=0.75)
    rby1_parser.add_argument("--max_base_adjustment_steps", type=int, default=120)
    rby1_parser.add_argument(
        "--allow_back_approach",
        action="store_true",
        help="Allow a container interaction pose on the inferred rear side.",
    )
    rby1_parser.add_argument(
        "--door_tcp_offset",
        type=float,
        default=0.03,
        help="Initial door-handle TCP depth; failed grasps try nearby offsets.",
    )
    rby1_parser.add_argument(
        "--robot_base_pose",
        nargs=7,
        type=float,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Use the algorithm's current robot pose instead of automatic placement.",
    )
    rby1_parser.add_argument(
        "--robot_pose_mode",
        choices=["auto", "current", "current_or_adjust"],
        default="auto",
        help=(
            "Use automatic placement, require the supplied navigation endpoint, "
            "or preserve it when safe and otherwise fall back to a nearby stance."
        ),
    )
    rby1_parser.add_argument("--success_threshold", type=float, default=0.8)
    rby1_parser.add_argument("--force_duration_seconds", type=float, default=2.0)
    rby1_parser.add_argument("--max_steps", type=int, default=400)
    rby1_parser.add_argument("--door_max_steps_per_waypoint", type=int, default=35)
    rby1_parser.add_argument("--door_max_planning_reattempts", type=int, default=5)
    rby1_parser.add_argument(
        "--door_joint_position_tolerance",
        type=float,
        default=0.10,
        help="Per-run RBY1 door-policy joint waypoint tolerance.",
    )
    rby1_parser.add_argument(
        "--door_articulation_delta_deg",
        type=float,
        default=7.0,
        help="Per-run door articulation increment in degrees.",
    )
    rby1_parser.add_argument(
        "--allow_force_fallback",
        action="store_true",
        help="After a contact phase, optionally finish the joint with explicit force-drive; recorded separately.",
    )
    rby1_parser.add_argument("--force_fallback_target_fraction", type=float, default=1.0)
    rby1_parser.add_argument("--force_fallback_max_steps", type=int, default=1000)
    rby1_parser.add_argument("--container_max_steps_per_waypoint", type=int, default=30)
    rby1_parser.add_argument("--container_max_batch_plan_attempts", type=int, default=8)
    rby1_parser.add_argument("--container_max_planning_reattempts", type=int, default=8)
    rby1_parser.add_argument(
        "--container_pregrasp_tolerance",
        type=float,
        default=0.04,
        help="Optional per-run pre-grasp joint tolerance; keep 0.04 for the official behavior.",
    )
    rby1_parser.add_argument("--video_fps", type=float)
    rby1_parser.add_argument(
        "--curobo_server_url",
        type=str,
        help="Optional remote Curobo gRPC address. Omit to use the local GPU planner.",
    )
    rby1_parser.add_argument(
        "--camera_names",
        nargs="+",
        default=list(DEFAULT_RBY1_INTERACTION_CAMERAS),
    )
    rby1_parser.set_defaults(func=command_run_rby1_interaction)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
