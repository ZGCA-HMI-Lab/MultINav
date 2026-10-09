"""Standalone, reproducible evaluator for the frozen InteractiveNav V3 benchmark.

This implementation intentionally does not modify ``molmo_spaces/evaluation``.
The upstream evaluator remains useful for ordinary NavToObj policies, while this
runner owns V3's interaction action protocol, force execution, terminal
semantics, asset-drift handling, and reporting.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import re
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping

# Must be set before importing modules that can allocate a renderer.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

from multinav.sim import container_scene_probe as probe
from multinav.sim.force_interaction_runtime import ground_drawer_open_regions
from multinav.sim import interactive_nav_v3

from .benchmark_metrics import (
    DEFAULT_PAPER_COST_ERROR_SURCHARGE,
    DEFAULT_PAPER_COST_FAILURE_PENALTY,
    DEFAULT_PAPER_COST_INTERACTION_ATTEMPT,
    PAPER_METRIC_SCHEMA_VERSION,
    PaperMetricConfig,
    SUCCESS_OPEN_FRACTION,
    joint_open_fraction,
    paper_episode_total_cost,
    paper_interaction_attempt_score,
    path_length_bin,
    reference_path_length_m,
    score_interactions,
    spl,
    summarise_results,
    target_candidate_names,
    target_metric_rows,
    target_metrics,
    oracle_terminal_goal_consistency,
)
from .benchmark_interaction_executor import execute_open_articulation_group
from .benchmark_interaction_adapter import (
    validate_public_interaction_observation,
    validate_public_interaction_pose,
)
from .benchmark_io import load_benchmark_episodes
from .benchmark_policies import (
    BenchmarkPolicy,
    NoOpPolicy,
    ScriptedOraclePolicy,
    build_factory_policy,
    build_ros_bridge_policy,
)
from .benchmark_types import EpisodeResult, InteractionAttempt, PolicyAction, PolicyObservation, PublicEpisode
from .goal_status import (
    GoalClaimVerification,
    PublicGoalEvidenceLedger,
    RosGoalStatusObserver,
    is_exploration_terminal,
    is_target_goal_success_claim,
    verify_target_goal_claim,
)
from .public_goal import build_public_target_context as build_language_target_context
from .public_ids import opaque_door_instance_id
from .restricted_gt_perception import (
    RestrictedGTPerceptionPublisher,
    build_private_object_specs_from_env,
)
from .ros_object_goal_adapter import (
    RosObjectGoalEvaluatorAdapter,
    build_public_target_context as build_ros_target_context,
)
from .ros_navigation_stall import (
    CROSS_SUBGOAL_STALL_REASON,
    CrossSubgoalStallConfig,
    CrossSubgoalStallTracker,
    RosBehaviorFeedbackObserver,
)
from .ros_policy_termination import (
    RosPolicyTerminationConfig,
    RosPolicyTerminationGuard,
    counts_toward_applied_action_budget,
)
from .trusted_interaction_skill import (
    JointOpenResult,
    OpaqueObjectRegistry,
    OpenPostconditionSpec,
    TrustedInteractionSkill,
)

PROTOCOL_VERSION = "interactive_nav_v3_benchmark_eval_v15"
PUBLIC_OBSERVATION_SCHEMA_VERSION = "interactive_nav_public_observation_v2"
INITIAL_STATE_REPLAY_VERSION = "interactive_nav_v3_initial_state_replay_v2"
INTERACTION_EXECUTION_MODE = "canonical_locked_force"
STEP_BUDGET_FORMULA_VERSION = "interactive_nav_v3_step_budget_v1"
DYNAMIC_MAX_STEPS_CAP = 2000


@dataclass(frozen=True)
class SimulatorProtocolProfile:
    """Physics/control semantics that must not drift between code branches."""

    policy_dt_ms: float
    ctrl_dt_ms: float
    sim_dt_ms: float
    holo_base_yaw_control_mode: Literal[
        "nearest_equivalent", "legacy_branch_reset"
    ]


SIMULATOR_PROTOCOL_PROFILES: dict[str, SimulatorProtocolProfile] = {
    # This is the timing/yaw behavior used by the complete InteractiveNav
    # experiments and is therefore the canonical frozen benchmark protocol.
    "interactive_nav_v3": SimulatorProtocolProfile(
        policy_dt_ms=200.0,
        ctrl_dt_ms=10.0,
        sim_dt_ms=10.0,
        holo_base_yaw_control_mode="legacy_branch_reset",
    ),
    # Retain an explicit upstream-main profile for controlled migration tests.
    "upstream_main": SimulatorProtocolProfile(
        policy_dt_ms=200.0,
        ctrl_dt_ms=2.0,
        sim_dt_ms=2.0,
        holo_base_yaw_control_mode="nearest_equivalent",
    ),
}
DEFAULT_SIMULATOR_PROFILE = "interactive_nav_v3"


def resolve_simulator_protocol(
    profile_name: str,
    *,
    policy_dt_ms: float | None = None,
    ctrl_dt_ms: float | None = None,
    sim_dt_ms: float | None = None,
    holo_base_yaw_control_mode: str | None = None,
) -> dict[str, Any]:
    """Resolve a named simulator profile or validate a fully explicit custom one."""

    profile = SIMULATOR_PROTOCOL_PROFILES.get(profile_name)
    provided = {
        "policy_dt_ms": policy_dt_ms,
        "ctrl_dt_ms": ctrl_dt_ms,
        "sim_dt_ms": sim_dt_ms,
        "holo_base_yaw_control_mode": holo_base_yaw_control_mode,
    }
    if profile is not None:
        defaults = asdict(profile)
        return {
            name: defaults[name] if value is None else value
            for name, value in provided.items()
        }
    if profile_name != "custom":
        raise ValueError(f"Unsupported simulator profile: {profile_name!r}")
    missing = [name for name, value in provided.items() if value is None]
    if missing:
        raise ValueError(
            "custom simulator profile requires explicit values for: "
            + ", ".join(missing)
        )
    return provided

# Established ROS exploration posture.  The asymmetric shoulder roll tucks the
# arms beside the torso so they neither move after the first navigation action
# nor enter the head-camera depth map.
ROS_NAVIGATION_ARM_QPOS: dict[str, tuple[float, ...]] = {
    "left_arm": (0.28, 0.0, -0.45, -0.64, 0.39, -0.26, -0.04),
    "right_arm": (0.28, 0.0, 0.45, -0.64, 0.39, -0.26, -0.04),
}


@dataclass
class BenchmarkEvaluationConfig:
    benchmark: Path
    output_dir: Path
    policy: Literal["noop", "scripted_oracle", "ros_bridge", "ros_object_goal_rule", "factory"] = "noop"
    policy_factory: str | None = None
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    workers: int = 1
    max_steps: int = 500
    step_budget_mode: Literal["fixed", "dynamic"] = "fixed"
    min_steps: int = 300
    dynamic_path_free_m: float = 3.0
    dynamic_steps_per_path_m: float = 25.0
    dynamic_channel_interaction_steps: int = 150
    dynamic_container_interaction_steps: int = 200
    dynamic_container_joint_steps: int = 40
    dynamic_step_quantum: int = 50
    episode_indices: list[int] | None = None
    max_episodes: int | None = None
    resume: bool = False
    record_video: bool = False
    video_fps: float = 5.0
    camera_names: list[str] = field(default_factory=lambda: ["head_camera"])
    image_resolution: tuple[int, int] | None = (640, 480)
    simulator_profile: Literal[
        "interactive_nav_v3", "upstream_main", "custom"
    ] = DEFAULT_SIMULATOR_PROFILE
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    holo_base_yaw_control_mode: Literal[
        "nearest_equivalent", "legacy_branch_reset"
    ] = "legacy_branch_reset"
    force_duration_seconds: float = 2.0
    force_collection_hz: float = 5.0
    force_target_fraction: float = 1.0
    force_max_internal_steps: int = 1500
    interaction_max_distance_m: float = 1.75
    require_interaction_visible: bool = True
    require_runtime_goal_consistency: bool = True
    allow_internal_object_names: bool = False
    oracle_navigation_mode: Literal["direct_pose", "task_action"] = "direct_pose"
    ros_observation_topic: str = "/molmo_spaces/head_camera/image"
    ros_action_topic: str = "/molmo_spaces/action"
    ros_action_timeout_s: float = 5.0
    ros_cmd_vel_linear_gain: float = 3.0
    ros_require_move_base_active: bool = True
    ros_map_warmup_skip_frames: int = 0
    ros_step_ready_topic: str = "/semantic_decision/step_ready"
    ros_step_ready_barrier_enabled: bool = False
    ros_step_ready_warmup_skip_frames: int = 0
    ros_step_ready_timeout_s: float = 2.0
    ros_step_ready_bootstrap_timeout_s: float = 10.0
    # Optional evaluator-owned RGB persistence for exact offline six-panel
    # reconstruction.  This remains outside the public policy contract: the
    # bridge writes only the already-published head-camera frame plus a step
    # manifest, while the recorder retains raw map/graph JSON.
    ros_step_frame_dir: str = ""
    ros_step_frame_queue_size: int = 4
    ros_step_capture_ack_topic: str = "/molmo_spaces/step_capture_ack"
    ros_step_capture_ack_barrier_enabled: bool = False
    ros_step_capture_ack_timeout_s: float = 2.0
    ros_target_topic: str = "/semantic_decision/target"
    ros_goal_status_topic: str = "/semantic_decision/goal_status"
    ros_restricted_gt_topic: str = "/semantic_mapping/gt_observations"
    ros_interaction_command_topic: str = "/semantic_decision/interaction_command"
    ros_interaction_result_topic: str = "/semantic_mapping/interaction_result"
    ros_behavior_feedback_topic: str = "/semantic_decision/behavior_feedback"
    ros_stall_early_stop_enabled: bool = True
    ros_stall_min_failed_subgoals: int = 8
    ros_stall_max_displacement_m: float = 0.15
    ros_stall_min_no_progress_steps: int = 20
    ros_command_starvation_timeout_s: float = 60.0
    ros_observation_turn_multiplier: float = 4.0
    ros_final_goal_status_drain_timeout_s: float = 5.0
    restricted_gt_min_visible_pixels: int = 16
    restricted_gt_min_bbox_area_pixels: int = 512
    restricted_gt_min_bbox_short_side_pixels: int = 1
    restricted_gt_min_portal_bbox_short_side_pixels: int = 8
    restricted_gt_min_visible_fraction: float = 0.2
    restricted_gt_max_distance_m: float = 4.0
    quality_gate_only: bool = False
    allow_runtime_ineligible: bool = False
    runtime_joint_position_tolerance: float = 0.02
    runtime_joint_fraction_tolerance: float = 0.05
    runtime_base_position_tolerance_m: float = 0.05
    runtime_base_yaw_tolerance_rad: float = 0.05
    paper_cost_interaction_attempt: float = DEFAULT_PAPER_COST_INTERACTION_ATTEMPT
    paper_cost_error_surcharge: float = DEFAULT_PAPER_COST_ERROR_SURCHARGE
    paper_cost_failure_penalty: float = DEFAULT_PAPER_COST_FAILURE_PENALTY
    progress_every: int = 10

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        _paper_metric_config(self).validate()
        if self.step_budget_mode not in {"fixed", "dynamic"}:
            raise ValueError("step_budget_mode must be fixed or dynamic")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self.step_budget_mode == "dynamic":
            if self.min_steps > self.max_steps:
                raise ValueError("min_steps must be <= max_steps in dynamic mode")
            if self.max_steps > DYNAMIC_MAX_STEPS_CAP:
                raise ValueError(
                    f"dynamic max_steps must be <= {DYNAMIC_MAX_STEPS_CAP}"
                )
        if not math.isfinite(float(self.dynamic_path_free_m)) or self.dynamic_path_free_m < 0:
            raise ValueError("dynamic_path_free_m must be finite and non-negative")
        for name, value in (
            ("dynamic_steps_per_path_m", self.dynamic_steps_per_path_m),
            ("dynamic_channel_interaction_steps", self.dynamic_channel_interaction_steps),
            ("dynamic_container_interaction_steps", self.dynamic_container_interaction_steps),
            ("dynamic_container_joint_steps", self.dynamic_container_joint_steps),
        ):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.dynamic_step_quantum < 1:
            raise ValueError("dynamic_step_quantum must be >= 1")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("max_episodes must be >= 1")
        if self.episode_indices is not None:
            if not self.episode_indices:
                raise ValueError("episode_indices must not be empty")
            normalized_indices = [int(index) for index in self.episode_indices]
            if len(normalized_indices) != len(set(normalized_indices)):
                raise ValueError("episode_indices must not contain duplicates")
        if self.policy == "factory" and not self.policy_factory:
            raise ValueError("--policy-factory is required with --policy factory")
        if self.policy != "factory" and self.policy_factory:
            raise ValueError("--policy-factory is only valid with --policy factory")
        if self.policy in {"ros_bridge", "ros_object_goal_rule"} and self.workers != 1:
            raise ValueError(f"{self.policy} requires --workers 1 because a ROS master is stateful")
        if self.policy in {"ros_bridge", "ros_object_goal_rule"} and "head_camera" not in self.camera_names:
            raise ValueError(f"{self.policy} requires head_camera for RGB/depth and pose publication")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        if self.simulator_profile != "custom":
            expected_profile = SIMULATOR_PROTOCOL_PROFILES.get(self.simulator_profile)
            if expected_profile is None:
                raise ValueError(
                    f"Unsupported simulator_profile: {self.simulator_profile!r}"
                )
            actual = (
                float(self.policy_dt_ms),
                float(self.ctrl_dt_ms),
                float(self.sim_dt_ms),
                self.holo_base_yaw_control_mode,
            )
            expected = (
                expected_profile.policy_dt_ms,
                expected_profile.ctrl_dt_ms,
                expected_profile.sim_dt_ms,
                expected_profile.holo_base_yaw_control_mode,
            )
            if actual != expected:
                raise ValueError(
                    f"simulator_profile {self.simulator_profile!r} requires "
                    f"policy/ctrl/sim/yaw={expected!r}, got {actual!r}; use "
                    "simulator_profile='custom' for an intentional variant"
                )
        if self.image_resolution is not None and min(self.image_resolution) <= 0:
            raise ValueError("image_resolution must be positive")
        for name, value in (
            ("policy_dt_ms", self.policy_dt_ms),
            ("ctrl_dt_ms", self.ctrl_dt_ms),
            ("sim_dt_ms", self.sim_dt_ms),
            ("force_duration_seconds", self.force_duration_seconds),
            ("force_collection_hz", self.force_collection_hz),
            ("video_fps", self.video_fps),
            ("interaction_max_distance_m", self.interaction_max_distance_m),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.force_target_fraction <= 1.0:
            raise ValueError("force_target_fraction must be in (0, 1]")
        if self.force_max_internal_steps < 1:
            raise ValueError("force_max_internal_steps must be >= 1")
        if self.restricted_gt_min_visible_pixels < 1:
            raise ValueError("restricted_gt_min_visible_pixels must be >= 1")
        if self.restricted_gt_min_bbox_area_pixels < 1:
            raise ValueError("restricted_gt_min_bbox_area_pixels must be >= 1")
        if self.restricted_gt_min_bbox_short_side_pixels < 1:
            raise ValueError("restricted_gt_min_bbox_short_side_pixels must be >= 1")
        if (
            self.restricted_gt_min_portal_bbox_short_side_pixels
            < self.restricted_gt_min_bbox_short_side_pixels
        ):
            raise ValueError(
                "restricted_gt_min_portal_bbox_short_side_pixels must be >= "
                "restricted_gt_min_bbox_short_side_pixels"
            )
        if (
            not math.isfinite(float(self.restricted_gt_min_visible_fraction))
            or not 0.0 <= float(self.restricted_gt_min_visible_fraction) <= 1.0
        ):
            raise ValueError("restricted_gt_min_visible_fraction must be in [0, 1]")
        if self.ros_step_frame_queue_size < 1:
            raise ValueError("ros_step_frame_queue_size must be >= 1")
        if self.ros_step_ready_warmup_skip_frames < 0:
            raise ValueError("ros_step_ready_warmup_skip_frames must be non-negative")
        if self.ros_step_ready_timeout_s < 0.0:
            raise ValueError("ros_step_ready_timeout_s must be non-negative")
        if self.ros_step_ready_bootstrap_timeout_s < 0.0:
            raise ValueError("ros_step_ready_bootstrap_timeout_s must be non-negative")
        if self.ros_step_capture_ack_timeout_s < 0.0:
            raise ValueError("ros_step_capture_ack_timeout_s must be non-negative")
        if (
            not math.isfinite(float(self.restricted_gt_max_distance_m))
            or self.restricted_gt_max_distance_m < 0.0
        ):
            raise ValueError("restricted_gt_max_distance_m must be finite and non-negative")
        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        if self.ros_stall_min_failed_subgoals < 2:
            raise ValueError("ros_stall_min_failed_subgoals must be >= 2")
        if (
            not math.isfinite(float(self.ros_stall_max_displacement_m))
            or self.ros_stall_max_displacement_m <= 0.0
        ):
            raise ValueError("ros_stall_max_displacement_m must be finite and positive")
        if self.ros_stall_min_no_progress_steps < 0:
            raise ValueError("ros_stall_min_no_progress_steps must be >= 0")
        RosPolicyTerminationConfig(
            command_starvation_timeout_s=self.ros_command_starvation_timeout_s,
            observation_turn_multiplier=self.ros_observation_turn_multiplier,
        ).validate()
        if (
            not math.isfinite(float(self.ros_final_goal_status_drain_timeout_s))
            or self.ros_final_goal_status_drain_timeout_s < 0.0
        ):
            raise ValueError(
                "ros_final_goal_status_drain_timeout_s must be finite and non-negative"
            )
        for name, value in (
            ("runtime_joint_position_tolerance", self.runtime_joint_position_tolerance),
            ("runtime_joint_fraction_tolerance", self.runtime_joint_fraction_tolerance),
            ("runtime_base_position_tolerance_m", self.runtime_base_position_tolerance_m),
            ("runtime_base_yaw_tolerance_rad", self.runtime_base_yaw_tolerance_rad),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isclose(self.policy_dt_ms / self.ctrl_dt_ms, round(self.policy_dt_ms / self.ctrl_dt_ms)):
            raise ValueError("policy_dt_ms must be an integer multiple of ctrl_dt_ms")
        if not math.isclose(self.ctrl_dt_ms / self.sim_dt_ms, round(self.ctrl_dt_ms / self.sim_dt_ms)):
            raise ValueError("ctrl_dt_ms must be an integer multiple of sim_dt_ms")


def _paper_metric_config(config: BenchmarkEvaluationConfig) -> PaperMetricConfig:
    """Project evaluator CLI values into the frozen paper metric protocol."""

    return PaperMetricConfig(
        interaction_attempt_cost=float(config.paper_cost_interaction_attempt),
        error_interaction_surcharge=float(config.paper_cost_error_surcharge),
        failure_penalty=float(config.paper_cost_failure_penalty),
    )


class V3BenchmarkTaskSampler(JsonEvalTaskSampler):
    """Json sampler with strict critical-name checks and safe asset drift repair."""

    runtime_compatibility: dict[str, Any]

    def __init__(self, exp_config: Any, episode_spec: EpisodeSpec, interactive_nav: dict[str, Any]) -> None:
        self._interactive_nav = interactive_nav
        self.runtime_compatibility = {}
        super().__init__(exp_config, episode_spec)

    def set_joint_values(self, env: Any) -> None:
        # JsonEvalTaskSampler treats every articulated ``pickup_obj_name`` as a
        # manipulation target and requires a grasp file.  NavToObj reuses that
        # field only to identify the navigation target; its authoritative joint
        # state is restored later from ``scene_modifications``.
        if self.episode_spec.task.get("task_type") == "nav_to_obj":
            return
        super().set_joint_values(env)

    def randomize_scene(self, env: Any, robot_view: Any) -> None:
        model = env.current_model
        bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(model.nbody)
        }
        joints = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        }
        bodies.discard(None)
        joints.discard(None)
        nav = self._interactive_nav
        critical_bodies = set(target_candidate_names(self.episode_spec.model_dump()))
        critical_bodies.update(str(row["object_name"]) for row in nav.get("interactions", []))
        missing_bodies = sorted(critical_bodies - bodies)
        if missing_bodies:
            raise RuntimeError("Runtime scene is missing task-critical bodies: " + ", ".join(missing_bodies))
        articulation_names = {
            str(row.joint_name) for row in self.episode_spec.scene_modifications.articulation_states
        }
        interaction_joints = {str(row["joint_name"]) for row in nav.get("interactions", [])}
        missing_joints = sorted((articulation_names | interaction_joints) - joints)
        if missing_joints:
            raise RuntimeError("Runtime scene is missing recorded articulation joints: " + ", ".join(missing_joints))
        poses = dict(self.episode_spec.scene_modifications.object_poses)
        missing_poses = sorted(name for name in poses if name not in bodies)
        if missing_poses:
            self.episode_spec.scene_modifications.object_poses = {
                name: pose for name, pose in poses.items() if name in bodies
            }
        self.runtime_compatibility = {
            "runtime_body_count": len(bodies),
            "runtime_joint_count": len(joints),
            "dropped_noncritical_object_pose_count": len(missing_poses),
            "dropped_noncritical_object_pose_names": missing_poses,
        }
        super().randomize_scene(env, robot_view)


@dataclass(frozen=True)
class RuntimeJoint:
    object_name: str
    object_category: str | None
    domain: Literal["channel", "container"]
    joint_name: str
    joint_index: int
    body_id: int
    aabb_center: np.ndarray
    aabb_size: np.ndarray


class InteractionCatalog:
    """Live articulated-object catalog used only inside the evaluator."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._joints: list[RuntimeJoint] = []
        ctx = SimpleNamespace(env=env)
        for record in probe.collect_door_records(ctx):
            self._joints.append(
                RuntimeJoint(
                    object_name=str(record["name"]),
                    object_category=str(record.get("category") or "Door"),
                    domain="channel",
                    joint_name=str(record["hinge_joint_name"]),
                    joint_index=int(record["hinge_joint_index"]),
                    body_id=int(record["body_id"]),
                    aabb_center=np.asarray(record["aabb_center"], dtype=float),
                    aabb_size=np.asarray(record["aabb_size"], dtype=float),
                )
            )
        _records, containers = probe.collect_scene_records(ctx)
        for record in containers:
            for joint in record.get("joints", []):
                self._joints.append(
                    RuntimeJoint(
                        object_name=str(record["name"]),
                        object_category=None if record.get("category") is None else str(record["category"]),
                        domain="container",
                        joint_name=str(joint["joint_name"]),
                        joint_index=int(joint["joint_index"]),
                        body_id=int(record["body_id"]),
                        aabb_center=np.asarray(record["aabb_center"], dtype=float),
                        aabb_size=np.asarray(record["aabb_size"], dtype=float),
                    )
                )

    @property
    def joints(self) -> list[RuntimeJoint]:
        return list(self._joints)

    def by_name(self, object_name: str, joint_index: int | None) -> RuntimeJoint | None:
        matches = [item for item in self._joints if item.object_name == object_name]
        if joint_index is not None:
            matches = [item for item in matches if item.joint_index == int(joint_index)]
        return matches[0] if len(matches) == 1 else None

    def by_pixel(
        self,
        *,
        camera_name: str,
        pixel_xy: tuple[int, int] | None,
        normalized_pixel_xy: tuple[float, float] | None,
        joint_index: int | None,
    ) -> tuple[RuntimeJoint | None, dict[str, Any]]:
        if camera_name not in self.env.camera_manager.registry:
            return None, {"reason": "unknown_camera"}
        try:
            segmentation = np.asarray(self.env.render_segmentation_frame(camera_name))
        except Exception as exc:
            return None, {"reason": "segmentation_render_failed", "error": f"{type(exc).__name__}: {exc}"}
        if segmentation.ndim < 3 or segmentation.shape[-1] < 2:
            return None, {"reason": "invalid_segmentation_shape", "shape": list(segmentation.shape)}
        height, width = segmentation.shape[:2]
        if normalized_pixel_xy is not None:
            u = int(round(float(normalized_pixel_xy[0]) * (width - 1)))
            v = int(round(float(normalized_pixel_xy[1]) * (height - 1)))
        elif pixel_xy is not None:
            u, v = int(pixel_xy[0]), int(pixel_xy[1])
        else:
            return None, {"reason": "missing_pixel_selector"}
        if not (0 <= u < width and 0 <= v < height):
            return None, {"reason": "pixel_out_of_bounds", "pixel_xy": [u, v], "image_size": [width, height]}
        geom_id, object_type = [int(value) for value in segmentation[v, u, :2]]
        if object_type != int(mujoco.mjtObj.mjOBJ_GEOM) or not (0 <= geom_id < self.env.current_model.ngeom):
            return None, {"reason": "pixel_not_on_geometry", "pixel_xy": [u, v], "geom_id": geom_id, "object_type": object_type}
        body_id = int(self.env.current_model.geom_bodyid[geom_id])
        ancestors: list[int] = []
        current = body_id
        while current >= 0:
            ancestors.append(current)
            if current == 0:
                break
            parent = int(self.env.current_model.body_parentid[current])
            if parent == current:
                break
            current = parent
        candidates = [item for item in self._joints if item.body_id in ancestors]
        if joint_index is not None:
            candidates = [item for item in candidates if item.joint_index == int(joint_index)]
        # The closest ancestor is the most specific visible articulated body.
        candidates.sort(key=lambda item: ancestors.index(item.body_id))
        if not candidates:
            return None, {"reason": "pixel_has_no_articulated_ancestor", "pixel_xy": [u, v], "body_id": body_id}
        if len(candidates) > 1 and joint_index is None:
            return None, {
                "reason": "ambiguous_joint_selector",
                "pixel_xy": [u, v],
                "candidate_joints": [item.joint_index for item in candidates],
            }
        return candidates[0], {"pixel_xy": [u, v], "body_id": body_id, "segmentation_image_size": [width, height]}


@dataclass
class RestrictedRosObjectGoalRuntime:
    """Evaluator-private state for one restricted-GT ROS episode."""

    perception: RestrictedGTPerceptionPublisher
    adapter: RosObjectGoalEvaluatorAdapter
    skill: TrustedInteractionSkill
    opaque_to_source_name: dict[str, str]
    opaque_to_joints: dict[str, tuple[RuntimeJoint, ...]]
    target_source_by_opaque_id: dict[str, str]
    goal_evidence: PublicGoalEvidenceLedger
    published_frame_sink: Callable[[Mapping[str, Any]], bool] | None = None


def _body_root_id(model: Any, body_id: int) -> int | None:
    """Return the MuJoCo body-tree root for one evaluator-private body."""

    try:
        current = int(body_id)
        parent_ids = getattr(model, "body_parentid")
        body_count = len(parent_ids)
    except (TypeError, ValueError, AttributeError):
        return None
    if not 0 <= current < body_count:
        return None
    root_ids = getattr(model, "body_rootid", None)
    if root_ids is not None:
        try:
            root_id = int(root_ids[current])
        except (TypeError, ValueError, IndexError):
            root_id = -1
        if 0 <= root_id < body_count:
            return root_id
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        try:
            parent = int(parent_ids[current])
        except (TypeError, ValueError, IndexError):
            return None
        if parent == current or parent < 0 or parent >= body_count:
            return current
        if parent == 0:
            return current
        current = parent
    return None


def _perception_source_skill_aliases(
    *,
    model: Any,
    private_specs: list[Any],
    joints_by_object: dict[str, list[RuntimeJoint]],
) -> dict[str, str]:
    """Map rendered body sources to their sealed object-skill source.

    A scene metadata record commonly names the static root of a doorway,
    whereas :class:`InteractionCatalog` names the articulated leaf body that
    owns its hinge.  Restricted GT can render either body.  Both therefore
    need episode-local opaque IDs that resolve to the same private skill,
    without ever publishing the raw body names to ROS.
    """

    skill_source_by_root: dict[int, str] = {}
    ambiguous_roots: set[int] = set()
    for source_name in sorted(joints_by_object):
        joints = joints_by_object[source_name]
        if not joints:
            continue
        root_id = _body_root_id(model, int(joints[0].body_id))
        if root_id is None:
            continue
        existing = skill_source_by_root.get(root_id)
        if existing is None:
            skill_source_by_root[root_id] = source_name
        elif existing != source_name:
            ambiguous_roots.add(root_id)

    aliases: dict[str, str] = {}
    for spec in private_specs:
        source_name = str(getattr(spec, "source_name", "") or "")
        body_id = getattr(spec, "body_id", None)
        if not source_name or body_id is None:
            continue
        root_id = _body_root_id(model, int(body_id))
        if root_id is None or root_id in ambiguous_roots:
            continue
        skill_source = skill_source_by_root.get(root_id)
        if skill_source is not None:
            aliases[source_name] = skill_source

    # Some ProcTHOR doorway assets put the render/root body and the articulated
    # leaf in different MuJoCo body roots.  Their stable asset instance stem is
    # still shared (for example ``doorway_<hash>_1_0_2`` and
    # ``doorway_<hash>_1_2_2``).  Preserve the outer instance identifiers and
    # drop only the root/leaf hierarchy index, so two copies of the same asset
    # do not collapse into one candidate.  Use that evaluator-private relation
    # only as a conservative fallback, and only when exactly one channel skill
    # owns the key.  This repairs routability without granting a container a
    # door alias.
    def instance_stem(source_name: str) -> str:
        value = str(source_name)
        match = re.match(r"^(.*)_(\d+)_\d+_(\d+)$", value)
        if match:
            return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        return value

    channel_by_stem: dict[str, set[str]] = {}
    for source_name, joints in joints_by_object.items():
        if any(joint.domain == "channel" for joint in joints):
            channel_by_stem.setdefault(instance_stem(source_name), set()).add(source_name)
    for spec in private_specs:
        source_name = str(getattr(spec, "source_name", "") or "")
        if not source_name or source_name in aliases:
            continue
        candidates = channel_by_stem.get(instance_stem(source_name), set())
        if len(candidates) == 1:
            aliases[source_name] = next(iter(candidates))
    return aliases


def _build_restricted_ros_object_goal_runtime(
    *,
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
    public: PublicEpisode,
    config: BenchmarkEvaluationConfig,
    episode_index: int,
    frame_callback: callable | None = None,
    goal_status_observer: RosGoalStatusObserver | None = None,
    published_frame_sink: Callable[[Mapping[str, Any]], bool] | None = None,
) -> RestrictedRosObjectGoalRuntime:
    """Create evaluator-owned restricted perception and sealed force skills.

    The external ROS graph sees only the public target language and canonical
    restricted-GT observations.  The source-name/joint mapping lives entirely
    in this function's returned private runtime object.
    """

    perception = RestrictedGTPerceptionPublisher(
        camera_name="head_camera",
        min_visible_pixels=int(config.restricted_gt_min_visible_pixels),
        min_bbox_area_pixels=int(config.restricted_gt_min_bbox_area_pixels),
        min_bbox_short_side_pixels=int(
            getattr(config, "restricted_gt_min_bbox_short_side_pixels", 2)
        ),
        min_portal_bbox_short_side_pixels=int(
            getattr(config, "restricted_gt_min_portal_bbox_short_side_pixels", 8)
        ),
        min_visible_fraction=float(getattr(config, "restricted_gt_min_visible_fraction", 0.2)),
        max_distance_m=float(config.restricted_gt_max_distance_m),
        step_interval=1,
        frame_id="world",
        initial_episode_index=int(episode_index) + 1,
    )
    private_registry = OpaqueObjectRegistry()
    opaque_to_source_name: dict[str, str] = {}
    opaque_to_joints: dict[str, tuple[RuntimeJoint, ...]] = {}
    joints_by_object: dict[str, list[RuntimeJoint]] = {}
    for joint in catalog.joints:
        joints_by_object.setdefault(joint.object_name, []).append(joint)
    # The restricted renderer can report a door frame/root body while the
    # catalog resolves the articulated child body.  Pre-register every render
    # source and bind root/child aliases to one sealed evaluator skill so the
    # opaque ID observed by ROS is always routable when it denotes an actual
    # portal or container.  Non-articulated scene objects remain visible but
    # deliberately receive no interaction capability.
    private_specs = build_private_object_specs_from_env(task.env)
    source_skill_aliases = _perception_source_skill_aliases(
        model=task.env.current_model,
        private_specs=private_specs,
        joints_by_object=joints_by_object,
    )
    perception_sources = {
        str(getattr(spec, "source_name", "") or "")
        for spec in private_specs
    }
    source_names = sorted(
        source_name
        for source_name in (set(joints_by_object) | perception_sources)
        if source_name
    )
    for source_name in source_names:
        opaque_id = perception.registry.public_id_for(source_name)
        skill_source_name = source_skill_aliases.get(source_name)
        if skill_source_name is None and source_name in joints_by_object:
            skill_source_name = source_name
        if skill_source_name is None:
            continue
        all_joints = tuple(
            sorted(joints_by_object[skill_source_name], key=lambda item: item.joint_index)
        )
        # Public execution capability depends only on the live articulation.
        # Hidden episode interactions/plans are consulted after execution for
        # private scoring, never while choosing which joints the action opens.
        joints = all_joints
        private_registry.register(
            opaque_id,
            joints=joints,
            object_ref=skill_source_name,
            open_postcondition=OpenPostconditionSpec(
                success_fraction=float(SUCCESS_OPEN_FRACTION),
                minimum_open_joints=1,
            ),
        )
        opaque_to_source_name[opaque_id] = skill_source_name
        opaque_to_joints[opaque_id] = joints

    def execute_open_joint(joint: RuntimeJoint) -> JointOpenResult:
        succeeded, metadata, before, after, simulated_seconds = _execute_locked_force(
            task.env,
            joint,
            config,
            frame_callback=frame_callback,
        )
        return JointOpenResult(
            executor_succeeded=bool(succeeded),
            open_fraction_before=before,
            open_fraction_after=after,
            simulated_seconds=float(simulated_seconds),
            metadata=metadata,
        )

    skill = TrustedInteractionSkill(private_registry, execute_open_joint)
    adapter = RosObjectGoalEvaluatorAdapter(
        target_topic=config.ros_target_topic,
        gt_observations_topic=config.ros_restricted_gt_topic,
        interaction_command_topic=config.ros_interaction_command_topic,
        interaction_result_topic=config.ros_interaction_result_topic,
        require_public_interaction_evidence=True,
        queue_rejected_interactions=True,
    )
    language_target = build_language_target_context(
        episode.get("language") if isinstance(episode.get("language"), dict) else {},
        public.instruction,
    )
    target_name = str(language_target.get("target_name") or public.instruction or "object")
    # The public semantic mapper intentionally normalizes a portal's opaque
    # ``obj_######`` identity to a generic ``door_####`` token.  Keep that
    # normalization evaluator-private: register the derived token as an alias
    # for the canonical opaque object, without publishing source/body names.
    instance_aliases: dict[str, str] = {}
    alias_collisions: set[str] = set()
    for canonical_id, joints in opaque_to_joints.items():
        # The public graph can only emit this form after classifying the same
        # public observation as a portal.  Use the sealed joint domain rather
        # than root/leaf scene metadata, which often disagrees about a door.
        # A container must never acquire a door alias merely because its node
        # was misclassified upstream.  Unregistered geometry-only doorways
        # also remain evaluator-invalid by design.
        if not any(joint.domain == "channel" for joint in joints):
            continue
        alias = opaque_door_instance_id(canonical_id)
        previous = instance_aliases.get(alias)
        if previous is None:
            instance_aliases[alias] = canonical_id
        elif previous != canonical_id:
            # The public generic token has only a four-digit hash suffix.  On
            # a collision, fail closed rather than route an interaction to a
            # different physical portal.
            alias_collisions.add(alias)
    for alias in alias_collisions:
        instance_aliases.pop(alias, None)
    # This evaluator-only startup diagnostic contains public generic door
    # tokens only.  It lets a short smoke confirm routing coverage without
    # exposing source-body or joint names to ROS or the semantic policy.
    print(
        "[v3-interaction-routing] "
        f"registered_channel_alias_count={len(instance_aliases)} "
        f"collision_count={len(alias_collisions)} "
        f"aliases={','.join(sorted(instance_aliases))}"
    )

    # Subscribe and establish the episode freshness boundary before publishing
    # the target or initial restricted frame.  Otherwise a fast semantic node
    # can publish a legitimate latched completion while the initial frame is
    # rendering, only for a later ``begin_episode`` call to clear it as stale.
    if goal_status_observer is not None:
        goal_status_observer.begin_episode(perception.episode_id)
    adapter.reset(
        episode_id=perception.episode_id,
        target_context=build_ros_target_context(
            episode_id=perception.episode_id,
            target_name=target_name,
            instruction=public.instruction,
            object_labels=list(language_target.get("object_labels") or [target_name]),
            enabled=bool(language_target.get("enabled", True)),
            min_visible_pixels=int(config.restricted_gt_min_visible_pixels),
            min_visible_fraction=0.0,
            min_consecutive_observations=1,
            completion_requires_visibility=True,
            require_current_visibility=False,
        ),
        # This mapping is evaluator-private.  The adapter uses its keys only to
        # validate opaque IDs from ROS interaction commands.
        private_instances={opaque_id: opaque_id for opaque_id in opaque_to_source_name},
        instance_aliases=instance_aliases,
    )
    target_source_by_opaque_id = {
        perception.registry.public_id_for(source_name): source_name
        for source_name in target_candidate_names(episode)
    }
    goal_evidence = PublicGoalEvidenceLedger(
        episode_id=perception.episode_id,
        target_instance_ids=target_source_by_opaque_id,
    )
    initial_frame = perception.build(task, force=True)
    if initial_frame is not None:
        published = adapter.publish_restricted_gt_frame(initial_frame, capture_step=0)
        goal_evidence.record_frame(published, capture_step=0)
        if published_frame_sink is not None:
            published_frame_sink(published)
    return RestrictedRosObjectGoalRuntime(
        perception=perception,
        adapter=adapter,
        skill=skill,
        opaque_to_source_name=opaque_to_source_name,
        opaque_to_joints=opaque_to_joints,
        target_source_by_opaque_id=target_source_by_opaque_id,
        goal_evidence=goal_evidence,
        published_frame_sink=published_frame_sink,
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_episodes(benchmark: Path) -> tuple[Path, list[dict[str, Any]]]:
    return load_benchmark_episodes(benchmark)


def _selected_indices(config: BenchmarkEvaluationConfig, episodes: list[dict[str, Any]]) -> list[int]:
    indices = list(range(len(episodes))) if config.episode_indices is None else list(config.episode_indices)
    if config.max_episodes is not None:
        indices = indices[: int(config.max_episodes)]
    normalized = [int(index) for index in indices]
    if len(normalized) != len(set(normalized)):
        raise ValueError("episode_indices must not contain duplicates")
    for index in normalized:
        if not 0 <= index < len(episodes):
            raise IndexError(f"episode index {index} outside [0,{len(episodes)})")
    return normalized


def _skip_replay_freeze_task_config(
    _observation: Any,
    *,
    task: Any | None = None,
) -> None:
    """V3 JSON is already frozen; satisfy ``BaseMujocoTask.reset`` only."""

    del task
    return None


def _build_replay_config(
    config: BenchmarkEvaluationConfig,
    output_dir: Path,
    *,
    task_horizon: int | None = None,
) -> Any:
    """Build the minimal RBY1 config consumed by JSON replay."""

    from molmo_spaces.configs.task_configs import NavToObjTaskConfig
    from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
    from molmo_spaces.robots.rby1 import RBY1
    from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView
    from molmo_spaces.molmo_spaces_constants import get_robot_path

    class ReplayRBY1(RBY1):
        """RBY1 scene insertion without legacy post-insertion XML lookup."""

        @classmethod
        def apply_control_overrides(cls, spec, robot_config):
            # add_robot_to_scene already creates the holonomic actuators with
            # the intended ranges.  The legacy override re-queries attached
            # joints through an API that no longer exposes them on MjSpec.
            return None

    init_qpos = {
        "base": np.array([0.0, 0.0, 0.0]),
        "head": np.array([0.0, 0.6]),
        "left_arm": np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0]),
        "left_gripper": np.array([-0.05]),
        "right_arm": np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0]),
        "right_gripper": np.array([-0.05]),
        "torso": np.zeros(6, dtype=float),
    }
    init_qpos_noise_range = {
        group_name: np.zeros_like(values, dtype=float)
        for group_name, values in init_qpos.items()
    }
    robot_config = SimpleNamespace(
        robot_cls=ReplayRBY1,
        robot_factory=ReplayRBY1,
        robot_namespace="robot_0/",
        init_qpos=init_qpos,
        init_qpos_noise_range=init_qpos_noise_range,
        default_world_pose=[0.0, 0.0, 0.0],
        use_holo_base=True,
        command_mode={
            "arm": "joint_position",
            "gripper": "joint_position",
            "base": "holo_joint_planar_position",
            "head": None,
        },
        name="rby1",
        robot_xml_path=Path("rby1_site_control.xml"),
        gravcomp=True,
        K_stiffness=None,
        K_damping=None,
        action_noise_config=SimpleNamespace(enabled=False),
        holo_base_position_limit_m=100.0,
        holo_base_yaw_control_mode=config.holo_base_yaw_control_mode,
        robot_view_factory=lambda mj_data, namespace: RBY1RobotView(
            mj_data,
            namespace,
            holo_base=True,
            holo_base_yaw_control_mode=config.holo_base_yaw_control_mode,
        ),
        # BaseRobotConfig exposes this method; keep the lightweight replay
        # namespace compatible without invoking Pydantic model copying.
        get_robot_xml_path=lambda: get_robot_path("rby1") / "rby1_site_control.xml",
    )
    replay = SimpleNamespace(
        num_envs=1,
        num_workers=1,
        num_threads=1,
        task_type="nav_to_obj",
        use_passive_viewer=False,
        viewer_camera=None,
        viewer_cam_dict={},
        policy_dt_ms=float(config.policy_dt_ms),
        ctrl_dt_ms=float(config.ctrl_dt_ms),
        sim_dt_ms=float(config.sim_dt_ms),
        task_horizon=int(config.max_steps if task_horizon is None else task_horizon),
        end_on_success=False,
        collision_free_pose_limit=3,
        scene_dataset="procthor-10k",
        data_split="val",
        camera_config=None,
        robot_config=robot_config,
        task_sampler_config=NavToObjTaskSamplerConfig(
            task_sampler_class=None, house_variant="base"
        ),
        task_config=NavToObjTaskConfig(task_cls=None),
        task_config_preset_exp=None,
        task_config_preset_scn=None,
        # JsonEvalTaskSampler reads this flag even for the evaluator-only
        # adapter; provide the minimal policy contract without constructing a
        # planner policy (which would pull in optional manipulation deps).
        policy_config=SimpleNamespace(force_enable_depth=False),
        benchmark_path=None,
        eval_runtime_params=None,
        output_dir=output_dir,
        profile=False,
        profiler=None,
        datagen_profiler=False,
        log_level="info",
        use_wandb=False,
        wandb_project=None,
        wandb_name=None,
        filter_for_successful_trajectories=False,
        use_filament=False,
        environment_light_intensity=15000.0,
        record_videos=False,
        seed=None,
        seed_torch=False,
        freeze_task_config=_skip_replay_freeze_task_config,
    )
    if _is_current_ros_policy(config):
        for group_name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
            replay.robot_config.init_qpos[group_name] = np.asarray(qpos, dtype=float)
            replay.robot_config.init_qpos_noise_range[group_name] = np.zeros(len(qpos), dtype=float)
    return replay


def _budget_oracle_plans(nav: dict[str, Any]) -> list[dict[str, Any]]:
    plans = list(nav.get("oracle_plans") or [])
    if not plans and nav.get("oracle_plan"):
        plans = [dict(nav["oracle_plan"])]
    return plans


def _budget_plan_components(
    nav: dict[str, Any],
    config: BenchmarkEvaluationConfig,
) -> dict[str, Any]:
    """Return the least-complex valid oracle-plan shape for budgeting only."""

    interactions = {
        str(row.get("interaction_id")): row
        for row in nav.get("interactions", [])
        if row.get("interaction_id") is not None
    }
    plans = _budget_oracle_plans(nav)
    if not plans:
        plans = [{"plan_id": None, "required_interaction_ids": list(interactions)}]

    candidates: list[dict[str, Any]] = []
    for plan in plans:
        required_ids = [str(value) for value in plan.get("required_interaction_ids", [])]
        type_counts = {"channel": 0, "container": 0, "unknown": 0}
        for interaction_id in required_ids:
            interaction_type = str(interactions.get(interaction_id, {}).get("type", ""))
            if interaction_type.startswith("channel_"):
                type_counts["channel"] += 1
            elif interaction_type.startswith("container_"):
                type_counts["container"] += 1
            else:
                type_counts["unknown"] += 1
        candidates.append(
            {
                "plan_id": plan.get("plan_id"),
                "required_interaction_count": len(required_ids),
                "required_interaction_type_counts": type_counts,
            }
        )

    def complexity(row: dict[str, Any]) -> tuple[int, int, str]:
        counts = row["required_interaction_type_counts"]
        weighted = (
            int(counts["channel"]) * int(config.dynamic_channel_interaction_steps)
            + int(counts["container"]) * int(config.dynamic_container_interaction_steps)
            + int(counts["unknown"]) * int(config.dynamic_container_interaction_steps)
        )
        return weighted, int(row["required_interaction_count"]), str(row.get("plan_id") or "")

    return min(candidates, key=complexity)


def _budget_reference_path_length_m(episode: dict[str, Any], plan_id: Any) -> tuple[float, str]:
    try:
        reference = reference_path_length_m(episode)
    except (KeyError, TypeError, ValueError):
        reference = None
    if reference is not None:
        try:
            reference = float(reference)
        except (TypeError, ValueError):
            reference = None
        if reference is not None and math.isfinite(reference):
            return max(0.0, reference), "frozen_navigation_validation"

    nav = episode.get("interactive_nav", {})
    plans = _budget_oracle_plans(nav)
    selected = next((plan for plan in plans if plan.get("plan_id") == plan_id), plans[0] if plans else None)
    pose = episode.get("task", {}).get("robot_base_pose", [])
    if selected is not None and len(pose) >= 2:
        current = np.asarray(pose[:2], dtype=float)
        distance = 0.0
        used = False
        for step in selected.get("steps", []):
            if step.get("type") != "navigate":
                continue
            goal = np.asarray(step.get("goal_point", [])[:2], dtype=float)
            if goal.shape != (2,):
                continue
            distance += float(np.linalg.norm(goal - current))
            current = goal
            used = True
        if used:
            return distance, "oracle_waypoint_polyline_fallback"
    return 0.0, "missing_path_fallback"


def _safe_reference_path_length_m(episode: dict[str, Any]) -> float | None:
    try:
        value = reference_path_length_m(episode)
    except (KeyError, TypeError, ValueError):
        return None
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _budget_initial_target_visible(episode: dict[str, Any], minimum_visible_pixels: int) -> bool:
    nav = episode.get("interactive_nav", {})
    if nav.get("initial_state", {}).get("target_visible") is True:
        return True
    validation = nav.get("generation_validation", {}).get("navigation_validation", {})
    pixels = validation.get("start_visible_pixels")
    fraction = validation.get("start_visibility_fraction")
    return bool(
        pixels is not None
        and fraction is not None
        and int(pixels) >= int(minimum_visible_pixels)
        and float(fraction) > 0.0
    )


def _budget_container_joint_count(episode: dict[str, Any]) -> int:
    container_name = episode.get("interactive_nav", {}).get("target", {}).get("container_name")
    if not container_name:
        return 0
    joints = {
        (row.get("joint_name"), row.get("joint_index"))
        for row in episode.get("scene_modifications", {}).get("articulation_states", [])
        if row.get("object_name") == container_name
    }
    return len(joints)


def episode_step_budget(
    config: BenchmarkEvaluationConfig,
    episode: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Compute a deterministic evaluator-private rollout budget for one episode."""

    if config.step_budget_mode == "fixed":
        return int(config.max_steps), {
            "evaluator_private": True,
            "formula_version": STEP_BUDGET_FORMULA_VERSION,
            "mode": "fixed",
            "effective_max_steps": int(config.max_steps),
        }

    nav = episode.get("interactive_nav", {})
    plan = _budget_plan_components(nav, config)
    path_length_m, path_source = _budget_reference_path_length_m(episode, plan.get("plan_id"))
    initial_target_visible = _budget_initial_target_visible(
        episode,
        config.restricted_gt_min_visible_pixels,
    )
    counts = dict(plan["required_interaction_type_counts"])
    container_joint_count = _budget_container_joint_count(episode)

    # A missing/invalid frozen path is not evidence of a short task.  Use the
    # hard cap so a malformed record cannot silently create a premature timeout.
    if path_source != "frozen_navigation_validation":
        return int(config.max_steps), {
            "evaluator_private": True,
            "formula_version": STEP_BUDGET_FORMULA_VERSION,
            "mode": "dynamic",
            "effective_max_steps": int(config.max_steps),
            "minimum_steps": int(config.min_steps),
            "maximum_steps": int(config.max_steps),
            "conservative_fallback": True,
            "fallback_reason": path_source,
            "reference_path_length_m": None,
            "required_plan_id": plan.get("plan_id"),
            "required_interaction_count": int(plan["required_interaction_count"]),
            "required_interaction_type_counts": counts,
            "container_joint_count": container_joint_count,
        }

    path_steps = float(config.dynamic_steps_per_path_m) * max(
        0.0,
        path_length_m - float(config.dynamic_path_free_m),
    )
    channel_steps = (
        0.0
        if initial_target_visible
        else float(config.dynamic_channel_interaction_steps) * int(counts["channel"])
    )
    container_steps = (
        0.0
        if initial_target_visible
        else float(config.dynamic_container_interaction_steps) * int(counts["container"])
    )
    unknown_steps = (
        0.0
        if initial_target_visible
        else float(config.dynamic_container_interaction_steps) * int(counts["unknown"])
    )
    container_joint_steps = 0.0
    if not initial_target_visible and int(counts["container"]) > 0:
        container_joint_steps = float(config.dynamic_container_joint_steps) * max(0, container_joint_count - 1)

    raw_steps = (
        float(config.min_steps)
        + path_steps
        + channel_steps
        + container_steps
        + unknown_steps
        + container_joint_steps
    )
    quantum = int(config.dynamic_step_quantum)
    rounded_steps = int(math.ceil(raw_steps / quantum) * quantum)
    effective = min(int(config.max_steps), max(int(config.min_steps), rounded_steps))
    basis = {
        "evaluator_private": True,
        "formula_version": STEP_BUDGET_FORMULA_VERSION,
        "mode": "dynamic",
        "effective_max_steps": effective,
        "minimum_steps": int(config.min_steps),
        "maximum_steps": int(config.max_steps),
        "step_quantum": quantum,
        "reference_path_length_m": path_length_m,
        "reference_path_source": path_source,
        "free_path_m": float(config.dynamic_path_free_m),
        "initial_target_visible": initial_target_visible,
        "required_plan_id": plan.get("plan_id"),
        "required_interaction_count": int(plan["required_interaction_count"]),
        "required_interaction_type_counts": counts,
        "container_joint_count": container_joint_count,
        "components": {
            "base_steps": int(config.min_steps),
            "path_steps": path_steps,
            "channel_interaction_steps": channel_steps,
            "container_interaction_steps": container_steps,
            "unknown_interaction_steps": unknown_steps,
            "container_joint_steps": container_joint_steps,
        },
        "raw_steps": raw_steps,
        "rounded_steps": rounded_steps,
    }
    return effective, basis


def _public_step_budget_basis(basis: dict[str, Any]) -> dict[str, Any]:
    """Expose only the rollout limit, never the GT inputs used to derive it."""

    return {
        "evaluator_private": True,
        "formula_version": basis.get("formula_version", STEP_BUDGET_FORMULA_VERSION),
        "mode": basis.get("mode"),
        "budget_unit": "applied_action_step",
        "effective_max_steps": basis.get("effective_max_steps"),
        "conservative_fallback": bool(basis.get("conservative_fallback", False)),
    }


def _apply_ros_navigation_arm_posture(spec: EpisodeSpec) -> None:
    """Use the deterministic ROS navigation posture without mutating benchmark JSON."""

    for group_name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
        if group_name in spec.robot.init_qpos:
            spec.robot.init_qpos[group_name] = list(qpos)


def _is_current_ros_policy(config: BenchmarkEvaluationConfig) -> bool:
    """Whether the policy consumes the repository's RGB/depth ROS bridge."""

    return bool(
        config.policy in {"ros_bridge", "ros_object_goal_rule"}
        or (
            config.policy == "factory"
            and config.policy_factory
            and "ros_navigation_factory" in config.policy_factory
        )
    )


def _is_ros_object_goal_rule(config: BenchmarkEvaluationConfig) -> bool:
    """Whether this rollout owns restricted-GT and object-skill ROS topics."""

    return config.policy == "ros_object_goal_rule"


def _public_episode(episode: dict[str, Any], camera_names: list[str], resolution: tuple[int, int]) -> PublicEpisode:
    return PublicEpisode(
        house_index=int(episode["house_index"]),
        scene_dataset=str(episode["scene_dataset"]),
        data_split=str(episode["data_split"]),
        instruction=str(episode.get("language", {}).get("task_description", "")),
        task_type=str(episode.get("task", {}).get("task_type", "nav_to_obj")),
        camera_names=list(camera_names),
        image_resolution=tuple(int(value) for value in resolution),
    )


def _public_policy_observation(
    observation: Mapping[str, Any] | list[Mapping[str, Any]],
    public_episode: PublicEpisode,
) -> dict[str, Any]:
    """Copy only sensors that are part of the external-policy contract.

    The task observation also contains evaluator-private ground truth such as
    ``env_states`` and ``pickup_obj_pose``.  Passing the task dictionary through
    unchanged would let an external policy accidentally depend on hidden state.
    """

    if isinstance(observation, Mapping):
        batch_item = observation
    elif (
        isinstance(observation, list)
        and len(observation) == 1
        and isinstance(observation[0], Mapping)
    ):
        batch_item = observation[0]
    else:
        raise ValueError(
            "External-policy observation must be a mapping or a single-item mapping batch"
        )

    allowed_keys = {"robot_base_pose", "qpos"}
    for camera_name in public_episode.camera_names:
        allowed_keys.add(camera_name)
        allowed_keys.add(f"{camera_name}_depth")
        allowed_keys.add(f"sensor_param_{camera_name}")
    return copy.deepcopy(
        {key: value for key, value in batch_item.items() if key in allowed_keys}
    )


def _build_policy(config: BenchmarkEvaluationConfig, public: PublicEpisode) -> BenchmarkPolicy:
    if config.policy == "noop":
        return NoOpPolicy()
    if config.policy == "scripted_oracle":
        return ScriptedOraclePolicy()
    if config.policy == "factory":
        assert config.policy_factory is not None
        return build_factory_policy(config.policy_factory, public_episode=public, kwargs=config.policy_kwargs)
    if config.policy in {"ros_bridge", "ros_object_goal_rule"}:
        return build_ros_bridge_policy(
            policy_dt_ms=config.policy_dt_ms,
            observation_topic=config.ros_observation_topic,
            action_topic=config.ros_action_topic,
            action_timeout_s=config.ros_action_timeout_s,
            cmd_vel_linear_gain=config.ros_cmd_vel_linear_gain,
            require_move_base_active=config.ros_require_move_base_active,
            map_warmup_skip_frames=config.ros_map_warmup_skip_frames,
            step_ready_topic=config.ros_step_ready_topic,
            step_ready_barrier_enabled=config.ros_step_ready_barrier_enabled,
            step_ready_warmup_skip_frames=config.ros_step_ready_warmup_skip_frames,
            step_ready_timeout_s=config.ros_step_ready_timeout_s,
            step_ready_bootstrap_timeout_s=config.ros_step_ready_bootstrap_timeout_s,
            step_frame_dir=config.ros_step_frame_dir,
            step_frame_queue_size=config.ros_step_frame_queue_size,
            step_capture_ack_topic=config.ros_step_capture_ack_topic,
            step_capture_ack_barrier_enabled=config.ros_step_capture_ack_barrier_enabled,
            step_capture_ack_timeout_s=config.ros_step_capture_ack_timeout_s,
            name=config.policy,
        )
    raise ValueError(f"Unsupported policy: {config.policy}")


def _discard_task_rollout_cache(task: Any) -> None:
    # Upstream tasks retain every observation for collection.  Benchmark eval
    # writes a compact trace, so keeping image tensors leaks memory per episode.
    for name in ("observation_cache", "reward_cache", "terminal_cache", "truncated_cache", "success_cache", "action_cache"):
        cache = getattr(task, name, None)
        if isinstance(cache, list):
            cache.clear()


def _base_pose_xyyaw(task: Any) -> np.ndarray:
    pose = np.asarray(task.env.current_robot.robot_view.base.pose, dtype=float)
    return np.asarray([pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0])], dtype=float)


def _wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def _expected_runtime_base_xyyaw(episode: dict[str, Any]) -> np.ndarray:
    """Return the task-authoritative start pose in planar coordinates."""

    pose = np.asarray(episode.get("task", {}).get("robot_base_pose", []), dtype=float)
    if pose.shape == (7,):
        x, y, _z, qw, qx, qy, qz = [float(value) for value in pose]
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return np.asarray([x, y, yaw], dtype=float)
    pose = np.asarray(
        episode.get("robot", {}).get("init_qpos", {}).get("base", []),
        dtype=float,
    )
    return pose


def _runtime_consistency_checks(
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
    sampler: V3BenchmarkTaskSampler,
    config: BenchmarkEvaluationConfig,
) -> dict[str, Any]:
    """Validate that a frozen episode can be scored in the live scene.

    These checks intentionally stop before any policy action.  They cover the
    runtime identities and initial state that formal metrics depend on, while
    leaving navigation and manipulation performance to the evaluated policy.
    """

    nav = episode["interactive_nav"]
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    goal = oracle_terminal_goal_consistency(task, episode)
    checks["oracle_terminal_goal"] = goal
    if goal.get("checked") and not goal.get("consistent"):
        reasons.append("terminal_goal_target_mismatch")

    expected_base = _expected_runtime_base_xyyaw(episode)
    actual_base = _base_pose_xyyaw(task)
    if expected_base.shape == (3,):
        base_position_error = float(np.linalg.norm(actual_base[:2] - expected_base[:2]))
        base_yaw_error = abs(_wrap_angle(float(actual_base[2] - expected_base[2])))
        base_passed = bool(
            base_position_error <= float(config.runtime_base_position_tolerance_m)
            and base_yaw_error <= float(config.runtime_base_yaw_tolerance_rad)
        )
    else:
        base_position_error = None
        base_yaw_error = None
        base_passed = False
    checks["robot_start_pose"] = {
        "passed": base_passed,
        "expected_xyyaw": expected_base.tolist(),
        "actual_xyyaw": actual_base.tolist(),
        "position_error_m": base_position_error,
        "yaw_error_rad": base_yaw_error,
        "position_tolerance_m": float(config.runtime_base_position_tolerance_m),
        "yaw_tolerance_rad": float(config.runtime_base_yaw_tolerance_rad),
    }
    if not base_passed:
        reasons.append("robot_start_pose_mismatch")

    articulation_rows: list[dict[str, Any]] = []
    for expected in episode.get("scene_modifications", {}).get("articulation_states", []):
        joint_name = str(expected["joint_name"])
        joint_id = mujoco.mj_name2id(task.env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            articulation_rows.append({
                "joint_name": joint_name,
                "passed": False,
                "reason": "joint_not_found",
            })
            continue
        actual_position = float(probe.joint_value_by_name(task.env, joint_name))
        expected_position = float(expected["position"])
        actual_fraction = joint_open_fraction(task.env, {"joint_name": joint_name})
        expected_fraction = float(expected["open_fraction"])
        position_error = abs(actual_position - expected_position)
        fraction_error = abs(actual_fraction - expected_fraction)
        articulation_rows.append({
            "object_name": str(expected.get("object_name", "")),
            "joint_name": joint_name,
            "expected_position": expected_position,
            "actual_position": actual_position,
            "position_error": position_error,
            "expected_open_fraction": expected_fraction,
            "actual_open_fraction": actual_fraction,
            "open_fraction_error": fraction_error,
            "passed": bool(
                position_error <= float(config.runtime_joint_position_tolerance)
                and fraction_error <= float(config.runtime_joint_fraction_tolerance)
            ),
        })
    # ``unnecessary`` V3 episodes legitimately have no articulated joints.
    # Treat the empty set as a passing readback instead of excluding those
    # episodes before the policy gets a chance to navigate.
    articulation_passed = all(row["passed"] for row in articulation_rows)
    checks["articulation_state_readback"] = {
        "passed": articulation_passed,
        "joint_count": len(articulation_rows),
        "position_tolerance": float(config.runtime_joint_position_tolerance),
        "open_fraction_tolerance": float(config.runtime_joint_fraction_tolerance),
        "failed_joints": [row for row in articulation_rows if not row["passed"]],
    }
    if not articulation_passed:
        reasons.append("articulation_state_mismatch")

    resolution_rows: list[dict[str, Any]] = []
    for interaction in nav.get("interactions", []):
        object_name = str(interaction["object_name"])
        joint_index = int(interaction["joint_index"])
        expected_joint_name = str(interaction["joint_name"])
        resolved = catalog.by_name(object_name, joint_index)
        passed = bool(resolved is not None and resolved.joint_name == expected_joint_name)
        resolution_rows.append({
            "interaction_id": str(interaction["interaction_id"]),
            "object_name": object_name,
            "joint_index": joint_index,
            "expected_joint_name": expected_joint_name,
            "resolved_joint_name": None if resolved is None else resolved.joint_name,
            "passed": passed,
        })
    resolution_passed = all(row["passed"] for row in resolution_rows)
    checks["interaction_resolution"] = {
        "passed": resolution_passed,
        "interaction_count": len(resolution_rows),
        "failed_interactions": [row for row in resolution_rows if not row["passed"]],
    }
    if not resolution_passed:
        reasons.append("interaction_resolution_mismatch")

    expected_visible = nav.get("initial_state", {}).get("target_visible")
    if isinstance(expected_visible, bool):
        criteria = nav.get("success_criteria", {})
        camera_name = str(criteria.get("visibility", {}).get("camera_name", "head_camera"))
        target_rows = target_metric_rows(task, episode)
        chosen_target = min(target_rows, key=lambda row: float(row["distance_m"]))
        target_names = [str(row["target_name"]) for row in target_rows]
        candidate_visibility = {
            str(row["target_name"]): float(row["visibility_fraction"])
            for row in target_rows
        }
        actual_visibility = float(chosen_target["visibility_fraction"])
        actual_visible = actual_visibility > 0.0
        visibility_passed = actual_visible is expected_visible
        checks["initial_target_visibility"] = {
            "applicable": True,
            "passed": visibility_passed,
            "camera_name": camera_name,
            "target_names": target_names,
            "chosen_target_name": str(chosen_target["target_name"]),
            "expected_visible": expected_visible,
            "actual_visible": actual_visible,
            "actual_visibility_fraction": actual_visibility,
            "candidate_visibility_fractions": candidate_visibility,
        }
        if not visibility_passed:
            reasons.append("initial_target_visibility_mismatch")
    else:
        checks["initial_target_visibility"] = {"applicable": False, "passed": True}

    checks["scene_runtime_compatibility"] = {
        "passed": True,
        **sampler.runtime_compatibility,
    }
    return {
        "schema_version": "interactive_nav_v3_runtime_consistency_v1",
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "checks": checks,
    }


def _redact_runtime_consistency_for_restricted_policy(
    runtime_consistency: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Project evaluator-private replay checks into a safe public summary.

    The checks are necessary for formal scoring, but their raw target names,
    joint names, joint values and poses must not become an accidental side
    channel in a restricted-GT policy's result directory.
    """

    if runtime_consistency is None:
        return None, None
    checks = runtime_consistency.get("checks") if isinstance(runtime_consistency, dict) else {}
    checks = checks if isinstance(checks, dict) else {}
    goal = checks.get("oracle_terminal_goal") if isinstance(checks.get("oracle_terminal_goal"), dict) else {}
    public_goal = {
        "checked": bool(goal.get("checked", False)),
        "consistent": bool(goal.get("consistent", False)),
    }
    public_checks: dict[str, Any] = {"oracle_terminal_goal": public_goal}
    for name, source in checks.items():
        if name == "oracle_terminal_goal" or not isinstance(source, dict):
            continue
        summary: dict[str, Any] = {"passed": bool(source.get("passed", False))}
        for key in ("applicable", "joint_count", "interaction_count"):
            if key in source:
                summary[key] = source[key]
        public_checks[name] = summary
    return public_goal, {
        "schema_version": "interactive_nav_v3_runtime_consistency_public_v1",
        "eligible": bool(runtime_consistency.get("eligible", False)),
        "exclusion_reasons": list(runtime_consistency.get("exclusion_reasons", [])),
        "checks": public_checks,
    }


def _capture_head_frame(task: Any, frames: list[np.ndarray], enabled: bool) -> None:
    if not enabled:
        return
    try:
        frame = np.asarray(task.env.render_rgb_frame("head_camera"), dtype=np.uint8)
    except Exception:
        return
    frames.append(frame.copy())


def _save_video(frames: list[np.ndarray], destination: Path, fps: float) -> None:
    """Load Torch-backed video utilities only when video output is requested."""

    from molmo_spaces.utils.save_utils import save_frames_to_mp4

    save_frames_to_mp4(frames, str(destination), fps=float(fps))


def _robot_lock_snapshot(
    env: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    robot_view = env.current_robot.robot_view
    base = robot_view.base
    base_pose = np.asarray(base.pose, dtype=float).copy()
    base_ctrl = np.asarray(base.ctrl, dtype=float).copy()
    base_hold_target = np.asarray(
        [
            float(base_pose[0, 3]),
            float(base_pose[1, 3]),
            math.atan2(float(base_pose[1, 0]), float(base_pose[0, 0])),
        ],
        dtype=float,
    )
    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("left_arm", "right_arm", "left_gripper", "right_gripper", "torso", "head"):
        if name in robot_view.move_group_ids():
            group = robot_view.get_move_group(name)
            groups[name] = (
                np.asarray(group.joint_pos, dtype=float).copy(),
                np.zeros_like(np.asarray(group.joint_vel, dtype=float)),
                np.asarray(group.noop_ctrl, dtype=float).copy(),
            )
    return base_pose, base_ctrl, base_hold_target, groups


def _apply_robot_lock(
    env: Any,
    snapshot: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ],
) -> None:
    base_pose, base_ctrl, base_hold_target, groups = snapshot
    robot_view = env.current_robot.robot_view
    base = robot_view.base
    base.pose = base_pose.copy()
    base.joint_vel = np.zeros_like(base.joint_vel)
    try:
        # Match the collection force executor: hold the holonomic base at the
        # current world x/y/yaw target instead of restoring a stale/no-op
        # actuator command.  The latter lets the base controller fight the
        # external articulation force and can prevent a door from opening.
        base.ctrl = (
            base_hold_target.copy()
            if np.asarray(base.ctrl).shape == base_hold_target.shape
            else base_ctrl.copy()
        )
    except (AttributeError, ValueError):
        pass
    for name, (qpos, qvel, ctrl) in groups.items():
        group = robot_view.get_move_group(name)
        group.joint_pos = qpos.copy()
        group.joint_vel = qvel.copy()
        try:
            group.ctrl = ctrl.copy()
        except (AttributeError, ValueError):
            pass
    mujoco.mj_forward(env.current_model, env.current_data)
    env.camera_manager.registry.update_all_cameras(env)


def _execute_locked_force(
    env: Any,
    runtime_joint: RuntimeJoint,
    config: BenchmarkEvaluationConfig,
    *,
    frame_callback: callable | None = None,
    target_open_fraction: float | None = None,
) -> tuple[bool, dict[str, Any], float, float, float]:
    """Execute a force ramp while locking the robot at its current pose.

    ``target_open_fraction`` defaults to the configured opening target.  The
    drawer scan skill also uses this trusted routine with ``0.0`` to close one
    drawer before moving to the next one, preserving the same force schedule,
    collection frequency, and base/arm lock as ordinary V3 force opening.
    """

    before = joint_open_fraction(
        env,
        {"joint_name": runtime_joint.joint_name},
    )
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, runtime_joint.joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {runtime_joint.joint_name}")
    lower, upper = [float(value) for value in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lower, upper])
    start_fraction = float(before)
    target_fraction = float(
        config.force_target_fraction if target_open_fraction is None else target_open_fraction
    )
    target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
    target = closed + target_fraction * (opened - closed)
    controller = probe.ForceJointController(env, runtime_joint.joint_name, target)
    sim_dt = float(env.current_model.opt.timestep)
    sample_count = max(1, int(round(float(config.force_duration_seconds) * float(config.force_collection_hz))))
    internal_per_sample = max(1, int(round((1.0 / float(config.force_collection_hz)) / sim_dt)))
    internal_total = sample_count * internal_per_sample
    if internal_total > int(config.force_max_internal_steps):
        raise ValueError(
            f"force schedule requires {internal_total} internal steps, exceeding force_max_internal_steps="
            f"{config.force_max_internal_steps}"
        )
    snapshot = _robot_lock_snapshot(env)
    max_drift = 0.0
    effort_values: list[float] = []
    joint_values: list[float] = []
    direction_signs: list[float] = []
    try:
        for sample_index in range(sample_count):
            progress = (sample_index + 1) / sample_count
            scheduled_fraction = start_fraction + progress * (target_fraction - start_fraction)
            controller.target_value = closed + scheduled_fraction * (opened - closed)
            for _ in range(internal_per_sample):
                _apply_robot_lock(env, snapshot)
                row = controller.step()
                effort_values.append(float(row["effort"]))
                joint_values.append(float(row["joint_value"]))
                direction_signs.append(float(row["direction_sign"]))
                current_pose = env.current_robot.robot_view.base.pose
                max_drift = max(max_drift, float(np.linalg.norm(current_pose[:2, 3] - snapshot[0][:2, 3])))
                _apply_robot_lock(env, snapshot)
            if frame_callback is not None:
                frame_callback()
    finally:
        controller.finish()
        _apply_robot_lock(env, snapshot)
    after = joint_open_fraction(env, {"joint_name": runtime_joint.joint_name})
    metadata = controller.result()
    metadata.update(
        {
            "force_duration_seconds": float(config.force_duration_seconds),
            "force_collection_hz": float(config.force_collection_hz),
            "target_open_fraction": target_fraction,
            "dataset_steps": sample_count,
            "internal_steps": internal_total,
            "simulated_seconds": internal_total * sim_dt,
            "base_lock_enabled": True,
            "upper_body_lock_enabled": True,
            "base_max_xy_drift_m": max_drift,
            "mean_effort": None if not effort_values else float(np.mean(effort_values)),
            "joint_value_min": None if not joint_values else float(np.min(joint_values)),
            "joint_value_max": None if not joint_values else float(np.max(joint_values)),
            "force_direction_readback_adaptive": bool(controller.use_generalized_force),
            "observed_direction_signs": sorted(set(direction_signs)),
        }
    )
    reached = (
        after >= SUCCESS_OPEN_FRACTION
        if target_fraction >= 0.5
        else after <= 1.0 - SUCCESS_OPEN_FRACTION
    )
    return bool(reached), metadata, before, after, internal_total * sim_dt


def _drawer_scan_target_evidence(
    task: Any,
    episode: dict[str, Any],
    config: BenchmarkEvaluationConfig,
) -> dict[str, Any] | None:
    """Record a private target observation made while one drawer is open."""

    _visible, distance, visibility = target_metrics(task, episode)
    resolution = config.image_resolution or tuple(episode.get("img_resolution") or (640, 480))
    width, height = int(resolution[0]), int(resolution[1])
    visible_pixels = int(round(float(visibility) * max(1, width) * max(1, height)))
    threshold = float(episode["interactive_nav"]["success_criteria"]["distance"]["threshold_m"])
    if not (
        float(distance) < threshold
        and visible_pixels >= int(config.restricted_gt_min_visible_pixels)
    ):
        return None
    return {
        "distance_m": float(distance),
        "visibility_fraction": float(visibility),
        "visible_pixels": visible_pixels,
        "minimum_visible_pixels": int(config.restricted_gt_min_visible_pixels),
    }


def _drawer_scan_runtime_groups(
    task: Any,
    joints: tuple[RuntimeJoint, ...],
    open_regions: tuple[tuple[float, float], ...],
    *,
    allowed_joint_indices: set[int] | None = None,
    fallback_to_all: bool = False,
) -> list[tuple[RuntimeJoint, dict[str, Any]]]:
    """Ground a public drawer plan to private slide joints in visual order.

    ``fallback_to_all`` exists only for non-restricted compatibility callers.
    The restricted benchmark always passes ``False``: a public container box is
    object provenance, not evidence for every simulator-private drawer front.
    """

    model = task.env.current_model
    data = task.env.current_data
    by_name: dict[str, RuntimeJoint] = {}
    joint_rows: list[dict[str, Any]] = []
    body_heights: dict[str, float] = {}
    for joint in joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.joint_name)
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        by_name[joint.joint_name] = joint
        joint_rows.append({"joint_name": joint.joint_name, "joint_type": "slide"})
        body_heights[joint.joint_name] = float(data.xpos[body_id][2])
    regions = [{"center": [float(x), float(y)]} for x, y in open_regions]
    grounded = ground_drawer_open_regions(
        joint_rows,
        regions,
        body_heights,
        fallback_to_all=bool(fallback_to_all),
    )
    result: list[tuple[RuntimeJoint, dict[str, Any]]] = []
    for group in grounded:
        names = list(group.get("joint_names") or [])
        if len(names) != 1:
            continue
        joint = by_name.get(str(names[0]))
        if joint is not None and (
            allowed_joint_indices is None or int(joint.joint_index) in allowed_joint_indices
        ):
            result.append((joint, group))
    return result


def _execute_private_drawer_scan(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
    joints: tuple[RuntimeJoint, ...],
    open_regions: tuple[tuple[float, float], ...],
    allowed_joint_indices: set[int],
    fallback_to_all: bool = False,
    config: BenchmarkEvaluationConfig,
    decision_index: int,
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Run the trusted low-view, open-observe-close drawer macro skill.

    Only normalized visual regions enter from the method.  Whether this macro
    is available is determined by the live physical articulation group (slide
    joints plus valid regions), not by a hidden V3 interaction recipe.  The
    mapping to slide joints, force execution, target-observation evidence and
    all joint results remain evaluator-private.
    """

    groups = _drawer_scan_runtime_groups(
        task,
        joints,
        open_regions,
        allowed_joint_indices=allowed_joint_indices,
        fallback_to_all=fallback_to_all,
    )
    if not groups:
        return {
            "success": False,
            "joint_results": (),
            "opened_joints": (),
            "simulated_seconds": 0.0,
            "target_discovery": None,
            "metadata": {
                "execution_mode": "drawer_scan",
                "reason": "no_grounded_visible_drawers",
                "group_count": 0,
            },
            "observation": task.get_observations(),
        }

    initial_head = probe.get_head_joint_position(task.env)
    initial_torso = probe.get_torso_joint_position(task.env)
    view_applied = False
    group_records: list[dict[str, Any]] = []
    opened_pairs: list[tuple[RuntimeJoint, JointOpenResult]] = []
    touched_joints: list[RuntimeJoint] = []
    cleanup_records: list[dict[str, Any]] = []
    total_seconds = 0.0
    observed_target_discovery: dict[str, Any] | None = None
    completed = True
    cleanup_succeeded = True
    observation = task.get_observations()
    try:
        probe.lower_head_for_drawer_view(task.env, tilt_delta=0.30)
        if initial_torso is not None:
            probe.lean_torso_for_drawer_view(
                task.env,
                default_qpos=initial_torso,
                pitch_delta=0.35,
            )
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        view_applied = True
        _capture_head_frame(task, frames, config.record_video)
        for group_index, (joint, group) in enumerate(groups):
            opened = False
            closed = False
            group_record: dict[str, Any] = {
                "group_index": group_index,
                "grounding_source": str(group.get("grounding_source") or ""),
            }
            touched_joints.append(joint)
            try:
                opened, open_meta, before, after_open, open_seconds = _execute_locked_force(
                    task.env,
                    joint,
                    config,
                    frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                )
                total_seconds += float(open_seconds)
                open_result = JointOpenResult(
                    executor_succeeded=bool(opened),
                    open_fraction_before=float(before),
                    open_fraction_after=float(after_open),
                    simulated_seconds=float(open_seconds),
                    metadata=open_meta,
                )
                group_record.update(
                    {
                        "opened": bool(opened),
                        "open_fraction_after_open": float(after_open),
                        "open_seconds": float(open_seconds),
                        "open_metadata": open_meta,
                    }
                )
                if opened:
                    opened_pairs.append((joint, open_result))
                    observation = task.get_observations()
                    _capture_head_frame(task, frames, config.record_video)
                    published_open_frame = _publish_restricted_ros_frame(
                        runtime,
                        task,
                        decision_index=decision_index,
                    )
                    group_record["published_open_frame"] = bool(published_open_frame)
                    if published_open_frame:
                        evidence = _drawer_scan_target_evidence(task, episode, config)
                        if evidence is not None and observed_target_discovery is None:
                            observed_target_discovery = {**evidence, "group_index": group_index}
                else:
                    completed = False
            except Exception as exc:
                completed = False
                group_record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                # The close is in the per-drawer cleanup block rather than
                # after image/semantic updates.  A rendering or publication
                # failure must never leave this drawer open or allow the next
                # drawer to be opened concurrently.
                try:
                    closed, close_meta, _close_before, after_close, close_seconds = _execute_locked_force(
                        task.env,
                        joint,
                        config,
                        frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                        target_open_fraction=0.0,
                    )
                    total_seconds += float(close_seconds)
                    group_record.update(
                        {
                            "closed": bool(closed),
                            "open_fraction_after_close": float(after_close),
                            "close_seconds": float(close_seconds),
                            "close_metadata": close_meta,
                        }
                    )
                    if not closed:
                        completed = False
                except Exception as exc:
                    closed = False
                    completed = False
                    group_record["closed"] = False
                    group_record["close_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    observation = task.get_observations()
                    _capture_head_frame(task, frames, config.record_video)
                    group_record["published_closed_frame"] = bool(
                        _publish_restricted_ros_frame(
                            runtime,
                            task,
                            decision_index=decision_index,
                        )
                    )
                except Exception as exc:
                    # The physical close result remains authoritative for this
                    # macro.  Do not expose the exception to the method.
                    group_record["post_close_observation_error"] = f"{type(exc).__name__}: {exc}"
                group_records.append(group_record)
            if not (opened and closed):
                # A failed group is first cleaned up above, then terminates the
                # macro.  Continuing could open a second drawer while recovery
                # is already known to be unreliable.
                completed = False
                break
    finally:
        # A force/controller/render exception can occur between the main close
        # and result recording.  Verify every touched drawer is closed before
        # restoring the view; recovery remains evaluator-private and a failed
        # recovery makes the macro unsuccessful.
        seen_joint_names: set[str] = set()
        for joint in reversed(touched_joints):
            if joint.joint_name in seen_joint_names:
                continue
            seen_joint_names.add(joint.joint_name)
            try:
                remaining_fraction = joint_open_fraction(
                    task.env,
                    {"joint_name": joint.joint_name},
                )
                recovery_record: dict[str, Any] = {
                    "remaining_open_fraction_before": float(remaining_fraction),
                }
                if remaining_fraction > 1.0 - SUCCESS_OPEN_FRACTION:
                    recovered, recovery_meta, _before, after, recovery_seconds = _execute_locked_force(
                        task.env,
                        joint,
                        config,
                        frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                        target_open_fraction=0.0,
                    )
                    total_seconds += float(recovery_seconds)
                    recovery_record.update(
                        {
                            "recovery_attempted": True,
                            "recovered": bool(recovered),
                            "open_fraction_after_recovery": float(after),
                            "recovery_seconds": float(recovery_seconds),
                            "recovery_metadata": recovery_meta,
                        }
                    )
                    if not recovered:
                        cleanup_succeeded = False
                else:
                    recovery_record["recovery_attempted"] = False
                cleanup_records.append(recovery_record)
            except Exception as exc:
                cleanup_succeeded = False
                cleanup_records.append({"recovery_error": f"{type(exc).__name__}: {exc}"})
        if initial_head is not None:
            probe.set_head_joint_position(task.env, initial_head)
        if initial_torso is not None:
            probe.set_torso_joint_position(task.env, initial_torso)
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        observation = task.get_observations()
        _capture_head_frame(task, frames, config.record_video)
        _publish_restricted_ros_frame(runtime, task, decision_index=decision_index)

    success = bool(completed and cleanup_succeeded and len(opened_pairs) == len(groups))
    return {
        "success": success,
        "joint_results": tuple(result for _, result in opened_pairs),
        "opened_joints": tuple(joint for joint, _ in opened_pairs),
        "simulated_seconds": float(total_seconds),
        # A transient target observation becomes a task endpoint only when the
        # whole open-observe-close macro completed and all drawers were cleaned
        # up.  This prevents a failed/partially open scan from earning success.
        "target_discovery": observed_target_discovery if success else None,
        "metadata": {
            "execution_mode": "drawer_scan",
            "view_profile": "drawer_low_view",
            "view_applied": view_applied,
            "group_count": len(groups),
            "groups": group_records,
            "cleanup": cleanup_records,
        },
        "observation": observation,
    }


def _execute_private_drawer_open(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    joints: tuple[RuntimeJoint, ...],
    open_regions: tuple[tuple[float, float], ...],
    allowed_joint_indices: set[int],
    execute_group: Callable[[tuple[RuntimeJoint, ...]], tuple[Any | None, dict[str, Any]]],
    config: BenchmarkEvaluationConfig,
    decision_index: int,
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Run drawer-open without allowing one skill failure to abort the episode."""

    try:
        return _execute_private_drawer_open_impl(
            task=task,
            runtime=runtime,
            joints=joints,
            open_regions=open_regions,
            allowed_joint_indices=allowed_joint_indices,
            execute_group=execute_group,
            config=config,
            decision_index=decision_index,
            frames=frames,
        )
    except Exception as exc:
        try:
            observation = task.get_observations()
        except Exception:
            observation = {}
        return {
            "success": False,
            "joint_results": (),
            "selected_joints": (),
            "simulated_seconds": 0.0,
            "physics_substeps": 0,
            "pre_state": "unknown",
            "metadata": {
                "execution_mode": "ordinary_drawer_open_fast_sequence",
                "reason": "drawer_open_execution_failed",
                "error_type": type(exc).__name__,
                "group_count": 0,
            },
            "observation": observation,
        }


def _execute_private_drawer_open_impl(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    joints: tuple[RuntimeJoint, ...],
    open_regions: tuple[tuple[float, float], ...],
    allowed_joint_indices: set[int],
    execute_group: Callable[[tuple[RuntimeJoint, ...]], tuple[Any | None, dict[str, Any]]],
    config: BenchmarkEvaluationConfig,
    decision_index: int,
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Mirror the ordinary fast drawer-open lifecycle with public observations.

    Each M1-grounded drawer is opened separately from the low drawer view.  A
    fresh restricted frame is published while that drawer remains open, then
    the view is restored and every selected joint is verified still open.  The
    evaluator never falls back to unobserved simulator slide joints.
    """

    groups = _drawer_scan_runtime_groups(
        task,
        joints,
        open_regions,
        allowed_joint_indices=allowed_joint_indices,
        fallback_to_all=False,
    )
    if not groups:
        return {
            "success": False,
            "joint_results": (),
            "selected_joints": (),
            "simulated_seconds": 0.0,
            "physics_substeps": 0,
            "pre_state": "unknown",
            "metadata": {
                "execution_mode": "ordinary_drawer_open_fast_sequence",
                "reason": "no_grounded_visible_drawers",
                "group_count": 0,
            },
            "observation": task.get_observations(),
        }

    initial_head = probe.get_head_joint_position(task.env)
    initial_torso = probe.get_torso_joint_position(task.env)
    selected_joints = tuple(joint for joint, _group in groups)
    joint_results: list[JointOpenResult] = []
    group_records: list[dict[str, Any]] = []
    total_seconds = 0.0
    physics_substeps = 0
    completed = True
    view_applied = False
    pre_state = "unknown"
    execution_error: str | None = None
    view_restore_error: str | None = None
    observation = task.get_observations()
    try:
        probe.lower_head_for_drawer_view(task.env, tilt_delta=0.30)
        if initial_torso is not None:
            probe.lean_torso_for_drawer_view(
                task.env,
                default_qpos=initial_torso,
                pitch_delta=0.35,
            )
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        view_applied = True
        _capture_head_frame(task, frames, config.record_video)
        for group_index, (joint, group) in enumerate(groups):
            execution, private_metadata = execute_group((joint,))
            group_record: dict[str, Any] = {
                "group_index": int(group_index),
                "grounding_source": str(group.get("grounding_source") or ""),
                "executor": private_metadata,
            }
            if execution is None:
                current_fraction = joint_open_fraction(
                    task.env,
                    {"joint_name": joint.joint_name},
                )
                result = JointOpenResult(
                    executor_succeeded=False,
                    open_fraction_before=current_fraction,
                    open_fraction_after=current_fraction,
                    error="articulation_resolution_failed",
                )
                joint_results.append(result)
                group_record.update(
                    {
                        "opened": False,
                        "reason": "articulation_resolution_failed",
                    }
                )
                group_records.append(group_record)
                completed = False
                break
            execution_results = tuple(execution.joint_results)
            if execution_results:
                result = execution_results[0]
            else:
                after_fraction = joint_open_fraction(
                    task.env,
                    {"joint_name": joint.joint_name},
                )
                result = JointOpenResult(
                    executor_succeeded=bool(execution.success),
                    open_fraction_before=None,
                    open_fraction_after=after_fraction,
                    simulated_seconds=float(execution.simulated_seconds),
                )
            joint_results.append(result)
            total_seconds += float(execution.simulated_seconds)
            physics_substeps += int(execution.physics_substeps)
            if group_index == 0:
                pre_state = str(execution.pre_state)
            opened = bool(
                execution.success
                and result.open_fraction_after is not None
                and float(result.open_fraction_after) >= SUCCESS_OPEN_FRACTION
            )
            group_record.update(
                {
                    "opened": opened,
                    "open_fraction_after": result.open_fraction_after,
                    "simulated_seconds": float(execution.simulated_seconds),
                    "physics_substeps": int(execution.physics_substeps),
                }
            )
            if opened:
                observation = task.get_observations()
                _capture_head_frame(task, frames, config.record_video)
                group_record["published_open_frame"] = bool(
                    _publish_restricted_ros_frame(
                        runtime,
                        task,
                        decision_index=decision_index,
                    )
                )
            else:
                completed = False
            group_records.append(group_record)
            if not opened:
                break
    except Exception as exc:
        # Keep simulator/runtime details evaluator-private while ensuring the
        # public interaction receives a normal failed postcondition instead of
        # crashing the whole episode.
        completed = False
        execution_error = type(exc).__name__
        group_records.append(
            {
                "group_index": len(group_records),
                "opened": False,
                "reason": "drawer_open_execution_failed",
                "error_type": execution_error,
            }
        )
    finally:
        try:
            if initial_head is not None:
                probe.set_head_joint_position(task.env, initial_head)
            if initial_torso is not None:
                probe.set_torso_joint_position(task.env, initial_torso)
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            task.env.camera_manager.registry.update_all_cameras(task.env)
            observation = task.get_observations()
            _capture_head_frame(task, frames, config.record_video)
            _publish_restricted_ros_frame(
                runtime,
                task,
                decision_index=decision_index,
            )
        except Exception as exc:
            completed = False
            view_restore_error = type(exc).__name__

    final_open_fractions = {
        joint.joint_name: float(
            joint_open_fraction(task.env, {"joint_name": joint.joint_name})
        )
        for joint in selected_joints
    }
    final_state_satisfied = bool(selected_joints) and all(
        fraction >= SUCCESS_OPEN_FRACTION
        for fraction in final_open_fractions.values()
    )
    success = bool(
        completed
        and len(joint_results) == len(selected_joints)
        and final_state_satisfied
    )
    return {
        "success": success,
        "joint_results": tuple(joint_results),
        "selected_joints": selected_joints,
        "simulated_seconds": float(total_seconds),
        "physics_substeps": int(physics_substeps),
        "pre_state": pre_state,
        "metadata": {
            "execution_mode": "ordinary_drawer_open_fast_sequence",
            "view_profile": "drawer_low_view",
            "view_applied": view_applied,
            "group_count": len(groups),
            "groups": group_records,
            "final_state_satisfied": final_state_satisfied,
            "final_open_fractions": final_open_fractions,
            "execution_error_type": execution_error,
            "view_restore_error_type": view_restore_error,
        },
        "observation": observation,
    }


def _runtime_joint_from_v3(interaction: dict[str, Any]) -> RuntimeJoint:
    return RuntimeJoint(
        object_name=str(interaction["object_name"]),
        object_category=None if interaction.get("object_category") is None else str(interaction["object_category"]),
        domain="channel" if str(interaction["type"]).startswith("channel_") else "container",
        joint_name=str(interaction["joint_name"]),
        joint_index=int(interaction["joint_index"]),
        body_id=-1,
        aabb_center=np.zeros(3, dtype=float),
        aabb_size=np.zeros(3, dtype=float),
    )


def _aabb_distance_xy(robot_xy: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    if not np.any(size):
        return 0.0
    delta = np.maximum(np.abs(np.asarray(robot_xy) - np.asarray(center)[:2]) - np.asarray(size)[:2] / 2.0, 0.0)
    return float(np.linalg.norm(delta))


def _resolve_interaction(
    *,
    env: Any,
    episode: dict[str, Any],
    catalog: InteractionCatalog,
    action: PolicyAction,
    uses_oracle_gt: bool,
    allow_internal_object_names: bool,
) -> tuple[RuntimeJoint | None, str | None, str, dict[str, Any]]:
    """Resolve a policy request without exposing V3 identifiers to ordinary policies."""

    if action.operation != "open":
        return None, None, "invalid", {"reason": f"unsupported_operation:{action.operation}"}
    nav_rows = list(episode["interactive_nav"].get("interactions", []))
    by_id = {str(row["interaction_id"]): row for row in nav_rows}
    oracle_id = action.metadata.get("oracle_interaction_id")
    details: dict[str, Any] = {}
    if oracle_id is not None:
        if not uses_oracle_gt:
            return None, None, "invalid", {"reason": "oracle_interaction_id_from_nonoracle"}
        interaction = by_id.get(str(oracle_id))
        if interaction is None:
            return None, None, "invalid", {"reason": "unknown_oracle_interaction_id"}
        return _runtime_joint_from_v3(interaction), str(oracle_id), "required_valid", {"resolver": "oracle_id"}
    candidate: RuntimeJoint | None
    if action.pixel_xy is not None or action.normalized_pixel_xy is not None:
        candidate, details = catalog.by_pixel(
            camera_name=action.camera_name,
            pixel_xy=action.pixel_xy,
            normalized_pixel_xy=action.normalized_pixel_xy,
            joint_index=action.joint_index,
        )
        if candidate is None:
            return None, None, "invalid", details
        details["resolver"] = "pixel"
    elif action.object_name is not None and allow_internal_object_names:
        candidate = catalog.by_name(str(action.object_name), action.joint_index)
        if candidate is None:
            return None, None, "invalid", {"reason": "unknown_or_ambiguous_internal_object_name"}
        details = {"resolver": "internal_object_name_debug"}
    else:
        return None, None, "invalid", {"reason": "nonoracle_interact_requires_pixel_selector"}
    matching = [
        row
        for row in nav_rows
        if str(row["object_name"]) == candidate.object_name and int(row["joint_index"]) == candidate.joint_index
    ]
    if len(matching) == 1:
        return candidate, str(matching[0]["interaction_id"]), "required_valid", details
    return candidate, None, "extra_valid", details


def _check_interaction_access(
    env: Any,
    runtime_joint: RuntimeJoint,
    *,
    camera_name: str,
    max_distance_m: float,
    require_visible: bool,
) -> tuple[bool, dict[str, Any]]:
    robot_xy = np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    distance = _aabb_distance_xy(robot_xy, runtime_joint.aabb_center, runtime_joint.aabb_size)
    visible = None
    if require_visible:
        try:
            visible = float(env.check_visibility(camera_name, runtime_joint.object_name))
        except Exception as exc:
            return False, {"reason": "visibility_check_failed", "error": f"{type(exc).__name__}: {exc}", "distance_m": distance}
    if distance > max_distance_m:
        return False, {"reason": "interaction_too_far", "distance_m": distance, "max_distance_m": max_distance_m, "visibility": visible}
    if require_visible and (visible is None or visible <= 0.0):
        return False, {"reason": "interaction_not_visible", "distance_m": distance, "visibility": visible}
    return True, {"distance_m": distance, "visibility": visible}


def _resolved_interaction_ids(attempt: dict[str, Any]) -> list[str]:
    """Return all private V3 IDs credited by one evaluator-side attempt."""

    values = attempt.get("resolved_interaction_ids")
    if isinstance(values, (list, tuple)):
        resolved = [str(value) for value in values if value is not None]
        if resolved:
            return resolved
    value = attempt.get("resolved_interaction_id")
    return [] if value is None else [str(value)]


def _episode_oracle_plans(episode: dict[str, Any]) -> list[dict[str, Any]]:
    nav = episode["interactive_nav"]
    plans = list(nav.get("oracle_plans") or [])
    if not plans and nav.get("oracle_plan"):
        plans = [dict(nav["oracle_plan"])]
    return plans


def _order_interaction_ids_by_oracle_plan(
    episode: dict[str, Any], interaction_ids: list[str]
) -> list[str]:
    """Put one macro action's credited IDs in a valid plan/prerequisite order."""

    unique_ids = list(dict.fromkeys(str(value) for value in interaction_ids))
    available = set(unique_ids)
    best_order: list[str] = []
    for plan in _episode_oracle_plans(episode):
        ordered = list(
            dict.fromkeys(
                str(value)
                for value in plan.get("required_interaction_ids", [])
                if str(value) in available
            )
        )
        if len(ordered) > len(best_order):
            best_order = ordered
    used = set(best_order)
    return [*best_order, *(value for value in unique_ids if value not in used)]


def _successful_object_skill_interaction_ids(
    *,
    episode: dict[str, Any],
    source_name: str,
    joints: tuple[RuntimeJoint, ...],
    joint_results: tuple[JointOpenResult, ...],
) -> list[str]:
    """Privately map sealed object-skill postconditions to frozen V3 rows.

    The navigation method receives just one opaque ``open(instance_id)`` result.
    V3's evaluator still needs to know which recorded articulation postconditions
    are actually satisfied.  This mapping intentionally remains in the runner
    and considers only final fractions, never controller substeps.
    """

    rows = list(episode["interactive_nav"].get("interactions", []))
    resolved: list[str] = []
    for joint, result in zip(joints, joint_results, strict=False):
        if result.open_fraction_after is None or float(result.open_fraction_after) < SUCCESS_OPEN_FRACTION:
            continue
        for row in rows:
            if (
                str(row.get("object_name")) == source_name
                and int(row.get("joint_index", -1)) == int(joint.joint_index)
            ):
                resolved.append(str(row["interaction_id"]))
    return _order_interaction_ids_by_oracle_plan(episode, resolved)


def _successful_drawer_scan_interaction_ids(
    *,
    episode: dict[str, Any],
    source_name: str,
    opened_joints: tuple[RuntimeJoint, ...],
) -> list[str]:
    """Credit only private drawer joints that reached the open observation."""

    ids_by_joint_index: dict[int, list[str]] = {}
    for row in episode["interactive_nav"].get("interactions", []):
        if str(row.get("object_name")) != source_name:
            continue
        ids_by_joint_index.setdefault(int(row.get("joint_index", -1)), []).append(
            str(row["interaction_id"])
        )
    resolved: list[str] = []
    seen: set[str] = set()
    # ``opened_joints`` retains the physical top-to-bottom drawer-scan order.
    # Preserve it for prerequisite scoring instead of reordering by JSON rows.
    for joint in opened_joints:
        for interaction_id in ids_by_joint_index.get(int(joint.joint_index), []):
            if interaction_id not in seen:
                resolved.append(interaction_id)
                seen.add(interaction_id)
    return resolved


def _publish_restricted_ros_frame(
    runtime: RestrictedRosObjectGoalRuntime,
    task: Any,
    *,
    decision_index: int,
) -> bool:
    """Render and publish one canonical public perception frame for ROS."""

    payload = runtime.perception.build(task, step_index=int(decision_index), force=True)
    if payload is None:
        return False
    published = runtime.adapter.publish_restricted_gt_frame(
        payload,
        capture_step=int(decision_index),
    )
    # Only observations that crossed the same restricted adapter boundary as
    # the policy are eligible as goal evidence.  A private MuJoCo visibility
    # result, including a transient drawer-open result, is never sufficient.
    runtime.goal_evidence.record_frame(
        published,
        capture_step=int(decision_index),
    )
    if runtime.published_frame_sink is not None:
        runtime.published_frame_sink(published)
    return True


def _private_target_distances_m(
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
) -> dict[str, float]:
    """Return private target ranges keyed by policy-visible opaque ids.

    This helper performs no visibility render and its result never crosses the
    evaluator boundary.  It is called only after the policy declares goal
    completion.
    """

    objects = task.env.object_managers[task.env.current_batch_index]
    robot_xy = np.asarray(
        task.env.current_robot.robot_view.base.pose[:2, 3],
        dtype=float,
    )
    result: dict[str, float] = {}
    for opaque_id, source_name in runtime.target_source_by_opaque_id.items():
        try:
            target = objects.get_object_by_name(source_name)
            target_xy = np.asarray(target.position[:2], dtype=float)
            result[opaque_id] = float(np.linalg.norm(target_xy - robot_xy))
        except Exception:
            continue
    return result


def _verify_restricted_goal_status(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
    payload: dict[str, Any],
) -> GoalClaimVerification:
    threshold = float(
        episode["interactive_nav"]["success_criteria"]["distance"]["threshold_m"]
    )
    return verify_target_goal_claim(
        payload,
        episode_id=runtime.perception.episode_id,
        evidence=runtime.goal_evidence,
        private_distances_m=_private_target_distances_m(task, runtime),
        distance_threshold_m=threshold,
    )


def _poll_restricted_goal_status(
    *,
    observer: RosGoalStatusObserver,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
) -> tuple[str, GoalClaimVerification | None, dict[str, Any]] | None:
    """Consume fresh public status messages and return the first terminal one."""

    for payload in observer.drain():
        if is_target_goal_success_claim(payload):
            verification = _verify_restricted_goal_status(
                task=task,
                runtime=runtime,
                episode=episode,
                payload=payload,
            )
            reason = "target_found" if verification.accepted else "target_claim_unverified"
            return reason, verification, payload
        if is_exploration_terminal(payload):
            status = str(payload.get("status") or "").strip().upper()
            reason_by_status = {
                "EXPLORATION_EXHAUSTED": "policy_exploration_exhausted",
                "EXPLORATION_STALLED": "policy_exploration_stalled",
                "STARTUP_SCAN_FAILED": "policy_startup_scan_failed",
            }
            return reason_by_status.get(status, "policy_exploration_exhausted"), None, payload
    return None


def _drain_restricted_goal_status(
    *,
    observer: RosGoalStatusObserver,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
    timeout_s: float,
    poll_interval_s: float = 0.02,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, GoalClaimVerification | None, dict[str, Any]] | None:
    """Give the final published frame one bounded, zero-step status drain.

    The last applied action may publish its restricted frame exactly when the
    action budget becomes exhausted.  ROS semantic processing is asynchronous,
    so a single immediate poll can miss a valid completion claim.  This drain
    never advances MuJoCo and never consumes an applied-action step.
    """

    timeout = max(0.0, float(timeout_s))
    interval = max(0.001, float(poll_interval_s))
    deadline = float(monotonic()) + timeout
    while True:
        terminal = _poll_restricted_goal_status(
            observer=observer,
            task=task,
            runtime=runtime,
            episode=episode,
        )
        if terminal is not None:
            return terminal
        remaining = deadline - float(monotonic())
        if remaining <= 0.0:
            return None
        sleep(min(interval, remaining))


def _compatibility_target_precheck(
    task: Any,
    episode: dict[str, Any],
    *,
    restricted_public_mode: bool,
    transient_target_discovery: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
) -> bool:
    """Preserve legacy private endpoints outside the restricted ROS protocol.

    The restricted object-goal method must declare and verify success through
    public goal evidence.  Compatibility/oracle policies historically use the
    live NavToObj metric as their rollout endpoint and must retain that behavior.
    """

    if restricted_public_mode:
        return False
    current_nav_ok, _target_distance, _target_visibility = target_metrics(
        task, episode
    )
    target_found = bool(current_nav_ok or transient_target_discovery is not None)
    if not target_found:
        return False
    nav = episode.get("interactive_nav") or {}
    if str(nav.get("interaction_requirement")) == "unnecessary":
        return True
    interaction_score = score_interactions(task.env, episode, attempts)
    return bool(
        interaction_score.required_interaction_success
        and interaction_score.sequence_success
    )


def _public_force_execution_outcome(
    *,
    execution: Any | None,
    executor_metadata: Mapping[str, Any],
    pose_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the ordinary bridge's leak-safe fast-force result contract."""

    physical_completed = bool(execution is not None and execution.success)
    public_failure_reason = str(
        executor_metadata.get("public_failure_reason") or ""
    ).strip()
    unsafe_open_sweep = public_failure_reason == "unsafe_open_sweep"
    if physical_completed:
        state = "open"
        capability = "articulated"
    elif execution is None:
        state = "unavailable"
        capability = "unavailable"
    elif unsafe_open_sweep:
        state = "unknown"
        capability = "articulated"
    else:
        state = "blocked"
        capability = "blocked"
    failure_reason = (
        ""
        if physical_completed
        else public_failure_reason
        if public_failure_reason
        else "articulation_resolution_failed"
        if execution is None
        else "force_target_not_reached"
    )
    verification_source = (
        "executor_open_sweep_preflight"
        if unsafe_open_sweep
        else "executor_resolution_failure"
        if execution is None
        else "executor_state_verification"
    )
    physics_substeps = 0 if execution is None else int(execution.physics_substeps)
    default_task_steps = 0 if execution is None or unsafe_open_sweep else 1
    task_steps_consumed = int(
        executor_metadata.get("task_steps_consumed", default_task_steps)
        or 0
    )
    result: dict[str, Any] = {
        "state": state,
        "pre_state": "unknown" if execution is None else str(execution.pre_state),
        "post_state": state,
        "interaction_capability": capability,
        "interactable": True if unsafe_open_sweep else physical_completed,
        "retryable": unsafe_open_sweep,
        "failure_reason": failure_reason,
        "verification_source": verification_source,
        "interaction_pose_validation": pose_validation,
        "physics_substeps": physics_substeps,
        "task_steps_consumed": task_steps_consumed,
        "execution_cost": 0.0 if unsafe_open_sweep else 1.0,
    }
    preflight = executor_metadata.get("open_sweep_preflight")
    if unsafe_open_sweep and isinstance(preflight, Mapping):
        try:
            recommended_retreat_m = float(
                preflight.get("recommended_retreat_m", 0.25)
            )
        except (TypeError, ValueError):
            recommended_retreat_m = 0.25
        if math.isfinite(recommended_retreat_m) and recommended_retreat_m >= 0.0:
            result["recommended_retreat_m"] = recommended_retreat_m
    return result


def _consume_pending_ros_object_goal_interaction(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
    private_attempts: list[dict[str, Any]],
    config: BenchmarkEvaluationConfig,
    decision_index: int,
    frames: list[np.ndarray],
) -> dict[str, Any] | None:
    """Execute at most one queued opaque object request inside the evaluator.

    The ROS navigation graph can only submit an opaque ID and ``open``.  This
    function resolves that ID, checks live approach access, and invokes the
    evaluator-owned force skill.  Its return value intentionally has separate
    private scoring and public trace projections.
    """

    request = runtime.adapter.pop_next_interaction_request()
    if request is None:
        return None
    rejection_reason = str(getattr(request, "rejection_reason", "") or "")
    if rejection_reason:
        # A geometric semantic doorway can be selectable by the navigation
        # method even when it has no evaluator-registered articulated object.
        # This is an invalid interaction request, not a failed force skill:
        # record it in the formal attempt counts while preserving genuine
        # execution failures for registered objects below.
        is_public_provenance_failure = rejection_reason == "interaction_not_visible"
        if is_public_provenance_failure:
            completion = runtime.adapter.complete_interaction(
                request.command_id,
                success=False,
                status="FAILED",
                outcome={
                    "state": "unknown",
                    "post_state": "unknown",
                    "interaction_capability": "unknown",
                    "retryable": True,
                    "failure_reason": rejection_reason,
                    "verification_source": "executor_capability_check",
                    "execution_cost": 0.0,
                },
            )
        else:
            completion = runtime.adapter.complete_interaction(
                request.command_id,
                success=False,
                status="INVALID",
                reason=rejection_reason,
                outcome={
                    "state": "unavailable",
                    "post_state": "unavailable",
                    "interaction_capability": "unavailable",
                    "interactable": False,
                    "retryable": False,
                    "failure_reason": rejection_reason,
                    "verification_source": "executor_capability_check",
                    "execution_cost": 0.0,
                },
            )
        public_request = {
            "request_id": request.command_id,
            "instance_id": request.instance_id,
            "operation": request.action,
        }
        # The adapter may resolve an unknown semantic portal from public
        # aperture evidence (for example ``static_open``).  The trace must use
        # that same public result instead of contradicting ROS with a hand-made
        # ``invalid`` status.
        public_result = {**public_request, **completion}
        action_skipped = bool(
            completion.get("success")
            and str(completion.get("state") or "") == "static_open"
        )
        private_attempt = InteractionAttempt(
            requested=public_request,
            classification="invalid",
            resolved_object_name=None,
            resolved_joint_name=None,
            resolved_joint_index=None,
            resolved_interaction_id=None,
            resolved_interaction_ids=[],
            success=False,
            joint_fraction_before=None,
            joint_fraction_after=None,
            prerequisite_satisfied=None,
            executor=None,
            simulated_seconds=0.0,
            metadata={
                "rejection": {
                    "reason": rejection_reason,
                    "node_id": request.node_id,
                    "candidate_id": request.candidate_id,
                    "rejected_before_execution": True,
                    "action_skipped": action_skipped,
                    "public_state": str(completion.get("state") or "unknown"),
                },
                "requested_interaction_ids": [],
            },
        ).to_dict()
        public_attempt = {
            **public_result,
            "decision_step": int(decision_index),
            "simulated_seconds": 0.0,
            "result_status": str(completion.get("status") or "INVALID"),
        }
        observation = task.get_observations()
        _capture_head_frame(task, frames, config.record_video)
        _publish_restricted_ros_frame(runtime, task, decision_index=decision_index)
        _discard_task_rollout_cache(task)
        return {
            "private_attempt": private_attempt,
            "public_attempt": public_attempt,
            "simulated_seconds": 0.0,
            "observation": observation,
            "target_discovery": None,
        }
    source_name = runtime.opaque_to_source_name.get(request.instance_id)
    joints = runtime.opaque_to_joints.get(request.instance_id, ())
    public_request = {
        "request_id": request.command_id,
        "instance_id": request.instance_id,
        "operation": request.action,
    }
    access_ok = False
    access_meta: dict[str, Any] = {"reason": "unknown_opaque_instance"}
    base_pose = np.asarray(task.env.current_robot.robot_view.base.pose, dtype=float)
    actual_pose_xyyaw = (
        float(base_pose[0, 3]),
        float(base_pose[1, 3]),
        float(math.atan2(base_pose[1, 0], base_pose[0, 0])),
    )
    pose_validation = validate_public_interaction_pose(
        request.public_command,
        actual_pose_xyyaw=actual_pose_xyyaw,
    )
    observation_validation = validate_public_interaction_observation(
        request.public_observation,
        actual_pose_xyyaw=actual_pose_xyyaw,
        max_distance_m=float(config.interaction_max_distance_m),
    )
    if not bool(observation_validation.get("checked")):
        access_meta = {
            "reason": "interaction_not_visible",
            "interaction_pose_validation": pose_validation,
            "public_observation_validation": observation_validation,
        }
    elif not bool(observation_validation.get("valid")):
        access_meta = {
            "reason": "interaction_too_far",
            "interaction_pose_validation": pose_validation,
            "public_observation_validation": observation_validation,
        }
    elif not bool(pose_validation.get("checked")) or not bool(
        pose_validation.get("valid")
    ):
        access_meta = {
            "reason": "interaction_pose_invalid",
            "interaction_pose_validation": pose_validation,
            "public_observation_validation": observation_validation,
        }
    elif source_name is not None and joints:
        # Restricted ROS execution mirrors the ordinary bridge: eligibility is
        # the method-produced public approach pose plus runtime capability.  A
        # private AABB/rerender visibility gate would make identical public
        # commands behave differently based on hidden geometry.
        access_ok = True
        access_meta = {
            "reason": (
                "public_interaction_pose_valid"
                if bool(pose_validation.get("checked"))
                else "no_expected_approach_pose"
            ),
            "interaction_pose_validation": pose_validation,
            "public_observation_validation": observation_validation,
        }

    joint_results: tuple[JointOpenResult, ...] = ()
    public_result: dict[str, Any]
    successful_ids: list[str] = []
    transient_satisfied_ids: list[str] = []
    target_discovery: dict[str, Any] | None = None
    executor_metadata: dict[str, Any] = {}
    executor_name = "trusted_object_force" if source_name is not None else None
    observation = task.get_observations()

    def execute_allowed_group(
        selected_joints: tuple[RuntimeJoint, ...],
    ) -> tuple[Any | None, dict[str, Any]]:
        """Run one evaluator-allowed group through the ordinary force backend."""

        robot_lock = _robot_lock_snapshot(task.env)
        try:
            execution = execute_open_articulation_group(
                task.env,
                object_name=str(source_name),
                joints=selected_joints,
                max_physics_substeps=int(config.force_max_internal_steps),
                success_fraction=float(SUCCESS_OPEN_FRACTION),
                robot_lock_callback=lambda: _apply_robot_lock(task.env, robot_lock),
                public_command=request.public_command,
            )
            return execution, dict(execution.private_metadata)
        except Exception as exc:
            # The exception type is evaluator-private diagnostic metadata.  The
            # ROS method receives only the public failure contract below.
            return None, {
                "execution_mode": "ordinary_force_group_fast",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            _apply_robot_lock(task.env, robot_lock)

    if access_ok:
        is_drawer_scan = bool(
            request.sequence_type == "drawer_scan"
            and source_name is not None
            and joints
        )
        if is_drawer_scan:
            # The method requested the public drawer-scan skill.  Its live slide
            # joints and visually grounded regions determine capability; the
            # frozen oracle recipe is used only after execution for scoring.
            allowed_drawer_joint_indices = {
                int(joint.joint_index) for joint in joints
            }
            scan = _execute_private_drawer_scan(
                task=task,
                runtime=runtime,
                episode=episode,
                joints=joints,
                open_regions=request.open_regions,
                allowed_joint_indices=allowed_drawer_joint_indices,
                # Match the ordinary interaction bridge: a public container box
                # selects the object, but only M1-grounded visible regions may
                # select drawers.  Never enumerate hidden slide joints.
                fallback_to_all=False,
                config=config,
                decision_index=decision_index,
                frames=frames,
            )
            joint_results = tuple(scan["joint_results"])
            opened_joints = tuple(scan["opened_joints"])
            successful_ids = _successful_drawer_scan_interaction_ids(
                episode=episode,
                source_name=str(source_name),
                opened_joints=opened_joints,
            )
            transient_satisfied_ids = list(successful_ids)
            skill_completed = bool(scan["success"])
            # The macro itself only returns discovery after cleanup, but retain
            # this boundary check so a future executor cannot accidentally turn
            # a failed physical interaction into a target task success.
            target_discovery = scan.get("target_discovery") if skill_completed else None
            simulated_seconds = float(scan["simulated_seconds"])
            postcondition = "drawer_scan_satisfied" if skill_completed else "drawer_scan_failed"
            executor_metadata = dict(scan.get("metadata") or {})
            executor_name = "trusted_drawer_scan"
            public_outcome = {
                "state": "closed" if skill_completed else "unknown",
                "pre_state": "closed",
                "post_state": "closed" if skill_completed else "unknown",
                "interaction_capability": (
                    "articulated"
                ),
                "interactable": True,
                "retryable": False,
                "failure_reason": (
                    "" if skill_completed else "drawer_scan_execution_failed"
                ),
                "verification_source": "drawer_scan_backend",
                "interaction_pose_validation": pose_validation,
                "execution_cost": 1.0,
            }
            completion = runtime.adapter.complete_interaction(
                request.command_id,
                success=skill_completed,
                outcome=public_outcome,
            )
            public_result = {**public_request, **completion}
            observation = scan["observation"]
        elif request.sequence_type == "drawer_open":
            allowed_drawer_joint_indices = {
                int(joint.joint_index) for joint in joints
            }
            drawer_open = _execute_private_drawer_open(
                task=task,
                runtime=runtime,
                joints=joints,
                open_regions=request.open_regions,
                allowed_joint_indices=allowed_drawer_joint_indices,
                execute_group=execute_allowed_group,
                config=config,
                decision_index=decision_index,
                frames=frames,
            )
            selected_joints = tuple(drawer_open["selected_joints"])
            joint_results = tuple(drawer_open["joint_results"])
            successful_ids = _successful_object_skill_interaction_ids(
                episode=episode,
                source_name=str(source_name),
                joints=selected_joints,
                joint_results=joint_results,
            )
            physical_completed = bool(drawer_open["success"])
            # Attempt success is the physical group postcondition.  Hidden
            # recipe relevance/order is scored separately from successful IDs.
            skill_completed = physical_completed
            simulated_seconds = float(drawer_open["simulated_seconds"])
            physics_substeps = int(drawer_open["physics_substeps"])
            pre_state = str(drawer_open["pre_state"])
            executor_metadata = dict(drawer_open["metadata"])
            executor_name = "ordinary_drawer_open_fast_sequence"
            postcondition = (
                "drawer_open_satisfied"
                if physical_completed
                else "drawer_open_failed"
            )
            completion = runtime.adapter.complete_interaction(
                request.command_id,
                success=physical_completed,
                outcome={
                    "state": "open" if physical_completed else "blocked",
                    "pre_state": pre_state,
                    "post_state": "open" if physical_completed else "blocked",
                    "interaction_capability": (
                        "articulated" if physical_completed else "blocked"
                    ),
                    "interactable": bool(physical_completed),
                    "retryable": False,
                    "failure_reason": (
                        "" if physical_completed else "drawer_open_execution_failed"
                    ),
                    "verification_source": "executor_state_verification",
                    "interaction_pose_validation": pose_validation,
                    "physics_substeps": physics_substeps,
                    "task_steps_consumed": len(selected_joints),
                    "execution_cost": 1.0,
                },
            )
            public_result = {**public_request, **completion}
            observation = drawer_open["observation"]
        else:
            execution, executor_metadata = execute_allowed_group(joints)
            joint_results = (
                () if execution is None else tuple(execution.joint_results)
            )
            successful_ids = _successful_object_skill_interaction_ids(
                episode=episode,
                source_name=str(source_name),
                joints=joints,
                joint_results=joint_results,
            )
            physical_completed = bool(
                execution is not None and execution.success
            )
            # Attempt success is the physical group postcondition.  Hidden
            # recipe relevance/order is scored separately from successful IDs.
            skill_completed = physical_completed
            public_outcome = _public_force_execution_outcome(
                execution=execution,
                executor_metadata=executor_metadata,
                pose_validation=pose_validation,
            )
            completion = runtime.adapter.complete_interaction(
                request.command_id,
                success=physical_completed,
                outcome=public_outcome,
            )
            public_result = {**public_request, **completion}
            simulated_seconds = (
                0.0 if execution is None else float(execution.simulated_seconds)
            )
            postcondition = (
                "satisfied" if physical_completed else "not_satisfied"
            )
            executor_name = "ordinary_force_group_fast"
    else:
        access_reason = str(access_meta.get("reason") or "")
        # Keep the ordinary public precondition taxonomy intact.  In
        # particular, missing/stale public evidence and excessive public-box
        # distance are not yaw/pose errors and drive different retry behavior.
        public_access_reason = access_reason or "interaction_unsupported"
        completion = runtime.adapter.complete_interaction(
            request.command_id,
            success=False,
            outcome={
                "state": "unknown",
                "interaction_capability": (
                    "articulated" if source_name is not None else "unavailable"
                ),
                "interactable": source_name is not None,
                "retryable": source_name is not None,
                "failure_reason": public_access_reason,
                "verification_source": "executor_pose_precondition",
                "interaction_pose_validation": pose_validation,
                "execution_cost": 0.0,
            },
        )
        public_result = {**public_request, **completion}
        simulated_seconds = 0.0
        skill_completed = False
        postcondition = None

    matching_ids = [
        str(row["interaction_id"])
        for row in episode["interactive_nav"].get("interactions", [])
        if source_name is not None and str(row.get("object_name")) == source_name
    ]
    classification = "required_valid" if matching_ids and access_ok else "extra_valid" if access_ok else "invalid"
    prior_success = {
        interaction_id
        for attempt in private_attempts
        if attempt.get("classification") == "required_valid" and bool(attempt.get("success"))
        for interaction_id in _resolved_interaction_ids(attempt)
    }
    prerequisite_satisfied: bool | None = None
    if matching_ids:
        completed_for_attempt = set(prior_success)
        prerequisite_satisfied = True
        rows_by_id = {
            str(row["interaction_id"]): row
            for row in episode["interactive_nav"].get("interactions", [])
        }
        for interaction_id in successful_ids:
            prerequisites = {
                str(prerequisite["interaction_id"])
                for prerequisite in rows_by_id.get(interaction_id, {}).get(
                    "prerequisites", []
                )
            }
            if not prerequisites.issubset(completed_for_attempt):
                prerequisite_satisfied = False
            completed_for_attempt.add(interaction_id)
    private_attempt = InteractionAttempt(
        requested=public_request,
        classification=classification,  # type: ignore[arg-type]
        resolved_object_name=source_name,
        resolved_joint_name=None,
        resolved_joint_index=None,
        resolved_interaction_id=successful_ids[0] if successful_ids else None,
        resolved_interaction_ids=successful_ids,
        success=skill_completed,
        joint_fraction_before=None,
        joint_fraction_after=None,
        prerequisite_satisfied=prerequisite_satisfied,
        executor=executor_name,
        simulated_seconds=simulated_seconds,
        metadata={
            # This private trace is retained only for formal V3 scoring and is
            # never handed to the ROS method.
            "access": access_meta,
            "postcondition": postcondition,
            "joint_result_count": len(joint_results),
            # The task-level acknowledgement may remain false while another
            # required effect is pending.  Preserve physical effects so the
            # paper's V/E accounting remains per attempted interaction.
            "effect_achieved_interaction_ids": successful_ids,
            # Retain the matched task-relevant IDs even when approach gating
            # prevents force execution.  This private detail lets the paper
            # breakdown label such an attempt as failed rather than unrelated.
            "requested_interaction_ids": matching_ids,
            "transient_satisfied_interaction_ids": transient_satisfied_ids,
            "drawer_scan": executor_metadata,
        },
    ).to_dict()
    public_attempt = {
        **public_result,
        "decision_step": int(decision_index),
        "simulated_seconds": simulated_seconds,
        "result_status": str(completion.get("status") or "FAILED"),
    }
    _capture_head_frame(task, frames, config.record_video)
    _publish_restricted_ros_frame(runtime, task, decision_index=decision_index)
    _discard_task_rollout_cache(task)
    return {
        "private_attempt": private_attempt,
        "public_attempt": public_attempt,
        "observation": observation,
        "simulated_seconds": simulated_seconds,
        "target_discovery": target_discovery,
    }


def _apply_view(task: Any, action: PolicyAction) -> dict[str, Any]:
    if action.head_qpos is None and action.torso_qpos is None:
        raise ValueError("view action must specify head_qpos and/or torso_qpos")
    result: dict[str, Any] = {}
    if action.head_qpos is not None:
        result["head_qpos"] = None if not action.head_qpos else probe.set_head_joint_position(task.env, action.head_qpos).astype(float).tolist()
    if action.torso_qpos is not None:
        result["torso_qpos"] = None if not action.torso_qpos else probe.set_torso_joint_position(task.env, action.torso_qpos).astype(float).tolist()
    mujoco.mj_forward(task.env.current_model, task.env.current_data)
    task.env.camera_manager.registry.update_all_cameras(task.env)
    return result


def _execute_oracle_waypoint(task: Any, action: PolicyAction, config: BenchmarkEvaluationConfig) -> tuple[Any, float, bool, dict[str, Any]]:
    metadata = action.metadata
    target = np.asarray(metadata["goal_point"], dtype=float)
    target_yaw = float(metadata.get("goal_yaw", 0.0))
    before = _base_pose_xyyaw(task)
    if config.oracle_navigation_mode == "task_action":
        observation, reward, terminated, truncated, infos = task.step({"base": np.asarray([target[0], target[1], target_yaw], dtype=float)})
        del reward
        terminated_flag = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_flag = bool(np.asarray(truncated).reshape(-1)[0])
        details: dict[str, Any] = {"mode": "task_action", "terminated": terminated_flag, "truncated": truncated_flag, "infos": _safe_json(infos)}
    else:
        robot = task.env.current_robot
        pose = robot.robot_view.base.pose.copy()
        pose[0, 3], pose[1, 3] = float(target[0]), float(target[1])
        c, s = math.cos(target_yaw), math.sin(target_yaw)
        pose[:2, :2] = np.asarray([[c, -s], [s, c]], dtype=float)
        robot.update_control({})
        robot.robot_view.base.pose = pose
        robot.robot_view.base.joint_vel = np.zeros_like(robot.robot_view.base.joint_vel)
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        observation = task.get_observations()
        details = {"mode": "direct_pose"}
    after = _base_pose_xyyaw(task)
    distance = float(np.linalg.norm(after[:2] - target[:2]))
    yaw_error = abs(_wrap_angle(float(after[2]) - target_yaw))
    reached = distance <= float(metadata.get("position_tolerance_m", 0.25)) and yaw_error <= float(metadata.get("yaw_tolerance_rad", 0.35))
    details.update({"goal_point": target.tolist(), "goal_yaw": target_yaw, "distance_error_m": distance, "yaw_error_rad": yaw_error, "reached": reached})
    return observation, float(np.linalg.norm(after[:2] - before[:2])), reached, details


def _task_base_step(task: Any, action: PolicyAction) -> tuple[Any, float, bool, dict[str, Any]]:
    if action.base_action is None:
        raise ValueError("base action is missing base_action")
    before = _base_pose_xyyaw(task)
    observation, reward, terminated, truncated, infos = task.step(action.base_action)
    after = _base_pose_xyyaw(task)
    return observation, float(np.linalg.norm(after[:2] - before[:2])), bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0]), {
        "reward": float(np.asarray(reward).reshape(-1)[0]),
        "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
        "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
        "infos": _safe_json(infos),
        "base_pose_xyyaw": after.tolist(),
    }


def _execute_evaluator_base_action(
    task: Any,
    action: PolicyAction,
    config: BenchmarkEvaluationConfig,
    policy: Any,
) -> tuple[Any, float, bool, dict[str, Any], str]:
    """Execute one base action and return an explicit task-end flag.

    Direct-pose oracle waypoints move without ``task.step`` and are explicitly
    nonterminal.  The compatibility ``task_action`` mode preserves the native
    terminated/truncated flags returned by the task.
    """

    if action.metadata.get("oracle_waypoint"):
        observation, increment, reached, details = _execute_oracle_waypoint(
            task,
            action,
            config,
        )
        notify = getattr(policy, "notify_action_result", None)
        if callable(notify):
            notify(action, reached=reached)
        task_ended = bool(details.get("terminated") or details.get("truncated"))
        return observation, increment, task_ended, details, "oracle_navigation"
    observation, increment, task_ended, details = _task_base_step(task, action)
    return observation, increment, task_ended, details, "base"


def _native_base_terminal_reason(
    *,
    task_ended: bool,
    details: Mapping[str, Any],
    applied_action_step_count: int,
    effective_max_steps: int,
) -> str | None:
    """Map native task flags without hiding final-frame goal-status claims."""

    if not task_ended:
        return None
    if bool(details.get("terminated")):
        return "native_task_terminated"
    if bool(details.get("truncated")):
        if applied_action_step_count + 1 >= effective_max_steps:
            return "max_steps"
        return "native_task_truncated"
    return "native_task_terminated"


def _recipe(episode: dict[str, Any]) -> str | None:
    nav = episode["interactive_nav"]
    return nav.get("legacy_case_type") or nav.get("generation_validation", {}).get("legacy_case_type")


def _episode_result_base(episode_index: int, episode: dict[str, Any], policy_name: str, uses_oracle_gt: bool) -> dict[str, Any]:
    nav = episode["interactive_nav"]
    return {
        "episode_index": int(episode_index),
        "case_id": str(nav["case_id"]),
        "house_index": int(episode["house_index"]),
        "domains": list(nav["interaction_domains"]),
        "recipe": _recipe(episode),
        "interaction_types": sorted({str(row["type"]) for row in nav.get("interactions", [])}),
        "path_length_bin": path_length_bin(reference_path_length_m(episode)),
        "interaction_requirement": str(nav["interaction_requirement"]),
        "policy_name": policy_name,
        "uses_oracle_gt": uses_oracle_gt,
    }


@dataclass
class _PhaseTimings:
    samples_ms: dict[str, list[float]] = field(default_factory=dict)

    def record(self, name: str, started: float) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.samples_ms.setdefault(name, []).append(elapsed_ms)
        return elapsed_ms

    def summary(self) -> dict[str, Any]:
        phases: dict[str, Any] = {}
        for name, values in self.samples_ms.items():
            samples = np.asarray(values, dtype=float)
            phases[name] = {
                "count": int(samples.size),
                "mean": float(np.mean(samples)),
                "p50": float(np.percentile(samples, 50)),
                "p95": float(np.percentile(samples, 95)),
                "max": float(np.max(samples)),
                "total": float(np.sum(samples)),
            }
        return {"unit": "ms", "phases": phases}


def _private_episode_visualization_context(
    *,
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
) -> dict[str, Any]:
    """Capture post-evaluation-only GT geometry without exposing it to a policy.

    This is deliberately a sidecar, not an ``EpisodeResult`` field: restricted
    ROS policies must never receive target or internal articulated-object names
    through their public V3 trace while an episode is being evaluated.
    """

    nav = episode["interactive_nav"]
    target_names = target_candidate_names(episode)
    objects = task.env.object_managers[task.env.current_batch_index]
    target_positions = {
        target_name: np.asarray(
            objects.get_object_by_name(target_name).position[:2], dtype=float
        ).tolist()
        for target_name in target_names
    }
    interactions: list[dict[str, Any]] = []
    for row in nav.get("interactions", []):
        runtime_joint = catalog.by_name(str(row["object_name"]), int(row["joint_index"]))
        entry: dict[str, Any] = {
            "interaction_id": str(row["interaction_id"]),
            "type": str(row.get("type", "interaction")),
        }
        if runtime_joint is not None:
            entry.update(
                {
                    "xy": runtime_joint.aabb_center[:2].astype(float).tolist(),
                    "aabb_center": runtime_joint.aabb_center.astype(float).tolist(),
                    "aabb_size": runtime_joint.aabb_size.astype(float).tolist(),
                }
            )
        interactions.append(entry)
    return {
        "schema_version": "interactive_nav_v3_evaluator_private_visualization_v1",
        "evaluator_private": True,
        "case_id": str(nav["case_id"]),
        "target": {
            "xy": target_positions[target_names[0]],
            "candidate_xy": target_positions,
        },
        "gt_interactions": interactions,
        "actual_interactions": [],
    }


def _private_actual_interaction_markers(
    attempts: list[dict[str, Any]],
    catalog: InteractionCatalog,
) -> list[dict[str, Any]]:
    """Project resolved evaluator attempts to object locations for the sidecar."""

    markers: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str | None]] = set()
    for attempt in attempts:
        object_name = attempt.get("resolved_object_name")
        if object_name is None:
            continue
        joint_index = attempt.get("resolved_joint_index")
        joints = [joint for joint in catalog.joints if joint.object_name == str(object_name)]
        if joint_index is not None:
            joints = [joint for joint in joints if joint.joint_index == int(joint_index)]
        for joint in joints:
            request = attempt.get("requested")
            request = request if isinstance(request, dict) else {}
            request_id = request.get("request_id")
            key = (joint.object_name, joint.joint_index, None if request_id is None else str(request_id))
            if key in seen:
                continue
            seen.add(key)
            markers.append(
                {
                    "request_id": request_id,
                    "xy": joint.aabb_center[:2].astype(float).tolist(),
                    "aabb_center": joint.aabb_center.astype(float).tolist(),
                    "aabb_size": joint.aabb_size.astype(float).tolist(),
                    "success": bool(attempt.get("success")),
                    "classification": attempt.get("classification"),
                }
            )
    return markers


@lru_cache(maxsize=1)
def _protocol_implementation_sha256() -> str:
    """Hash the evaluator plus all MolmoSpaces Python code it can execute.

    Hashing the full core package is intentionally conservative: replay and
    scoring depend on sampler, robot, camera, sensor, and environment modules
    transitively. This also fingerprints uncommitted edits instead of relying
    only on the current Git commit identity.
    """

    digest = hashlib.sha256()
    repo_root = Path(__file__).resolve().parents[3]
    interactive_nav_root = Path(__file__).resolve().parent.parent
    paths = [
        *sorted((interactive_nav_root / "evaluation").glob("*.py")),
        interactive_nav_root / "container_scene_probe.py",
        interactive_nav_root / "force_interaction_runtime.py",
        interactive_nav_root / "force_interaction_bridge.py",
        Path(interactive_nav_v3.__file__).resolve(),
        Path(interactive_nav_v3.SCHEMA_PATH).resolve(),
        *sorted((repo_root / "molmo_spaces").rglob("*.py")),
    ]
    for path in paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pinned_assets_signature() -> dict[str, Any] | None:
    """Return the content identity of the active asset-version override."""

    raw_path = os.environ.get("MLSPACES_PINNED_ASSETS_FILE")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned assets file does not exist: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "versions": json.loads(path.read_text(encoding="utf-8")),
    }


def _policy_factory_signature(config: BenchmarkEvaluationConfig) -> dict[str, Any] | None:
    """Fingerprint the factory's defining module for safe local resume.

    This cannot identify transitive imports or model/checkpoint contents, which
    remain caller-owned, but it prevents silently reusing traces after the
    directly configured factory module changes.
    """

    factory_spec = config.policy_factory
    if config.policy != "factory" or not factory_spec:
        return None
    module_name, separator, qualname = factory_spec.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError("policy factory must use 'module.path:callable' syntax")
    try:
        module_spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ImportError(f"Cannot resolve policy factory module {module_name!r}") from exc
    origin = None if module_spec is None else module_spec.origin
    if not origin or origin in {"built-in", "frozen"}:
        raise ImportError(f"Policy factory module {module_name!r} has no hashable source file")
    source_path = Path(origin).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Policy factory module source does not exist: {source_path}"
        )
    source_bytes = source_path.read_bytes()
    return {
        "factory": factory_spec,
        "module": module_name,
        "qualname": qualname,
        "module_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "module_source_size_bytes": len(source_bytes),
    }


def _run_signature(config: BenchmarkEvaluationConfig, benchmark_sha256: str, indices: list[int]) -> tuple[str, dict[str, Any]]:
    config_payload = asdict(config)
    for key in ("benchmark", "output_dir", "resume", "workers"):
        config_payload.pop(key, None)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "public_observation_schema_version": PUBLIC_OBSERVATION_SCHEMA_VERSION,
        "initial_state_replay_version": INITIAL_STATE_REPLAY_VERSION,
        "interaction_execution_mode": INTERACTION_EXECUTION_MODE,
        "paper_metric_config": _paper_metric_config(config).to_dict(),
        "protocol_implementation_sha256": _protocol_implementation_sha256(),
        "pinned_assets": _pinned_assets_signature(),
        "policy_factory_implementation": _policy_factory_signature(config),
        "benchmark_sha256": benchmark_sha256,
        "episode_indices": indices,
        "evaluation_config": _safe_json(config_payload),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _load_completed_trace(trace_path: Path, signature: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "complete" or payload.get("run_signature") != signature:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def evaluate_episode(
    config: BenchmarkEvaluationConfig,
    episode_index: int,
    episode: dict[str, Any],
    *,
    run_signature: str,
) -> dict[str, Any]:
    """Evaluate one frozen V3 episode in exactly one MuJoCo context."""

    config.output_dir = Path(config.output_dir)
    nav = episode["interactive_nav"]
    restricted_public_mode = _is_ros_object_goal_rule(config)
    case_id = str(nav["case_id"])
    episode_dir = config.output_dir / "episodes" / f"{episode_index:04d}_{case_id[:96]}"
    trace_path = episode_dir / "episode_result.json"
    if config.resume and trace_path.is_file():
        existing = _load_completed_trace(trace_path, run_signature)
        if existing is not None:
            return existing

    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    task = None
    sampler = None
    policy: BenchmarkPolicy | None = None
    scene_install_module: Any | None = None
    scene_install_original: Any | None = None
    attempts: list[dict[str, Any]] = []
    public_attempts: list[dict[str, Any]] = []
    restricted_ros_runtime: RestrictedRosObjectGoalRuntime | None = None
    ros_goal_status: RosGoalStatusObserver | None = None
    ros_behavior_feedback: RosBehaviorFeedbackObserver | None = None
    ros_policy_termination = (
        RosPolicyTerminationGuard(
            RosPolicyTerminationConfig(
                command_starvation_timeout_s=float(
                    config.ros_command_starvation_timeout_s
                ),
                observation_turn_multiplier=float(
                    config.ros_observation_turn_multiplier
                ),
            )
        )
        if _is_current_ros_policy(config)
        else None
    )
    stall_tracker = CrossSubgoalStallTracker(
        CrossSubgoalStallConfig(
            enabled=bool(
                _is_current_ros_policy(config) and config.ros_stall_early_stop_enabled
            ),
            min_failed_subgoals=int(config.ros_stall_min_failed_subgoals),
            max_displacement_m=float(config.ros_stall_max_displacement_m),
            min_no_progress_steps=int(config.ros_stall_min_no_progress_steps),
        )
    )
    navigation_steps = 0
    view_actions = 0
    nav_path_length = 0.0
    nav_sim_seconds = 0.0
    interaction_sim_seconds = 0.0
    applied_action_step_count = 0
    decision_index = 0
    terminal_reason = "max_steps"
    runtime_goal_consistency: dict[str, Any] | None = None
    runtime_consistency: dict[str, Any] | None = None
    scoring_exclusion_reasons: list[str] = []
    scoring_eligible = True
    runtime_goal_blocked = False
    catalog: InteractionCatalog | None = None
    private_visualization: dict[str, Any] | None = None
    accepted_goal_claim = False
    last_goal_claim_verification: GoalClaimVerification | None = None
    effective_max_steps = int(config.max_steps)
    step_budget_basis: dict[str, Any] = {
        "evaluator_private": True,
        "formula_version": STEP_BUDGET_FORMULA_VERSION,
        "mode": config.step_budget_mode,
        "effective_max_steps": effective_max_steps,
        "conservative_fallback": True,
        "fallback_reason": "budget_computation_not_completed",
    }
    reference: float | None = None
    phase_timings = _PhaseTimings()
    started = time.monotonic()
    base_data = _episode_result_base(episode_index, episode, config.policy, config.policy == "scripted_oracle")
    paper_metric_config = _paper_metric_config(config)
    paper_metric_config_payload = paper_metric_config.to_dict()
    try:
        phase_started = time.perf_counter()
        # Older frozen V3 benchmark releases omitted the empty articulation
        # state list from ``scene_modifications``.  The current schema keeps
        # the field required so downstream replay can rely on a stable shape;
        # normalize the legacy payload in memory without changing the frozen
        # benchmark artifact.
        episode.setdefault("scene_modifications", {}).setdefault(
            "articulation_states", []
        )
        interactive_nav_v3.validate_interactive_nav_v3_episode(episode, expected_domains=list(nav["interaction_domains"]))
        phase_timings.record("episode_validation", phase_started)
        spec = EpisodeSpec.model_validate(episode)
        effective_max_steps, step_budget_basis = episode_step_budget(config, episode)
        print(
            "[episode-step-budget] "
            f"episode={episode_index} mode={config.step_budget_mode} "
            f"effective={effective_max_steps} cap={config.max_steps} "
            f"fallback={bool(step_budget_basis.get('conservative_fallback', False))}",
            flush=True,
        )
        original_cameras = [camera.name for camera in spec.cameras]
        spec.cameras = [camera for camera in spec.cameras if camera.name in set(config.camera_names)]
        missing_cameras = sorted(set(config.camera_names) - {camera.name for camera in spec.cameras})
        if missing_cameras:
            raise ValueError(f"Requested unavailable camera(s): {missing_cameras}; available={original_cameras}")
        if config.image_resolution is not None:
            spec.img_resolution = tuple(int(item) for item in config.image_resolution)
        # The frozen light benchmark stores head RGB only.  The current ROS
        # navigation stack consumes an RGB-D point cloud, so enable the paired
        # live depth sensor locally without mutating the benchmark JSON.
        if _is_current_ros_policy(config):
            for camera in spec.cameras:
                if camera.name == "head_camera":
                    camera.record_depth = True
            # The stored V3 default was selected for manipulation and puts both
            # arms in the navigation camera.  This evaluator-only reset posture
            # is the established ROS navigation configuration; the frozen JSON
            # on disk remains unchanged.
            _apply_ros_navigation_arm_posture(spec)
        replay = _build_replay_config(config, episode_dir, task_horizon=effective_max_steps)
        sampler = V3BenchmarkTaskSampler(replay, spec, nav)
        # Collection and this evaluator use a writable scene mirror.  Do not let
        # the upstream sampler attempt to mutate the shared resource cache.
        import molmo_spaces.tasks.task_sampler as task_sampler_module

        scene_install_module = task_sampler_module
        scene_install_original = task_sampler_module.install_scene_with_objects_and_grasps_from_path
        task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *args, **kwargs: None
        # ``_get_dataset_index_map`` is a process-global cache.  Install the
        # writable mirror only while constructing this task, then restore the
        # cached source path so a later episode in the same house can replay.
        variants = sampler._get_dataset_index_map()[spec.data_split][spec.house_index]
        source_scene_path = variants["base"]
        variants["base"] = probe.prepare_writable_scene_path(Path(source_scene_path))
        try:
            phase_started = time.perf_counter()
            # Current BaseMujocoTaskSampler selects the base scene variant
            # internally; older releases exposed a variant keyword here.
            task = sampler.sample_task(house_index=spec.house_index)
            phase_timings.record("task_sample", phase_started)
        finally:
            variants["base"] = source_scene_path
        if task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task")
        phase_started = time.perf_counter()
        observation, _info = task.reset()
        phase_timings.record("task_reset", phase_started)
        # JsonEvalTaskSampler's nav-to-object path intentionally skips its
        # manipulation-only joint initializer. Reapply the frozen object and
        # articulation state explicitly before runtime consistency checks. A
        # released benchmark can contain decorative object poses from an older
        # asset revision; the sampler already validated that task-critical
        # bodies exist, so drop only non-critical poses absent from this runtime
        # scene instead of turning an otherwise scoreable episode into a setup
        # exception. The frozen JSON is never modified on disk.
        runtime_body_names = {
            mujoco.mj_id2name(task.env.current_model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(task.env.current_model.nbody)
        }
        runtime_body_names.discard(None)
        critical_pose_names = set(target_candidate_names(episode))
        critical_pose_names.update(
            str(row["object_name"])
            for row in nav.get("interactions", [])
            if isinstance(row, Mapping) and row.get("object_name")
        )
        modifications = episode.get("scene_modifications") or {}
        recorded_poses = modifications.get("object_poses") or {}
        dropped_pose_names = sorted(
            str(name)
            for name in recorded_poses
            if str(name) not in runtime_body_names
        )
        missing_critical = sorted(set(dropped_pose_names) & critical_pose_names)
        if missing_critical:
            raise RuntimeError(
                "Runtime scene is missing task-critical object pose bodies: "
                + ", ".join(missing_critical)
            )
        if dropped_pose_names:
            episode["scene_modifications"] = dict(modifications)
            episode["scene_modifications"]["object_poses"] = {
                name: pose
                for name, pose in recorded_poses.items()
                if str(name) not in set(dropped_pose_names)
            }
        probe.apply_episode_scene_state(task.env, episode)
        for object_manager in task.env.object_managers:
            object_manager.invalidate_data_cache()
        task.env.camera_manager.registry.update_all_cameras(task.env)
        observation = task.get_observations()
        catalog = InteractionCatalog(task.env)
        private_visualization = _private_episode_visualization_context(
            task=task,
            catalog=catalog,
            episode=episode,
        )
        private_visualization["step_budget"] = step_budget_basis
        phase_started = time.perf_counter()
        runtime_consistency = _runtime_consistency_checks(task, catalog, episode, sampler, config)
        phase_timings.record("runtime_consistency", phase_started)
        runtime_goal_consistency = runtime_consistency["checks"]["oracle_terminal_goal"]
        scoring_exclusion_reasons = list(runtime_consistency["exclusion_reasons"])
        scoring_eligible = bool(runtime_consistency["eligible"])
        runtime_goal_blocked = bool(
            config.require_runtime_goal_consistency
            and not scoring_eligible
        )
        public = _public_episode(episode, [camera.name for camera in spec.cameras], tuple(spec.img_resolution))
        if not runtime_goal_blocked and not config.quality_gate_only:
            phase_started = time.perf_counter()
            policy = _build_policy(config, public)
            policy.reset(public)
            if isinstance(policy, ScriptedOraclePolicy):
                policy.reset_oracle(list(nav["oracle_plan"]["steps"]))
            if _is_ros_object_goal_rule(config):
                ros_goal_status = RosGoalStatusObserver(config.ros_goal_status_topic)
                published_frame_sink = getattr(
                    policy,
                    "queue_step_frame_public_payload",
                    None,
                )
                if not callable(published_frame_sink):
                    published_frame_sink = None
                restricted_ros_runtime = _build_restricted_ros_object_goal_runtime(
                    task=task,
                    catalog=catalog,
                    episode=episode,
                    public=public,
                    config=config,
                    episode_index=episode_index,
                    frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                    goal_status_observer=ros_goal_status,
                    published_frame_sink=published_frame_sink,
                )
            if stall_tracker.config.enabled:
                try:
                    ros_behavior_feedback = RosBehaviorFeedbackObserver(
                        config.ros_behavior_feedback_topic
                    )
                    ros_behavior_feedback.begin_episode()
                    stall_tracker.observe_pose(_base_pose_xyyaw(task)[:2], 0)
                except Exception as exc:
                    # A missing optional observer must not turn an otherwise
                    # valid benchmark episode into an exception.  Its absence
                    # remains visible in the evaluator diagnostics.
                    stall_tracker.disable(f"{type(exc).__name__}: {exc}")
            phase_timings.record("policy_setup", phase_started)
            base_data["policy_name"] = str(getattr(policy, "name", config.policy))
            base_data["uses_oracle_gt"] = bool(getattr(policy, "uses_oracle_gt", False))
        public_runtime_goal_consistency = runtime_goal_consistency
        public_runtime_consistency = runtime_consistency
        if restricted_public_mode:
            public_runtime_goal_consistency, public_runtime_consistency = _redact_runtime_consistency_for_restricted_policy(
                runtime_consistency
            )
        public_runtime_compatibility = sampler.runtime_compatibility
        if restricted_public_mode:
            public_runtime_compatibility = {
                key: value
                for key, value in sampler.runtime_compatibility.items()
                if key != "dropped_noncritical_object_pose_names"
            }
        runtime_trace: dict[str, Any] = {
            "runtime_compatibility": public_runtime_compatibility,
            "oracle_terminal_goal_consistency": public_runtime_goal_consistency,
            "runtime_consistency": public_runtime_consistency,
            "runtime_cameras": [camera.name for camera in spec.cameras],
            "runtime_depth_cameras": [camera.name for camera in spec.cameras if camera.record_depth],
            "runtime_image_resolution": list(spec.img_resolution),
            "benchmark_image_resolution": list(episode.get("img_resolution", [])),
            "interaction_catalog_joint_count": len(catalog.joints),
            "step_budget": _public_step_budget_basis(step_budget_basis),
        }
        if ros_policy_termination is not None:
            runtime_trace["ros_step_accounting"] = {
                "observation_turn_unit": "published_policy_observation",
                "applied_step_unit": "evaluator_applied_action",
                "no_fresh_consumes_applied_step": False,
                "command_starvation_timeout_s": float(
                    config.ros_command_starvation_timeout_s
                ),
                "observation_turn_limit": (
                    ros_policy_termination.observation_turn_limit(
                        effective_max_steps
                    )
                ),
                "final_goal_status_drain_timeout_s": float(
                    config.ros_final_goal_status_drain_timeout_s
                ),
            }
        if _is_current_ros_policy(config):
            runtime_trace["effective_navigation_arm_qpos"] = {
                name: list(spec.robot.init_qpos[name])
                for name in ROS_NAVIGATION_ARM_QPOS
                if name in spec.robot.init_qpos
            }
        if restricted_public_mode:
            runtime_trace["restricted_gt_protocol"] = {
                "enabled": True,
                "camera_name": "head_camera",
                "minimum_visible_pixels": int(config.restricted_gt_min_visible_pixels),
                "minimum_bbox_area_pixels": int(config.restricted_gt_min_bbox_area_pixels),
                "minimum_visible_fraction": float(config.restricted_gt_min_visible_fraction),
                "maximum_distance_m": float(config.restricted_gt_max_distance_m),
                "interaction_endpoint": "opaque_object_open",
            }
        trace.append(
            {
                "runtime": runtime_trace
            }
        )
        _capture_head_frame(task, frames, config.record_video)
        previous_action: dict[str, Any] | None = None
        transient_target_discovery: dict[str, Any] | None = None

        if runtime_goal_blocked:
            terminal_reason = "runtime_consistency_ineligible"
        elif config.quality_gate_only:
            terminal_reason = "quality_gate_complete"
        while (
            not runtime_goal_blocked
            and not config.quality_gate_only
            and applied_action_step_count < effective_max_steps
        ):
            decision_timing: dict[str, float] = {}
            decision_started = time.perf_counter()
            phase_started = decision_started
            goal_terminal = None
            if ros_goal_status is not None and restricted_ros_runtime is not None:
                goal_terminal = _poll_restricted_goal_status(
                    observer=ros_goal_status,
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                )
            compatibility_target_found = _compatibility_target_precheck(
                task,
                episode,
                restricted_public_mode=restricted_public_mode,
                transient_target_discovery=transient_target_discovery,
                attempts=attempts,
            )
            decision_timing["precheck"] = phase_timings.record("step_precheck", phase_started)
            if goal_terminal is not None:
                terminal_reason, verification, goal_payload = goal_terminal
                if verification is not None:
                    last_goal_claim_verification = verification
                    accepted_goal_claim = bool(verification.accepted)
                if ros_policy_termination is not None and (
                    verification is None or bool(verification.accepted)
                ):
                    ros_policy_termination.observe_goal_status(goal_payload)
                phase_timings.record("step_total", decision_started)
                break
            if compatibility_target_found:
                terminal_reason = "target_found"
                phase_timings.record("step_total", decision_started)
                break
            if ros_behavior_feedback is not None:
                phase_started = time.perf_counter()
                base_pose = _base_pose_xyyaw(task)
                stall_tracker.observe_pose(base_pose[:2], decision_index)
                stalled = False
                for feedback in ros_behavior_feedback.drain():
                    stalled = stall_tracker.note_feedback(
                        feedback,
                        base_pose[:2],
                        decision_index,
                    ) or stalled
                decision_timing["ros_stall_guard"] = phase_timings.record(
                    "ros_stall_guard", phase_started
                )
                if stalled:
                    terminal_reason = CROSS_SUBGOAL_STALL_REASON
                    decision_timing["total"] = phase_timings.record(
                        "step_total", decision_started
                    )
                    # This guard fires before ``policy.act``.  It therefore
                    # has no corresponding public observation, action, or
                    # RosBridge step-sync marker, and must not inflate the
                    # counted evaluator steps beyond the recordable video.
                    # The trigger remains in ``EpisodeResult.early_stop``
                    # and the terminal trace below.
                    break
            if restricted_ros_runtime is not None:
                phase_started = time.perf_counter()
                consumed = _consume_pending_ros_object_goal_interaction(
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                    private_attempts=attempts,
                    config=config,
                    decision_index=decision_index,
                    frames=frames,
                )
                decision_timing["restricted_interaction_poll"] = phase_timings.record(
                    "restricted_interaction_poll", phase_started
                )
                if consumed is not None:
                    if ros_policy_termination is not None:
                        ros_policy_termination.note_progress()
                    applied_action_step_count += 1
                    attempts.append(consumed["private_attempt"])
                    public_attempts.append(consumed["public_attempt"])
                    interaction_sim_seconds += float(consumed["simulated_seconds"])
                    observation = consumed["observation"]
                    previous_action = {
                        "kind": "interact",
                        **consumed["public_attempt"],
                    }
                    decision_timing["total"] = phase_timings.record(
                        "step_total", decision_started
                    )
                    trace.append(
                        {
                            "decision_step": decision_index,
                            "interaction": consumed["public_attempt"],
                            "timing_ms": decision_timing,
                        }
                    )
                    if consumed.get("target_discovery") is not None:
                        transient_target_discovery = dict(consumed["target_discovery"])
                    # A transient private drawer-open observation is diagnostic
                    # only.  Completion still requires a policy goal-status
                    # declaration grounded in the restricted frame published
                    # during the scan.
                    decision_index += 1
                    continue
            # Drain terminal/status and evaluator-side interaction feedback
            # before enforcing the non-applied observation-turn cap. A status
            # or interaction result can arrive asynchronously during the last
            # allowed turn; it is real progress and must not be discarded by
            # the liveness guard before policy.act() is called again.
            if ros_policy_termination is not None:
                observation_limit_terminal = (
                    ros_policy_termination.observe_observation_turn_count(
                        decision_index,
                        applied_step_budget=effective_max_steps,
                    )
                )
                if observation_limit_terminal is not None:
                    terminal_reason = observation_limit_terminal.reason
                    break
            policy_observation = PolicyObservation(
                observation=_public_policy_observation(observation, public),
                instruction=public.instruction,
                step_index=decision_index,
                elapsed_seconds=time.monotonic() - started,
                previous_action=previous_action,
            )
            policy_wait_started_mono_s = time.monotonic()
            phase_started = time.perf_counter()
            action = policy.act(policy_observation)
            decision_timing["policy_act"] = phase_timings.record("policy_act", phase_started)
            policy_wait_finished_mono_s = time.monotonic()
            pending_policy_terminal = None
            event: dict[str, Any] = {
                "decision_step": decision_index,
                "action": action.to_dict(),
                "timing_ms": decision_timing,
            }
            if ros_goal_status is not None and restricted_ros_runtime is not None:
                goal_terminal = _poll_restricted_goal_status(
                    observer=ros_goal_status,
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                )
                if goal_terminal is not None:
                    terminal_reason, verification, goal_payload = goal_terminal
                    if verification is not None:
                        last_goal_claim_verification = verification
                        accepted_goal_claim = bool(verification.accepted)
                    if ros_policy_termination is not None and (
                        verification is None or bool(verification.accepted)
                    ):
                        ros_policy_termination.observe_goal_status(goal_payload)
                    decision_timing["total"] = phase_timings.record(
                        "step_total", decision_started
                    )
                    trace.append(event)
                    decision_index += 1
                    break
            if ros_policy_termination is not None:
                pending_policy_terminal = ros_policy_termination.observe_action(
                    action,
                    wait_started_mono_s=policy_wait_started_mono_s,
                    now_mono_s=policy_wait_finished_mono_s,
                )
            if restricted_ros_runtime is not None:
                phase_started = time.perf_counter()
                consumed = _consume_pending_ros_object_goal_interaction(
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                    private_attempts=attempts,
                    config=config,
                    decision_index=decision_index,
                    frames=frames,
                )
                decision_timing["restricted_interaction_poll_after_act"] = phase_timings.record(
                    "restricted_interaction_poll_after_act", phase_started
                )
                if consumed is not None:
                    if ros_policy_termination is not None:
                        # A side-channel interaction command is real policy
                        # progress even when the velocity bridge timed out on
                        # the same observation turn.
                        ros_policy_termination.note_progress()
                    pending_policy_terminal = None
                    applied_action_step_count += 1
                    attempts.append(consumed["private_attempt"])
                    public_attempts.append(consumed["public_attempt"])
                    interaction_sim_seconds += float(consumed["simulated_seconds"])
                    observation = consumed["observation"]
                    previous_action = {
                        "kind": "interact",
                        **consumed["public_attempt"],
                    }
                    decision_timing["total"] = phase_timings.record(
                        "step_total", decision_started
                    )
                    trace.append(
                        {
                            "decision_step": decision_index,
                            "interaction": consumed["public_attempt"],
                            "timing_ms": decision_timing,
                        }
                    )
                    if consumed.get("target_discovery") is not None:
                        transient_target_discovery = dict(consumed["target_discovery"])
                    decision_index += 1
                    continue
            if pending_policy_terminal is not None:
                terminal_reason = pending_policy_terminal.reason
                event["policy_termination"] = {
                    "reason": pending_policy_terminal.reason,
                    "source": pending_policy_terminal.source,
                    "detail": pending_policy_terminal.detail,
                }
                decision_timing["total"] = phase_timings.record(
                    "step_total", decision_started
                )
                trace.append(event)
                decision_index += 1
                break
            if action.kind == "stop":
                terminal_reason = str(action.metadata.get("reason", "policy_stop"))
                decision_timing["total"] = phase_timings.record(
                    "step_total", decision_started
                )
                trace.append(event)
                decision_index += 1
                break
            if action.kind == "base":
                phase_started = time.perf_counter()
                (
                    observation,
                    increment,
                    task_ended,
                    details,
                    event_key,
                ) = _execute_evaluator_base_action(task, action, config, policy)
                event[event_key] = details
                native_terminal_reason = _native_base_terminal_reason(
                    task_ended=task_ended,
                    details=details,
                    applied_action_step_count=applied_action_step_count,
                    effective_max_steps=effective_max_steps,
                )
                if native_terminal_reason is not None:
                    # Horizon truncation at the evaluator budget is kept as
                    # max_steps so the last published frame receives the
                    # bounded zero-step goal-status drain below.
                    terminal_reason = native_terminal_reason
                nav_path_length += increment
                nav_sim_seconds += float(config.policy_dt_ms) / 1000.0
                navigation_steps += 1
                applied_action_step_count += 1
                decision_timing["base_action"] = phase_timings.record("base_action", phase_started)
                phase_started = time.perf_counter()
                _capture_head_frame(task, frames, config.record_video)
                decision_timing["frame_capture"] = phase_timings.record(
                    "frame_capture", phase_started
                )
                if restricted_ros_runtime is not None:
                    phase_started = time.perf_counter()
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
                    decision_timing["restricted_gt_publish"] = phase_timings.record(
                        "restricted_gt_publish", phase_started
                    )
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                decision_timing["total"] = phase_timings.record(
                    "step_total", decision_started
                )
                trace.append(event)
                decision_index += 1
                if terminal_reason in {
                    "native_task_terminated",
                    "native_task_truncated",
                    "max_steps",
                } and task_ended:
                    break
                continue
            if action.kind == "view":
                phase_started = time.perf_counter()
                view_actions += 1
                applied_action_step_count += 1
                event["view"] = _apply_view(task, action)
                observation = task.get_observations()
                decision_timing["view_action"] = phase_timings.record("view_action", phase_started)
                phase_started = time.perf_counter()
                _capture_head_frame(task, frames, config.record_video)
                decision_timing["frame_capture"] = phase_timings.record(
                    "frame_capture", phase_started
                )
                if restricted_ros_runtime is not None:
                    phase_started = time.perf_counter()
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
                    decision_timing["restricted_gt_publish"] = phase_timings.record(
                        "restricted_gt_publish", phase_started
                    )
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                decision_timing["total"] = phase_timings.record(
                    "step_total", decision_started
                )
                trace.append(event)
                decision_index += 1
                continue
            if action.kind == "observe":
                phase_started = time.perf_counter()
                event["observe"] = {"refreshed": True}
                observation = task.get_observations()
                decision_timing["observe_action"] = phase_timings.record(
                    "observe_action", phase_started
                )
                phase_started = time.perf_counter()
                _capture_head_frame(task, frames, config.record_video)
                decision_timing["frame_capture"] = phase_timings.record(
                    "frame_capture", phase_started
                )
                if restricted_ros_runtime is not None:
                    phase_started = time.perf_counter()
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
                    decision_timing["restricted_gt_publish"] = phase_timings.record(
                        "restricted_gt_publish", phase_started
                    )
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                if counts_toward_applied_action_budget(action):
                    applied_action_step_count += 1
                decision_timing["total"] = phase_timings.record(
                    "step_total", decision_started
                )
                trace.append(event)
                decision_index += 1
                continue
            if action.kind != "interact":
                raise ValueError(f"Unsupported action kind: {action.kind}")
            if restricted_ros_runtime is not None:
                raise ValueError(
                    "ros_object_goal_rule must submit opaque interaction commands on "
                    f"{config.ros_interaction_command_topic}, not a direct PolicyAction"
                )

            phase_started = time.perf_counter()
            runtime_joint, interaction_id, classification, resolver_meta = _resolve_interaction(
                env=task.env,
                episode=episode,
                catalog=catalog,
                action=action,
                uses_oracle_gt=bool(getattr(policy, "uses_oracle_gt", False)),
                allow_internal_object_names=config.allow_internal_object_names,
            )
            prereq_satisfied: bool | None = None
            before = after = None
            success = False
            executor_meta: dict[str, Any] = {}
            simulated_seconds = 0.0
            if runtime_joint is None:
                classification = "invalid"
            else:
                if interaction_id is not None:
                    interaction = next(row for row in nav.get("interactions", []) if str(row["interaction_id"]) == interaction_id)
                    prior_success = {
                        str(item["resolved_interaction_id"])
                        for item in attempts
                        if item.get("classification") == "required_valid" and bool(item.get("success")) and item.get("resolved_interaction_id")
                    }
                    prerequisites = {str(item["interaction_id"]) for item in interaction.get("prerequisites", [])}
                    prereq_satisfied = prerequisites.issubset(prior_success)
                    if interaction_id in prior_success:
                        classification = "extra_valid"
                access_ok, access_meta = _check_interaction_access(
                    task.env,
                    runtime_joint,
                    camera_name=action.camera_name,
                    max_distance_m=config.interaction_max_distance_m,
                    require_visible=config.require_interaction_visible,
                )
                resolver_meta["access"] = access_meta
                # Oracle references are grounded in a frozen plan.  They still
                # need a live body/joint, but their planned approach is trusted
                # when renderer occlusion differs slightly across asset builds.
                if bool(getattr(policy, "uses_oracle_gt", False)) and action.metadata.get("oracle_interaction_id"):
                    access_ok = True
                    resolver_meta["access"]["oracle_access_override"] = True
                if access_ok:
                    success, executor_meta, before, after, simulated_seconds = _execute_locked_force(
                        task.env,
                        runtime_joint,
                        config,
                        frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                    )
                    observation = task.get_observations()
                else:
                    classification = "invalid"
            attempt = InteractionAttempt(
                requested=action.to_dict(),
                classification=classification,  # type: ignore[arg-type]
                resolved_object_name=None if runtime_joint is None else runtime_joint.object_name,
                resolved_joint_name=None if runtime_joint is None else runtime_joint.joint_name,
                resolved_joint_index=None if runtime_joint is None else runtime_joint.joint_index,
                resolved_interaction_id=interaction_id,
                success=bool(success),
                joint_fraction_before=before,
                joint_fraction_after=after,
                prerequisite_satisfied=prereq_satisfied,
                executor=None if runtime_joint is None else "force_locked",
                simulated_seconds=float(simulated_seconds),
                metadata={"resolver": resolver_meta, "executor": executor_meta},
            ).to_dict()
            attempts.append(attempt)
            public_attempts.append(attempt)
            interaction_sim_seconds += float(simulated_seconds)
            event["interaction"] = attempt
            decision_timing["interaction_action"] = phase_timings.record(
                "interaction_action", phase_started
            )
            phase_started = time.perf_counter()
            _capture_head_frame(task, frames, config.record_video)
            decision_timing["frame_capture"] = phase_timings.record(
                "frame_capture", phase_started
            )
            _discard_task_rollout_cache(task)
            previous_action = action.to_dict()
            applied_action_step_count += 1
            decision_timing["total"] = phase_timings.record(
                "step_total", decision_started
            )
            trace.append(event)
            decision_index += 1

        if (
            ros_goal_status is not None
            and restricted_ros_runtime is not None
            and terminal_reason == "max_steps"
            and applied_action_step_count >= effective_max_steps
        ):
            final_goal_terminal = _drain_restricted_goal_status(
                observer=ros_goal_status,
                task=task,
                runtime=restricted_ros_runtime,
                episode=episode,
                timeout_s=float(config.ros_final_goal_status_drain_timeout_s),
            )
            if final_goal_terminal is not None:
                final_reason, verification, goal_payload = final_goal_terminal
                terminal_reason = final_reason
                if verification is not None:
                    last_goal_claim_verification = verification
                    accepted_goal_claim = bool(verification.accepted)
                if ros_policy_termination is not None and (
                    verification is None or bool(verification.accepted)
                ):
                    ros_policy_termination.observe_goal_status(goal_payload)

        current_nav_ok, target_distance, target_visibility = target_metrics(task, episode)
        # Raw simulator visibility remains a private diagnostic, not a terminal
        # signal.  Restricted ROS policies receive task credit only after their
        # explicit goal-status declaration passes the public-frame evidence and
        # private-distance verifier above.
        nav_ok = bool(accepted_goal_claim) if restricted_public_mode else bool(current_nav_ok)
        terminal_score = score_interactions(task.env, episode, attempts)
        requirement = str(nav["interaction_requirement"])
        task_success = bool(nav_ok)
        interaction_conditioned_success = bool(
            nav_ok
            and terminal_score.required_interaction_success
            and terminal_score.sequence_success
            and (terminal_score.non_interaction_success if requirement == "unnecessary" else True)
        )
        reference = _safe_reference_path_length_m(episode)
        correct_count = terminal_score.correct_action_count
        extra_count = sum(row.get("classification") == "extra_valid" for row in attempts)
        invalid_count = sum(row.get("classification") == "invalid" for row in attempts)
        paper_interaction_score = paper_interaction_attempt_score(episode, attempts)
        episode_total_cost, episode_total_cost_breakdown = paper_episode_total_cost(
            nav_success=nav_ok,
            navigation_path_length_m=nav_path_length,
            interaction_score=paper_interaction_score,
            config=paper_metric_config,
        )
        video_path = None
        if config.record_video and frames:
            destination = episode_dir / "head_camera.mp4"
            _save_video(frames, destination, config.video_fps)
            video_path = str(destination)
        policy_termination_snapshot = (
            ros_policy_termination.snapshot()
            if ros_policy_termination is not None
            else {}
        )
        result = EpisodeResult(
            **base_data,
            status="complete",
            # Preserve ``success`` as the historical formal V3 score while
            # exposing the visible task endpoint independently.
            success=interaction_conditioned_success,
            task_success=task_success,
            interaction_conditioned_success=interaction_conditioned_success,
            nav_success=nav_ok,
            required_interaction_success=terminal_score.required_interaction_success,
            sequence_success=terminal_score.sequence_success,
            non_interaction_success=terminal_score.non_interaction_success,
            terminal_reason=terminal_reason,
            step_count=sum("decision_step" in item for item in trace),
            navigation_step_count=navigation_steps,
            view_action_count=view_actions,
            interaction_action_count=len(attempts),
            correct_interaction_action_count=correct_count,
            extra_interaction_action_count=int(extra_count),
            invalid_interaction_action_count=int(invalid_count),
            navigation_path_length_m=nav_path_length,
            reference_path_length_m=reference,
            spl=spl(nav_ok, reference, nav_path_length),
            navigation_simulated_seconds=nav_sim_seconds,
            interaction_simulated_seconds=interaction_sim_seconds,
            total_simulated_seconds=nav_sim_seconds + interaction_sim_seconds + view_actions * float(config.policy_dt_ms) / 1000.0,
            elapsed_seconds=time.monotonic() - started,
            target_distance_m=target_distance,
            target_visibility_fraction=target_visibility,
            interaction_attempts=public_attempts if restricted_public_mode else attempts,
            applied_action_step_count=applied_action_step_count,
            no_fresh_action_count=int(
                policy_termination_snapshot.get("total_no_fresh_action_count", 0)
            ),
            policy_termination=policy_termination_snapshot,
            episode_step_budget=effective_max_steps,
            step_budget_mode=config.step_budget_mode,
            step_budget_basis=_public_step_budget_basis(step_budget_basis),
            trace_path=str(trace_path),
            video_path=video_path,
            scoring_eligible=scoring_eligible,
            scoring_exclusion_reasons=scoring_exclusion_reasons,
            runtime_goal_consistency=public_runtime_goal_consistency,
            runtime_consistency=public_runtime_consistency,
            timing_summary=phase_timings.summary(),
            early_stop=stall_tracker.snapshot(),
            paper_metric_schema_version=PAPER_METRIC_SCHEMA_VERSION,
            paper_metric_config=paper_metric_config_payload,
            valid_interaction_attempt_count=(
                paper_interaction_score.valid_interaction_attempt_count
            ),
            error_interaction_attempt_count=(
                paper_interaction_score.error_interaction_attempt_count
            ),
            task_irrelevant_interaction_attempt_count=(
                paper_interaction_score.task_irrelevant_interaction_attempt_count
            ),
            failed_interaction_attempt_count=(
                paper_interaction_score.failed_interaction_attempt_count
            ),
            repeated_interaction_attempt_count=(
                paper_interaction_score.repeated_interaction_attempt_count
            ),
            interaction_precision_episode=(
                paper_interaction_score.interaction_precision_episode
            ),
            episode_total_cost=episode_total_cost,
            episode_total_cost_breakdown=episode_total_cost_breakdown,
        ).to_dict()
        if transient_target_discovery is not None and private_visualization is not None:
            # Keep the scan-time target evidence beside other evaluator-only
            # visualization GT.  The public trace retains only the real final
            # camera metrics after the drawer is closed.
            private_visualization["transient_target_discovery"] = {
                "distance_m": float(transient_target_discovery["distance_m"]),
                "visibility_fraction": float(transient_target_discovery["visibility_fraction"]),
                "visible_pixels": int(transient_target_discovery["visible_pixels"]),
                "group_index": int(transient_target_discovery["group_index"]),
            }
        terminal_trace: dict[str, Any] = {
            "task_success": task_success,
            "interaction_conditioned_success": interaction_conditioned_success,
            "success": interaction_conditioned_success,
        }
        if not restricted_public_mode:
            terminal_trace["interaction_score"] = terminal_score.to_dict()
        else:
            # The complete V3 score remains evaluator-private.  Do not write
            # frozen interaction IDs/fractions into a trace that may be shared
            # with the method being evaluated.
            terminal_trace["interaction_score"] = {
                "required_interaction_success": terminal_score.required_interaction_success,
                "sequence_success": terminal_score.sequence_success,
                "non_interaction_success": terminal_score.non_interaction_success,
            }
        if stall_tracker.snapshot().get("triggered"):
            terminal_trace["early_stop"] = stall_tracker.snapshot()
        if policy_termination_snapshot:
            terminal_trace["policy_termination"] = policy_termination_snapshot
        trace.append({"terminal": terminal_trace})
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if reference is None:
            reference = _safe_reference_path_length_m(episode)
        public_runtime_goal_consistency = runtime_goal_consistency
        public_runtime_consistency = runtime_consistency
        if restricted_public_mode:
            # The public episode document intentionally hides evaluator-private
            # details.  Preserve the original traceback in the evaluator-owned
            # attempt directory, though, so a startup/runtime failure can be
            # diagnosed and retried instead of being indistinguishable from a
            # policy failure in a large batch.
            try:
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "restricted_runtime_exception.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except OSError:
                pass
            public_runtime_goal_consistency, public_runtime_consistency = _redact_runtime_consistency_for_restricted_policy(
                runtime_consistency
            )
            error = "restricted evaluator exception"
        paper_interaction_score = paper_interaction_attempt_score(episode, attempts)
        episode_total_cost, episode_total_cost_breakdown = paper_episode_total_cost(
            nav_success=False,
            navigation_path_length_m=nav_path_length,
            interaction_score=paper_interaction_score,
            config=paper_metric_config,
        )
        policy_termination_snapshot = (
            ros_policy_termination.snapshot()
            if ros_policy_termination is not None
            else {}
        )
        result = EpisodeResult(
            **base_data,
            status="exception",
            success=False,
            task_success=False,
            interaction_conditioned_success=False,
            nav_success=False,
            required_interaction_success=False,
            sequence_success=False,
            non_interaction_success=None,
            terminal_reason="exception",
            step_count=sum("decision_step" in item for item in trace),
            navigation_step_count=navigation_steps,
            view_action_count=view_actions,
            interaction_action_count=len(attempts),
            correct_interaction_action_count=0,
            extra_interaction_action_count=sum(row.get("classification") == "extra_valid" for row in attempts),
            invalid_interaction_action_count=sum(row.get("classification") == "invalid" for row in attempts),
            navigation_path_length_m=nav_path_length,
            reference_path_length_m=reference,
            spl=0.0 if reference is not None else None,
            navigation_simulated_seconds=nav_sim_seconds,
            interaction_simulated_seconds=interaction_sim_seconds,
            total_simulated_seconds=nav_sim_seconds + interaction_sim_seconds,
            elapsed_seconds=time.monotonic() - started,
            target_distance_m=None,
            target_visibility_fraction=None,
            interaction_attempts=public_attempts if restricted_public_mode else attempts,
            applied_action_step_count=applied_action_step_count,
            no_fresh_action_count=int(
                policy_termination_snapshot.get("total_no_fresh_action_count", 0)
            ),
            policy_termination=policy_termination_snapshot,
            episode_step_budget=effective_max_steps,
            step_budget_mode=config.step_budget_mode,
            step_budget_basis=_public_step_budget_basis(step_budget_basis),
            trace_path=str(trace_path),
            error=error,
            scoring_eligible=False,
            scoring_exclusion_reasons=(
                scoring_exclusion_reasons
                if scoring_exclusion_reasons
                else ["runtime_exception"]
            ),
            runtime_goal_consistency=public_runtime_goal_consistency,
            runtime_consistency=public_runtime_consistency,
            timing_summary=phase_timings.summary(),
            early_stop=stall_tracker.snapshot(),
            paper_metric_schema_version=PAPER_METRIC_SCHEMA_VERSION,
            paper_metric_config=paper_metric_config_payload,
            valid_interaction_attempt_count=(
                paper_interaction_score.valid_interaction_attempt_count
            ),
            error_interaction_attempt_count=(
                paper_interaction_score.error_interaction_attempt_count
            ),
            task_irrelevant_interaction_attempt_count=(
                paper_interaction_score.task_irrelevant_interaction_attempt_count
            ),
            failed_interaction_attempt_count=(
                paper_interaction_score.failed_interaction_attempt_count
            ),
            repeated_interaction_attempt_count=(
                paper_interaction_score.repeated_interaction_attempt_count
            ),
            interaction_precision_episode=(
                paper_interaction_score.interaction_precision_episode
            ),
            episode_total_cost=episode_total_cost,
            episode_total_cost_breakdown=episode_total_cost_breakdown,
        ).to_dict()
        if restricted_public_mode:
            trace.append({"exception": error})
        else:
            trace.append({"exception": error, "traceback": traceback.format_exc()})
        status = "exception"
    finally:
        if ros_goal_status is not None:
            ros_goal_status.close()
        if ros_behavior_feedback is not None:
            ros_behavior_feedback.close()
        if restricted_ros_runtime is not None:
            try:
                restricted_ros_runtime.adapter.close()
            except Exception:
                pass
        if policy is not None:
            try:
                policy.close()
            except Exception:
                pass
        if sampler is not None:
            try:
                sampler.close()
            except Exception:
                pass
        if scene_install_module is not None and scene_install_original is not None:
            scene_install_module.install_scene_with_objects_and_grasps_from_path = scene_install_original
    if private_visualization is not None and catalog is not None:
        private_visualization["actual_interactions"] = _private_actual_interaction_markers(attempts, catalog)
        _atomic_json(episode_dir / "episode_visualization.json", private_visualization)
    _atomic_json(trace_path, {"status": status, "run_signature": run_signature, "result": result, "trace": trace})
    return result


def _worker(payload: tuple[dict[str, Any], int, dict[str, Any], str]) -> dict[str, Any]:
    raw_config, index, episode, signature = payload
    raw_config["benchmark"] = Path(raw_config["benchmark"])
    raw_config["output_dir"] = Path(raw_config["output_dir"])
    raw_config["image_resolution"] = None if raw_config["image_resolution"] is None else tuple(raw_config["image_resolution"])
    return evaluate_episode(BenchmarkEvaluationConfig(**raw_config), index, episode, run_signature=signature)


def _write_reports(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    _atomic_json(output_dir / "results.json", rows)
    _atomic_json(output_dir / "summary.json", summary)
    scoring_path = output_dir / "scoring_manifest.jsonl"
    scoring_tmp = scoring_path.with_name(f".{scoring_path.name}.{os.getpid()}.tmp")
    with scoring_tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe_json({
                "schema_version": "interactive_nav_v3_scoring_eligibility_v1",
                "episode_index": row.get("episode_index"),
                "case_id": row.get("case_id"),
                "house_index": row.get("house_index"),
                "domains": row.get("domains", []),
                "interaction_requirement": row.get("interaction_requirement"),
                "status": row.get("status"),
                "scoring_eligible": bool(row.get("scoring_eligible", False)),
                "exclusion_reasons": row.get("scoring_exclusion_reasons", []),
                "nav_success": row.get("nav_success"),
                "spl": row.get("spl"),
                "required_interaction_success": row.get("required_interaction_success"),
                "interaction_precision_episode": row.get("interaction_precision_episode"),
                "episode_total_cost": row.get("episode_total_cost"),
                "paper_metric_config": row.get("paper_metric_config"),
                "runtime_consistency": row.get("runtime_consistency"),
            }), ensure_ascii=False) + "\n")
    scoring_tmp.replace(scoring_path)
    if rows:
        columns = sorted({key for row in rows for key in row if key not in {"interaction_attempts"}})
        csv_path = output_dir / "results.csv"
        temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(_safe_json(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in columns})
        temporary.replace(csv_path)
    lines = [
        "# InteractiveNav V3 benchmark evaluation",
        "",
        f"- Episodes replayed: {len(rows)}",
        f"- Scoring-eligible: {summary.get('scoring_eligible_episode_count', len(rows))}",
        f"- Runtime-ineligible: {summary.get('runtime_ineligible_episode_count', 0)}",
        f"- Early-stopped scored failures: {summary.get('early_stop_episode_count', 0)}",
        f"- Early-stop reasons: {summary.get('early_stop_reason_counts', {})}",
        f"- Paper metric configuration: {summary.get('paper_metric_config', {})}",
        "",
        "## Formal score groups",
        "",
    ]
    if not summary.get("groups"):
        lines.append("No scoring-eligible episodes were available for formal metrics.")
    for name, values in summary.get("groups", {}).items():
        lines.append(
            f"- `{name}`: n={values['episode_count']}, SR={values['success_rate']}, "
            f"TaskSR={values['task_success_rate']}, ICS={values['interaction_conditioned_success_rate']}, "
            f"NavSR={values['nav_success_rate']}, ISR={values['required_interaction_success_rate']}, "
            f"IP={values['interaction_precision']}, SPL={values['mean_spl']}, "
            f"TotalCost={values.get('mean_total_cost')}, "
            f"ObsTurns={values.get('mean_step_count')}, "
            f"AppliedSteps={values.get('mean_applied_action_step_count')}, "
            f"NoFresh={values.get('mean_no_fresh_action_count')}, "
            f"EarlyStop={values.get('early_stop_episode_count', 0)}"
        )
    report = output_dir / "report.md"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(report)


def run_evaluation(config: BenchmarkEvaluationConfig) -> dict[str, Any]:
    """Run independent V3 evaluation with spawn-safe MuJoCo multiprocessing."""

    config.validate()
    config.benchmark = Path(config.benchmark).resolve()
    config.output_dir = Path(config.output_dir).resolve()
    benchmark_file, episodes = _load_episodes(config.benchmark)
    indices = _selected_indices(config, episodes)
    benchmark_hash = _sha256(benchmark_file)
    signature, manifest_payload = _run_signature(config, benchmark_hash, indices)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_signature = existing.get("run_signature")
        if config.resume:
            if existing_signature != signature:
                raise ValueError("--resume refused: existing output has a different benchmark/evaluation signature")
        else:
            raise FileExistsError(f"Output directory already contains run_manifest.json: {config.output_dir}; use --resume or a new directory")
    else:
        _atomic_json(manifest_path, {"run_signature": signature, **manifest_payload, "created_unix_s": time.time()})
    raw_config = asdict(config)
    raw_config["benchmark"] = str(config.benchmark)
    raw_config["output_dir"] = str(config.output_dir)
    total_payloads = len(indices)

    def make_payload(index: int) -> tuple[dict[str, Any], int, dict[str, Any], str]:
        return raw_config, index, episodes[index], signature

    rows: list[dict[str, Any]] = []
    progress_started = time.monotonic()

    def record_progress(*, force: bool = False) -> None:
        completed = len(rows)
        if not force and completed % int(config.progress_every) != 0:
            return
        elapsed = max(time.monotonic() - progress_started, 1e-9)
        rate = completed / elapsed
        remaining = total_payloads - completed
        eta_seconds = None if rate <= 0 else remaining / rate
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "processed": completed,
            "total": total_payloads,
            "progress": 0.0 if not total_payloads else completed / total_payloads,
            "eligible": sum(bool(row.get("scoring_eligible")) for row in rows),
            "ineligible": sum(not bool(row.get("scoring_eligible")) for row in rows),
            "exceptions": sum(row.get("status") == "exception" for row in rows),
            "rate_episodes_per_second": rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
        }
        # Keep the live progress artifact useful to an outer scheduler as well
        # as to humans watching the terminal. ``success`` is the historical
        # formal V3 score; only complete, scoring-eligible rows contribute to
        # the interim SR so runtime exceptions and consistency exclusions do not
        # silently dilute the metric.
        eligible_rows = [
            row
            for row in rows
            if row.get("status") == "complete" and bool(row.get("scoring_eligible"))
        ]
        payload.update(
            {
                "scored": len(eligible_rows),
                "success_count": sum(bool(row.get("success")) for row in eligible_rows),
                "task_success_count": sum(bool(row.get("task_success")) for row in eligible_rows),
                "nav_success_count": sum(bool(row.get("nav_success")) for row in eligible_rows),
                "interaction_success_count": sum(
                    bool(row.get("required_interaction_success")) for row in eligible_rows
                ),
                "success_rate": (
                    sum(bool(row.get("success")) for row in eligible_rows) / len(eligible_rows)
                    if eligible_rows
                    else None
                ),
            }
        )
        _atomic_json(config.output_dir / "progress.json", payload)
        progress_sr = "NA" if payload["success_rate"] is None else format(payload["success_rate"], ".4f")
        progress_task_sr = (
            "NA"
            if not eligible_rows
            else format(payload["task_success_count"] / len(eligible_rows), ".4f")
        )
        progress_nav_sr = (
            "NA"
            if not eligible_rows
            else format(payload["nav_success_count"] / len(eligible_rows), ".4f")
        )
        try:
            with (config.output_dir / "progress.log").open("a", encoding="utf-8") as handle:
                handle.write(
                    "[benchmark-progress] "
                    f"processed={completed}/{total_payloads} "
                    f"scored={payload['scored']} "
                    f"success={payload['success_count']} "
                    f"SR={progress_sr} taskSR={progress_task_sr} navSR={progress_nav_sr} "
                    f"rate={rate:.4f}/s "
                    f"eta_s={'unknown' if eta_seconds is None else int(eta_seconds)}\n"
                )
        except OSError:
            # Progress reporting must never turn an otherwise valid evaluator
            # result into a failed episode when a log filesystem is transient.
            pass
        print(
            "[quality-gate-progress] "
            f"time={payload['time']} processed={completed}/{total_payloads} "
            f"eligible={payload['eligible']} ineligible={payload['ineligible']} "
            f"exceptions={payload['exceptions']} SR={'NA' if payload['success_rate'] is None else format(payload['success_rate'], '.3f')} "
            f"rate={rate:.3f}/s "
            f"eta_s={'unknown' if eta_seconds is None else int(eta_seconds)}",
            flush=True,
        )

    if config.workers == 1:
        for index in indices:
            rows.append(_worker(make_payload(index)))
            record_progress()
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            # Keep only a small bounded queue of serialized episodes.  Submitting
            # the entire benchmark at once needlessly duplicates large JSON
            # payloads in the parent and multiprocessing feeder threads.
            index_iter = iter(indices)
            pending: set[concurrent.futures.Future] = set()
            for _ in range(max(1, int(config.workers) * 2)):
                try:
                    pending.add(pool.submit(_worker, make_payload(next(index_iter))))
                except StopIteration:
                    break
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    rows.append(future.result())
                    record_progress()
                    try:
                        pending.add(pool.submit(_worker, make_payload(next(index_iter))))
                    except StopIteration:
                        pass
    record_progress(force=True)
    rows.sort(key=lambda row: int(row["episode_index"]))
    summary = summarise_results(rows)
    summary.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "paper_metric_config": _paper_metric_config(config).to_dict(),
            "run_signature": signature,
            "benchmark": str(benchmark_file),
            "benchmark_sha256": benchmark_hash,
            "policy": config.policy,
            "workers": config.workers,
            "max_steps": config.max_steps,
            "step_budget_mode": config.step_budget_mode,
            "min_steps": config.min_steps,
            "episode_step_budget_min": min((int(row["episode_step_budget"]) for row in rows), default=None),
            "episode_step_budget_max": max((int(row["episode_step_budget"]) for row in rows), default=None),
            "episode_step_budget_mean": (
                sum(int(row["episode_step_budget"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "episode_indices": indices,
            "result_count": len(rows),
            "complete_count": sum(row.get("status") == "complete" for row in rows),
            "exception_count": sum(row.get("status") == "exception" for row in rows),
        }
    )
    _write_reports(config.output_dir, rows, summary)
    return {"results": rows, "summary": summary}


def _parse_policy_kwargs(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--policy-kwargs-json must decode to an object")
    return parsed


def parse_args() -> BenchmarkEvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=["noop", "scripted_oracle", "ros_bridge", "ros_object_goal_rule", "factory"],
        default="noop",
    )
    parser.add_argument("--policy-factory")
    parser.add_argument("--policy-kwargs-json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help=(
            "Fixed evaluator-applied action budget, or the hard cap when "
            "--step-budget-mode=dynamic; ROS no-fresh waits do not consume it."
        ),
    )
    parser.add_argument("--step-budget-mode", choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument("--min-steps", type=int, default=300)
    parser.add_argument("--dynamic-path-free-m", type=float, default=3.0)
    parser.add_argument("--dynamic-steps-per-path-m", type=float, default=25.0)
    parser.add_argument("--dynamic-channel-interaction-steps", type=int, default=150)
    parser.add_argument("--dynamic-container-interaction-steps", type=int, default=200)
    parser.add_argument("--dynamic-container-joint-steps", type=int, default=40)
    parser.add_argument("--dynamic-step-quantum", type=int, default=50)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--camera-names", nargs="+", default=["head_camera"])
    parser.add_argument("--image-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[640, 480])
    parser.add_argument(
        "--simulator-profile",
        choices=[*SIMULATOR_PROTOCOL_PROFILES, "custom"],
        default=DEFAULT_SIMULATOR_PROFILE,
        help="Named physics/control protocol; custom requires all dt/yaw arguments.",
    )
    parser.add_argument("--policy-dt-ms", type=float)
    parser.add_argument("--ctrl-dt-ms", type=float)
    parser.add_argument("--sim-dt-ms", type=float)
    parser.add_argument(
        "--holo-base-yaw-control-mode",
        choices=["nearest_equivalent", "legacy_branch_reset"],
    )
    parser.add_argument("--force-duration-seconds", type=float, default=2.0)
    parser.add_argument("--force-collection-hz", type=float, default=5.0)
    parser.add_argument("--force-target-fraction", type=float, default=1.0)
    parser.add_argument("--force-max-internal-steps", type=int, default=1500)
    parser.add_argument("--interaction-max-distance-m", type=float, default=1.75)
    parser.add_argument(
        "--require-interaction-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the evaluator-private interaction target to be visible before sealed force execution.",
    )
    parser.add_argument(
        "--require-runtime-goal-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse formal scoring when a frozen terminal oracle waypoint does not reach "
            "the live selected target. Use --no-require-runtime-goal-consistency only for "
            "integration debugging; such rows remain scoring-ineligible."
        ),
    )
    parser.add_argument("--allow-internal-object-names", action="store_true", help="debug-only non-oracle name selector")
    parser.add_argument("--oracle-navigation-mode", choices=["direct_pose", "task_action"], default="direct_pose")
    parser.add_argument("--ros-observation-topic", default="/molmo_spaces/head_camera/image")
    parser.add_argument("--ros-action-topic", default="/molmo_spaces/action")
    parser.add_argument(
        "--ros-action-timeout-s",
        type=float,
        default=5.0,
        help="Maximum wait for a fresh ROS action per decision; 0 waits indefinitely.",
    )
    parser.add_argument("--ros-cmd-vel-linear-gain", type=float, default=3.0)
    parser.add_argument("--ros-require-move-base-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ros-map-warmup-skip-frames", type=int, default=0)
    parser.add_argument("--ros-step-ready-topic", default="/semantic_decision/step_ready")
    parser.add_argument(
        "--ros-step-ready-barrier-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for semantic mapping/decision readiness before opening each fresh-command gate.",
    )
    parser.add_argument("--ros-step-ready-warmup-skip-frames", type=int, default=0)
    parser.add_argument("--ros-step-ready-timeout-s", type=float, default=2.0)
    parser.add_argument("--ros-step-ready-bootstrap-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--ros-step-frame-dir",
        default="",
        help="Optional evaluator-owned directory for exact per-step RGB PNGs and manifest.jsonl.",
    )
    parser.add_argument("--ros-step-frame-queue-size", type=int, default=4)
    parser.add_argument(
        "--ros-step-capture-ack-topic",
        default="/molmo_spaces/step_capture_ack",
    )
    parser.add_argument(
        "--ros-step-capture-ack-barrier-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait until the recorder queues each raw step boundary before advancing the V3 bridge.",
    )
    parser.add_argument("--ros-step-capture-ack-timeout-s", type=float, default=2.0)
    parser.add_argument("--ros-target-topic", default="/semantic_decision/target")
    parser.add_argument(
        "--ros-goal-status-topic",
        default="/semantic_decision/goal_status",
    )
    parser.add_argument("--ros-restricted-gt-topic", default="/semantic_mapping/gt_observations")
    parser.add_argument("--ros-interaction-command-topic", default="/semantic_decision/interaction_command")
    parser.add_argument("--ros-interaction-result-topic", default="/semantic_mapping/interaction_result")
    parser.add_argument("--ros-behavior-feedback-topic", default="/semantic_decision/behavior_feedback")
    parser.add_argument(
        "--ros-stall-early-stop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "End a ROS rollout after repeated distinct NAVIGATE/EXPLORE failures "
            "with no material simulated-base displacement."
        ),
    )
    parser.add_argument("--ros-stall-min-failed-subgoals", type=int, default=8)
    parser.add_argument("--ros-stall-max-displacement-m", type=float, default=0.15)
    parser.add_argument("--ros-stall-min-no-progress-steps", type=int, default=20)
    parser.add_argument(
        "--ros-command-starvation-timeout-s",
        type=float,
        default=60.0,
        help=(
            "End a ROS rollout as command_starvation after this continuous wall-clock "
            "interval of bridge no-fresh waits; 0 disables this guard."
        ),
    )
    parser.add_argument(
        "--ros-observation-turn-multiplier",
        type=float,
        default=4.0,
        help=(
            "Hard observation-turn limit relative to the applied-action budget; "
            "bridge no-fresh waits consume observation turns but not action steps."
        ),
    )
    parser.add_argument(
        "--ros-final-goal-status-drain-timeout-s",
        type=float,
        default=5.0,
        help=(
            "Zero-step wall-clock grace after the final applied action so an "
            "asynchronous goal status for that published frame is not lost."
        ),
    )
    parser.add_argument("--restricted-gt-min-visible-pixels", type=int, default=16)
    parser.add_argument("--restricted-gt-min-bbox-area-pixels", type=int, default=512)
    parser.add_argument(
        "--restricted-gt-min-bbox-short-side-pixels",
        type=int,
        default=1,
        help="Reject restricted-GT components thinner than this many pixels.",
    )
    parser.add_argument(
        "--restricted-gt-min-portal-bbox-short-side-pixels",
        type=int,
        default=8,
        help="Stricter restricted-GT short-side gate for door/portal observations.",
    )
    parser.add_argument(
        "--restricted-gt-min-visible-fraction",
        type=float,
        default=0.2,
        help="Minimum observed 2-D extent divided by the projected 3-D AABB extent; zero disables this sanity filter.",
    )
    parser.add_argument(
        "--restricted-gt-max-distance-m",
        type=float,
        default=4.0,
        help="Maximum head-camera distance for evaluator-published restricted-GT observations; 0 disables the limit.",
    )
    parser.add_argument(
        "--quality-gate-only",
        action="store_true",
        help="Load and validate runtime consistency without executing policy actions.",
    )
    parser.add_argument(
        "--allow-runtime-ineligible",
        action="store_true",
        help="Return success despite runtime-ineligible rows (audit/debug only).",
    )
    parser.add_argument("--runtime-joint-position-tolerance", type=float, default=0.02)
    parser.add_argument("--runtime-joint-fraction-tolerance", type=float, default=0.05)
    parser.add_argument("--runtime-base-position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--runtime-base-yaw-tolerance-rad", type=float, default=0.05)
    parser.add_argument(
        "--paper-cost-interaction-attempt",
        type=float,
        default=DEFAULT_PAPER_COST_INTERACTION_ATTEMPT,
        help="Paper Total Cost lambda: base cost for every interaction attempt.",
    )
    parser.add_argument(
        "--paper-cost-error-surcharge",
        type=float,
        default=DEFAULT_PAPER_COST_ERROR_SURCHARGE,
        help="Paper Total Cost mu: additional cost for an erroneous attempt (must exceed lambda).",
    )
    parser.add_argument(
        "--paper-cost-failure-penalty",
        type=float,
        default=DEFAULT_PAPER_COST_FAILURE_PENALTY,
        help="Paper Total Cost kappa: fixed penalty for NavToObj terminal failure.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    simulator_protocol = resolve_simulator_protocol(
        args.simulator_profile,
        policy_dt_ms=args.policy_dt_ms,
        ctrl_dt_ms=args.ctrl_dt_ms,
        sim_dt_ms=args.sim_dt_ms,
        holo_base_yaw_control_mode=args.holo_base_yaw_control_mode,
    )
    return BenchmarkEvaluationConfig(
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        policy=args.policy,
        policy_factory=args.policy_factory,
        policy_kwargs=_parse_policy_kwargs(args.policy_kwargs_json),
        workers=args.workers,
        max_steps=args.max_steps,
        step_budget_mode=args.step_budget_mode,
        min_steps=args.min_steps,
        dynamic_path_free_m=args.dynamic_path_free_m,
        dynamic_steps_per_path_m=args.dynamic_steps_per_path_m,
        dynamic_channel_interaction_steps=args.dynamic_channel_interaction_steps,
        dynamic_container_interaction_steps=args.dynamic_container_interaction_steps,
        dynamic_container_joint_steps=args.dynamic_container_joint_steps,
        dynamic_step_quantum=args.dynamic_step_quantum,
        episode_indices=args.episode_indices,
        max_episodes=args.max_episodes,
        resume=args.resume,
        record_video=args.record_video,
        video_fps=args.video_fps,
        camera_names=list(args.camera_names),
        image_resolution=tuple(args.image_resolution),
        simulator_profile=args.simulator_profile,
        policy_dt_ms=simulator_protocol["policy_dt_ms"],
        ctrl_dt_ms=simulator_protocol["ctrl_dt_ms"],
        sim_dt_ms=simulator_protocol["sim_dt_ms"],
        holo_base_yaw_control_mode=simulator_protocol[
            "holo_base_yaw_control_mode"
        ],
        force_duration_seconds=args.force_duration_seconds,
        force_collection_hz=args.force_collection_hz,
        force_target_fraction=args.force_target_fraction,
        force_max_internal_steps=args.force_max_internal_steps,
        interaction_max_distance_m=args.interaction_max_distance_m,
        require_interaction_visible=args.require_interaction_visible,
        require_runtime_goal_consistency=args.require_runtime_goal_consistency,
        allow_internal_object_names=args.allow_internal_object_names,
        oracle_navigation_mode=args.oracle_navigation_mode,
        ros_observation_topic=args.ros_observation_topic,
        ros_action_topic=args.ros_action_topic,
        ros_action_timeout_s=args.ros_action_timeout_s,
        ros_cmd_vel_linear_gain=args.ros_cmd_vel_linear_gain,
        ros_require_move_base_active=args.ros_require_move_base_active,
        ros_map_warmup_skip_frames=args.ros_map_warmup_skip_frames,
        ros_step_ready_topic=args.ros_step_ready_topic,
        ros_step_ready_barrier_enabled=args.ros_step_ready_barrier_enabled,
        ros_step_ready_warmup_skip_frames=args.ros_step_ready_warmup_skip_frames,
        ros_step_ready_timeout_s=args.ros_step_ready_timeout_s,
        ros_step_ready_bootstrap_timeout_s=args.ros_step_ready_bootstrap_timeout_s,
        ros_step_frame_dir=args.ros_step_frame_dir,
        ros_step_frame_queue_size=args.ros_step_frame_queue_size,
        ros_step_capture_ack_topic=args.ros_step_capture_ack_topic,
        ros_step_capture_ack_barrier_enabled=args.ros_step_capture_ack_barrier_enabled,
        ros_step_capture_ack_timeout_s=args.ros_step_capture_ack_timeout_s,
        ros_target_topic=args.ros_target_topic,
        ros_goal_status_topic=args.ros_goal_status_topic,
        ros_restricted_gt_topic=args.ros_restricted_gt_topic,
        ros_interaction_command_topic=args.ros_interaction_command_topic,
        ros_interaction_result_topic=args.ros_interaction_result_topic,
        ros_behavior_feedback_topic=args.ros_behavior_feedback_topic,
        ros_stall_early_stop_enabled=args.ros_stall_early_stop_enabled,
        ros_stall_min_failed_subgoals=args.ros_stall_min_failed_subgoals,
        ros_stall_max_displacement_m=args.ros_stall_max_displacement_m,
        ros_stall_min_no_progress_steps=args.ros_stall_min_no_progress_steps,
        ros_command_starvation_timeout_s=args.ros_command_starvation_timeout_s,
        ros_observation_turn_multiplier=args.ros_observation_turn_multiplier,
        ros_final_goal_status_drain_timeout_s=(
            args.ros_final_goal_status_drain_timeout_s
        ),
        restricted_gt_min_visible_pixels=args.restricted_gt_min_visible_pixels,
        restricted_gt_min_bbox_area_pixels=args.restricted_gt_min_bbox_area_pixels,
        restricted_gt_min_bbox_short_side_pixels=(
            args.restricted_gt_min_bbox_short_side_pixels
        ),
        restricted_gt_min_portal_bbox_short_side_pixels=(
            args.restricted_gt_min_portal_bbox_short_side_pixels
        ),
        restricted_gt_min_visible_fraction=args.restricted_gt_min_visible_fraction,
        restricted_gt_max_distance_m=args.restricted_gt_max_distance_m,
        quality_gate_only=args.quality_gate_only,
        allow_runtime_ineligible=args.allow_runtime_ineligible,
        runtime_joint_position_tolerance=args.runtime_joint_position_tolerance,
        runtime_joint_fraction_tolerance=args.runtime_joint_fraction_tolerance,
        runtime_base_position_tolerance_m=args.runtime_base_position_tolerance_m,
        runtime_base_yaw_tolerance_rad=args.runtime_base_yaw_tolerance_rad,
        paper_cost_interaction_attempt=args.paper_cost_interaction_attempt,
        paper_cost_error_surcharge=args.paper_cost_error_surcharge,
        paper_cost_failure_penalty=args.paper_cost_failure_penalty,
        progress_every=args.progress_every,
    )


def _evaluation_exit_code(
    output: Mapping[str, Any], *, allow_runtime_ineligible: bool
) -> int:
    summary = output["summary"]
    planned_count = len(summary.get("episode_indices", ()))
    return int(
        int(summary.get("exception_count", 0)) > 0
        or int(summary.get("result_count", 0)) != planned_count
        or (
            int(summary.get("runtime_ineligible_episode_count", 0)) > 0
            and not allow_runtime_ineligible
        )
    )


def main() -> int:
    config = parse_args()
    output = run_evaluation(config)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return _evaluation_exit_code(
        output, allow_runtime_ineligible=config.allow_runtime_ineligible
    )


if __name__ == "__main__":
    raise SystemExit(main())
