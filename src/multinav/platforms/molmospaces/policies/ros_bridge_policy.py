import json
import queue
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy
from multinav.platforms.molmospaces.policies.organized_depth_scan import (
    OrganizedDepthScanProjector,
)
from multinav.platforms.molmospaces.policies.realtime_gt_observation import RealtimeGTObservationPublisher
from molmo_spaces.tasks.task import BaseMujocoTask


class RosBridgePolicy(BasePolicy):
    """
    ROS bridge policy for closed-loop external control.

    - Publishes observations to a ROS topic.
    - Receives actions from another ROS node.
    """

    def __init__(
        self,
        config,
        task: BaseMujocoTask | None = None,
        observation_topic: str = "/molmo_spaces/head_camera/image",
        action_topic: str = "/molmo_spaces/action",
        pointcloud_topic: str = "/registered_scan",
        camera_info_topic: str = "/molmo_spaces/head_camera/camera_info",
        depth_topic: str = "/molmo_spaces/head_camera/depth",
        action_timeout_s: float = 0.0,
        blocking_observation_republish_period_s: float = 0.25,
        blocking_republish_pointcloud: bool = False,
        queue_size: int = 1,
        observation_queue_size: int = 1,
        extra_image_queue_size: int = 16,
        publish_pointcloud: bool = True,
        publish_camera_info: bool = True,
        publish_depth_scan: bool = False,
        depth_scan_topic: str = "/molmo_spaces/organized_depth_scan",
        depth_camera_name: str = "head_camera",
        pointcloud_frame_id: str = "tf_frame_lidar",
        optical_frame_id: str = "head_camera_optical_frame",
        depth_fov_deg: float = 90.0,
        depth_min_m: float = 0.1,
        depth_max_m: float = 30.0,
        pointcloud_stride: int = 1,
        pointcloud_self_filter_radius_m: float = 0.32,
        pointcloud_roll_correction_deg: float = 0.0,
        odom_topic: str = "/odom",
        publish_odom: bool = True,
        publish_odom_twist: bool = False,
        odom_twist_source: str = "instantaneous",
        map_frame_id: str = "tf_frame_map",
        odom_frame_id: str = "tf_frame_odom",
        base_frame_id: str = "tf_frame_base_link",
        lidar_offset_x_m: float = 0.0,
        lidar_offset_y_m: float = 0.0,
        lidar_offset_z_m: float = 1.6,
        lidar_calib_x_m: float = 0.0,
        lidar_calib_y_m: float = 0.0,
        lidar_calib_z_m: float = 0.0,
        lidar_calib_roll_deg: float = 0.0,
        lidar_calib_pitch_deg: float = 0.0,
        lidar_calib_yaw_deg: float = 0.0,
        allow_static_lidar_tf_fallback: bool = False,
        cmd_vel_topic: str = "/cmd_vel_stamped",
        cmd_vel_timeout_s: float = 0.5,
        cmd_vel_control_dt_s: float | None = None,
        cmd_vel_linear_gain: float = 1.0,
        require_fresh_cmd_vel: bool = True,
        require_move_base_active_for_cmd_vel: bool = False,
        move_base_status_topic: str = "/move_base/status",
        map_warmup_skip_frames: int = 10,
        immediate_noop_after_publish: bool = False,
        timing_log_every_n_frames: int = 30,
        extra_image_topic: str = "/molmo_spaces/debug_front_camera/image",
        extra_image_camera_name: str = "debug_front_camera",
        publish_realtime_gt: bool = False,
        realtime_gt_topic: str = "/semantic_mapping/gt_observations",
        realtime_gt_camera_name: str = "head_camera",
        realtime_gt_min_visible_pixels: int = 16,
        realtime_gt_min_visible_bbox_short_side_px: int = 1,
        realtime_gt_min_portal_bbox_short_side_px: int = 8,
        realtime_gt_min_visible_fraction: float = 0.2,
        realtime_gt_required_consecutive_observations: int = 2,
        realtime_gt_step_interval: int = 3,
        realtime_gt_max_distance_m: float = 4.0,
        realtime_gt_emit_interaction_approach_axis: bool = False,
        step_frame_dir: str = "",
        step_frame_queue_size: int = 4,
        step_sync_topic: str = "/molmo_spaces/step_sync",
        step_capture_ack_topic: str = "/molmo_spaces/step_capture_ack",
        step_capture_ack_barrier_enabled: bool = False,
        step_capture_ack_timeout_s: float = 2.0,
        fresh_command_gate_topic: str = "/molmo_spaces/fresh_cmd_gate",
        step_ready_topic: str = "/semantic_decision/step_ready",
        step_ready_barrier_enabled: bool = False,
        step_ready_warmup_skip_frames: int | None = None,
        step_ready_timeout_s: float = 30.0,
        step_ready_bootstrap_timeout_s: float | None = None,
        step_ready_bootstrap_republish_period_s: float = 0.5,
        tf_keepalive_period_s: float = 0.25,
        allow_lateral_cmd_vel: bool = True,
    ) -> None:
        super().__init__(config, task)
        self.observation_topic = observation_topic
        self.action_topic = action_topic
        self.pointcloud_topic = pointcloud_topic
        self.camera_info_topic = camera_info_topic
        self.depth_topic = depth_topic
        self.action_timeout_s = float(action_timeout_s)
        self.last_action_timed_out = False
        self.blocking_observation_republish_period_s = max(
            0.0, float(blocking_observation_republish_period_s)
        )
        self.blocking_republish_pointcloud = bool(blocking_republish_pointcloud)
        # The evaluator can spend wall-clock seconds inside a sealed simulator
        # interaction.  Keep the last public pose transform fresh during those
        # pauses without replaying RGB/depth/point-cloud observations.
        self.tf_keepalive_period_s = max(0.0, float(tf_keepalive_period_s))
        self.queue_size = queue_size
        self.observation_queue_size = int(observation_queue_size)
        self.extra_image_queue_size = int(extra_image_queue_size)
        self.publish_pointcloud = publish_pointcloud
        self.publish_camera_info = publish_camera_info
        self.publish_depth_scan = bool(publish_depth_scan)
        self.depth_scan_topic = str(depth_scan_topic)
        self._organized_depth_scan_projector = OrganizedDepthScanProjector()
        self.depth_camera_name = depth_camera_name
        self.pointcloud_frame_id = pointcloud_frame_id
        self.optical_frame_id = optical_frame_id
        self.depth_fov_deg = float(depth_fov_deg)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.pointcloud_stride = max(1, int(pointcloud_stride))
        self.pointcloud_self_filter_radius_m = max(0.0, float(pointcloud_self_filter_radius_m))
        self.pointcloud_roll_correction_deg = float(pointcloud_roll_correction_deg)
        self.odom_topic = odom_topic
        self.publish_odom = bool(publish_odom)
        self.publish_odom_twist = bool(publish_odom_twist)
        if odom_twist_source not in {"instantaneous", "step_delta"}:
            raise ValueError("odom_twist_source must be instantaneous or step_delta")
        self.odom_twist_source = odom_twist_source
        self.map_frame_id = map_frame_id
        self.odom_frame_id = odom_frame_id
        self.base_frame_id = base_frame_id
        self.lidar_offset_x_m = float(lidar_offset_x_m)
        self.lidar_offset_y_m = float(lidar_offset_y_m)
        self.lidar_offset_z_m = float(lidar_offset_z_m)
        self.lidar_calib_x_m = float(lidar_calib_x_m)
        self.lidar_calib_y_m = float(lidar_calib_y_m)
        self.lidar_calib_z_m = float(lidar_calib_z_m)
        self.lidar_calib_roll_deg = float(lidar_calib_roll_deg)
        self.lidar_calib_pitch_deg = float(lidar_calib_pitch_deg)
        self.lidar_calib_yaw_deg = float(lidar_calib_yaw_deg)
        self.allow_static_lidar_tf_fallback = bool(allow_static_lidar_tf_fallback)
        self.cmd_vel_topic = cmd_vel_topic
        self.cmd_vel_timeout_s = float(cmd_vel_timeout_s)
        self.cmd_vel_linear_gain = max(0.0, float(cmd_vel_linear_gain))
        self.allow_lateral_cmd_vel = bool(allow_lateral_cmd_vel)
        self.last_cmd_vel_lateral_rejected = False
        self.lateral_cmd_vel_rejection_count = 0
        self.require_fresh_cmd_vel = bool(require_fresh_cmd_vel)
        self.require_move_base_active_for_cmd_vel = bool(require_move_base_active_for_cmd_vel)
        self.move_base_status_topic = move_base_status_topic
        self.map_warmup_skip_frames = max(0, int(map_warmup_skip_frames))
        self.immediate_noop_after_publish = bool(immediate_noop_after_publish)
        self.timing_log_every_n_frames = max(0, int(timing_log_every_n_frames))
        self.extra_image_topic = extra_image_topic
        self.extra_image_camera_name = extra_image_camera_name
        self.step_frame_dir = Path(step_frame_dir).expanduser().resolve() if step_frame_dir else None
        self.step_sync_topic = str(step_sync_topic)
        self.step_capture_ack_topic = str(step_capture_ack_topic)
        self.step_capture_ack_barrier_enabled = bool(step_capture_ack_barrier_enabled)
        self.step_capture_ack_timeout_s = max(0.0, float(step_capture_ack_timeout_s))
        self.fresh_command_gate_topic = str(fresh_command_gate_topic)
        self.step_ready_topic = str(step_ready_topic)
        self.step_ready_barrier_enabled = bool(step_ready_barrier_enabled)
        self.step_ready_warmup_skip_frames = max(
            0,
            int(
                self.map_warmup_skip_frames
                if step_ready_warmup_skip_frames is None
                else step_ready_warmup_skip_frames
            ),
        )
        self.step_ready_timeout_s = max(0.0, float(step_ready_timeout_s))
        self.step_ready_bootstrap_timeout_s = max(
            0.0,
            float(
                max(5.0, self.step_ready_timeout_s)
                if step_ready_bootstrap_timeout_s is None
                else step_ready_bootstrap_timeout_s
            ),
        )
        self.step_ready_bootstrap_republish_period_s = max(
            0.01, float(step_ready_bootstrap_republish_period_s)
        )
        self._step_ready_bootstrap_complete = False
        self._latest_step_ready: dict[str, Any] = {}
        self._latest_step_ready_mono_s = 0.0
        # Per-simulator-step causal barrier evidence.  It is intentionally
        # separate from numeric timing so the JSON timing trace can retain the
        # source tuple and mapper stage watermarks used to unlock (or time out)
        # a step.
        self.last_step_ready_diagnostics: dict[str, Any] = {}
        self._latest_step_capture_ack: dict[str, Any] = {}
        self._latest_step_capture_ack_mono_s = 0.0
        self._current_step_stamp_sec = 0.0
        self._step_frame_queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(step_frame_queue_size))
        )
        self._step_frame_thread = None
        self._latest_gt_payload = None
        # V3 publishes its restricted public perception through an evaluator
        # adapter rather than this policy's task-backed realtime-GT publisher.
        # Keep one adapter payload for the next successfully published RGB so
        # the media manifest has the exact public evidence available to the
        # policy for that image.  ``None`` means no external payload was
        # provided; an empty observations list remains a real payload.
        self._pending_step_frame_public_payload: dict[str, Any] | None = None
        self.publish_realtime_gt = bool(publish_realtime_gt)
        if cmd_vel_control_dt_s is None:
            cfg_dt_ms = getattr(config, "policy_dt_ms", None)
            if cfg_dt_ms is not None and float(cfg_dt_ms) > 0.0:
                self.cmd_vel_control_dt_s = float(cfg_dt_ms) / 1000.0
            else:
                self.cmd_vel_control_dt_s = 0.1
        else:
            self.cmd_vel_control_dt_s = max(1e-3, float(cmd_vel_control_dt_s))
        # Cache per-intrinsics projection lookup tables for depth->pointcloud.
        # This avoids rebuilding uv grids every frame.
        self._pointcloud_projection_cache: dict[
            tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}

        self._step_idx = 0
        self._latest_action: dict[str, Any] | None = None
        self._latest_action_step: int = -1
        self._latest_action_mono_s: float = 0.0
        self._last_consumed_action_step: int = -1
        self._latest_cmd_vel: np.ndarray | None = None
        self._latest_cmd_vel_mono_s: float = 0.0
        self._move_base_active: bool = False
        self._last_base_position_xyz: np.ndarray | None = None
        self._last_base_pose_xyyaw: np.ndarray | None = None
        self._odom_step_sample = None
        self._last_common_stamp_s: float | None = None
        self._stamp_lock = threading.Lock()
        self._tf_cache_lock = threading.Lock()
        self._latest_odom_tf_state: tuple[float, ...] | None = None
        self._latest_base_to_lidar_tf: tuple[float, ...] | None = None
        self._tf_keepalive_timer = None
        self._base_position_jump_warn_m: float = 1.0
        self._timing_frame_count: int = 0
        self._timing_acc_ms: dict[str, float] = {
            "total": 0.0,
            "odom_tf": 0.0,
            "realtime_gt": 0.0,
            "realtime_gt_snapshot_hit": 0.0,
            "rgb_extract": 0.0,
            "rgb_encode": 0.0,
            "rgb_ros_publish": 0.0,
            "step_frame_enqueue": 0.0,
            "extra_rgb_extract": 0.0,
            "extra_rgb_encode": 0.0,
            "extra_rgb_ros_publish": 0.0,
            "rgb_publish": 0.0,
            "depth_extract_intrinsics": 0.0,
            "depth_msg_publish": 0.0,
            "pointcloud_convert": 0.0,
            "pointcloud_publish": 0.0,
            "depth_scan_convert": 0.0,
            "depth_scan_publish": 0.0,
            "camera_info_publish": 0.0,
            "blocking_republish": 0.0,
            "step_ready_wait": 0.0,
            "step_ready_satisfied": 0.0,
            "step_ready_timed_out": 0.0,
            "action_wait": 0.0,
            # Separate navigation-side scheduling from the mandatory readiness
            # barrier.  The two fresh-command fields use the command callback
            # timestamp, so they reveal whether an observed delay is before or
            # after move_base/DWA actually produced a command.
            "action_wait_after_ready": 0.0,
            "fresh_cmd_after_gate": 0.0,
            "fresh_action_after_gate": 0.0,
            "postprocess_action": 0.0,
            "step_sync_publish": 0.0,
            "step_capture_ack_wait": 0.0,
        }
        self._timing_samples_ms: dict[str, list[float]] = {
            key: [] for key in self._timing_acc_ms
        }
        self.last_timing_ms: dict[str, float] = {}
        self._latest_depth_scan_diagnostics: dict[str, int | float] = {}
        # Per-frame organized-depth provenance.  The ROS LaserScan message has
        # no field for why a beam is NaN, so retain the projector counters for
        # the recorder/timing JSONL without changing scan semantics.
        self.last_depth_scan_diagnostics: dict[str, int | float] = {}
        self.last_action_source: str = ""
        self._step_frame_queue_peak: int = 0
        self._lock = threading.Lock()

        self._patch_rosgraph_logger_for_py311()
        import rospy
        from actionlib_msgs.msg import GoalID, GoalStatusArray
        from geometry_msgs.msg import TransformStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image, LaserScan
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Empty, String
        from std_srvs.srv import Empty as EmptyService
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._rospy = rospy
        self._TransformStamped = TransformStamped
        self._TwistStamped = TwistStamped
        self._Image = Image
        self._LaserScan = LaserScan
        self._CameraInfo = CameraInfo
        self._PointCloud2 = PointCloud2
        self._PointField = PointField
        self._String = String
        self._GoalStatusArray = GoalStatusArray
        self._GoalID = GoalID
        self._Empty = Empty
        self._Odometry = Odometry
        if not rospy.core.is_initialized():
            rospy.init_node("molmo_spaces_ros_policy", anonymous=True, disable_signals=True)

        observation_queue_size = (
            None if self.observation_queue_size <= 0 else self.observation_queue_size
        )
        self._obs_pub = rospy.Publisher(
            self.observation_topic,
            Image,
            queue_size=observation_queue_size,
        )
        self._step_sync_pub = (
            rospy.Publisher(self.step_sync_topic, String, queue_size=4096)
            if self.step_sync_topic
            else None
        )
        # This is distinct from step_sync: it marks that the current RGB
        # observation's fresh-cmd wait has begun.  A startup scan can use it
        # to publish one command in the same evaluator action without relying
        # on timer phase or guessing whether an RGB callback ran too early.
        self._fresh_command_gate_pub = (
            rospy.Publisher(self.fresh_command_gate_topic, String, queue_size=32)
            if self.fresh_command_gate_topic
            else None
        )
        self._extra_image_pub = None
        if self.extra_image_topic and self.extra_image_camera_name:
            extra_image_queue_size = (
                None if self.extra_image_queue_size <= 0 else self.extra_image_queue_size
            )
            self._extra_image_pub = rospy.Publisher(
                self.extra_image_topic,
                Image,
                queue_size=extra_image_queue_size,
            )
        self._depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=self.queue_size)
        self._pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=self.queue_size)
        self._depth_scan_pub = (
            rospy.Publisher(self.depth_scan_topic, LaserScan, queue_size=self.queue_size)
            if self.publish_depth_scan and self.depth_scan_topic
            else None
        )
        self._camera_info_pub = rospy.Publisher(
            self.camera_info_topic, CameraInfo, queue_size=self.queue_size
        )
        self._image_camera_info_pub = rospy.Publisher(
            f"{self.observation_topic}/camera_info", CameraInfo, queue_size=self.queue_size
        )
        self._depth_camera_info_pub = rospy.Publisher(
            f"{self.depth_topic}/camera_info", CameraInfo, queue_size=self.queue_size
        )
        self._odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=self.queue_size)
        self._static_tf_pub = StaticTransformBroadcaster()
        self._tf_broadcaster = TransformBroadcaster()
        self._publish_static_tfs()
        self._action_sub = rospy.Subscriber(self.action_topic, String, self._action_callback)
        self._step_ready_sub = (
            rospy.Subscriber(self.step_ready_topic, String, self._step_ready_callback, queue_size=32)
            if self.step_ready_barrier_enabled and self.step_ready_topic
            else None
        )
        self._step_capture_ack_sub = (
            rospy.Subscriber(
                self.step_capture_ack_topic,
                String,
                self._step_capture_ack_callback,
                queue_size=32,
            )
            if self.step_capture_ack_barrier_enabled and self.step_capture_ack_topic
            else None
        )
        self._cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, TwistStamped, self._cmd_vel_callback)
        self._realtime_gt_publisher = None
        if self.publish_realtime_gt:
            self._realtime_gt_publisher = RealtimeGTObservationPublisher(
                rospy,
                String,
                topic=realtime_gt_topic,
                camera_name=realtime_gt_camera_name,
                min_visible_pixels=realtime_gt_min_visible_pixels,
                min_visible_bbox_short_side_px=(
                    realtime_gt_min_visible_bbox_short_side_px
                ),
                min_portal_bbox_short_side_px=(
                    realtime_gt_min_portal_bbox_short_side_px
                ),
                min_visible_fraction=realtime_gt_min_visible_fraction,
                required_consecutive_observations=realtime_gt_required_consecutive_observations,
                step_interval=realtime_gt_step_interval,
                max_distance_m=realtime_gt_max_distance_m,
                emit_interaction_approach_axis=(
                    realtime_gt_emit_interaction_approach_axis
                ),
                # GT is a latest-state stream.  A deep ROS queue makes newly
                # revealed container contents wait behind stale observations.
                queue_size=1,
            )
        if self.step_frame_dir is not None:
            self.step_frame_dir.mkdir(parents=True, exist_ok=True)
            self._step_frame_manifest = (self.step_frame_dir / "manifest.jsonl").open("w", encoding="utf-8")
            self._step_frame_thread = threading.Thread(
                target=self._run_step_frame_writer,
                name="ros-bridge-step-frame-writer",
                daemon=True,
            )
            self._step_frame_thread.start()
        self._move_base_status_sub = rospy.Subscriber(
            self.move_base_status_topic,
            GoalStatusArray,
            self._move_base_status_callback,
        )
        self._move_base_cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self._mapping_reset_pub = rospy.Publisher("/nav_system/reset", Empty, queue_size=1)
        self._explorer_reset_pub = rospy.Publisher("/explore_py/reset", Empty, queue_size=1)
        self._clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", EmptyService)
        if self.publish_odom and self.tf_keepalive_period_s > 0.0:
            self._tf_keepalive_timer = rospy.Timer(
                rospy.Duration.from_sec(self.tf_keepalive_period_s),
                self._tf_keepalive_callback,
            )
        self._episode_count = 0

    @staticmethod
    def _rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
        tr = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)

    def _build_optical_static_tf(self):
        """Build static TF from robot-centric frame to optical frame."""
        # parent(robot): x forward, y left, z up
        # child(optical): x right, y down, z forward
        # optical <- robot
        rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(rot)
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = self._rospy.Time.now()
        tf_msg.header.frame_id = self.pointcloud_frame_id
        tf_msg.child_frame_id = self.optical_frame_id
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        return tf_msg

    def _build_base_to_lidar_static_tf(self):
        """Base->lidar is published dynamically from observation camera extrinsics."""
        return None

    def _publish_static_tfs(self) -> None:
        """
        Publish all static transforms in one latched TF message.
        In rospy tf2, each sendTransform() publishes only provided transforms.
        """
        tfs = []
        tfs.append(self._build_optical_static_tf())
        self._static_tf_pub.sendTransform(tfs)

    @staticmethod
    def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """Convert scalar-first quaternion to a 3x3 rotation matrix."""
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        rot_y = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        rot_z = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rot_z @ rot_y @ rot_x

    def _apply_lidar_calibration(self, T_base_lidar: np.ndarray) -> np.ndarray:
        """Apply a small tunable correction in the lidar frame."""
        if (
            abs(self.lidar_calib_x_m) < 1e-9
            and abs(self.lidar_calib_y_m) < 1e-9
            and abs(self.lidar_calib_z_m) < 1e-9
            and abs(self.lidar_calib_roll_deg) < 1e-9
            and abs(self.lidar_calib_pitch_deg) < 1e-9
            and abs(self.lidar_calib_yaw_deg) < 1e-9
        ):
            return T_base_lidar

        T_lidar_calib = np.eye(4, dtype=np.float64)
        T_lidar_calib[:3, 3] = np.array(
            [self.lidar_calib_x_m, self.lidar_calib_y_m, self.lidar_calib_z_m],
            dtype=np.float64,
        )
        T_lidar_calib[:3, :3] = self._rpy_to_rotmat(
            np.deg2rad(self.lidar_calib_roll_deg),
            np.deg2rad(self.lidar_calib_pitch_deg),
            np.deg2rad(self.lidar_calib_yaw_deg),
        )
        return T_base_lidar @ T_lidar_calib

    def _extract_lidar_pose_rel_base(self, observation: Any) -> np.ndarray | None:
        """Return lidar pose in base frame as a 4x4 transform."""
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            return None

        sensor_params = obs_dict.get(f"sensor_param_{self.depth_camera_name}")
        if not isinstance(sensor_params, dict):
            return None

        cam2world_gl = sensor_params.get("cam2world_gl")
        if cam2world_gl is None:
            return None

        T_world_optical = np.asarray(cam2world_gl, dtype=np.float64)
        if T_world_optical.shape != (4, 4):
            return None

        px, py, pz = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        T_world_base = np.eye(4, dtype=np.float64)
        T_world_base[:3, :3] = self._quat_wxyz_to_rotmat(qw, qx, qy, qz)
        T_world_base[:3, 3] = np.array([px, py, pz], dtype=np.float64)

        # The pointcloud is published in a robot-centric lidar/body frame:
        # x forward, y left, z up. The optical frame is related by the fixed
        # lidar->optical rotation used in _build_optical_static_tf().
        R_lidar_to_optical = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        T_optical_lidar = np.eye(4, dtype=np.float64)
        T_optical_lidar[:3, :3] = R_lidar_to_optical.T

        return np.linalg.inv(T_world_base) @ T_world_optical @ T_optical_lidar

    def _publish_base_to_lidar_tf(self, observation: Any, stamp) -> bool:
        if self.base_frame_id == self.pointcloud_frame_id:
            return True

        T_base_lidar = self._extract_lidar_pose_rel_base(observation)
        if T_base_lidar is None:
            if not self.allow_static_lidar_tf_fallback:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: sensor_param_%s missing; skip mapping observation instead of publishing fixed base->lidar TF.",
                    self.depth_camera_name,
                )
                return False

            self._rospy.logwarn_throttle(
                5.0,
                "RosBridgePolicy: sensor_param_%s missing; falling back to configured fixed base->lidar TF.",
                self.depth_camera_name,
            )
            T_base_lidar = np.eye(4, dtype=np.float64)
            T_base_lidar[:3, 3] = np.array(
                [self.lidar_offset_x_m, self.lidar_offset_y_m, self.lidar_offset_z_m],
                dtype=np.float64,
            )
            T_base_lidar = self._apply_lidar_calibration(T_base_lidar)
            qx, qy, qz, qw = self._rotation_matrix_to_quaternion(T_base_lidar[:3, :3])
            state = (
                float(T_base_lidar[0, 3]),
                float(T_base_lidar[1, 3]),
                float(T_base_lidar[2, 3]),
                qx,
                qy,
                qz,
                qw,
            )
            self._publish_base_to_lidar_tf_from_state(state, stamp)
            with self._tf_cache_lock:
                self._latest_base_to_lidar_tf = state
            return False

        T_base_lidar = self._apply_lidar_calibration(T_base_lidar)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(T_base_lidar[:3, :3])
        self._rospy.loginfo_throttle(
            2.0,
            (
                "RosBridgePolicy: publishing dynamic base->lidar TF from sensor_param_%s: "
                "xyz=(%.3f, %.3f, %.3f), quat=(%.4f, %.4f, %.4f, %.4f)"
            ),
            self.depth_camera_name,
            float(T_base_lidar[0, 3]),
            float(T_base_lidar[1, 3]),
            float(T_base_lidar[2, 3]),
            qx,
            qy,
            qz,
            qw,
        )
        state = (
            float(T_base_lidar[0, 3]),
            float(T_base_lidar[1, 3]),
            float(T_base_lidar[2, 3]),
            qx,
            qy,
            qz,
            qw,
        )
        self._publish_base_to_lidar_tf_from_state(state, stamp)
        with self._tf_cache_lock:
            self._latest_base_to_lidar_tf = state
        return True

    @staticmethod
    def _patch_rosgraph_logger_for_py311() -> None:
        """
        Patch rosgraph's custom logger to avoid a known infinite-loop edge case.

        In some Python 3.11 environments with ROS Noetic, rosgraph's RospyLogger.findCaller
        can loop forever when walking stack frames if no matching frame is found and f_back is None.
        """
        try:
            import os
            import sys
            import inspect
            import rosgraph.roslogging as roslogging
        except Exception:
            # If rosgraph isn't importable yet, let rospy import handle errors.
            return

        if not hasattr(roslogging, "RospyLogger"):
            return

        def _safe_find_caller(self, *args, **kwargs):
            file_name, lineno, func_name = super(roslogging.RospyLogger, self).findCaller(
                *args, **kwargs
            )[:3]
            file_name = os.path.normcase(file_name)

            f = inspect.currentframe()
            if f is not None:
                f = f.f_back
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                    break
                if f.f_back:
                    f = f.f_back
                else:
                    # Critical fix: break instead of infinite loop.
                    break

            if f is None or not hasattr(f, "f_code"):
                if sys.version_info > (3, 2):
                    return file_name, lineno, func_name, None
                return file_name, lineno, func_name

            if f.f_back and f.f_code and f.f_code.co_name == "_base_logger":
                f = f.f_back
                if f.f_back:
                    f = f.f_back
            co = f.f_code
            func_name2 = co.co_name
            try:
                class_name = f.f_locals["self"].__class__.__name__
                func_name2 = f"{class_name}.{func_name2}"
            except KeyError:
                pass

            if sys.version_info > (3, 2):
                return co.co_filename, f.f_lineno, func_name2, None
            return co.co_filename, f.f_lineno, func_name2

        roslogging.RospyLogger.findCaller = _safe_find_caller

    def reset(self):
        self._step_idx = 0
        self._step_ready_bootstrap_complete = False
        self._current_step_stamp_sec = 0.0
        self._timing_frame_count = 0
        for key in self._timing_acc_ms:
            self._timing_acc_ms[key] = 0.0
        with self._lock:
            self._latest_gt_payload = None
            self._pending_step_frame_public_payload = None
            self._latest_action = None
            self._latest_action_step = -1
            self._latest_action_mono_s = 0.0
            self._last_consumed_action_step = -1
            self._latest_cmd_vel = None
            self._latest_cmd_vel_mono_s = 0.0
            self._move_base_active = False
            self._latest_step_ready = {}
            self._latest_step_ready_mono_s = 0.0
            self.last_step_ready_diagnostics = {}
            self._latest_step_capture_ack = {}
            self._latest_step_capture_ack_mono_s = 0.0
        self._last_base_position_xyz = None
        self._last_base_pose_xyyaw = None
        self._odom_step_sample = None
        with self._tf_cache_lock:
            # Do not let the keepalive publish a previous house's transform
            # while the navigation stack is resetting its map/costmaps.
            self._latest_odom_tf_state = None
            self._latest_base_to_lidar_tf = None
        if self._realtime_gt_publisher is not None:
            self._realtime_gt_publisher.reset()

    def publish_realtime_gt_now(self, step_index: int | None = None):
        if self._realtime_gt_publisher is None or self.task is None:
            return None
        stamp = self._next_common_stamp()
        payload = self._realtime_gt_publisher.publish(
            self.task,
            stamp=stamp,
            step_index=self._step_idx if step_index is None else int(step_index),
            force=True,
        )
        if payload is not None:
            self._latest_gt_payload = payload
        return payload

    def publish_public_rgb_frame(self, observation: Any, *, stamp_sec: float) -> None:
        """Publish RGB for an evaluator-owned observation, including macro views.

        This does not consume an action step or emit a recorder step marker.
        The evaluator supplies the timestamp of the same-state public GT frame.
        """
        stamp = self._rospy.Time.from_sec(float(stamp_sec))
        frame = self._extract_image_from_observation(observation)
        if frame is None:
            raise ValueError("Evaluator public observation has no RGB frame")
        msg = self._to_image_msg(frame, stamp=stamp, seq=self._step_idx)
        if msg is None:
            raise ValueError("Cannot encode evaluator public RGB frame")
        self._publish_odom_and_tf(observation, stamp)
        self._obs_pub.publish(msg)

    def queue_step_frame_public_payload(self, payload: Mapping[str, Any]) -> bool:
        """Attach an already-published public perception frame to next RGB.

        The standalone V3 evaluator owns restricted-GT publication because this
        bridge intentionally has no simulator task.  Recording must nevertheless
        preserve the same public payload that preceded the next policy RGB.  A
        deep copy makes this a self-contained snapshot and
        prevents later evaluator-side mutations from changing queued media.

        This is deliberately a next-frame one-shot: a newer post-action public
        state replaces an older pending state until an RGB is actually emitted.
        Normal task-backed realtime-GT continues to use ``_latest_gt_payload``.
        """

        if not isinstance(payload, Mapping):
            return False
        try:
            snapshot = deepcopy(dict(payload))
        except (TypeError, ValueError, RecursionError):
            return False
        if not isinstance(snapshot, dict):
            return False
        with self._lock:
            self._pending_step_frame_public_payload = snapshot
        return True

    def prepare_realtime_gt_snapshot_for_next_step(self, next_step_index: int) -> None:
        """Request a private task snapshot only when the next GT frame is due.

        The task evaluates visibility during ``task.step`` and this policy
        publishes realtime-GT immediately afterward.  Scheduling the request
        here avoids deep-copying a segmentation image on the intervening
        non-GT steps while preserving the exact same-state reuse guarantee.
        """
        task = self.task
        request_snapshot = getattr(
            task, "request_private_realtime_gt_segmentation_snapshot", None
        )
        if not callable(request_snapshot):
            return
        publisher = self._realtime_gt_publisher
        should_publish_step = getattr(publisher, "should_publish_step", None)
        due = bool(
            callable(should_publish_step)
            and should_publish_step(int(next_step_index))
        )
        request_snapshot(due)

    def prepare_episode_reset(self) -> None:
        self._episode_count += 1
        if self._episode_count <= 1:
            return
        self._rospy.logwarn("RosBridgePolicy: resetting ROS navigation state before episode %d", self._episode_count)
        self._move_base_cancel_pub.publish(self._GoalID())
        self._explorer_reset_pub.publish(self._Empty())
        self._mapping_reset_pub.publish(self._Empty())
        try:
            self._clear_costmaps.wait_for_service(timeout=2.0)
            self._clear_costmaps()
        except Exception as exc:
            self._rospy.logwarn("RosBridgePolicy: clear_costmaps during scene reset failed: %s", exc)
        self._rospy.sleep(1.0)

    def _record_timing(self, stage_ms: dict[str, float]) -> None:
        self.last_timing_ms = {
            key: float(stage_ms.get(key, 0.0)) for key in self._timing_acc_ms
        }
        if self.timing_log_every_n_frames <= 0:
            return
        self._timing_frame_count += 1
        for key in self._timing_acc_ms:
            value = float(stage_ms.get(key, 0.0))
            self._timing_acc_ms[key] += value
            self._timing_samples_ms[key].append(value)

        if self._timing_frame_count < self.timing_log_every_n_frames:
            return

        n = float(self._timing_frame_count)
        avg = {k: self._timing_acc_ms[k] / n for k in self._timing_acc_ms}
        p95 = {
            key: float(np.percentile(values, 95)) if values else 0.0
            for key, values in self._timing_samples_ms.items()
        }
        max_values = {
            key: max(values) if values else 0.0
            for key, values in self._timing_samples_ms.items()
        }
        avg_total = max(avg["total"], 1e-6)
        fps = 1000.0 / avg_total
        queue_size = self._step_frame_queue.qsize() if self._step_frame_queue is not None else 0
        queue_capacity = self._step_frame_queue.maxsize if self._step_frame_queue is not None else 0
        self._rospy.loginfo(
            (
                "RosBridgePolicy timing window=%d: total avg/p95/max=%.2f/%.2f/%.2fms "
                "(%.2fHz), action_wait=%.2f/%.2f/%.2fms post_ready=%.2fms "
                "fresh_cmd=%.2fms fresh_action=%.2fms, gt=%.2f/%.2fms hit=%.2f, "
                "rgb_total=%.2f/%.2fms [extract=%.2f encode=%.2f ros_pub=%.2f "
                "frame_enqueue=%.2f/%.2f], extra_rgb=%.2fms, "
                "depth+pcd=%.2fms, depth_scan=%.2fms [convert=%.2f pub=%.2f], "
                "odom_tf=%.2fms, republish=%.2fms, "
                "postprocess=%.2fms, step_ready=%.2fms sat=%.2f timeout=%.2f, "
                "step_sync=%.2fms, capture_ack=%.2fms, frame_queue=%d/%d peak=%d, "
                "action_source=%s timeout=%s"
            ),
            int(n),
            avg["total"],
            p95["total"],
            max_values["total"],
            fps,
            avg["action_wait"],
            p95["action_wait"],
            max_values["action_wait"],
            avg["action_wait_after_ready"],
            avg["fresh_cmd_after_gate"],
            avg["fresh_action_after_gate"],
            avg["realtime_gt"],
            p95["realtime_gt"],
            avg["realtime_gt_snapshot_hit"],
            avg["rgb_publish"],
            p95["rgb_publish"],
            avg["rgb_extract"],
            avg["rgb_encode"],
            avg["rgb_ros_publish"],
            avg["step_frame_enqueue"],
            p95["step_frame_enqueue"],
            avg["extra_rgb_extract"] + avg["extra_rgb_encode"] + avg["extra_rgb_ros_publish"],
            avg["depth_extract_intrinsics"]
            + avg["depth_msg_publish"]
            + avg["pointcloud_convert"]
            + avg["pointcloud_publish"]
            + avg["camera_info_publish"],
            avg["depth_scan_convert"] + avg["depth_scan_publish"],
            avg["depth_scan_convert"],
            avg["depth_scan_publish"],
            avg["odom_tf"],
            avg["blocking_republish"],
            avg["postprocess_action"],
            avg["step_ready_wait"],
            avg["step_ready_satisfied"],
            avg["step_ready_timed_out"],
            avg["step_sync_publish"],
            avg["step_capture_ack_wait"],
            queue_size,
            queue_capacity,
            self._step_frame_queue_peak,
            self.last_action_source or "unknown",
            self.last_action_timed_out,
        )
        self._timing_frame_count = 0
        for key in self._timing_acc_ms:
            self._timing_acc_ms[key] = 0.0
            self._timing_samples_ms[key].clear()

    def _action_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self._rospy.logwarn("RosBridgePolicy: received non-JSON action payload, ignored.")
            return

        step = int(payload.get("step", -1))
        raw_action = payload.get("action", payload)
        action = self._coerce_action(raw_action)
        with self._lock:
            self._latest_action = action
            self._latest_action_step = step
            self._latest_action_mono_s = time.monotonic()

    def _step_ready_callback(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._latest_step_ready = payload
            self._latest_step_ready_mono_s = time.monotonic()

    def _step_capture_ack_callback(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._latest_step_capture_ack = payload
            self._latest_step_capture_ack_mono_s = time.monotonic()

    def _wait_for_step_capture_ack(self, stamp) -> bool:
        """Hold the simulator until recorder froze this exact step's raw state."""
        if not self.step_capture_ack_barrier_enabled or not self.step_capture_ack_topic:
            return True
        expected_step = int(self._step_idx)
        expected_stamp = float(stamp.to_sec())
        deadline = time.monotonic() + self.step_capture_ack_timeout_s
        while not self._rospy.is_shutdown():
            with self._lock:
                payload = dict(self._latest_step_capture_ack)
            try:
                ack_step = int(payload.get("step_index"))
            except (TypeError, ValueError):
                ack_step = -1
            try:
                ack_stamp = float(payload.get("stamp_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                ack_stamp = 0.0
            if (
                bool(payload.get("ready"))
                and ack_step == expected_step
                and (ack_stamp <= 0.0 or abs(ack_stamp - expected_stamp) <= 1e-6)
            ):
                return True
            if self.step_capture_ack_timeout_s <= 0.0 or time.monotonic() >= deadline:
                self._rospy.logwarn(
                    "RosBridgePolicy: step-capture acknowledgment timeout step=%d stamp=%.6f ack_step=%s ack_stamp=%s.",
                    expected_step,
                    expected_stamp,
                    payload.get("step_index"),
                    payload.get("stamp_sec"),
                )
                return False
            time.sleep(0.001)
        return False

    def _step_ready_for_current(
        self,
        *,
        ignore_warmup: bool = False,
        require_current_stamp: bool = True,
    ) -> bool:
        if (
            not self.step_ready_barrier_enabled
            or (
                not ignore_warmup
                and self._step_idx < self.step_ready_warmup_skip_frames
            )
        ):
            return True
        with self._lock:
            payload = dict(self._latest_step_ready)
        if not bool(payload.get("ready")):
            return False
        if not require_current_stamp:
            return True
        aggregate_stamp = float(payload.get("stamp_sec", 0.0) or 0.0)
        if aggregate_stamp > 0.0 and self._current_step_stamp_sec > 0.0:
            return aggregate_stamp >= self._current_step_stamp_sec - 1e-6
        return int(payload.get("step_index", -1)) >= int(self._step_idx)

    def _cmd_vel_callback(self, msg) -> None:
        twist = msg.twist
        cmd = np.array(
            [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)],
            dtype=np.float32,
        )
        with self._lock:
            self._latest_cmd_vel = cmd
            self._latest_cmd_vel_mono_s = time.monotonic()

    def _move_base_status_callback(self, msg) -> None:
        active = any(int(status.status) == 1 for status in msg.status_list)
        with self._lock:
            self._move_base_active = active

    @staticmethod
    def _quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _cmd_vel_to_base_action(self, cmd_vel: np.ndarray, observation: Any) -> dict[str, Any] | None:
        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            return None

        px, py = float(base_pose[0]), float(base_pose[1])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        yaw = self._quat_wxyz_to_yaw(qw, qx, qy, qz)
        vx, vy, wz = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
        self.last_cmd_vel_lateral_rejected = False
        if not getattr(self, "allow_lateral_cmd_vel", True):
            if not np.isfinite(vy) or abs(vy) > 1e-6:
                # Reject the whole trajectory, not just its lateral component:
                # the remaining forward/turn motion has not been collision checked.
                self.last_cmd_vel_lateral_rejected = True
                self.lateral_cmd_vel_rejection_count += 1
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: rejecting lateral cmd_vel in nonholonomic mode "
                    "(vx=%.6f vy=%.6f wz=%.6f, rejected=%d); holding pose.",
                    vx, vy, wz, self.lateral_cmd_vel_rejection_count,
                )
                vx = wz = 0.0
            vy = 0.0
        vx *= self.cmd_vel_linear_gain
        vy *= self.cmd_vel_linear_gain
        dt = self.cmd_vel_control_dt_s

        dx_world = (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
        dy_world = (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
        target_yaw = self._wrap_to_pi(yaw + wz * dt)

        return {
            "base": np.array([px + dx_world, py + dy_world, target_yaw], dtype=np.float32),
            "done": False,
        }

    def _coerce_action(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._coerce_action(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if all(isinstance(v, (int, float)) for v in obj):
                return np.asarray(obj, dtype=np.float32)
            return [self._coerce_action(v) for v in obj]
        return obj

    def _sanitize_observation(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitize_observation(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, np.ndarray):
            # Keep payloads bounded for ROS String transport.
            if obj.size <= 256:
                return obj.tolist()
            return {
                "__ndarray_summary__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.nanmin(obj)) if obj.size > 0 else None,
                "max": float(np.nanmax(obj)) if obj.size > 0 else None,
            }
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _build_noop_action(self) -> dict[str, Any]:
        if self.task is not None:
            robot_view = self.task.env.current_robot.robot_view
            return {**robot_view.get_noop_ctrl_dict(["base"]), "done": False}
        return {"done": False}

    @staticmethod
    def _fill_missing_navigation_holds(chosen_action: dict[str, Any], robot_view: Any) -> None:
        """Complete a base-only action by holding each omitted robot group in place."""

        if "base" not in chosen_action:
            chosen_action["base"] = robot_view.get_noop_ctrl_dict(["base"])["base"]
        move_group_ids = set(robot_view.move_group_ids())
        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in move_group_ids or arm_name in chosen_action:
                continue
            chosen_action[arm_name] = np.asarray(
                robot_view.get_noop_ctrl_dict([arm_name])[arm_name], dtype=np.float32
            ).copy()

    def _extract_image_from_observation(self, observation: Any) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        # Prefer common camera keys, then fallback to first image-like tensor.
        preferred_keys = ("head_camera", "exo_camera_1", "wrist_camera", "rgb", "image")
        for key in preferred_keys:
            value = obs_dict.get(key)
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value

        for value in obs_dict.values():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value
        return None

    def _extract_named_image_from_observation(self, observation: Any, camera_name: str) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None or not camera_name:
            return None
        value = obs_dict.get(camera_name)
        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            return value
        return None

    def _extract_observation_dict(self, observation: Any) -> dict[str, Any] | None:
        if isinstance(observation, list) and len(observation) > 0:
            observation = observation[0]
        if not isinstance(observation, dict):
            return None
        return observation

    def _extract_depth_from_observation(self, observation: Any) -> tuple[str, np.ndarray] | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        preferred_key = f"{self.depth_camera_name}_depth"
        value = obs_dict.get(preferred_key)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return preferred_key, value
        return None

    def _extract_intrinsics_from_observation(
        self, observation: Any, depth_key: str
    ) -> tuple[float, float, float, float] | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        camera_name = depth_key[:-6] if depth_key.endswith("_depth") else depth_key
        sensor_key = f"sensor_param_{camera_name}"
        params = obs_dict.get(sensor_key)
        if not isinstance(params, dict):
            return None

        intrinsic_cv = params.get("intrinsic_cv")
        if intrinsic_cv is None:
            return None
        intrinsic = np.asarray(intrinsic_cv, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            return None

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        if fx <= 0 or fy <= 0:
            return None
        return fx, fy, cx, cy

    def _get_camera_fov_deg(self, camera_name: str) -> float | None:
        cam_cfg = getattr(getattr(self.config, "camera_config", None), "cameras", None)
        if cam_cfg is None:
            return None
        for cam in cam_cfg:
            if getattr(cam, "name", None) == camera_name:
                fov = getattr(cam, "fov", None)
                if fov is not None:
                    return float(fov)
        return None

    def _intrinsics_from_fov(
        self, camera_name: str, width: int, height: int
    ) -> tuple[float, float, float, float] | None:
        fov_deg = self._get_camera_fov_deg(camera_name)
        if fov_deg is None or width <= 0 or height <= 0:
            return None
        fov_rad = np.deg2rad(max(1e-3, fov_deg))
        # Treat configured fov as vertical FoV (matches sensor generation path).
        fy = (height * 0.5) / np.tan(fov_rad * 0.5)
        fx = fy
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _normalize_intrinsics_to_image_shape(
        intrinsics: tuple[float, float, float, float] | None,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        """Heuristically rescale intrinsics when source resolution mismatches depth image."""
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        if width <= 0 or height <= 0:
            return None

        # Typical principal point should be near image center; if far away, assume mismatched resolution.
        target_cx = 0.5 * width
        target_cy = 0.5 * height
        sx = target_cx / cx if cx > 1e-6 else 1.0
        sy = target_cy / cy if cy > 1e-6 else 1.0

        # Apply only when mismatch is significant to avoid perturbing valid intrinsics.
        if abs(cx - target_cx) > 0.1 * width:
            fx *= sx
            cx *= sx
        if abs(cy - target_cy) > 0.1 * height:
            fy *= sy
            cy *= sy

        if fx <= 0 or fy <= 0:
            return None
        return fx, fy, cx, cy

    def _get_pointcloud_projection_lut(
        self, height: int, width: int, fx: float, fy: float, cx: float, cy: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return cached normalized projection maps used by depth->pointcloud:
          x_cam = proj_x * depth
          y_cam = proj_y * depth
        """
        key = (
            int(height),
            int(width),
            round(float(fx), 6),
            round(float(fy), 6),
            round(float(cx), 6),
            round(float(cy), 6),
        )
        cached = self._pointcloud_projection_cache.get(key)
        if cached is not None:
            return cached

        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        proj_x_row = (u - float(cx)) / max(float(fx), 1e-6)
        proj_y_col = (v - float(cy)) / max(float(fy), 1e-6)
        proj_x = np.broadcast_to(proj_x_row[None, :], (height, width)).copy()
        proj_y = np.broadcast_to(proj_y_col[:, None], (height, width)).copy()
        self._pointcloud_projection_cache[key] = (proj_x, proj_y)
        return proj_x, proj_y

    def _depth_to_pointcloud_msg(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float] | None = None,
        stamp=None,
    ):
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)

        orig_h, orig_w = depth.shape
        stride = self.pointcloud_stride
        if self.pointcloud_stride > 1:
            depth = depth[::stride, ::stride]

        h, w = depth.shape
        if h == 0 or w == 0:
            return None

        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
            if stride > 1:
                fx /= stride
                fy /= stride
                cx /= stride
                cy /= stride
        else:
            fov_rad = np.deg2rad(max(1e-3, self.depth_fov_deg))
            fy = (h * 0.5) / np.tan(fov_rad * 0.5)
            fx = fy
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5

        valid = np.isfinite(depth) & (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if not np.any(valid):
            return None

        proj_x, proj_y = self._get_pointcloud_projection_lut(h, w, fx, fy, cx, cy)
        # Camera optical frame:
        # x_cam: right, y_cam: down, z_cam: forward
        z_cam = depth[valid]
        x_cam = proj_x[valid] * z_cam
        y_cam = proj_y[valid] * z_cam

        # Convert to robot-centric frame for mapping:
        # x: forward, y: left, z: up
        points = np.empty((z_cam.shape[0], 3), dtype=np.float32)
        points[:, 0] = z_cam
        points[:, 1] = -x_cam
        points[:, 2] = -y_cam

        if self.pointcloud_self_filter_radius_m > 0.0:
            horizontal_radius_sq = points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]
            keep = horizontal_radius_sq > self.pointcloud_self_filter_radius_m**2
            points = points[keep]
            if points.shape[0] == 0:
                return None

        # Optional roll correction around forward axis (robot +x).
        # Positive follows right-hand rule; if you observe clockwise tilt in view,
        # use a negative value to compensate.
        if abs(self.pointcloud_roll_correction_deg) > 1e-6:
            theta = np.deg2rad(self.pointcloud_roll_correction_deg)
            c, s = np.cos(theta), np.sin(theta)
            y_corr = points[:, 1] * c - points[:, 2] * s
            z_corr = points[:, 1] * s + points[:, 2] * c
            points[:, 1] = y_corr
            points[:, 2] = z_corr

        msg = self._PointCloud2()
        msg.header.seq = int(self._step_idx)
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        msg.header.frame_id = self.pointcloud_frame_id
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            self._PointField(name="x", offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="y", offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="z", offset=8, datatype=self._PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.tobytes()
        return msg

    def _depth_to_organized_scan_msg(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        base_from_lidar: np.ndarray,
        stamp=None,
    ):
        projection = self._organized_depth_scan_projector.project(
            depth,
            intrinsics,
            base_from_lidar,
        )
        msg = self._LaserScan()
        msg.header.seq = int(self._step_idx)
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        # The projection has already leveled every return in base coordinates.
        # Publishing in base keeps GMapping's planar-laser invariant explicit.
        msg.header.frame_id = self.base_frame_id
        msg.angle_min = float(projection.angle_min_rad)
        msg.angle_increment = float(projection.angle_increment_rad)
        msg.angle_max = float(
            projection.angle_min_rad
            + (projection.ranges_m.size - 1) * projection.angle_increment_rad
        )
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = float(projection.range_min_m)
        msg.range_max = float(projection.range_max_m)
        msg.ranges = projection.ranges_m.tolist()
        msg.intensities = projection.intensities.tolist()
        return msg, projection.diagnostics

    def _to_depth_msg(self, depth: np.ndarray, stamp=None):
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)
        msg = self._Image()
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        msg.header.frame_id = self.optical_frame_id
        msg.height = int(depth.shape[0])
        msg.width = int(depth.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = int(depth.shape[1] * 4)
        msg.data = depth.tobytes()
        return msg

    def _build_camera_info_msg(
        self,
        width: int,
        height: int,
        intrinsics: tuple[float, float, float, float] | None,
        stamp=None,
    ):
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        info = self._CameraInfo()
        info.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        info.header.frame_id = self.optical_frame_id
        info.width = int(width)
        info.height = int(height)
        info.distortion_model = "plumb_bob"
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _to_image_msg(self, frame: np.ndarray, stamp=None, seq: int | None = None):
        img = frame
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                # Best effort normalization for float images.
                max_val = float(np.nanmax(img)) if img.size > 0 else 1.0
                if max_val <= 1.0:
                    img = np.clip(img, 0.0, 1.0) * 255.0
                img = np.nan_to_num(img, nan=0.0)
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            h, w = img.shape
            encoding = "mono8"
            step = w
            data = img.tobytes()
        elif img.ndim == 3 and img.shape[2] >= 3:
            # Keep first 3 channels as rgb8.
            img = img[:, :, :3]
            h, w, _ = img.shape
            encoding = "rgb8"
            step = w * 3
            data = img.tobytes()
        else:
            return None

        msg = self._Image()
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        if seq is not None:
            msg.header.seq = int(seq)
        msg.header.frame_id = self.optical_frame_id
        msg.height = h
        msg.width = w
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = data
        return msg

    def _extract_base_pose_from_observation(self, observation: Any) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None
        pose = obs_dict.get("robot_base_pose")
        if pose is None:
            return None
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.size < 7:
            return None
        return pose_arr[:7]

    @classmethod
    def _estimate_planar_twist(
        cls,
        previous_pose_xyyaw: np.ndarray,
        current_pose_xyyaw: np.ndarray,
        dt_s: float,
    ) -> tuple[float, float, float]:
        dt_s = max(1e-6, float(dt_s))
        dx = float(current_pose_xyyaw[0] - previous_pose_xyyaw[0])
        dy = float(current_pose_xyyaw[1] - previous_pose_xyyaw[1])
        yaw_delta = cls._wrap_to_pi(float(current_pose_xyyaw[2] - previous_pose_xyyaw[2]))
        midpoint_yaw = float(previous_pose_xyyaw[2]) + 0.5 * yaw_delta
        vx = (np.cos(midpoint_yaw) * dx + np.sin(midpoint_yaw) * dy) / dt_s
        vy = (-np.sin(midpoint_yaw) * dx + np.cos(midpoint_yaw) * dy) / dt_s
        return float(vx), float(vy), float(yaw_delta / dt_s)

    @staticmethod
    def _world_planar_velocity_to_body(
        vx_world: float,
        vy_world: float,
        wz: float,
        yaw: float,
    ) -> tuple[float, float, float]:
        vx = np.cos(yaw) * vx_world + np.sin(yaw) * vy_world
        vy = -np.sin(yaw) * vx_world + np.cos(yaw) * vy_world
        return float(vx), float(vy), float(wz)

    def _step_delta_twist(self, pose_xyyaw: np.ndarray, *, position_jump: bool = False):
        """Report executed motion once per control step, stable under republish."""
        sample = getattr(self, "_odom_step_sample", None)
        step = int(self._step_idx)
        if sample is not None and sample[0] == step and not position_jump:
            return sample[2]
        twist = (0.0, 0.0, 0.0)
        if sample is not None and step > sample[0] and not position_jump:
            twist = self._estimate_planar_twist(
                sample[1], pose_xyyaw, self.cmd_vel_control_dt_s * (step - sample[0])
            )
        self._odom_step_sample = (step, pose_xyyaw.copy(), twist)
        return twist

    def _extract_planar_twist_from_task(self, yaw: float) -> tuple[float, float, float] | None:
        if self.task is None:
            return None
        try:
            base_group = self.task.env.current_robot.robot_view.get_move_group("base")
            joint_vel = np.asarray(base_group.joint_vel, dtype=np.float64).reshape(-1)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if joint_vel.size != 3 or not np.all(np.isfinite(joint_vel)):
            return None
        return self._world_planar_velocity_to_body(
            float(joint_vel[0]),
            float(joint_vel[1]),
            float(joint_vel[2]),
            yaw,
        )

    def _next_common_stamp(self):
        with self._stamp_lock:
            stamp = self._rospy.Time.now()
            stamp_s = float(stamp.to_sec())
            if self._last_common_stamp_s is not None and stamp_s <= self._last_common_stamp_s:
                # Keep publish timestamps strictly monotonic to avoid occasional time back-jumps.
                stamp_s = self._last_common_stamp_s + 1e-6
                stamp = self._rospy.Time.from_sec(stamp_s)
            self._last_common_stamp_s = stamp_s
            return stamp

    def _publish_odom_and_base_tf_from_state(self, state: tuple[float, ...], stamp) -> None:
        """Publish a numeric odom/base snapshot with the supplied ROS timestamp."""

        px, py, pz, qx, qy, qz, qw, twist_x, twist_y, twist_yaw = state
        odom_msg = self._Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.base_frame_id
        odom_msg.pose.pose.position.x = px
        odom_msg.pose.pose.position.y = py
        odom_msg.pose.pose.position.z = pz
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = twist_x
        odom_msg.twist.twist.linear.y = twist_y
        odom_msg.twist.twist.angular.z = twist_yaw
        self._odom_pub.publish(odom_msg)

        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame_id
        tf_msg.child_frame_id = self.base_frame_id
        tf_msg.transform.translation.x = px
        tf_msg.transform.translation.y = py
        tf_msg.transform.translation.z = pz
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_base_to_lidar_tf_from_state(self, state: tuple[float, ...], stamp) -> None:
        """Publish a cached base-to-lidar transform without touching MuJoCo."""

        x, y, z, qx, qy, qz, qw = state
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.base_frame_id
        tf_msg.child_frame_id = self.pointcloud_frame_id
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf_msg)

    def _tf_keepalive_callback(self, _event: Any) -> None:
        """Refresh only public pose TF while simulator control is synchronously blocked."""

        if not self.publish_odom:
            return
        with self._tf_cache_lock:
            odom_state = self._latest_odom_tf_state
            lidar_state = self._latest_base_to_lidar_tf
        if odom_state is None:
            return
        if self.base_frame_id != self.pointcloud_frame_id and lidar_state is None:
            return
        try:
            stamp = self._next_common_stamp()
            self._publish_odom_and_base_tf_from_state(odom_state, stamp)
            if lidar_state is not None:
                self._publish_base_to_lidar_tf_from_state(lidar_state, stamp)
        except Exception as exc:
            # The timer must never terminate a rollout because ROS is shutting down.
            self._rospy.logwarn_throttle(
                5.0,
                "RosBridgePolicy: TF keepalive publication failed: %s",
                str(exc),
            )

    def _publish_odom_and_tf(self, observation: Any, stamp) -> bool:
        if not self.publish_odom:
            return True

        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            self._rospy.logwarn_throttle(
                5.0,
                "RosBridgePolicy: robot_base_pose missing; /odom and odom->base TF not published.",
            )
            return False

        # robot_base_pose layout: [x, y, z, qw, qx, qy, qz]
        px, py, pz = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        curr_pos = np.array([px, py, pz], dtype=np.float32)
        curr_yaw = self._quat_wxyz_to_yaw(qw, qx, qy, qz)
        curr_pose_xyyaw = np.array([px, py, curr_yaw], dtype=np.float32)
        twist_x = 0.0
        twist_y = 0.0
        twist_yaw = 0.0
        position_jump = False
        if self._last_base_position_xyz is not None:
            jump_dist = float(np.linalg.norm(curr_pos - self._last_base_position_xyz))
            if jump_dist > self._base_position_jump_warn_m:
                position_jump = True
                self._rospy.logwarn(
                    (
                        "RosBridgePolicy: detected base position jump > %.2fm (dist=%.3fm). "
                        "prev=(%.3f, %.3f, %.3f), curr=(%.3f, %.3f, %.3f), step=%d"
                    ),
                    self._base_position_jump_warn_m,
                    jump_dist,
                    float(self._last_base_position_xyz[0]),
                    float(self._last_base_position_xyz[1]),
                    float(self._last_base_position_xyz[2]),
                    px,
                    py,
                    pz,
                    self._step_idx,
                )
        if self.publish_odom_twist and getattr(self, "odom_twist_source", "instantaneous") == "step_delta":
            twist_x, twist_y, twist_yaw = self._step_delta_twist(
                curr_pose_xyyaw, position_jump=position_jump
            )
        elif self.publish_odom_twist and not position_jump:
            instantaneous_twist = self._extract_planar_twist_from_task(curr_yaw)
            if instantaneous_twist is not None:
                twist_x, twist_y, twist_yaw = instantaneous_twist
            elif self._last_base_pose_xyyaw is not None:
                twist_x, twist_y, twist_yaw = self._estimate_planar_twist(
                    self._last_base_pose_xyyaw,
                    curr_pose_xyyaw,
                    self.cmd_vel_control_dt_s,
                )
        self._last_base_position_xyz = curr_pos
        self._last_base_pose_xyyaw = curr_pose_xyyaw
        odom_state = (px, py, pz, qx, qy, qz, qw, twist_x, twist_y, twist_yaw)
        self._publish_odom_and_base_tf_from_state(odom_state, stamp)
        with self._tf_cache_lock:
            self._latest_odom_tf_state = odom_state
        return self._publish_base_to_lidar_tf(observation, stamp)

    def _republish_observation_messages(
        self,
        observation: Any,
        messages: dict[str, Any],
        *,
        force_pointcloud: bool = False,
        stamp=None,
    ):
        stamp = self._next_common_stamp() if stamp is None else stamp
        self._publish_odom_and_tf(observation, stamp)

        # RGB topics are recording-only in the realtime-GT ROS pipeline. Do not
        # duplicate them while waiting for a fresh navigation command; repeated
        # large images create recorder backlog without adding mapping evidence.
        topic_messages = [(self._depth_pub, messages.get("depth"))]
        if force_pointcloud or self.blocking_republish_pointcloud:
            topic_messages.append((self._pointcloud_pub, messages.get("pointcloud")))
            topic_messages.append((self._depth_scan_pub, messages.get("depth_scan")))
        for publisher, message in topic_messages:
            if publisher is None or message is None:
                continue
            message.header.stamp = stamp
            publisher.publish(message)

        camera_info = messages.get("camera_info")
        if camera_info is not None:
            camera_info.header.stamp = stamp
            self._camera_info_pub.publish(camera_info)
            self._image_camera_info_pub.publish(camera_info)
            self._depth_camera_info_pub.publish(camera_info)
        return stamp

    def _step_ready_payload(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_step_ready)

    @staticmethod
    def _step_ready_stage_evidence(payload: dict[str, Any]) -> dict[str, Any]:
        """Keep the causal evidence compact enough for per-step JSON traces."""

        payload = payload if isinstance(payload, dict) else {}
        modules = payload.get("modules") or {}
        mapping = modules.get("semantic_mapping") or {}
        return {
            "aggregate": {
                "ready": bool(payload.get("ready")),
                "step_index": payload.get("step_index"),
                "stamp_sec": payload.get("stamp_sec"),
                "source_alignment_required": bool(
                    payload.get("source_alignment_required", False)
                ),
                "source_aligned": payload.get("source_aligned"),
                "source_match_mode": payload.get("source_match_mode"),
                "source_identity_key": payload.get("source_identity_key"),
                "source_tuples": payload.get("source_tuples") or {},
                "source_identities": payload.get("source_identities") or {},
                "missing_modules": payload.get("missing_modules") or [],
            },
            "semantic_mapping": {
                "causal_contract": mapping.get("causal_contract"),
                "raw_occ_ready": mapping.get("raw_occ_ready"),
                "room_segmentation_ready": mapping.get("room_segmentation_ready"),
                "unified_graph_ready": mapping.get("unified_graph_ready"),
                "missing_stages": mapping.get("missing_stages") or [],
                "occupancy_source": mapping.get("occupancy_source") or {},
                "room_segmentation_source": mapping.get(
                    "room_segmentation_source"
                )
                or {},
                "room_commit_source": mapping.get("room_commit_source") or {},
                "unified_graph_room_source": mapping.get(
                    "unified_graph_room_source"
                )
                or {},
                "room_commit_graph_revision": mapping.get(
                    "room_commit_graph_revision"
                ),
                "published_graph_revision": mapping.get(
                    "published_graph_revision"
                ),
                "room_commit_latency_ms": mapping.get("room_commit_latency_ms"),
                "room_worker_total_ms": mapping.get("room_worker_total_ms"),
            },
        }

    def _record_step_ready_outcome(
        self,
        *,
        phase: str,
        outcome: str,
        wait_started_mono_s: float,
        payload: dict[str, Any],
    ) -> None:
        """Record the precise readiness proof or timeout for this step."""

        expected = {
            "step_index": int(self._step_idx),
            "stamp_sec": float(self._current_step_stamp_sec),
        }
        self.last_step_ready_diagnostics = {
            "enabled": bool(self.step_ready_barrier_enabled),
            "phase": str(phase),
            "outcome": str(outcome),
            "ready_satisfied": str(outcome) == "satisfied",
            "timed_out": str(outcome) == "timeout",
            "wait_ms": max(0.0, (time.monotonic() - wait_started_mono_s) * 1000.0),
            "expected_source": expected,
            "stage_evidence": self._step_ready_stage_evidence(payload),
        }

    def _wait_for_step_ready(
        self,
        *,
        timeout_s: float,
        ignore_warmup: bool,
        require_current_stamp: bool = True,
        observation: Any | None = None,
        messages: dict[str, Any] | None = None,
        force_pointcloud_republish: bool = False,
        republish_stamp=None,
        reason: str = "barrier",
    ) -> bool:
        """Wait for aggregate decision readiness for the current observation.

        Bootstrap resends only depth/pointcloud/camera-info at the *same* shared
        stamp.  One bounded retry handles late subscriber connection without
        continually advancing the required stamp faster than map consumers can
        process it, and without duplicating RGB recordings while paused.
        """

        wait_started_mono_s = time.monotonic()
        deadline = wait_started_mono_s + max(0.0, float(timeout_s))
        republish_period_s = self.step_ready_bootstrap_republish_period_s
        remaining_republishes = 1 if observation is not None and messages is not None else 0
        next_republish_mono = (
            time.monotonic() + republish_period_s
            if remaining_republishes
            else None
        )
        while not self._rospy.is_shutdown():
            if self._step_ready_for_current(
                ignore_warmup=ignore_warmup,
                require_current_stamp=require_current_stamp,
            ):
                self._record_step_ready_outcome(
                    phase=reason,
                    outcome="satisfied",
                    wait_started_mono_s=wait_started_mono_s,
                    payload=self._step_ready_payload(),
                )
                return True
            now_mono = time.monotonic()
            if timeout_s <= 0.0 or now_mono >= deadline:
                ready_payload = self._step_ready_payload()
                self._record_step_ready_outcome(
                    phase=reason,
                    outcome="timeout",
                    wait_started_mono_s=wait_started_mono_s,
                    payload=ready_payload,
                )
                self._rospy.logwarn(
                    "RosBridgePolicy: %s step-ready timeout step=%d "
                    "current_stamp=%.6f aggregate_ready=%s aggregate_step=%s "
                    "aggregate_stamp=%s modules=%s.",
                    reason,
                    self._step_idx,
                    self._current_step_stamp_sec,
                    bool(ready_payload.get("ready")),
                    ready_payload.get("step_index"),
                    ready_payload.get("stamp_sec"),
                    ready_payload.get("modules", {}),
                )
                return False
            if next_republish_mono is not None and now_mono >= next_republish_mono:
                stamp = self._republish_observation_messages(
                    observation,
                    messages,
                    force_pointcloud=force_pointcloud_republish,
                    stamp=republish_stamp,
                )
                if republish_stamp is None:
                    self._current_step_stamp_sec = float(stamp.to_sec())
                remaining_republishes -= 1
                next_republish_mono = (
                    now_mono + republish_period_s
                    if remaining_republishes > 0
                    else None
                )
            time.sleep(0.005)

    def _enqueue_step_frame(self, image_msg, stamp, step_index: int) -> None:
        if self._step_frame_thread is None or image_msg is None:
            return
        with self._lock:
            # An externally published V3 payload is consumed exactly once by
            # the next RGB.  Do not clear it on a skipped/failed RGB encode:
            # the following successful image is still the first policy frame
            # that can causally observe that state.
            gt_payload = self._pending_step_frame_public_payload
            if gt_payload is not None:
                self._pending_step_frame_public_payload = None
            else:
                gt_payload = self._latest_gt_payload
        self._step_frame_queue.put(
            (
                int(step_index),
                float(stamp.to_sec()),
                int(image_msg.width),
                int(image_msg.height),
                bytes(image_msg.data),
                gt_payload,
            )
        )

    def _publish_step_sync(self, stamp) -> None:
        if self._step_sync_pub is None:
            return
        payload = {
            "step_index": int(self._step_idx),
            "stamp_sec": float(stamp.to_sec()),
            # This lets a step-indexed controller distinguish a command that
            # actually became evaluator action N from a late relay publish.
            "action_source": str(self.last_action_source or ""),
            # cmd_vel is converted to one fixed target increment per evaluator
            # action.  Record the contract instead of inferring motion from
            # wall-clock callback timing.
            "cmd_vel_control_dt_s": float(self.cmd_vel_control_dt_s),
        }
        self._step_sync_pub.publish(
            self._String(data=json.dumps(payload, separators=(",", ":")))
        )

    def _publish_fresh_command_gate(self, stamp) -> None:
        """Announce that this observation's fresh-cmd wait is open."""

        if self._fresh_command_gate_pub is None:
            return
        payload = {
            "step_index": int(self._step_idx),
            "stamp_sec": float(stamp.to_sec()),
        }
        self._fresh_command_gate_pub.publish(
            self._String(data=json.dumps(payload, separators=(",", ":")))
        )

    def _run_step_frame_writer(self) -> None:
        import cv2

        while True:
            job = self._step_frame_queue.get()
            try:
                if job is None:
                    return
                step_index, stamp_sec, width, height, rgb_bytes, gt_payload = job
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
                frame_path = self.step_frame_dir / f"step_{step_index:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                record = {
                    "step_index": step_index,
                    "stamp_sec": stamp_sec,
                    "width": width,
                    "height": height,
                    "frame": str(frame_path),
                    "gt_observations": gt_payload,
                }
                self._step_frame_manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                self._step_frame_manifest.flush()
            finally:
                self._step_frame_queue.task_done()

    def get_action(self, observation, *, hold_navigation: bool = False):
        self.last_action_timed_out = False
        self.last_action_source = ""
        frame_t0 = time.perf_counter()
        stage_ms = {k: 0.0 for k in self._timing_acc_ms}
        published_messages: dict[str, Any] = {}

        t0 = time.perf_counter()
        common_stamp = self._next_common_stamp()
        self._current_step_stamp_sec = float(common_stamp.to_sec())
        # Do not carry provenance from a previous frame when this step has no
        # organized-depth input or projection failure.
        self._latest_depth_scan_diagnostics = {}
        self.last_depth_scan_diagnostics = {}
        self.last_step_ready_diagnostics = {
            "enabled": bool(self.step_ready_barrier_enabled),
            "phase": "not_attempted",
            "outcome": "not_attempted",
            "ready_satisfied": False,
            "timed_out": False,
            "wait_ms": 0.0,
            "expected_source": {
                "step_index": int(self._step_idx),
                "stamp_sec": float(self._current_step_stamp_sec),
            },
            "stage_evidence": {},
        }
        tf_ready = self._publish_odom_and_tf(observation, common_stamp)
        stage_ms["odom_tf"] = (time.perf_counter() - t0) * 1000.0
        if self._realtime_gt_publisher is not None:
            t0 = time.perf_counter()
            gt_payload = self._realtime_gt_publisher.publish(
                self.task,
                stamp=common_stamp,
                step_index=self._step_idx,
            )
            if gt_payload is not None:
                self._latest_gt_payload = gt_payload
            stage_ms["realtime_gt"] = (time.perf_counter() - t0) * 1000.0
            stage_ms["realtime_gt_snapshot_hit"] = float(
                bool(getattr(self._realtime_gt_publisher, "last_snapshot_used", False))
            )

        bootstrap_requires_mapping = (
            self.step_ready_barrier_enabled
            and not self._step_ready_bootstrap_complete
        )
        skip_mapping_observation = (
            (self._step_idx < self.map_warmup_skip_frames and not bootstrap_requires_mapping)
            or (not tf_ready)
        )
        if skip_mapping_observation:
            if self._step_idx < self.map_warmup_skip_frames:
                self._rospy.loginfo_throttle(
                    2.0,
                    "RosBridgePolicy: map warmup active, skipping observation publish (%d/%d)",
                    self._step_idx + 1,
                    self.map_warmup_skip_frames,
                )
            elif not tf_ready:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: odom/tf not ready this frame, skip depth/pointcloud publish to keep timestamps aligned.",
                )

        rgb_block_t0 = time.perf_counter()
        t0 = time.perf_counter()
        frame = self._extract_image_from_observation(observation)
        stage_ms["rgb_extract"] = (time.perf_counter() - t0) * 1000.0
        if frame is not None:
            t0 = time.perf_counter()
            msg = self._to_image_msg(frame, stamp=common_stamp, seq=self._step_idx)
            stage_ms["rgb_encode"] = (time.perf_counter() - t0) * 1000.0
            if msg is not None:
                t0 = time.perf_counter()
                self._obs_pub.publish(msg)
                stage_ms["rgb_ros_publish"] = (time.perf_counter() - t0) * 1000.0
                published_messages["rgb"] = msg
                t0 = time.perf_counter()
                self._enqueue_step_frame(msg, common_stamp, self._step_idx)
                stage_ms["step_frame_enqueue"] = (time.perf_counter() - t0) * 1000.0
                self._step_frame_queue_peak = max(
                    self._step_frame_queue_peak, self._step_frame_queue.qsize()
                )
            else:
                self._rospy.logwarn_throttle(2.0, "RosBridgePolicy: failed to encode image.")
        else:
            self._rospy.logwarn_throttle(
                2.0,
                "RosBridgePolicy: no image-like tensor found in observation."
            )
        if self._extra_image_pub is not None:
            t0 = time.perf_counter()
            extra_frame = self._extract_named_image_from_observation(observation, self.extra_image_camera_name)
            stage_ms["extra_rgb_extract"] = (time.perf_counter() - t0) * 1000.0
            if extra_frame is not None:
                t0 = time.perf_counter()
                extra_msg = self._to_image_msg(
                    extra_frame,
                    stamp=common_stamp,
                    seq=self._step_idx,
                )
                stage_ms["extra_rgb_encode"] = (time.perf_counter() - t0) * 1000.0
                if extra_msg is not None:
                    t0 = time.perf_counter()
                    self._extra_image_pub.publish(extra_msg)
                    stage_ms["extra_rgb_ros_publish"] = (
                        time.perf_counter() - t0
                    ) * 1000.0
                    published_messages["extra_rgb"] = extra_msg
        stage_ms["rgb_publish"] = (time.perf_counter() - rgb_block_t0) * 1000.0

        if self.publish_pointcloud and not skip_mapping_observation:
            t0_depth_extract = time.perf_counter()
            depth_data = self._extract_depth_from_observation(observation)
            if depth_data is not None:
                depth_key, depth = depth_data
                camera_name = depth_key[:-6] if depth_key.endswith("_depth") else depth_key
                intrinsics = self._extract_intrinsics_from_observation(observation, depth_key)
                intrinsics = self._normalize_intrinsics_to_image_shape(
                    intrinsics, width=depth.shape[1], height=depth.shape[0]
                )
                # If principal-point scaling is anisotropic, prefer FoV-based intrinsics
                # to avoid aspect-ratio distortion in RViz DepthCloud.
                if intrinsics is not None:
                    fx, fy, cx, cy = intrinsics

                    ratio = fx / max(fy, 1e-6)
                    if ratio < 0.9 or ratio > 1.1:
                        fov_intrinsics = self._intrinsics_from_fov(
                            camera_name, width=depth.shape[1], height=depth.shape[0]
                        )
                        if fov_intrinsics is not None:
                            intrinsics = fov_intrinsics
                elif camera_name:
                    intrinsics = self._intrinsics_from_fov(
                        camera_name, width=depth.shape[1], height=depth.shape[0]
                    )
                # For the selected depth camera (default head_camera), enforce FoV-based
                # intrinsics to avoid mixed-resolution artifacts from sensor_param_*.
                if camera_name == self.depth_camera_name:
                    fov_intrinsics = self._intrinsics_from_fov(
                        camera_name, width=depth.shape[1], height=depth.shape[0]
                    )
                    if fov_intrinsics is not None:
                        intrinsics = fov_intrinsics
                stage_ms["depth_extract_intrinsics"] = (
                    time.perf_counter() - t0_depth_extract
                ) * 1000.0

                stamp = common_stamp
                t0_depth_msg = time.perf_counter()
                depth_msg = self._to_depth_msg(depth, stamp=stamp)
                self._depth_pub.publish(depth_msg)
                published_messages["depth"] = depth_msg
                stage_ms["depth_msg_publish"] = (time.perf_counter() - t0_depth_msg) * 1000.0
                if self._depth_scan_pub is not None:
                    t0_depth_scan = time.perf_counter()
                    base_from_lidar = self._extract_lidar_pose_rel_base(observation)
                    if base_from_lidar is not None:
                        base_from_lidar = self._apply_lidar_calibration(base_from_lidar)
                        try:
                            depth_scan_msg, depth_scan_diagnostics = self._depth_to_organized_scan_msg(
                                depth,
                                intrinsics=intrinsics,
                                base_from_lidar=base_from_lidar,
                                stamp=stamp,
                            )
                            stage_ms["depth_scan_convert"] = (
                                time.perf_counter() - t0_depth_scan
                            ) * 1000.0
                            t0_depth_scan_publish = time.perf_counter()
                            published_messages["depth_scan"] = depth_scan_msg
                            self._depth_scan_pub.publish(depth_scan_msg)
                            stage_ms["depth_scan_publish"] = (
                                time.perf_counter() - t0_depth_scan_publish
                            ) * 1000.0
                            self._latest_depth_scan_diagnostics = depth_scan_diagnostics
                            self.last_depth_scan_diagnostics = dict(depth_scan_diagnostics)
                        except (TypeError, ValueError, FloatingPointError) as exc:
                            self._rospy.logwarn_throttle(
                                2.0,
                                "RosBridgePolicy: organized depth scan projection failed: %s",
                                exc,
                            )
                    else:
                        self._rospy.logwarn_throttle(
                            2.0,
                            "RosBridgePolicy: no current base<-lidar transform; depth scan skipped.",
                        )
                    if stage_ms["depth_scan_convert"] <= 0.0:
                        stage_ms["depth_scan_convert"] = (
                            time.perf_counter() - t0_depth_scan
                        ) * 1000.0
                t0_pcd_convert = time.perf_counter()
                cloud_msg = self._depth_to_pointcloud_msg(depth, intrinsics=intrinsics, stamp=stamp)
                stage_ms["pointcloud_convert"] = (time.perf_counter() - t0_pcd_convert) * 1000.0
                if cloud_msg is not None:
                    t0_pcd_pub = time.perf_counter()
                    self._pointcloud_pub.publish(cloud_msg)
                    published_messages["pointcloud"] = cloud_msg
                    stage_ms["pointcloud_publish"] = (time.perf_counter() - t0_pcd_pub) * 1000.0
                    if self.publish_camera_info:
                        t0_cam_info = time.perf_counter()
                        info_msg = self._build_camera_info_msg(
                            width=depth.shape[1],
                            height=depth.shape[0],
                            intrinsics=intrinsics,
                            stamp=stamp,
                        )
                        if info_msg is not None:
                            self._camera_info_pub.publish(info_msg)
                            self._image_camera_info_pub.publish(info_msg)
                            self._depth_camera_info_pub.publish(info_msg)
                            published_messages["camera_info"] = info_msg
                        stage_ms["camera_info_publish"] = (
                            time.perf_counter() - t0_cam_info
                        ) * 1000.0
                else:
                    self._rospy.logwarn_throttle(
                        2.0,
                        "RosBridgePolicy: depth found but no valid points for PointCloud2.",
                    )
            else:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: no depth tensor found; PointCloud2 not published.",
                )

        if self.immediate_noop_after_publish:
            # Debug mode: isolate simulator publish/step throughput from policy inference/wait time.
            t0_post = time.perf_counter()
            chosen_action = self._build_noop_action()
            if isinstance(chosen_action, dict):
                chosen_action.setdefault("done", False)
            stage_ms["postprocess_action"] = (time.perf_counter() - t0_post) * 1000.0
            self.last_action_source = "immediate_noop"
            t0_sync = time.perf_counter()
            self._publish_step_sync(common_stamp)
            stage_ms["step_sync_publish"] = (time.perf_counter() - t0_sync) * 1000.0
            t0_capture_ack = time.perf_counter()
            self._wait_for_step_capture_ack(common_stamp)
            stage_ms["step_capture_ack_wait"] = (
                time.perf_counter() - t0_capture_ack
            ) * 1000.0
            stage_ms["total"] = (time.perf_counter() - frame_t0) * 1000.0
            self._record_timing(stage_ms)
            self._step_idx += 1
            return chosen_action

        t0_wait = time.perf_counter()
        ready_wait_t0 = time.perf_counter()
        if self.step_ready_barrier_enabled:
            if not self._step_ready_bootstrap_complete:
                bootstrap_ready = self._wait_for_step_ready(
                    timeout_s=self.step_ready_bootstrap_timeout_s,
                    ignore_warmup=True,
                    require_current_stamp=True,
                    observation=observation,
                    messages=published_messages,
                    force_pointcloud_republish=True,
                    republish_stamp=common_stamp,
                    reason="bootstrap",
                )
                if bootstrap_ready:
                    self._step_ready_bootstrap_complete = True
                    self._rospy.loginfo(
                        "RosBridgePolicy: step-ready bootstrap complete at step=%d stamp=%.6f.",
                        self._step_idx,
                        self._current_step_stamp_sec,
                    )
            if self._step_ready_bootstrap_complete:
                # A ready message from an older sensor stamp is insufficient for
                # recording or planning this simulator frame.  Each step waits
                # for the aggregate that covers its own published observation.
                self._wait_for_step_ready(
                    timeout_s=self.step_ready_timeout_s,
                    ignore_warmup=False,
                    require_current_stamp=True,
                    reason="barrier",
                )
        else:
            self.last_step_ready_diagnostics = {
                "enabled": False,
                "phase": "disabled",
                "outcome": "not_required",
                "ready_satisfied": False,
                "timed_out": False,
                "wait_ms": 0.0,
                "expected_source": {
                    "step_index": int(self._step_idx),
                    "stamp_sec": float(self._current_step_stamp_sec),
                },
                "stage_evidence": {},
            }
        stage_ms["step_ready_wait"] = (
            time.perf_counter() - ready_wait_t0
        ) * 1000.0
        stage_ms["step_ready_satisfied"] = float(
            bool(self.last_step_ready_diagnostics.get("ready_satisfied"))
        )
        stage_ms["step_ready_timed_out"] = float(
            bool(self.last_step_ready_diagnostics.get("timed_out"))
        )
        wait_start_mono = time.monotonic()
        self._publish_fresh_command_gate(common_stamp)
        deadline = (
            wait_start_mono + self.action_timeout_s if self.action_timeout_s > 0.0 else None
        )
        next_republish_mono = (
            wait_start_mono + self.blocking_observation_republish_period_s
            if self.blocking_observation_republish_period_s > 0.0
            else None
        )
        chosen_action = None
        if (
            hold_navigation
            and (
                not self.step_ready_barrier_enabled
                or self.last_step_ready_diagnostics.get("ready_satisfied", False)
            )
            and not self._rospy.is_shutdown()
        ):
            # The caller owns navigation for this step; retain ready/capture barriers.
            chosen_action = self._build_noop_action()
            self.last_action_source = "navigation_hold"
        while chosen_action is None and not self._rospy.is_shutdown():
            now_mono = time.monotonic()
            if deadline is not None and now_mono >= deadline:
                break

            cmd_vel = None
            cmd_vel_ts = 0.0
            action_ts = 0.0
            move_base_active = False
            with self._lock:
                if self._latest_action is not None:
                    action_ts = self._latest_action_mono_s
                    action_is_fresh = action_ts >= wait_start_mono
                    if self._latest_action_step < 0 and action_is_fresh:
                        # Allow action payloads without explicit step field.
                        chosen_action = self._latest_action
                        self.last_action_source = "action_topic"
                        stage_ms["fresh_action_after_gate"] = max(
                            0.0, (action_ts - wait_start_mono) * 1000.0
                        )
                        self._last_consumed_action_step += 1
                        break
                    if (
                        self._latest_action_step > self._last_consumed_action_step
                        and action_is_fresh
                    ):
                        chosen_action = self._latest_action
                        self.last_action_source = "action_topic"
                        stage_ms["fresh_action_after_gate"] = max(
                            0.0, (action_ts - wait_start_mono) * 1000.0
                        )
                        self._last_consumed_action_step = self._latest_action_step
                        break
                cmd_vel = self._latest_cmd_vel
                cmd_vel_ts = self._latest_cmd_vel_mono_s
                move_base_active = self._move_base_active

            cmd_is_recent = cmd_vel is not None and (now_mono - cmd_vel_ts) <= self.cmd_vel_timeout_s
            cmd_is_fresh = (not self.require_fresh_cmd_vel) or cmd_vel_ts >= wait_start_mono
            nav_is_ready = (
                (not self.require_move_base_active_for_cmd_vel) or move_base_active
            )
            if cmd_is_recent and cmd_is_fresh and nav_is_ready:
                cmd_action = self._cmd_vel_to_base_action(cmd_vel, observation)
                if cmd_action is not None:
                    chosen_action = cmd_action
                    self.last_action_source = (
                        "lateral_cmd_vel_rejected"
                        if self.last_cmd_vel_lateral_rejected else "cmd_vel"
                    )
                    stage_ms["fresh_cmd_after_gate"] = max(
                        0.0, (cmd_vel_ts - wait_start_mono) * 1000.0
                    )
                    break

            if next_republish_mono is not None and now_mono >= next_republish_mono:
                republish_t0 = time.perf_counter()
                self._republish_observation_messages(observation, published_messages)
                stage_ms["blocking_republish"] += (
                    time.perf_counter() - republish_t0
                ) * 1000.0
                next_republish_mono = now_mono + self.blocking_observation_republish_period_s
                self._rospy.loginfo_throttle(
                    5.0,
                    "RosBridgePolicy: blocking for a fresh navigation command; observation republished.",
                )
            time.sleep(0.005)
        stage_ms["action_wait"] = (time.perf_counter() - t0_wait) * 1000.0
        stage_ms["action_wait_after_ready"] = max(
            0.0, stage_ms["action_wait"] - stage_ms["step_ready_wait"]
        )

        if chosen_action is None:
            if self._rospy.is_shutdown():
                self.last_action_source = "ros_shutdown"
                stage_ms["total"] = (time.perf_counter() - frame_t0) * 1000.0
                self._record_timing(stage_ms)
                return None
            chosen_action = self._build_noop_action()
            if deadline is not None:
                self.last_action_timed_out = True
                self.last_action_source = "timeout_noop"
                self._rospy.logwarn_throttle(
                    2.0, "RosBridgePolicy: action timeout, using noop action."
                )

        t0_post = time.perf_counter()
        if isinstance(chosen_action, dict):
            # Keep downstream navigation task behavior consistent.
            chosen_action.setdefault("done", False)
            if self.task is not None:
                robot_view = self.task.env.current_robot.robot_view
                # A base-only navigation command must hold the reset posture.
                # Do not inject a separate hard-coded arm pose on the first
                # action: it creates a visible arm swing and can contaminate
                # the head-camera depth map.
                self._fill_missing_navigation_holds(chosen_action, robot_view)
        stage_ms["postprocess_action"] = (time.perf_counter() - t0_post) * 1000.0
        if not self.last_action_source:
            self.last_action_source = "fallback_noop"
        t0_sync = time.perf_counter()
        self._publish_step_sync(common_stamp)
        stage_ms["step_sync_publish"] = (time.perf_counter() - t0_sync) * 1000.0
        t0_capture_ack = time.perf_counter()
        self._wait_for_step_capture_ack(common_stamp)
        stage_ms["step_capture_ack_wait"] = (
            time.perf_counter() - t0_capture_ack
        ) * 1000.0
        stage_ms["total"] = (time.perf_counter() - frame_t0) * 1000.0
        self._record_timing(stage_ms)

        self._step_idx += 1
        return chosen_action

    def close(self):
        if self._tf_keepalive_timer is not None:
            self._tf_keepalive_timer.shutdown()
            self._tf_keepalive_timer = None
        if self._step_frame_thread is not None:
            self._step_frame_queue.put(None)
            self._step_frame_thread.join()
            self._step_frame_thread = None
            self._step_frame_manifest.close()
        if self._realtime_gt_publisher is not None:
            self._realtime_gt_publisher.close()
        if hasattr(self, "_action_sub") and self._action_sub is not None:
            self._action_sub.unregister()
            self._action_sub = None
        if hasattr(self, "_step_ready_sub") and self._step_ready_sub is not None:
            self._step_ready_sub.unregister()
            self._step_ready_sub = None
        if hasattr(self, "_step_capture_ack_sub") and self._step_capture_ack_sub is not None:
            self._step_capture_ack_sub.unregister()
            self._step_capture_ack_sub = None
        if hasattr(self, "_cmd_vel_sub") and self._cmd_vel_sub is not None:
            self._cmd_vel_sub.unregister()
            self._cmd_vel_sub = None
        if hasattr(self, "_move_base_status_sub") and self._move_base_status_sub is not None:
            self._move_base_status_sub.unregister()
            self._move_base_status_sub = None
