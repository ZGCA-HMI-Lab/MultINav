"""Policy adapters for the standalone InteractiveNav benchmark evaluator.

The adapters deliberately avoid handing a live task or replay configuration to
non-oracle policies.  Doing so would let a policy inspect ``pickup_obj_name``
or other benchmark-only annotations instead of solving the visual task.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .benchmark_types import PolicyAction, PolicyObservation, PublicEpisode


class BenchmarkPolicy(Protocol):
    name: str
    uses_oracle_gt: bool

    def reset(self, episode: PublicEpisode) -> None: ...

    def act(self, observation: PolicyObservation) -> PolicyAction: ...

    def close(self) -> None: ...


class NoOpPolicy:
    name = "noop"
    uses_oracle_gt = False

    def reset(self, episode: PublicEpisode) -> None:
        del episode

    def act(self, observation: PolicyObservation) -> PolicyAction:
        del observation
        return PolicyAction(kind="stop", metadata={"reason": "noop"})

    def close(self) -> None:
        return None


class ScriptedOraclePolicy:
    """Explicit GT-only execution upper bound.

    It is never used for a learned-policy result.  The runner supplies the
    oracle plan only to this class and marks every produced result with
    ``uses_oracle_gt=True``.
    """

    name = "scripted_oracle"
    uses_oracle_gt = True

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._cursor = 0

    def reset(self, episode: PublicEpisode) -> None:
        # The runner injects the private plan immediately after this public
        # reset.  Keeping this method GT-free makes accidental reuse by an
        # ordinary policy impossible.
        del episode
        self._steps = []
        self._cursor = 0

    def reset_oracle(self, steps: list[dict[str, Any]]) -> None:
        self._steps = copy.deepcopy(steps)
        self._cursor = 0

    def act(self, observation: PolicyObservation) -> PolicyAction:
        del observation
        if self._cursor >= len(self._steps):
            return PolicyAction(kind="stop", metadata={"reason": "oracle_complete"})
        step = self._steps[self._cursor]
        step_type = str(step.get("type", ""))
        if step_type == "navigate":
            return PolicyAction(
                kind="base",
                metadata={
                    "oracle_waypoint": True,
                    "goal_point": list(step["goal_point"]),
                    "goal_yaw": float(step.get("goal_yaw", 0.0)),
                    "position_tolerance_m": float(step.get("position_tolerance_m", 0.25)),
                    "yaw_tolerance_rad": float(step.get("yaw_tolerance_rad", 0.35)),
                    "oracle_cursor": self._cursor,
                    "reason": step.get("reason"),
                },
            )
        self._cursor += 1
        if step_type == "open_joint":
            return PolicyAction(
                kind="interact",
                object_name=str(step["object_name"]),
                joint_index=int(step["joint_index"]),
                operation="open",
                metadata={
                    "oracle_interaction_id": str(step["interaction_id"]),
                    "oracle_joint_name": str(step["joint_name"]),
                    "oracle_target_fraction": float(step.get("target_fraction", 1.0)),
                },
            )
        if step_type == "set_view":
            return PolicyAction(
                kind="view",
                head_qpos=[float(value) for value in step.get("head_qpos", [])],
                torso_qpos=[float(value) for value in step.get("torso_qpos", [])],
                metadata={"oracle_view_profile": step.get("view_profile"), "reason": step.get("reason")},
            )
        if step_type == "observe_target":
            return PolicyAction(kind="observe", metadata={"oracle_observe": True, "reason": step.get("reason")})
        raise ValueError(f"Unsupported oracle plan step type: {step_type!r}")

    def notify_action_result(self, action: PolicyAction, *, reached: bool) -> None:
        if action.metadata.get("oracle_waypoint") and reached:
            self._cursor += 1

    def close(self) -> None:
        return None


def _as_pair(value: Any, *, cast) -> tuple[Any, Any] | None:
    if value is None:
        return None
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"Expected a two-element coordinate, got {value!r}") from exc
    if len(values) != 2:
        raise ValueError(f"Expected a two-element coordinate, got {value!r}")
    return cast(values[0]), cast(values[1])


def normalize_policy_action(value: Any) -> PolicyAction:
    """Normalize a generic/ROS policy payload into the evaluator protocol."""

    if isinstance(value, PolicyAction):
        return value
    if value is None:
        return PolicyAction(kind="stop", metadata={"reason": "policy_returned_none"})
    if not isinstance(value, dict):
        raise TypeError(f"Policy returned {type(value).__name__}; expected dict or PolicyAction")
    payload = value.get("action", value)
    if not isinstance(payload, dict):
        raise TypeError("Policy action wrapper must contain a mapping in 'action'")
    if bool(payload.get("done", False)):
        return PolicyAction(kind="stop", metadata={"wrapped_action": _json_safe(payload)})
    kind = str(payload.get("kind", ""))
    if not kind and ("head_qpos" in payload or "torso_qpos" in payload):
        kind = "view"
    if not kind:
        kind = "base"
    metadata = {"wrapped_action": _json_safe(payload)}
    if kind == "stop":
        return PolicyAction(kind="stop", metadata=metadata)
    if kind == "base":
        base = payload.get("base_action")
        if base is None:
            # Preserve the normal MolmoSpaces action dictionary as-is, except
            # for evaluator protocol keys that task.step() must not receive.
            base = {
                key: item
                for key, item in payload.items()
                if key
                not in {
                    "kind", "pixel_xy", "normalized_pixel_xy", "camera_name", "operation",
                    "instance_id", "object_name", "joint_index", "head_qpos", "torso_qpos", "done",
                }
            }
        if not isinstance(base, dict):
            raise TypeError("base_action must be a mapping")
        return PolicyAction(kind="base", base_action=base, metadata=metadata)
    if kind == "interact":
        pixel = payload.get("pixel_xy", payload.get("target_pixel", payload.get("pixel")))
        normalized = payload.get("normalized_pixel_xy", payload.get("normalized_pixel"))
        return PolicyAction(
            kind="interact",
            camera_name=str(payload.get("camera_name", "head_camera")),
            pixel_xy=_as_pair(pixel, cast=lambda item: int(round(float(item)))),
            normalized_pixel_xy=_as_pair(normalized, cast=float),
            instance_id=None if payload.get("instance_id") is None else str(payload["instance_id"]),
            joint_index=None if payload.get("joint_index") is None else int(payload["joint_index"]),
            object_name=payload.get("object_name"),
            operation=str(payload.get("operation", "open")),
            metadata=metadata,
        )
    if kind == "view":
        return PolicyAction(
            kind="view",
            head_qpos=None if payload.get("head_qpos") is None else [float(v) for v in payload["head_qpos"]],
            torso_qpos=None if payload.get("torso_qpos") is None else [float(v) for v in payload["torso_qpos"]],
            metadata=metadata,
        )
    if kind == "observe":
        return PolicyAction(kind="observe", metadata=metadata)
    raise ValueError(f"Unsupported policy action kind: {kind!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class ExternalPolicyAdapter:
    """Adapt an external policy without exposing a live MuJoCo task."""

    uses_oracle_gt = False

    def __init__(self, policy: Any, *, name: str | None = None) -> None:
        self.policy = policy
        self.name = name or getattr(policy, "name", type(policy).__name__)

    def reset(self, episode: PublicEpisode) -> None:
        reset = getattr(self.policy, "reset", None)
        if not callable(reset):
            return
        try:
            reset(episode.to_dict())
        except TypeError:
            # Existing MolmoSpaces policies commonly expose reset() with no
            # arguments.  They still never receive a task through this adapter.
            reset()

    def act(self, observation: PolicyObservation) -> PolicyAction:
        act = getattr(self.policy, "act", None)
        if callable(act):
            return normalize_policy_action(act(observation))
        get_action = getattr(self.policy, "get_action", None)
        if callable(get_action):
            return normalize_policy_action(get_action(observation.observation))
        if callable(self.policy):
            return normalize_policy_action(self.policy(observation))
        raise TypeError(f"External policy {type(self.policy).__name__} has no act/get_action callable")

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


class RosBridgePolicyAdapter(ExternalPolicyAdapter):
    """Adapter for the repository's live ROS navigation stack.

    :class:`RosBridgePolicy` is a normal MolmoSpaces policy and therefore
    expects a raw task observation in ``get_action``.  The standalone V3
    evaluator deliberately must not give it a live task, because that would
    make private benchmark annotations reachable through ``task``.  This
    adapter supplies only the public sensor observation and translates an
    action-timeout/no-op into ``observe`` instead of attempting
    ``task.step({})``.

    The ROS process itself receives RGB/depth/point-cloud/odometry on its
    existing topics and publishes either a normal action JSON or
    ``/cmd_vel_stamped``.  It never receives V3 interaction metadata.
    """

    def reset(self, episode: PublicEpisode) -> None:
        # ``episode`` intentionally remains unused.  The ROS graph is reset by
        # RosBridgePolicy.prepare_episode_reset() between live episodes; this
        # evaluator does not pass a MuJoCo task or private replay object.
        del episode
        prepare_episode_reset = getattr(self.policy, "prepare_episode_reset", None)
        if callable(prepare_episode_reset):
            # The canonical ROS rollout runner owns one bridge policy across
            # scenes and therefore calls this public method before every reset.
            # The standalone evaluator creates one policy per isolated MuJoCo
            # episode, so seed the bridge's local counter such that its first
            # invocation also clears the persistent external map/costmaps.
            # This touches no simulator task and is necessary to prevent a
            # previous house leaking into the next single-worker ROS episode.
            if getattr(self.policy, "_episode_count", 0) < 1:
                try:
                    self.policy._episode_count = 1
                except (AttributeError, TypeError):
                    pass
            prepare_episode_reset()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: PolicyObservation) -> PolicyAction:
        get_action = getattr(self.policy, "get_action", None)
        if not callable(get_action):
            raise TypeError(f"ROS bridge {type(self.policy).__name__} has no get_action callable")
        raw_action = get_action(observation.observation)
        if raw_action is None:
            # ``None`` is another no-command bridge outcome, not an algorithm-
            # requested observe action.  Canonicalising it keeps applied-step
            # accounting and the wall-clock starvation guard honest.
            return PolicyAction(
                kind="observe",
                metadata={
                    "reason": "ros_bridge_no_fresh_action",
                    "bridge_action_source": "returned_none",
                },
            )
        if isinstance(raw_action, dict):
            payload = raw_action.get("action", raw_action)
            # With ``task=None``, RosBridgePolicy represents an action timeout
            # as ``{\"done\": false}``.  Treat it as a pure sensor refresh; a
            # base action with an empty control dictionary is not a valid task
            # command and can spuriously advance a rollout.
            if isinstance(payload, dict) and set(payload).issubset({"done"}) and not bool(payload.get("done", False)):
                return PolicyAction(
                    kind="observe",
                    metadata={
                        "reason": "ros_bridge_no_fresh_action",
                        "wrapped_action": _json_safe(payload),
                        # Evaluator-visible diagnostics only.  They explain
                        # whether the empty task-less payload came from the
                        # configured action deadline without changing the
                        # public action or simulator state.
                        "bridge_action_timed_out": bool(
                            getattr(self.policy, "last_action_timed_out", False)
                        ),
                        "bridge_action_source": str(
                            getattr(self.policy, "last_action_source", "") or ""
                        ),
                    },
                )
        return normalize_policy_action(raw_action)

    def queue_step_frame_public_payload(self, payload: dict[str, Any]) -> bool:
        """Forward evaluator-owned public perception to the bridge recorder.

        This is intentionally not part of ``BenchmarkPolicy``: only the V3 ROS
        bridge records per-RGB simulator frames, while all other policies remain
        unaware of evaluator perception internals.
        """

        queue_payload = getattr(self.policy, "queue_step_frame_public_payload", None)
        if not callable(queue_payload):
            return False
        return bool(queue_payload(payload))


def _import_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("policy factory must use 'module.path:callable' syntax")
    module_name, qualname = spec.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def build_factory_policy(
    factory_spec: str,
    *,
    public_episode: PublicEpisode,
    kwargs: dict[str, Any] | None = None,
) -> ExternalPolicyAdapter:
    """Build a generic policy using only public episode/configuration data.

    Factories may accept ``public_episode`` and/or ``kwargs`` as keyword
    parameters, a single positional public-episode dictionary, or no argument.
    """

    factory = _import_symbol(factory_spec)
    options = dict(kwargs or {})
    signature = inspect.signature(factory)
    parameters = signature.parameters
    call_kwargs: dict[str, Any] = {}
    if "public_episode" in parameters:
        call_kwargs["public_episode"] = public_episode.to_dict()
    if "episode" in parameters:
        call_kwargs["episode"] = public_episode.to_dict()
    if "kwargs" in parameters:
        call_kwargs["kwargs"] = options
    for key, value in options.items():
        if key in parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            call_kwargs[key] = value
    try:
        policy = factory(**call_kwargs)
    except TypeError as exc:
        if call_kwargs:
            raise TypeError(f"Could not construct policy factory {factory_spec}: {exc}") from exc
        policy = factory(public_episode.to_dict())
    return ExternalPolicyAdapter(policy, name=factory_spec)


@dataclass(frozen=True)
class RosConfigFacade:
    """Minimal config accepted by RosBridgePolicy without task-level GT."""

    policy_dt_ms: float
    policy_config: Any


def build_ros_bridge_policy(
    *,
    policy_dt_ms: float,
    observation_topic: str,
    action_topic: str,
    action_timeout_s: float,
    cmd_vel_linear_gain: float,
    require_move_base_active: bool,
    map_warmup_skip_frames: int,
    step_ready_topic: str = "/semantic_decision/step_ready",
    step_ready_barrier_enabled: bool = False,
    step_ready_warmup_skip_frames: int = 0,
    step_ready_timeout_s: float = 2.0,
    step_ready_bootstrap_timeout_s: float = 10.0,
    step_frame_dir: str = "",
    step_frame_queue_size: int = 4,
    step_capture_ack_topic: str = "/molmo_spaces/step_capture_ack",
    step_capture_ack_barrier_enabled: bool = False,
    step_capture_ack_timeout_s: float = 2.0,
    name: str = "ros_bridge",
) -> RosBridgePolicyAdapter:
    """Attach to an already-running ROS graph with no live task reference."""

    from multinav.platforms.molmospaces.policies.ros_bridge_policy import RosBridgePolicy

    policy = RosBridgePolicy(
        config=RosConfigFacade(
            policy_dt_ms=float(policy_dt_ms),
            policy_config=type("RosPolicyConfig", (), {"force_enable_depth": False})(),
        ),
        task=None,
        observation_topic=observation_topic,
        action_topic=action_topic,
        action_timeout_s=float(action_timeout_s),
        cmd_vel_linear_gain=float(cmd_vel_linear_gain),
        # V3's pre-rotation sends at most one nonzero command for each RGB
        # evaluator step.  Keep a command fresh-only so it cannot spill into a
        # later get_action call and exceed that control-step budget.
        require_fresh_cmd_vel=True,
        require_move_base_active_for_cmd_vel=bool(require_move_base_active),
        map_warmup_skip_frames=int(map_warmup_skip_frames),
        step_ready_topic=str(step_ready_topic),
        step_ready_barrier_enabled=bool(step_ready_barrier_enabled),
        step_ready_warmup_skip_frames=int(step_ready_warmup_skip_frames),
        step_ready_timeout_s=float(step_ready_timeout_s),
        step_ready_bootstrap_timeout_s=float(step_ready_bootstrap_timeout_s),
        publish_realtime_gt=False,
        step_frame_dir=str(step_frame_dir),
        step_frame_queue_size=int(step_frame_queue_size),
        step_capture_ack_topic=str(step_capture_ack_topic),
        step_capture_ack_barrier_enabled=bool(step_capture_ack_barrier_enabled),
        step_capture_ack_timeout_s=float(step_capture_ack_timeout_s),
    )
    return RosBridgePolicyAdapter(policy, name=str(name))
