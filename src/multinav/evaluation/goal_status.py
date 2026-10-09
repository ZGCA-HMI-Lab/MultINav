"""Evaluator-side goal completion boundary for the restricted ROS benchmark.

The simulator owns the private target pose and the evaluator owns the mapping
from an opaque perception id back to that pose.  A policy may *declare* that it
has completed its object goal, but the declaration is accepted only when the
same target id has appeared in a frame that was actually published through the
restricted-GT adapter and the private distance criterion is satisfied.  In
particular, a raw MuJoCo visibility check is never a terminal event.

This module deliberately has no dependency on rospy at import time.  That keeps
the verification logic unit-testable and lets the evaluator run in quality-gate
or non-ROS environments.  ``RosGoalStatusObserver`` is a small optional topic
observer; callers can also inject payloads directly into
``GoalStatusObserver.ingest`` in tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any, Callable, Iterable, Mapping


TARGET_SUCCESS_STATUS = "SUCCEEDED"
TARGET_SUCCESS_REASON = "target_goal_succeeded"
OBJECT_GOAL_MISSION_MODES = frozenset(
    {"object_goal", "semantic_interaction_object_goal"}
)
EXPLORATION_TERMINAL_STATUSES = frozenset(
    {"EXPLORATION_EXHAUSTED", "EXPLORATION_STALLED", "STARTUP_SCAN_FAILED"}
)


def _normalise_episode_id(value: Any) -> str:
    return str(value or "").strip()


def _payload_episode_id(payload: Mapping[str, Any]) -> str:
    context = payload.get("target_context")
    if isinstance(context, Mapping):
        return _normalise_episode_id(context.get("episode_id"))
    # Some deployments put the episode id at the top level.  Accepting this
    # compatibility form is safe because it is still matched exactly against
    # the evaluator-owned active episode id.
    return _normalise_episode_id(payload.get("episode_id"))


def _payload_timestamp(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("timestamp", payload.get("stamp_sec"))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def is_target_goal_success_claim(payload: Mapping[str, Any] | None) -> bool:
    """Return whether a payload is an object-goal success declaration.

    ``SUCCEEDED`` alone is intentionally insufficient.  Exploration completion
    and generic action success must not be interpreted as finding the benchmark
    object.
    """

    if not isinstance(payload, Mapping):
        return False
    status = str(payload.get("status") or "").strip().upper()
    if status != TARGET_SUCCESS_STATUS:
        return False
    detail = payload.get("detail")
    reason = detail.get("reason") if isinstance(detail, Mapping) else None
    if str(reason or "").strip() != TARGET_SUCCESS_REASON:
        return False
    mission_mode = str(payload.get("mission_mode") or "").strip().casefold()
    return mission_mode in OBJECT_GOAL_MISSION_MODES


def is_exploration_terminal(payload: Mapping[str, Any] | None) -> bool:
    """Return whether a payload ends exploration without claiming the target."""

    if not isinstance(payload, Mapping):
        return False
    status = str(payload.get("status") or "").strip().upper()
    # Keep the terminal vocabulary unambiguous: ``SUCCEEDED`` is reserved for
    # the exact target-goal declaration checked above.  Exploration completion
    # must use one of its dedicated statuses instead of smuggling a terminal
    # through a generic success message.
    return status in EXPLORATION_TERMINAL_STATUSES


@dataclass(frozen=True)
class GoalStatusMessage:
    """One fresh, episode-matched goal-status payload."""

    payload: dict[str, Any]
    received_at_wall_time: float


class GoalStatusObserver:
    """Thread-safe queue with strict episode and freshness filtering.

    The observer is transport agnostic.  ROS callbacks call :meth:`ingest`,
    while tests and non-ROS adapters can inject dictionaries directly.  Calling
    :meth:`begin_episode` invalidates all messages from the previous episode;
    this is important for a latched ``goal_status`` topic.
    """

    def __init__(
        self,
        *,
        expected_episode_id: str | None = None,
        clock: Callable[[], float] | None = None,
        stale_tolerance_s: float = 0.25,
        queue_size: int = 64,
    ) -> None:
        self._clock = clock or time.time
        self._stale_tolerance_s = max(0.0, float(stale_tolerance_s))
        self._queue_size = max(1, int(queue_size))
        self._lock = threading.Lock()
        self._queue: deque[GoalStatusMessage] = deque(maxlen=self._queue_size)
        self._expected_episode_id = _normalise_episode_id(expected_episode_id)
        self._started_at_wall_time = float(self._clock())
        self._closed = False

    @property
    def expected_episode_id(self) -> str:
        with self._lock:
            return self._expected_episode_id

    def begin_episode(self, episode_id: str) -> None:
        normalized = _normalise_episode_id(episode_id)
        if not normalized:
            raise ValueError("goal-status episode_id must be non-empty")
        with self._lock:
            self._expected_episode_id = normalized
            self._started_at_wall_time = float(self._clock())
            self._queue.clear()
            self._closed = False

    def ingest(
        self,
        payload: Mapping[str, Any] | str,
        *,
        received_at_wall_time: float | None = None,
    ) -> bool:
        """Queue a JSON/dict payload if it belongs to the active episode.

        Returning ``False`` for stale or malformed messages makes it easy for a
        ROS callback to count rejected latched messages without exposing them to
        the policy.
        """

        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
        else:
            decoded = dict(payload) if isinstance(payload, Mapping) else None
        if not isinstance(decoded, dict):
            return False
        received = float(self._clock() if received_at_wall_time is None else received_at_wall_time)
        if not math.isfinite(received):
            return False
        with self._lock:
            if self._closed or not self._expected_episode_id:
                return False
            if _payload_episode_id(decoded) != self._expected_episode_id:
                return False
            timestamp = _payload_timestamp(decoded)
            # A latched message from a prior run can carry the same episode id.
            # Require its producer timestamp to be no older than the episode
            # boundary, when a timestamp is supplied.  Missing timestamps remain
            # admissible for compatibility with older semantic nodes.
            if timestamp is not None and timestamp < self._started_at_wall_time - self._stale_tolerance_s:
                return False
            self._queue.append(
                GoalStatusMessage(payload=decoded, received_at_wall_time=received)
            )
            return True

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            payloads = [message.payload for message in self._queue]
            self._queue.clear()
            return payloads

    def close(self) -> None:
        with self._lock:
            self._queue.clear()
            self._closed = True


class RosGoalStatusObserver(GoalStatusObserver):
    """Optional ROS ``std_msgs/String`` transport for :class:`GoalStatusObserver`."""

    def __init__(
        self,
        topic: str = "/semantic_decision/goal_status",
        *,
        rospy_module: Any | None = None,
        string_type: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.topic = str(topic)
        self._rospy = rospy_module
        self._string_type = string_type
        self._subscriber: Any | None = None

    def _ensure_ros(self) -> bool:
        if self._rospy is not None and self._string_type is not None:
            return True
        try:
            import rospy  # type: ignore
            from std_msgs.msg import String  # type: ignore
        except Exception:
            return False
        self._rospy = rospy
        self._string_type = String
        return True

    def begin_episode(self, episode_id: str) -> None:
        super().begin_episode(episode_id)
        if self._subscriber is not None or not self._ensure_ros():
            return
        try:
            self._subscriber = self._rospy.Subscriber(
                self.topic,
                self._string_type,
                self._callback,
                queue_size=16,
            )
        except Exception:
            self._subscriber = None

    def _callback(self, message: Any) -> None:
        self.ingest(getattr(message, "data", message))

    def close(self) -> None:
        subscriber = self._subscriber
        self._subscriber = None
        if subscriber is not None:
            unregister = getattr(subscriber, "unregister", None)
            if callable(unregister):
                try:
                    unregister()
                except Exception:
                    pass
        super().close()


@dataclass(frozen=True)
class PublicGoalEvidence:
    """A target id observed in one published restricted frame."""

    instance_id: str
    capture_step: int
    received_at_wall_time: float


class PublicGoalEvidenceLedger:
    """Bounded ledger of policy-visible target observations.

    Only opaque IDs and capture metadata are retained.  Source/body names stay
    in the runner's private mapping and are never serialized by this class.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        target_instance_ids: Iterable[str],
        max_frames: int = 128,
    ) -> None:
        self.episode_id = _normalise_episode_id(episode_id)
        if not self.episode_id:
            raise ValueError("evidence ledger episode_id must be non-empty")
        self.target_instance_ids = frozenset(
            str(value).strip() for value in target_instance_ids if str(value).strip()
        )
        self._frames: deque[PublicGoalEvidence] = deque(maxlen=max(1, int(max_frames)))

    def record_frame(
        self,
        payload: Mapping[str, Any],
        *,
        capture_step: int,
        received_at_wall_time: float | None = None,
    ) -> tuple[str, ...]:
        if _payload_episode_id(payload) != self.episode_id:
            return ()
        observations = payload.get("observations")
        if not isinstance(observations, Iterable) or isinstance(observations, (str, bytes, Mapping)):
            return ()
        observed: list[str] = []
        timestamp = float(time.time() if received_at_wall_time is None else received_at_wall_time)
        for raw in observations:
            if not isinstance(raw, Mapping):
                continue
            instance_id = str(raw.get("instance_id", raw.get("id", "")) or "").strip()
            if instance_id and instance_id in self.target_instance_ids:
                observed.append(instance_id)
                self._frames.append(
                    PublicGoalEvidence(
                    instance_id=instance_id,
                    capture_step=int(capture_step),
                    received_at_wall_time=timestamp,
                    )
                )
        return tuple(dict.fromkeys(observed))

    @property
    def observed_instance_ids(self) -> frozenset[str]:
        return frozenset(item.instance_id for item in self._frames)

    @property
    def frames(self) -> tuple[PublicGoalEvidence, ...]:
        return tuple(self._frames)

    def has_reliable_target_evidence(self) -> bool:
        return bool(self._frames)


@dataclass(frozen=True)
class GoalClaimVerification:
    accepted: bool
    reason: str
    target_instance_id: str | None = None
    distance_m: float | None = None
    evidence_capture_step: int | None = None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
            "target_instance_id": self.target_instance_id,
            "distance_m": self.distance_m,
            "evidence_capture_step": self.evidence_capture_step,
        }


def verify_target_goal_claim(
    payload: Mapping[str, Any] | None,
    *,
    episode_id: str,
    evidence: PublicGoalEvidenceLedger,
    private_distances_m: Mapping[str, float],
    distance_threshold_m: float,
) -> GoalClaimVerification:
    """Verify one policy declaration against public evidence and private range.

    ``private_distances_m`` is keyed by the same opaque IDs as the ledger.  The
    mapping is supplied by the evaluator at claim time and is never sent to
    ROS.  A declaration for a stale/mismatched episode, a non-object mission,
    or a target absent from published frames is rejected.
    """

    if not is_target_goal_success_claim(payload):
        return GoalClaimVerification(False, "not_target_goal_claim")
    if _payload_episode_id(payload or {}) != _normalise_episode_id(episode_id):
        return GoalClaimVerification(False, "episode_mismatch")
    if not evidence.has_reliable_target_evidence():
        return GoalClaimVerification(False, "no_published_target_evidence")
    threshold = float(distance_threshold_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        return GoalClaimVerification(False, "invalid_distance_threshold")
    # Match the native NavToObj endpoint: select the currently nearest target
    # candidate first, then require public evidence for that *same* opaque
    # instance.  Accepting any historically visible candidate would silently
    # change ``any_candidate`` episodes into a different success definition.
    finite_distances: list[tuple[float, str]] = []
    for instance_id, raw_distance in private_distances_m.items():
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance):
            finite_distances.append((distance, str(instance_id)))
    if not finite_distances:
        return GoalClaimVerification(False, "private_distance_unavailable")
    # ``min(..., key=distance)`` deliberately preserves the candidate mapping's
    # insertion order on an exact distance tie, matching ``target_metrics``.
    distance, nearest_instance_id = min(
        finite_distances,
        key=lambda item: item[0],
    )
    if distance >= threshold:
        return GoalClaimVerification(False, "private_distance_failed")
    matching_frame = next(
        (
            frame
            for frame in reversed(evidence.frames)
            if frame.instance_id == nearest_instance_id
        ),
        None,
    )
    if matching_frame is None:
        return GoalClaimVerification(False, "nearest_target_not_published")
    return GoalClaimVerification(
        True,
        "verified",
        target_instance_id=nearest_instance_id,
        distance_m=distance,
        evidence_capture_step=matching_frame.capture_step,
    )
