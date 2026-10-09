"""Evaluator-side guard for repeated ROS navigation failures without progress.

The semantic executor intentionally retries after an individual navigation
failure.  This small observer is deliberately outside that recovery policy:
it stops a benchmark rollout only when *different* navigation/exploration
subgoals keep failing while the simulated base makes no material progress.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any


CROSS_SUBGOAL_STALL_REASON = "cross_subgoal_navigation_stall"


@dataclass(frozen=True)
class CrossSubgoalStallConfig:
    """Conservative thresholds for evaluator-side ROS stall termination."""

    enabled: bool = True
    min_failed_subgoals: int = 8
    max_displacement_m: float = 0.15
    min_no_progress_steps: int = 20


def _finite_xy(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        x, y = value[:2]
        x = float(x)
        y = float(y)
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _feedback_reason(payload: dict[str, Any]) -> str:
    detail = payload.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    for key in ("reason", "status", "error"):
        value = detail.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "behavior_failed"


def _subgoal_key(payload: dict[str, Any]) -> str | None:
    """Return a stable executor-issued key, never an observer-local counter."""

    for key in ("decision_id", "candidate_id", "command_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return None


class CrossSubgoalStallTracker:
    """Pure state tracker used by the evaluator and unit tests.

    A sequence is reset by either meaningful base displacement or a successful
    navigation/exploration result.  Repeated terminal feedback for one
    subgoal is de-duplicated, so a latched ROS topic cannot fabricate a stall.
    """

    def __init__(self, config: CrossSubgoalStallConfig) -> None:
        self.config = config
        self._baseline_pose_xy: tuple[float, float] | None = None
        self._baseline_step: int | None = None
        self._last_pose_xy: tuple[float, float] | None = None
        self._failed_subgoals: set[str] = set()
        self._seen_terminal_feedback: set[tuple[str, str]] = set()
        self._failure_reasons: Counter[str] = Counter()
        self._behavior_failures: Counter[str] = Counter()
        self._observed_navigation_failure_count = 0
        self._last_failure: dict[str, Any] | None = None
        self._triggered = False
        self._trigger_step: int | None = None
        self._observer_error: str | None = None

    def disable(self, error: str) -> None:
        self._observer_error = str(error)

    def _reset_sequence(self, pose_xy: tuple[float, float] | None, step_index: int) -> None:
        self._baseline_pose_xy = pose_xy
        self._baseline_step = int(step_index)
        self._failed_subgoals.clear()
        self._seen_terminal_feedback.clear()
        self._last_failure = None

    def observe_pose(self, pose_xy: Any, step_index: int) -> None:
        pose = _finite_xy(pose_xy)
        if pose is None:
            return
        self._last_pose_xy = pose
        if self._baseline_pose_xy is None:
            self._baseline_pose_xy = pose
            self._baseline_step = int(step_index)
            return
        if math.dist(pose, self._baseline_pose_xy) >= self.config.max_displacement_m:
            self._reset_sequence(pose, step_index)

    def note_feedback(self, payload: dict[str, Any], pose_xy: Any, step_index: int) -> bool:
        """Consume one executor feedback payload and return whether to stop."""

        self.observe_pose(pose_xy, step_index)
        if not self.config.enabled or self._triggered:
            return self._triggered
        behavior_type = str(payload.get("behavior_type") or "").upper()
        if behavior_type not in {"NAVIGATE", "EXPLORE"}:
            return False
        status = str(payload.get("status") or "").upper()
        success = payload.get("success")
        subgoal = _subgoal_key(payload)
        if status in {"SUCCEEDED", "SUCCESS"} or success is True:
            self._reset_sequence(self._last_pose_xy, step_index)
            return False
        # Only explicit terminal navigation failures advance the evaluator
        # stall budget.  A CANCELED result can be a normal target preemption
        # (for example ``preempted_by_target``), not an unsuccessful subgoal.
        if status not in {"FAILED", "ABORTED"}:
            return False
        # A feedback message is terminal only once per executor-issued
        # subgoal/status pair.  Missing IDs cannot prove a cross-subgoal run.
        if subgoal is None:
            return False
        terminal_key = (subgoal, status or "FAILED")
        if terminal_key in self._seen_terminal_feedback:
            return False
        self._seen_terminal_feedback.add(terminal_key)
        self._failed_subgoals.add(subgoal)
        reason = _feedback_reason(payload)
        self._failure_reasons[reason] += 1
        self._behavior_failures[behavior_type] += 1
        self._observed_navigation_failure_count += 1
        self._last_failure = {
            "subgoal_key": subgoal,
            "behavior_type": behavior_type,
            "status": status or "FAILED",
            "reason": reason,
        }
        if self._baseline_step is None:
            self._baseline_step = int(step_index)
        if (
            len(self._failed_subgoals) >= self.config.min_failed_subgoals
            and int(step_index) - int(self._baseline_step) >= self.config.min_no_progress_steps
        ):
            self._triggered = True
            self._trigger_step = int(step_index)
        return self._triggered

    def snapshot(self) -> dict[str, Any]:
        displacement_m: float | None = None
        if self._baseline_pose_xy is not None and self._last_pose_xy is not None:
            displacement_m = float(math.dist(self._baseline_pose_xy, self._last_pose_xy))
        return {
            "enabled": bool(self.config.enabled),
            "active": bool(self.config.enabled and self._observer_error is None),
            "observer_error": self._observer_error,
            "triggered": self._triggered,
            "reason": CROSS_SUBGOAL_STALL_REASON if self._triggered else None,
            "trigger_step": self._trigger_step,
            "reference_step": self._baseline_step,
            "reference_pose_xy": (
                list(self._baseline_pose_xy) if self._baseline_pose_xy is not None else None
            ),
            "last_pose_xy": list(self._last_pose_xy) if self._last_pose_xy is not None else None,
            "displacement_m": displacement_m,
            "min_failed_subgoals": int(self.config.min_failed_subgoals),
            "max_displacement_m": float(self.config.max_displacement_m),
            "min_no_progress_steps": int(self.config.min_no_progress_steps),
            "failed_subgoal_count": len(self._failed_subgoals),
            "observed_navigation_failure_count": self._observed_navigation_failure_count,
            "failure_reason_counts": dict(sorted(self._failure_reasons.items())),
            "behavior_failure_counts": dict(sorted(self._behavior_failures.items())),
            "last_failure": self._last_failure,
        }


class RosBehaviorFeedbackObserver:
    """Minimal, lazy ROS subscriber for semantic executor feedback.

    The source topic is latched.  ``begin_episode`` therefore drops queued
    messages and ignores source timestamps predating this episode.
    """

    def __init__(self, topic: str) -> None:
        import rospy  # Imported only for the ROS policy path.
        from std_msgs.msg import String

        self._lock = threading.Lock()
        self._messages: list[dict[str, Any]] = []
        self._not_before_timestamp = float("-inf")
        self._subscriber = rospy.Subscriber(
            str(topic),
            String,
            self._callback,
            queue_size=100,
        )

    def _callback(self, message: Any) -> None:
        try:
            payload = json.loads(str(message.data))
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._messages.append(payload)

    def begin_episode(self) -> None:
        with self._lock:
            self._messages.clear()
            self._not_before_timestamp = time.time() - 0.25

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            messages = self._messages
            self._messages = []
            not_before = self._not_before_timestamp
        fresh: list[dict[str, Any]] = []
        for payload in messages:
            timestamp = payload.get("timestamp")
            try:
                if timestamp is not None and float(timestamp) < not_before:
                    continue
            except (TypeError, ValueError):
                pass
            fresh.append(payload)
        return fresh

    def close(self) -> None:
        try:
            self._subscriber.unregister()
        except Exception:
            pass
