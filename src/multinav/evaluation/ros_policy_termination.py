"""Evaluator-owned liveness and public ROS terminal-state handling.

The ROS navigation graph can legitimately need several observation turns before
it emits a control command.  A single bridge timeout is therefore not a policy
stop.  This module owns the narrower contract used by the V3 evaluator:

* explicit public goal-status terminal messages end the rollout immediately;
* repeated ``ros_bridge_no_fresh_action`` observations become a scored
  command-starvation failure only after a configurable wall-clock duration;
* any real policy action or evaluator-consumed interaction resets the current
  starvation streak.

The guard never assigns task success.  MuJoCo target and interaction metrics
remain authoritative after the rollout ends.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping


NO_FRESH_ACTION_REASON = "ros_bridge_no_fresh_action"
COMMAND_STARVATION_REASON = "ros_bridge_command_starvation"
OBSERVATION_TURN_LIMIT_REASON = "ros_bridge_observation_turn_limit"

TERMINAL_GOAL_STATUS_REASONS: dict[str, str] = {
    "SUCCEEDED": "policy_goal_succeeded",
    "EXPLORATION_EXHAUSTED": "policy_exploration_exhausted",
    "EXPLORATION_STALLED": "policy_exploration_stalled",
    "STARTUP_SCAN_FAILED": "policy_startup_scan_failed",
}


@dataclass(frozen=True)
class RosPolicyTerminationConfig:
    """Public policy-terminal and command-starvation settings."""

    command_starvation_timeout_s: float = 60.0
    observation_turn_multiplier: float = 4.0

    def validate(self) -> None:
        timeout = float(self.command_starvation_timeout_s)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError(
                "command_starvation_timeout_s must be finite and non-negative"
            )
        multiplier = float(self.observation_turn_multiplier)
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError(
                "observation_turn_multiplier must be finite and at least 1"
            )


@dataclass(frozen=True)
class RosPolicyTermination:
    """One evaluator terminal decision without a success assertion."""

    reason: str
    source: str
    detail: dict[str, Any]


def _action_kind(action: Any) -> str:
    if isinstance(action, Mapping):
        return str(action.get("kind") or "")
    return str(getattr(action, "kind", "") or "")


def _action_metadata(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        value = action.get("metadata")
    else:
        value = getattr(action, "metadata", None)
    return dict(value) if isinstance(value, Mapping) else {}


def is_no_fresh_action(action: Any) -> bool:
    """Return whether an evaluator action is only a bridge timeout refresh."""

    metadata = _action_metadata(action)
    return (
        _action_kind(action) == "observe"
        and str(metadata.get("reason") or "") == NO_FRESH_ACTION_REASON
    )


def counts_toward_applied_action_budget(action: Any) -> bool:
    """Classify normalized actions for the evaluator's applied-step budget."""

    kind = _action_kind(action)
    if kind == "stop" or is_no_fresh_action(action):
        return False
    return kind in {"base", "interact", "view", "observe"}


class RosPolicyTerminationGuard:
    """Track policy terminals and no-command liveness through one interface."""

    def __init__(self, config: RosPolicyTerminationConfig | None = None) -> None:
        self.config = config or RosPolicyTerminationConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self._terminal: RosPolicyTermination | None = None
        self._latest_goal_status: dict[str, Any] = {}
        self._total_no_fresh = 0
        self._consecutive_no_fresh = 0
        self._max_consecutive_no_fresh = 0
        self._no_fresh_started_mono_s: float | None = None
        self._last_no_fresh_mono_s: float | None = None
        self._max_no_fresh_wall_s = 0.0
        self._progress_reset_count = 0

    @property
    def terminal(self) -> RosPolicyTermination | None:
        return self._terminal

    def observation_turn_limit(self, applied_step_budget: int) -> int:
        budget = max(1, int(applied_step_budget))
        return max(
            budget,
            int(math.ceil(budget * float(self.config.observation_turn_multiplier))),
        )

    def observe_observation_turn_count(
        self,
        observation_turn_count: int,
        *,
        applied_step_budget: int,
    ) -> RosPolicyTermination | None:
        """Fail closed if non-applied turns evade every other liveness guard."""

        if self._terminal is not None:
            return self._terminal
        limit = self.observation_turn_limit(applied_step_budget)
        count = max(0, int(observation_turn_count))
        if count < limit:
            return None
        self._terminal = RosPolicyTermination(
            reason=OBSERVATION_TURN_LIMIT_REASON,
            source="observation_turn_limit",
            detail={
                "observation_turn_count": count,
                "observation_turn_limit": limit,
                "applied_step_budget": max(1, int(applied_step_budget)),
            },
        )
        return self._terminal

    def observe_goal_status(
        self, payload: Mapping[str, Any] | None
    ) -> RosPolicyTermination | None:
        """Consume one already-fresh public goal-status payload."""

        if not isinstance(payload, Mapping):
            return self._terminal
        document = {str(key): value for key, value in payload.items()}
        self._latest_goal_status = document
        status = str(document.get("status") or "").strip().upper()
        detail = document.get("detail")
        detail = dict(detail) if isinstance(detail, Mapping) else {}
        # ``SUCCEEDED`` is a generic word on several ROS feedback paths.  The
        # goal-status topic is terminal-success evidence only when the decision
        # layer explicitly reports its object-goal completion transition.
        if status == "SUCCEEDED":
            mission_mode = str(document.get("mission_mode") or "").strip().casefold()
            if (
                str(detail.get("reason") or "").strip() != "target_goal_succeeded"
                or mission_mode
                not in {"object_goal", "semantic_interaction_object_goal"}
            ):
                return self._terminal
        reason = TERMINAL_GOAL_STATUS_REASONS.get(status)
        if reason is None or self._terminal is not None:
            return self._terminal
        self._terminal = RosPolicyTermination(
            reason=reason,
            source="goal_status",
            detail={
                "status": status,
                "mission_mode": document.get("mission_mode"),
                "decision_id": document.get("decision_id"),
                "reported_reason": detail.get("reason"),
            },
        )
        return self._terminal

    def observe_action(
        self,
        action: Any,
        *,
        wait_started_mono_s: float | None = None,
        now_mono_s: float | None = None,
    ) -> RosPolicyTermination | None:
        """Consume one normalized action and return a terminal when proven.

        ``wait_started_mono_s`` should be captured immediately before
        ``policy.act``.  It lets the first timeout contribute its actual wait to
        the wall-clock liveness interval instead of starting at zero afterward.
        """

        if self._terminal is not None:
            return self._terminal
        now = time.monotonic() if now_mono_s is None else float(now_mono_s)
        if not is_no_fresh_action(action):
            self.note_progress(now_mono_s=now)
            return None

        started = now if wait_started_mono_s is None else float(wait_started_mono_s)
        started = min(started, now)
        if self._no_fresh_started_mono_s is None:
            self._no_fresh_started_mono_s = started
        self._last_no_fresh_mono_s = now
        self._total_no_fresh += 1
        self._consecutive_no_fresh += 1
        self._max_consecutive_no_fresh = max(
            self._max_consecutive_no_fresh, self._consecutive_no_fresh
        )
        elapsed = max(0.0, now - self._no_fresh_started_mono_s)
        self._max_no_fresh_wall_s = max(self._max_no_fresh_wall_s, elapsed)
        timeout = float(self.config.command_starvation_timeout_s)
        if timeout <= 0.0 or elapsed < timeout:
            return None
        self._terminal = RosPolicyTermination(
            reason=COMMAND_STARVATION_REASON,
            source="command_starvation",
            detail={
                "consecutive_no_fresh_action_count": self._consecutive_no_fresh,
                "no_fresh_wall_seconds": elapsed,
                "timeout_seconds": timeout,
            },
        )
        return self._terminal

    def note_progress(self, *, now_mono_s: float | None = None) -> None:
        """Reset starvation after an action or side-channel interaction.

        A bridge wait can cross the wall threshold in the same evaluator turn
        that a queued interaction is consumed.  That interaction is real
        progress and must retract the provisional command-starvation terminal.
        Goal-status and observation-turn terminals remain monotonic.
        """

        now = time.monotonic() if now_mono_s is None else float(now_mono_s)
        if self._no_fresh_started_mono_s is not None:
            end = self._last_no_fresh_mono_s
            if end is None:
                end = now
            self._max_no_fresh_wall_s = max(
                self._max_no_fresh_wall_s,
                max(0.0, float(end) - self._no_fresh_started_mono_s),
            )
            self._progress_reset_count += 1
        self._consecutive_no_fresh = 0
        self._no_fresh_started_mono_s = None
        self._last_no_fresh_mono_s = None
        if (
            self._terminal is not None
            and self._terminal.source == "command_starvation"
        ):
            self._terminal = None

    def snapshot(self, *, now_mono_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_mono_s is None else float(now_mono_s)
        current_wall_s = 0.0
        if self._no_fresh_started_mono_s is not None:
            current_wall_s = max(0.0, now - self._no_fresh_started_mono_s)
        terminal = self._terminal
        return {
            "enabled": float(self.config.command_starvation_timeout_s) > 0.0,
            "triggered": terminal is not None,
            "reason": None if terminal is None else terminal.reason,
            "source": None if terminal is None else terminal.source,
            "terminal_detail": {} if terminal is None else dict(terminal.detail),
            "command_starvation_timeout_s": float(
                self.config.command_starvation_timeout_s
            ),
            "observation_turn_multiplier": float(
                self.config.observation_turn_multiplier
            ),
            "total_no_fresh_action_count": int(self._total_no_fresh),
            "consecutive_no_fresh_action_count": int(self._consecutive_no_fresh),
            "max_consecutive_no_fresh_action_count": int(
                self._max_consecutive_no_fresh
            ),
            "current_no_fresh_wall_seconds": float(current_wall_s),
            "max_no_fresh_wall_seconds": float(
                max(self._max_no_fresh_wall_s, current_wall_s)
            ),
            "progress_reset_count": int(self._progress_reset_count),
            "latest_goal_status": dict(self._latest_goal_status),
        }
