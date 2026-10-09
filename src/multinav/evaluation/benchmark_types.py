"""Public protocol types for the standalone InteractiveNav benchmark evaluator.

This module deliberately lives under ``scripts/InteractiveNav`` instead of the
upstream :mod:`molmo_spaces.evaluation` package.  The V3 benchmark adds a
stateful interaction protocol that is not part of the upstream NavToObj API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ActionKind = Literal["base", "interact", "view", "observe", "stop"]


@dataclass(frozen=True)
class PublicEpisode:
    """Episode context that may be supplied to a non-oracle policy.

    Instance names, interaction annotations, oracle waypoints and task objects
    intentionally do not appear here.  A policy receives its instruction and
    sensor observation through :class:`PolicyObservation` separately.
    """

    house_index: int
    scene_dataset: str
    data_split: str
    instruction: str
    task_type: str
    camera_names: list[str]
    image_resolution: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyObservation:
    """Per-step policy-visible data without V3 GT annotations."""

    observation: Any
    instruction: str
    step_index: int
    elapsed_seconds: float
    previous_action: dict[str, Any] | None


@dataclass
class PolicyAction:
    """Normalized action accepted by the standalone evaluator.

    Generic external policies select an interaction with ``pixel_xy`` (or
    ``normalized_pixel_xy``). ``instance_id`` is reserved for the restricted-GT
    ROS protocol, which is the only interface that publishes opaque instance
    IDs. ``object_name`` remains available only for an explicitly oracle/debug
    policy because it is a simulator-internal identifier.
    """

    kind: ActionKind
    base_action: dict[str, Any] | None = None
    camera_name: str = "head_camera"
    pixel_xy: tuple[int, int] | None = None
    normalized_pixel_xy: tuple[float, float] | None = None
    instance_id: str | None = None
    joint_index: int | None = None
    object_name: str | None = None
    operation: Literal["open"] = "open"
    head_qpos: list[float] | None = None
    torso_qpos: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionAttempt:
    """One policy-requested interaction and its evaluator-side outcome."""

    requested: dict[str, Any]
    classification: Literal["required_valid", "extra_valid", "invalid"]
    resolved_object_name: str | None
    resolved_joint_name: str | None
    resolved_joint_index: int | None
    resolved_interaction_id: str | None
    success: bool
    joint_fraction_before: float | None
    joint_fraction_after: float | None
    prerequisite_satisfied: bool | None
    executor: str | None
    simulated_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # One object-level skill can internally operate several joints.  This is
    # evaluator-private V3 bookkeeping and is never sent back to the policy.
    resolved_interaction_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Stable, JSON-serialisable result for a benchmark episode."""

    episode_index: int
    case_id: str
    house_index: int
    domains: list[str]
    recipe: str | None
    interaction_types: list[str]
    path_length_bin: str | None
    interaction_requirement: str
    policy_name: str
    uses_oracle_gt: bool
    status: Literal["complete", "exception"]
    # ``success`` remains the historical interaction-conditioned V3 score for
    # compatibility.  These explicit fields separate the task endpoint from
    # interaction-plan correctness.
    success: bool
    task_success: bool
    interaction_conditioned_success: bool
    nav_success: bool
    required_interaction_success: bool
    sequence_success: bool
    non_interaction_success: bool | None
    terminal_reason: str
    step_count: int
    navigation_step_count: int
    view_action_count: int
    interaction_action_count: int
    correct_interaction_action_count: int
    extra_interaction_action_count: int
    invalid_interaction_action_count: int
    navigation_path_length_m: float
    reference_path_length_m: float | None
    spl: float | None
    navigation_simulated_seconds: float
    interaction_simulated_seconds: float
    total_simulated_seconds: float
    elapsed_seconds: float
    target_distance_m: float | None
    target_visibility_fraction: float | None
    interaction_attempts: list[dict[str, Any]]
    # ``step_count`` remains the recordable ROS observation-turn count.  These
    # fields make bridge waits and evaluator-applied actions explicit instead
    # of forcing callers to infer them from a large JSON trace.
    applied_action_step_count: int | None = None
    no_fresh_action_count: int = 0
    policy_termination: dict[str, Any] = field(default_factory=dict)
    episode_step_budget: int | None = None
    step_budget_mode: str = "fixed"
    step_budget_basis: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None
    video_path: str | None = None
    error: str | None = None
    # A run may be technically complete but ineligible for a formal score when
    # the frozen V3 record is inconsistent with the live replayed scene.
    scoring_eligible: bool = True
    scoring_exclusion_reasons: list[str] = field(default_factory=list)
    runtime_goal_consistency: dict[str, Any] | None = None
    runtime_consistency: dict[str, Any] | None = None
    timing_summary: dict[str, Any] = field(default_factory=dict)
    # Evaluator-side termination diagnostics.  A triggered guard is a normal
    # scored failure, not a runtime exception or scoring exclusion.
    early_stop: dict[str, Any] = field(default_factory=dict)
    # Paper-facing metric provenance and per-episode quantities.  These are
    # scalar aggregates only: evaluator-private interaction IDs remain out of
    # restricted-policy result traces.
    paper_metric_schema_version: str = ""
    paper_metric_config: dict[str, Any] = field(default_factory=dict)
    valid_interaction_attempt_count: int = 0
    error_interaction_attempt_count: int = 0
    task_irrelevant_interaction_attempt_count: int = 0
    failed_interaction_attempt_count: int = 0
    repeated_interaction_attempt_count: int = 0
    interaction_precision_episode: float | None = None
    episode_total_cost: float | None = None
    episode_total_cost_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
