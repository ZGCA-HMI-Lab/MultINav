"""Evaluator-owned, opaque object-level interaction skills.

The InteractiveNav policy is allowed to request a high-level action such as
``open(obj_000017)``.  This module deliberately keeps the mapping from that
episode-local opaque ID to simulator objects and joints private to the
evaluator.  It is therefore suitable for an evaluator which delegates the
low-level motion to a trusted force policy without leaking articulation
metadata to the navigation method.

The module has no MuJoCo or ROS dependency.  The evaluator supplies its own
runtime-joint objects and an ``execute_open_joint`` callable, which makes the
protocol straightforward to unit test with fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Literal


TrustedOperation = Literal["open"]
PublicSkillStatus = Literal["completed", "failed", "rejected"]


class OpenPostcondition(str, Enum):
    """Evaluator-private classification of an object-level ``open`` result."""

    SATISFIED = "satisfied"
    ALREADY_SATISFIED = "already_satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class OpenPostconditionSpec:
    """Private evaluator rule for one object's generic ``open`` skill.

    ``minimum_open_joints`` defaults to one so that a multi-part articulated
    object may be reported open once one of its accessible parts reaches the
    configured threshold.  A caller that needs every supplied joint to open
    can set it to ``len(joints)`` when registering the object.  This choice is
    private evaluator behaviour, not a field exposed to the navigation method.
    """

    success_fraction: float = 0.8
    minimum_open_joints: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.success_fraction) <= 1.0:
            raise ValueError("success_fraction must be in [0, 1]")
        if isinstance(self.minimum_open_joints, bool) or not isinstance(self.minimum_open_joints, int):
            raise TypeError("minimum_open_joints must be an integer")
        if self.minimum_open_joints < 1:
            raise ValueError("minimum_open_joints must be >= 1")


@dataclass(frozen=True)
class ObjectInteractionRequest:
    """The full public request accepted by the evaluator skill endpoint.

    It intentionally has no source-object name, joint name/index, articulation
    state, pixel selector, or simulator-specific metadata.  ``instance_id`` is
    an opaque, episode-local identifier previously supplied by the restricted
    perception publisher.
    """

    request_id: str
    instance_id: str
    operation: TrustedOperation = "open"

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")

    @classmethod
    def from_public_payload(cls, payload: Mapping[str, Any]) -> "ObjectInteractionRequest":
        """Parse an exact allow-listed public action payload.

        Rejecting unrecognised fields is intentional: silently accepting a
        ``joint_name`` or ``source_object_name`` field would create an accidental
        side channel as the protocol evolves.
        """

        allowed = {"request_id", "instance_id", "operation"}
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"unsupported public interaction fields: {sorted(unexpected)}")
        missing = {"request_id", "instance_id"} - set(payload)
        if missing:
            raise ValueError(f"missing public interaction fields: {sorted(missing)}")
        operation = payload.get("operation", "open")
        if operation != "open":
            raise ValueError(f"unsupported interaction operation: {operation!r}")
        for field_name in ("request_id", "instance_id"):
            if not isinstance(payload[field_name], str):
                raise TypeError(f"{field_name} must be a string")
        return cls(
            request_id=payload["request_id"],
            instance_id=payload["instance_id"],
            operation="open",
        )

    def to_public_dict(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class ObjectInteractionResult:
    """Minimal result returned to the navigation method.

    The status reports only whether the sealed skill completed.  It never
    exposes a raw object/joint identifier, actuator state, opening fraction,
    contact trace, force magnitude, or low-level failure reason.
    """

    request_id: str
    instance_id: str
    operation: TrustedOperation
    status: PublicSkillStatus

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_public_dict(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "operation": self.operation,
            "status": self.status,
        }


@dataclass(frozen=True)
class JointOpenResult:
    """Private result from one evaluator-owned low-level joint execution.

    ``open_fraction_after`` is required to mark the high-level skill complete.
    This checks the object-level postcondition without treating the intermediate
    force-control trace as part of the navigation-method evaluation.
    """

    executor_succeeded: bool
    open_fraction_before: float | None
    open_fraction_after: float | None
    simulated_seconds: float = 0.0
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("open_fraction_before", self.open_fraction_before),
            ("open_fraction_after", self.open_fraction_after),
            ("simulated_seconds", self.simulated_seconds),
        ):
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied")
        if float(self.simulated_seconds) < 0.0:
            raise ValueError("simulated_seconds must be non-negative")


@dataclass(frozen=True)
class TrustedObject:
    """Private registry entry for a public opaque instance ID.

    ``object_ref`` and ``joints`` may contain ``RuntimeJoint`` instances or
    any evaluator-defined runtime objects.  They are deliberately omitted from
    repr and must never be serialised into ROS messages or policy traces.
    """

    opaque_id: str
    joints: tuple[Any, ...] = field(repr=False, compare=False)
    object_ref: Any = field(default=None, repr=False, compare=False)
    open_postcondition: OpenPostconditionSpec = field(default_factory=OpenPostconditionSpec, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_id, str) or not self.opaque_id:
            raise ValueError("opaque_id must be a non-empty string")
        if not self.joints:
            raise ValueError("trusted objects must provide at least one private joint")
        if self.open_postcondition.minimum_open_joints > len(self.joints):
            raise ValueError("minimum_open_joints cannot exceed the number of private joints")


class OpaqueObjectRegistry:
    """Episode-local private mapping from opaque IDs to runtime objects.

    Create a new registry for every episode, or call :meth:`clear` before
    repopulating it.  The only public lookup intentionally exposed here is
    :meth:`contains`; resolving a record is evaluator-internal.
    """

    def __init__(self) -> None:
        self._records: dict[str, TrustedObject] = {}

    def register(
        self,
        opaque_id: str,
        *,
        joints: Iterable[Any],
        object_ref: Any = None,
        open_postcondition: OpenPostconditionSpec | None = None,
    ) -> None:
        if opaque_id in self._records:
            raise ValueError(f"duplicate opaque object ID: {opaque_id!r}")
        record = TrustedObject(
            opaque_id=opaque_id,
            joints=tuple(joints),
            object_ref=object_ref,
            open_postcondition=open_postcondition or OpenPostconditionSpec(),
        )
        self._records[opaque_id] = record

    def contains(self, opaque_id: str) -> bool:
        """Return whether an opaque ID is registered, without revealing a record."""

        return opaque_id in self._records

    def clear(self) -> None:
        self._records.clear()

    def _resolve_private(self, opaque_id: str) -> TrustedObject | None:
        """Evaluator-only lookup; never hand its result to a policy process."""

        return self._records.get(opaque_id)


def classify_open_postcondition(
    joint_results: Iterable[JointOpenResult],
    *,
    spec: OpenPostconditionSpec,
) -> OpenPostcondition:
    """Classify only the before/after object state of a sealed open skill.

    This deliberately does *not* inspect or score individual controller steps.
    An executor success flag without a readable final opening fraction is not
    accepted as proof of an open postcondition.
    """

    results = tuple(joint_results)
    if not results:
        return OpenPostcondition.NOT_SATISFIED

    threshold = float(spec.success_fraction)
    observed = [result for result in results if result.open_fraction_after is not None]
    if not observed:
        return OpenPostcondition.UNOBSERVABLE

    satisfied_count = sum(
        float(result.open_fraction_after) >= threshold
        for result in observed
        if result.open_fraction_after is not None
    )
    if satisfied_count < spec.minimum_open_joints:
        return OpenPostcondition.NOT_SATISFIED

    changed_count = sum(
        result.open_fraction_before is not None
        and result.open_fraction_after is not None
        and float(result.open_fraction_before) < threshold
        and float(result.open_fraction_after) >= threshold
        for result in observed
    )
    if changed_count:
        return OpenPostcondition.SATISFIED
    return OpenPostcondition.ALREADY_SATISFIED


@dataclass(frozen=True)
class TrustedInteractionEvent:
    """Evaluator-private execution record with an explicitly safe projection."""

    request: ObjectInteractionRequest
    public_result: ObjectInteractionResult
    postcondition: OpenPostcondition | None
    joint_results: tuple[JointOpenResult, ...] = field(default_factory=tuple, repr=False, compare=False)
    private_object: TrustedObject | None = field(default=None, repr=False, compare=False)
    internal_reason: str | None = field(default=None, repr=False, compare=False)

    def redacted(self) -> dict[str, str]:
        return redact_interaction_event(self)


def redact_interaction_event(event: TrustedInteractionEvent) -> dict[str, str]:
    """Return the only representation safe to emit outside the evaluator.

    Building this dictionary from scratch, rather than deleting a few private
    fields from an internal event, prevents future low-level metadata from
    becoming public accidentally.
    """

    return event.public_result.to_public_dict()


OpenJointExecutor = Callable[[Any], JointOpenResult]


class TrustedInteractionSkill:
    """Resolve opaque object-level requests and execute sealed ``open`` skills."""

    def __init__(self, registry: OpaqueObjectRegistry, execute_open_joint: OpenJointExecutor) -> None:
        self._registry = registry
        self._execute_open_joint = execute_open_joint

    def handle_public_request(self, request: ObjectInteractionRequest) -> ObjectInteractionResult:
        """Execute a request and return only public, redacted feedback."""

        return self.execute_private(request).public_result

    def execute_private(self, request: ObjectInteractionRequest) -> TrustedInteractionEvent:
        """Evaluator-only execution path retaining the private diagnostic record."""

        if request.operation != "open":
            return self._rejected(request, "unsupported_operation")
        record = self._registry._resolve_private(request.instance_id)
        if record is None:
            return self._rejected(request, "unknown_opaque_id")

        joint_results: list[JointOpenResult] = []
        for joint in record.joints:
            try:
                result = self._execute_open_joint(joint)
            except Exception as exc:  # Low-level errors remain evaluator-private.
                joint_results.append(
                    JointOpenResult(
                        executor_succeeded=False,
                        open_fraction_before=None,
                        open_fraction_after=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if not isinstance(result, JointOpenResult):
                raise TypeError("execute_open_joint must return JointOpenResult")
            joint_results.append(result)

        postcondition = classify_open_postcondition(joint_results, spec=record.open_postcondition)
        completed = postcondition in {OpenPostcondition.SATISFIED, OpenPostcondition.ALREADY_SATISFIED}
        result = ObjectInteractionResult(
            request_id=request.request_id,
            instance_id=request.instance_id,
            operation="open",
            status="completed" if completed else "failed",
        )
        return TrustedInteractionEvent(
            request=request,
            public_result=result,
            postcondition=postcondition,
            joint_results=tuple(joint_results),
            private_object=record,
            internal_reason=None if completed else postcondition.value,
        )

    @staticmethod
    def _rejected(request: ObjectInteractionRequest, reason: str) -> TrustedInteractionEvent:
        return TrustedInteractionEvent(
            request=request,
            public_result=ObjectInteractionResult(
                request_id=request.request_id,
                instance_id=request.instance_id,
                operation="open",
                status="rejected",
            ),
            postcondition=None,
            internal_reason=reason,
        )


__all__ = [
    "JointOpenResult",
    "ObjectInteractionRequest",
    "ObjectInteractionResult",
    "OpaqueObjectRegistry",
    "OpenPostcondition",
    "OpenPostconditionSpec",
    "TrustedInteractionEvent",
    "TrustedInteractionSkill",
    "TrustedObject",
    "classify_open_postcondition",
    "redact_interaction_event",
]
