"""Minimal external policy factory for validating the benchmark integration.

Use this as a wiring smoke test, not as a navigation baseline.  Real policies
may implement the same ``reset``/``act``/``close`` interface and return any
action dictionary accepted by ``normalize_policy_action``.
"""

from __future__ import annotations

from typing import Any


class ExampleStopPolicy:
    """Observe once and stop, proving the external-policy seam end to end."""

    name = "example_external_stop"

    def __init__(self, *, reason: str = "external_policy_smoke_test") -> None:
        self.reason = str(reason)
        self.public_episode: dict[str, Any] | None = None

    def reset(self, episode: dict[str, Any]) -> None:
        self.public_episode = dict(episode)

    def act(self, observation: Any) -> dict[str, Any]:
        if self.public_episode is None:
            raise RuntimeError("reset() must be called before act()")
        if not hasattr(observation, "observation"):
            raise TypeError("Expected the public PolicyObservation contract")
        return {"kind": "stop", "reason": self.reason}

    def close(self) -> None:
        self.public_episode = None


def build_policy(
    *,
    public_episode: dict[str, Any] | None = None,
    reason: str = "external_policy_smoke_test",
    **_kwargs: Any,
) -> ExampleStopPolicy:
    """Factory compatible with ``--policy-factory module:callable``."""

    policy = ExampleStopPolicy(reason=reason)
    # Construction may inspect only the same public episode contract later
    # supplied to reset; the adapter never exposes simulator-private GT.
    if public_episode is not None:
        policy.public_episode = dict(public_episode)
    return policy
