"""The interactive navigation episode contract.

An episode is a self-contained record: scene identity, robot initialization,
cameras, task, language and the interactive navigation ground truth. MultINav
owns that contract; ``benchmarks/schema/interactive_nav_episode.schema.json`` is
its source of truth.

The Pydantic model implementing the contract still comes from MolmoSpaces, so it
is fetched lazily and never imported at module import time. This keeps
``multinav.evaluation`` importable without a simulator installed.

Migration note: once the Isaac Sim adapter lands, this module owns the model
directly and the simulator import disappears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multinav.core.simulator import load

if TYPE_CHECKING:  # pragma: no cover - typing only
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

_EPISODE_SCHEMA_MODULE = "molmo_spaces.evaluation.benchmark_schema"


def get_episode_spec_class() -> Any:
    """Return the ``EpisodeSpec`` model, importing the simulator lazily."""
    return getattr(load(_EPISODE_SCHEMA_MODULE), "EpisodeSpec")


def load_episode_spec(payload: dict[str, Any]) -> "EpisodeSpec":
    """Validate ``payload`` and return an ``EpisodeSpec`` instance."""
    return get_episode_spec_class().model_validate(payload)


__all__ = ["get_episode_spec_class", "load_episode_spec"]
