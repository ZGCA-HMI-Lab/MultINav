"""Public task-language helpers for V3 ROS evaluation.

The evaluator may pass an episode's natural-language goal to a policy, but it
must never derive a target from ``interactive_nav.target``: that record names
the benchmark-selected simulator instance.  This module intentionally accepts
only public language and uses a small, static synonym lexicon to make the
existing rule stack's category matching useful.
"""

from __future__ import annotations

import re
from typing import Any


_LEADING_TASK_WORDS = re.compile(
    r"^(?:find|locate|go to|navigate to|reach|pick up|approach|look for|search for)\s+(?:the\s+|a\s+|an\s+)?",
    flags=re.IGNORECASE,
)

# This is public task ontology, not a scene- or benchmark-derived lookup.  The
# values deliberately contain only semantic aliases that an ordinary language
# model or hand-built rule system could know from the instruction itself.
_PUBLIC_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "alarm clock": ("alarm clock", "alarmclock", "clock"),
    "alarmclock": ("alarmclock", "alarm clock", "clock"),
    "atomizer": ("atomizer", "spray bottle", "sprayer", "bottle"),
    "bed": ("bed",),
    "bottle": ("bottle", "atomizer", "spray bottle"),
    "bowl": ("bowl",),
    "cabinet": ("cabinet", "cupboard"),
    "chair": ("chair",),
    "clock": ("clock", "alarmclock", "alarm clock"),
    "couch": ("couch", "sofa"),
    "dresser": ("dresser", "chest_of_drawers", "chest of drawers"),
    "drawer": ("drawer",),
    "fridge": ("fridge", "refrigerator"),
    "garbage can": ("garbage can", "trash can", "trashcan", "ashcan", "bin"),
    "laptop": ("laptop", "laptop computer", "computer"),
    "microwave": ("microwave",),
    "refrigerator": ("refrigerator", "fridge"),
    "sofa": ("sofa", "couch"),
    "spray bottle": ("spray bottle", "atomizer", "sprayer", "bottle"),
    "television": ("television", "tv", "tv set"),
    "toilet": ("toilet", "crapper", "commode", "potty"),
    "tv": ("tv", "television"),
    "wardrobe": ("wardrobe", "closet"),
}


def _normalise_text(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    )


def public_goal_phrase(language: dict[str, Any] | None, instruction: str = "") -> str:
    """Return only a task-language noun phrase, never an internal target ID."""

    raw_language = dict(language or {})
    referral = raw_language.get("referral_expressions") or {}
    if isinstance(referral, dict):
        phrase = str(referral.get("object_name") or "").strip()
        if phrase:
            return phrase
    text = str(raw_language.get("task_description") or instruction or "").strip()
    text = _LEADING_TASK_WORDS.sub("", text).strip().rstrip(".?!")
    return text


def public_goal_labels(language: dict[str, Any] | None, instruction: str = "") -> list[str]:
    """Create deterministic, language-only labels for the existing ROS rule stack."""

    phrase = public_goal_phrase(language, instruction)
    normalised = _normalise_text(phrase)
    labels: list[str] = []
    if normalised:
        labels.append(normalised)
    for key, aliases in _PUBLIC_LABEL_ALIASES.items():
        if key == normalised or (key and key in normalised):
            labels.extend(aliases)
    # A noun phrase often includes colour/shape adjectives.  Retaining its
    # individual content tokens is a harmless fallback for category labels
    # such as ``mug`` and ``vase`` that need no explicit synonym entry.
    labels.extend(token for token in normalised.split() if len(token) > 2)
    return list(dict.fromkeys(label for label in labels if label))


def build_public_target_context(
    language: dict[str, Any] | None,
    instruction: str = "",
) -> dict[str, Any]:
    """Extract public language goal fields without selected-instance leakage.

    The ROS adapter adds transport-specific reliability settings later.  This
    helper deliberately carries no interaction-required flag: whether a door or
    container must be opened is part of the evaluated method's inference.
    """

    phrase = public_goal_phrase(language, instruction)
    labels = public_goal_labels(language, instruction)
    return {
        "enabled": bool(labels),
        "target_name": phrase,
        "object_labels": labels,
    }
