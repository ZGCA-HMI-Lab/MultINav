"""Stable public identifiers shared by the standalone evaluator.

This module intentionally has no ROS or semantic-mapping dependency so the
simulator benchmark branch can run from a normal Python environment.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_PUBLIC_DOOR_ID_RE = re.compile(r"^door_(?:[0-9]{1,8}|x)$")


def opaque_door_instance_id(value: Any) -> str:
    """Return the legacy-compatible generic identity for a scene door alias."""

    raw = str(value or "").strip().casefold()
    if _PUBLIC_DOOR_ID_RE.fullmatch(raw):
        return raw
    canonical = raw.replace(" ", "_")
    canonical = re.sub(
        r"(?:^|_)(?:doorframe|doorway|door_leaf|door|portal)(?=_|$)",
        "_",
        canonical,
    )
    canonical = re.sub(r"_+", "_", canonical).strip("_") or "door"
    digest = hashlib.blake2s(canonical.encode("utf-8"), digest_size=4).digest()
    ordinal = int.from_bytes(digest, "big") % 9_999 + 1
    return f"door_{ordinal:04d}"
