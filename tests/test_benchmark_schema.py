"""Structural checks for the tracked V3 benchmark schema and examples."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "schema" / "interactive_nav_episode.schema.json"
EXAMPLES = sorted((REPO_ROOT / "benchmarks" / "examples").glob("*.json"))


def test_schema_and_examples_are_tracked():
    assert SCHEMA_PATH.is_file()
    assert EXAMPLES, "expected at least one example episode"


def test_examples_have_required_episode_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    for path in EXAMPLES:
        episode = json.loads(path.read_text(encoding="utf-8"))
        missing = required - set(episode)
        assert not missing, f"{path.name} is missing {sorted(missing)}"
        assert episode["interactive_nav"]["schema_version"] == "interactive_nav_v3"
