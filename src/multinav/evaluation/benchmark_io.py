"""Load InteractiveNav benchmark episodes from JSON or deterministic gzip archives."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def resolve_benchmark_file(path: Path) -> Path:
    """Resolve a benchmark file, including the repository's compressed bundle."""

    path = Path(path)
    if not path.is_dir():
        return path
    for filename in ("benchmark.json", "benchmark.json.gz"):
        candidate = path / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Benchmark directory contains neither benchmark.json nor benchmark.json.gz: {path}"
    )


def load_benchmark_document(path: Path) -> tuple[Path, Any]:
    """Return the resolved input path and decoded JSON document."""

    benchmark_file = resolve_benchmark_file(path)
    opener = gzip.open if benchmark_file.suffix == ".gz" else open
    with opener(benchmark_file, "rt", encoding="utf-8") as stream:
        return benchmark_file, json.load(stream)


def load_benchmark_episodes(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load and validate the benchmark's episode collection."""

    benchmark_file, payload = load_benchmark_document(path)
    episodes = payload.get("episodes", []) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    if not isinstance(episodes, list):
        raise ValueError(f"Expected an episode list in {benchmark_file}")
    if not all(isinstance(item, dict) for item in episodes):
        raise ValueError(f"Every episode in {benchmark_file} must be a JSON object")
    return benchmark_file, [dict(item) for item in episodes]
