"""The front page must render: every local reference points at a tracked file.

The README carries the paper figure and links into `docs/`, `benchmarks/` and the
license files. A renamed path would silently degrade the front page, so the
references are resolved against the repository.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _local_targets(pattern: re.Pattern[str]) -> list[str]:
    text = README.read_text(encoding="utf-8")
    return [
        target.split("#", 1)[0]
        for target in pattern.findall(text)
        if target
        and not target.startswith(("http://", "https://", "mailto:"))
        and not target.startswith("#")
    ]


def test_front_page_shows_a_figure():
    assert _local_targets(IMAGE), "expected the front page to show the paper figure"


def test_front_page_image_references_are_tracked():
    missing = [
        target
        for target in _local_targets(IMAGE)
        if not (REPO_ROOT / target).is_file()
    ]
    assert not missing, f"README references missing images: {missing}"


def test_front_page_links_resolve():
    missing = [
        target
        for target in _local_targets(LINK)
        if not (REPO_ROOT / target).exists()
    ]
    assert not missing, f"README links to untracked paths: {missing}"
