"""The published dataset card is tracked here and must stay consistent.

The card in ``benchmarks/`` is the text uploaded as the Hugging Face dataset
README. Editing it in the repository and re-uploading is the supported flow, so
its front matter, its release layout and its self-links are checked here rather
than discovered after a release.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = REPO_ROOT / "benchmarks" / "dataset_card.md"
SELF_LINK = "https://github.com/ZGCA-HMI-Lab/MultINav/blob/main/"
EXPECTED_CONFIGS = {"channel", "container", "mixed", "all"}


def _front_matter(text: str) -> dict:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "the card must open with YAML front matter"
    return yaml.safe_load(match.group(1))


def test_card_is_tracked_as_the_hub_readme():
    text = CARD_PATH.read_text(encoding="utf-8")
    front_matter = _front_matter(text)

    assert front_matter["pretty_name"] == "MultINav-Bench"
    assert front_matter["license"] == "cc-by-4.0"
    assert {config["config_name"] for config in front_matter["configs"]} == EXPECTED_CONFIGS


def test_card_configs_point_at_the_released_layout():
    front_matter = _front_matter(CARD_PATH.read_text(encoding="utf-8"))

    for config in front_matter["configs"]:
        data_files = config["data_files"]
        assert data_files.startswith("v1.2/benchmark/"), data_files
        assert data_files.endswith(".json"), data_files


def test_card_links_to_files_that_exist_in_this_repository():
    text = CARD_PATH.read_text(encoding="utf-8")

    broken = []
    for target in re.findall(re.escape(SELF_LINK) + r"([^\s)\"]+)", text):
        if not (REPO_ROOT / target).is_file():
            broken.append(target)
    assert not broken, f"the card links to untracked files: {sorted(set(broken))}"
