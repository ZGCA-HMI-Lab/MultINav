"""The scoring core must import without a simulator installed.

``docs/architecture.md`` requires ``multinav.evaluation`` never to import a
simulator. That is a property of the import graph, so it is checked in a
subprocess where ``molmo_spaces`` and ``mujoco`` cannot be imported and every
package module is imported one by one.

The two lists below are the current partition of the package. Importing a new
module fails until it is classified, and moving a module into
``SIMULATOR_FREE_MODULES`` is the step that retires simulator imports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

SIMULATOR_FREE_MODULES = [
    "multinav.collection",
    "multinav.collection.full_rollout_recorder",
    "multinav.collection.gt_trajectory",
    "multinav.collection.gt_operation",
    "multinav.collection.gt_parking",
    "multinav.collection.gt_topdown",
    "multinav.core",
    "multinav.core.episode",
    "multinav.core.joints",
    "multinav.core.simulator",
    "multinav.evaluation",
    "multinav.evaluation.benchmark_interaction_adapter",
    "multinav.evaluation.benchmark_interaction_executor",
    "multinav.evaluation.benchmark_io",
    "multinav.evaluation.benchmark_metrics",
    "multinav.evaluation.benchmark_policies",
    "multinav.evaluation.benchmark_types",
    "multinav.evaluation.episode_topdown",
    "multinav.evaluation.example_external_policy",
    "multinav.evaluation.goal_status",
    "multinav.evaluation.public_goal",
    "multinav.evaluation.public_ids",
    "multinav.evaluation.restricted_gt_perception",
    "multinav.evaluation.ros_navigation_stall",
    "multinav.evaluation.ros_object_goal_adapter",
    "multinav.evaluation.ros_policy_termination",
    "multinav.evaluation.trusted_interaction_skill",
    "multinav.platforms",
    "multinav.platforms.base",
    "multinav.sim",
    "multinav.sim.collect_full_gt",
    "multinav.sim.force_interaction_bridge",
    "multinav.sim.force_interaction_runtime",
    "multinav.sim.interactive_nav_v3",
    "multinav.sim.navigation_posture",
    "multinav.sim.render_full_gt",
]

# Modules that legitimately need the simulator, and why. This list may only
# shrink: each remaining entry is a roadmap item in ``docs/architecture.md``.
SIMULATOR_DEPENDENT_MODULES = {
    "multinav.collection.gt_map": "imports MolmoSpaces scene maps at module level",
    "multinav.evaluation.benchmark_runner": (
        "imports the MolmoSpaces JSON task sampler and the scene probe"
    ),
    "multinav.platforms.molmospaces": "the MolmoSpaces adapter itself",
    "multinav.sim.container_scene_probe": (
        "subclasses MolmoSpaces task samplers at class-creation time"
    ),
}

_PROBE = """
import importlib, json, pkgutil, sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"molmo_spaces", "mujoco"}:
            raise ImportError(f"blocked: {name}")
        return None

sys.path.insert(0, __SRC__)
sys.meta_path.insert(0, Blocker())

import multinav

try:
    import molmo_spaces  # noqa: F401
    blocked = False
except ImportError:
    blocked = True

names = sorted(m.name for m in pkgutil.walk_packages(multinav.__path__, prefix="multinav."))
result = {}
for name in names:
    try:
        importlib.import_module(name)
        result[name] = "ok"
    except ImportError as exc:
        result[name] = str(exc).splitlines()[0]
    except Exception as exc:  # pragma: no cover - surfaced as a test failure
        result[name] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"blocked": blocked, "modules": result}))
"""


@pytest.fixture(scope="module")
def import_probe() -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), env.get("PYTHONPATH", "")] if env.get("PYTHONPATH") else [str(SRC_ROOT)]
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.replace("__SRC__", repr(str(SRC_ROOT)))],
        capture_output=True,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_probe_really_blocks_the_simulator(import_probe):
    assert import_probe["blocked"], "molmo_spaces was importable; the probe proves nothing"


def test_package_modules_are_classified(import_probe):
    reported = set(import_probe["modules"])
    classified = set(SIMULATOR_FREE_MODULES) | set(SIMULATOR_DEPENDENT_MODULES)
    assert reported == classified, (
        f"unclassified modules: {sorted(reported - classified)}; "
        f"stale entries: {sorted(classified - reported)}"
    )


def test_simulator_free_modules_import_without_importing_the_simulator(import_probe):
    modules = import_probe["modules"]
    failures = {name: modules[name] for name in SIMULATOR_FREE_MODULES if modules[name] != "ok"}
    assert not failures, f"these modules must not need the simulator: {failures}"


def test_simulator_dependent_modules_are_blocked_by_the_simulator(import_probe):
    modules = import_probe["modules"]
    unexpected = {
        name: modules[name]
        for name in SIMULATOR_DEPENDENT_MODULES
        if not str(modules[name]).startswith("blocked:")
    }
    assert not unexpected, f"no longer simulator-dependent; reclassify them: {unexpected}"
