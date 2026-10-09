# Architecture

## Goal

The interactive navigation algorithm must be independent of any single simulator.
MolmoSpaces is the first target; Isaac Sim is the next. Therefore this repository
contains only the *incremental* interactive navigation layer and talks to a simulator
through one narrow seam.

## Layers

```text
benchmarks/            episode schema + small examples (full release on Hugging Face)
        |
multinav.evaluation/   benchmark runner, metrics, goal status, policies   [platform-agnostic]
        |
multinav.sim/          interactive_nav_v3 episode model, interaction runtime  [platform-agnostic]
        |
multinav.platforms/    PlatformInteractionInterface + registry             [seam]
        |
multinav.platforms.molmospaces/   MuJoCo / MolmoSpaces adapter
multinav.platforms.isaacsim/      (planned)

multinav.core/         episode contract, articulation math, deferred simulator access
                       shared by every layer above; imports no simulator
```

Dependencies point downwards only. `multinav.evaluation` and `multinav.core`
must import without a simulator installed, and
`multinav.platforms.molmospaces` is the intended place for `molmo_spaces`
imports.

## Simulator-free imports

A simulator-free import path is a property of the import graph, so it is
enforced rather than asserted: `tests/test_simulator_free_imports.py` imports
every package module in a subprocess where `molmo_spaces` and `mujoco` raise on
import, and compares the result against an explicit partition of the package.
A new module fails that test until it is classified, and a module that becomes
importable without the simulator fails its "still dependent" expectation, so
the list can only shrink.

Modules that still resolve a simulator package at import time do so through
`multinav.core.simulator.lazy_module`, which defers the import to first
attribute access. That keeps the ported call sites unchanged while the seam is
being built; each proxy disappears as its symbol moves behind
`multinav.platforms`.

## The seam

`multinav.platforms.base.PlatformInteractionInterface` exposes three operations:

| Method | Meaning |
|---|---|
| `scan()` | every door, container and articulation with public state |
| `open_door(name, fraction)` | open one door through its hinge |
| `set_joint_open_fraction(name, joint_index, fraction)` | move exactly one hinge or slider |

Only *public* state crosses the seam: object name, category, joint index, joint type,
closed/open positions and normalized `open_fraction`. Joint axes, hinge geometry,
task recipes and evaluator-private identifiers never cross it.

## Decoupling roadmap

The ported code still touches `molmo_spaces` in the following places. Each becomes
either a seam implementation or is removed:

| Area | Current import | Target |
|---|---|---|
| `platforms/molmospaces/*` | `molmo_spaces.env.*`, `molmo_spaces.policy.*` | stays (this is the adapter) |
| `evaluation/benchmark_runner.py` | `multinav.core.episode` for the contract (done); still imports `molmo_spaces.tasks.json_eval_task_sampler`, `molmo_spaces.configs.*`, `molmo_spaces.robots.*` | simulator construction into the adapter |
| `evaluation/episode_topdown.py` | `molmo_spaces.*` deferred inside functions | move scene lookup behind the platform seam |
| `evaluation/restricted_gt_perception.py` | `molmo_spaces.utils.mj_model_and_data_utils.body_aabb` deferred inside a function | move AABB query into the seam |
| `evaluation/benchmark_policies.py` | `molmo_spaces.policy.learned_policy.ros_bridge_policy` deferred inside a function | optional ROS adapter |
| `sim/container_scene_probe.py` | `molmo_spaces.*` at module level | move geometry/scene queries behind the seam; the class-level task-sampler subclasses are the blocker |
| `sim/interactive_nav_v3.py` | `multinav.core.episode` for the contract (done) | none remaining |
| `sim/force_interaction_runtime.py`, `sim/force_interaction_bridge.py` | `mujoco`, `molmo_spaces.env.data_views.Door` through `lazy_module` | move articulation geometry behind the seam |
| `collection/*` | `molmo_spaces.*` in `gt_map` only | optional collection extra |

## Status

- [x] Package skeleton, license, attribution.
- [x] Platform seam (`PlatformInteractionInterface`, registry).
- [x] V3 episode schema, examples and evaluation core ported.
- [x] Fork-only policies ported and pinned simulator version recorded.
- [x] Episode contract and articulation math owned by `multinav.core`.
- [x] Scoring core (`benchmark_metrics`, `benchmark_interaction_executor`) and
      the collection helpers import without a simulator; the partition is pinned
      by `tests/test_simulator_free_imports.py`.
- [ ] Move the remaining MolmoSpaces-specific imports behind the seam
      (`benchmark_runner`, `container_scene_probe`, `collection/gt_map`).
- [ ] Isaac Sim adapter.
- [ ] Contract tests that run the same fixtures on both platforms.
