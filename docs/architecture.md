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
```

Dependencies point downwards only. `multinav.evaluation` must never import a
simulator, and `multinav.platforms.molmospaces` is the only place allowed to import
`molmo_spaces`.

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
| `evaluation/benchmark_runner.py` | `molmo_spaces.evaluation.benchmark_schema`, `molmo_spaces.tasks.json_eval_task_sampler`, `molmo_spaces.configs.*`, `molmo_spaces.robots.*` | split: schema loading into `multinav` core, simulator construction into the adapter |
| `evaluation/episode_topdown.py` | `molmo_spaces.molmo_spaces_constants`, `molmo_spaces.utils.*` | move scene lookup behind the platform seam |
| `evaluation/restricted_gt_perception.py` | `molmo_spaces.utils.mj_model_and_data_utils.body_aabb` | move AABB query into the seam |
| `evaluation/benchmark_policies.py` | `molmo_spaces.policy.learned_policy.ros_bridge_policy` | optional ROS adapter |
| `sim/container_scene_probe.py`, `sim/interactive_nav_v3.py`, `sim/force_interaction_runtime.py` | `molmo_spaces.*` | move geometry/scene queries behind the seam |
| `collection/*` | `molmo_spaces.*` | optional collection extra |

## Status

- [x] Package skeleton, license, attribution.
- [x] Platform seam (`PlatformInteractionInterface`, registry).
- [x] V3 episode schema, examples and evaluation core ported.
- [x] Fork-only policies ported and pinned simulator version recorded.
- [ ] Move MolmoSpaces-specific imports behind the seam (tracked above).
- [ ] Isaac Sim adapter.
- [ ] Contract tests that run the same fixtures on both platforms.
