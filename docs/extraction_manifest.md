# Extraction manifest

## Source

| | |
|---|---|
| Repository | `https://github.com/piqiuni/molmospaces-interactive-nav.git` |
| Branch | `interactive-nav/sim` |
| Commit | `16a5a5abdb5740556774f7b1ea8c1934f5163ea3` (merge of PR #1) |
| Upstream base | `allenai/molmospaces` `1320b266d2b47aaa81c5f7a419cb9d3474e6994d` |

`interactive-nav/sim` is a fork branch: upstream MolmoSpaces plus 108 changed files
(83 added, 25 modified). This repository keeps only the incremental interactive
navigation layer; upstream MolmoSpaces is an external dependency.

## Ported

| Source (on `interactive-nav/sim`) | Destination | Notes |
|---|---|---|
| `scripts/InteractiveNav/evaluation/*.py` (17 modules, 13.6k lines) | `src/multinav/evaluation/` | runner, metrics, goal status, ROS protocol adapters |
| `scripts/InteractiveNav/evaluation/evaluation_protocol.md` | `docs/evaluation_protocol.md` | |
| `scripts/InteractiveNav/{interactive_nav_v3,force_interaction_runtime,force_interaction_bridge,container_scene_probe,navigation_posture,collect_full_gt,render_full_gt}.py` (12.1k lines) | `src/multinav/sim/` | simulator-side episode model and interaction runtime |
| `scripts/InteractiveNav/collection/*` (8 files) | `src/multinav/collection/` | GT trajectory collection helpers |
| `scripts/InteractiveNav/dataset_definition/v3/interactive_nav_episode.schema.json` | `benchmarks/schema/` | |
| `scripts/InteractiveNav/dataset_definition/v3/examples/*.json` (5) | `benchmarks/examples/` | |
| `molmo_spaces/env/interaction_interface.py` | `src/multinav/platforms/molmospaces/interface.py` | newly added upstream-of-the-fork module; now implements the seam |
| `molmo_spaces/policy/interactive_nav_sim_policy.py` | `src/multinav/platforms/molmospaces/demo_policy.py` | demonstration policy, explicitly *not* a navigation algorithm |
| `molmo_spaces/policy/learned_policy/{organized_depth_scan,realtime_gt_observation,ros_bridge_policy}.py` | `src/multinav/platforms/molmospaces/policies/` | fork-only modules (4,001 lines): the evaluator imports `ros_bridge_policy`, so they must travel with this repository |
| `docs/interactive_navigation_simulator_branch.md` | `docs/simulator_branch.md` | |

## Import rewrites applied

```text
scripts.InteractiveNav.evaluation.<mod>      -> multinav.evaluation.<mod>
scripts.InteractiveNav.collection.<mod>      -> multinav.collection.<mod>
scripts.InteractiveNav.force_interaction_*   -> multinav.sim.force_interaction_*
scripts.InteractiveNav.navigation_posture    -> multinav.sim.navigation_posture
scripts.InteractiveNav.render_full_gt        -> multinav.sim.render_full_gt
scripts.InteractiveNav import container_scene_probe -> multinav.sim import container_scene_probe
scripts.InteractiveNav import interactive_nav_v3    -> multinav.sim import interactive_nav_v3
molmo_spaces.env.interaction_interface       -> multinav.platforms.molmospaces.interface
molmo_spaces.policy.learned_policy.<mod>     -> multinav.platforms.molmospaces.policies.<mod>
```

## External simulator pin

`molmo_spaces` is pinned to `1320b266d2b47aaa81c5f7a419cb9d3474e6994d`, the commit the
interactive navigation layer was built against. Later upstream commits rename or
remove modules this adapter uses (for example `molmo_spaces.env.mj_extensions`), so
the pin is a requirement, not a convenience.

## Deliberately not ported

| Item | Reason |
|---|---|
| `Interactive-Nav-SG-nav/` (ROS catkin workspace, ~3,500 files) | private algorithm implementation; stays in the working fork |
| Vendored third-party (`pytorch3d-main/`, `or-tools/`, `openslam_gmapping/`, `bin/wheels/`, `*.zip`) | must not be vendored; install as dependencies |
| 25 in-place modifications to upstream `molmo_spaces/` files | upstream simulator changes, not incremental interactive navigation; see `docs/architecture.md` for the ones still needed behind the seam |
| Benchmark archives (`*.json.gz`, ~39 MB in the source branch) | hosted on Hugging Face instead (`Piqiuni/MultINav-Bench`) |
| Machine paths (`/home/ldl/...`) in docs and scripts | local paths; not portable |

## Local paths

The ported files carry no machine paths: absolute paths in the source branch
became package-relative imports during the port. Simulator packages are resolved
through `multinav.core.simulator.lazy_module` where an import cannot yet move
behind the seam; `docs/architecture.md` lists the modules that remain.

## Added after the port

| Item | Contents |
|---|---|
| `src/multinav/core/` | episode contract, articulation math, deferred simulator access |
| `tests/` | platform seam, V3 schema, articulation convention, simulator-free imports |
