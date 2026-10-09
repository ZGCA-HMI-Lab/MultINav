---
license: cc-by-4.0
pretty_name: MultINav-Bench
language:
  - en
task_categories:
  - robotics
tags:
  - robot-navigation
  - interactive-navigation
  - object-goal-navigation
  - embodied-ai
  - simulation
  - molmospaces
  - procthor-10k
  - benchmark
size_categories:
  - 1K<n<10K
configs:
  - config_name: channel
    data_files: v1.2/benchmark/channel.json
  - config_name: container
    data_files: v1.2/benchmark/container.json
  - config_name: mixed
    data_files: v1.2/benchmark/mixed.json
  - config_name: all
    data_files: v1.2/benchmark/benchmark.json
---

# MultINav-Bench

**A benchmark for interaction-aware navigation: navigation tasks in which doors, sliding doors,
gate-like barriers, fridges, cabinets, and drawers change the structure of the navigable space.**

MultINav-Bench evaluates whether a policy can decide *when* an interaction is worth performing,
not only *how* to perform it. Each episode attaches interactive objects, their state
(`unknown` / `closed` / `open`), and the resulting state transition to the scene topology, so a
planner can exploit how interaction changes **reachability**, **visibility**, and **path cost**.

The benchmark is defined over [MolmoSpaces](https://github.com/allenai/molmospaces) scenes from
`procthor-10k` (val split) with the `rby1` mobile manipulator, and uses the
`interactive_nav_v3` episode schema.

## Domains

| Domain | Episodes | Interaction type | Effect on the navigation graph |
|---|---:|---|---|
| `channel` | 1000 | door, sliding door, gate-like barrier, movable blocker | changes connectivity / reachability |
| `container` | 1000 | fridge, cabinet, drawer | changes target visibility and accessibility |
| `mixed` | 1000 | channel then container | chained interaction: one interaction enables the next |
| **all** | **3000** | | |

## Files

```
v1.2/
├── benchmark/
│   ├── benchmark.json     # all 3000 episodes (list of episodes)
│   ├── channel.json       # 1000 channel episodes
│   ├── container.json     # 1000 container episodes
│   └── mixed.json         # 1000 mixed episodes
├── analysis/              # QC, distribution, and paper-facing statistics
│   ├── dataset_qc_report.md
│   ├── distribution_summary.json
│   ├── episodes_flat.{csv,jsonl}
│   ├── paper_data/*.csv
│   └── figures/*.{pdf,png}
├── provenance/            # resolved collection config and repair manifests
├── checksums.sha256
└── README.md
```

Verify integrity from inside `v1.2/`:

```bash
sha256sum -c checksums.sha256
```

## Episode format

Each episode follows the
[`interactive_nav_episode.schema.json`](https://github.com/ZGCA-HMI-Lab/MultINav/blob/main/benchmarks/schema/interactive_nav_episode.schema.json)
schema (`interactive_nav_v3`). Top-level fields:

| Field | Meaning |
|---|---|
| `house_index`, `scene_dataset`, `data_split` | source scene (`procthor-10k`, val) |
| `robot` | `rby1` with initial qpos |
| `cameras`, `img_resolution` | camera rig and resolution (640x480) |
| `scene_modifications` | scene edits applied before the episode |
| `task` | `nav_to_obj` target, thresholds, horizon |
| `language` | instruction type, disclosure, referral expressions |
| `interactive_nav` | the interactive-navigation ground truth (see below) |
| `source` | upstream provenance of the underlying NavToObj episode |

`interactive_nav` contains:

| Field | Meaning |
|---|---|
| `schema_version` | `interactive_nav_v3` |
| `case_id` | globally unique episode id |
| `interaction_domains` | `["channel"]`, `["container"]`, or `["channel", "container"]` |
| `interaction_requirement` | `required` (interaction needed for success) or `unnecessary` |
| `target` | target category / instance and grounding |
| `success_criteria` | distance + head-camera visibility conditions |
| `initial_state.interaction_states` | per-joint initial state and `semantic_state` |
| `interactions` | interaction objects, joints, and state transitions |
| `oracle_plan` / `oracle_plans` | reference interaction-then-navigation plan |

## Statistics

From `analysis/dataset_qc_report.md` and `analysis/dataset_health_report.json`:

- 3000 episodes, 3000 unique `case_id`, **0 duplicate** and **0 semantic issue**.
- Interaction required in 2654 / 3000 episodes (88.5%); `channel` 65.4%, `container` and `mixed` 100%.
- Effect labels: `reveal_target_object` 2000, `restore_reachability` 1654, `enable_interaction` 8.
- Visibility gain: `container` 100%, `mixed` 100%, `channel` 0% (by construction).
- Scenes: 639 houses total (`channel` 253, `container` 569, `mixed` 369).
- Reference GT path length (mean / median, m): `channel` 8.70 / 8.12, `container` 7.99 / 5.93,
  `mixed` 13.58 / 11.50, `all` 10.09 / 8.69.
- Target categories: 31 across all domains; container categories: 2 (fridge, cabinet/drawer).

## Evaluation

MultINav-Bench is evaluated with the MolmoSpaces `nav_to_obj` success criterion (distance
threshold plus head-camera visibility). The main reported metrics are:

| Metric | Meaning |
|---|---|
| `SR` | final task success rate |
| `SPL` | success-weighted path efficiency (0 on failure) |
| `Interaction Success Rate` | necessary interactions completed / necessary interactions, averaged over episodes that need interaction |
| `Interaction Precision` | completed interactions of the target category / all interaction attempts |
| `Total Cost` | `min((L_exec + lambda * A + mu * E) / B, 1)` on success, `1` on failure; defaults `lambda=0.3, mu=1, B=30` |

Results are reported per split (`all`, `channel`, `container`, `mixed`, `no-interaction`).
`no-interaction` episodes measure interaction restraint, so their interaction success rate may be
reported as `N/A`. See the
[evaluation protocol](https://github.com/ZGCA-HMI-Lab/MultINav/blob/main/docs/evaluation_protocol.md)
for full definitions.

## Usage

1. Install the evaluator from
   [MultINav](https://github.com/ZGCA-HMI-Lab/MultINav) and download the
   `procthor-10k` scenes and `molmospaces-bench-v2` resources:

```bash
pip install -e ".[molmospaces]"
hf download Piqiuni/MultINav-Bench --repo-type dataset --local-dir benchmarks/data
```

2. Point the evaluator at the benchmark file:

```bash
multinav-eval --benchmark benchmarks/data/v1.2/benchmark/benchmark.json --policy <name>
```

This card is tracked at `benchmarks/dataset_card.md` in that repository, next to
the episode schema it describes.

## Provenance

Episodes are derived from the upstream MolmoSpaces `NavToObjProcthor10kBench_20260112` benchmark
(val split). The `source` field records the upstream episode that each sample was built from and
is provenance-only: the released episodes are self-contained and do not require the referenced
upstream H5 files. The `provenance/` directory records the resolved collection config and the
repair manifests used to build this release.

This release is `interactive-nav-v3-procthor10k-val-release-v1.2`.

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en), consistent with
the MolmoSpaces data subsets it is derived from. Objaverse-derived content in the underlying
scenes is licensed under ODC-BY 1.0.

## Citation

```bibtex
@article{kim2026molmospaces,
  title={MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation},
  author={Kim, Yejin and Pumacay, Wilbert and Rayyan, Omar and Argus, Max and Han, Winson and VanderBilt, Eli and Salvador, Jordi and Deshpande, Abhay and Hendrix, Rose and Jauhri, Snehal and others},
  journal={arXiv preprint arXiv:2602.11337},
  year={2026}
}
```

The MultINav paper citation will be added upon publication.
