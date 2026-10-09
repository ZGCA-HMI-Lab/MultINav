# InteractiveNav V3 evaluation protocol

This evaluator is intentionally separate from `molmo_spaces/evaluation`.  It
uses the same JSON episode representation and the same `JsonEvalTaskSampler`,
but V3 interaction supervision remains evaluator-only.

The standalone bundled-benchmark entry point is:

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --output-dir ./outputs/interactive-nav/evaluation \
  --policy noop --workers 1
```

It routes to `scripts/InteractiveNav/evaluation/benchmark_runner.py`; it does
not invoke or modify `molmo_spaces/evaluation`. The lower-level
`evaluate_interactive_nav_v3.py` entry point remains available for evaluating a
single benchmark shard or a caller-supplied V3 benchmark. When using that
lower-level entry point directly, set `MLSPACES_PINNED_ASSETS_FILE` to the
bundle's `pinned_assets.json`; the standalone wrapper does this automatically.

## Fixed input

The formal validation input is the complete v1.2 release bundled as three
losslessly compressed domain shards:

```text
scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2/
```

The formal scoring denominator is 3000: 1000 Channel, 1000 Container and 1000
Mixed. This release contains every candidate episode; no domain is truncated or
quality-gate filtered at bundle publication time. The evaluator must not modify
the frozen JSON. Each episode's `scene_modifications` is the authoritative
initial object/articulation state.

The evaluator transparently reads `.json` and `.json.gz`. The omitted aggregate
has the same episode list as `channel + container + mixed` in that order, so the
three shards retain all 3,000 formal episodes without duplicating the large
aggregate in Git. `manifest.json` records both archive and uncompressed hashes,
as well as the source aggregate hash.

### Runtime compatibility gate

Before formal scoring, the current protocol checks the live selected target against any
frozen terminal goal, authoritative robot start pose, every recorded
articulation state, all interaction object/joint bindings, initial target
visibility when specified, and critical scene-name compatibility.  A failed
check records `scoring_eligible=false` and is excluded from formal aggregation.

## Policy-visible information

The ordinary policy interface receives first-person observations, the task
language, elapsed time and its own action history. The raw sensor payload is a
deep-copied allow-list containing requested RGB camera keys, paired
`<camera>_depth` when enabled, matching `sensor_param_<camera>` calibration,
`robot_base_pose`, and `qpos`. It excludes
`env_states`, `task_info`, target/object poses, action sensors, segmentation, and
all other evaluator-private sensors. It does not receive
`interactive_nav`, selected-instance identifiers, controlling joints,
interaction ids, oracle plans, or validation evidence.  `scripted_oracle` is
the only exception; all its results are marked `uses_oracle_gt=true` and may
only be reported as an execution upper bound.

## Terminal conditions

The evaluator-private target check remains the final navigation score. It
becomes a pre-action rollout endpoint
only after the required interaction plan is complete (or immediately for an
`interaction_requirement=unnecessary` episode), so reaching an approach pose
cannot prevent the policy from issuing the required open action.

`interaction_conditioned_success` requires `task_success` plus all of:

1. every interaction in at least one valid oracle plan reaches 0.8 semantic
   open fraction;
2. all prerequisite interactions were executed before their dependent action;
3. for `interaction_requirement=unnecessary`, no interaction action occurred.

For backward compatibility, result field `success` is the same as
`interaction_conditioned_success`. Formal report `SR` follows the paper metric
and uses navigation success; reports expose interaction-conditioned success as
`interaction_conditioned_success_rate` / `interaction_conditioned_sr` alongside
required-interaction success, sequence success, wrong interaction count, path
length, and terminal reason.

## External policy factory

Use the mixed-domain wrapper to evaluate an ordinary Python policy across all
three domains without starting ROS:

```bash
python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --output-dir ./outputs/interactive-nav/my_policy \
  --policy factory \
  --policy-factory my_package.my_policy:build_policy \
  --policy-kwargs-json '{"checkpoint":"./checkpoints/model.pt"}' \
  --workers 3
```

The factory is constructed once per episode and may accept `public_episode`,
`episode`, `kwargs`, named policy options, or `**kwargs`. Its returned object may
provide `reset(episode_dict)` (or `reset()`), `act(PolicyObservation)` (or
`get_action(raw_observation)`), and optional `close()`.

The Python factory is a cooperative in-process plugin, not an isolation or
security boundary. It must be trusted: the public API does not pass benchmark
GT, but arbitrary plugin code can still open local files or import other
modules. Untrusted/leaderboard submissions require a separate process or
container with filesystem and network restrictions.

The generic factory receives only `PublicEpisode` fields and public sensor
observations; it never receives the live task, `interactive_nav`, oracle plans,
object names, or joint names. It may return `PolicyAction`, an action dictionary,
or `{ "action": ... }`. Supported kinds are `base`, `interact`, `view`, `observe`,
and `stop`. Generic interaction actions must select a visible target using
`pixel_xy` or `normalized_pixel_xy`; opaque `instance_id` belongs to the separate
restricted-GT ROS interface, and simulator `object_name` is debug/oracle-only.

`PublicEpisode` treats scene dataset, split, and house index as public benchmark
metadata, so scene-prior policies are permitted by this contract. A sensor-only
leaderboard must additionally hide or anonymize those fields in its isolated
submission boundary.

`scripts.InteractiveNav.evaluation.example_external_policy:build_policy` is a
zero-performance stop policy intended only to verify environment, policy, and
result wiring. Use `--episodes-per-domain 1 --max-steps 1 --no-render-topdown`
for the smallest real-scene integration run.

## Reproducibility and parallelism

The evaluator records an explicit simulator profile instead of inheriting a
branch-local default. The formal `interactive_nav_v3` profile is
`policy_dt_ms=200`, `ctrl_dt_ms=10`, `sim_dt_ms=10`, with
`legacy_branch_reset` RBY1 yaw mapping. The `upstream_main` comparison profile
uses `200/2/2 ms` and `nearest_equivalent` yaw. A `custom` profile is accepted
only when all dt and yaw values are supplied explicitly. Changing profile
changes the run signature and results must not be compared as the same protocol.

One episode owns one MuJoCo context.  `--workers N` starts up to N independent
processes; this is required because MuJoCo renderers and episode state are not
thread safe.  `--resume` only skips episode directories whose completed trace
has the identical run signature (benchmark hash, evaluation configuration, and
evaluator protocol implementation).  A partial, failed, or differently
configured trace is rerun. The mixed-domain `run_manifest.json` records the
wrapper implementation hash and each domain's unhashed signature payload,
including the full evaluation configuration, paper metric configuration,
protocol version, evaluator implementation hash, and direct policy-factory
module hash. These sit alongside the signature so the resolved run configuration
can be reconstructed and audited. Transitive policy dependencies and model artifacts remain caller-owned:
version them separately and do not resume an old output after changing them
without also changing the factory/kwargs identity.

## Required validation order

1. Run the V3 schema and `JsonEvalTaskSampler` tests, including initial joint
   readback.
2. Run the wrapper with `--dry-run` to verify the frozen bundle, pins, selection,
   signatures, and output manifest.
3. Run the example external stop policy for one episode per domain to verify
   scene replay, policy wiring, and report generation.
4. Use `scripted_oracle` only as a force-execution and prerequisite diagnostic.
5. Before a full run, smoke-test the intended worker count to verify per-process
   MuJoCo isolation and deterministic result merging.

## Optional ROS adapters

The lower-level runner retains optional `ros_bridge` and
`ros_object_goal_rule` protocol adapters, but this branch intentionally does not
ship or start the external ROS navigation workspace or its configuration. They
are not required by the bundled wrapper and are not part of this branch's
self-contained smoke-test contract. Integrators must supply and version that
workspace separately and use one evaluator worker per ROS master.
