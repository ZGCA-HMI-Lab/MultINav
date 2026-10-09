# Interactive navigation simulator branch

`interactive-nav/sim` descends from MolmoSpaces commit
`1320b266d2b47aaa81c5f7a419cb9d3474e6994d` and contains
simulator-facing interaction support plus a self-contained InteractiveNav V3
benchmark evaluator. It intentionally excludes the navigation/semantic decision
implementation maintained separately.

## Interface policy

`InteractiveNavSimulatorPolicy` is a safe demonstration policy. Its default
mode scans the scene and lists available operations without changing scene
state. Explicit `InteractionDemoCommand` entries demonstrate opening a door or
one selected drawer/cabinet joint.

The simulator seam is `SimulatorInteractionInterface`:

- `scan()` lists doors, containers, joints, ranges, positions, and normalized
  open fractions.
- `open_door(name, fraction)` moves only the door hinge.
- `set_joint_open_fraction(name, joint_index, fraction)` moves exactly one
  hinge or slider, including one drawer in a multi-drawer cabinet.

## Real-scene smoke test

`scripts/smoke_test_interaction_interface.py` attaches RBY1 to a local MuJoCo
scene, scans its articulations, and searches around each target for a robot pose
that is collision-free with the target both closed and open. It then checks one
door through `open_door` and every joint of the largest detected container
through `set_joint_open_fraction`. Each target runs through open fractions
`0, 0.5, 1, 0`, and container sibling joints must remain unchanged.

The scene XML must be in a layout where its relative mesh and texture paths
resolve. Run from an environment where this checkout is installed, or set
`PYTHONPATH` to the checkout:

```bash
python scripts/smoke_test_interaction_interface.py \
  --scene-xml <SCENE_XML> \
  --metadata <OPTIONAL_SCENE_METADATA_JSON> \
  --output <RESULT_JSON>
```

The command exits nonzero on a missing interface category, an invalid state
transition, sibling-joint movement, or failure to find a collision-free pose.

## Standalone benchmark evaluation

The repository includes the complete frozen ProcTHOR validation release in
`scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2`:

- Channel: 1,000 episodes
- Container: 1,000 episodes
- Mixed: 1,000 episodes

The three shards therefore contain the full 3,000-episode candidate set. The
previous v1.1 runtime-qualified bundle remains in the repository only for
historical/reproduction use; it is not selected by the default evaluator.

The archives, their SHA-256 digests, and the release-pinned robot/scene/object
versions are tracked in Git. The wrapper applies those asset versions, restores
all recorded object and articulation state before the first observation, and
evaluates the three domains in deterministic round-robin order.

The default `interactive_nav_v3` simulator profile explicitly locks the policy,
control, and physics periods to `200/10/10 ms` and uses the legacy RBY1 yaw
branch mapping shared with the complete experiment stack. The `upstream_main`
profile selects `200/2/2 ms` and nearest-equivalent yaw for controlled upstream
comparisons. Use `custom` only with all four dt/yaw options; the
resolved profile is included in every run signature and manifest.

Use the bundled stop policy for the smallest real-scene wiring check:

```bash
python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --output-dir ./outputs/interactive-nav/smoke \
  --episodes-per-domain 1 --max-steps 1 \
  --policy factory \
  --policy-factory scripts.InteractiveNav.evaluation.example_external_policy:build_policy \
  --no-render-topdown
```

Replace the factory path with an importable `module:callable` to evaluate a
policy. The callable receives `PublicEpisode` and per-step `PolicyObservation`
objects. Public observations are deep-copied and allow-listed to requested RGB
cameras, paired depth when enabled, their `sensor_param_*` calibration,
`robot_base_pose`, and `qpos`;
`env_states`, target/object poses, action sensors, segmentation, live tasks,
oracle plans, and simulator object/joint names are not exposed. Generic
interaction actions select visible objects by image pixel, not internal name.

The factory runs in-process as a cooperative plugin, not as a security sandbox.
It should be trusted code and could read local files on its own. Use process or
container isolation for untrusted submissions. Runtime/setup exceptions,
missing rows, and runtime-ineligible formal episodes produce a nonzero wrapper
exit code; ordinary policy failures remain valid scored outcomes and do not fail
the command. Top-down rendering is enabled by default and uses the frozen oracle
stage endpoints plus the tracked core scene-map loader; it does not import the
private benchmark-generation or navigation-method scripts.
Rendering remains a best-effort reporting artifact and does not change scoring
or the exit code. Custom benchmark audits may explicitly use
`--allow-runtime-ineligible`.

`scripted_oracle` is evaluator diagnostics only. It follows frozen waypoints and
uses the canonical locked-force interaction executor; its scores are not an
external-policy baseline. The evaluator retains optional protocol adapters for
compatibility, but this distribution does not ship or start the ROS navigation stack.

## Synchronization rule

Changes flow in one direction:

```text
MolmoSpaces upstream -> fork main -> interactive-nav/sim -> private algorithm integration
```

Review simulator changes before applying them to the private integration.
Keep the private algorithm implementation out of this public repository. The
root `.gitignore` excludes `Interactive-Nav-SG-nav/` and defaults to excluding
`scripts/InteractiveNav/*`, with an explicit allow-list for the evaluator,
schema, frozen benchmark, and simulator support files. Git ignore rules do not
remove already tracked files, so review every public commit and tag before
pushing.

`scripts/InteractiveNav/simulator_scope.txt` is the machine-readable shared
surface. In an integration checkout with both refs available, verify that its
blobs are identical by supplying the actual simulator and algorithm refs:

```bash
python scripts/InteractiveNav/check_simulator_scope_parity.py \
  --sim-ref <PUBLIC_SIMULATOR_REF> \
  --full-ref <PRIVATE_ALGORITHM_REF>
```
