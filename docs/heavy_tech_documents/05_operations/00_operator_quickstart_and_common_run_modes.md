# Operator Quickstart and Common Run Modes

> Scope: Provide a conservative startup guide for launching the repository through its public surface and understanding the common operating modes visible in the uploaded dump.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- launch surface
- minimal startup sequence
- headless benchmark and soak modes
- what a successful startup should produce
- safety checks before resuming from checkpoints

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)
- [Runtime assembly](../02_system/02_runtime_assembly_launch_sequence_and_session_graph.md)

## 1. Canonical startup route

Use the repository-root launch surface that routes into `tensor_crypt.app.launch.main`. The dump identifies `run.py` as the canonical root-level start surface for repository users. The compatibility entrypoint `main.py` also forwards into the same canonical launch path.

## 2. Expected startup actions

A normal interactive launch should:
1. validate runtime configuration
2. set deterministic seeds
3. create a run directory
4. print startup diagnostics such as device, grid size, initial agent count, and run directory
5. assemble the runtime graph
6. launch the viewer

## 3. Common run modes visible in the dump

### Interactive viewer run
Use the normal launch path. This is the standard operator mode.

### Headless benchmark harness
The dump contains a benchmark script with arguments for ticks, warmup ticks, device, grid size, agent count, wall count, H-zone count, PPO settings, checkpoint cadence, and experimental family-vmap toggles. Use this when the goal is reproducible measurement rather than visual inspection.

### Headless soak audit
The dump also contains a soak runner that:
- runs for many ticks
- checks invariants
- periodically validates checkpoint save/load surfaces
- reports basic alive-count summaries

This is an integrity mode, not a throughput benchmark.

## 4. Startup checklist

- confirm the intended device selection
- confirm the config root directory for logs
- treat unsupported enum combinations as errors rather than as acceptable warnings
- if resuming, validate checkpoint artifacts before trusting the run

## 5. Dependency note

The test surface in the dump expects `pygame-ce` rather than the legacy `pygame` package name in dependency surfaces. Treat viewer startup issues through that lens first.


## Read next
- [Viewer UI controls, HUD, and inspector manual](01_viewer_ui_controls_hud_and_inspector_manual.md)
- [Run directory artifacts and file outputs](02_run_directory_artifacts_and_file_outputs.md)
- [Checkpointing, manifests, restore, and latest pointer](03_checkpointing_manifests_restore_and_latest_pointer.md)

## Related reference
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `interactive run`
- `headless benchmark`
- `soak audit`
- `startup checklist`
