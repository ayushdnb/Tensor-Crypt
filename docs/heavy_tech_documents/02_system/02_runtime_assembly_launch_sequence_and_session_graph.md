# Runtime Assembly, Launch Sequence, and Session Graph

> Scope: Explain the launch-time subsystem construction order and why the order is treated as a stability boundary rather than an arbitrary implementation choice.

## Who this document is for
Technical readers, maintainers, and operators who need to understand what is assembled before the first simulation tick.

## What this document covers
- entrypoint to viewer path
- determinism setup
- run-directory creation
- runtime validation
- subsystem assembly order
- why ordering matters

## What this document does not cover
- deep physics rules
- deep PPO math

## Prerequisite reading
- [Repository identity and public contract](00_repository_identity_entry_surfaces_and_public_contract.md)
- [Package layout and wrappers](01_package_layout_canonical_modules_and_compatibility_wrappers.md)

## 1. Top-level launch path

The root-level start surface routes into `tensor_crypt.app.launch.main`. That function performs:
1. determinism setup
2. run-directory creation
3. startup prints
4. runtime assembly through `build_runtime`
5. viewer launch

The launch module’s own docstring explicitly says it owns launch-time concerns and intentionally does **not** own simulation rules.

## 2. Runtime validation before assembly

`setup_determinism()` calls runtime configuration validation before setting seeds. That matters because the repository rejects unsupported configuration combinations early rather than letting the simulation drift into ambiguous runtime states.

## 3. Stable assembly sequence

The runtime builder assembles the following major graph:

1. `DataLogger`
2. `Grid`
3. `Registry`
4. `Physics`
5. `Perception`
6. `PPO`
7. `Evolution`
8. procedural map generation (`add_random_walls`, `add_random_hzones`)
9. static wall cache refresh
10. initial population spawn
11. `Engine`
12. `Viewer`

> **Invariant**
> Runtime assembly treats the order of map generation, initial spawn, engine construction, and viewer construction as stability-sensitive unless simulation semantics are being changed intentionally.

## 4. Why launch order matters

- map generation must happen before spawn so that spawn placement can respect walls and zones
- the static wall cache should be refreshed after map creation so physics uses the current geometry
- initial spawn must happen before engine construction if the logger is expected to bootstrap initial population state
- engine must exist before the viewer because the viewer reads engine state every render cycle

## 5. Session graph

See the dedicated asset:
- [Runtime assembly sequence](../assets/diagrams/system/runtime_assembly_sequence.md)

## 6. Launch-time boundary versus tick-time boundary

A useful separation is:
- **launch time**: validate configuration, seed randomness, create paths, construct subsystems
- **tick time**: catastrophes, observations, inference, physics, rewards, deaths, respawn, PPO, telemetry, checkpoints

This prevents the common documentation mistake of mixing “how the program starts” with “how the world evolves.”
## Read next
- [Runtime config taxonomy and knob safety](03_runtime_config_taxonomy_and_knob_safety.md)
- [Operator quickstart and common run modes](../05_operations/00_operator_quickstart_and_common_run_modes.md)

## Related reference
- [Run directory artifacts and file outputs](../05_operations/02_run_directory_artifacts_and_file_outputs.md)
- [Checkpointing, manifests, restore, and latest pointer](../05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `runtime assembly`
- `session graph`
- `launch-time boundary`
- `tick-time boundary`
