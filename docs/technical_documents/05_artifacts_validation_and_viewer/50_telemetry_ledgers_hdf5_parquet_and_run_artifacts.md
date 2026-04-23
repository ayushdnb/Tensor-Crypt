# D50 - Telemetry Ledgers, HDF5, Parquet, and Run Artifacts

## Purpose

This document describes the ordinary run artifacts produced by Tensor Crypt and explains which runtime component owns each artifact family.

## Scope Boundary

This chapter covers run-directory artifacts and their producing surfaces. It does not restate checkpoint publication semantics in full; those belong to [D51](./51_checkpointing_atomic_publish_resume_and_schema_safety.md).

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.telemetry.run_paths`
- `tensor_crypt.telemetry.data_logger`
- `tensor_crypt.telemetry.lineage_export`
- `tensor_crypt.simulation.engine`

## Run Directory Creation

At startup, `tensor_crypt.telemetry.run_paths.create_run_directory` creates:

- the run root
- `snapshots/`
- `brains/`
- `heatmaps/`
- `config.json`
- `run_metadata.json`

The remaining ordinary artifacts are created by the logger and related runtime paths as the run proceeds.

## `run_metadata.json`

`run_metadata.json` records the run's metadata posture, including schema-version and subsystem-posture fields such as identity, observation, checkpoint, migration, viewer, catastrophe, and telemetry summaries.

This file is useful for interpreting a run without reading source code, but it should not be mistaken for the full state of a runtime checkpoint.

## HDF5 Surface

The ordinary HDF5 artifact is `simulation_data.hdf5`. Its top-level groups currently include:

- `agent_snapshots`
- `heatmaps`
- `agent_identity`

This is the dense snapshot-style telemetry surface, distinct from the row-oriented Parquet ledgers.

## Parquet Ledger Surface

The current logger writes the following ordinary Parquet ledgers:

- `birth_ledger.parquet`
- `genealogy.parquet`
- `life_ledger.parquet`
- `death_ledger.parquet`
- `collisions.parquet`
- `ppo_events.parquet`
- `tick_summary.parquet`
- `family_summary.parquet`
- `catastrophes.parquet`

## Important Ledger Notes

### `genealogy.parquet`

`genealogy.parquet` is a backward-compatible alias surface. It is populated with the same spawn-lineage rows used by the canonical birth-lineage path, rather than representing a distinct canonical lineage substrate.

### `life_ledger.parquet`

The logger can flush still-open life rows on close when `cfg.TELEMETRY.FLUSH_OPEN_LIVES_ON_CLOSE` is enabled.

### `catastrophes.parquet`

Catastrophe rows depend on the active catastrophe path and catastrophe logging behavior. The file name is stable, but its content volume depends on whether catastrophes are enabled and exercised.

## Lineage Export

The current durable lineage export is `lineage_graph.json`, produced from the canonical UID and parent-role substrate rather than from slot history alone.

## Brain Snapshot Surface

Periodic brain snapshots are written under:

- `brains/brains_tick_<tick>.pt`

This is separate from runtime checkpoints. Brain snapshots do not by themselves encode the full runtime, registry, and optimizer substrate needed for faithful resume.

Operator-selected live brain exports are written under:

- `brains/selected_exports/u<uid>/t<tick>_s<slot>_<family>.pt`
- `brains/selected_exports/u<uid>/t<tick>_s<slot>_<family>.json`

These exports are deliberate inspection artifacts for the live selected agent. They carry weights plus identity, family, topology, lineage, session, and PPO-state metadata, but they are not full runtime checkpoints.

## Buffering and Flush Behavior

The logger buffers Parquet rows and flushes by ledger when `cfg.TELEMETRY.PARQUET_BATCH_ROWS` is reached or when the logger is closed. This is a performance and visibility tradeoff rather than a semantic change to the ledger schema.

## Initial Population Artifacts

When deep ledgers are enabled, the initial spawn path writes birth-like lineage rows for the bootstrap population and opens life rows accordingly. This matters when interpreting a run that has not yet experienced much death or reproduction.

## Practical Consequences

Readers should carry forward the following:

- ordinary run artifacts begin with `config.json` and `run_metadata.json`
- HDF5 and Parquet surfaces are complementary rather than redundant
- `genealogy.parquet` is a compatibility alias surface
- lineage export is UID-based
- brain snapshots are not full runtime checkpoints
- selected-brain exports are logger-owned artifacts under `brains/selected_exports/`, not repository-root files

## Cross References

- Boot-time artifact creation: [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
- Tick-order consequences for telemetry timing: [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
- Checkpoint artifacts: [D51](./51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Validation of artifact integrity: [D52](./52_validation_determinism_resume_consistency_and_soak_methods.md)
