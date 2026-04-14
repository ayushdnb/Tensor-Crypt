# D03 - Figure, Artifact, and Source Reference Index

## Purpose

This document standardizes how the technical-document bundle refers to figures, tables, run artifacts, checkpoint artifacts, scripts, and implementation sources.

## Scope Boundary

This document defines naming and reference conventions. It does not replace the artifact semantics in [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md), [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md), or [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md).

## Evidence Basis

The artifact names listed here were verified against:

- `tensor_crypt.telemetry.run_paths`
- `tensor_crypt.telemetry.data_logger`
- `tensor_crypt.checkpointing.atomic_checkpoint`
- `tensor_crypt.checkpointing.runtime_checkpoint`
- `scripts/benchmark_runtime.py`
- `scripts/run_soak_audit.py`

## Figure and Table Conventions

Use document-prefixed numbering.

- Figures: `Figure Dxx-n`
- Tables: `Table Dxx-n`

Examples:

- `Figure D31-1`
- `Table D50-2`

If a figure is referenced before it exists, mark it explicitly as deferred. Do not imply that a deferred figure is already present in the repository.

## Code Reference Conventions

Use inline code formatting for implementation references, and prefer the narrowest owner that supports the claim.

| Claim type | Preferred reference |
|---|---|
| package ownership | module path such as `tensor_crypt.app.runtime` |
| runtime assembly site | function path such as `tensor_crypt.app.runtime.build_runtime` |
| behavior owner | class or method path such as `tensor_crypt.learning.ppo.PPO.update` |
| config truth | field path such as `cfg.CHECKPOINT.STRICT_SCHEMA_VALIDATION` |
| artifact producer | function or method path such as `tensor_crypt.telemetry.data_logger.DataLogger.close` |

## Run Artifact Reference Index

The following names are safe to reference as checked runtime outputs when the corresponding feature paths execute.

| Artifact | Producer or owner | Notes |
|---|---|---|
| `config.json` | `tensor_crypt.telemetry.run_paths.create_run_directory` | Run-local config snapshot |
| `run_metadata.json` | `tensor_crypt.telemetry.run_paths.create_run_directory` | Run-local metadata snapshot |
| `simulation_data.hdf5` | `tensor_crypt.telemetry.data_logger.DataLogger` | HDF5 container for snapshots and identity groups |
| `birth_ledger.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Birth rows |
| `genealogy.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Backward-compatible alias surface for spawn lineage rows |
| `life_ledger.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Lifecycle-open and close rows |
| `death_ledger.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Death rows |
| `collisions.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Collision and contest events |
| `ppo_events.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | PPO update events |
| `tick_summary.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Tick-level summary rows |
| `family_summary.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Family summary rows |
| `catastrophes.parquet` | `tensor_crypt.telemetry.data_logger.DataLogger` | Catastrophe rows when the corresponding path is active |
| `lineage_graph.json` | `tensor_crypt.telemetry.lineage_export` | Durable lineage export |
| `brains/brains_tick_<tick>.pt` | `tensor_crypt.telemetry.data_logger.DataLogger` | Periodic brain snapshots |
| `snapshots/` | `tensor_crypt.telemetry.run_paths.create_run_directory` | Snapshot directory |
| `heatmaps/` | `tensor_crypt.telemetry.run_paths.create_run_directory` | Heatmap directory |

## Checkpoint Artifact Reference Index

| Artifact or pattern | Producer or owner | Notes |
|---|---|---|
| `checkpoints/` | checkpoint publication callers | Target directory for periodic checkpoint publication when checkpoint saving is active |
| `runtime_tick_*.pt` | `tensor_crypt.checkpointing.atomic_checkpoint.atomic_save_checkpoint_files` | Runtime checkpoint bundle pattern |
| `runtime_tick_*.pt.manifest.json` | `tensor_crypt.checkpointing.atomic_checkpoint.atomic_save_checkpoint_files` | Manifest file paired with bundle |
| `latest_checkpoint.json` | `tensor_crypt.checkpointing.atomic_checkpoint.atomic_save_checkpoint_files` | Optional latest pointer written only on the atomic-manifest path when enabled |

When discussing checkpoint publication semantics, cite both the file name and the publishing function.

## Validation and Script Reference Index

| Surface | Owner | Notes |
|---|---|---|
| Determinism probe | `tensor_crypt.audit.final_validation.run_determinism_probe` | Validation harness, not normal runtime mode |
| Resume-consistency probe | `tensor_crypt.audit.final_validation.run_resume_consistency_probe` | Validation harness |
| Catastrophe reproducibility probe | `tensor_crypt.audit.final_validation.run_catastrophe_repro_probe` | Validation harness |
| Save-load-save signature probe | `tensor_crypt.audit.final_validation.save_load_save_surface_signature` | Validation helper |
| Headless benchmark script | `scripts/benchmark_runtime.py` | Produces JSON benchmark output |
| Soak audit script | `scripts/run_soak_audit.py` | Headless long-run audit harness |

## Figure and Asset Honesty

The current audit did not find a checked-in publication figure corpus under `docs/technical_documents/99_assets/` that should be treated as present technical evidence. The `diagrams/`, `figure_sources/`, and `tables/` directories are therefore reserved locations rather than evidence of existing reviewed figures.

If future work adds figures:

1. place the rendered asset under `99_assets/diagrams/` or `99_assets/tables/` as appropriate
2. place the editable or source-form input under `99_assets/figure_sources/` when such a source exists
3. update the owning document and this index together
4. avoid phantom references to figures that are not checked in

## Cross References

- Artifact semantics: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint semantics: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Validation methods: [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
- Documentation governance: [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md)
