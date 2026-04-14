# D99 - Documentation Audit Change Log

## Purpose

This document records the repository-truth enforcement pass performed on the Tensor Crypt technical-document bundle.

## Scope Boundary

This log summarizes the audit and repair work. It is not a substitute for the repaired chapters themselves.

## Evidence Basis

The entries below are based on the code inspection, targeted repository searches, manual runtime smoke checks, and partial test execution performed during this audit.

## Scope of This Audit

This file records the documentation verification and repair pass performed over `docs/technical_documents/`.

## What Was Reorganized

- moved the published bundle into numbered public folders:
  - `00_meta/`
  - `01_operations_and_config/`
  - `02_system_foundations/`
  - `03_world_and_simulation/`
  - `04_perception_brains_and_learning/`
  - `05_artifacts_validation_and_viewer/`
  - `06_boundaries_and_appendices/`
- moved planning prompts and drafting inputs into `98_authoring_inputs/`
- reserved `99_assets/`, `99_assets/diagrams/`, `99_assets/figure_sources/`, and `99_assets/tables/` for reviewed figure materials
- added `README.md` files to clarify that `98_authoring_inputs/` is archival and `99_assets/` is reserved rather than evidentiary

## Factual Corrections Made

- corrected the bundle to describe `tensor_crypt.*` as the canonical implementation owner and root files plus `engine/*` and `viewer/*` as compatibility or entry surfaces
- corrected the launch narrative to the actual path through `tensor_crypt.app.launch.main`, `setup_determinism`, `create_run_directory`, `build_runtime`, and `viewer.run()`
- corrected the world chapter to the current grid-channel contract and current catastrophe roster
- corrected the tick-order chapter to the current `Engine.step()` ordering, including the placement of reward staging, telemetry death finalization, respawn, and PPO updates
- corrected the reproduction chapter to the currently validated `binary_parented` mode and the explicit brain-parent, trait-parent, and anchor-parent roles
- corrected the observation chapter to the current canonical feature order and legacy-compatibility boundary
- corrected the brain chapter to the current five-family roster and the actual same-family vmap conditions
- corrected the PPO chapter to the UID-owned ownership model, the current reward form, and the active bootstrap and optimizer-validation behavior
- corrected the artifact chapter to the current run-directory files, HDF5 groups, Parquet ledgers, lineage export, and brain-snapshot naming
- corrected the checkpoint chapter to the current capture set, restore order, atomic publish path, manifest dependencies, and latest-pointer conditions
- corrected the validation chapter to the actual validation helpers and the current soak-script path `scripts/run_soak_audit.py`
- corrected the viewer chapter to the current control set, layout semantics, selection behavior, and diagnostic exposure

## Terminology Conflicts Resolved

- normalized "slot" versus "UID" across the bundle
- normalized "canonical observation surface" versus "legacy observation bridge"
- normalized "compatibility wrapper" versus "canonical implementation owner"
- normalized status labels:
  - `Active`
  - `Guarded`
  - `Compatibility surface`
  - `Currently unread`
  - `Adjacent background`
- normalized artifact references so manifests, checkpoint bundles, latest pointers, and ordinary run artifacts are not conflated

## Claims Downgraded Because Evidence Was Insufficient

- public config fields with no live read path were downgraded to currently unread rather than described as active features
- validation harnesses were described as evidence of testing rather than universal guarantees
- checkpoint publication was described as conditional rather than implied for every run
- `seed_bank_bootstrap` and `admin_spawn_defaults` extinction policies were not described as behaviorally distinct in the current controller because the implementation currently routes both through the same bootstrap-spawn loop
- `98_authoring_inputs/` was explicitly marked archival rather than authoritative
- `99_assets/` was explicitly marked reserved rather than treated as evidence of a reviewed figure corpus

## What Was Added for Clarity

- explicit purpose, scope boundary, and evidence-basis sections in the published chapters
- cross references among dependent chapters
- compact operator-safe config guidance
- explicit checkpoints-versus-ordinary-artifacts separation
- explicit implementation-versus-background boundary chapter
- contributor truth contract for future documentation edits

## What Could Not Be Fully Verified

- the full focused pytest sweep did not complete successfully in this environment
- repeated `pytest` attempts entered an `ERROR` state on `tests/test_identity_substrate.py::test_uid_is_monotonic_across_slot_reuse` and did not return a traceback before timing out under Python `3.14.2`
- because of that harness behavior, the validation phase relied on:
  - targeted repository searches
  - direct code inspection
  - existing test-file inspection
  - a manual runtime-build smoke check that succeeded

## Heaviest-Repair Documents

The most extensive repairs were applied to:

- `00_meta/00_documentation_bundle_index.md`
- `01_operations_and_config/10_operator_runbook_and_game_manual.md`
- `01_operations_and_config/11_config_reference_active_guarded_dead.md`
- `02_system_foundations/20_project_identity_runtime_boot_and_package_map.md`
- `05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md`

These chapters had the highest concentration of stale planning-era path references, outdated evidence framing, or high-risk opportunities for implementation drift.

## Verification Actions Performed

- inventoried the markdown corpus under `docs/technical_documents/`
- inventoried the main runtime modules and tests they describe
- reorganized published chapters away from planning prompts
- ran targeted repository searches over:
  - identity substrate terms
  - checkpoint validation and manifest terms
  - currently unread config candidates
  - catastrophe identifiers
- validated published markdown relative links
- validated that published markdown files were ASCII-only
- performed a manual runtime-build smoke check through the canonical build path
