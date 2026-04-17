# Telemetry Spawn Schema Hardening Phase Report

## Objective

Harden the spawn-event parquet ledger family so `birth_ledger.parquet` and `genealogy.parquet` no longer infer `null` Arrow fields from an all-null first batch and then fail on later non-null writes.

## Source Artifacts Used

- `AGENTS.md`
- `c:\Users\ayush\Downloads\patch_telemetry_schema_hardening_and_adjacent_audit.md`
- live code under `tensor_crypt/telemetry/`, `tensor_crypt/population/`, and `tests/`

## Files Touched

- `tensor_crypt/telemetry/data_logger.py`
- `tests/test_telemetry_spawn_schema_hardening.py`
- `docs/technical_documents/99_assets/telemetry_spawn_schema_hardening_phase_report.md`

## Patch Adaptations

- The patch intent was preserved.
- One minimal live-code adaptation was required: the explicit spawn-ledger schema includes `mutation_extinction_bootstrap` because `tensor_crypt/population/respawn_controller.py` already emits that field in the extinction-bootstrap spawn path.
- The focused regression file uses a self-contained local runtime-build helper for the bootstrap-first birth-ledger scenario. This keeps the regression durable even though the shared pytest runtime-builder path did not exit cleanly in this shell environment.

## Tests Executed

### Attempted pytest runs

- `python -m pytest -q tests/test_telemetry_spawn_schema_hardening.py --basetemp=.pytest_tmp_spawn_schema -o cache_dir=.pytest_tmp_spawn_schema/cache`
  - test progress reached `.. [100%]`
  - in this shell environment the pytest process did not return before the command timeout, so this run was treated as an environment-level execution issue rather than a logic failure
- `python -m pytest -q tests/test_telemetry_ledgers.py::test_initial_population_bootstraps_birth_and_open_life_ledgers --basetemp=.pytest_tmp_spawn_schema -o cache_dir=.pytest_tmp_spawn_schema/cache`
  - produced the same non-returning pytest behavior after test execution began

### Direct assertion-based verification

- Focused spawn-ledger verification script:
  - null-only first flush followed by non-null string/bool/int values
  - verified both `birth_ledger.parquet` and `genealogy.parquet`
- Adjacent runtime verification script:
  - bootstrap-first birth-ledger path
  - later crowding-overlay failure row with `placement_crowding_policy_applied = "block_birth"`
  - genealogy compatibility fields including `child_idx`, `parent_idx`, `child_slot`, `parent_slot`, `child_uid`, `parent_uid`, `identity_schema_version`, and `telemetry_schema_version`

Result: direct verification passed.

## Pre-Existing / Environment Issues

- In this shell environment, pytest did not exit cleanly after reporting test progress completion for targeted runs. The same behavior appeared on an existing runtime-builder telemetry test, so this was treated as pre-existing or environment-specific rather than introduced by this patch.

## Compatibility and Migration Notes

- No simulation semantics changed.
- No checkpoint schema or checkpoint publication behavior changed.
- Artifact names are unchanged.
- `genealogy.parquet` remains the backward-compatible alias surface.
- Existing historical parquet files are not migrated; the explicit schema applies to new writes only.
- Bootstrap rows now materialize later optional spawn columns as nulls instead of letting early inference omit or null-poison those fields.

## Final Status

- Explicit spawn-ledger schema added and used for both `birth` and `genealogy`
- spawn-event rows normalized to the schema before buffering
- schema-backed parquet writes now reindex to schema column order
- focused regression coverage added for the exact null-first poison class and the adjacent bootstrap-first birth-ledger surface
- patch ready for review
