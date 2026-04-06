# Tensor Crypt Audit Report

## Objective
Perform a research-grade verification pass over the Tensor Crypt simulation system: inspect the architecture, validate subsystem wiring, reproduce real defects, harden the code with minimal fixes, add meaningful automated tests, and leave a reproducible audit trail.

## Repository Overview
- Public entry surfaces: `config.py`, `main.py`, `run.py`
- Internal implementation: `tensor_crypt/*`
- Legacy compatibility facades: `engine/*`, `viewer/*`
- Major subsystems audited:
  - runtime/bootstrap: `tensor_crypt.app`
  - orchestration: `tensor_crypt.simulation.engine`
  - state/brains: `tensor_crypt.agents`
  - world/grid/perception/physics/mapgen: `tensor_crypt.world`
  - PPO: `tensor_crypt.learning`
  - evolution/respawn: `tensor_crypt.population`
  - telemetry/artifacts: `tensor_crypt.telemetry`
  - viewer: `tensor_crypt.viewer`

## Architecture Summary
- `run.py` and `main.py` are thin wrappers into `tensor_crypt.app.launch.main()`.
- `tensor_crypt.app.runtime.build_runtime()` assembles the runtime graph in this order: logger, grid, registry, physics, perception, PPO, evolution, map generation, initial spawn, engine, viewer.
- `tensor_crypt.simulation.engine.Engine.step()` is the sequencing authority for ticks.
- The registry is slot-based. Brains are owned per slot; life-cycle and respawn reuse those slots.
- Telemetry writes HDF5 snapshots and parquet event streams.
- The viewer is pygame-based but can be exercised headlessly with SDL dummy drivers.

## Baseline State Before Changes
- No automated pytest suite existed. `pytest -q` reported `no tests ran`.
- Direct module import auditing initially failed because `h5py`, `pandas`, and `pyarrow` were not installed in the environment. Those were installed so the real telemetry path could be verified.
- A direct runtime smoke uncovered multiple real defects once telemetry dependencies were available.

## Testing Strategy
The audit added unit, integration, regression, determinism, viewer-smoke, and soak coverage.

### Unit / Contract Tests
- Grid borders, cell occupancy helpers, heal-zone overlap modes, gradients
- Procedural map bounds safety
- Perception empty-batch contract, self-hit avoidance, zero-vision stability
- Physics wall collisions, ram behavior, contest resolution, death clearing
- PPO return/advantage math, buffer lifecycle, invalid mini-batch validation, update cleanup
- Logger artifact creation and parquet schema stability
- Import/package compatibility for root and legacy paths

### Integration / Regression Tests
- Runtime build and multi-tick engine execution with real telemetry
- Respawn metadata and lineage wiring
- Respawn inheritance when a slot has no existing brain object
- Extinction recovery path
- Seeded deterministic replay across separate runs
- Headless viewer draw path with dead-agent and heal-zone selections

### Soak / Endurance Verification
- Added `scripts/run_soak_audit.py` for reproducible headless engine stress runs with invariant checks.
- Executed a 128-tick soak in this environment with repeated PPO updates and no invariant failures.

## Commands Used
### Baseline
- `pytest -q`
- targeted import audit via inline Python import script
- targeted smoke scripts using `build_runtime()` and direct engine stepping

### Verification After Fixes
- `pytest -q`
- `python scripts/run_soak_audit.py --ticks 128`

## Defect Inventory
### 1. Run-directory collisions on launches started within the same second
- Severity: High
- Subsystem: telemetry / launch
- Manifestation: two launches in the same second could resolve to the same `logs/run_YYYYMMDD_HHMMSS` path, causing HDF5 locking/open failures.
- Cause: `create_run_directory()` used second-resolution timestamps only and did not reserve uniqueness.
- Fix: `tensor_crypt/telemetry/run_paths.py` now reserves the run directory and appends a numeric suffix when the timestamp path already exists.
- Regression coverage: `tests/test_logging_and_run_paths.py::test_create_run_directory_deduplicates_same_timestamp`

### 2. Collision parquet logging failed after an all-empty `contenders` batch
- Severity: High
- Subsystem: telemetry
- Manifestation: collision logging emitted `Invalid null value` / schema conversion errors once a later contest row introduced integer contenders after an earlier writer schema had inferred `list<null>`.
- Cause: schema inference on a first batch containing only empty contender lists.
- Fix: `tensor_crypt/telemetry/data_logger.py` now uses an explicit collision schema and normalizes contender lists to integer lists before writing.
- Regression coverage: `tests/test_logging_and_run_paths.py::test_collision_parquet_schema_handles_empty_then_nonempty_contenders`

### 3. Perception rays could report the observing agent as a hit
- Severity: High
- Subsystem: perception
- Manifestation: in empty space, some rays returned hit type `2` because flooring/clamping could keep a ray sample in the source cell and the code did not exclude the observing slot.
- Cause: `agent_hit` only checked `agent_vals >= 0`.
- Fix: `tensor_crypt/world/perception.py` now receives self slot indices and excludes self-hits; it also clamps vision ranges to avoid divide-by-zero instability.
- Regression coverage:
  - `tests/test_perception.py::test_perception_skips_self_hits_and_detects_cardinal_walls`
  - `tests/test_perception.py::test_zero_vision_is_stable_and_finite`

### 4. `TICK_BORN` was never wired and always stayed at zero after respawn
- Severity: High
- Subsystem: registry / respawn
- Manifestation: respawned agents recorded `TICK_BORN = 0` even when spawned much later.
- Cause: `Registry.tick_counter` was never advanced or passed into respawn.
- Fix: `tensor_crypt/agents/state_registry.py` now accepts explicit `tick_born`; `tensor_crypt/population/respawn_controller.py` passes the current tick.
- Regression coverage: `tests/test_registry_respawn.py::test_respawn_assigns_birth_tick_new_identity_and_parent_identity`

### 5. Respawn lineage metadata did not advance identities across lives
- Severity: Medium
- Subsystem: registry / respawn
- Manifestation: slot IDs were reused as agent identity forever; `next_unique_id` existed but was unused.
- Cause: respawn never assigned a fresh identity.
- Fix: respawned agents now get a new unique `AGENT_ID`, and `PARENT_ID` stores the parent agent identity rather than the parent slot index.
- Regression coverage: `tests/test_registry_respawn.py::test_respawn_assigns_birth_tick_new_identity_and_parent_identity`

### 6. Respawn into a slot with no existing brain produced a random policy instead of inherited policy
- Severity: High
- Subsystem: respawn / evolution
- Manifestation: if a slot had never owned a brain, respawn skipped inheritance and `spawn_agent()` created a fresh random brain.
- Cause: inheritance only happened when both parent and child brain objects already existed.
- Fix: `tensor_crypt/population/respawn_controller.py` now deep-copies the parent brain into an empty child slot before spawning.
- Regression coverage: `tests/test_registry_respawn.py::test_respawn_inherits_parent_brain_for_slot_without_existing_brain`

### 7. Population extinction was unrecoverable
- Severity: High
- Subsystem: respawn / engine integration
- Manifestation: once alive count reached zero, the engine advanced ticks but the respawn controller returned immediately because there were no alive parents.
- Cause: respawn required at least one alive parent.
- Fix: `tensor_crypt/population/respawn_controller.py` now supports extinction recovery by spawning default-trait agents with `parent_id = -1` when no parents exist.
- Regression coverage: `tests/test_registry_respawn.py::test_extinction_path_respawns_from_default_traits`

### 8. PPO buffers survived death and could mix trajectories across lives
- Severity: High
- Subsystem: PPO / evolution
- Manifestation: dead slots lost optimizer state but kept buffered transitions, so a respawned child could inherit old-life rollout data.
- Cause: `Evolution.process_deaths()` only cleared the optimizer.
- Fix: `tensor_crypt/learning/ppo.py` gained `clear_agent_state()` and `tensor_crypt/population/evolution.py` now clears both optimizer and buffer on death.
- Regression coverage: `tests/test_ppo.py::test_process_deaths_clears_agent_state`

### 9. PPO return/advantage computation crossed done boundaries incorrectly
- Severity: High
- Subsystem: PPO
- Manifestation: terminal transitions could bootstrap from future values and produce incorrect returns.
- Cause: the recursion used the wrong done boundary when walking the trajectory backward.
- Fix: `tensor_crypt/learning/ppo.py` now uses the current transition's done flag to gate bootstrapping and the next state's value correctly.
- Regression coverage: `tests/test_ppo.py::test_compute_returns_respect_done_boundaries`

### 10. Invalid PPO mini-batch configurations failed with an opaque `range()` error
- Severity: Medium
- Subsystem: PPO
- Manifestation: `cfg.PPO.MINI_BATCHES > batch_size` produced `ValueError: range() arg 3 must not be zero`.
- Cause: `mini_batch_size` could become zero without validation.
- Fix: `tensor_crypt/learning/ppo.py` now raises an explicit validation error before training starts.
- Regression coverage: `tests/test_ppo.py::test_invalid_minibatch_configuration_raises_clear_error`

### 11. CPU runs emitted deprecated AMP warnings
- Severity: Low
- Subsystem: PPO
- Manifestation: default construction of `torch.cuda.amp.GradScaler` warned on CPU-only verification runs.
- Cause: scaler creation ignored actual CUDA availability.
- Fix: PPO now only creates a CUDA grad scaler when AMP is enabled and CUDA is available.
- Regression coverage: covered implicitly by the clean pytest and soak runs.

## Files Added / Updated
### Updated code
- `tensor_crypt/telemetry/run_paths.py`
- `tensor_crypt/telemetry/data_logger.py`
- `tensor_crypt/world/perception.py`
- `tensor_crypt/agents/state_registry.py`
- `tensor_crypt/population/respawn_controller.py`
- `tensor_crypt/population/evolution.py`
- `tensor_crypt/learning/ppo.py`

### Added tests and audit tooling
- `pytest.ini`
- `tests/conftest.py`
- `tests/test_imports_and_compat.py`
- `tests/test_grid_and_mapgen.py`
- `tests/test_perception.py`
- `tests/test_physics.py`
- `tests/test_registry_respawn.py`
- `tests/test_ppo.py`
- `tests/test_logging_and_run_paths.py`
- `tests/test_engine_and_viewer_integration.py`
- `scripts/run_soak_audit.py`

## Test Inventory
- 28 pytest tests added
- Coverage areas:
  - imports / compatibility / config bridge
  - grid / mapgen / heal-zone behavior
  - perception contracts and stability
  - physics collision and death rules
  - registry spawn and respawn metadata
  - PPO math, cleanup, and validation
  - telemetry artifact creation and schema stability
  - end-to-end engine stepping
  - deterministic replay across separate seeded runs
  - headless viewer rendering smoke path

## Results After Fixes
- `pytest -q` => `28 passed`
- `python scripts/run_soak_audit.py --ticks 128` => completed successfully
- Observed soak result in this environment:
  - `SOAK_OK ticks=128 alive_final=12 min_alive=12 max_alive=12`
  - multiple PPO updates completed without invariant failures
  - no NaN/Inf state drift detected in registry or grid

## Remaining Risks / Unresolved Concerns
- Several config fields still appear unused by the current runtime (`SPAWN_MODE`, `MOVE_FAIL_COST`, `TIE_BREAKER`, `RAY_FIELD_AGG`, `RAY_STEP_SAMPLER`, `REWARD_FORM`, `SELECTION`, `METAB_FORM`). They were audited as wiring risks but not changed because there was no safe, code-grounded fix without changing semantics.
- Telemetry parquet writing still reports errors via `print()` rather than surfacing exceptions. The collision-schema defect is fixed, but fail-fast behavior is still a design decision left unchanged.
- Determinism is verified for separate seeded runs, not for concurrently stepped runtimes in the same process. The simulation still relies on global RNG state.
- Viewer interaction coverage is smoke-level only. Full human-input behavior, resizing stress, and long interactive sessions remain outside automated coverage.
- Large-map, GPU, and very long soak behavior beyond the 128-tick executed run were not exercised in this session, though the soak runner supports larger/manual runs.

## Recommended Next Steps
- Add CI that installs telemetry dependencies and runs `pytest -q` on every change.
- Add a scheduled longer soak job, for example `python scripts/run_soak_audit.py --ticks 512`.
- Decide whether unused config knobs are intentional compatibility placeholders or incomplete wiring, then either remove them or implement them with tests.
- Consider converting telemetry write failures from console prints into explicit errors under assertion/debug modes.

## Reproducible Commands
- `pytest -q`
- `python scripts/run_soak_audit.py --ticks 128`
- `python scripts/run_soak_audit.py --ticks 256`