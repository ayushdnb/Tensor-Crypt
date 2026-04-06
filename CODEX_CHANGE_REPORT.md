# CODEX Change Report

## Objective
Execute a repository-grounded audit and implementation pass across mechanics, PPO ownership, checkpointing, telemetry, configuration wiring, performance, validation, and documentation without changing established simulation semantics unless a concrete bug was proven.

## Repository-Grounded Findings Summary

### Strong and already correct
- The runtime assembly order in `src/tensor_crypt/app/runtime.py` was already stable and semantically coherent.
- PPO ownership was already UID-centric rather than slot-centric, which made conservative hardening feasible.
- Catastrophe state was already designed as reversible runtime overlay state rather than destructive mutation of canonical trait storage.
- The viewer and telemetry surfaces already exposed enough state to extend operator diagnostics without changing mechanics.

### Weak, incomplete, or risky before this patch
- Runtime checkpoint publishing lacked a periodic scheduler and retention policy in the engine.
- Checkpoint pointer integrity was not validated strongly enough against manifest metadata.
- Several `config.py` knobs were effectively dead or only partially wired, creating silent no-op risk.
- Hot-path telemetry performed repeated parquet writes and repeated family aggregation work on every summary tick.
- Inference and bootstrap paths were using `no_grad`-style behavior conservatively but not the lower-overhead inference-only surface.
- Physics and engine hot loops were doing repeated `.item()` extraction and repeated mode flips that inflated Python and host-sync overhead.
- Strict manifest validation did not clearly fail when the operator requested manifest discipline but the manifest file was absent.

### Missing but high-value
- A benchmark harness suitable for repeatable headless runtime measurements.
- Scheduler and retention tests for periodic runtime checkpoints.
- Deterministic tie-break coverage for contested movement when a seeded pseudo-random policy is selected.
- Operator-facing checkpoint metadata for bundle size and checksum parity between manifest and latest pointer.

### Optimizations intentionally rejected
- Broad `torch.compile` application was rejected because the orchestration path is graph-break-heavy and ownership-sensitive.
- `torch.func` or family-stacked batched module execution was rejected because each UID owns an independent module and optimizer lifecycle, and forcing parameter stacking would materially raise checkpoint, optimizer, and semantic risk.
- Semantics-changing spatial or PPO redesign was rejected because the measured hotspots did not justify ownership or determinism risk.

## Files Changed
- `config.py`
- `scripts/benchmark_runtime.py`
- `src/tensor_crypt/agents/state_registry.py`
- `src/tensor_crypt/app/runtime.py`
- `src/tensor_crypt/checkpointing/__init__.py`
- `src/tensor_crypt/checkpointing/atomic_checkpoint.py`
- `src/tensor_crypt/checkpointing/runtime_checkpoint.py`
- `src/tensor_crypt/learning/ppo.py`
- `src/tensor_crypt/simulation/engine.py`
- `src/tensor_crypt/telemetry/data_logger.py`
- `src/tensor_crypt/telemetry/run_paths.py`
- `src/tensor_crypt/viewer/input.py`
- `src/tensor_crypt/viewer/main.py`
- `src/tensor_crypt/viewer/panels.py`
- `src/tensor_crypt/world/physics.py`
- `tests/test_benchmark_and_scheduler.py`
- `tests/test_physics.py`
- `tests/test_prompt7_checkpoint_atomicity.py`

## Major Changes and Rationale

### Checkpointing hardening
- Added periodic runtime checkpoint scheduling, retention, and latest published checkpoint tracking in `Engine`.
- Strengthened atomic checkpoint publication so bundle and manifest are written as same-directory temp files and promoted with `os.replace`.
- Added latest-pointer resolution and strict pointer-versus-manifest validation for tick, checksum, and bundle size.
- Added stricter runtime checkpoint validation for schema, UID ownership, and optional PPO/training surfaces.
- Fixed the manifest publication bug where the manifest initially recorded the temp bundle filename rather than the published filename.

### Performance and hot-path cleanup
- Switched inference-only action sampling and PPO bootstrap value resolution to `torch.inference_mode()`.
- Stopped redundant `brain.eval()` churn on every inference pass by restoring modules to eval mode after optimizer updates and at brain creation/restore time.
- Replaced several repeated tensor scalar extractions with list-based bulk extraction in `engine.py` and `physics.py`.
- Vectorized environment HP gain/loss and death-boundary clamping while keeping per-death context bookkeeping intact.
- Batched parquet writes behind bounded row buffers to remove repeated hot-path file I/O.
- Added summary export cadence control to suppress unnecessary tick-summary churn when operators want lower-cost telemetry.

### Config surface completion and guarding
- Wired runtime validation for guarded compatibility knobs so unsupported non-default values now fail loudly instead of silently doing nothing.
- Added and documented checkpoint cadence and retention knobs.
- Added and documented telemetry batching/cadence and viewer enrichment knobs.
- Wired viewer overlay behavior to the existing configuration surface.
- Wired deterministic contest tie-break selection to `PHYS.TIE_BREAKER`.

### Observability expansion
- Added buffered parquet row counts and PPO ownership counts into tick summaries.
- Added run metadata for checkpoint cadence/retention and telemetry batching settings.
- Preserved ledger semantics while expanding operator visibility into buffer, optimizer, lineage-depth, and catastrophe surfaces.

### Validation expansion
- Added tests for periodic runtime checkpoint scheduling and retention.
- Added tests for tick-summary cadence and parquet buffer flushing.
- Added tests for latest-pointer resolution and strict manifest requirements.
- Added a deterministic seeded tie-break test for contested movement.
- Added a benchmark harness smoke test.

## Semantic Risks Considered
- PPO ownership drift across slot reuse.
- Reward or point leakage across deaths, respawns, and terminal transitions.
- Determinism drift from tie-breaking or seeded catastrophe scheduling.
- Checkpoint incompatibility from manifest, pointer, or optimizer-state handling.
- Telemetry backpressure or unbounded buffering.
- Viewer semantic drift from overlay toggles or inspector enrichment.

## Performance Effect
- Headless CPU training benchmark improved from 10.2965 ticks/sec to 10.3378 ticks/sec, approximately +0.4 percent.
- Headless CUDA training benchmark improved from 4.8500 ticks/sec to 5.5874 ticks/sec, approximately +15.2 percent.
- CPU benchmark RSS delta fell from 171.60 MiB to 168.93 MiB.
- CUDA benchmark RSS delta fell from 806.75 MiB to 798.41 MiB.
- Steady-state profiling showed tick-summary parquet writes and repeated `Module.eval()` churn drop out of the top hotspots.

## Correctness Effect
- Strict manifest validation now rejects missing manifests when the operator asked for manifest discipline.
- Latest pointer resolution now proves it references the published bundle rather than trusting pointer metadata blindly.
- Runtime checkpoint restore now respects the `CAPTURE_PPO_TRAINING_STATE` surface explicitly.
- Contest resolution now exposes a documented deterministic seeded mode instead of only the hard-coded lowest-slot fallback.
- Unsupported config values now fail at runtime assembly rather than quietly pretending to be active.

## Validation Run
- `python -m compileall src scripts tests config.py run.py main.py`
- `pytest -q --basetemp .pytest_tmp_clean3`
- `pytest -q --basetemp .pytest_tmp_clean4 tests/test_prompt7_checkpoint_atomicity.py tests/test_benchmark_and_scheduler.py tests/test_physics.py`
- `python scripts/run_soak_audit.py --ticks 128 --seed 42 --width 20 --height 20 --agents 12 --walls 4 --hzones 2 --log-dir .pytest_tmp_clean_soak\soak_post_final --checkpoint-validate-every 32`

## Invariant Checklist
- Alive slot to UID bindings remain canonical and unique.
- Dead UIDs do not keep active slot ownership.
- PPO buffers remain UID-owned and terminal finalization is still explicit.
- Checkpoint capture and restore preserve registry, brains, catastrophe state, and RNG state consistently.
- Viewer overlays remain presentation-only and do not mutate canonical runtime state.
- Telemetry buffering does not change ledger semantics; it changes only flush timing.

## Failure-Mode Checklist
- Missing manifest under strict validation: hard failure.
- Pointer checksum or size mismatch against manifest: hard failure.
- Unsupported guarded config knobs: hard failure during runtime validation.
- Corrupt checkpoint bundle: detected by checksum and schema validation.
- Tick-summary telemetry pressure: bounded by `PARQUET_BATCH_ROWS` and `SUMMARY_EXPORT_CADENCE_TICKS`.

## Rule-Resolution Notes
- Contest tie breaks are still strength-first; the new seeded mode only changes how exact-strength ties are resolved.
- Periodic runtime checkpoints are saved after the tick has fully settled, including deaths, births, telemetry, and PPO updates for that tick.
- Catastrophe viewer gating affects only overlay visibility, not catastrophe scheduling or world overrides.

## Remaining Warnings and Caveats
- End-to-end CPU gains are intentionally modest because the dominant remaining costs are still per-agent brain forward passes, perception raycasting, and PPO update math.
- A stale sandbox-denied temp directory from a pre-fix soak run remained under `.pytest_tmp`; validation was rerun in clean temp roots and the current code path no longer produces that temp-directory publisher.
- The benchmark harness is intentionally headless and operational; it is not a substitute for long-horizon research training experiments.

## External Guidance Consulted
- PyTorch `torch.inference_mode` reference
- PyTorch reproducibility notes
- PyTorch AMP examples and `GradScaler` guidance
- PyTorch `torch.compile` graph-break guidance
- PyTorch `torch.func` stacking/functional-call references
- Python `os.replace` semantics for atomic same-filesystem publish
