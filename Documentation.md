# Documentation

## Release Identity

Tensor Crypt Five-Brain is the multi-family bloodline release line. It supports five configured brain families, canonical observations, UID-owned PPO, binary-parented reproduction, catastrophe controls, telemetry ledgers, and runtime checkpoints.

## Operator Defaults

- Entry points: `python run.py`, `python main.py`, `tensor-crypt`
- Default family assignment: `BRAIN.INITIAL_FAMILY_ASSIGNMENT = "round_robin"`
- Default inference path: per-brain loop
- Optional benchmark accelerator: `SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False` by default

## Validation Summary

- Targeted release suite: 42 passed.
- Full pytest: 180 passed.
- Benchmark smoke: passed; final tick 18, final alive 4, vmap disabled.
- Benchmark record: `artifacts/release/five_brain_benchmark_smoke.json`.

## Known Limits

- `SIM.DTYPE` is guarded to `float32`.
- `RESPAWN.MODE` is guarded to `binary_parented`.
- The family-vmap path is a guarded benchmark path, not a semantic replacement for UID-owned brains.

## Release Notes

### 1.0.0 Release

- Establishes the five-family line as the canonical public runtime.
- Removes the single-family launch preset from this branch.
- Refreshes branch-facing README and architecture documentation.
- Validated with full pytest and a headless benchmark smoke.
