# Scope And Repo Map

## Evidence Order Used
1. Live repository under `tensor_crypt/`, `tests/`, packaging surfaces, and scripts.
2. Supplied blueprint file: `tensor_crypt_validation_blueprint (2).md`.
3. Additional `evolution.txt` evidence: not present in the working tree at audit start.

## Audit Scope
- Canonical runtime assembly and launch surfaces.
- UID/slot/lifecycle ownership.
- PPO buffer, optimizer, and training-state ownership by UID.
- Checkpoint capture, validation, load, restore, manifest, and latest-pointer behavior.
- Telemetry schema, flush, and artifact integrity.
- Catastrophe, respawn, observation, brain, viewer, and compatibility-wrapper integration.
- Long-run stability proxies, soak execution, memory-growth evidence, and non-finite detection.

## Canonical Implementation Map
- `tensor_crypt/app`: launch and runtime assembly.
- `tensor_crypt/agents`: brains and state registry.
- `tensor_crypt/world`: observation, perception, grid, map, physics.
- `tensor_crypt/simulation`: engine and catastrophes.
- `tensor_crypt/population`: reproduction and respawn.
- `tensor_crypt/learning`: PPO ownership and updates.
- `tensor_crypt/telemetry`: run paths, ledgers, lineage export.
- `tensor_crypt/checkpointing`: capture, validation, atomic publish, restore.
- `tensor_crypt/viewer`: headless-testable pygame-ce viewer surfaces.
- `tensor_crypt/audit`: determinism/resume/save-load-save harness.

## Public / Compatibility Surfaces
- `run.py`
- `main.py`
- `config.py`
- `engine/*`
- `viewer/*`

## Existing Validation Surfaces Observed
- `tests/` contains 29 test modules at audit start.
- `scripts/benchmark_runtime.py`
- `scripts/run_soak_audit.py`
- `tensor_crypt/audit/final_validation.py`

## Initial High-Risk Fault Domains
- UID/slot reuse drift and historical UID rebinding.
- PPO optimizer/buffer leakage across slots or same-family agents.
- Checkpoint topology mismatch or incomplete strict validation.
- Observation contract mismatch, especially canonical vs experimental surfaces.
- Telemetry schema poisoning from null-only early batches.
- Catastrophe overlay reversibility and scheduler restore drift.
- Viewer/headless divergence and compatibility-wrapper drift.
- Memory/backlog growth during longer runs or repeated resume chains.
