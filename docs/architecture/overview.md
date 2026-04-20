# Tensor Crypt Single-Brain Vmap Architecture Overview

This branch is the self-centric single-family release line. Public entrypoints
apply the single-family preset before runtime assembly, while retaining
compatibility surfaces needed by telemetry, checkpoints, tests, and viewer code.

## Canonical Structure

```text
Tensor_Crypt/
  tensor_crypt/
    agents/
    app/
    audit/
    checkpointing/
    learning/
    population/
    simulation/
    telemetry/
    viewer/
    world/
    config_bridge.py
    runtime_config.py
  engine/
    *.py                  # compatibility re-exports only
  viewer/
    *.py                  # compatibility re-exports only
  scripts/
  docs/
    architecture/
    technical_documents/
  tests/
  config.py
  run.py
  main.py
  pyproject.toml
  README.md
```

## Module Responsibilities

- `tensor_crypt.app`: launch-time setup and runtime graph assembly.
- `tensor_crypt.simulation`: simulation tick orchestration, catastrophe scheduling, and loop/vmap inference paths for the active branch family.
- `tensor_crypt.world`: grid substrate, procedural map generation, perception, and physics.
- `tensor_crypt.agents`: brains plus UID and slot lifecycle state.
- `tensor_crypt.learning`: PPO storage, ownership, optimizer state, and updates.
- `tensor_crypt.population`: evolution helpers, reproduction, and respawn control.
- `tensor_crypt.checkpointing`: runtime checkpoint capture, validation, and atomic publish flow.
- `tensor_crypt.telemetry`: run path creation, parquet/HDF5 logging, lineage export.
- `tensor_crypt.viewer`: pygame viewer runtime, input controls, panels, and rendering helpers.
- `tensor_crypt.audit`: deterministic, resume, and save-load validation harnesses.
- `tensor_crypt.runtime_config`: canonical config dataclasses and singleton `cfg`.

## Public Entrypoints

- `config.py` is the public config import surface.
- `run.py` is the canonical repository-root launch surface.
- `main.py` remains an equivalent compatibility launcher.
- `tensor-crypt` is the installed console entrypoint.
- `tensor_crypt.*` is the canonical implementation import surface.
- `engine.*` and `viewer.*` remain compatibility-only import surfaces.

## Config Contract

- `cfg` is defined canonically in `tensor_crypt.runtime_config`.
- Root `config.py` is a thin wrapper that re-exports the canonical surface.
- Internal modules consume config through `tensor_crypt.config_bridge` so there is one shared singleton.

## Compatibility Policy

- Compatibility modules are allowed only in `engine/`, `viewer/`, `config.py`, `run.py`, and `main.py`.
- Compatibility code must stay thin and never own simulation, training, checkpoint, telemetry, or viewer behavior.
- New implementation code belongs under `tensor_crypt/`, not at repository root.

## Extension Rules

1. Add new implementation modules under `tensor_crypt/<domain>/`.
2. Keep cross-domain orchestration in `tensor_crypt/app/runtime.py` or `tensor_crypt/simulation/engine.py`.
3. Keep compatibility-only re-exports in `engine/` or `viewer/`.
4. Avoid introducing another implementation root or package bridge.
5. Document any new public surface in `README.md`, this overview, and tests.
6. Keep the vmap path benchmarkable and reversible; do not treat enablement as proof that every workload benefits from it.
