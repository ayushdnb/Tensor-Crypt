# Tensor Crypt Architecture Overview

## Public Release Posture

This branch is the single-brain-vmap public line. The repository launch path applies the self-centric single-family preset before runtime construction and enables same-family vmap inference when eligible. The package still contains the broader bloodline-family substrate because family topology, UID ownership, and checkpoint metadata are compatibility-visible surfaces.

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
  docs/
    architecture/
    technical_documents/
  config.py
  run.py
  main.py
  pyproject.toml
  README.md
```

## Module Responsibilities

- `tensor_crypt.app`: launch-time setup, resume handling, and runtime graph assembly.
- `tensor_crypt.simulation`: simulation tick orchestration, catastrophe scheduling, and inference routing.
- `tensor_crypt.world`: grid substrate, procedural map generation, perception, and physics.
- `tensor_crypt.agents`: brain modules and UID/slot lifecycle state.
- `tensor_crypt.learning`: PPO storage, optimizer ownership, and updates.
- `tensor_crypt.population`: evolution helpers, reproduction, and respawn control.
- `tensor_crypt.checkpointing`: checkpoint capture, validation, resume policy, and atomic publication.
- `tensor_crypt.telemetry`: run path creation, Parquet/HDF5 logging, lineage export, and selected-brain export.
- `tensor_crypt.viewer`: pygame viewer runtime, input controls, panels, and rendering helpers.
- `tensor_crypt.audit`: programmatic determinism, resume, checkpoint, and artifact validation probes.
- `tensor_crypt.runtime_config`: canonical config dataclasses and singleton `cfg`.

## Entrypoints

- `python run.py` is the canonical repository-root launch command.
- `python main.py` remains an equivalent compatibility launcher.
- `tensor-crypt` is the installed console entrypoint.
- `config.py` is the public config import surface.
- `tensor_crypt.*` is the canonical implementation import surface.
- `engine.*` and `viewer.*` are compatibility-only import surfaces.

## Config Contract

- `cfg` is defined canonically in `tensor_crypt.runtime_config`.
- Root `config.py` re-exports the canonical config surface.
- Internal modules consume config through `tensor_crypt.config_bridge` so there is one shared singleton.
- Launch-time single-family defaults are applied in `tensor_crypt.app.launch`.

## Compatibility Policy

- Compatibility modules are allowed only in `engine/`, `viewer/`, `config.py`, `run.py`, and `main.py`.
- Compatibility modules must remain thin re-exports or launch shims.
- New implementation code belongs under `tensor_crypt/`.
- Compatibility code must not own simulation, PPO, checkpointing, telemetry, or viewer behavior.

## Extension Rules

1. Add implementation modules under `tensor_crypt/<domain>/`.
2. Keep cross-domain orchestration in `tensor_crypt/app/runtime.py` or `tensor_crypt/simulation/engine.py`.
3. Treat observation shape, family topology, UID ownership, schema versions, and checkpoint manifests as high-risk compatibility surfaces.
4. Keep experimental accelerators benchmarkable, reversible, and guarded by explicit config.
5. Update README and owning technical documents when public behavior changes.
