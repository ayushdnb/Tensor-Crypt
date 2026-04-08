# Tensor Crypt Architecture Overview

## Canonical Structure

```text
Tensor_Crypt/
  docs/
    architecture/
      compatibility.md
      overview.md
    reports/
      audits/
      history/
      patches/
      restructure/
      validation/
  src/
    engine/
    viewer/
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
  tensor_crypt/
    __init__.py
  engine/
    __init__.py
  viewer/
    __init__.py
  scripts/
  tests/
  config.py
  run.py
  main.py
  pyproject.toml
  requirements.txt
  README.md
```

## Module Responsibilities

- `src/tensor_crypt/app`: launch-time setup and runtime graph assembly.
- `src/tensor_crypt/simulation`: simulation tick orchestration and catastrophe scheduling.
- `src/tensor_crypt/world`: grid substrate, procedural map generation, perception, and physics.
- `src/tensor_crypt/agents`: brains plus UID and slot lifecycle state.
- `src/tensor_crypt/learning`: PPO storage, ownership, optimizer state, and updates.
- `src/tensor_crypt/population`: evolution helpers, reproduction, and respawn control.
- `src/tensor_crypt/checkpointing`: runtime checkpoint capture, validation, and atomic publish flow.
- `src/tensor_crypt/telemetry`: run path creation, parquet/HDF5 logging, lineage export.
- `src/tensor_crypt/viewer`: pygame viewer runtime, input controls, panels, and rendering helpers.
- `src/tensor_crypt/audit`: deterministic, resume, and save-load validation harnesses.
- `src/tensor_crypt/runtime_config.py`: packaged backing module for the public `config.py` surface.
- `src/engine`: installable legacy re-export package for `engine.*`.
- `src/viewer`: installable legacy re-export package for `viewer.*`.

## Public Entrypoints

- `config.py` is the canonical configuration surface.
- `run.py` and `main.py` are the supported launch entrypoints.
- `tensor-crypt` is the installed console entrypoint.
- `tensor_crypt.*` is the canonical implementation import surface.
- `engine.*` and `viewer.*` remain compatibility-only import surfaces.

## Config Contract

- `cfg` is defined only in root `config.py`.
- Internal modules consume config through `src/tensor_crypt/config_bridge.py`.
- The bridge preserves a single config source of truth while remaining repo-root safe.

## Compatibility Policy

- Repo-root `tensor_crypt/__init__.py` extends package path to `src/tensor_crypt`.
- Repo-root `engine/__init__.py` and `viewer/__init__.py` are package stubs only.
- Thin legacy wrapper modules live under `src/engine` and `src/viewer`.
- No compatibility layer owns simulation, training, checkpoint, telemetry, or viewer behavior.

## Extension Rules

1. Add new implementation modules under `src/tensor_crypt/<domain>/`.
2. Keep cross-domain orchestration in `app/runtime.py` or `simulation/engine.py`.
3. Keep compatibility-only re-exports under `src/engine/` or `src/viewer/`.
4. Avoid adding new root-level implementation packages.
5. Document any new public surface in `README.md`, this overview, and tests.
