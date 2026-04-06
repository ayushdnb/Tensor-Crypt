# Tensor Crypt Architecture Overview

## Final Folder Tree

```text
Tensor_Crypt/
  .agent/
    AGENTS.md
    PLANS.md
  docs/
    architecture_overview.md
    restructure_execution_plan.md
    restructure_mapping_audit.md
    restructure_validation_report.md
    history/
      ARCHITECTURE.md
      AUDIT_REPORT.md
      CLEANUP_AUDIT_REPORT.md
  src/
    tensor_crypt/
      app/
      agents/
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
    __init__.py                  # repo-root namespace shim for src package
  engine/                        # legacy compatibility wrappers
  viewer/                        # legacy compatibility wrappers
  scripts/
  tests/
  config.py
  run.py
  main.py
  pyproject.toml
  requirements.txt
  README.md
  pytest.ini
```

## Module Responsibilities
- `src/tensor_crypt/app`: launch-time setup and runtime graph assembly.
- `src/tensor_crypt/simulation`: simulation tick orchestration and catastrophe scheduling.
- `src/tensor_crypt/world`: grid substrate, procedural mapgen, perception, and physics.
- `src/tensor_crypt/agents`: brain architecture and UID/slot registry lifecycle contracts.
- `src/tensor_crypt/learning`: PPO storage, ownership, optimizer state, and updates.
- `src/tensor_crypt/population`: evolution helpers, binary reproduction, respawn/extinction policy.
- `src/tensor_crypt/checkpointing`: runtime checkpoint capture/restore and atomic manifest/checksum flow.
- `src/tensor_crypt/telemetry`: run path creation, parquet/HDF5 logging, lineage export.
- `src/tensor_crypt/viewer`: pygame viewer runtime, input controls, panels, rendering helpers.
- `src/tensor_crypt/audit`: deterministic/replay/save-load validation harness.

## Public Entrypoints
- `config.py` is the canonical configuration surface.
- `run.py` and `main.py` are public launch entrypoints.
- `tensor_crypt.*` is the primary import surface for implementation modules.

## Config Surface Contract
- `cfg` is defined only in root `config.py`.
- Internal modules access `cfg` through `src/tensor_crypt/config_bridge.py`.
- The bridge preserves root config ownership and avoids duplicated configuration state.

## Compatibility Policy
- `engine/*` and `viewer/*` remain explicit compatibility wrappers for legacy imports.
- Repo-root `tensor_crypt/__init__.py` extends package path to `src/tensor_crypt`, preserving `tensor_crypt.*` imports from source checkout without requiring path hacks.
- No legacy wrapper contains business logic; all behavior lives in `src/tensor_crypt/*`.

## Extending the Codebase Without Reintroducing Sprawl
1. Add new implementation modules under `src/tensor_crypt/<domain>/`.
2. Keep cross-domain orchestration in `simulation/engine.py` or `app/runtime.py`, not in low-level domain modules.
3. Avoid adding new root-level top-level packages for implementation logic.
4. Add compatibility wrappers only when preserving a proven external import contract.
5. If a new public surface is introduced, document it in README + this architecture overview and add import/behavior tests.
