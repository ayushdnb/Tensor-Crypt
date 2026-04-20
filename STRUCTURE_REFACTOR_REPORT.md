# Structure Refactor Report

## Goal

Refactor the repository into a single clear implementation layout while preserving simulation, PPO, checkpoint, viewer, and public execution semantics.

## Before / After Rationale

Before this refactor, the repository had three overlapping structural stories:

- the real implementation lived under `src/tensor_crypt/`
- the repository root exposed a same-name `tensor_crypt/` namespace bridge
- legacy `engine.*` and `viewer.*` imports were split between root package stubs and `src/engine` / `src/viewer` wrapper modules

That created duplicate truth locations, root/package confusion, and packaging metadata that depended on bridge behavior rather than a direct source tree.

After this refactor:

- `tensor_crypt/` at repository root is the single implementation package
- `config.py`, `run.py`, and `main.py` are the only root public entry modules
- `engine/` and `viewer/` remain only as thin legacy compatibility packages
- `src/` compatibility duplication is removed
- packaging metadata now describes the real tree directly instead of routing through `src`

## Canonical Decisions

### Canonical package decision

- Canonical implementation root: `tensor_crypt/`
- Removed duplicate implementation under `src/tensor_crypt/`
- Root `tensor_crypt/__init__.py` is now a real package marker, not a namespace/path-extension bridge

### Canonical entrypoint decision

- Canonical repository-root launcher: `run.py`
- Preserved equivalent launcher: `main.py`
- Preserved installed console entrypoint: `tensor-crypt -> tensor_crypt.app.launch:main`

### Canonical config decision

- Canonical config module: `tensor_crypt/runtime_config.py`
- Root `config.py` is now a thin compatibility wrapper that re-exports the canonical config surface
- This also fixed a live split-truth problem where root `config.py` and packaged `runtime_config.py` had diverged

## Compatibility Strategy

The following compatibility surfaces remain intentionally:

- `config`
- `main`
- `run`
- `engine.*`
- `viewer.*`

They remain because tests and real public usage depend on them. Each retained surface is thin and contains no simulation or training logic.

## Moves / Removals

### Moved

- `src/tensor_crypt/**` -> `tensor_crypt/**`
- `src/engine/*.py` -> `engine/*.py`
- `src/viewer/*.py` -> `viewer/*.py`

### Removed

- `src/` implementation duplication
- root/package path-extension bridge behavior
- `src/config.py`, `src/main.py`, `src/run.py`
- `src/tensor_crypt.egg-info`

### Updated

- `pyproject.toml` now discovers packages directly from repository root
- `README.md` and architecture docs now describe the flat canonical package layout
- `tensor_crypt/checkpointing/runtime_checkpoint.py` now tolerates minimal runtime shims without a `physics` attribute during restore; this fixed a validation-only assumption without changing normal runtime behavior

## Generated Artifact Hygiene Strategy

- Runtime and test artifacts remain git-ignored
- New repository docs point to `docs/` as the source/report boundary
- Validation had to avoid legacy locked workspace temp trees by running pytest from user-temp basetemp/cache locations

Note:

- Several old workspace temp/artifact directories were already ACL-locked before this refactor and could not be fully removed from inside the environment
- They are not part of the tracked source tree, and validation was redirected away from them

## Verification Summary

Completed validation:

- `python -m compileall tensor_crypt engine viewer config.py main.py run.py scripts tests`
- import smoke for `config`, `main`, `run`, `tensor_crypt.*`, `engine.*`, and `viewer.*`
- focused pytest smoke:
  - `tests/test_imports_and_compat.py`
  - `tests/test_viewer_interactions.py::test_camera_fit_clamp_and_zoom_bounds`
  - `tests/test_ppo_reward_surface.py::test_validate_runtime_config_rejects_invalid_reward_surface`
- runtime/checkpoint slice under user-temp basetemp:
  - `tests/test_binary_reproduction.py::test_binary_birth_assigns_distinct_parent_roles`
  - `tests/test_checkpoint_atomicity.py::test_atomic_checkpoint_emits_manifest_and_latest_pointer`
  - `tests/test_engine_and_viewer_integration.py::test_seeded_runtime_is_deterministic`
- full pytest suite under user-temp basetemp/cache:
  - `128 passed`
- headless benchmark harness:
  - `python scripts/benchmark_runtime.py --device cpu --ticks 2 --warmup-ticks 0 --width 12 --height 12 --agents 6 --walls 0 --hzones 1 --update-every 8 --batch-size 4 --mini-batches 1 --epochs 1 --log-dir benchmark_smoke_logs --output benchmark_smoke.json`
- public launcher smoke through `run.py` with dummy SDL and `MAX_TICKS = 1`

Attempted but environment-blocked:

- wheel build smoke via `pip wheel . --no-deps --no-build-isolation`
- failure cause was host temp/build tracker permission denial, not a Python import or packaging metadata error inside the repo

## Final Tree Summary

```text
.
├── config.py
├── run.py
├── main.py
├── tensor_crypt/
├── engine/
├── viewer/
├── scripts/
├── docs/
├── tests/
├── pyproject.toml
├── README.md
└── STRUCTURE_REFACTOR_REPORT.md
```
