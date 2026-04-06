# Restructure Validation Report

## Purpose
Verification evidence for the repository-structure migration, focused on semantic preservation.

## Baseline Discovery Evidence (Pre-Migration)

| Command | Result |
|---|---|
| `git status --short --branch` | Confirmed branch/context and pre-existing local modifications in telemetry/test files. |
| `Get-ChildItem -Force` | Captured original root structure and public surfaces. |
| `rg --files` | Enumerated source/test/doc inventory for mapping audit. |
| `git ls-files` | Captured tracked file truth used in old->new mapping. |
| `Get-Content` across runtime modules/tests | Completed discovery of import graph, runtime wiring, compatibility wrappers, and behavior-sensitive modules. |

## Migration Actions Verified
- Moved canonical implementation tree from `tensor_crypt/*` to `src/tensor_crypt/*`.
- Added root `tensor_crypt/__init__.py` namespace shim to preserve `tensor_crypt.*` imports.
- Preserved root compatibility wrappers `engine/*` and `viewer/*`.
- Added `pyproject.toml` with `src` packaging configuration.
- Moved historical reports into `docs/history/`.

## Validation Commands and Results

### A. Import/Path Validation

| Command | Result |
|---|---|
| `python -c "import tensor_crypt; import tensor_crypt.app.runtime; import engine.physics; import viewer.main; print('IMPORT_OK')"` | Passed (`IMPORT_OK`) |
| `python -c "from tensor_crypt.config_bridge import cfg; import config; print(cfg is config.cfg)"` | Passed (`True`) |
| `pytest -q tests/test_imports_and_compat.py` | Passed (`2 passed`) |

### B. Entrypoint Validation

Direct `python run.py` / `python main.py` do not self-terminate due interactive viewer loop, so startup-path validation was executed by patching `Viewer.run` at runtime to return immediately after full startup assembly:

| Command | Result |
|---|---|
| `python -c "from tensor_crypt.viewer.main import Viewer; import run; Viewer.run=lambda self: None; run.main(); print('RUN_ENTRY_OK')"` | Passed (`RUN_ENTRY_OK`) |
| `python -c "from tensor_crypt.viewer.main import Viewer; import main; Viewer.run=lambda self: None; main.main(); print('MAIN_ENTRY_OK')"` | Passed (`MAIN_ENTRY_OK`) |

### C. Simulation Smoke Validation

| Command | Result |
|---|---|
| `python scripts/run_soak_audit.py --ticks 64 --seed 42 --width 20 --height 20 --agents 12 --walls 4 --hzones 2 --log-dir audit_tmp/restructure_soak_logs` | Passed (`SOAK_OK ticks=64 alive_final=12 min_alive=12 max_alive=12`) |

### D/E/F/G/H. Checkpoint, Learning Path, Viewer, Compatibility, Regression Suite

| Command | Result |
|---|---|
| `pytest -q` | Passed (`88 passed in 25.11s`) |
| `pytest -q` (final rerun after docs/finalization commits) | Passed (`88 passed in 20.78s`) |

The full suite includes coverage for:
- checkpoint save/load/manifest/checksum (`test_runtime_checkpoint_substrate.py`, `test_prompt7_checkpoint_atomicity.py`, prompt4/prompt5 checkpoint cases)
- PPO ownership/buffer/update contracts (`test_ppo.py`, `test_prompt4_uid_ppo_hardening.py`)
- viewer integration/hotkeys (`test_engine_and_viewer_integration.py`, `test_prompt6_catastrophes.py`)
- telemetry schemas/ledger behavior (`test_logging_and_run_paths.py`, `test_prompt7_telemetry.py`)
- compatibility import surfaces (`test_imports_and_compat.py`)

## Semantic Preservation Audit
- Intentional behavior changes: none.
- Modules moved without logic changes: implementation tree under `src/tensor_crypt/*`.
- Non-move logic edit risk: none required for passing parity gates.
- Compatibility wrappers retained:
  - root `engine/*`
  - root `viewer/*`
  - root `tensor_crypt/__init__.py` namespace shim for `src` package relocation.
- Drift risks identified:
  - import/path drift from `src` migration
  - entrypoint startup path drift
  - checkpoint/telemetry path contract drift
- Evidence of no observed drift:
  - import checks passed
  - startup path checks passed
  - soak smoke passed
  - full regression suite passed

## Unresolved Issues
- None detected in executed validation scope.

## Final Verdict
- Restructure validated as behavior-preserving within current automated and smoke coverage.
