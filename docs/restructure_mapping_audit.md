# Restructure Mapping Audit

## Purpose
Map every existing tracked module/file/folder responsibility to its target location and migration treatment, proving no responsibility is dropped.

## Execution Status
- Structural migration executed.
- Tables below reflect final old->new mapping outcomes.

## Status Legend
- `keep`: retained in place
- `move`: relocated without semantic changes
- `wrap`: retained as compatibility shim
- `update-path`: same responsibility, code edited only for path/import safety
- `doc`: documentation-only move/update

## Root-Level Inventory

| Old Path | Current Responsibility | Migration Treatment | Target Path |
|---|---|---|---|
| `.gitignore` | Ignore generated/runtime artifacts | keep | `.gitignore` |
| `README.md` | User-facing project overview and run instructions | doc | `README.md` |
| `ARCHITECTURE.md` | Legacy architecture notes | doc | `docs/history/ARCHITECTURE.md` |
| `AUDIT_REPORT.md` | Prior audit report | doc | `docs/history/AUDIT_REPORT.md` |
| `CLEANUP_AUDIT_REPORT.md` | Prior cleanup audit report | doc | `docs/history/CLEANUP_AUDIT_REPORT.md` |
| `config.py` | Canonical configuration surface (`cfg`) | keep | `config.py` |
| `run.py` | Public root launcher | keep | `run.py` |
| `main.py` | Alternate public root launcher | keep | `main.py` |
| `requirements.txt` | Runtime dependency pin list | keep | `requirements.txt` |
| `pytest.ini` | pytest defaults | keep | `pytest.ini` |
| `dump_py_to_text.py` | Utility script for source dump | keep | `dump_py_to_text.py` |

## Compatibility Packages (Root)

| Old Path | Current Responsibility | Migration Treatment | Target Path |
|---|---|---|---|
| `engine/__init__.py` | Legacy engine package surface | wrap | `engine/__init__.py` |
| `engine/brain.py` | Re-export brain | wrap | `engine/brain.py` |
| `engine/evolution.py` | Re-export evolution | wrap | `engine/evolution.py` |
| `engine/grid.py` | Re-export grid | wrap | `engine/grid.py` |
| `engine/logger.py` | Re-export data logger | wrap | `engine/logger.py` |
| `engine/mapgen.py` | Re-export map generation helpers | wrap | `engine/mapgen.py` |
| `engine/perception.py` | Re-export perception | wrap | `engine/perception.py` |
| `engine/physics.py` | Re-export physics | wrap | `engine/physics.py` |
| `engine/ppo.py` | Re-export PPO | wrap | `engine/ppo.py` |
| `engine/registry.py` | Re-export registry | wrap | `engine/registry.py` |
| `engine/respawn.py` | Re-export respawn controller | wrap | `engine/respawn.py` |
| `engine/simulation.py` | Re-export simulation engine | wrap | `engine/simulation.py` |
| `viewer/__init__.py` | Legacy viewer package surface | wrap | `viewer/__init__.py` |
| `viewer/camera.py` | Re-export camera | wrap | `viewer/camera.py` |
| `viewer/colors.py` | Re-export colors | wrap | `viewer/colors.py` |
| `viewer/input.py` | Re-export input handler | wrap | `viewer/input.py` |
| `viewer/layout.py` | Re-export layout manager | wrap | `viewer/layout.py` |
| `viewer/main.py` | Re-export viewer runtime | wrap | `viewer/main.py` |
| `viewer/panels.py` | Re-export panel/render classes | wrap | `viewer/panels.py` |
| `viewer/text_cache.py` | Re-export text cache | wrap | `viewer/text_cache.py` |

## Implementation Tree (`tensor_crypt/*`)

All implementation modules below move to `src/tensor_crypt/*` with identical module names and responsibilities.

| Old Path | Current Responsibility | Migration Treatment | Target Path |
|---|---|---|---|
| `tensor_crypt/__init__.py` | Implementation package root marker/docs | move + wrap | `src/tensor_crypt/__init__.py` (+ new root shim at `tensor_crypt/__init__.py`) |
| `tensor_crypt/config_bridge.py` | Root config bridge and fallback loader | move + update-path | `src/tensor_crypt/config_bridge.py` |
| `tensor_crypt/app/__init__.py` | App package marker | move | `src/tensor_crypt/app/__init__.py` |
| `tensor_crypt/app/launch.py` | Launch entry logic | move | `src/tensor_crypt/app/launch.py` |
| `tensor_crypt/app/runtime.py` | Runtime graph assembly + determinism setup | move | `src/tensor_crypt/app/runtime.py` |
| `tensor_crypt/agents/__init__.py` | Agents package marker | move | `src/tensor_crypt/agents/__init__.py` |
| `tensor_crypt/agents/brain.py` | Brain architecture + family contracts | move | `src/tensor_crypt/agents/brain.py` |
| `tensor_crypt/agents/state_registry.py` | UID-slot lifecycle registry | move | `src/tensor_crypt/agents/state_registry.py` |
| `tensor_crypt/audit/__init__.py` | Audit helper surface | move | `src/tensor_crypt/audit/__init__.py` |
| `tensor_crypt/audit/final_validation.py` | Determinism/resume/save-load probes | move | `src/tensor_crypt/audit/final_validation.py` |
| `tensor_crypt/checkpointing/__init__.py` | Checkpoint public API | move | `src/tensor_crypt/checkpointing/__init__.py` |
| `tensor_crypt/checkpointing/atomic_checkpoint.py` | Manifest/checksum atomic save/load helpers | move | `src/tensor_crypt/checkpointing/atomic_checkpoint.py` |
| `tensor_crypt/checkpointing/runtime_checkpoint.py` | Runtime capture/validate/restore checkpoint logic | move | `src/tensor_crypt/checkpointing/runtime_checkpoint.py` |
| `tensor_crypt/learning/__init__.py` | Learning package marker | move | `src/tensor_crypt/learning/__init__.py` |
| `tensor_crypt/learning/ppo.py` | PPO buffers, training, optimizer ownership | move | `src/tensor_crypt/learning/ppo.py` |
| `tensor_crypt/population/__init__.py` | Population package marker | move | `src/tensor_crypt/population/__init__.py` |
| `tensor_crypt/population/evolution.py` | Death processing + policy-noise helpers | move | `src/tensor_crypt/population/evolution.py` |
| `tensor_crypt/population/reproduction.py` | Binary reproduction + trait mutation/placement | move | `src/tensor_crypt/population/reproduction.py` |
| `tensor_crypt/population/respawn_controller.py` | Respawn/extinction controller | move | `src/tensor_crypt/population/respawn_controller.py` |
| `tensor_crypt/simulation/__init__.py` | Simulation package marker | move | `src/tensor_crypt/simulation/__init__.py` |
| `tensor_crypt/simulation/catastrophes.py` | Catastrophe scheduler/runtime modifiers | move | `src/tensor_crypt/simulation/catastrophes.py` |
| `tensor_crypt/simulation/engine.py` | Tick orchestration and subsystem sequencing | move | `src/tensor_crypt/simulation/engine.py` |
| `tensor_crypt/telemetry/__init__.py` | Telemetry package marker | move | `src/tensor_crypt/telemetry/__init__.py` |
| `tensor_crypt/telemetry/data_logger.py` | HDF5/parquet run logging | move | `src/tensor_crypt/telemetry/data_logger.py` |
| `tensor_crypt/telemetry/lineage_export.py` | Lineage graph export | move | `src/tensor_crypt/telemetry/lineage_export.py` |
| `tensor_crypt/telemetry/run_paths.py` | Run directory and run metadata creation | move | `src/tensor_crypt/telemetry/run_paths.py` |
| `tensor_crypt/viewer/__init__.py` | Viewer package marker | move | `src/tensor_crypt/viewer/__init__.py` |
| `tensor_crypt/viewer/camera.py` | Viewer camera transforms | move | `src/tensor_crypt/viewer/camera.py` |
| `tensor_crypt/viewer/colors.py` | Viewer colors + bloodline color helpers | move | `src/tensor_crypt/viewer/colors.py` |
| `tensor_crypt/viewer/input.py` | Viewer event/hotkey routing | move | `src/tensor_crypt/viewer/input.py` |
| `tensor_crypt/viewer/layout.py` | Viewer panel geometry layout | move | `src/tensor_crypt/viewer/layout.py` |
| `tensor_crypt/viewer/main.py` | Pygame viewer loop | move | `src/tensor_crypt/viewer/main.py` |
| `tensor_crypt/viewer/panels.py` | World/HUD/side panel drawing | move | `src/tensor_crypt/viewer/panels.py` |
| `tensor_crypt/viewer/text_cache.py` | Cached text rendering | move | `src/tensor_crypt/viewer/text_cache.py` |
| `tensor_crypt/world/__init__.py` | World package marker | move | `src/tensor_crypt/world/__init__.py` |
| `tensor_crypt/world/observation_schema.py` | Canonical/legacy observation adapters | move | `src/tensor_crypt/world/observation_schema.py` |
| `tensor_crypt/world/perception.py` | Raycasting and observation construction | move | `src/tensor_crypt/world/perception.py` |
| `tensor_crypt/world/physics.py` | Movement/collision/environment/death processing | move | `src/tensor_crypt/world/physics.py` |
| `tensor_crypt/world/procedural_map.py` | Wall and h-zone generation | move | `src/tensor_crypt/world/procedural_map.py` |
| `tensor_crypt/world/spatial_grid.py` | Tensor-backed grid substrate | move | `src/tensor_crypt/world/spatial_grid.py` |

## Scripts and Tests

| Old Path | Current Responsibility | Migration Treatment | Target Path |
|---|---|---|---|
| `scripts/run_soak_audit.py` | Headless soak audit runner | keep | `scripts/run_soak_audit.py` |
| `tests/conftest.py` | Test fixtures/runtime builder | keep (path update only if required) | `tests/conftest.py` |
| `tests/test_imports_and_compat.py` | Public/compat import contracts | keep | `tests/test_imports_and_compat.py` |
| `tests/test_engine_and_viewer_integration.py` | End-to-end runtime/viewer smoke | keep | `tests/test_engine_and_viewer_integration.py` |
| `tests/test_logging_and_run_paths.py` | Telemetry/run-path contracts | keep | `tests/test_logging_and_run_paths.py` |
| `tests/test_grid_and_mapgen.py` | Grid/mapgen contracts | keep | `tests/test_grid_and_mapgen.py` |
| `tests/test_physics.py` | Physics contracts | keep | `tests/test_physics.py` |
| `tests/test_perception.py` | Perception/observation contracts | keep | `tests/test_perception.py` |
| `tests/test_ppo.py` | PPO math/training contracts | keep | `tests/test_ppo.py` |
| `tests/test_registry_respawn.py` | Registry/respawn contracts | keep | `tests/test_registry_respawn.py` |
| `tests/test_bloodline_brains.py` | Bloodline brain contracts | keep | `tests/test_bloodline_brains.py` |
| `tests/test_identity_substrate.py` | UID substrate contracts | keep | `tests/test_identity_substrate.py` |
| `tests/test_runtime_checkpoint_substrate.py` | Checkpoint substrate validation | keep | `tests/test_runtime_checkpoint_substrate.py` |
| `tests/test_prompt4_uid_ppo_hardening.py` | Prompt4 hardening contracts | keep | `tests/test_prompt4_uid_ppo_hardening.py` |
| `tests/test_prompt5_binary_reproduction.py` | Prompt5 reproduction contracts | keep | `tests/test_prompt5_binary_reproduction.py` |
| `tests/test_prompt6_catastrophes.py` | Prompt6 catastrophe contracts | keep | `tests/test_prompt6_catastrophes.py` |
| `tests/test_prompt7_checkpoint_atomicity.py` | Prompt7 checkpoint atomicity contracts | keep | `tests/test_prompt7_checkpoint_atomicity.py` |
| `tests/test_prompt7_telemetry.py` | Prompt7 telemetry contracts | keep | `tests/test_prompt7_telemetry.py` |
| `tests/test_prompt7_audit.py` | Prompt7 validation harness contracts | keep | `tests/test_prompt7_audit.py` |

## New Paths Introduced by Migration

| New Path | Responsibility |
|---|---|
| `pyproject.toml` | Packaging metadata and `src` layout configuration |
| `tensor_crypt/__init__.py` (root shim) | Preserve `tensor_crypt.*` import surface from repo root while implementation moves to `src/tensor_crypt` |
| `docs/architecture_overview.md` | Final architecture and module boundaries |
| `docs/restructure_execution_plan.md` | Migration planning and risk controls |
| `docs/restructure_mapping_audit.md` | Responsibility-preservation mapping |
| `docs/restructure_validation_report.md` | Validation evidence and final verdict |
| `.agent/AGENTS.md` | Agent operating constraints |
| `.agent/PLANS.md` | Plan ledger and status |

## Responsibility Preservation Check
- Runtime assembly remains in `app/runtime.py` (moved, not rewritten).
- Tick orchestration remains in `simulation/engine.py` (moved, not rewritten).
- PPO core remains in `learning/ppo.py` (moved, not rewritten).
- Checkpoint logic remains in `checkpointing/*` (moved, not rewritten).
- Telemetry logic remains in `telemetry/*` (moved, not rewritten).
- Viewer logic remains in `viewer/*` (moved, not rewritten).
- Compatibility import surfaces remain via root wrappers and root namespace shim.

No subsystem responsibility is intentionally dropped in the target map.
