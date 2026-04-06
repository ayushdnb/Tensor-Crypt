# Cleanup Audit Report

## 1. Executive Summary

This pass treated the project as behavior-frozen and focused only on hygiene work that could be justified as semantics-preserving.

High-confidence cleanup completed:
- removed generated caches, temporary test/runtime artifacts, and one stale generated dump output
- removed a small amount of clearly dead internal code and unused imports
- made compatibility shims explicit rather than looking like accidental unused imports
- normalized the root config file's formatting/comments without changing any knob names, values, or meanings
- normalized text/source encoding from UTF-8 with BOM to plain UTF-8 for consistency and tool compatibility

Behavior-sensitive subsystems were left intact: simulation sequencing, physics, perception, PPO update semantics, respawn behavior, config meanings, and root launch behavior were not changed.

Validation status:
- `ruff check . --exclude __pycache__,logs,audit_tmp,.pytest_cache,.pytest_tmp` passed
- `pytest -q` passed: `28 passed`
- `python scripts/run_soak_audit.py --ticks 64` passed

## 2. Repo Map

Repository layout after audit:
- Root public surfaces: `config.py`, `run.py`, `main.py`
- Architecture/history docs: `ARCHITECTURE.md`, `AUDIT_REPORT.md`
- Internal implementation package: `tensor_crypt/`
- Compatibility facades: `engine/`, `viewer/`
- Test suite: `tests/`
- Maintained utility script: `scripts/run_soak_audit.py`
- Suspicious-but-retained ad hoc utility: `dump_py_to_text.py`

Internal package map:
- `tensor_crypt.app`: bootstrap and runtime assembly
- `tensor_crypt.simulation`: tick orchestration
- `tensor_crypt.world`: grid, map generation, perception, physics
- `tensor_crypt.agents`: registry and brain modules
- `tensor_crypt.learning`: PPO buffering/update logic
- `tensor_crypt.population`: evolution helpers and respawn control
- `tensor_crypt.telemetry`: run directories and artifact persistence
- `tensor_crypt.viewer`: pygame viewer/runtime rendering

Behavior-sensitive areas identified during recon:
- `tensor_crypt/simulation/engine.py`
- `tensor_crypt/world/physics.py`
- `tensor_crypt/world/perception.py`
- `tensor_crypt/learning/ppo.py`
- `tensor_crypt/population/respawn_controller.py`
- `tensor_crypt/agents/state_registry.py`
- `tensor_crypt/app/runtime.py`
- `tensor_crypt/telemetry/data_logger.py`
- `tensor_crypt/telemetry/run_paths.py`

## 3. Public / Compatibility Surfaces Identified

Primary public surfaces retained unchanged:
- `config.py`: authoritative root configuration surface
- `run.py`: root launch entrypoint
- `main.py`: alternate root launch entrypoint
- `tensor_crypt/config_bridge.py`: controlled bridge back to root `config.py`

Compatibility contracts retained unchanged:
- `engine.*` import paths remain valid
- `viewer.*` import paths remain valid
- package layout under `tensor_crypt.*` remains unchanged
- tests in `tests/test_imports_and_compat.py` continue to exercise these contracts

Additional contractual surfaces treated conservatively:
- persisted artifact names under `tensor_crypt.telemetry`
- registry column layout in `tensor_crypt.agents.state_registry.Registry`
- observation tensor layout in `tensor_crypt.world.perception.Perception`
- viewer selection/inspector behavior in `tensor_crypt.viewer.*`

## 4. Dead Code Findings

Removed dead code:
- `tensor_crypt/population/evolution.py`: removed private method `_select_parent`
  - Why it was considered dead: there were no call sites in the repository, including tests and runtime assembly.
  - How verified: `rg -n "_select_parent\("` returned only the method definition.
  - Risk assessment: low. It was private, unused, and not part of a compatibility surface.
- `tensor_crypt/learning/ppo.py`: removed unused attribute `self.tick_count`
  - Why it was considered dead: written once in `__init__` and never read.
  - How verified: `rg -n "tick_count\b"` returned only the assignment.
  - Risk assessment: low. Internal attribute with no behavioral effect.
- `tests/conftest.py`: removed unused helper `create_logger_run_dir`
  - Why it was considered dead: no tests imported or called it; the logging test file uses its own local helper.
  - How verified: `rg -n "create_logger_run_dir\("` showed only the definition.
  - Risk assessment: none for runtime behavior; test-only dead helper.
- `tensor_crypt/viewer/camera.py`: removed unused `pygame` import.
  - Why it was considered dead: the module does not reference `pygame`.
  - How verified: `ruff check` flagged the import and no symbol references exist in the file.
  - Risk assessment: none.
- `tests/test_registry_respawn.py`: removed unused `torch` import.
  - Why it was considered dead: the module does not reference `torch`.
  - How verified: `ruff check` flagged the import and no symbol references exist in the file.
  - Risk assessment: none.

Suspicious items retained:
- `dump_py_to_text.py`
  - Why it looked removable: one-off utility shape; no in-repo references.
  - Why kept: it is a user-invokable script, not an imported module, and there was not enough evidence that it is non-contractual.
- `AUDIT_REPORT.md`
  - Why it looked removable: appears to be a previous generated audit document rather than core project documentation.
  - Why kept: could still be useful historical context; deleting it would be policy rather than hygiene.
- unused-looking config knobs in `config.py` such as `SPAWN_MODE`, `MOVE_FAIL_COST`, `TIE_BREAKER`, `RAY_FIELD_AGG`, `RAY_STEP_SAMPLER`, `REWARD_FORM`, `SELECTION`, `METAB_FORM`
  - Why they looked removable: several are not wired to current runtime logic.
  - Why kept: they are part of the public root config surface and may be relied on by users, scripts, or future compatibility expectations.
- `Engine.last_obs_dict` / `Engine.last_dones_dict`
  - Why they looked removable: current code writes them but does not read them.
  - Why kept: they sit in the most behavior-sensitive module and are public-ish attributes on a runtime object; cleanup value did not justify the contract risk.

## 5. Dead File Findings

Deleted generated/cache/artifact paths:
- `.pytest_cache`
  - Why considered dead: pytest cache directory.
  - How verified: generated by test tooling; not imported or documented as project data.
  - Risk assessment: none.
- `.pytest_tmp`
  - Why considered dead: pytest temporary output directory created by tests.
  - How verified: directory contents were test-generated run artifacts.
  - Risk assessment: none.
- `__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `engine/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/agents/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/app/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/learning/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/population/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/simulation/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/telemetry/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/viewer/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tensor_crypt/world/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `tests/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `viewer/__pycache__`
  - Why considered dead: Python bytecode cache directory.
  - How verified: standard interpreter cache naming and `.pyc` contents.
  - Risk assessment: none.
- `.ruff_cache`
  - Why considered dead: ruff cache directory created during validation.
  - How verified: standard linter cache naming.
  - Risk assessment: none.
- `logs`
  - Why considered dead: generated runtime artifact tree containing run directories, HDF5, and parquet outputs.
  - How verified: file names and contents matched `tensor_crypt.telemetry` outputs and were recreated by validation.
  - Risk assessment: low; generated outputs only.
- `audit_tmp`
  - Why considered dead: temporary audit/soak output tree.
  - How verified: contents were soak/log artifacts created by tests and soak tooling.
  - Risk assessment: low; generated outputs only.
- `codes/evolution.txt`
  - Why considered dead: stale generated concatenated source dump.
  - How verified: contents mirrored repository source and were produced by `dump_py_to_text.py`; no in-repo consumers.
  - Risk assessment: low.
- `codes`
  - Why considered dead: empty directory left after removing the generated dump file.
  - How verified: directory was empty and recreatable by `dump_py_to_text.py` if needed.
  - Risk assessment: none.

## 6. Consistency Findings

Consistency changes made:
- File encoding normalization
  - Old: text/source files stored as UTF-8 with BOM.
  - New: text/source files stored as plain UTF-8.
  - Why safe: file contents were unchanged; only the BOM marker was removed.
  - Why worthwhile: tooling such as AST parsing and linting no longer trips over BOM-prefixed source files.
- Compatibility shim intent
  - Old: shim modules in `engine/*` and `viewer/*` were bare imports that looked like unused code.
  - New: each shim now has a short module docstring and explicit `__all__` re-export list.
  - Why safe: import targets did not change.
  - Why worthwhile: compatibility purpose is now explicit and lint-clean.
- Root config readability
  - Old: `config.py` had inconsistent spacing and stale section markers such as `NEW SECTION` comments.
  - New: imports, spacing, inline comments, and module/class docstrings were normalized while preserving every field name and value.
  - Why safe: comments/formatting only.
  - Why worthwhile: the root public knob surface is materially easier to read and audit.

Renames performed:
- None. No files, classes, functions, config fields, or public symbols were renamed.

## 7. Comment / Docstring Improvements

Improvements made:
- `config.py`
  - Added a module docstring clarifying that it remains the authoritative public config surface.
  - Removed noisy section-marker comments and replaced them with cleaner structure.
- compatibility shims under `engine/*` and `viewer/*`
  - Added concise module docstrings documenting that they are deliberate legacy re-export surfaces.
- `tests/conftest.py`
  - Added a short module docstring so the purpose of the shared fixture layer is explicit.
- `scripts/run_soak_audit.py`
  - Added a short module docstring so the file reads as maintained audit tooling rather than an anonymous script.

## 8. Changes Made

Code changes:
- `config.py`: formatting/comment cleanup only; no config meaning changes.
- `tensor_crypt/population/evolution.py`: removed dead private helper `_select_parent`.
- `tensor_crypt/learning/ppo.py`: removed dead attribute `tick_count`.
- `tensor_crypt/viewer/camera.py`: removed unused import.
- `tests/conftest.py`: removed dead helper, documented intentional import-order exception.
- `tests/test_registry_respawn.py`: removed unused import.
- `scripts/run_soak_audit.py`: documented intentional import-order exception.
- `engine/*.py`, `viewer/*.py`: made compatibility re-exports explicit with module docstrings and `__all__`.
- repo-wide text files: stripped UTF-8 BOM markers.

Non-code cleanup:
- removed caches and generated runtime/test/lint artifacts listed in Section 5
- removed stale generated dump output `codes/evolution.txt`

## 9. Changes Explicitly Avoided

Avoided on purpose:
- no changes to simulation tick order or orchestration behavior
- no changes to physics, perception math, PPO update logic, respawn behavior, or viewer interaction semantics
- no changes to public root entrypoint behavior in `run.py` / `main.py`
- no changes to config field names or meanings
- no removal of compatibility modules under `engine/` or `viewer/`
- no removal of suspicious historical/manual files without stronger proof (`dump_py_to_text.py`, `AUDIT_REPORT.md`)
- no broad internal refactor inside behavior-sensitive files where cleanup value was smaller than contract risk

## 10. Validation Performed

Validation executed after code cleanup:
- `ruff check . --exclude __pycache__,logs,audit_tmp,.pytest_cache,.pytest_tmp`
  - Result: passed
- `pytest -q`
  - Result: `28 passed in 6.11s`
- `python scripts/run_soak_audit.py --ticks 64`
  - Result: completed successfully
  - Observed output summary: `SOAK_OK ticks=64 alive_final=12 min_alive=12 max_alive=12`

Compatibility/import validation specifically covered by tests:
- root imports `config`, `main`, `run`
- legacy import paths such as `engine.physics`, `engine.simulation`, `viewer.main`, `viewer.camera`, `viewer.layout`
- config bridge identity sharing

## 11. Risk Notes

Residual risks and decisions:
- `AUDIT_REPORT.md` and `dump_py_to_text.py` were retained conservatively; they still look non-core, but there was not enough evidence to remove them safely.
- Some config knobs remain unused-looking; they were left untouched because the root config surface is public and potentially user-facing.
- Encoding normalization touched many text/source files, but only at the BOM layer; validation passed afterward.
- Generated artifact deletion was intentionally deferred until after validation so runtime/test verification could still execute normally.

## 12. Final Status

Cleanup pass completed.

Outcome:
- repository tree is materially cleaner
- compatibility surfaces remain intact
- no validated behavioral drift was introduced by the cleanup set
- required audit report written at repository root as `CLEANUP_AUDIT_REPORT.md`
