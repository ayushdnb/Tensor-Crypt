# Findings And Fixes

## Finding 1
- Surface: checkpoint latest-pointer resolution
- Evidence: manual reproduction of `load_runtime_checkpoint(checkpoint_dir)` raised `FileNotFoundError` because the resolver doubled the relative run-directory prefix when interpreting `latest_checkpoint.json`.
- Fix: `tensor_crypt/checkpointing/atomic_checkpoint.py`
- Change: resolver now honors already-existing relative paths before rebasing; newly written latest pointers store checkpoint and manifest paths relative to the pointer directory.
- Verification: full `tests/test_checkpoint_atomicity.py`, `tests/test_benchmark_and_scheduler.py`, and the full repo suite all passed.

## Finding 2
- Surface: repo pytest temp-path infrastructure on Windows/Python 3.14
- Evidence: runtime-backed tests errored before meaningful execution because pytest's forced `--basetemp=.pytest_tmp_run` and built-in tempdir/cache behavior produced `PermissionError` failures in temp-path setup and cleanup.
- Fix: `pyproject.toml`, `tests/conftest.py`
- Change: removed forced `--basetemp`, disabled cacheprovider for repo tests, and replaced built-in `tmp_path` usage with a repo-owned workspace fixture that creates ordinary directories via `Path.mkdir()`.
- Verification: full suite passed from a clean `python -m pytest -q` invocation with `180` passing tests.

## Finding 3
- Surface: long-run checkpoint/resume confidence gap
- Evidence: the repo had single-cycle resume coverage but no reusable repeated resume-chain harness.
- Fix: `tensor_crypt/audit/final_validation.py`, `scripts/run_resume_chain_audit.py`, associated tests
- Change: added `run_resume_chain_probe()` and a scriptable harness that performs repeated save/load/restore cycles and compares each restored state to a fresh same-seed reference at the same total tick count.
- Verification: `scripts/run_resume_chain_audit.py` passed for `5` cycles x `8` ticks with signature matches at every cycle, including after a respawned UID appeared.
