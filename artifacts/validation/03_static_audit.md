# Static Audit

## Initial Observations
- Canonical runtime assembly and validation live in `tensor_crypt/app/runtime.py`.
- Launch defaults for repo-root entrypoints are modified by `apply_experimental_single_family_launch_defaults()` in `tensor_crypt/runtime_config.py`, then invoked from launch surfaces.
- Existing test coverage already targets many contracts, but gaps are still being measured against real execution.
- The worktree contains pre-existing local edits across several canonical modules; any defects must be separated from this audit's changes.

## Findings
- `tensor_crypt.checkpointing.atomic_checkpoint.resolve_latest_checkpoint_bundle()` mis-resolved relative checkpoint paths by rebasing already-valid relative paths onto the pointer directory, duplicating parent segments and breaking directory-based checkpoint loads.
- The repo-level pytest config forced `--basetemp=.pytest_tmp_run`, and on Windows/Python 3.14 the resulting tempdir/cache strategy produced `PermissionError` failures before meaningful runtime tests could execute.
- The root launch surface had import coverage but no executable verification that it actually applies the documented experimental single-family preset before runtime construction.

## Static Audit Targets Covered
- Runtime assembly/launch preset behavior: covered by new launch-entrypoint test.
- Checkpoint manifest/latest-pointer failure paths: extended with checksum-mismatch and temp-file-cleanup cases.
- Resume-chain restore stability: extended with repeated save/load/restore verification.
- Test harness portability: repaired via repo-owned `tmp_path` fixture and pytest config hardening.
