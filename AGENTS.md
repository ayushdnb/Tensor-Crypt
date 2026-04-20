# AGENTS.md

## Repository Contract

Tensor Crypt Five-Brain is the canonical multi-family release line. Real implementation code belongs under `tensor_crypt/`; root files such as `run.py`, `main.py`, and `config.py` are public compatibility surfaces only.

## Release Identity

- Default startup uses the five configured bloodline families.
- Canonical observations are the supported brain input contract.
- UID ownership, lineage, PPO buffers, checkpoints, and telemetry remain UID-scoped rather than slot-scoped.
- The optional family-vmap path is benchmark-only and disabled by default.

## Change Rules

- Keep wrappers thin and free of simulation logic.
- Preserve engine tick order, checkpoint capture/restore order, and telemetry artifact names unless the task explicitly changes those contracts.
- If config support is widened, update runtime validation, tests, README, and docs together.
- Do not treat generated logs, validation runs, or staged worktrees as implementation truth.

## Validation Focus

For launch/config changes, run `python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py`.
For brain, observation, PPO, checkpoint, or lifecycle changes, include the matching subsystem tests under `tests/`.
