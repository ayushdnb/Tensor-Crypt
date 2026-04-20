# AGENTS.md

## Repository Contract

Tensor Crypt Single-Brain Vmap is the self-centric single-family release line. Real implementation code belongs under `tensor_crypt/`; root files such as `run.py`, `main.py`, and `config.py` are public compatibility surfaces only.

## Release Identity

- Public startup applies the self-centric single-family preset.
- The active branch family is `BRAIN.EXPERIMENTAL_BRANCH_FAMILY`.
- Experimental observation tensors are the branch brain input contract.
- Family-shift mutation stays disabled for one-family ownership semantics.
- Vmap inference is enabled by public startup defaults and remains guarded by `torch.func` availability.

## Change Rules

- Keep wrappers thin and free of simulation logic.
- Preserve UID ownership, engine tick order, checkpoint capture/restore order, and telemetry artifact names unless a task explicitly changes those contracts.
- If config support is widened, update runtime validation, tests, README, and docs together.
- Do not treat generated logs, validation runs, or staged worktrees as implementation truth.

## Validation Focus

For launch/config changes, run `python -m pytest tests\test_imports_and_compat.py tests\test_experimental_family_vmap_inference.py`.
For brain, observation, PPO, checkpoint, or lifecycle changes, include the matching subsystem tests under `tests/`.
