# Restructure Plan Ledger

## Active Plan
1. Discovery and inventory (completed)
2. Planning and mapping documentation (completed)
3. Structural migration to target hierarchy (completed)
4. Documentation alignment (completed)
5. Validation and parity audit (completed)
6. Final report and commit grouping (in progress)

## Discovery Snapshot
- Current runtime implementation lives in `tensor_crypt/*`.
- Legacy compatibility modules exist at root in `engine/*` and `viewer/*`.
- Root public surfaces are `config.py`, `run.py`, `main.py`.
- No packaging metadata file (`pyproject.toml`) exists yet.
- Test suite heavily exercises imports, compatibility layers, runtime wiring, checkpointing, telemetry, and viewer behavior.

## Planned Target
- Consolidate implementation under `src/tensor_crypt/*`.
- Keep root launch/config surfaces.
- Keep root `engine/*` and `viewer/*` compatibility wrappers.
- Add a root `tensor_crypt` namespace shim to preserve `tensor_crypt.*` imports from repository-root execution context.
- Add packaging metadata (`pyproject.toml`) and keep `requirements.txt`.
