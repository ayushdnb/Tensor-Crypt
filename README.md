# Tensor Crypt

Tensor Crypt is a PyTorch-based grid simulation with per-agent neural policies, PPO training, evolutionary respawn, persistent telemetry, checkpointing, catastrophe scheduling, and a Pygame viewer.

This repository is structured for clear public surfaces plus a single authoritative implementation tree.

## Public Surfaces
- `config.py`: canonical user configuration (`cfg`).
- `run.py`: primary launch entrypoint.
- `main.py`: alternate launch entrypoint.

## Repository Layout
- `src/tensor_crypt/`: canonical implementation package.
- `tensor_crypt/__init__.py`: root namespace shim so `tensor_crypt.*` imports work from source checkout.
- `engine/` and `viewer/`: legacy compatibility wrappers for old import paths.
- `tests/`: regression and compatibility test suite.
- `scripts/`: operational scripts such as soak audit runner.
- `docs/`: architecture/restructure documentation and history reports.

See `docs/architecture_overview.md` for module-level boundaries and extension guidance.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Running

```powershell
python run.py
```

Equivalent alternate entrypoint:

```powershell
python main.py
```

## Testing

```powershell
pytest -q
```

## Headless Soak Audit

```powershell
python scripts/run_soak_audit.py --ticks 256
```

## Notes
- Runtime outputs are generated under `logs/` by default.
- Checkpointing and telemetry schema/behavior contracts are covered by tests under `tests/`.
- Historical audit documents are preserved under `docs/history/`.
