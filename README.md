# Tensor Crypt

Tensor Crypt is a PyTorch-based grid simulation project that combines per-agent neural policies, procedural world generation, PPO training, evolutionary respawn, persistent telemetry, and a Pygame viewer. The repository keeps the public control surface at the root while the implementation lives under the `tensor_crypt/` package.

## Key Capabilities

- Grid-based multi-agent simulation backed by PyTorch tensors.
- Procedural wall and healing-zone generation.
- Per-agent brain modules with PPO buffering and optimization.
- Canonical UID lifecycle tracking for active and historical agents.
- Persistent run artifacts written as HDF5 snapshots and Parquet event streams.
- Runtime checkpoint capture utilities under `tensor_crypt/checkpointing`.
- Interactive Pygame viewer plus a headless soak-audit script for longer invariant checks.

## Repository Structure

- `config.py`: primary configuration surface exposed as nested dataclasses through `cfg`.
- `run.py` and `main.py`: thin root entrypoints that launch the application.
- `tensor_crypt/app`: launch-time setup, determinism, and runtime assembly.
- `tensor_crypt/simulation`: tick orchestration and subsystem sequencing.
- `tensor_crypt/world`: spatial grid, procedural map generation, perception, and physics.
- `tensor_crypt/agents`: brain model and slot-based state registry.
- `tensor_crypt/learning`: PPO storage and optimization logic.
- `tensor_crypt/population`: evolutionary mutation helpers and respawn control.
- `tensor_crypt/telemetry`: run-directory management and persistent logging.
- `tensor_crypt/viewer`: rendering, layout, input handling, and UI panels.
- `tensor_crypt/checkpointing`: runtime checkpoint capture and validation helpers.
- `engine/` and `viewer/`: compatibility facades that re-export package modules.
- `tests/`: pytest-based coverage for runtime assembly, logging, physics, perception, PPO, and identity/checkpointing behavior.
- `scripts/run_soak_audit.py`: headless audit runner for deterministic soak checks.
- `ARCHITECTURE.md`: concise architectural notes for the current package layout.

## Requirements

- Python 3.10 or newer.
- A working PyTorch installation.
- A graphical environment for the interactive viewer when running `run.py` or `main.py`.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

If you need a CUDA-specific PyTorch build, install the matching wheel for your platform before or instead of the default `torch` dependency in `requirements.txt`.

## Running The Project

Launch the interactive simulation from the repository root:

```powershell
python run.py
```

`main.py` is an equivalent alternate entrypoint:

```powershell
python main.py
```

On startup the application:

- applies deterministic seeds from `cfg.SIM.SEED`
- creates a new run directory under `cfg.LOG.DIR`
- writes a config snapshot and run metadata
- builds the runtime graph
- opens the Pygame viewer

By default run artifacts are written under `logs/run_YYYYMMDD_HHMMSS/`.

## Configuration

The main configuration surface is the root-level `config.py` module. The application imports the shared `cfg` object through `tensor_crypt.config_bridge`, so the repository should be run from source rather than treated as an installed package.

Common configuration entry points include:

- `cfg.SIM`: seed, device, tick rate, and runtime limits.
- `cfg.GRID` and `cfg.MAPGEN`: world size, overlap handling, walls, and healing zones.
- `cfg.AGENTS`, `cfg.TRAITS`, and `cfg.RESPAWN`: population size, trait initialization, and respawn policy.
- `cfg.PPO` and `cfg.EVOL`: learning cadence and evolutionary mutation parameters.
- `cfg.VIEW`: viewer window and rendering settings.
- `cfg.LOG`: artifact directory, snapshot cadence, and assertions.

## Testing And Audit Utilities

The repository includes a pytest suite:

```powershell
pip install pytest
pytest -q
```

Tests set dummy SDL drivers automatically so the viewer can be exercised in headless environments.

For a longer headless audit run:

```powershell
python scripts/run_soak_audit.py --ticks 256
```

That script writes its output under `audit_tmp/` by default.

## Notes

- Root-level `engine/` and `viewer/` modules are compatibility wrappers over `tensor_crypt/`.
- Generated runtime outputs such as `logs/`, `audit_tmp/`, and `.pytest_tmp/` are local artifacts and are intentionally not part of source control.
- Additional implementation notes are available in `ARCHITECTURE.md`.
