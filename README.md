# Tensor Crypt Single-Brain Vmap

Tensor Crypt Single-Brain Vmap is the self-centric single-family release line of the Tensor Crypt simulation runtime. It keeps the UID-owned simulation, PPO, telemetry, checkpoint, catastrophe, and viewer contracts from Tensor Crypt while making the single-family self-centric observation path the public startup mode.

This is a simulation and learning project, not a cryptography library. The name refers to dense tensor state and durable runtime records such as identity ledgers, lineage, telemetry, and checkpoints.

## Release Identity

- Public entrypoints apply the self-centric single-family preset before runtime assembly.
- The preset forces root and respawned agents onto `BRAIN.EXPERIMENTAL_BRANCH_FAMILY`.
- Brain inputs use `PERCEPT.OBS_MODE = "experimental_selfcentric_v1"` with experimental observation tensors enabled.
- Family-shift mutation is disabled to preserve one-family ownership semantics.
- `SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE` is enabled by the public startup and benchmark defaults when `torch.func` is available.
- The retained five-family registry surfaces are compatibility scaffolding for telemetry, checkpoints, viewer legends, and existing tests; only the configured branch family is active in normal startup.

The term `experimental` remains in config field names because those fields are checkpoint-visible compatibility surfaces. This branch is a research-preview release line rather than the stable five-brain line.

## Main Capabilities

- Interactive `pygame-ce` viewer with pan, zoom, selection, overlays, and catastrophe controls
- Self-centric observation bundle with experimental ray, self, and context tensors
- Single active brain family using a compact split-input topology
- Optional same-family `torch.func` vmap inference path with loop parity tests
- UID-owned PPO buffers, optimizers, checkpoints, lineage, and telemetry
- Binary-parented reproduction with branch-family enforcement
- Atomic checkpoint publishing with manifests, checksums, and a latest pointer
- Pytest coverage for imports, observation contracts, vmap parity, registry identity, PPO hardening, checkpoints, telemetry, and viewer controls

## Repository Structure

```text
.
|-- config.py                  # Public config compatibility wrapper
|-- run.py                     # Primary launch entrypoint
|-- main.py                    # Alternate launch entrypoint
|-- tensor_crypt/              # Canonical implementation package
|   |-- runtime_config.py      # Canonical config dataclasses and singleton cfg
|   |-- agents/                # Brains and slot-backed registry
|   |-- app/                   # Launch and runtime assembly
|   |-- audit/                 # Determinism and checkpoint probes
|   |-- checkpointing/         # Capture, restore, atomic publish, validation
|   |-- learning/              # PPO
|   |-- population/            # Evolution, reproduction, respawn
|   |-- simulation/            # Engine and catastrophe manager
|   |-- telemetry/             # Run paths, logger, lineage export
|   |-- viewer/                # Pygame viewer
|   `-- world/                 # Grid, map generation, perception, physics
|-- engine/                    # Legacy compatibility imports
|-- viewer/                    # Legacy compatibility imports
|-- scripts/                   # Benchmark and audit harnesses
|-- docs/                      # Architecture and technical documents
`-- tests/                     # Pytest suite
```

`tensor_crypt/` is the only implementation root. `run.py`, `main.py`, `config.py`, `engine/*`, and root `viewer/*` remain thin compatibility surfaces.

## Installation

The viewer backend is pinned to the `pygame-ce` 2.5.x line. The code imports it as `pygame`, because `pygame-ce` provides the `pygame` module namespace. The checked-in manifests require `pygame-ce>=2.5.6,<2.6`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Quick Start

Run the simulation from the repository root:

```bash
python run.py
```

`main.py` and the installed `tensor-crypt` console script use the same canonical package entrypoint.

Startup applies `apply_experimental_single_family_launch_defaults()`, prints the selected device, branch family, population size, grid size, and run directory, then enters the viewer.

## Configuration

The public config surface is `config.py`, which re-exports the canonical dataclasses and singleton `cfg` from `tensor_crypt/runtime_config.py`.

Operator startup uses this branch preset:

- `PERCEPT.OBS_MODE = "experimental_selfcentric_v1"`
- `PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True`
- `BRAIN.EXPERIMENTAL_BRANCH_PRESET = True`
- `BRAIN.EXPERIMENTAL_BRANCH_FAMILY = "House Nocthar"` by default
- `EVOL.ENABLE_FAMILY_SHIFT_MUTATION = False`
- `SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True`
- `LOG.AMP = False` to avoid scaler-only AMP behavior on the preset path

Runtime validation still rejects unsupported or misleading combinations, including missing `torch.func` support when vmap is enabled.

## Viewer Controls

- `Esc`: quit
- `Space`: pause or resume
- `.`: advance one tick while paused
- `+` / `-`: change simulation speed
- `WASD` or arrow keys: pan
- Mouse wheel: zoom at cursor
- Left click: select an agent or zone
- `R`, `B`, `H`, `G`: toggle rays, HP bars, heal-zone overlay, and grid
- `Shift+1`, `Shift+2`, `Shift+3`, `Shift+0`: reproduction doctrine overrides
- `F1` through `F12`: trigger catastrophes manually
- `C`, `Y`, `U`, `I`, `O`: catastrophe controls and panel toggles
- `Alt+Enter`: fullscreen

## Outputs

Each run creates a timestamped directory under `cfg.LOG.DIR` containing:

```text
logs/
`-- run_YYYYMMDD_HHMMSS/
    |-- config.json
    |-- run_metadata.json
    |-- simulation_data.hdf5
    |-- birth_ledger.parquet
    |-- genealogy.parquet
    |-- life_ledger.parquet
    |-- death_ledger.parquet
    |-- collisions.parquet
    |-- ppo_events.parquet
    |-- tick_summary.parquet
    |-- family_summary.parquet
    |-- catastrophes.parquet
    |-- lineage_graph.json
    |-- brains/
    |   `-- brains_tick_<tick>.pt
    `-- checkpoints/
```

Runtime checkpoint manifests and `latest_checkpoint.json` are published only when `ATOMIC_WRITE_ENABLED`, `MANIFEST_ENABLED`, and `SAVE_CHECKPOINT_MANIFEST` are all true.

## Benchmarking

The benchmark harness applies the single-brain-vmap preset by default:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --output benchmark.json
```

To compare against the loop path in the same branch, disable vmap explicitly:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --disable-experimental-family-vmap-inference --output benchmark_loop.json
```

Vmap counters in the benchmark output show whether any slot work actually used that path.

## Testing

Run the suite with:

```bash
python -m pytest
```

Focused checks for this release line:

```powershell
python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py
python -m pytest tests\test_experimental_family_vmap_inference.py tests\test_bloodline_brains.py tests\test_perception.py
python -m pytest tests\test_registry_respawn.py tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py
```

## License

MIT
