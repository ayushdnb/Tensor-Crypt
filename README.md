# Tensor Crypt Five-Brain

Tensor Crypt Five-Brain is the canonical multi-family release line of the Tensor Crypt simulation runtime. It is a tensor-backed multi-agent simulation with UID-owned identity, five bloodline-specific policy/value brain families, PPO learning, binary-parented reproduction, telemetry ledgers, runtime checkpoints, catastrophe controls, and an interactive `pygame-ce` viewer.

This is a simulation and learning project, not a cryptography library. The name refers to dense tensor state and durable runtime records such as identity ledgers, lineage, telemetry, and checkpoints.

## Release Identity

- Default startup uses the five configured bloodline families.
- Brain inputs use the canonical observation contract.
- Slots are dense storage; UIDs own identity, lineage, PPO buffers, checkpoints, and telemetry.
- The optional family-vmap inference path is disabled by default and is treated as a benchmark accelerator, not as the branch identity.
- Root launch and config files are public compatibility surfaces over the canonical `tensor_crypt/` package.

## Main Capabilities

- Interactive viewer with pan, zoom, selection, overlays, and catastrophe controls
- Procedural grid world with walls and heal or harm zones
- Five bloodline-aware MLP policy/value network families with distinct topology signatures
- Batched perception with canonical observations and a legacy observation adapter
- PPO training keyed by UID rather than by slot
- Structured reproduction overlays: The Ashen Press, The Widow Interval, and The Bloodhold Radius
- HDF5, Parquet, JSON, and PyTorch checkpoint artifacts
- Atomic checkpoint publishing with manifests, checksums, and a latest pointer
- Pytest coverage for imports, runtime invariants, UID ownership, PPO, checkpoints, telemetry, viewer controls, and catastrophe scheduling

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
|-- scripts/                   # Benchmark and soak harnesses
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

Startup prints the selected device, population size, grid size, and run directory. The launcher writes `config.json` and `run_metadata.json` before entering the viewer.

## Configuration

The public config surface is `config.py`, which re-exports the canonical dataclasses and singleton `cfg` from `tensor_crypt/runtime_config.py`.

Important supported defaults for this release line:

- `SIM.DTYPE == "float32"`
- `AGENTS.SPAWN_MODE == "uniform"`
- `TRAITS.METAB_FORM == "affine_combo"`
- `BRAIN.INITIAL_FAMILY_ASSIGNMENT == "round_robin"`
- `RESPAWN.MODE == "binary_parented"`
- `PPO.OWNERSHIP_MODE == "uid_strict"`
- `TELEMETRY.LINEAGE_EXPORT_FORMAT == "json"`
- `SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE == False`

Unsupported guarded values fail during runtime validation rather than drifting silently.

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

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --output benchmark.json
```

The benchmark harness can also exercise the optional family-vmap path for controlled comparisons:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --experimental-family-vmap-inference --experimental-family-vmap-min-bucket 8 --output benchmark_vmap.json
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
python -m pytest tests\test_bloodline_brains.py tests\test_perception.py tests\test_registry_respawn.py
python -m pytest tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py tests\test_experimental_family_vmap_inference.py
```

## License

MIT
