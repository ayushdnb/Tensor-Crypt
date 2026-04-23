# Tensor Crypt

Tensor Crypt is a tensor-backed multi-agent simulation runtime with a pygame-ce viewer, UID-owned PPO training state, structured telemetry, and atomic checkpoint publication.

This public line is the single-brain-vmap release surface. The repository launch path applies the self-centric single-family preset before startup: observation mode is `experimental_selfcentric_v1`, the active launch family is `BRAIN.EXPERIMENTAL_BRANCH_FAMILY`, family-shift mutation is disabled, and same-family `torch.func` inference batching is enabled when the live bucket is eligible. The broader bloodline-family substrate remains in the package because it is checkpoint-visible and compatibility-relevant, but it is not presented here as the active launch posture.

Tensor Crypt is a simulation and learning project. It is not a cryptography library.

## Capabilities

- Interactive viewer with pan, zoom, selection, overlays, catastrophe controls, manual checkpoint save, and selected-brain export
- Procedural grid worlds with walls, heal zones, harmful fields, and reversible catastrophe overlays
- Slot-backed agent storage with monotonic UID identity, lifecycle ledgers, family bindings, and parent-role lineage
- Self-centric observation support for the active single-family launch preset
- Policy/value MLP execution with per-UID optimizer and rollout ownership
- Same-family vmap inference acceleration when the runtime bucket satisfies topology and eligibility checks
- Binary-parent reproduction with structured runtime doctrine overrides
- HDF5, Parquet, JSON, and PyTorch output surfaces
- Atomic checkpoint bundles with manifests, checksums, latest-pointer publication, resume-policy validation, and lifecycle metadata

## Repository Layout

```text
.
|-- config.py                 # Public config compatibility wrapper
|-- run.py                    # Canonical repository-root launcher
|-- main.py                   # Equivalent compatibility launcher
|-- pyproject.toml            # Packaging metadata
|-- requirements.txt          # Runtime dependency floor
|-- tensor_crypt/             # Canonical implementation package
|   |-- agents/               # Brains and UID/slot registry
|   |-- app/                  # Launch and runtime assembly
|   |-- audit/                # Programmatic validation probes
|   |-- checkpointing/        # Capture, restore, resume policy, atomic publish
|   |-- learning/             # PPO
|   |-- population/           # Evolution, reproduction, respawn
|   |-- simulation/           # Engine and catastrophe manager
|   |-- telemetry/            # Run directories, ledgers, lineage export
|   |-- viewer/               # Pygame viewer
|   `-- world/                # Grid, map generation, perception, physics
|-- engine/                   # Thin legacy import compatibility package
|-- viewer/                   # Thin legacy import compatibility package
`-- docs/
    |-- architecture/         # Public architecture and compatibility notes
    |-- heavy_tech_documents/ # Full technical corpus and diagram assets
    `-- technical_documents/  # Technical reference chapters
```

`tensor_crypt/` is the canonical implementation root. Root-level `config.py`, `run.py`, and `main.py` are public entry surfaces. The root `engine/` and `viewer/` packages are retained only as thin compatibility imports for older source-tree callers.

## Installation

Tensor Crypt requires Python 3.10 or newer. The viewer dependency is `pygame-ce`, which provides the import namespace `pygame`; the runtime dependency files pin it to the 2.5.x line.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Launch

Run from the repository root:

```bash
python run.py
```

Equivalent entrypoints:

```bash
python main.py
tensor-crypt
```

Startup prints the selected device, launch mode, grid size, initial population size, and run directory. A fresh run writes `config.json` and `run_metadata.json` before entering the viewer.

## Viewer Controls

- `Esc`: request a graceful viewer shutdown and publish the shutdown checkpoint when enabled
- `Space`: pause or resume
- `.`: advance one tick while paused
- `Ctrl+S`: publish a runtime checkpoint through the canonical checkpoint path
- `Ctrl+E`: export the live selected agent brain and metadata
- `+` / `-`: increase or decrease simulation speed
- `WASD` or arrow keys: pan
- Mouse wheel: zoom at cursor
- Left click: select an agent or heal zone
- `R`: toggle rays
- `B`: toggle HP bars
- `H`: toggle heal-zone overlay
- `G`: toggle grid
- `Shift+1`, `Shift+2`, `Shift+3`: toggle reproduction doctrine overrides
- `Shift+0`: clear reproduction doctrine overrides
- `F1`-`F12`: trigger catastrophes manually
- `C`: clear active catastrophes
- `Y`: cycle catastrophe mode
- `U`: arm or disarm the catastrophe scheduler
- `I`: toggle catastrophe panel
- `O`: pause or resume the catastrophe scheduler

The side-panel inspector exposes UID, family, parameter count, slot, birth tick, lineage depth, parent-role UIDs, health, position, trait summary, PPO counters, and catastrophe exposure when the corresponding runtime surfaces are enabled.

## Configuration

The public configuration import surface is `config.py`. It re-exports the canonical dataclasses and singleton `cfg` from `tensor_crypt.runtime_config`.

The main configuration sections are:

- `SIM`: seed, device, dtype, runtime limits, and vmap inference gates
- `GRID` and `MAPGEN`: world dimensions, walls, heal zones, and procedural substrate
- `AGENTS`, `TRAITS`, `RESPAWN`, `EVOL`: population, trait decoding, parented respawn, doctrine overrides, and mutation
- `PERCEPT`: ray casting and observation layout
- `BRAIN`: action/value dimensions, family topology, and the self-centric branch preset
- `PPO`: reward form, reward gates, rollout cadence, and UID ownership enforcement
- `VIEW`: window sizing, overlay defaults, and operator UI controls
- `LOG`, `TELEMETRY`, `CHECKPOINT`, `VALIDATION`: ledgers, exports, checkpoint policy, and programmatic validation helpers
- `IDENTITY`, `SCHEMA`, `MIGRATION`, `CATASTROPHE`: UID invariants, schema versions, compatibility visibility, and catastrophe scheduling

`apply_experimental_single_family_launch_defaults()` is called by the repository launch path. Direct programmatic callers may still configure `cfg` explicitly before building a runtime, but checkpoint-visible surfaces such as observation shape, family topology, UID ownership, and schema versions should be treated as compatibility-critical.

## Outputs And Checkpoints

Each fresh run creates a timestamped directory under `cfg.LOG.DIR`:

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
    |   `-- selected_exports/
    |       `-- u<uid>/
    |           |-- t<tick>_s<slot>_<family>.pt
    |           `-- t<tick>_s<slot>_<family>.json
    `-- checkpoints/
```

Manual checkpoint save, scheduled tick checkpoints, wallclock autosave, and shutdown checkpoints use the same engine publication path. When atomic checkpoint publication is enabled, each checkpoint can publish a bundle, a manifest with checksums and lifecycle metadata, and `latest_checkpoint.json`.

Selected-brain export writes only the currently live selected agent. The `.pt` payload carries the brain state dict and metadata; the `.json` sidecar records UID, slot, family, topology signature, observation contract, lineage fields, session identifiers, and live PPO state presence.

Viewer shutdown through `Esc`, window close, or Ctrl+C routes through lifecycle finalization, prints shutdown details, and publishes a shutdown checkpoint when `ENABLE_SHUTDOWN_CHECKPOINT` remains enabled.

Runtime-generated outputs under `logs/`, `artifacts/`, checkpoints, selected-brain exports, caches, and local scratch paths are generated artifacts. They are intentionally ignored by Git.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Compatibility notes](docs/architecture/compatibility.md)
- [Technical document index](docs/technical_documents/00_meta/00_documentation_bundle_index.md)
- [Heavy technical documentation index](docs/heavy_tech_documents/00_program/00_documentation_index_and_reading_guide.md)

## License

MIT
