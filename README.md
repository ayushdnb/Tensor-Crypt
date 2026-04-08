# Tensor Crypt

Tensor Crypt is a PyTorch-based multi-agent evolution simulation with an interactive Pygame viewer. Agents live on a 2D grid with walls and heal zones, perceive the world through batched ray casting, act through per-agent policy/value networks, learn with PPO, reproduce through a binary parent model, and emit structured logs, checkpoints, and validation data.

## Main capabilities

- Interactive viewer with pan, zoom, selection, overlays, and catastrophe controls
- Procedural map generation with walls and heal zones
- Slot-backed agent storage with canonical UID ownership and lineage tracking
- Bloodline-aware MLP policy/value networks with multiple family topologies
- Batched perception with canonical observations and a legacy observation adapter
- PPO training keyed by UID rather than by slot
- Structured telemetry in HDF5, Parquet, JSON, and PyTorch checkpoint files
- Atomic checkpoint publishing with manifests, checksums, and a latest-pointer file
- A pytest suite covering determinism, checkpoint restore, catastrophe scheduling, and runtime invariants

## How the system works

At startup the launcher seeds all random sources, creates a run directory, builds the runtime graph, generates a procedural map, spawns the initial population, and starts the viewer.

Each tick follows the same broad order:

1. Update catastrophe scheduling and apply temporary world modifiers.
2. Build observations for all alive agents.
3. Run each agent's brain to sample actions and value estimates.
4. Resolve movement, collisions, contests, and environment effects.
5. Compute PPO rewards and store transitions.
6. Finalize deaths, evolve the population, handle respawn, and write telemetry.
7. Optionally publish a runtime checkpoint.

The runtime keeps dense tensors for speed, but identity is defined by monotonic UIDs. That distinction matters for lineage, checkpoints, and PPO ownership: slot reuse does not recycle agent identity or optimizer state.

## Repository structure

```text
.
├── config.py                  # Main configuration file
├── run.py                     # Primary launch entrypoint
├── main.py                    # Alternate launch entrypoint
├── engine/                    # Legacy import shim
├── tensor_crypt/              # Root namespace shim for direct repository execution
├── scripts/
│   └── benchmark_runtime.py   # Headless benchmark harness
├── src/
│   └── tensor_crypt/
│       ├── _compat/           # Legacy import re-exports
│       ├── agents/            # Brains and slot-backed registry
│       ├── app/               # Launch and runtime assembly
│       ├── audit/             # Determinism and checkpoint probes
│       ├── checkpointing/     # Capture, restore, atomic publish, validation
│       ├── learning/          # PPO
│       ├── population/        # Evolution, reproduction, respawn
│       ├── simulation/        # Engine and catastrophe manager
│       ├── telemetry/         # Run paths, logger, lineage export
│       ├── viewer/            # Pygame viewer
│       └── world/             # Grid, map generation, perception, physics
└── tests/                     # Pytest suite
```

The real package lives under `src/tensor_crypt`. The repository root keeps thin launch and compatibility shims so the project can still be run directly from the root directory.

## Installation

The source dump does not show packaging metadata, so the safest setup is a plain Python environment with the libraries imported by the repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pygame pandas pyarrow h5py psutil pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy pygame pandas pyarrow h5py psutil pytest
```

## Quick start

Run the simulation from the repository root:

```bash
python run.py
```

`main.py` is equivalent:

```bash
python main.py
```

Startup prints the selected device, the grid size, the configured population size, and the run directory. The launcher also writes `config.json` and `run_metadata.json` into that run directory before entering the viewer.

## Viewer controls

The viewer binds a small set of direct controls in `viewer.input.InputHandler`:

- `Esc`: quit
- `Space`: pause / resume
- `.`: advance one tick while paused
- `+` / `-`: increase or decrease simulation speed
- `WASD` or arrow keys: pan
- Mouse wheel: zoom at cursor
- Left click: select an agent or heal zone
- `R`: toggle rays
- `B`: toggle HP bars
- `H`: toggle heal-zone overlay
- `G`: toggle grid
- `F1`-`F12`: trigger catastrophes manually
- `C`: clear active catastrophes
- `Y`: cycle catastrophe mode
- `U`: toggle catastrophe auto mode
- `I`: toggle catastrophe panel
- `O`: pause or resume the catastrophe scheduler

## Configuration

Configuration lives in `config.py` as nested dataclasses and a single live object, `cfg`.

The file is organized by concern:

- `SIM`: seed, device, tick timing, action-buffer reuse
- `GRID` and `MAPGEN`: world size, wall generation, heal-zone generation
- `AGENTS`, `TRAITS`, `RESPAWN`, `EVOL`: population size, trait bounds, parent selection, mutation
- `PERCEPT`: ray casting and observation layout
- `BRAIN`: action/value dimensions, bloodline families, family colors, family network shapes
- `PPO`: reward form, reward gating, rollout/update cadence, ownership checks
- `VIEW`: window size, overlays, catastrophe UI
- `LOG`, `TELEMETRY`, `CHECKPOINT`, `VALIDATION`: logging cadence, ledger export, checkpoint policy, audit switches
- `IDENTITY`, `SCHEMA`, `MIGRATION`, `CATASTROPHE`: UID substrate, schema versions, compatibility flags, catastrophe scheduling

Treat the checked-in values as one concrete scenario, not as universal recommendations. Many settings trade off visibility, logging volume, checkpoint frequency, and runtime cost.

The runtime also rejects several unsupported compatibility modes during startup instead of accepting them silently. For example, the current code path requires:

- `SIM.DTYPE == "float32"`
- `AGENTS.SPAWN_MODE == "uniform"`
- `TRAITS.METAB_FORM == "affine_combo"`
- `RESPAWN.MODE == "binary_parented"`
- `PPO.OWNERSHIP_MODE == "uid_strict"`

## Outputs, logs, and checkpoints

Each run creates a timestamped directory under `cfg.LOG.DIR`:

```text
logs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── run_metadata.json
    ├── simulation_data.hdf5
    ├── birth_ledger.parquet
    ├── genealogy.parquet
    ├── life_ledger.parquet
    ├── death_ledger.parquet
    ├── collisions.parquet
    ├── ppo_events.parquet
    ├── tick_summary.parquet
    ├── family_summary.parquet
    ├── catastrophes.parquet
    ├── lineage_graph.json
    ├── brains/
    │   └── brains_tick_<tick>.pt
    └── checkpoints/          # Created when periodic runtime checkpointing is enabled
```

`simulation_data.hdf5` stores agent snapshots, heatmaps, and identity datasets. The run directory is also created with `snapshots/`, `brains/`, and `heatmaps/` subdirectories; in the current logger, snapshots and heatmaps are written into the HDF5 file, while brain state files are written into `brains/`.

Runtime checkpoints are controlled by `cfg.CHECKPOINT`. When periodic checkpointing is enabled, the engine publishes bundle files under the run directory's checkpoint folder using the configured filename prefix. With atomic publishing enabled, each checkpoint can include:

- a `.pt` bundle
- a manifest file with checksums and metadata
- `latest_checkpoint.json` pointing to the most recent published checkpoint

The checkpoint code validates schema versions, UID bindings, PPO state, and manifest metadata during load.

## Benchmarking

The repository includes a headless benchmark harness:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --output benchmark.json
```

The benchmark configures a small runtime, executes a fixed number of ticks, and writes a JSON summary with elapsed time, ticks per second, memory use, final tick, final alive count, and the run directory.

## Testing and validation

Run the test suite with:

```bash
python -m pytest
```

The repository includes a substantial pytest suite. Based on the test names and helper modules, coverage includes:

- deterministic seeding and repeatable runtime traces
- public and compatibility imports
- observation-shape checks and legacy observation bridging
- bloodline family instantiation and topology checks
- UID ownership, slot reuse, and PPO buffer ownership
- reward gating behavior
- checkpoint round-trip and restore validation
- atomic checkpoint publish and manifest validation
- catastrophe scheduling, replay, and viewer state
- lineage export and telemetry schema checks
- benchmark smoke coverage

There is also a programmatic validation package under `tensor_crypt.audit` with helpers for determinism probes, resume-consistency probes, save-load-save checks, catastrophe replay checks, and a combined final validation suite.

## License

MIT
