# Tensor Crypt

Tensor Crypt is a tensor-backed multi-agent simulation runtime with an interactive pygame-ce viewer. Agents live on a 2D grid with walls and heal zones, perceive the world through batched ray casting, act through per-agent policy/value networks, learn with PPO, reproduce through a binary parent model, and emit structured logs, checkpoints, and validation data.

The project name refers to the repository's dense tensor substrate and its emphasis on durable runtime records such as identity ledgers, lineage, telemetry, and checkpoints. It is a simulation and learning project, not a cryptography library.

## Main capabilities

- Interactive viewer with pan, zoom, selection, overlays, catastrophe controls, manual checkpoint save, and selected-brain export
- Procedural map generation with walls and heal zones
- Slot-backed agent storage with canonical UID ownership and lineage tracking
- Bloodline-aware MLP policy/value networks with multiple family topologies
- Batched perception with canonical observations and a legacy observation adapter
- PPO training keyed by UID rather than by slot
- Structured reproduction overlay doctrines: The Ashen Press, The Widow Interval, and The Bloodhold Radius
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
├── config.py                  # Public config compatibility wrapper
├── run.py                     # Primary launch entrypoint
├── main.py                    # Alternate launch entrypoint
├── tensor_crypt/              # Canonical implementation package
│   ├── runtime_config.py      # Canonical config dataclasses and singleton cfg
│   ├── agents/                # Brains and slot-backed registry
│   ├── app/                   # Launch and runtime assembly
│   ├── audit/                 # Determinism and checkpoint probes
│   ├── checkpointing/         # Capture, restore, atomic publish, validation
│   ├── learning/              # PPO
│   ├── population/            # Evolution, reproduction, respawn
│   ├── simulation/            # Engine and catastrophe manager
│   ├── telemetry/             # Run paths, logger, lineage export
│   ├── viewer/                # Pygame viewer
│   └── world/                 # Grid, map generation, perception, physics
├── engine/                    # Legacy compatibility imports
├── viewer/                    # Legacy compatibility imports
├── scripts/
│   ├── benchmark_runtime.py   # Headless benchmark harness
│   ├── run_soak_audit.py      # Headless soak audit
│   └── dump_py_to_text.py     # Source dump helper
├── docs/
│   ├── architecture/          # Structure and compatibility notes
│   ├── reports/               # Audit, validation, and patch reports
│   └── technical_documents/   # Deep technical reference material
└── tests/                     # Pytest suite
```

`tensor_crypt/` is the only implementation root. The repository root keeps a small public surface (`config.py`, `run.py`, `main.py`) plus compatibility-only `engine/` and `viewer/` packages for legacy imports.

## Installation

The repository ships with standard packaging metadata. An editable install keeps imports, scripts, and tests aligned with the checked-out tree.

The viewer backend is intentionally pinned to the `pygame-ce` 2.5.x line. The code imports it as `pygame`, because `pygame-ce` provides the `pygame` module namespace. The checked-in manifests currently require `pygame-ce>=2.5.6,<2.6`, which matches the version line validated in this repository audit while still allowing patch-level updates within 2.5.x.

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
- `Ctrl+S`: manually publish a runtime checkpoint through the canonical checkpoint path
- `Ctrl+E`: export the live selected agent's brain weights and metadata
- `+` / `-`: increase or decrease simulation speed
- `WASD` or arrow keys: pan
- Mouse wheel: zoom at cursor
- Left click: select an agent or heal zone
- `R`: toggle rays
- `B`: toggle HP bars
- `H`: toggle heal-zone overlay
- `G`: toggle grid
- `Shift+1`: toggle The Ashen Press runtime override
- `Shift+2`: toggle The Widow Interval runtime override
- `Shift+3`: toggle The Bloodhold Radius runtime override
- `Shift+0`: clear reproduction doctrine runtime overrides
- `F1`-`F12`: trigger catastrophes manually
- `C`: clear active catastrophes
- `Y`: cycle catastrophe mode
- `U`: arm or disarm the catastrophe scheduler
- `I`: toggle catastrophe panel
- `O`: pause or resume the catastrophe scheduler

The side-panel inspector includes a compact Actions block for the same manual save/export operations. Manual save remains available while paused. Brain export is enabled only when a live agent is selected.

When a live agent is selected, the inspector always surfaces the canonical UID, family, and trainable parameter count. It also keeps compact forensic fields such as slot, age/birth tick, lineage depth, parent-role UIDs, health/position, trait summary, PPO counters, and catastrophe exposure when the corresponding enrichment surfaces are enabled.

## Configuration

The public configuration entry surface is `config.py`, which re-exports the canonical dataclasses and singleton `cfg` from `tensor_crypt/runtime_config.py`.

The file is organized by concern:

- `SIM`: seed, device, run length, and top-level runtime posture
- `GRID` and `MAPGEN`: world size, heal/harm-field composition, and procedural substrate
- `AGENTS`, `TRAITS`, `RESPAWN`, `EVOL`: population size, latent trait budgets/clamps, binary-parent respawn, structured overlay doctrines, and mutation
- `PERCEPT`: ray casting and observation layout
- `BRAIN`: action/value dimensions, bloodline families, topology, and observation-compatibility policy
- `PPO`: reward form, reward gating, rollout/update cadence, and UID ownership enforcement
- `VIEW`: window size, supported startup overlays, and catastrophe UI
- `LOG`, `TELEMETRY`, `CHECKPOINT`, `VALIDATION`: logging cadence, export cadence, checkpoint policy, and audit switches
- `IDENTITY`, `SCHEMA`, `MIGRATION`, `CATASTROPHE`: UID invariant strictness, schema versions, legacy visibility flags, and catastrophe scheduling
- `SIM.EXPERIMENTAL_FAMILY_VMAP_*`: opt-in same-family inference batching for benchmarking on headless workloads without changing the default per-brain ownership-preserving loop

Treat the checked-in values as one concrete scenario, not as universal recommendations. Many settings trade off visibility, logging volume, checkpoint frequency, and runtime cost.

The reproduction surface now includes a structured `RESPAWN.OVERLAYS` subtree. `CROWDING` configures The Ashen Press (crowding-gated reproduction overlay), `COOLDOWN` configures The Widow Interval (parent refractory reproduction overlay), `LOCAL_PARENT` configures The Bloodhold Radius (local lineage parent-selection overlay), and `VIEWER` controls HUD and hotkey exposure for the runtime override surface. When all three doctrines are disabled, the controller falls back to legacy binary-parent selection and placement behavior.

The surface is intentionally narrower than older audit prose may imply. The dead and documentary-only knobs that were not wired have been removed. Two notable special cases remain: `TRAITS.INIT` is a legacy/template container retained for compatibility and documentation even though the live birth path uses latent decoding, and `TELEMETRY.ENABLE_DEEP_LEDGERS` only gates initial root-seed deep-ledger seeding rather than the broader telemetry stack.

The runtime also rejects unsupported or misleading combinations during startup instead of accepting them silently. For example, the current code path requires:

- `SIM.DTYPE == "float32"`
- `AGENTS.SPAWN_MODE == "uniform"`
- `TRAITS.METAB_FORM == "affine_combo"`
- `RESPAWN.MODE == "binary_parented"`
- `PPO.OWNERSHIP_MODE == "uid_strict"`
- `TELEMETRY.LINEAGE_EXPORT_FORMAT == "json"`
- manifest strictness and latest-pointer features to run only on the manifest-publishing atomic path (`ATOMIC_WRITE_ENABLED`, `MANIFEST_ENABLED`, and `SAVE_CHECKPOINT_MANIFEST`)

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
    │   ├── brains_tick_<tick>.pt
    │   └── selected_exports/
    │       └── uid_<uid>/
    │           ├── uid_<uid>_tick_<tick>_slot_<slot>_<family>.pt
    │           └── uid_<uid>_tick_<tick>_slot_<slot>_<family>.json
    └── checkpoints/          # Created when periodic runtime checkpointing is enabled
```

`simulation_data.hdf5` stores agent snapshots, heatmaps, and identity datasets. The run directory is also created with `snapshots/`, `brains/`, and `heatmaps/` subdirectories; in the current logger, snapshots and heatmaps are written into the HDF5 file, while brain state files are written into `brains/`.

Selected-brain export is an operator action for the currently live selected agent only. The logger writes a weight-bearing `.pt` bundle plus a `.json` metadata sidecar under the session-aware `brains/selected_exports/uid_<uid>/` hierarchy. Metadata includes UID, slot, family, parameter count, topology signature, observation contract, lineage and parent-role fields, export tick, session identifiers, and live PPO state presence/counters.

Runtime checkpoints are controlled by `cfg.CHECKPOINT`. When periodic checkpointing is enabled, the engine publishes bundle files under the run directory's checkpoint folder using the configured filename prefix. `Ctrl+S` and the side-panel Save action call the same canonical engine publication path with save reason `manual_operator` and `force=True`; they do not bypass telemetry flushing, manifests, latest-pointer publication, retention pruning, or session metadata updates. In the current runtime, manifest files and `latest_checkpoint.json` are published only when `ATOMIC_WRITE_ENABLED`, `MANIFEST_ENABLED`, and `SAVE_CHECKPOINT_MANIFEST` are all true. On that path each checkpoint can include:

- a `.pt` bundle
- a manifest file with checksums and metadata
- `latest_checkpoint.json` pointing to the most recent published checkpoint

The checkpoint code validates schema versions, UID bindings, PPO state, and manifest metadata during load. Checkpoint bundles also persist reproduction doctrine runtime state: viewer-toggled doctrine overrides plus The Widow Interval cooldown ledgers are restored on resume instead of snapping back to config defaults.

Runtime-generated outputs under `logs/`, `artifacts/`, checkpoints, selected-brain exports, cache directories, and local task scratch files are generated artifacts. Do not commit them.

## Benchmarking

The repository includes a headless benchmark harness:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --output benchmark.json
```

The benchmark configures a small runtime, executes a fixed number of ticks, and writes a JSON summary with elapsed time, ticks per second, memory use, final tick, final alive count, and the run directory.

The harness also exposes the experimental inference fast path so you can compare the canonical loop against the family-local `torch.func` path under identical seeds and workloads:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --experimental-family-vmap-inference --experimental-family-vmap-min-bucket 8 --output benchmark_experimental.json
```

When enabled, the output JSON includes `experimental_family_vmap_inference`, `experimental_family_vmap_min_bucket`, and `inference_path_stats` so benchmark runs can distinguish loop-routed versus vmap-routed slots and buckets.

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
- reproduction doctrine behavior, runtime hotkeys, and overlay checkpoint round-trips
- lineage export and telemetry schema checks
- benchmark smoke coverage

There is also a programmatic validation package under `tensor_crypt.audit` with helpers for determinism probes, resume-consistency probes, save-load-save checks, catastrophe replay checks, and a combined final validation suite.

## License

MIT
