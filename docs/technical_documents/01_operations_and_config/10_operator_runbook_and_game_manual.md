# D10 - Operator Runbook and Game Manual

## Purpose

This document is the operator-facing entry chapter for Tensor Crypt. It explains how to launch the program, what the viewer displays, which controls are implemented, and which generated artifacts an operator should expect from an ordinary run.

## Scope Boundary

This chapter is operational rather than exhaustive. It does not define the slot-versus-UID identity contract in full, does not restate the full checkpoint schema, and does not attempt to derive PPO or reproduction internals from first principles. Those topics belong primarily to:

- [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
- [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
- [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
- [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)

## Evidence Basis

This runbook is grounded in the current launch, runtime, viewer, and artifact code, primarily:

- `run.py`
- `main.py`
- `tensor_crypt.app.launch`
- `tensor_crypt.app.runtime`
- `tensor_crypt.telemetry.run_paths`
- `tensor_crypt.viewer.main`
- `tensor_crypt.viewer.input`
- `tensor_crypt.viewer.panels`

## Canonical Launch Surfaces

The repository provides two root-level launch files:

- `run.py`
- `main.py`

Both delegate to `tensor_crypt.app.launch.main`. The canonical construction path then:

1. validates and seeds deterministic surfaces
2. creates the run directory
3. builds the runtime objects
4. constructs the viewer
5. enters the interactive viewer loop

The canonical implementation owners live under `tensor_crypt.*`. The root files are public entry points, not separate runtime implementations.

## Runtime Assembly Seen by an Operator

The current runtime build path creates, in order:

- `DataLogger`
- `Grid`
- `Registry`
- `Physics`
- `Perception`
- `PPO`
- `Evolution`
- procedural walls and H-zones
- initial population
- `Engine`
- `Viewer`

This matters operationally because early run artifacts, viewer state, and initial world layout are all downstream of that construction order.

## Operator Vocabulary

| Term | Practical meaning |
|---|---|
| Tick | One engine step. |
| World grid | The 2-D simulation arena containing walls, H-zone field values, and live occupancy. |
| Slot | Dense storage location for a live or reusable agent position in the registry. |
| UID | Canonical identity used for lifecycle, lineage, telemetry, and PPO ownership. |
| Family | Bloodline grouping that determines brain architecture family and viewer coloration. |
| H-zone | Rectangular heal-or-harm field contribution stored in the grid substrate. |
| Catastrophe | Temporary world or runtime modifier applied by the catastrophe manager. |
| Run directory | Timestamped output directory created for one launched run. |

## What the Viewer Shows

The viewer uses a resizable window and splits the interface into three main regions:

- world panel
- lower HUD strip
- right-side inspection panel

### World panel

The world panel can show:

- walls
- H-zone coloration
- live agents colored by family
- optional HP bars
- optional selected-agent rays
- optional grid lines
- catastrophe overlays and catastrophe-specific cues when enabled

### HUD strip

The HUD exposes:

- current tick
- pause state and speed multiplier
- alive count
- family counts
- catastrophe status when enabled
- reproduction-doctrine status when enabled

### Inspection panel

Depending on the current selection, the panel can show:

- selected agent details
- selected H-zone details
- neutral guidance text when nothing is selected

When an agent is selected, the panel can include slot, UID, family, age, parent-role fields, HP, position, mass, vision, metabolism, and optional viewer-enrichment fields. When an H-zone is selected, the panel shows the zone identifier, rectangle, and current rate.

## Implemented Controls

The current input layer in `tensor_crypt.viewer.input.InputHandler` implements the following operator controls.

| Control | Effect |
|---|---|
| `Esc` | Exit the viewer |
| `Alt+Enter` | Toggle fullscreen |
| `Space` | Pause or resume |
| `.` | Advance exactly one tick while paused |
| `+` or `=` | Double simulation speed, or raise selected H-zone rate if an H-zone is selected |
| `-` | Halve simulation speed, or lower selected H-zone rate if an H-zone is selected |
| `R` | Toggle ray rendering |
| `B` | Toggle HP bars |
| `H` | Toggle H-zone rendering |
| `G` | Toggle grid lines |
| `F` | Fit the camera to the world |
| `W`, `A`, `S`, `D` or arrows | Pan the camera |
| mouse wheel over world | Zoom |
| mouse wheel over side panel | Scroll inspection panel |
| left click | Select nearest visible agent, then cell occupant, then H-zone |

### Reproduction-doctrine hotkeys

When `cfg.RESPAWN.OVERLAYS.VIEWER.HOTKEYS_ENABLED` is true:

| Control | Effect |
|---|---|
| `Shift+1` | Toggle crowding doctrine override |
| `Shift+2` | Toggle cooldown doctrine override |
| `Shift+3` | Toggle local-parent doctrine override |
| `Shift+0` | Clear doctrine overrides |

### Catastrophe hotkeys

When `cfg.CATASTROPHE.VIEWER_CONTROLS_ENABLED` is true:

| Control | Effect |
|---|---|
| `F1` through `F12` | Trigger catastrophe by configured index |
| `C` | Clear active catastrophes |
| `Y` | Cycle catastrophe mode |
| `U` | Toggle scheduler armed state |
| `I` | Toggle catastrophe panel visibility |
| `O` | Toggle scheduler pause |

## Operator-Visible World Semantics

### Walls and H-zones

The grid always contains border walls. Additional random one-cell wall segments and positive H-zones are added by procedural generation during runtime construction. Negative H-zone values can still appear later because catastrophes can repaint the field and the viewer allows selected H-zone rate edits.

### Families and color

Live agents are colored by family, not by UID and not by health. HP can still affect rendering details when the corresponding viewer logic is enabled, but family remains the primary color grouping.

### Selection behavior

Selection is screen-space aware. The input layer first tries to select the nearest visible agent under the cursor, then falls back to direct cell occupancy, then to H-zone selection. This makes zoomed inspection more stable than strict cell-only picking.

## Run Directory Overview

Each run creates a timestamped directory below `cfg.LOG.DIR`. At launch time, `tensor_crypt.telemetry.run_paths.create_run_directory` creates:

- the run root
- `snapshots/`
- `brains/`
- `heatmaps/`
- `config.json`
- `run_metadata.json`

Additional artifacts are then created by runtime components as the run proceeds.

### Common artifacts

| Artifact | Meaning |
|---|---|
| `config.json` | Run-local config snapshot |
| `run_metadata.json` | Run-local metadata including schema and posture summaries |
| `simulation_data.hdf5` | HDF5 store for snapshot-style telemetry |
| `birth_ledger.parquet` | Birth ledger |
| `genealogy.parquet` | Backward-compatible spawn-lineage alias ledger |
| `life_ledger.parquet` | Lifecycle ledger |
| `death_ledger.parquet` | Death ledger |
| `collisions.parquet` | Collision and contest events |
| `ppo_events.parquet` | PPO event rows |
| `tick_summary.parquet` | Tick summary rows |
| `family_summary.parquet` | Family summary rows |
| `catastrophes.parquet` | Catastrophe event rows |
| `lineage_graph.json` | UID-lineage export |

The `checkpoints/` directory is conditional. It appears only if periodic runtime checkpoint publication is enabled and actually executes.

## Safe First Operator Checks

For a first interactive run, the most useful immediate checks are:

1. the startup output shows the device, grid size, initial population, and run directory
2. the viewer window opens and can pan, zoom, pause, and resume
3. the world contains visible walls, H-zones, and family-colored agents
4. selecting an agent exposes both slot and UID when viewer migration flags permit it
5. the run directory contains `config.json`, `run_metadata.json`, and the expected telemetry files after some ticks have elapsed

## Common Misreadings to Avoid

| Misreading | Correct interpretation |
|---|---|
| Slot is the lasting identity of an agent | Slot is reusable storage; UID is the lasting identity substrate. |
| Hiding an overlay removes the underlying simulation effect | Viewer visibility and simulation truth are separate. |
| `+` and `-` always change speed | They edit H-zone rate when an H-zone is selected. |
| Catastrophe code implies catastrophes are always active | Activation depends on catastrophe enablement, mode, and scheduler state. |
| Checkpoint code implies a checkpoint directory must appear in every run | The directory is conditional on active checkpoint publication settings. |

## Where to Continue

- Config status and dormant surfaces: [D11](./11_config_reference_active_guarded_dead.md)
- Viewer diagnostics: [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)
- Run artifacts: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
