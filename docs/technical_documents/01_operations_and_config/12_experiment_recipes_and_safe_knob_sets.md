# D12 - Experiment Recipes and Safe Knob Sets

## Purpose

This document translates common operator goals into small, evidence-backed config bundles. Each recipe is intentionally conservative: it favors changes that are active in the current runtime and avoids crossing into schema migration, checkpoint compatibility, or brain-topology surgery.

## Scope Boundary

These recipes are not proofs of subsystem behavior. They are practical run setups. For semantics of the underlying systems, use the subsystem chapters. For config status classification, use [D11](./11_config_reference_active_guarded_dead.md).

## Evidence Basis

The recipes below are grounded in:

- active config and runtime-validation paths in `tensor_crypt.runtime_config` and `tensor_crypt.app.runtime`
- viewer control and rendering paths under `tensor_crypt.viewer`
- benchmark harness `scripts/benchmark_runtime.py`
- soak harness `scripts/run_soak_audit.py`

## Recipe Format

Each recipe states:

- goal
- recommended knob bundle
- why the bundle is low risk
- what to inspect afterward

## Recipe 1: Baseline Interactive Smoke Run

### Goal

Confirm that launch, viewer, telemetry, and ordinary world evolution all work on the current machine.

### Recommended knob bundle

- explicit `cfg.SIM.SEED`
- bounded `cfg.SIM.MAX_TICKS`
- moderate `cfg.GRID.W` and `cfg.GRID.H`
- moderate `cfg.AGENTS.N`
- default viewer size unless the display requires adjustment

### Why this is low risk

It changes scale and stop conditions only. It does not alter observation schema, brain topology, or checkpoint posture.

### Inspect afterward

- startup output and run directory path
- viewer pause, zoom, and selection behavior
- presence of `config.json`, `run_metadata.json`, `simulation_data.hdf5`, and the expected Parquet ledgers

## Recipe 2: Small-Map Visual Inspection Run

### Goal

Make world geometry, H-zones, and selection easier to inspect.

### Recommended knob bundle

- reduce `cfg.GRID.W`
- reduce `cfg.GRID.H`
- reduce `cfg.AGENTS.N`
- optionally reduce `cfg.MAPGEN.RANDOM_WALLS`
- keep `cfg.MAPGEN.HEAL_ZONE_COUNT` nonzero

### Why this is low risk

It changes spatial scale, not subsystem contracts.

### Inspect afterward

- ability to select agents and H-zones reliably
- visibility of wall boundaries and H-zone colors
- effect of `R`, `B`, `H`, `G`, and `F` viewer toggles

## Recipe 3: Sparse-World Comparison

### Goal

Observe how lower encounter density changes the apparent pace of contests, births, and deaths.

### Recommended knob bundle

- increase `cfg.GRID.W`
- increase `cfg.GRID.H`
- keep `cfg.AGENTS.N` moderate rather than scaling it proportionally

### Why this is low risk

It changes density and travel distance without touching learning or identity contracts.

### Inspect afterward

- alive-count slope
- family spread across the world
- whether catastrophe geometry is easier to interpret in a sparser arena

## Recipe 4: H-zone Inspection and Live Edit Demo

### Goal

Learn how H-zone visibility and H-zone rate edits behave in the current viewer.

### Recommended knob bundle

- nonzero `cfg.MAPGEN.HEAL_ZONE_COUNT`
- moderate `cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO`
- visible overlay defaults in `cfg.VIEW.SHOW_OVERLAYS`

### Why this is low risk

It uses active map-generation and viewer controls only.

### Inspect afterward

- left-click H-zone selection
- `+` and `-` changing selected H-zone rate instead of speed
- resulting color change in the world panel

## Recipe 5: Catastrophe Visibility Run

### Goal

Make catastrophe state easy to see in the HUD, panel, and world.

### Recommended knob bundle

- `cfg.CATASTROPHE.ENABLED = True`
- choose `cfg.CATASTROPHE.DEFAULT_MODE` deliberately
- `cfg.CATASTROPHE.VIEWER_CONTROLS_ENABLED = True`
- `cfg.CATASTROPHE.VIEWER_OVERLAY_ENABLED = True`
- `cfg.VIEW.SHOW_CATASTROPHE_PANEL = True`
- `cfg.VIEW.SHOW_CATASTROPHE_OVERLAY = True`
- `cfg.VIEW.SHOW_CATASTROPHE_STATUS_IN_HUD = True`

### Why this is low risk

It uses active catastrophe and viewer exposure paths without changing unrelated mechanics.

### Inspect afterward

- catastrophe status line in the HUD
- catastrophe panel content
- `F1` through `F12`, `C`, `Y`, `U`, `I`, and `O` controls when enabled
- special overlay cues such as the Thorn March safe rectangle or the Woundtide front

## Recipe 6: Reproduction-Doctrine Visibility Run

### Goal

Observe doctrine status, doctrine overrides, and floor-recovery behavior.

### Recommended knob bundle

- enable doctrine viewer exposure under `cfg.RESPAWN.OVERLAYS.VIEWER.*`
- keep `cfg.RESPAWN.OVERLAYS.VIEWER.HOTKEYS_ENABLED = True`
- choose explicit doctrine enablement for crowding, cooldown, and local-parent overlays
- set `cfg.RESPAWN.POPULATION_FLOOR` high enough that below-floor behavior is reachable

### Why this is low risk

It stays inside the active overlay and viewer-status surface.

### Inspect afterward

- doctrine status in the panel and HUD
- `Shift+1`, `Shift+2`, `Shift+3`, and `Shift+0` behavior
- override markers when the live doctrine state differs from config defaults

## Recipe 7: Periodic Checkpoint Run

### Goal

Generate checkpoint artifacts suitable for manual inspection or later resume testing.

### Recommended knob bundle

- `cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS = True`
- positive `cfg.CHECKPOINT.SAVE_EVERY_TICKS`
- small positive `cfg.CHECKPOINT.KEEP_LAST`
- `cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED = True`
- `cfg.CHECKPOINT.MANIFEST_ENABLED = True`
- `cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST = True`
- optionally `cfg.CHECKPOINT.WRITE_LATEST_POINTER = True`

### Why this is low risk

It uses the active checkpoint publication path rather than inventing a custom save surface.

### Inspect afterward

- `checkpoints/` directory creation
- `runtime_tick_*.pt` bundles
- paired manifest files
- `latest_checkpoint.json` only when its dependency constraints are satisfied

## Recipe 8: Summary-Heavy Telemetry Run

### Goal

Produce dense summary rows for later artifact inspection.

### Recommended knob bundle

- keep `cfg.TELEMETRY.LOG_TICK_SUMMARY = True`
- keep `cfg.TELEMETRY.LOG_FAMILY_SUMMARY = True`
- `cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS = 1`
- moderate `cfg.TELEMETRY.PARQUET_BATCH_ROWS`
- keep `cfg.TELEMETRY.EXPORT_LINEAGE = True`

### Why this is low risk

It changes telemetry density, not simulation ownership or state layout.

### Inspect afterward

- `tick_summary.parquet`
- `family_summary.parquet`
- `lineage_graph.json`
- buffer flush behavior on clean shutdown

## Recipe 9: Headless Benchmark Run

### Goal

Measure throughput and resource use without the interactive viewer.

### Recommended surface

Use `scripts/benchmark_runtime.py`, which already supports arguments for:

- tick count and warmup ticks
- seed
- device
- width, height, and agent count
- wall and H-zone count
- ray count
- PPO update cadence
- telemetry summary cadence
- parquet batch rows
- checkpoint cadence and retention
- optional profiling
- optional experimental family-vmap inference

### Why this is low risk

It is an existing headless harness rather than a custom benchmark wrapper.

### Inspect afterward

- JSON benchmark output
- elapsed time and ticks per second
- memory fields
- final tick and alive count
- checkpoint and inference-path fields when requested

## Recipe 10: Soak Audit Run

### Goal

Exercise the system for longer headless validation with periodic invariant and checkpoint checks.

### Recommended surface

Use `scripts/run_soak_audit.py`.

### Why this is low risk

It is the repository's explicit soak harness and already performs periodic verification logic instead of relying on ad hoc manual monitoring.

### Inspect afterward

- invariant-check output
- periodic checkpoint validation output
- final run directory contents
- any reported finiteness, ownership, or checkpoint-round-trip failures

## Unsafe Edit Classes to Exclude from Routine Recipes

The following changes should not be folded into routine operator recipes:

- canonical observation feature-count changes
- brain family order or family spec rewrites
- schema-version edits
- identity-substrate migration edits
- checkpoint strictness changes without understanding manifest dependencies
- direct reinterpretation of currently unread config fields as if they were active

## Cross References

- Operational launch and controls: [D10](./10_operator_runbook_and_game_manual.md)
- Config-status map: [D11](./11_config_reference_active_guarded_dead.md)
- Telemetry and artifacts: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint publication and restore semantics: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Validation harnesses: [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
