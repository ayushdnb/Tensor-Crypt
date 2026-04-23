# D53 - Viewer Controls, Inspection, and Diagnostics

## Purpose

This document focuses on the viewer as an inspection and diagnostic surface: what it renders, how selection behaves, how the layout is organized, and which controls are implemented for runtime inspection.

## Scope Boundary

This chapter does not restate world mechanics or launch semantics in full. It focuses on the current viewer behavior and its diagnostic utility.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.viewer.main`
- `tensor_crypt.viewer.input`
- `tensor_crypt.viewer.panels`
- `tensor_crypt.viewer.layout`
- `tensor_crypt.viewer.camera`
- `tensor_crypt.viewer.colors`
- tests including `tests/test_viewer_layout_cleanup.py`, `tests/test_viewer_color_semantics.py`, and `tests/test_engine_and_viewer_integration.py`

## Window and Layout

The viewer uses a resizable window and maintains three major regions:

- world region
- lower HUD region
- side inspection panel

The layout manager keeps these regions within the window across tested sizes. The viewer camera computes cell size dynamically; the public config field `cfg.VIEW.CELL_SIZE` is present but not used by the current viewer path.

## Selection Semantics

The current selection behavior is:

1. choose the nearest visible agent under the cursor in screen space when possible
2. otherwise fall back to the grid occupant at the clicked cell
3. otherwise fall back to H-zone selection
4. otherwise clear selection

This makes zoomed interaction more stable than a strict cell-only picker.

## World Rendering Controls

The current viewer exposes toggles for:

- rays
- HP bars
- H-zones
- grid lines
- catastrophe panel and catastrophe overlay

Camera fit, zoom, and pan are also active diagnostic controls.

## Side-Panel Diagnostics

The side panel can show:

- selected agent identity and family information
- selected agent trainable parameter count
- manual checkpoint and selected-brain export actions
- generation depth and parent-role information
- physical and trait-like scalar information
- catastrophe information
- doctrine status information
- control legend text

For a live selected agent, UID, family, and parameter count are mandatory inspector facts. They are not hidden by migration-era visibility flags.

When `cfg.TELEMETRY.ENABLE_VIEWER_INSPECTOR_ENRICHMENT` is true, additional inspection fields can be exposed without changing simulation semantics.

## Operator Artifacts

The viewer exposes two deliberate operator actions:

- `Ctrl+S` and the side-panel Save action publish a manual checkpoint through `Engine.publish_runtime_checkpoint(...)` with save reason `manual_operator`.
- `Ctrl+E` and the side-panel Export action export only the live selected agent's brain through `DataLogger.export_selected_brain(...)`.

`Esc`, window close, and Ctrl+C request graceful viewer shutdown. The viewer routes the shutdown reason into lifecycle finalization, prints the final tick, alive count, run directory, checkpoint result, and telemetry close result, and uses the shutdown checkpoint path when that checkpoint policy is enabled.

Selected-brain exports land below the logger-managed `brains/selected_exports/uid_<uid>/` hierarchy and include both a `.pt` weight bundle and a `.json` metadata sidecar.

## Catastrophe Diagnostics

When catastrophe viewer exposure is enabled, the viewer can show:

- catastrophe mode
- scheduler armed or paused state
- active catastrophe list
- catastrophe panel content
- catastrophe overlay wash
- special cues such as the Thorn March safe rectangle and the Woundtide front

## Reproduction-Doctrine Diagnostics

When doctrine viewer exposure is enabled, the viewer can show:

- doctrine status in the HUD
- doctrine status in the panel
- override markers
- live response to `Shift+1`, `Shift+2`, `Shift+3`, and `Shift+0`

This is diagnostic visibility of active reproduction overlays, not a separate reproduction implementation.

## Mouse-Wheel and Scroll Behavior

The current input layer distinguishes:

- mouse wheel over the world panel: zoom
- mouse wheel over the side panel: panel scroll

This is part of the implemented viewer ergonomics and is tested indirectly through layout and integration coverage.

## Color Semantics

Family color, health-related display modulation, and other viewer colors are owned by the current color utilities and renderer path. These colors are presentation surfaces; they do not redefine the underlying simulation state.

## Practical Consequences

Readers should carry forward the following:

- the viewer is an inspection layer over the runtime, not a separate simulation
- selection prefers visible agents before H-zones
- layout and camera behavior are active implementation surfaces with test coverage
- some public viewer config fields remain present without driving the current viewer implementation

## Cross References

- Operator controls in brief: [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md)
- World and catastrophe substrate: [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
- Artifact interpretation for viewer-correlated runs: [D50](./50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
