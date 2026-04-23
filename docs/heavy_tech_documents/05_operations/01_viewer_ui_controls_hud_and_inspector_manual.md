# Viewer UI Controls, HUD, and Inspector Manual

> Scope: Document the interactive pygame-ce viewer, including camera controls, overlays, selection behavior, catastrophe controls, reproduction-doctrine toggles, and the information shown in the HUD and side panel.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- camera controls
- selection semantics
- HUD fields
- side panel controls
- catastrophe hotkeys
- reproduction overlay hotkeys
- operator save/export actions
- inspector fields

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Operator quickstart](00_operator_quickstart_and_common_run_modes.md)
- [Catastrophe system](../03_mechanics/08_catastrophe_system_scheduler_and_world_overlays.md)

## 1. Viewer role

The viewer owns the interactive render loop and UI state. It is not a passive screenshot surface. It reads engine state every frame, prepares a structured state snapshot, and supports world, HUD, and side-panel rendering.

## 2. Core controls

| Control | Effect |
| --- | --- |
| `WASD` / arrows | pan camera |
| mouse wheel | zoom |
| `F` | fit world |
| `Alt+Enter` | toggle fullscreen |
| `Space` | pause / unpause |
| `.` while paused | advance one tick |
| `Esc` | request graceful viewer shutdown and a shutdown checkpoint when enabled |
| `Ctrl+S` | publish a manual runtime checkpoint through the canonical engine path |
| `Ctrl+E` | export the live selected agent's brain weights and metadata |
| `+` / `-` | speed control when no H-zone is selected |
| `R` | toggle rays |
| `B` | toggle HP bars |
| `H` | toggle H-zones |
| `G` | toggle grid |

## 3. Selection model

Left-click in the world:
- prefers a nearby live agent when screen-space proximity matches
- otherwise can select an H-zone by spatial hit
- otherwise clears selection

The side panel then shows either:
- agent detail lines
- H-zone details
- an instruction to click an agent or H-zone

The side panel also shows a compact Actions block. Save checkpoint is always available. Export brain is enabled only when a live agent is selected.

## 4. HUD content

The HUD can show:
- current tick
- pause or speed state
- alive count
- family alive counts
- catastrophe state line
- reproduction overlay doctrine status line

## 5. Agent inspector content

The side panel can show:
- slot and UID
- bloodline family
- brain parameter count
- age, birth tick, generation depth
- brain/trait/anchor parent UIDs
- HP and position
- mass, vision, metabolism
- trait-budget allocations
- PPO counters
- catastrophe exposure summary

UID, family, and parameter count are canonical minimum inspector fields and are not suppressed by migration-era visibility flags.

## 6. H-zone editing path

When an H-zone is selected, `+` and `-` adjust its rate instead of simulation speed. This is a selection-sensitive control path and should be remembered during debugging.

## 7. Reproduction overlay hotkeys

With shift held:
- `Shift+1` toggles The Ashen Press override
- `Shift+2` toggles The Widow Interval override
- `Shift+3` toggles The Bloodhold Radius override
- `Shift+0` clears doctrine overrides

## 8. Catastrophe hotkeys

When viewer catastrophe controls are enabled:
- `F1` through `F12` manually trigger catastrophe entries by roster index
- `C` clears active catastrophes
- `Y` cycles catastrophe mode
- `U` toggles scheduler armed state
- `O` toggles scheduler pause
- `I` toggles catastrophe panel visibility

## 9. Operator save and export

Manual save uses the existing runtime checkpoint publication path with save reason `manual_operator`. It preserves telemetry flushing, manifest/latest-pointer policy, retention, and session metadata update behavior.

Viewer shutdown through `Esc`, window close, or Ctrl+C routes through lifecycle finalization. The finalizer flushes telemetry, publishes a shutdown checkpoint when enabled, closes telemetry, updates session metadata, and prints the shutdown reason, final tick, alive count, run directory, checkpoint result, and telemetry close result.

Selected-brain export writes the live selected agent's weights and metadata through the logger-managed `brains/selected_exports/uid_<uid>/` hierarchy. It does not export dead agents or write to the repository root.


## Read next
- [Run directory artifacts and file outputs](02_run_directory_artifacts_and_file_outputs.md)
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)

## Related reference
- [Viewer HUD and panels atlas](../assets/diagrams/operations/viewer_hud_and_panels_atlas.md)

## If debugging this, inspect…
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)

## Terms introduced here
- `HUD`
- `side panel`
- `selection semantics`
- `doctrine override`
- `catastrophe hotkey`
