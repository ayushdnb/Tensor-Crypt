# Catastrophe System, Scheduler, and World Overlays

> Scope: Explain the catastrophe manager’s scheduler modes, event lifecycle, world-field overrides, and runtime modifier behavior.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- modes and scheduler truth
- manual versus auto triggers
- duration and selection
- active catastrophe set
- world override application
- viewer-facing catastrophe state

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Procedural map generation and zones](01_procedural_map_generation_walls_and_zones.md)
- [Physics and death](06_physics_collisions_damage_healing_and_death.md)

## 1. Catastrophe modes

The catastrophe manager supports four modes:
- `off`
- `manual_only`
- `auto_dynamic`
- `auto_static`

It separately tracks whether the scheduler is armed and whether it is paused. This means catastrophe truth is not a single boolean.

## 2. Event lifecycle

An event has:
- event id
- catastrophe id
- display name
- technical class
- start tick
- end tick
- manual flag
- parameter bundle

Events are added to an active list, can expire naturally, and can be cleared manually if permissions allow.

## 3. Selection and planning

Dynamic auto mode uses weighted random selection with bounded gaps. Static auto mode uses a configured interval and a policy such as round robin, configured sequence, or fixed priority.

## 4. World override pattern

At the start of each tick, the engine:
1. expires or triggers catastrophes
2. repaints baseline H-zones
3. applies catastrophe world overrides and runtime modifiers

That sequencing makes the overlay behavior reversible and easier to reason about.

## 5. Modifier targets

Catastrophes can affect:
- zone fields
- perception vision scaling
- physics damage or metabolism multipliers
- respawn controller runtime modifiers
- viewer-visible overlay state

## 6. Viewer and telemetry surfaces

The catastrophe manager exposes structured status used by:
- HUD lines
- side-panel details
- overlay rendering
- catastrophe parquet logging
- validation probes

## 7. Why this subsystem is special

Catastrophes are not mere visual effects. They are cross-cutting runtime overlays that temporarily alter world or subsystem behavior while preserving the canonical substrate beneath them.


## Read next
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Validation, determinism, resume consistency, and soak](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Related reference
- [Catastrophe scheduler state map](../assets/diagrams/mechanics/catastrophe_scheduler_state_map.md)

## If debugging this, inspect…
- [Viewer UI controls, HUD, and inspector manual](../05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md)

## Terms introduced here
- `catastrophe mode`
- `scheduler armed`
- `scheduler paused`
- `world override`
- `active catastrophe`
