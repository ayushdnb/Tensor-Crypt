# D30 - World Grid, Map, H-zones, and Catastrophe Substrate

## Purpose

This document defines the static and semi-static world substrate of Tensor Crypt: the grid tensor, procedural map generation, H-zone composition, and the catastrophe layer that can temporarily override or reinterpret parts of that substrate.

## Scope Boundary

This chapter describes world-state structure and catastrophe substrate semantics. It does not describe the full per-tick execution order or PPO consequences of those world changes; those topics belong to [D31](./31_tick_order_physics_conflict_resolution_and_death.md) and [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md).

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.world.spatial_grid`
- `tensor_crypt.world.procedural_map`
- `tensor_crypt.simulation.catastrophes`
- `tensor_crypt.app.runtime`

## Grid Tensor Contract

The grid stores a four-channel tensor whose channels are currently used as follows:

| Channel | Meaning |
|---|---|
| `0` | occupancy and wall substrate |
| `1` | H-zone rate field |
| `2` | occupying agent slot id, with `-1` for empty |
| `3` | occupying agent mass |

This contract matters because later rendering, environment effects, and selection logic all assume these channel meanings.

## Border Walls and Procedural Walls

The world always includes border walls created by the grid implementation. Additional one-cell-thick random wall segments come from `tensor_crypt.world.procedural_map.add_random_walls`.

Operationally, this means a fresh world is never completely empty of barriers even when random wall generation is reduced.

## H-zone Substrate

### Stored representation

H-zones are stored as rectangles with associated identifiers, rates, and active flags. The grid rebuilds the H-zone rate field by repainting active zones.

### Default procedural generation

`tensor_crypt.world.procedural_map.add_random_hzones` creates positive rectangular zones using:

- `cfg.MAPGEN.HEAL_ZONE_COUNT`
- `cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO`
- `cfg.MAPGEN.HEAL_RATE`

The default procedural map therefore starts from positive zones, even though negative field values can appear later.

### Overlap modes

The current grid supports:

- `max_abs`
- `sum_clamped`
- `last_wins`

`sum_clamped` is the only mode that clamps the accumulated value to the configured sum bound during repaint.

### Point lookup

`find_hzone_at` searches active zones in reverse order, so later active zones take lookup priority when multiple rectangles overlap the same location.

## Catastrophe Substrate

### Modes

The catastrophe manager currently supports:

- `off`
- `manual_only`
- `auto_dynamic`
- `auto_static`

Scheduler arm and pause state are distinct from the selected mode.

### Active catastrophe roster

The current catastrophe roster includes:

- `ashfall_of_nocthar`
- `sanguine_bloom`
- `the_woundtide`
- `the_hollow_fast`
- `mirror_of_thorns`
- `veil_of_somnyr`
- `graveweight`
- `glass_requiem`
- `the_witchstorm`
- `the_thorn_march`
- `the_barren_hymn`
- `crimson_deluge`

### Implemented effect classes

The catastrophe layer currently implements several different classes of effects:

| Catastrophe | Effect class |
|---|---|
| `ashfall_of_nocthar` | zeroes selected positive zones |
| `sanguine_bloom` | turns selected zones negative |
| `the_woundtide` | applies a moving negative vertical band |
| `the_hollow_fast` | scales positive field values downward |
| `mirror_of_thorns` | inverts selected zone field values |
| `veil_of_somnyr` | applies perception-vision scaling only |
| `graveweight` | changes metabolism and effective mass burden |
| `glass_requiem` | scales collision damage |
| `the_witchstorm` | applies mutation overrides |
| `the_thorn_march` | imposes a shrinking safe rectangle with negative field outside it |
| `the_barren_hymn` | disables reproduction while active |
| `crimson_deluge` | paints negative patches |

These effects do not all act on the same substrate. Some repaint the world field, while others modify runtime parameters or subsystem behavior.

## Important Boundary: Temporary Override vs Persistent Trait

Catastrophes are runtime overlays or modifiers. They are not lineage-persistent trait state. A catastrophe can affect world field values, reproduction permissibility, or subsystem scalars during its active window without rewriting the inherited trait substrate of a bloodline.

## Checkpoint Visibility

Catastrophe state can be serialized and restored when catastrophe-state persistence is enabled. Checkpoint validation also checks catastrophe schema expectations when catastrophe state is present and strict catastrophe validation remains enabled.

## Practical Consequences

Readers should carry forward the following:

- the grid is multi-channel, not a single occupancy bitmap
- H-zones are rectangle-based and repaint into a field channel
- default procedural H-zones are positive, but runtime overrides can make field values negative
- catastrophe behavior is heterogeneous: not every catastrophe is just a field repaint
- catastrophe overlays are temporary runtime state, not bloodline inheritance

## Cross References

- Tick order, physics, and death sequencing: [D31](./31_tick_order_physics_conflict_resolution_and_death.md)
- Reproduction effects and catastrophe interactions: [D32](./32_respawn_reproduction_mutation_and_bloodline_dynamics.md)
- Observation construction over the world substrate: [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- Viewer presentation of catastrophe state: [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)
