# Grid Substrate and Spatial Field Model

> Scope: Document the dense tensor-backed world grid, its channels, and the meaning of the spatial field that agents inhabit.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- grid tensor channel contract
- walls
- H-zone field
- agent slot occupancy
- agent mass shadowing
- what the grid owns and does not own

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Runtime assembly](../02_system/02_runtime_assembly_launch_sequence_and_session_graph.md)

## 1. Core substrate

The `Grid` class owns a dense tensor `self.grid` with four channels:

| Channel | Meaning |
| --- | --- |
| `0` | static occupancy: `0` empty, `1` wall |
| `1` | H-zone rate field |
| `2` | agent slot id, with `-1` for empty |
| `3` | agent mass shadow |

This channel contract is one of the cleanest examples of why the repository distinguishes *runtime storage* from *identity meaning*. Channel `2` stores slot ids, not canonical UIDs.

## 2. Border walls

The grid constructor creates border walls immediately. That means the world is born with hard perimeter occupancy before random wall generation or heal-zone painting occurs.

## 3. H-zone field

The H-zone field is a signed scalar field. Positive values heal. Negative values harm. The grid supports multiple overlap modes:
- `max_abs`
- `sum_clamped`
- `last_wins`

The overlap policy is therefore a true mechanics control surface rather than a viewer-only concern.

## 4. Occupancy and mass shadows

When a live agent occupies a cell:
- channel `2` stores the live slot index
- channel `3` stores that agent’s mass

This allows perception and physics to query occupancy-related information without reading all registry state every time.

## 5. What the grid does not own

The grid does not own:
- canonical UID identity
- PPO state
- trait inheritance
- family binding
- optimizer ownership

It is a world substrate and spatial lookup surface, not the identity ledger.


## Read next
- [Procedural map generation, walls, and zones](01_procedural_map_generation_walls_and_zones.md)
- [Physics, collisions, damage, healing, and death](06_physics_collisions_damage_healing_and_death.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)

## If debugging this, inspect…
- [Run directory artifacts and file outputs](../05_operations/02_run_directory_artifacts_and_file_outputs.md)

## Terms introduced here
- `grid substrate`
- `channel contract`
- `H-zone field`
- `slot occupancy`
