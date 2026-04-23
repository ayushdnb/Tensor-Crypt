# Physics, Collisions, Damage, Healing, and Death

> Scope: Explain the movement referee role of the physics subsystem, including collision paths, environment effects, reversible runtime modifiers, and death-context bookkeeping.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- wall collisions
- rams
- contests
- zone effects
- metabolism
- death-context priority
- clamping and death processing

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Action surface and move resolution](05_action_surface_intents_and_move_resolution.md)
- [Grid substrate and field model](00_grid_substrate_and_spatial_field_model.md)

## 1. Physics is the referee

The physics module explicitly states that it determines physically valid outcomes without owning higher-level learning or respawn policy. That boundary matters: physics can kill, but it does not decide how replacement births work.

## 2. Collision paths

The module distinguishes at least:
- wall collision
- ram damage
- contest damage

It also maintains a collision log for telemetry.

## 3. Environment effects

After movement resolution, environment effects apply:
- signed zone rates add or subtract HP
- metabolism drains HP
- positive zone effects accumulate HP gained
- negative zone exposure can stage poison-zone death context

## 4. Reversible runtime modifiers

Catastrophe-aware modifiers can temporarily change:
- collision damage multiplier
- metabolism multiplier
- mass-related metabolism burden

The module emphasizes that these are runtime modifiers rather than permanent mutation of the canonical trait substrate.

## 5. Death processing

Death processing:
1. clamps HP into the legal interval
2. identifies slots at or below zero HP
3. resolves death context if available
4. marks the slot dead in the registry and clears grid occupancy

Death finalization as a UID-retirement event happens later in the evolution/registry path.


## Read next
- [Reproduction, respawn, mutation, and lineage](07_reproduction_respawn_mutation_and_lineage.md)
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)

## Related reference
- [Collision and death-context flow](../assets/diagrams/mechanics/collision_and_death_context_flow.md)

## If debugging this, inspect…
- [Telemetry ledgers and lineage graph](../05_operations/04_telemetry_ledgers_snapshots_exports_and_lineage_graph.md)

## Terms introduced here
- `death context`
- `ram damage`
- `contest damage`
- `reversible runtime modifier`
