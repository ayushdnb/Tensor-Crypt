# Action Surface, Intents, and Move Resolution

> Scope: Explain the discrete action surface, how intents are staged, and how the physics layer resolves competing moves and collisions.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- action dimension
- direction mapping
- intent staging
- non-movers
- contested cells
- approved moves versus blocked moves

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Observation schema and ray semantics](04_observation_schema_perception_and_ray_semantics.md)
- [Grid substrate and spatial field model](00_grid_substrate_and_spatial_field_model.md)

## 1. Action surface

The brain emits logits over the configured action dimension, which the code dump sets to `9`. The physics layer defines nine direction choices:
- stay
- the eight Moore-neighborhood directions

## 2. Intent staging

The physics step first converts live actions into intents:
- current slot
- current position
- proposed target position

This is important because movement is not executed immediately. The system first builds a conflict picture.

## 3. Wall and occupied-target handling

A move can fail or turn into damage before any final displacement occurs:
- wall collisions trigger wall-collision handling
- moving into an occupied cell whose occupant does not vacate can trigger a ram path
- multiple contenders for one cell trigger a contest path

## 4. Contest resolution

The contest path computes a strength signal based on mass and HP ratio, then applies a tie-breaker policy. The winner may move. Losers take the loser-damage path and remain blocked.

## 5. Resolution cache and chains

The engine uses a staged resolution process so that move legality across interdependent proposals is decided coherently rather than by naïve one-pass mutation of grid state.

## 6. Sparse action buffer

The engine can reuse a sparse action buffer rather than allocate a fresh full-width action tensor every step. That is an execution-path implementation detail, not a change in action semantics.


## Read next
- [Physics, collisions, damage, healing, and death](06_physics_collisions_damage_healing_and_death.md)
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)

## Related reference
- [Action-resolution flow](../assets/diagrams/mechanics/action_resolution_flow.md)

## If debugging this, inspect…
- [Benchmarking and performance probe manual](../05_operations/06_benchmarking_and_performance_probe_manual.md)

## Terms introduced here
- `intent staging`
- `non-mover`
- `contest`
- `tie-breaker`
