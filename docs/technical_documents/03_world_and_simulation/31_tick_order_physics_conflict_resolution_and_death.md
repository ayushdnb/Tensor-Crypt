# D31 - Tick Order, Physics, Conflict Resolution, and Death

## Purpose

This document defines the implemented per-tick execution order of Tensor Crypt and explains where movement, contests, environment effects, reward staging, death processing, and update scheduling fit into that order.

## Scope Boundary

This chapter focuses on step order and its immediate consequences. It does not restate the world substrate in full or reproduce PPO update equations.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.simulation.engine`
- `tensor_crypt.world.physics`
- `tensor_crypt.population.evolution`
- `tensor_crypt.telemetry.data_logger`

## Implemented Tick Order

The current `Engine.step()` order is:

1. synchronize the registry tick counter
2. run catastrophe pre-tick bookkeeping
3. repaint H-zones
4. apply catastrophe world overrides
5. build catastrophe status for downstream physics
6. identify alive slots
7. take the empty-tick path if no agents are alive
8. build observations
9. sample actions
10. fill the sparse action buffer
11. execute `physics.step()`
12. log physics events
13. apply environment effects
14. record catastrophe exposure
15. compute PPO reward tensor
16. process deaths inside physics
17. compute done flags
18. store transitions
19. finalize death-related telemetry rows
20. run evolution death finalization
21. run respawn controller step
22. check registry invariants
23. log tick summary
24. increment the engine tick
25. trigger PPO update if cadence conditions are satisfied
26. save snapshots if scheduled
27. save runtime checkpoints if scheduled
28. print periodic progress if configured

The order above is load-bearing. Later chapters should not describe these operations as freely commutative.

## Empty-Tick Behavior

If no agents are alive, the engine does not simply stop. It still:

- maintains catastrophe bookkeeping
- runs respawn-controller logic
- logs tick summary information
- increments the tick counter

This matters because extinction and recovery behavior are still possible in a tick with no live agents.

## Physics and Conflict Resolution

### Contest strength

The current contest strength uses:

- agent mass
- HP ratio

More precisely, the active contest logic multiplies mass by HP ratio to rank contenders.

### Tie breaking

The current runtime supports:

- `strength_then_lowest_id`
- `random_seeded`

The tie-break choice is validated at config-validation time; it is not an open-ended string field.

### Damage and environment surfaces

The current physics path includes:

- movement and wall-collision handling
- ram or contest damage
- metabolism
- zone-driven healing or poisoning
- death-context staging

`cfg.PHYS.MOVE_FAIL_COST` is present in config but not read by the current physics implementation.

## Reward Staging and Death Timing

The reward tensor is computed after physics and environment effects, but before UID finalization. This ordering allows the system to stage reward and done information for transitions that correspond to the just-processed world consequences.

The live death-finalization sequence is split across components:

1. physics identifies and marks deaths
2. telemetry death rows are finalized while the relevant context is still available
3. `Evolution.process_deaths()` clears PPO state for dead UIDs and then delegates to registry death finalization

This means "dead in world logic" and "UID fully finalized out of the live substrate" are related but not identical moments.

## Respawn Timing

Respawn runs after death processing and after death-related telemetry finalization. This preserves the intended semantics that a child entering a reused slot is a later event than the death and retirement of the prior UID.

## Update Cadence

PPO update triggering is tick-cadence based. In the current engine, updates occur only when:

- `tick > 0`
- `tick % cfg.PPO.UPDATE_EVERY_N_TICKS == 0`

The update is therefore downstream of the completed tick, not interleaved in the middle of physics or observation construction.

## Why the Order Matters

The current ordering prevents several classes of ambiguity:

- world overrides are applied before observation and physics for that tick
- transitions are stored against post-physics, post-environment, post-done state for the current tick
- telemetry sees death context before the UID is finalized out of the active substrate
- respawn occurs only after death has been processed
- learning updates occur after the tick has been fully staged

## Practical Reading Rules

When later chapters describe the system, they should assume:

- catastrophe field repaint precedes observation construction
- observation construction precedes action sampling
- physics precedes reward storage
- death telemetry finalization precedes registry UID finalization
- respawn is downstream of the death path
- PPO updates are downstream of tick completion

## Cross References

- World substrate and catastrophe classes: [D30](./30_world_grid_map_hzones_and_catastrophe_substrate.md)
- Reproduction and slot reuse after death: [D32](./32_respawn_reproduction_mutation_and_bloodline_dynamics.md)
- UID ownership and rollout storage: [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- Artifact consequences of this order: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
