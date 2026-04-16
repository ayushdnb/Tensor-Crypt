# Learning System Overview and Data Ownership

> Scope: Provide a system-level map of how observations, logits, actions, rewards, buffers, optimizers, counters, and checkpoints are connected, with ownership boundaries made explicit.

## Who this document is for
Technical readers, maintainers, and auditors studying the learning path, data ownership, and checkpoint-sensitive training surfaces.

## What this document covers
- engine-to-PPO data flow
- who owns what state
- brain lookup versus UID ownership
- where checkpoints intersect learning

## What this document does not cover
- generic RL exposition beyond what is needed for the repository
- viewer operation details

## Prerequisite reading
- [PPO from first principles to UID-owned rollouts](../01_foundations/07_ppo_from_first_principles_to_uid_owned_rollouts.md)
- [Agent registry and lifecycle](../03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md)

## 1. System map

A single tick of the live learning path can be summarized as:

1. the engine builds observations for alive slots
2. each live slot reaches a family-aligned brain
3. the actor head produces logits and the critic head produces values
4. actions are sampled
5. physics and environment effects update live state
6. rewards are computed
7. transitions are stored into UID-owned rollout buffers
8. when cadence and readiness conditions are met, PPO updates run
9. learning state becomes checkpoint-visible

See the learning asset:
- [UID-owned PPO lifecycle panel](../assets/diagrams/learning/uid_owned_ppo_lifecycle_panel.md)

## 2. Ownership table

| Surface | Ownership model |
| --- | --- |
| live brain hosting | slot-local at runtime |
| family identity | UID-owned |
| rollout buffer | UID-owned |
| optimizer state | UID-owned |
| training counters | UID-owned |
| bootstrap tail | UID-owned |
| action tensor for current step | ephemeral engine/runtime surface |
| checkpoint manifest | checkpoint publication surface |

## 3. Lookup versus ownership

The engine often locates the currently active brain through a slot because the registry maps active UIDs to live slots. That lookup is an execution convenience. Training ownership remains UID-keyed.

> **Invariant**
> Slot lookup must never be mistaken for slot ownership of PPO state.

## 4. Family-aware inference without ownership collapse

The engine can bucket alive slots by family and, in an explicitly experimental path, use a family-vmap inference route when topology signatures align. This is still not parameter sharing. Each live brain remains its own module instance and each UID retains its own optimizer and training history.

## 5. Learning surfaces that enter checkpoints

The checkpoint bundle can include:
- brain states by UID
- brain metadata by UID
- optimizer states by UID
- serialized buffers by UID
- serialized training state by UID
- scaler state
- RNG state

That breadth is why learning documentation and checkpoint documentation must cross-link tightly.


## Read next
- [PPO buffers, bootstrap, and update cadence](01_ppo_buffers_bootstrap_and_update_cadence.md)
- [Inference execution paths: loop versus family-vmap](03_inference_execution_paths_loop_vs_family_vmap.md)
- [Checkpoint-visible learning state and restore order](04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Run directory artifacts and outputs](../05_operations/02_run_directory_artifacts_and_file_outputs.md)

## If debugging this, inspect…
- [Validation harnesses](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Terms introduced here
- `learning ownership`
- `slot lookup`
- `UID-owned buffer`
- `family-aware inference`
