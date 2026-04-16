# PPO from First Principles to UID-Owned Rollouts

> Scope: Bridge standard PPO concepts to the repository’s stricter ownership model, where buffers, optimizers, counters, and bootstrap tails belong to canonical UIDs.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- trajectory collection
- old log-probs
- ratio clipping
- GAE intuition
- bootstrap tails
- why UID ownership changes the bookkeeping story

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Standard PPO ingredients

PPO typically needs:
- observations
- actions
- old log-probabilities
- rewards
- value estimates
- done flags
- a bootstrap value when the trajectory tail is non-terminal

Tensor Crypt carries all of these, but it adds a high-stakes ownership rule: the rollout is keyed by UID, not merely by whichever slot currently contains a live agent.

## 2. Why old log-probs matter

PPO compares the new policy to the data-collecting policy through a log-prob ratio. That only makes sense if the stored trajectory still belongs to the same conceptual agent identity. UID ownership provides that continuity discipline.

## 3. GAE intuition

Generalized Advantage Estimation blends temporal-difference style bootstrapping with multi-step return information. The practical consequence is that the final state of a partial trajectory matters. If the tail is active, a bootstrap observation/value is needed. If the tail is terminal, it must close cleanly at zero future continuation.

## 4. Ownership consequence

In a slot-reusing simulation, slot identity is unstable. If training state were slot-keyed, the system could accidentally transfer optimizer history or rollout continuity from one canonical agent identity to another. UID-keyed storage prevents that error.

## 5. What “UID-owned rollout” means operationally

It means the following surfaces are conceptually owned by the UID:
- rollout buffer
- optimizer state
- training counters
- truncated-rollout accounting
- bootstrap closure state

The live brain may still be *located* through a slot when the agent is active. But lookup location is not the same thing as ownership.


## Why this matters for Tensor Crypt
This chapter is the conceptual bridge into the learning layer. The repository’s PPO path is best understood as standard PPO plus a strict identity substrate that resists slot-reuse corruption.

## Read next
- [Grid-world simulation engines and state ownership primer](08_grid_world_simulation_engines_and_state_ownership_primer.md)
- [Learning system overview and data ownership](../04_learning/00_learning_system_overview_and_data_ownership.md)
- [Checkpoint-visible learning state and restore order](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
