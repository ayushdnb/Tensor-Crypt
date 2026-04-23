# Reinforcement Learning: MDP, Policy, Value, and Advantage

> Scope: Introduce the reinforcement-learning concepts that underlie the engine’s action-selection and PPO update path.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- state, action, reward
- policy
- value
- return
- advantage
- why dones and bootstraps matter

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. State, action, reward

In reinforcement learning, an agent observes a state, chooses an action, receives reward, and transitions into a new state. Tensor Crypt’s canonical observation bundle and action logits live inside that pattern.

## 2. Policy

A policy maps observations to action preferences or probabilities. In this repository the actor head produces logits over the configured action dimension.

## 3. Value

A value function estimates expected future return from a state. The critic head exists for that purpose.

## 4. Return

Return is the future reward signal that the learner is trying to predict or optimize. Because trajectories have finite length in practice, return computation often needs a bootstrap value at the tail if the rollout did not terminate naturally.

## 5. Advantage

Advantage answers a comparative question:
- how much better or worse did this action look than the baseline value estimate?

That makes optimization less noisy than using raw return alone.

## 6. Done boundaries

A `done` boundary prevents future reward estimates from leaking past a terminal event. If done handling is wrong, returns and advantages can cross death boundaries incorrectly.


## Why this matters for Tensor Crypt
The repository’s PPO update path, staged bootstrap state, and explicit terminal handling all depend on these concepts. The code is unusually ownership-aware, but it still relies on standard RL ideas underneath.

## Read next
- [PPO from first principles to UID-owned rollouts](07_ppo_from_first_principles_to_uid_owned_rollouts.md)
- [PPO buffers, bootstrap, and update cadence](../04_learning/01_ppo_buffers_bootstrap_and_update_cadence.md)
- [Reward design, gate logic, and value targets](../04_learning/02_reward_design_gate_logic_and_value_targets.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
