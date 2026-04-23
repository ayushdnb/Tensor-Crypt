# Probability, Statistics, and Expected Value for RL

> Scope: Introduce the probabilistic ideas needed to understand stochastic action sampling, expected return, entropy, and validation under seeded randomness.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- random variables
- categorical action sampling
- expectation and return
- variance and stabilization
- entropy as policy spread
- deterministic seeding versus stochastic policy

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Why probability appears even in a deterministic engine

Tensor Crypt uses deterministic seeding for reproducibility, but a policy can still be stochastic. The engine samples actions from categorical logits. That means the *world update logic* can be deterministic given sampled actions, while the *policy choice* remains probabilistic.

## 2. Expected value

Expected value is a weighted average over possibilities. In reinforcement learning, expected return means the average future reward one would obtain if the same state were encountered many times under the same policy.

A value head is therefore not a score of what happened; it is an estimate of what is expected to happen.

## 3. Variance

Learning from single trajectories is noisy. The same policy may experience good or bad luck. Variance-reduction devices matter:
- value baselines
- advantage centering and normalization
- clipping in PPO
- consistent rollout ownership

## 4. Entropy

Entropy measures spread. For a categorical action distribution:
- high entropy means the policy is still uncertain or exploratory
- low entropy means it is concentrated on a few actions

The repository logs entropy during PPO updates. That is useful because an apparently stable policy can in fact have collapsed too early.

## 5. Determinism is not the same as certainty

A fixed random seed makes repeated runs reproducible only if:
- all random streams are seeded
- restore order is correct
- stochastic decisions are replayed from the same state
- no hidden state is lost at checkpoint boundaries

That is why the repository carries RNG state into checkpoints and exposes determinism and resume-consistency probes.


## Why this matters for Tensor Crypt
This material explains why the repository can be both stochastic at the action level and rigorously deterministic under controlled seeds, and why checkpointing RNG state is part of correctness rather than an optional convenience.

## Read next
- [Calculus, gradients, and backprop for this project](03_calculus_gradients_and_backprop_for_this_project.md)
- [Reinforcement learning: MDP, policy, value, and advantage](06_reinforcement_learning_mdp_policy_value_and_advantage.md)
- [Validation, determinism, resume consistency, and soak](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
