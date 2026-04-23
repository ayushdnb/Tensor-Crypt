# Foundations Learning Roadmap

> Scope: Provide a staged path through the foundations layer so a reader can climb from elementary prerequisites to repository-specific understanding without conceptual jumps.

## Who this document is for
Beginners, career-switchers, and technical readers who want a compact study order before reading the deep implementation chapters.

## What this document covers
- a staged learning ladder
- what each foundations document unlocks
- where to jump into repository-specific documents afterward

## What this document does not cover
- full math derivations
- deep subsystem detail

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. The staircase

### Stage 1 — scaling and normalization
Read:
1. `01_high_school_math_for_scaling_rates_and_normalization.md`
2. `02_probability_statistics_and_expected_value_for_rl.md`

These chapters explain why Tensor Crypt normalizes mass, HP, vision, metabolism, distance, age, and zone rates before using them in observations or summaries.

### Stage 2 — gradients and tensor thinking
Read:
1. `03_calculus_gradients_and_backprop_for_this_project.md`
2. `04_linear_algebra_tensors_shapes_and_batching.md`

These chapters make it possible to read the bloodline brain, the canonical observation contract, and the family-vmap inference path without treating tensor shapes as magic.

### Stage 3 — approximation and neural architecture
Read:
1. `05_neural_networks_mlp_design_and_function_approximation.md`

This chapter is the bridge into the repository’s bloodline-aware MLP families.

### Stage 4 — reinforcement learning concepts
Read:
1. `06_reinforcement_learning_mdp_policy_value_and_advantage.md`
2. `07_ppo_from_first_principles_to_uid_owned_rollouts.md`

This stage explains why the repository stores observations, actions, values, rewards, dones, bootstrap tails, optimizers, and update counters the way it does.

### Stage 5 — simulation-engine thinking
Read:
1. `08_grid_world_simulation_engines_and_state_ownership_primer.md`

This chapter explains why deterministic update order, storage ownership, and reversible runtime modifiers matter.

## 2. Two recommended paths

### Compact path
For technically experienced readers:
- 04 tensors and batching
- 06 RL
- 07 PPO
- 08 simulation ownership primer
- then jump to `docs/heavy_tech_documents/02_system` and `docs/heavy_tech_documents/03_mechanics`

### Full path
For readers starting below repository complexity:
- 01 math
- 02 probability
- 03 calculus
- 04 tensors
- 05 neural networks
- 06 RL
- 07 PPO
- 08 simulation primer

## 3. Exit points into the repository docs

After this layer:
- go to `docs/heavy_tech_documents/02_system/00_repository_identity_entry_surfaces_and_public_contract.md` for repository orientation
- go to `docs/heavy_tech_documents/03_mechanics/04_observation_schema_perception_and_ray_semantics.md` if the observation path is the main interest
- go to `docs/heavy_tech_documents/04_learning/00_learning_system_overview_and_data_ownership.md` if PPO ownership is the main interest

## Read next
- [High-school math for scaling, rates, and normalization](01_high_school_math_for_scaling_rates_and_normalization.md)
- [Linear algebra, tensors, shapes, and batching](04_linear_algebra_tensors_shapes_and_batching.md)
- [Grid-world simulation engines and state ownership primer](08_grid_world_simulation_engines_and_state_ownership_primer.md)

## Related reference
- [Observation schema, perception, and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)
- [Learning system overview and data ownership](../04_learning/00_learning_system_overview_and_data_ownership.md)
