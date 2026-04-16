# Neural Networks, MLP Design, and Function Approximation

> Scope: Explain why an MLP can serve as a policy/value approximator and how family-specific topology choices affect capacity and compatibility.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- function approximation
- layers and activations
- residual blocks
- normalization placement
- split-input encoders
- parameter count and topology signature

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Function approximation

A neural network is a learnable function. In Tensor Crypt, the policy/value brain maps an observation bundle to:
- action logits
- a value estimate

This is function approximation: the network learns a useful mapping rather than storing a table for every possible state.

## 2. Why an MLP is plausible here

The observation surface is already engineered into dense feature tensors. That makes an MLP a natural candidate:
- it handles fixed-size dense inputs well
- it keeps per-family topology explicit
- it is easier to checkpoint and audit than more exotic architectures

## 3. Layer choices matter

The code dump exposes per-family choices for:
- hidden widths
- activation
- normalization placement
- residual usage
- gating
- split-input encoding
- dropout

Those are not merely tuning details. Together they define a **topology signature**, and the repository treats that signature as checkpoint-visible.

## 4. Split-input design

Some families separate ray-derived features from scalar features before mixing them. That is an architectural choice: it lets the model preserve some structure before the merged trunk.

## 5. Parameter count is descriptive, not evaluative

A larger parameter count does not automatically imply a better family. It only tells you how much trainable capacity the instantiated family contains. The viewer inspector exposes parameter count because it is operationally informative, not because the code dump proves one family is superior.


## Why this matters for Tensor Crypt
The bloodline brain chapter uses this vocabulary directly. Family specs, residual blocks, split-input paths, and topology signatures are repository facts rather than abstract neural-network folklore in this codebase.

## Read next
- [Reinforcement learning: MDP, policy, value, and advantage](06_reinforcement_learning_mdp_policy_value_and_advantage.md)
- [Traits, bloodlines, families, and brain instantiation](../03_mechanics/03_traits_bloodlines_families_and_brain_instantiation.md)
- [Learning system overview and data ownership](../04_learning/00_learning_system_overview_and_data_ownership.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
