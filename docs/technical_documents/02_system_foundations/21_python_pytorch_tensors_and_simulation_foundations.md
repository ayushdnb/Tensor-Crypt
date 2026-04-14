# D21 - Python, PyTorch, Tensors, and Simulation Foundations

## Purpose

This document provides the minimum conceptual background needed to read the implementation-heavy chapters without losing track of tensor shapes, batched inference, or discrete-time simulation flow.

## Scope Boundary

This chapter is explanatory background. It does not define implementation truth by itself. Whenever this chapter describes a Tensor Crypt-specific contract, the canonical details remain in the code-oriented chapters such as [D22](./22_state_identity_lineage_and_ownership_contracts.md), [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md), and [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md).

## Evidence Basis

Repository-specific examples in this chapter are drawn from:

- `tensor_crypt.agents.brain`
- `tensor_crypt.world.observation_schema`
- `tensor_crypt.world.perception`
- `tensor_crypt.simulation.engine`
- `tensor_crypt.learning.ppo`

## Python and Package Basics Relevant Here

Tensor Crypt is organized as importable Python modules. Examples:

- `tensor_crypt.app.runtime` is a module
- `tensor_crypt.world` is a package family
- `tensor_crypt.agents.brain.Brain` is a class inside a module

The shared runtime configuration is exposed as a dataclass aggregate accessed through `cfg`, not as an unstructured dictionary.

## Tensor Basics

A tensor is a numeric array with one or more ordered dimensions. Tensor Crypt uses tensors for:

- world-grid storage
- dense per-slot agent state
- observation batches
- policy logits and value estimates
- PPO rollout data

The order of tensor dimensions matters. A tensor shaped `[B, R, F]` is not interchangeable with `[R, B, F]`.

## Shape Notation Used in This Bundle

This bundle reuses the notation frozen in [D02](../00_meta/02_notation_glossary_and_shape_legend.md):

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `R` | number of rays |
| `F` | per-ray feature width |
| `S` | canonical self-feature width |
| `C` | canonical context-feature width |
| `A` | action dimension |

## Batched Observation Intuition

A single live agent can be described by:

- `canonical_rays`: `[R, F]`
- `canonical_self`: `[S]`
- `canonical_context`: `[C]`

For neural-network inference, even one agent is typically lifted into a batch:

- `canonical_rays`: `[1, R, F]`
- `canonical_self`: `[1, S]`
- `canonical_context`: `[1, C]`

For many live agents, Tensor Crypt batches them together:

- `canonical_rays`: `[B, R, F]`
- `canonical_self`: `[B, S]`
- `canonical_context`: `[B, C]`

## Brain Output Intuition

The live policy path returns:

- logits over discrete actions: `[B, A]`
- value estimates: `[B, 1]`

The actor output and critic output are distinct. The action logits are not value estimates, and the value head is not an action distribution.

## Flattening and Concatenation in Context

The current brain code derives:

- a flattened ray block from `[B, R, F]` to `[B, R*F]`
- a scalar block from self and context features

Depending on the family specification, the network may:

- process ray and scalar streams separately before mixing them, or
- consume a combined flat input block

This is why later chapters distinguish split-input families from non-split families.

## State Dictionaries and Learnable Modules

`tensor_crypt.agents.brain.Brain` subclasses `torch.nn.Module`. Its learnable state can be represented as a `state_dict`, which is the serializable parameter-and-buffer surface of the module.

This matters because:

- reproduction may copy brain weights from a parent of the same family
- checkpoints serialize and restore family-specific brain state
- optimizer validation depends on parameter names and shapes lining up with reconstructed live modules

## Discrete-Time Simulation Intuition

Tensor Crypt advances in ticks. A tick is a single ordered state transition, not a vague notion of continuous flow. At a high level:

1. observations are built
2. actions are sampled
3. world and physics logic are applied
4. rewards and death state are staged
5. transitions are stored
6. the tick counter advances

The exact implemented order is given in [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md). This chapter only establishes the discrete-time mental model needed to read that ordering correctly.

## One World, Many Batched Agents

Tensor Crypt does not primarily present itself as a bank of independent vectorized environments. Instead, it maintains one evolving world and performs batched tensor operations across the currently alive agents in that world.

This distinction matters when reading:

- live observation construction
- per-family inference paths
- UID-owned rollout accumulation

## Common Misreadings to Avoid

| Misreading | Correction |
|---|---|
| Slot count and batch size are the same concept | Slot capacity and alive-agent batch size are related but not identical. |
| A single observation never needs a batch axis | Neural-network calls typically still expect batch-first inputs. |
| Flattening removes semantic structure | Flattening changes representation shape, not the origin of the features. |
| Batched inference implies many independent worlds | Tensor Crypt mainly batches agents within one world. |

## Cross References

- Runtime and package map: [D20](./20_project_identity_runtime_boot_and_package_map.md)
- Identity substrate: [D22](./22_state_identity_lineage_and_ownership_contracts.md)
- Observation contract: [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- Brain topology: [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md)
- PPO ownership and update semantics: [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- Compact equations and shape reference: [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)
