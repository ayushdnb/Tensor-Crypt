# Linear Algebra, Tensors, Shapes, and Batching

> Scope: Build the tensor-shape literacy required to read the canonical observation contract, bloodline MLP families, and the family-aware inference path.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- vectors and matrices
- rank and shape
- batch dimensions
- reshape versus semantic contract
- why shape invariants are compatibility-sensitive

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Tensor literacy is essential here

Tensor Crypt stores world state in dense tensors, builds batched observations, performs batched ray casting, and runs actor/critic inference over batches of live agents. Shape mistakes are therefore architectural mistakes, not cosmetic ones.

## 2. Rank and shape

A scalar has rank `0`.
A vector such as `[x1, x2, x3]` has rank `1`.
A matrix such as `[[...], [...]]` has rank `2`.
A batched ray tensor such as `[B, R, Fr]` has rank `3`.

The shape tells you what axes exist. The semantic meaning of the axes must still be documented separately.

## 3. Batching

Batching means evaluating several items at once. If a single agent observation has self features of shape `[Fs]`, then a batch of agents has shape `[B, Fs]`.

The repository’s canonical observation contract is semantically:
- rays: `[B, R, Fr]`
- self: `[B, Fs]`
- context: `[B, Fc]`

Internally, a family may flatten rays and concatenate scalars. That implementation convenience does **not** erase the semantic boundary between ray features and scalar features.

## 4. Shape invariants

A shape invariant is a property that must keep matching across modules. In this repository:
- canonical observation feature counts must match perception output, brain input expectations, PPO storage, and checkpoint metadata
- family topology signatures must match checkpoint restore expectations
- action dimension must match engine semantics

## 5. Batch alignment

If rays, self features, and context do not share the same batch dimension, the brain cannot tell which rows belong to the same agent. The repository explicitly validates canonical batch alignment to avoid silent corruption.

## 6. Reshape is not always harmless

Two tensors can contain the same numbers but different meanings. Flattening `[R, Fr]` into `[R*Fr]` changes layout, but not necessarily semantic ownership. A good document therefore explains both:
- the tensor shape used at an interface
- the semantic meaning carried by each axis


## Why this matters for Tensor Crypt
This chapter directly unlocks the canonical observation docs, the bloodline brain family description, and the experimental family-vmap inference path. Without shape literacy, those chapters collapse into opaque implementation detail.

## Read next
- [Neural networks, MLP design, and function approximation](05_neural_networks_mlp_design_and_function_approximation.md)
- [Observation schema, perception, and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)
- [Inference execution paths: loop versus family-vmap](../04_learning/03_inference_execution_paths_loop_vs_family_vmap.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
