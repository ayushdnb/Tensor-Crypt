# Traits, Bloodlines, Families, and Brain Instantiation

> Scope: Explain how the repository separates trait inheritance, family identity, color mapping, and brain topology while keeping them aligned through UID-owned ledgers.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- family order
- default and initial family assignment
- family colors
- family specs
- trait latent and derived traits
- brain creation per slot

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Agent registry and lifecycle](02_agent_registry_uid_ownership_and_lifecycle.md)
- [Neural networks and MLP design](../01_foundations/05_neural_networks_mlp_design_and_function_approximation.md)

## 1. Family identity is structural

The repository’s bloodline families are not just UI labels. A family determines:
- a valid family id
- a color mapping
- a per-family topology specification
- an ordering used in round-robin assignment and family-aware update ordering

## 2. Family specs

Each family spec can set:
- hidden widths
- activation
- normalization placement
- residual usage
- gating
- split-input mode
- split ray width
- split scalar width
- dropout

That means family identity is tightly coupled to model topology.

## 3. Root assignment and inheritance

The registry can assign root families by:
- round robin
- weighted random

Once parented births occur, family inheritance flows from the brain parent unless explicitly shifted by a mutation path.

## 4. Trait latent versus exposed traits

The reproduction path stores a latent budget-plus-logit representation and derives exposed traits such as:
- HP max
- mass
- vision
- metabolism

This prevents the mistaken assumption that inheritance is copying the currently displayed scalar traits directly.

## 5. Brain instantiation

A live slot hosts a brain instance consistent with the bound UID’s family. If the slot lacks a correct family-aligned brain, the registry instantiates one. That is slot-local hosting of a UID-owned family choice.


## Read next
- [Observation schema, perception, and ray semantics](04_observation_schema_perception_and_ray_semantics.md)
- [Inference execution paths: loop versus family-vmap](../04_learning/03_inference_execution_paths_loop_vs_family_vmap.md)

## Related reference
- [Viewer UI controls, HUD, and inspector manual](../05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md)

## If debugging this, inspect…
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Terms introduced here
- `bloodline family`
- `family spec`
- `trait latent`
- `brain parent`
- `family shift`
