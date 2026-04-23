# Observation Schema, Perception, and Ray Semantics

> Scope: Document the canonical observation contract, the legacy fallback adapter, the batched ray-casting semantics, and the feature-order invariants shared across perception, brain, PPO, and checkpoints.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- canonical rays/self/context tensors
- legacy adapter path
- per-ray feature semantics
- context features
- shape validation
- why observation shape is checkpoint-sensitive

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Linear algebra, tensors, shapes, and batching](../01_foundations/04_linear_algebra_tensors_shapes_and_batching.md)
- [Traits, bloodlines, families, and brain instantiation](03_traits_bloodlines_families_and_brain_instantiation.md)

## 1. Canonical observation contract

The canonical contract is a three-part bundle:
- `canonical_rays`
- `canonical_self`
- `canonical_context`

The brain module validates the shapes of these tensors explicitly before forward execution.

## 2. Canonical ray features

The perception docstring lists the canonical per-ray contract as:

```text
[hit_none, hit_agent, hit_wall, hit_distance_norm,
 path_zone_peak_rate_norm, terminal_zone_rate_norm,
 target_mass_norm, target_hp_ratio]
```

This is one of the most important cross-module contracts in the repository.

## 3. Canonical self features

The observation builder constructs self features including:
- HP ratio
- HP deficit ratio
- normalized mass
- normalized HP max
- normalized vision
- normalized metabolism
- normalized `x`
- normalized `y`
- distance-to-center norm
- age norm
- current-zone-rate norm

## 4. Canonical context features

The observation builder constructs context features including:
- alive fraction
- mean mass norm
- mean HP ratio

These are global summary features rather than agent-local raw state.

## 5. Legacy fallback surface

The brain supports an adapter path from older keys:
- `rays`
- `state`
- `genome`
- `position`
- `context`

This path exists only when legacy observation fallback is allowed. The existence of the adapter is **not** evidence that canonical and legacy surfaces are interchangeable. The canonical contract remains the target interface.

## 6. Ray-casting semantics

Perception performs batched ray casting over alive agents. Rays:
- exclude self-hits explicitly
- record wall hits and agent hits
- track path-zone peak rate and terminal zone rate
- normalize target mass and target HP ratio only when the ray actually hits an agent

## 7. Checkpoint and compatibility consequence

Observation shape is checkpoint-sensitive because brain family topology and expected feature counts must align across:
- observation construction
- brain input validation
- PPO rollout storage
- checkpoint metadata and restore expectations

> **Common confusion**
> A family may flatten or split the canonical inputs internally, but that does not change the canonical observation contract itself.


## Read next
- [Action surface, intents, and move resolution](05_action_surface_intents_and_move_resolution.md)
- [Learning system overview and data ownership](../04_learning/00_learning_system_overview_and_data_ownership.md)
- [Checkpoint-visible learning state and restore order](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Observation tensor atlas](../assets/diagrams/mechanics/observation_tensor_atlas.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `canonical observation contract`
- `legacy observation fallback`
- `ray semantics`
- `feature-order invariant`
