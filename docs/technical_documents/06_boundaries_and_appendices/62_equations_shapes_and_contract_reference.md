# D62 - Equations, Shapes, and Contract Reference

## Purpose

This appendix provides a compact reference for the most reused tensor shapes, value relations, and contract statements in the bundle.

## Scope Boundary

This document is a reference sheet. Explanatory discussion belongs to the main chapters.

## Evidence Basis

This appendix compacts shapes and contract statements already established in:

- [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)

## Core Shapes

| Surface | Shape |
|---|---|
| `canonical_rays` | `[B, R, F]` |
| `canonical_self` | `[B, S]` |
| `canonical_context` | `[B, C]` |
| logits | `[B, A]` |
| value | `[B, 1]` |

Current canonical widths:

- `F = 8`
- `S = 11`
- `C = 3`
- `A = 9`

## Observation Feature Contracts

### `canonical_rays`

`[hit_none, hit_agent, hit_wall, hit_distance_norm, path_zone_peak_rate_norm, terminal_zone_rate_norm, target_mass_norm, target_hp_ratio]`

### `canonical_self`

`[hp_ratio, hp_deficit_ratio, mass_norm, hp_max_norm, vision_norm, metabolism_norm, x_norm, y_norm, distance_to_center_norm, age_norm, current_zone_rate_norm]`

### `canonical_context`

`[alive_fraction, mean_mass_norm, mean_hp_ratio]`

## Reward Reference

Current reward form:

- `sq_health_ratio`

Current base relation:

- `reward = clamp(hp / hp_max, 0, 1)^2`

Current gating modes:

- `off`
- `hp_ratio_min`
- `hp_abs_min`

## Identity Contract Reference

- slots are dense storage positions
- UIDs are canonical lifecycle identities
- PPO ownership is UID-based
- checkpoint identity capture is UID-ledger based

## World Contract Reference

- grid channel `0`: occupancy and walls
- grid channel `1`: H-zone rate field
- grid channel `2`: occupying slot id
- grid channel `3`: occupying mass

## Checkpoint Contract Reference

The runtime checkpoint captures:

- config snapshot
- registry data and UID ledgers
- grid and H-zone state
- active brains by UID
- PPO buffers and optimizer state as enabled
- respawn and catastrophe runtime state as enabled
- RNG state as enabled

## Cross References

- notation and glossary: [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- observation details: [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- PPO details: [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- checkpoint details: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
