# D40 - Observation Schema, Rays, and Feature Construction

## Purpose

This document defines the implemented observation contract consumed by Tensor Crypt brains. It distinguishes the canonical observation surface from the compatibility payload and records the exact feature ordering used by the current runtime.

## Scope Boundary

This chapter describes observation construction and feature ordering. It does not explain brain topology in full or PPO update logic.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.world.perception`
- `tensor_crypt.world.observation_schema`
- `tensor_crypt.agents.brain`
- tests including `tests/test_perception.py` and `tests/test_bloodline_brains.py`

## Canonical Observation Surface

The canonical observation keys are:

- `canonical_rays`
- `canonical_self`
- `canonical_context`

Their bundle-level shapes are:

- `canonical_rays`: `[B, R, F]`
- `canonical_self`: `[B, S]`
- `canonical_context`: `[B, C]`

where the notation follows [D02](../00_meta/02_notation_glossary_and_shape_legend.md).

## Canonical Ray Features

The current per-ray feature order is:

1. `hit_none`
2. `hit_agent`
3. `hit_wall`
4. `hit_distance_norm`
5. `path_zone_peak_rate_norm`
6. `terminal_zone_rate_norm`
7. `target_mass_norm`
8. `target_hp_ratio`

This ordering is load-bearing because brain input validation and checkpoint-visible topology expectations depend on the configured ray-feature width.

## Canonical Self Features

The current `canonical_self` feature order is:

1. `hp_ratio`
2. `hp_deficit_ratio`
3. `mass_norm`
4. `hp_max_norm`
5. `vision_norm`
6. `metabolism_norm`
7. `x_norm`
8. `y_norm`
9. `distance_to_center_norm`
10. `age_norm`
11. `current_zone_rate_norm`

## Canonical Context Features

The current `canonical_context` feature order is:

1. `alive_fraction`
2. `mean_mass_norm`
3. `mean_hp_ratio`

## Legacy Compatibility Payload

The runtime can also return legacy-shaped observation fields:

- `rays`
- `state`
- `genome`
- `position`
- `context`

These are compatibility outputs, not the canonical contract. The current compatibility shapes are:

- `rays`: last dimension `5`
- `state`: width `2`
- `genome`: width `4`
- `position`: width `2`
- `context`: width `3`

## Canonical Return Toggle

`cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS` controls whether the canonical keys are included in the returned observation bundle. This affects the returned dictionary surface, not the conceptual primacy of the canonical contract.

## Legacy Fallback Boundary

The brain layer can adapt legacy observations to the canonical input form only when `cfg.BRAIN.ALLOW_LEGACY_OBS_FALLBACK` is true.

This is a compatibility path. It should not be described as the primary observation contract.

## Currently Unread Observation Fields

The following public config fields remain present but are not read by the current observation runtime:

- `cfg.PERCEPT.RAY_FIELD_AGG`
- `cfg.PERCEPT.RAY_STEP_SAMPLER`
- `cfg.PERCEPT.LEGACY_ADAPTER_MODE`

Documentation should not imply that they actively steer the current perception logic.

## Practical Consequences

Readers should carry forward the following:

- the canonical observation contract is three-part, not one monolithic tensor
- feature ordering inside each part is fixed
- legacy keys remain available for compatibility, but they are not the authoritative surface
- canonical feature counts are architecture-sensitive and should not be treated as casual tuning knobs

## Cross References

- Tensor-shape background: [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
- Brain families and inference: [D41](./41_bloodline_brain_architecture_and_inference_paths.md)
- PPO rollout ownership and storage: [D42](./42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- Compact shape reference: [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)
