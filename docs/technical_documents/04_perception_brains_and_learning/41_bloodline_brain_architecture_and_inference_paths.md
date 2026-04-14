# D41 - Bloodline Brain Architecture and Inference Paths

## Purpose

This document describes the implemented brain-family roster of Tensor Crypt, the topology parameters that vary by family, and the currently supported inference paths.

## Scope Boundary

This chapter covers architecture and inference routing. It does not restate observation feature construction in full or the PPO update equations.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.agents.brain`
- `tensor_crypt.simulation.engine`
- `tensor_crypt.runtime_config`
- tests including `tests/test_bloodline_brains.py` and `tests/test_experimental_family_vmap_inference.py`

## Family Roster

The current runtime recognizes the following families:

- `House Nocthar`
- `House Vespera`
- `House Umbrael`
- `House Mourndveil`
- `House Somnyr`

The default family is `House Nocthar`.

## Shared Output Contract

Regardless of family, the current brain contract returns:

- action logits with action dimension `9`
- value output with dimension `1`

All families consume the canonical observation contract described in [D40](./40_observation_schema_rays_and_feature_construction.md).

## Family Specifications

The current family specifications differ as follows:

| Family | Widths | Activation | Norm order | Residual | Gated | Split inputs |
|---|---|---|---|---|---|---|
| House Nocthar | `[256, 256, 224, 192]` | `gelu` | pre | yes | no | no |
| House Vespera | `[160, 160, 160, 128, 128]` | `silu` | pre | yes | no | no |
| House Umbrael | `[320, 320, 224]` | `relu` | post | yes | no | no |
| House Mourndveil | `[224, 224, 192]` | `silu` | pre | yes | yes | yes |
| House Somnyr | `[256, 256, 256, 224, 192]` | `gelu` | pre | yes | yes | yes |

For the split-input families currently configured:

- House Mourndveil uses ray width `160` and scalar width `96`
- House Somnyr uses ray width `192` and scalar width `128`

House Somnyr also carries configured dropout `0.02`.

## Within-Family and Across-Family Invariants

The important architectural distinction is:

- topology is stable within a given family specification
- topology may differ across families

This matters for reproduction, checkpoint compatibility, and optimizer-state validation.

## Canonical Forward Path

`tensor_crypt.agents.brain.Brain.forward` consumes canonical observation tensors and returns `(logits, value)`.

The normal live inference path is therefore:

1. build canonical observations
2. route live agents by slot and family
3. forward through family-appropriate brain modules
4. return logits and values for downstream sampling and storage

## Experimental Same-Family VMAP Path

The engine also contains an experimental same-family vmap-style inference path. It is only used when all of the following hold:

- `cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE` is true
- `torch.func` is available
- the bucket size is at least `cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET`
- the candidate brains in the bucket have the same type and topology
- those brains are in evaluation mode

This path is an optimization path, not a different semantic contract.

## Important Boundary for the Experimental Path

The experimental vmap path stacks module state ephemerally for same-family inference. It does not merge brain ownership across UIDs, and it does not redefine per-UID optimizer ownership.

Documentation should therefore not describe the optimization path as if it created one persistent shared brain for all agents of a family.

## Practical Consequences

Readers should carry forward the following:

- family names are part of live architecture selection
- family topology differences are real and checkpoint-relevant
- split-input and non-split families both consume the same canonical observation semantics
- the experimental same-family vmap path is conditional and optional

## Cross References

- Observation contract: [D40](./40_observation_schema_rays_and_feature_construction.md)
- Reproduction and family inheritance: [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md)
- UID-owned PPO state and optimizer validation: [D42](./42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- Checkpoint topology safety: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
