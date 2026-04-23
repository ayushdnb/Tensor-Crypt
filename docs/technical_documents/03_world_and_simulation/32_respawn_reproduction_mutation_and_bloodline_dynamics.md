# D32 - Respawn, Reproduction, Mutation, and Bloodline Dynamics

## Purpose

This document describes the implemented birth pipeline of Tensor Crypt: parent-role selection, floor-recovery behavior, overlay doctrines, child placement, family inheritance, mutation effects, and extinction-handling behavior.

## Scope Boundary

This chapter focuses on the active reproduction and respawn logic. It does not restate the registry substrate in full or the full engine tick order.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.population.respawn_controller`
- `tensor_crypt.population.reproduction`
- `tensor_crypt.population.evolution`
- `tensor_crypt.agents.state_registry`

## Reproduction Mode

The active runtime validates `cfg.RESPAWN.MODE` to `binary_parented`. The current system should therefore be described as binary-parented reproduction rather than as a generic multi-parent or pluggable-parenting framework.

## Parent Roles

The current birth path distinguishes three roles:

| Role | Current meaning |
|---|---|
| Brain parent | supplies family and, when compatible, the copied brain weights |
| Trait parent | supplies latent trait substrate before child mutation is applied |
| Anchor parent | determines the local placement center used for offspring search |

These roles can coincide in a particular birth, but the implementation keeps them conceptually separate.

## Parent Selection and Floor Recovery

The live reproduction logic uses explicit ranking and threshold rules rather than the dormant selector-name fields still present in config.

- Brain parent selection is fitness-oriented.
- Trait parent selection uses health-oriented ranking terms.
- Floor recovery can suspend some parent thresholds when population falls below the configured floor.

The public fields `cfg.RESPAWN.BRAIN_PARENT_SELECTOR` and `cfg.RESPAWN.TRAIT_PARENT_SELECTOR` are present, but the current runtime does not read them as live selector switches.

## Overlay Doctrines

The active reproduction overlay layer currently exposes three doctrines:

| Doctrine | Operator-facing name | Effect family |
|---|---|---|
| crowding | The Ashen Press | crowding-gated reproduction behavior |
| cooldown | The Widow Interval | refractory window between births |
| local parent | The Bloodhold Radius | locality-sensitive parent selection and placement behavior |

These overlays are active runtime logic, not just viewer labels.

`cfg.RESPAWN.OVERLAYS.COOLDOWN.UNIFIED_UID_POLICY` is part of the active cooldown runtime logic.

## Child Placement

The current placement search uses shuffled square rings around the anchor-parent position. If local placement fails and the corresponding fallback is allowed, the controller can fall back to a broader global search.

This means child placement is neither purely random nor purely exact-anchor occupancy.

## Family and Brain Inheritance

The child family ordinarily inherits from the brain parent. A family shift can still occur through mutation.

If the child family remains the same as the brain parent family, the child brain state is initialized by copying the brain parent's state dict and then applying policy-noise mutation where configured.

If the child family changes, the system cannot treat the parent's state dict as topology-compatible by default.

## Trait Inheritance

The child latent trait substrate is derived from the trait parent and then mutated. Trait inheritance is therefore role-specific rather than being a blind average across every available parent-like signal.

## Birth HP

The current runtime supports birth HP modes of:

- `full`
- `fraction`

This is a validated supported subset, not an arbitrary free-form field.

## Extinction Policies

The runtime validates the extinction-policy field to supported named policies, including:

- `fail_run`
- `seed_bank_bootstrap`
- `admin_spawn_defaults`

However, the current controller implementation uses the same bootstrap-spawn loop for the two non-fail policies. Documentation should therefore not imply materially different runtime behavior between those two non-fail modes unless the implementation changes.

## Checkpoint Visibility

Respawn state is checkpoint-visible. The runtime checkpoint bundle captures:

- respawn timing state
- overlay runtime state
- cooldown-related state

This is important because reproduction doctrine status is not purely transient viewer decoration.

## Dormant or Currently Unread Reproduction Fields

The following public fields remain present but are not read as active runtime switches in the current audit:

- `cfg.RESPAWN.BRAIN_PARENT_SELECTOR`
- `cfg.RESPAWN.TRAIT_PARENT_SELECTOR`
- `cfg.RESPAWN.FLOOR_RECOVERY_REQUIRE_TWO_PARENTS`
- `cfg.RESPAWN.ASSERT_BINARY_PARENTING`

Documentation should not describe them as though toggling them changes current runtime behavior.

## Practical Consequences

Readers should carry forward the following:

- births are binary-parented
- parent roles are explicit and not interchangeable
- floor recovery can soften thresholds
- overlay doctrines are part of live reproduction semantics
- child placement is anchor-centered with fallback logic
- child family is brain-parent driven unless mutation shifts it
- trait inheritance is trait-parent driven before mutation
- the two non-fail extinction policies are currently not behaviorally distinguished by separate controller logic

## Cross References

- UID ownership substrate: [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
- Tick order and death-before-respawn ordering: [D31](./31_tick_order_physics_conflict_resolution_and_death.md)
- Brain families and topology implications: [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md)
- Viewer doctrine controls and catastrophe interactions: [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)
