# D42 - UID-Owned PPO Rollouts, Rewards, and Updates

## Purpose

This document defines the implemented PPO ownership model of Tensor Crypt, the reward surface, rollout-tail handling, and the conditions under which updates occur.

## Scope Boundary

This chapter covers the current PPO substrate. It does not restate all observation features or all checkpoint file semantics in full.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.learning.ppo`
- `tensor_crypt.simulation.engine`
- `tensor_crypt.population.evolution`

## Ownership Model

Tensor Crypt stores PPO state by UID, not by slot. The current PPO layer owns:

- buffers keyed by UID
- optimizers keyed by UID
- training-state entries keyed by UID

This is a direct continuation of the identity contract described in [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md).

## Why UID Ownership Matters

Slots are reusable. A slot-based PPO ownership model would allow a newborn occupant of a reused slot to inherit rollout or optimizer state that belonged to a dead historical agent. The current implementation avoids that failure mode by binding PPO state to UID instead.

## Reward Surface

The current reward implementation supports reward form:

- `sq_health_ratio`

The live base reward is:

- clamp `hp / hp_max` into `[0, 1]`
- square the result

Full-source validation also confirmed that negative HP values are clamped before squaring in the live reward path.

## Reward Gating

The current reward gate modes include:

- `off`
- `hp_ratio_min`
- `hp_abs_min`

When the selected gate blocks reward, the configured below-gate replacement value is used.

## Rollout Storage and Tail Closure

The current PPO logic distinguishes between:

- terminal stages for dead UIDs
- active stages that still require bootstrap value closure

The implemented behavior includes:

- dead-terminal stages marked bootstrap-complete without requiring future observation
- active stages that can carry bootstrap observations
- optional enforcement that active buffers must have bootstrap information when the corresponding requirement flag is enabled

This is why later checkpoint and validation chapters treat bootstrap state as part of meaningful training continuity.

## Update Triggering

In the current engine, PPO updates are attempted only when:

- the engine tick is greater than zero
- the tick is divisible by `cfg.PPO.UPDATE_EVERY_N_TICKS`

Ready UIDs must also satisfy the current minimum buffer-length conditions before participating in an update.

## Family-Aware Update Considerations

The update path is aware of family and topology constraints when validating optimizer state and reconstructing live learning surfaces. PPO ownership is therefore UID-based, but it is not blind to family-specific brain architecture.

## Optimizer-State Validation

The current PPO and checkpoint stack validate optimizer ownership and compatibility using checks such as:

- parameter-name agreement
- parameter-shape agreement
- parameter-group size agreement
- optional tensor-shape validation for optimizer state tensors

This is important because a restored or reconstructed optimizer state is only meaningful if it still matches the live module topology it is meant to optimize.

## Dormant or Misleading PPO Fields

`cfg.PPO.TRACK_TRAINING_STATE` remains present in config, but the current implementation tracks training state regardless of that flag. Documentation should therefore not imply that disabling it disables the training-state substrate.

## Practical Consequences

Readers should carry forward the following:

- PPO continuity belongs to UID, not slot
- reward is currently health-ratio based and squared after clamping
- rollout-tail closure distinguishes dead-terminal and active-bootstrap cases
- update cadence is downstream of tick completion
- optimizer compatibility checks are part of the live safety story, not an optional appendix detail

## Cross References

- UID ownership substrate: [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
- Tick order and reward staging: [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
- Artifact consequences of PPO events: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint capture and restore of PPO state: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Validation probes and PPO-related consistency checks: [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
