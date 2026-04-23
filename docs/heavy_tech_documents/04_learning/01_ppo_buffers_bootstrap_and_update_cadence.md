# PPO Buffers, Bootstrap, and Update Cadence

> Scope: Explain the UID-owned rollout buffer structure, staged bootstrap closure, readiness checks, and update ordering rules used by the repository’s PPO implementation.

## Who this document is for
Technical readers, maintainers, and auditors studying the learning path, data ownership, and checkpoint-sensitive training surfaces.

## What this document covers
- buffer structure
- bootstrap staging
- terminal finalization
- trajectory truncation
- family-aware update ordering
- optimizer creation and validation

## What this document does not cover
- generic RL exposition beyond what is needed for the repository
- viewer operation details

## Prerequisite reading
- [Learning system overview and data ownership](00_learning_system_overview_and_data_ownership.md)
- [Observation schema and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)

## 1. Buffer structure

Each UID-owned buffer stores aligned lists of:
- observations
- actions
- log-probabilities
- rewards
- values
- dones

It also stores:
- optional bootstrap observation
- bootstrap done flag
- finalization kind

The buffer validates both structure and finiteness.

## 2. Why bootstrap staging exists

A non-terminal rollout tail still needs a closing value estimate. The repository stages explicit bootstrap state so GAE and return computation can finish correctly at update time. A terminal tail, by contrast, should close with a done boundary and zero continuation.

## 3. Terminal closure and active closure

The implementation distinguishes:
- terminal closure after death
- active bootstrap closure for unfinished but update-ready rollouts

This distinction prevents a non-terminal buffer from being treated as terminal merely because the update cadence boundary was reached.

## 4. Readiness and update cadence

A UID becomes update-ready when:
- its buffer length reaches the configured batch size
- the UID is still active
- the engine reaches an update cadence boundary

The update path can also order ready UIDs by family when configured, which keeps the update order structured without merging ownership.

## 5. Truncated-rollout accounting

If a buffer is dropped while still non-terminal and not safely closed, the implementation can count it as a truncated rollout. This turns a subtle integrity risk into an auditable metric.

## 6. Optimizer validation

When optimizer state is restored from a checkpoint, the PPO subsystem validates that the saved parameter mapping and tensor shapes still match the currently instantiated family-aligned brain.


## Read next
- [Reward design, gate logic, and value targets](02_reward_design_gate_logic_and_value_targets.md)
- [Checkpoint-visible learning state and restore order](04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Buffer/bootstrap/update flow](../assets/diagrams/learning/buffer_bootstrap_update_flow.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `rollout buffer`
- `bootstrap closure`
- `truncated rollout`
- `ready-to-train ordering`
