# Checkpoint-Visible Learning State and Restore Order

> Scope: Document what learning-related state enters runtime checkpoints, how manifests and latest pointers fit around the bundle, and why restore ordering is part of correctness.

## Who this document is for
Technical readers, maintainers, and auditors studying the learning path, data ownership, and checkpoint-sensitive training surfaces.

## What this document covers
- bundle contents
- brain state by UID
- optimizer metadata validation
- buffer serialization
- RNG capture
- restore ordering
- manifest checks

## What this document does not cover
- generic RL exposition beyond what is needed for the repository
- viewer operation details

## Prerequisite reading
- [Learning system overview](00_learning_system_overview_and_data_ownership.md)
- [PPO buffers and update cadence](01_ppo_buffers_bootstrap_and_update_cadence.md)
- [Schema legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Learning state in the checkpoint bundle

The runtime checkpoint capture path can include:
- brain state by UID
- brain metadata by UID
- PPO buffer state by UID
- training state by UID
- optimizer state by UID
- optimizer metadata by UID
- scaler state
- RNG state

This is a broad snapshot of the learning substrate, not just a parameter dump.

## 2. Manifest and latest pointer around the bundle

Atomic checkpoint publication can create:
- the bundle file
- a manifest file describing schema version, tick, checksums, sizes, and config fingerprint
- a latest-pointer file pointing to the newest published bundle

These files are operationally important because resume logic and audit logic read them as a file set.

## 3. Restore ordering

The restore helper rebuilds state conservatively:
1. validate the bundle
2. restore registry tensors and UID ledgers
3. rebuild active UID-to-slot bindings
4. instantiate correct-family brains per active slot
5. restore brain state
6. restore grid state
7. restore engine tick and respawn runtime state
8. restore PPO buffers and training state
9. validate and restore optimizer state
10. restore scaler state
11. restore RNG state
12. restore catastrophe state
13. reassert registry and grid invariants

> **Invariant**
> Registry bindings must exist before brain and PPO restore logic can attach learning state safely.

## 4. Optimizer metadata validation

The restore path validates:
- parameter count
- parameter names
- parameter shapes
- param-group sizes
- tensor-state shape alignment

That protects against silent restore into an incompatible family topology.

## 5. Why this is a correctness boundary

A checkpoint that restores weights but not ownership, bootstraps, optimizer history, or RNG state is not merely “incomplete”; it can produce a semantically different continuation. The repository’s checkpoint system is therefore documented as an audited substrate rather than a convenience feature.


## Read next
- [Checkpointing, manifests, restore, and latest pointer](../05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md)
- [Validation, determinism, resume consistency, and soak](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Related reference
- [Checkpoint-visible learning-state map](../assets/diagrams/learning/checkpoint_visible_learning_state_map.md)

## If debugging this, inspect…
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Terms introduced here
- `checkpoint-visible state`
- `optimizer metadata`
- `latest pointer`
- `restore order`
- `config fingerprint`
