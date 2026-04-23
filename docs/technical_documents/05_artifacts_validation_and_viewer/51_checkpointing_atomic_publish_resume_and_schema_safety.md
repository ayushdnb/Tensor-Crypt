# D51 - Checkpointing, Atomic Publish, Resume, and Schema Safety

## Purpose

This document describes the runtime checkpoint substrate of Tensor Crypt: what the checkpoint captures, how publication works, how restore is ordered, and which validation flags materially affect load safety.

## Scope Boundary

This chapter covers runtime checkpoints, not ordinary run artifacts. Telemetry ledgers and viewer screenshots are outside its primary scope.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.checkpointing.runtime_checkpoint`
- `tensor_crypt.checkpointing.atomic_checkpoint`
- `tensor_crypt.app.runtime`
- `tensor_crypt.learning.ppo`
- tests including `tests/test_checkpoint_atomicity.py` and `tests/test_runtime_checkpoint_substrate.py`

## What a Runtime Checkpoint Captures

The current checkpoint capture path records a broad runtime bundle, including:

- schema-version values
- full config snapshot
- engine tick
- respawn timing and overlay runtime state
- catastrophe state when enabled
- registry tensors and UID-ledger state
- grid tensor, H-zones, and H-zone identifiers
- active brain state keyed by UID, with topology metadata
- PPO buffers, training state, and optimizer state as enabled
- scaler state as enabled
- RNG state as enabled
- metadata including config fingerprint

Before forced, scheduled, or shutdown checkpoint capture, the engine stages active PPO bootstrap observations when `CAPTURE_BOOTSTRAP_STATE` is enabled. This makes the serialized buffer tail explicit at the checkpoint boundary.

This is a state-continuity artifact, not just a model-weight dump.

## Registry and Identity Capture

The checkpoint captures identity-bearing registry surfaces such as:

- `slot_uid`
- `slot_parent_uid`
- `next_agent_uid`
- lifecycle ledger
- family ledger
- parent-role ledger
- trait-latent ledger
- generation-depth ledger

This is necessary because faithful restore depends on rebuilding the UID substrate before live brains and PPO state can be attached correctly.

## Restore Order

The current restore path proceeds in a conservative order:

1. validate the checkpoint bundle
2. restore registry tensors and UID-ledger state
3. rebuild active UID-to-slot mappings
4. recreate live brains by family and load their state dicts
5. restore grid and H-zones
6. refresh the wall cache
7. restore engine tick and respawn runtime state
8. restore PPO buffers, training state, optimizers, and scaler state
9. restore RNG state
10. restore catastrophe state or reset catastrophe state as required
11. resynchronize shadow columns and re-run invariants

This order is load-bearing because later restored surfaces depend on earlier ones already being coherent.

## Atomic Publication Path

The current atomic checkpoint publication path:

1. writes a temporary bundle file
2. writes a temporary manifest file
3. computes manifest metadata including checksums and config fingerprint
4. promotes the bundle with `os.replace`
5. promotes the manifest with `os.replace`
6. validates the published file set
7. optionally writes `latest_checkpoint.json`

This is stricter than a naive save call because it is designed to reduce partial-publication windows.

## Manifest and Latest-Pointer Semantics

The checkpoint bundle, manifest, and latest pointer are distinct:

- bundle: `runtime_tick_*.pt`
- manifest: `runtime_tick_*.pt.manifest.json`
- latest pointer: `latest_checkpoint.json`

`latest_checkpoint.json` is only written when the active save path publishes manifests atomically and `cfg.CHECKPOINT.WRITE_LATEST_POINTER` remains enabled.

## Active Validation Flags

The current restore and validation path materially uses:

- `cfg.CHECKPOINT.STRICT_SCHEMA_VALIDATION`
- `cfg.CHECKPOINT.STRICT_UID_VALIDATION`
- `cfg.CHECKPOINT.STRICT_PPO_STATE_VALIDATION`
- `cfg.CHECKPOINT.STRICT_CONFIG_FINGERPRINT_VALIDATION`
- `cfg.CHECKPOINT.VALIDATE_OPTIMIZER_TENSOR_SHAPES`
- `cfg.CHECKPOINT.VALIDATE_BUFFER_SCHEMA`

When catastrophe state is present, `cfg.CATASTROPHE.STRICT_CHECKPOINT_VALIDATION` is also part of the active validation surface.

## Dependency Constraints in the Save Path

The current runtime enforces dependencies among:

- `ATOMIC_WRITE_ENABLED`
- `MANIFEST_ENABLED`
- `SAVE_CHECKPOINT_MANIFEST`
- `STRICT_MANIFEST_VALIDATION`
- `WRITE_LATEST_POINTER`

This means documentation should not describe latest-pointer publication or strict manifest validation as standalone toggles. They are only meaningful when the atomic-manifest publication path is active.

## What Checkpoints Do Not Replace

A runtime checkpoint does not replace:

- ordinary Parquet ledgers
- ordinary HDF5 telemetry
- standalone brain snapshots under `brains/`

Those artifact families serve different purposes.

## Practical Consequences

Readers should carry forward the following:

- runtime checkpoints are full-state continuity artifacts
- restore safety depends on rebuilding identity before learning state
- manifest and latest-pointer behavior is conditional, not automatic
- strict validation flags are live runtime behavior, not aspirational comments

## Cross References

- Run artifacts outside the checkpoint path: [D50](./50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- UID substrate required for restore coherence: [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
- Validation probes that test resume behavior: [D52](./52_validation_determinism_resume_consistency_and_soak_methods.md)
