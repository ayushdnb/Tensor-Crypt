# Checkpointing, Manifests, Restore, and Latest Pointer

> Scope: Provide an operator-facing explanation of how runtime checkpoints are published, validated, pruned, resolved, and restored.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- periodic checkpoint scheduling
- atomic publication
- manifest files
- latest pointer
- strict validation
- keep-last pruning
- safe resume interpretation

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Checkpoint-visible learning state](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)
- [Run directory artifacts](02_run_directory_artifacts_and_file_outputs.md)

## 1. Periodic scheduling

The engine can publish post-tick runtime checkpoints on a configured interval, and records the most recent checkpoint tick and path after successful publication.

## 2. File-set publication model

When atomic manifest publication is enabled, the checkpointing path publishes a **file set**:
- bundle file
- manifest file
- latest pointer file, if configured

This is the operator-visible substrate. Treat the bundle and side files together.

## 3. Manifest purpose

The manifest records items such as:
- schema version
- tick
- active UID count
- artifact filenames
- artifact sizes
- checksums
- config fingerprint
- presence of catastrophe state, RNG state, optimizer state, and buffer state

## 4. Latest pointer purpose

The latest pointer lets operators or resume paths resolve the most recent published checkpoint safely instead of guessing from filenames alone.

## 5. Strict validation

Strict validation can compare:
- pointer tick versus manifest tick
- bundle sizes versus manifest sizes
- bundle checksum versus manifest checksum
- config fingerprint consistency
- schema version matches

## 6. Restore safety rule

A checkpoint should be trusted only if:
- the file set validates
- bundle validation passes
- restore order is respected
- the resume intention matches the desired operational semantics

A raw `.pt` file existing on disk is not, by itself, evidence of a safe resume surface.


## Read next
- [Validation, determinism, resume consistency, and soak](05_validation_determinism_resume_consistency_and_soak.md)
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)

## Related reference
- [Checkpoint publish/load sequence](../assets/diagrams/operations/checkpoint_publish_and_restore_sequence.md)

## If debugging this, inspect…
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Terms introduced here
- `atomic publication`
- `manifest`
- `latest pointer`
- `keep-last pruning`
- `strict validation`
