# Validation, Determinism, Resume Consistency, and Soak

> Scope: Explain the repository's credibility infrastructure: validation probes, deterministic run checks, resume-consistency tests, save-load-save checks, soak audits, and broader automated test coverage present in the repository tree.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- final validation suite
- determinism probe
- resume consistency probe
- save-load-save signature
- soak runner
- test-surface breadth

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Checkpointing, manifests, restore, and latest pointer](03_checkpointing_manifests_restore_and_latest_pointer.md)
- [Learning system overview](../04_learning/00_learning_system_overview_and_data_ownership.md)

## 1. Validation is a first-class subsystem

The repository exposes dedicated validation helpers rather than relying only on ad hoc confidence. The final validation harness can run:
- determinism probe
- resume consistency probe
- catastrophe reproducibility probe
- save-load-save surface signature

## 2. Determinism probe

The determinism probe builds two fresh runtimes under the same conditions and compares signatures over repeated ticks. That signature includes active UID mappings, slot bindings, family ledgers, grid digest, and PPO state digests.

## 3. Resume consistency probe

The resume probe runs forward, captures a checkpoint, restores from it, then compares the resumed continuation with the uninterrupted baseline continuation.

## 4. Save-load-save surface signature

This check verifies that saving, loading, restoring, and recapturing do not silently mutate core structural surfaces such as slot bindings, grid shape, family ledger, and key state digests.

## 5. Soak runner

The soak runner is a long-form invariant audit. It periodically validates runtime checkpoint surfaces while also checking that registry tensors, grid tensors, parameter tensors, and ownership maps remain finite and aligned.

## 6. Automated test breadth present in the repository

The repository contains a broad automated test surface. The table below summarizes the checked-in categories by test-name evidence.


## Read next
- [Benchmarking and performance probe manual](06_benchmarking_and_performance_probe_manual.md)
- [Extension safety, testing, and change protocol](07_extension_safety_testing_and_change_protocol.md)

| Validation category | Approximate visible tests |
| --- | ---: |
| checkpoint / resume / manifest | 18 |
| catastrophe scheduler and overlays | 22 |
| identity, registry, and ownership | 14 |
| observation and perception | 8 |
| physics and movement | 6 |
| PPO and reward path | 13 |
| viewer and UI | 17 |

## 7. What the validation surface does and does not prove

It **does** provide strong evidence that:
- checkpoint publication and restore are treated seriously
- UID and slot semantics are audited
- observation contracts are tested
- catastrophe scheduling and viewer controls are not undocumented side effects

It does **not** by itself prove:
- empirical learning quality
- generalization
- runtime throughput superiority
- scientific outcomes beyond covered invariants
## Read next
- [Benchmarking and performance probe manual](06_benchmarking_and_performance_probe_manual.md)
- [Extension safety, testing, and change protocol](07_extension_safety_testing_and_change_protocol.md)

## Related reference
- [Validation harness matrix](../assets/diagrams/operations/validation_harness_matrix.md)

## If debugging this, inspect…
- [Checkpointing, manifests, restore, and latest pointer](03_checkpointing_manifests_restore_and_latest_pointer.md)

## Terms introduced here
- `determinism probe`
- `resume consistency`
- `save-load-save`
- `soak audit`
- `credibility infrastructure`
