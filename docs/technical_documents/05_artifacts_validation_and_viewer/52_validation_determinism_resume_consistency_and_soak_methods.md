# D52 - Validation, Determinism, Resume Consistency, and Soak Methods

## Purpose

This document records the explicit validation surfaces that exist in the repository and separates them from ordinary runtime behavior.

## Scope Boundary

This chapter describes audit and validation harnesses, not interactive operation. It is concerned with evidence for determinism, resume consistency, catastrophe reproducibility, and long-run invariant checks.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.audit.final_validation`
- `scripts/benchmark_runtime.py`
- `scripts/run_soak_audit.py`
- tests including `tests/test_validation_harness.py`, `tests/test_runtime_checkpoint_substrate.py`, `tests/test_verification_telemetry_integrity.py`, and `tests/test_catastrophe_scheduler_controls.py`

## Validation Harnesses in `tensor_crypt.audit.final_validation`

The current repository exposes the following validation helpers:

- `run_determinism_probe`
- `run_resume_consistency_probe`
- `run_catastrophe_repro_probe`
- `save_load_save_surface_signature`
- `run_final_validation_suite`

These are validation surfaces. They should not be described as the normal operator loop or as a permanent runtime mode.

## Determinism Probe

The determinism probe is intended to compare repeated executions under fixed seed and comparable conditions. It is evidence that the repository has an explicit determinism-checking path; it is not a claim that every hardware and dependency combination is guaranteed bit-identical.

## Resume-Consistency Probe

The resume-consistency probe checks whether checkpoint save-and-resume behavior preserves the relevant runtime substrate closely enough to match the expected post-resume trajectory.

This is stronger than merely confirming that a checkpoint can be loaded without raising an exception.

## Catastrophe Reproducibility Probe

The catastrophe reproducibility probe specifically targets catastrophe-related state and scheduling behavior under controlled conditions. Its existence is important because catastrophe state is not just a visual layer; it can change world and runtime behavior materially.

## Save-Load-Save Signature Probe

`save_load_save_surface_signature` exists to compare checkpoint surfaces across a save-load-save cycle. This is a structural integrity check, not a general throughput or gameplay test.

## Final Validation Suite

`run_final_validation_suite` is the repository's umbrella validation entry point for these probe families. It is the correct documentation reference when discussing the existence of a bundled validation suite.

## Benchmark Harness

`scripts/benchmark_runtime.py` is a headless measurement harness, not just a manual timing convenience. It produces structured JSON including fields such as:

- elapsed time
- ticks per second
- memory summaries
- final tick
- final alive count
- checkpoint-related fields when relevant
- experimental inference-path fields when relevant

This script is useful for performance-oriented comparisons, but performance conclusions still depend on the chosen configuration and environment.

## Soak Audit Harness

`scripts/run_soak_audit.py` is the repository's headless soak runner. It performs long-run invariant-oriented checks rather than acting as a generic benchmark.

The soak path checks surfaces such as:

- registry and grid finiteness
- PPO buffer ownership consistency
- optimizer ownership consistency
- brain parameter finiteness
- checkpoint round-trip surfaces at intervals

## Validation Boundaries

The repository contains explicit validation machinery, but documentation should still remain careful about overclaiming:

- determinism probes are evidence of testing, not a universal determinism guarantee
- resume probes are evidence of continuity checks, not proof that every future schema change will remain compatible
- benchmark outputs are measurements for a chosen run configuration, not universal performance claims

## Currently Unread Validation Fields

The public validation fields:

- `cfg.VALIDATION.VALIDATION_STRICTNESS`
- `cfg.VALIDATION.SAVE_LOAD_SAVE_COMPARE_BUFFERS`
- `cfg.VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES`

remain present in config, but the current validation code does not read them as active behavior switches.

## Practical Consequences

Readers should carry forward the following:

- Tensor Crypt has explicit validation code, not merely informal testing aspirations
- validation harnesses are separate from ordinary runtime operation
- soak and benchmark scripts serve different purposes
- determinism and resume discussions should remain evidence-bounded

## Cross References

- Artifact surfaces that validation inspects: [D50](./50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint substrate under test: [D51](./51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Boundary discipline for implementation claims: [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md)
