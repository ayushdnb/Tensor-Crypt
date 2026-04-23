# D52 - Validation, Determinism, Resume Consistency, and Soak Methods

## Purpose

This document records the explicit validation surfaces that exist in the repository and separates them from ordinary runtime behavior.

## Scope Boundary

This chapter describes audit and validation harnesses, not interactive operation. It is concerned with evidence for determinism, resume consistency, catastrophe reproducibility, and long-run invariant checks.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.audit.final_validation`
- checkpoint capture and restore code under `tensor_crypt.checkpointing`
- runtime assembly and telemetry code under `tensor_crypt.app` and `tensor_crypt.telemetry`

## Validation Harnesses in `tensor_crypt.audit.final_validation`

The current repository exposes the following validation helpers:

- `run_determinism_probe`
- `run_resume_consistency_probe`
- `run_catastrophe_repro_probe`
- `save_load_save_surface_signature`
- `run_resume_policy_probe`
- `run_manual_checkpoint_probe`
- `run_selected_brain_export_probe`
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

## Validation Boundaries

The repository contains explicit validation machinery, but documentation should still remain careful about overclaiming:

- determinism probes are evidence from controlled execution, not a universal determinism guarantee
- resume probes are evidence of continuity checks, not proof that every future schema change will remain compatible
- artifact probes verify specific runtime surfaces; they are not a substitute for reviewing the owning code paths

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
- determinism and resume discussions should remain evidence-bounded

## Cross References

- Artifact surfaces that validation inspects: [D50](./50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint substrate under test: [D51](./51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- Boundary discipline for implementation claims: [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md)
