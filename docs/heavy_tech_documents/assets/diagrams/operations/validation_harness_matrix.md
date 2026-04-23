# Validation Harness Matrix

> Owning document: [Validation, determinism, resume consistency, and soak](../../../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## What this asset shows
- the main validation harness families

## What this asset intentionally omits
- exact test implementation content

```mermaid
flowchart TD
    A[Validation surface] --> B[Determinism probe]
    A --> C[Resume consistency probe]
    A --> D[Save-load-save signature]
    A --> E[Catastrophe repro probe]
    A --> F[Soak runner]
    A --> G[subsystem validation probes]

```
