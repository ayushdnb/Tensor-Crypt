# Inference Path Comparison

> Owning document: [Inference execution paths: loop versus family-vmap](../../../04_learning/03_inference_execution_paths_loop_vs_family_vmap.md)

## What this asset shows
- the decision split between baseline loop execution and optional family-vmap execution

## What this asset intentionally omits
- measured performance outcomes

```mermaid
flowchart TD
    A[alive family bucket] --> B{experimental gate on?}
    B -- no --> C[loop bucket forward]
    B -- yes --> D{torch.func available and eligible bucket?}
    D -- no --> C
    D -- yes --> E[topology and module-type checks]
    E --> F[vmap forward]
    C --> G[log loop bucket stats]
    F --> H[log vmap bucket stats]
    
```
