# UID-Owned PPO Lifecycle Panel

> Owning document: [Learning system overview and data ownership](../../../04_learning/00_learning_system_overview_and_data_ownership.md)

## What this asset shows
- the lifetime of PPO state under UID ownership

## What this asset intentionally omits
- optimizer math details

```mermaid
flowchart LR
    A[alive slot] --> B[lookup active UID]
    B --> C[brain forward]
    C --> D[store transition in buffer_by_uid]
    D --> E[stage bootstrap or terminal closure]
    E --> F[update readiness]
    F --> G[PPO update]
    G --> H[optimizer_state_by_uid + training_state_by_uid]
    H --> I[checkpoint capture]
    
```
