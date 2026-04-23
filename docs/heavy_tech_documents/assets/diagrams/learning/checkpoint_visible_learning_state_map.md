# Checkpoint-Visible Learning-State Map

> Owning document: [Checkpoint-visible learning state and restore order](../../../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)

## What this asset shows
- learning surfaces that enter the checkpoint bundle

## What this asset intentionally omits
- non-learning registry/grid state details

```mermaid
flowchart TD
    A[Checkpoint bundle]
    A --> B[brain_state_by_uid]
    A --> C[brain_metadata_by_uid]
    A --> D[buffer_state_by_uid]
    A --> E[training_state_by_uid]
    A --> F[optimizer_state_by_uid]
    A --> G[optimizer_metadata_by_uid]
    A --> H[scaler_state]
    A --> I[rng_state]
    A --> J[manifest + latest pointer]

```
