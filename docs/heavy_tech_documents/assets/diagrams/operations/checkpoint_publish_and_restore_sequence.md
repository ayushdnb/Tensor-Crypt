# Checkpoint Publish and Restore Sequence

> Owning document: [Checkpointing, manifests, restore, and latest pointer](../../../05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md)

## What this asset shows
- the operator-visible file-set lifecycle around checkpoint publication and restore

## What this asset intentionally omits
- every internal validation branch

```mermaid
sequenceDiagram
    participant E as Engine
    participant C as runtime_checkpoint
    participant A as atomic_checkpoint
    participant F as filesystem
    E->>C: capture_runtime_checkpoint()
    C->>A: atomic_save_checkpoint_files()
    A->>F: temp bundle + temp manifest
    A->>F: os.replace() publish bundle
    A->>F: os.replace() publish manifest
    A->>F: write latest pointer
    Note over E,F: restore later validates file set before loading bundle
    
```
