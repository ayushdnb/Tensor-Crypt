# Troubleshooting Decision Tree

> Owning document: [Troubleshooting and failure atlas](../../../05_operations/08_troubleshooting_and_failure_atlas.md)

## What this asset shows
- a compact first-check tree for common operational failures

## What this asset intentionally omits
- low-level code debugging recipes

```mermaid
flowchart TD
    A[Problem observed] --> B{Launch fails?}
    B -- yes --> C[check device / enum validation / pygame-ce / log path]
    B -- no --> D{Viewer issue?}
    D -- yes --> E[check pause, selection mode, camera fit, panel toggles]
    D -- no --> F{Resume/checkpoint issue?}
    F -- yes --> G[validate bundle + manifest + latest pointer]
    F -- no --> H{Determinism mismatch?}
    H -- yes --> I[inspect RNG capture, restore order, scheduler state]
    H -- no --> J[inspect benchmark interpretation and logging cadence]

```
