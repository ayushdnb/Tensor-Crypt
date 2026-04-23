# Config Taxonomy Map

> Owning document: [Runtime config taxonomy and knob safety](../../../02_system/03_runtime_config_taxonomy_and_knob_safety.md)

## What this asset shows
- the main semantic config families
- where high-risk and compatibility-sensitive sections live

## What this asset intentionally omits
- every individual field

```mermaid
flowchart TD
    C[Config]
    C --> S[Sim / device / seed]
    C --> W[World: Grid / Mapgen / Agents]
    C --> R[Respawn / overlays]
    C --> T[Traits]
    C --> P[Physics / Perception]
    C --> B[Brain]
    C --> L[PPO / Evolution]
    C --> O[Viewer / Log / Telemetry]
    C --> I[Identity / Migration]
    C --> K[Checkpoint / Validation / Schema]
    C --> X[Catastrophe]

```
