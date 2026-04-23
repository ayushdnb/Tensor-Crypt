# Collision and Death-Context Flow

> Owning document: [Physics, collisions, damage, healing, and death](../../../03_mechanics/06_physics_collisions_damage_healing_and_death.md)

## What this asset shows
- how collisions and environment effects stage death context before final death processing

## What this asset intentionally omits
- detailed numeric damage formulas

```mermaid
flowchart TD
    A[Physics step] --> B[Wall / ram / contest effects]
    B --> C[Collision log]
    B --> D[Pending death context]
    D --> E[Environment effects]
    E --> F[HP clamp]
    F --> G{HP <= 0?}
    G -- yes --> H[Resolved death context]
    H --> I[registry.mark_dead]
    I --> J[evolution.process_deaths + UID finalization]

```
