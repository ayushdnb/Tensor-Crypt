# Respawn and Lineage Flow

> Owning document: [Reproduction, respawn, mutation, and lineage](../../../03_mechanics/07_reproduction_respawn_mutation_and_lineage.md)

## What this asset shows
- parent-role selection, mutation, placement, and fresh-UID birth

## What this asset intentionally omits
- full overlay policy branches

```mermaid
flowchart TD
    A[Respawn controller step] --> B{Population below ceiling?}
    B -- no --> Z[No birth]
    B -- yes --> C[Select parent roles]
    C --> D[brain parent / trait parent / anchor parent]
    D --> E[Mutate trait latent]
    E --> F[Resolve family inheritance or shift]
    F --> G[Place near anchor or fallback]
    G --> H[Allocate fresh UID]
    H --> I[Bind UID to slot]
    I --> J[Open telemetry life record]

```
