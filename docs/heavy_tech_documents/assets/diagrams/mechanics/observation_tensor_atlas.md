# Observation Tensor Atlas

> Owning document: [Observation schema, perception, and ray semantics](../../../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)

## What this asset shows
- the canonical observation split into rays, self, and context
- the bridge from canonical to legacy keys

## What this asset intentionally omits
- exact code for every adapter assignment

```mermaid
flowchart LR
    A[Perception ray cast -> canonical_rays [B,R,Fr]]
    B[Self features -> canonical_self [B,Fs]]
    C[Context features -> canonical_context [B,Fc]]
    A --> D[Observation bundle]
    B --> D
    C --> D
    D --> E[canonical keys]
    D --> F[legacy adapter keys]
    F --> G[rays / state / genome / position / context]
    
```
