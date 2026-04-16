# Package and Compatibility Surface Map

> Owning document: [Package layout, canonical modules, and compatibility wrappers](../../../02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md)

## What this asset shows
- the canonical `tensor_crypt` regions and their responsibilities
- root-level public entry surfaces
- compatibility wrapper direction

## What this asset intentionally omits
- detailed call signatures
- intra-module algorithm details

```mermaid
flowchart TD
    A[run.py] --> B[tensor_crypt.app.launch]
    C[main.py] --> B
    D[config.py] --> E[tensor_crypt.runtime_config / config_bridge]
    F[engine.* wrappers] --> G[tensor_crypt canonical modules]
    B --> H[tensor_crypt.app.runtime]
    H --> I[agents]
    H --> J[world]
    H --> K[learning]
    H --> L[telemetry]
    H --> M[simulation.engine]
    H --> N[viewer]
    H --> O[checkpointing]
    H --> P[validation]
    
```
