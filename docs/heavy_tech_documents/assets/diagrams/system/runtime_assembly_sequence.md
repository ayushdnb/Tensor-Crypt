# Runtime Assembly Sequence

> Owning document: [Runtime assembly, launch sequence, and session graph](../../../02_system/02_runtime_assembly_launch_sequence_and_session_graph.md)

## What this asset shows
- startup ordering from launch to viewer
- stable construction sequence enforced by runtime assembly

## What this asset intentionally omits
- per-tick simulation behavior

```mermaid
sequenceDiagram
    participant U as User / root surface
    participant L as app.launch
    participant R as app.runtime
    participant D as DataLogger
    participant G as Grid
    participant S as Registry
    participant E as Engine
    participant V as Viewer
    U->>L: main()
    L->>R: setup_determinism()
    L->>R: build_runtime(run_dir)
    R->>D: create logger
    R->>G: create grid
    R->>S: create registry
    R->>R: create physics/perception/PPO/evolution
    R->>R: generate walls and H-zones
    R->>S: spawn initial population
    R->>E: construct engine
    R->>V: construct viewer
    L->>V: run()

```
