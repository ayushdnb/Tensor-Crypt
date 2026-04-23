# Action-Resolution Flow

> Owning document: [Action surface, intents, and move resolution](../../../03_mechanics/05_action_surface_intents_and_move_resolution.md)

## What this asset shows
- how actions become intents, contests, rams, or approved moves

## What this asset intentionally omits
- reward and PPO storage after movement

```mermaid
flowchart TD
    A[Sampled actions] --> B[Build intents]
    B --> C{Wall?}
    C -- yes --> D[Wall collision path]
    C -- no --> E{Occupied target?}
    E -- yes --> F[Ram or contest staging]
    E -- no --> G[Proposed move]
    F --> H{Multiple contenders?}
    H -- yes --> I[Contest resolution]
    H -- no --> J[Block / ram]
    G --> K[Resolution cache]
    I --> K
    K --> L[Successful moves]
    K --> M[Blocked moves]

```
