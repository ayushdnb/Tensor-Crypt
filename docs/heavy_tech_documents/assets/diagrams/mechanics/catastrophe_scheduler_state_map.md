# Catastrophe Scheduler State Map

> Owning document: [Catastrophe system, scheduler, and world overlays](../../../03_mechanics/08_catastrophe_system_scheduler_and_world_overlays.md)

## What this asset shows
- catastrophe mode truth and scheduler-armed / paused semantics

## What this asset intentionally omits
- per-catastrophe parameter details

```mermaid
stateDiagram-v2
    [*] --> Off
    Off --> ManualOnly: cycle_mode
    ManualOnly --> AutoDynamic: cycle_mode
    AutoDynamic --> AutoStatic: cycle_mode
    AutoStatic --> Off: cycle_mode
    AutoDynamic --> AutoDynamicPaused: toggle_scheduler_pause
    AutoDynamicPaused --> AutoDynamic: toggle_scheduler_pause
    AutoStatic --> AutoStaticPaused: toggle_scheduler_pause
    AutoStaticPaused --> AutoStatic: toggle_scheduler_pause
    
```
