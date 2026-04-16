# Procedural Map Generation, Walls, and Zones

> Scope: Explain the generated wall and zone geometry that the runtime paints before the initial population is spawned.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- random wall segments
- heal-zone creation
- bounds handling
- why map generation precedes initial spawn

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Grid substrate and spatial field model](00_grid_substrate_and_spatial_field_model.md)

## 1. Wall generation

The `add_random_walls` helper carves one-cell-thick wall segments. The code uses:
- configured wall count
- minimum and maximum segment length
- an avoidance margin
- directional persistence with occasional turns

The result is not arbitrary random noise. It is a bounded, one-cell-thick obstacle process.

## 2. H-zone generation

The `add_random_hzones` helper adds rectangular zones with a configured count, size ratio, and rate. Each zone becomes a persistent entry in the grid’s `hzones` ledger and contributes to the signed field in channel `1`.

## 3. Paint-before-spawn consequence

Map generation happens before initial spawn. This ensures:
- root agents are not spawned into walls
- initial H-zone geometry already exists for the first tick
- the static wall cache used by physics reflects the generated map

## 4. Overlap and repainting

The grid can repaint H-zones from the persistent zone ledger. Catastrophe overlays rely on this because the baseline field is repainted before reversible catastrophe-specific overrides are applied.


## Read next
- [Observation schema, perception, and ray semantics](04_observation_schema_perception_and_ray_semantics.md)
- [Catastrophe system, scheduler, and world overlays](08_catastrophe_system_scheduler_and_world_overlays.md)

## Related reference
- [Package and compatibility surface map](../assets/diagrams/system/package_and_compatibility_surface_map.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `procedural map generation`
- `wall segment`
- `H-zone repaint`
