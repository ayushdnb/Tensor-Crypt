# Tensor Crypt Game Manual and Rulebook

> Scope: Present the simulation as a coherent, rule-driven world in readable language while staying faithful to the repository’s actual mechanics and avoiding fiction or unsupported emergent claims.

## Who this document is for
Public readers, operators, and technical readers who want a behavior-first explanation before reading code-centric deep dives.

## What this document covers
- the board and its fields
- what agents are and how they are distinguished
- movement, collisions, damage, healing, and death
- birth, lineage, and bloodlines
- catastrophes and visible overlays
- how to read the viewer without needing to know the full codebase first

## What this document does not cover
- deep tensor-shape contracts
- checkpoint serialization details
- optimizer and PPO implementation math

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Grid substrate](../03_mechanics/00_grid_substrate_and_spatial_field_model.md)
- [Observation semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)

## 1. The world at a glance

Tensor Crypt is a grid-based living field. The world contains:
- border walls
- procedurally generated internal walls
- signed H-zones that may heal or harm
- live agents occupying cells
- temporary catastrophe overlays that can distort the field or subsystem behavior

The world is updated in ticks. A tick is one full cycle of catastrophe handling, perception, action choice, movement resolution, environment effects, death processing, reproduction, logging, and optional learning updates.

## 2. What an agent is

An agent is simultaneously:
- a live runtime occupant of one slot
- the carrier of a canonical UID identity
- a member of a bloodline family
- the host of inherited and mutated traits such as HP max, mass, vision, and metabolism
- a bearer of lineage information through parent-role records

In the viewer, the most obvious public identity is bloodline color. In the deeper mechanics, UID and family topology are more important.

## 3. The board and its fields

### Walls
Walls are impassable. The outer border is walled by construction. Internal walls are procedurally generated one-cell-thick segments.

### H-zones
H-zones are rectangular field regions with a signed rate:
- positive rates heal
- negative rates harm

When several zones overlap, the configured overlap policy decides how the net field is formed.

### Agents on the board
A live agent occupies a single cell. The runtime stores slot occupancy in the grid, while canonical identity lives in the registry ledgers.

## 4. What agents perceive

Each live agent receives a structured observation:
- radial ray information about emptiness, walls, other agents, and zone-rate exposure
- self features describing health, body, position, age, and current-zone context
- context features summarizing the current live population

This is not a literal camera. It is a compact engineered observation contract.

## 5. What agents can do

The current action surface is discrete and nine-way:
- stay in place
- or move to one of the eight neighboring directions

Agents do not teleport, and a move is not automatically successful. The physics layer arbitrates all intents.

## 6. How movement resolves

A move can lead to several outcomes:
- the target cell is free and the move succeeds
- the move hits a wall and becomes a wall-collision event
- the move runs into an occupied target and becomes a ram path
- several agents want the same cell and enter a contest

Contests are resolved deterministically from a strength calculation and tie-breaker policy. Winners may move. Losers take contest damage.

## 7. How agents live and die

After movement:
- environment effects apply
- H-zones can heal or harm
- metabolism drains HP
- catastrophe modifiers may alter active rates or burdens

Any agent whose HP crosses the death boundary is removed from the live board and later finalized as a historical UID.

## 8. Bloodlines and family identity

Bloodline families are visible in color, but they are more than color. A family also defines the architecture of the policy/value brain. The repository therefore treats bloodline as a structural property, not just a decorative label.

## 9. Birth and replacement

The world can repopulate after deaths. The coded active model is binary-parented. A birth can have:
- a brain parent
- a trait parent
- an anchor parent

Those roles can differ. The child receives a fresh UID, lineage records, a family binding, and mutated trait latent state. A live slot may be reused for the child, but the child is never the same canonical identity as the dead agent that previously occupied that slot.

## 10. Overlay doctrines for reproduction

The repository exposes three named doctrines that can soften, constrain, or localize reproduction:
- **The Ashen Press** — crowding-sensitive birth gating
- **The Widow Interval** — refractory windows on parent reuse
- **The Bloodhold Radius** — local candidate pools around the dead slot

These are not separate game modes in the sense of replacing the whole model. They are overlays on the reproduction path.

## 11. Catastrophes

Catastrophes are temporary world shocks. They can be:
- absent
- manually triggered
- driven automatically through dynamic or static scheduling

A catastrophe may change parts of the field, alter effective vision, change damage or metabolism behavior, or block or soften parts of the reproduction path. When the effect ends, the system returns to the baseline substrate.

## 12. How to read the viewer

The viewer lets an operator:
- pan and zoom the world
- pause and step
- inspect a chosen agent or H-zone
- view family counts
- watch catastrophe state
- toggle overlay visibility
- trigger or clear catastrophes
- toggle reproduction overlay overrides

The viewer is therefore part rulebook, part debugging console, and part operator console.

## 13. Lineage and bloodline reading notes {#lineage-and-bloodline-reading-notes}

A useful reading discipline is:
- **slot** tells you where something is currently hosted
- **UID** tells you who that agent is canonically
- **family** tells you what topology class it belongs to
- **parent roles** tell you where its architecture, traits, and placement anchor came from conceptually

## 14. What this manual intentionally does not claim

This manual does **not** claim:
- that any bloodline is superior
- that the system necessarily converges to a particular strategy
- that the simulation proves emergence in a scientific sense
- that validation coverage proves research outcomes

It describes rules and visible behavior only.
## Read next
- [Grid substrate and spatial field model](../03_mechanics/00_grid_substrate_and_spatial_field_model.md)
- [Physics, collisions, damage, healing, and death](../03_mechanics/06_physics_collisions_damage_healing_and_death.md)
- [Viewer UI controls, HUD, and inspector manual](../05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md)

## Related reference
- [Observation schema, perception, and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)
- [Reproduction, respawn, mutation, and lineage](../03_mechanics/07_reproduction_respawn_mutation_and_lineage.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `tick`
- `bloodline`
- `overlay doctrine`
- `catastrophe`
- `lineage`
