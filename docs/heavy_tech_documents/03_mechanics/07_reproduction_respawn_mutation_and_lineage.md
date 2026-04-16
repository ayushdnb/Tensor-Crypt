# Reproduction, Respawn, Mutation, and Lineage

> Scope: Document the binary-parented reproduction system, parent-role separation, overlay doctrines, trait mutation, placement, and extinction-handling surfaces.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- respawn cadence
- population floor and ceiling
- brain/trait/anchor parent roles
- overlay doctrines
- birth placement
- family shift
- lineage consequences

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Agent registry and lifecycle](02_agent_registry_uid_ownership_and_lifecycle.md)
- [Traits, bloodlines, families, and brain instantiation](03_traits_bloodlines_families_and_brain_instantiation.md)

## 1. Binary-parented baseline

The active runtime validates only the `binary_parented` reproduction mode. That is the live semantics documented here.

## 2. Parent roles

The repository separates:
- brain parent
- trait parent
- anchor parent

Those roles can differ. This is a major design choice because it avoids collapsing “inherit architecture”, “inherit latent traits”, and “inherit placement anchor” into one simplistic parent notion.

## 3. Overlay doctrines

The code exposes three named overlays:
- **The Ashen Press** — crowding-gated reproduction
- **The Widow Interval** — UID-scoped cooldown / refractory parent selection
- **The Bloodhold Radius** — local lineage parent-selection overlay

These doctrines can change parent eligibility or placement behavior without changing the baseline binary-parented model.

## 4. Floor recovery and extinction policy

Population floor and ceiling fields govern when recovery is aggressive or blocked. Extinction policy is runtime-validated and currently supports explicit policies rather than silent recovery magic.

## 5. Trait mutation

Births derive child trait latent state from the parent latent, then mutate it with configurable sigma surfaces. A rare-mutation path and a possible family-shift mutation path are both visible in the reproduction helpers.

## 6. Placement

Offspring placement is attempted near the anchor parent using a bounded search. Global fallback can be enabled or disabled, and crowding overlays can block or divert births.

## 7. Lineage consequence

A child receives:
- a fresh monotonic UID
- parent-role records
- a family binding
- trait latent state
- generation depth

No normal birth should be parentless under the coded binary-parented path.


## Read next
- [Catastrophe system, scheduler, and world overlays](08_catastrophe_system_scheduler_and_world_overlays.md)
- [Telemetry ledgers and lineage graph](../05_operations/04_telemetry_ledgers_snapshots_exports_and_lineage_graph.md)

## Related reference
- [Respawn and lineage flow](../assets/diagrams/mechanics/respawn_and_lineage_flow.md)

## If debugging this, inspect…
- [Validation harnesses](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Terms introduced here
- `brain parent`
- `trait parent`
- `anchor parent`
- `overlay doctrine`
- `floor recovery`
- `family shift`
