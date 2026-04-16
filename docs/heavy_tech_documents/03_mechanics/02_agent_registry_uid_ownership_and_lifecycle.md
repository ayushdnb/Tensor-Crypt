# Agent Registry, UID Ownership, and Lifecycle

> Scope: Document the slot-backed agent registry, the canonical UID lifecycle ledger, and the invariant that slot reuse must never recycle identity ownership.

## Who this document is for
Technical readers, maintainers, and operators who need precise repository behavior rather than general theory.

## What this document covers
- slot tensor layout
- uid lifecycle records
- slot-to-uid binding
- family binding
- trait latent storage
- generation depth
- death finalization
- shadow columns

## What this document does not cover
- broader RL theory unless needed for local context
- general game lore

## Prerequisite reading
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Grid substrate and spatial field model](00_grid_substrate_and_spatial_field_model.md)
- [PPO from first principles to UID-owned rollouts](../01_foundations/07_ppo_from_first_principles_to_uid_owned_rollouts.md)

## 1. Registry purpose

The registry owns the dense per-slot tensor state used for fast runtime access, but its docstring explicitly states that canonical semantics belong to monotonic UIDs. This single distinction controls almost every other subsystem that needs identity-safe behavior.

## 2. Slot columns

The registry stores many per-slot columns, including:
- alive flag
- position
- HP and HP max
- last action
- mass
- vision
- metabolism rate
- UID shadow column
- parent UID shadow column
- tick born
- reward-related accumulators
- optimization cycle marker

The shadow columns exist for compatibility and visibility; they do not replace the canonical UID ledgers.

## 3. UID lifecycle ledger

Each allocated UID receives a lifecycle record with:
- `uid`
- `parent_uid`
- `birth_tick`
- `death_tick`
- `current_slot`
- `is_active`

The registry also maintains:
- `active_uid_to_slot`
- `uid_family`
- `uid_parent_roles`
- `uid_trait_latent`
- `uid_generation_depth`

## 4. Bind and finalize

### Binding
A newly allocated UID is bound to a free slot. The registry checks:
- the UID exists
- the slot is unbound
- the UID is not already active
- the UID is not historical

### Death finalization
When a slot’s live agent dies:
- the UID is marked historical
- `death_tick` is filled
- `current_slot` becomes `None`
- the active map entry is removed
- the slot binding is cleared
- the slot family marker is cleared

> **Invariant**
> Finalizing death retires the slot binding, not the concept of the slot itself. The slot may be reused later; the UID may not.

## 5. Family and trait ownership

Family binding is UID-owned and inherited deliberately. Trait latent state is also UID-owned. The slot only hosts the currently live realization.

## 6. Generation depth

The registry records generation depth per UID. That gives the telemetry and game-manual layers a lineage depth surface without pretending that slot age and lineage depth are the same thing.

## 7. Identity shadow columns

If enabled, the registry mirrors canonical UID and parent UID into legacy float shadow columns. This is a bridge surface for logs, viewers, and compatibility-era visibility—not a replacement ownership model.

## 8. Identity invariants

The registry asserts that:
- every active UID maps to exactly one slot
- every live slot with an active UID hosts a brain with the correct family
- inactive UIDs do not retain live slot bindings
- lifecycle, family, parent-role, and trait-latent ledgers stay aligned

This makes the registry one of the main repository-truth anchors.


## Read next
- [Traits, bloodlines, families, and brain instantiation](03_traits_bloodlines_families_and_brain_instantiation.md)
- [Learning system overview and data ownership](../04_learning/00_learning_system_overview_and_data_ownership.md)
- [Checkpoint-visible learning state and restore order](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## If debugging this, inspect…
- [Validation, determinism, resume consistency, and soak](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Terms introduced here
- `slot`
- `UID`
- `lifecycle record`
- `active UID`
- `historical UID`
- `shadow column`
