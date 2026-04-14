# D22 - State, Identity, Lineage, and Ownership Contracts

## Purpose

This document defines the core ownership contract of Tensor Crypt: dense slots are runtime storage, but canonical identity is UID-based. Later chapters on death, reproduction, PPO ownership, telemetry, and checkpointing all depend on this distinction.

## Scope Boundary

This chapter focuses on state ownership and identity semantics. It does not restate world physics, reproduction heuristics, or optimizer mathematics in full.

## Evidence Basis

This chapter is grounded in:

- `tensor_crypt.agents.state_registry`
- `tensor_crypt.population.evolution`
- `tensor_crypt.population.respawn_controller`
- `tensor_crypt.learning.ppo`
- `tensor_crypt.checkpointing.runtime_checkpoint`
- tests including `tests/test_identity_substrate.py`, `tests/test_registry_respawn.py`, `tests/test_uid_ppo_hardening.py`, and `tests/test_runtime_checkpoint_substrate.py`

## Core Distinction

### Slot

A slot is a dense storage position in the registry substrate. Slots are useful because they support tensor-friendly state layout and can be reused after an agent dies.

### UID

A UID is the canonical monotonic identity token for an agent lifecycle. UIDs are used for:

- lifecycle tracking
- lineage tracking
- family ownership
- PPO buffer ownership
- optimizer ownership
- checkpoint-visible identity state

Slots can be reused. UIDs are not reused.

## Registry Surfaces That Carry the Contract

The current registry maintains the following load-bearing identity structures:

- `slot_uid`
- `slot_parent_uid`
- `active_uid_to_slot`
- `uid_lifecycle`
- `uid_family`
- `uid_parent_roles`
- `uid_trait_latent`
- `uid_generation_depth`
- `next_agent_uid`

The dense runtime state itself lives in `Registry.data`, but that tensor is not the full identity story.

## `AgentLifecycleRecord`

The lifecycle ledger stores `AgentLifecycleRecord` instances with fields:

- `uid`
- `parent_uid`
- `birth_tick`
- `death_tick`
- `current_slot`
- `is_active`

This makes identity historical, not merely present-tense. A dead UID remains represented in the lifecycle ledger even after the slot binding is cleared.

## Birth-Time Ownership

When the registry spawns an agent, it:

1. allocates a fresh UID
2. binds that UID to a slot
3. records family and parent-role ledgers
4. records trait latent and generation depth
5. populates dense runtime state
6. synchronizes legacy shadow columns when enabled

The parent-role substrate is explicit. The registry records:

- brain parent UID
- trait parent UID
- anchor parent UID

This is stricter than a single undifferentiated parent reference.

## Death Finalization

When a death is finalized, the registry:

- writes the death tick to the lifecycle record
- marks the UID inactive
- clears the UID's current slot
- removes the active UID-to-slot binding
- clears slot-local UID and parent-UID bindings
- clears slot-family state

The UID remains in history. Only the live slot binding is cleared.

## Shadow Columns Are Compatibility Mirrors

The registry also maintains shadow columns such as `AGENT_UID_SHADOW` and `PARENT_UID_SHADOW` in the dense data tensor when compatibility mirroring is enabled.

These are not the canonical source of identity truth. They exist so legacy or compatibility-oriented consumers can still observe UID-like values through older dense-column interfaces.

Canonical identity remains in the UID-ledger structures listed above.

## Invariants the Registry Enforces

The registry's invariant checks protect against:

- one UID appearing in more than one live slot
- live-slot bindings that disagree with lifecycle records
- dead or unbound slots still carrying active identity state
- family drift between slot state and UID ledgers
- parent references that point outside the historical UID substrate when corresponding checks are enabled

These checks are part of the implementation contract, not merely a testing convenience.

## Why the Contract Matters Outside the Registry

### PPO ownership

The PPO layer stores buffers, optimizers, and training state by UID. If slot identity were treated as canonical, a newborn agent could inherit rollout or optimizer state that belonged to a dead historical entity.

### Telemetry

Birth, life, death, and lineage exports depend on stable identity across slot reuse. Slot-only interpretation would make longitudinal artifact analysis misleading.

### Checkpoint restore

Checkpoint restore reconstructs registry bindings, active brains, and PPO ownership using the UID substrate. A slot-only ownership model would make faithful restore much harder to validate.

## Relationship to Reproduction

Reproduction operates on top of the UID contract rather than replacing it. Children receive:

- a fresh UID
- explicit parent-role bindings
- family assignment
- trait latent state
- generation-depth state

This is why the lineage chapter can discuss brain parent, trait parent, and anchor parent as separate roles instead of collapsing them into a single abstract parent.

## Relationship to Checkpointing

Runtime checkpoints capture and validate:

- dense registry data
- slot-to-UID bindings
- next UID counter
- lifecycle ledger
- family ledger
- parent-role ledger
- trait-latent ledger
- generation-depth ledger

Checkpoint safety therefore depends on preserving the identity substrate, not merely on serializing tensors with the right shapes.

## Practical Consequences for Readers

Readers should carry the following assumptions into later chapters:

- slot reuse is allowed and expected
- UID reuse is not allowed
- history belongs to UID-ledger structures
- dense columns alone do not define canonical identity
- any artifact or learning surface that claims continuity across death and birth should be interpreted through UID ownership first

## Cross References

- Runtime construction and package map: [D20](./20_project_identity_runtime_boot_and_package_map.md)
- Tick order and death sequencing: [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
- Reproduction and bloodline dynamics: [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md)
- UID-owned PPO state: [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- Run artifacts: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- Checkpoint publication and restore: [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
