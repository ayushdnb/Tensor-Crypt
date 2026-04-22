# D02 - Notation, Glossary, and Shape Legend

## Purpose

This document standardizes terminology, status labels, and tensor-shape notation across the Tensor Crypt technical-document bundle. Its role is to reduce ambiguity, especially where the code distinguishes between identity and storage, canonical and compatibility surfaces, or runtime behavior and background explanation.

## Scope Boundary

This document defines vocabulary and notation. It does not attempt to reteach every subsystem. Detailed subsystem claims remain in the domain chapters.

## Evidence Basis

The terminology below is anchored to the current repository implementation, especially:

- identity and lifecycle code in `tensor_crypt.agents.state_registry`
- runtime assembly in `tensor_crypt.app.launch` and `tensor_crypt.app.runtime`
- world, observation, reproduction, and PPO code in `tensor_crypt.world`, `tensor_crypt.population`, `tensor_crypt.learning`, and `tensor_crypt.simulation`
- checkpoint, telemetry, validation, and viewer code in `tensor_crypt.checkpointing`, `tensor_crypt.telemetry`, `tensor_crypt.audit`, and `tensor_crypt.viewer`

## Status Labels

These labels are used throughout the bundle and should not be paraphrased away when the distinction matters.

| Label | Meaning |
|---|---|
| Active | A direct current read path or behavior path exists. |
| Guarded | The surface is real, but accepted values or supported behavior are narrower than the public name alone suggests. |
| Compatibility surface | The surface exists for import or entry-point continuity while canonical ownership lives elsewhere. |
| Currently unread | The surface is present in configuration or compatibility code, but the current repository does not show a live runtime read path. |
| Adjacent background | The material is explanatory only and does not assert implemented behavior. |

## Shape Legend

The following symbols are used in later chapters unless a local chapter defines additional symbols explicitly.

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `R` | number of rays |
| `F` | per-ray feature width |
| `S` | canonical self-feature width |
| `C` | canonical context-feature width |
| `A` | action dimension |

Preferred shape notation uses bracketed ordered dimensions, for example:

- canonical rays: `[B, R, F]`
- canonical self: `[B, S]`
- canonical context: `[B, C]`
- policy logits: `[B, A]`

## Core Terms

| Term | Bundle meaning | Do not confuse with |
|---|---|---|
| Slot | Dense runtime storage position in the registry substrate. A slot can be reused over time. | UID |
| UID | Canonical monotonic identity for lifecycle, lineage, and PPO ownership. | Slot index, display order |
| Family | Bloodline grouping that selects brain topology and related architecture. | Parent role |
| Brain parent | Parent whose family and brain weights seed the child policy. | Trait parent |
| Trait parent | Parent whose latent trait state seeds the child trait substrate. | Brain parent |
| Anchor parent | Parent whose location is used for offspring placement search. | Brain parent, trait parent |
| Canonical observation surface | The observation bundle keyed by `canonical_rays`, `canonical_self`, and `canonical_context`. | Legacy compatibility bundle |
| Legacy observation bridge | Compatibility path that maps older observation fields into the canonical input contract when fallback is allowed. | Canonical observation surface |
| Compatibility wrapper | Root-level or compatibility-package surface that forwards to canonical `tensor_crypt.*` implementation modules. | Canonical implementation owner |
| Manifest | Checkpoint-side metadata file describing bundle identity, checksums, and schema values. | Latest pointer |
| Latest pointer | Optional locator file that names the latest published checkpoint bundle. | Manifest, checkpoint bundle |
| Validation harness | Determinism, resume-consistency, catastrophe-reproducibility, or soak audit logic used to test runtime behavior. | Runtime behavior itself |

## Preferred Wording

Use the forms below when precision matters.

| Preferred form | Use when |
|---|---|
| UID-owned PPO state | describing buffers, optimizers, or training state keyed by UID |
| Canonical implementation module | naming the `tensor_crypt.*` owner of current logic |
| Compatibility wrapper | naming root or alias surfaces such as `run.py`, `main.py`, `config.py`, `engine.*`, or `viewer.*` |
| No live read path is documented in the current implementation | describing currently unread config or compatibility surfaces |
| Adjacent background only | separating theory from implementation claims |

## Required Distinctions

The following distinctions are load-bearing and should remain explicit.

| Distinction | Why it matters |
|---|---|
| Slot vs UID | Slots are storage positions; UIDs are the identity substrate for lineage, logging, and PPO ownership. |
| Canonical observation surface vs legacy observation bridge | The compatibility bridge is not the authoritative observation contract. |
| Compatibility wrapper vs canonical implementation owner | Public entry points and imports do not by themselves identify where current logic lives. |
| Runtime catastrophe overlay vs inherited trait state | Catastrophe effects are temporary world or runtime modifiers, not lineage-persistent trait values. |
| Validation harness vs runtime behavior | Test and audit probes exercise the system; they do not define everyday operator semantics. |
| Manifest vs latest pointer | They are separate files with different roles and different validation implications. |

## Canonical Observation Naming

When discussing live policy inputs, use the exact implementation keys:

- `canonical_rays`
- `canonical_self`
- `canonical_context`

When discussing the compatibility payload, make the distinction explicit and use the legacy keys only in that context:

- `rays`
- `state`
- `genome`
- `position`
- `context`

## Document References

Later chapters that depend heavily on this terminology include:

- [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md) for slot, UID, family, and parent-role terminology
- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md) for canonical and legacy observation terminology
- [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md) for rollout, bootstrap, and UID-owned learning state
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md) for manifest and latest-pointer terminology
- [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md) for implementation-versus-background boundaries
