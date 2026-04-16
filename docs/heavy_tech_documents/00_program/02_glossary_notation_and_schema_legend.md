# Glossary, Notation, and Schema Legend

> Scope: Provide canonical definitions for high-load repository terms, the tensor-shape notation used across the corpus, and the schema-version legend used by checkpoint and telemetry documents.

## Who this document is for
All readers, especially anyone jumping directly into deep technical chapters.

## What this document covers
- canonical term definitions
- tensor and shape notation
- repository-specific symbol usage
- schema-version legend
- places where repository meaning differs from ordinary language

## What this document does not cover
- long-form subsystem explanation
- operator procedures

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)

## 1. Term glossary

> **Definition — canonical package**
> The `tensor_crypt` implementation package. It is the architecture center of gravity.

> **Definition — public entry surface**
> A user-facing repository-root launch or import surface such as `run.py`, `main.py`, or `config.py`.

> **Definition — compatibility wrapper**
> A thin re-export surface retained for older import paths. It is not the owning implementation.

> **Definition — runtime assembly**
> Launch-time construction of the subsystem graph: determinism setup, run-directory creation, map generation, initial spawn, engine build, viewer build.

> **Definition — slot**
> A dense runtime storage position in tensor-backed agent state.

> **Definition — UID**
> A monotonic canonical identity that survives slot reuse and owns lineage, family binding, and PPO state.

> **Definition — active UID**
> A UID currently bound to a live slot.

> **Definition — historical UID**
> A UID present in the lifecycle ledger but no longer active in a live slot.

> **Definition — bloodline family**
> A named family category that determines color identity and a fixed MLP topology signature.

> **Definition — trait latent**
> The latent budget-and-logit representation from which exposed trait values are derived.

> **Definition — parent roles**
> The repository distinguishes at least brain parent, trait parent, and anchor parent roles.

> **Definition — canonical observation contract**
> The brain-facing bundle consisting of canonical rays, canonical self features, and canonical context features.

> **Definition — legacy observation fallback**
> The adapter path that derives canonical observations from the older rays/state/genome/position/context surface when fallback is enabled.

> **Definition — bootstrap**
> The value-closure state needed to finish return and advantage calculation for an unfinished rollout tail.

> **Definition — rollout buffer**
> The UID-owned PPO transition store containing observations, actions, log-probs, rewards, values, dones, and staged bootstrap state.

> **Definition — optimizer ownership**
> The rule that optimizer state is keyed by agent UID rather than by slot.

> **Definition — topology signature**
> The ordered parameter-shape signature of a brain family. It is checkpoint-visible and migration-sensitive.

> **Definition — schema version**
> A version number carried for identity, observation, checkpoint, reproduction, catastrophe, telemetry, and logging surfaces.

> **Definition — manifest**
> The metadata file published alongside an atomic checkpoint bundle.

> **Definition — latest pointer**
> The file that points to the most recent published checkpoint bundle and embeds consistency metadata.

> **Definition — determinism probe**
> A validation harness that compares repeated runs under a fixed seed.

> **Definition — resume consistency**
> Validation that a resumed run matches the continued baseline when checkpoint restore is done correctly.

> **Definition — catastrophe mode**
> The scheduler mode controlling world-shock activity: off, manual-only, auto-dynamic, or auto-static.

> **Definition — overlay doctrine**
> A reproduction overlay such as crowding, cooldown, or local-parent selection.

> **Definition — run artifact**
> A durable file or directory emitted during a run, such as parquet ledgers, HDF5 snapshots, JSON lineage export, config snapshot, or checkpoints.


## 2. Tensor and shape notation

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `R` | number of rays in the canonical observation |
| `Fr` | canonical per-ray feature count |
| `Fs` | canonical self-feature count |
| `Fc` | canonical context-feature count |
| `A` | action dimension |
| `V` | value-head width; in the repository this is configured as `1` |
| `[B, R, Fr]` | batched ray tensor |
| `[B, Fs]` | batched self-feature tensor |
| `[B, Fc]` | batched context tensor |
| `[B, A]` | actor logits |
| `[B, V]` | critic value output |

The repository’s canonical observation path uses three tensors rather than a single monolithic feature vector at the interface boundary. Family implementations may flatten or split those internally, but the contract remains three-way at the semantic level.

## 3. Repository-specific meaning differences

| Everyday word | Repository meaning |
| --- | --- |
| identity | Canonical UID ownership, not slot occupancy |
| family | A bloodline topology class, not just a color tag |
| parent | Role-specific parent identity; there are separate brain, trait, and anchor roles |
| bootstrap | A staged rollout tail closure, not a generic startup term |
| manifest | Checkpoint publication metadata, not a generic project manifest |
| latest | The latest checkpoint pointer, not merely the file with the newest timestamp |

## 4. Schema-version legend

The runtime metadata and checkpoint surfaces expose version fields for:
- identity
- observation
- PPO state
- checkpoint
- reproduction
- catastrophe
- telemetry
- logging

A schema version in this corpus means **compatibility-sensitive structural meaning**, not a decorative build number.

## 5. Reading aid for high-load terms

When a technical chapter uses a term heavily, assume the following:
- `slot` refers to fast dense runtime storage
- `UID` refers to canonical ownership and lineage
- `canonical observation` refers to the three-tensor bundle
- `family` implies checkpoint-visible topology rules
- `checkpoint correctness` implies schema validation, manifest handling, ownership checks, and restore ordering

## Read next
- [Agent registry, UID ownership, and lifecycle](../03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md)
- [Observation schema, perception, and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)
- [Checkpoint-visible learning state and restore order](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)

## Related reference
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)
- [Config reference index](../07_reference/00_config_reference_index.md)

## If debugging this, inspect…
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)

## Terms introduced here
- `canonical package`
- `public entry surface`
- `compatibility wrapper`
- `runtime assembly`
- `slot`
- `UID`
- `active UID`
- `historical UID`
- `bloodline family`
- `trait latent`
