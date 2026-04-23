# D11 - Config Reference: Active, Guarded, and Currently Unread Surfaces

## Purpose

This document classifies the public configuration surface in `tensor_crypt.runtime_config` according to current implementation behavior. Its goal is to prevent operators and maintainers from treating every exported dataclass field as equally live, equally supported, or equally safe to edit.

## Scope Boundary

This chapter is a status map, not a duplicate of every config dataclass definition. It highlights the surfaces that materially affect runtime behavior, the surfaces constrained by validation, and the surfaces that are present but not shown to drive the current runtime.

## Evidence Basis

The classifications below were verified against the current repository, with emphasis on:

- `tensor_crypt.runtime_config`
- `tensor_crypt.app.runtime`
- `tensor_crypt.agents.state_registry`
- `tensor_crypt.world.*`
- `tensor_crypt.population.*`
- `tensor_crypt.learning.ppo`
- `tensor_crypt.telemetry.*`
- `tensor_crypt.checkpointing.*`
- `tensor_crypt.audit.final_validation`
- `tensor_crypt.viewer.*`

## Classification Rules

| Label | Meaning |
|---|---|
| Active | A direct runtime or validation read path exists. |
| Guarded | The field is active, but the accepted values or semantics are narrower than the public name suggests. |
| Compatibility surface | The field or module exists to preserve continuity while canonical ownership lives elsewhere. |
| Currently unread | The field is present in public config, but no live runtime or validation path reads it in the current implementation. |

## Operator-Safe Subset

The following knobs are active and comparatively low risk for ordinary operator work:

- `cfg.SIM.SEED`
- `cfg.SIM.MAX_TICKS`
- `cfg.GRID.W`
- `cfg.GRID.H`
- `cfg.MAPGEN.RANDOM_WALLS`
- `cfg.MAPGEN.HEAL_ZONE_COUNT`
- `cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO`
- `cfg.MAPGEN.HEAL_RATE`
- `cfg.AGENTS.N`
- `cfg.RESPAWN.RESPAWN_PERIOD`
- `cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE`
- `cfg.RESPAWN.POPULATION_FLOOR`
- `cfg.RESPAWN.POPULATION_CEILING`
- `cfg.LOG.DIR`
- `cfg.LOG.LOG_TICK_EVERY`
- `cfg.LOG.SNAPSHOT_EVERY`
- `cfg.VIEW.WINDOW_WIDTH`
- `cfg.VIEW.WINDOW_HEIGHT`
- `cfg.VIEW.SHOW_CATASTROPHE_PANEL`
- `cfg.VIEW.SHOW_CATASTROPHE_OVERLAY`
- `cfg.CHECKPOINT.SAVE_EVERY_TICKS`
- `cfg.CHECKPOINT.KEEP_LAST`
- `cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS`
- `cfg.TELEMETRY.PARQUET_BATCH_ROWS`

These fields still affect behavior, but they do not by themselves redefine schema versions, identity ownership, or brain topology.

## Domain-by-Domain Status Map

### Simulation, grid, map generation, and population scale

| Section | Active surfaces | Guarded surfaces | Currently unread notes |
|---|---|---|---|
| `SIM` | `SEED`, `MAX_TICKS`, `REUSE_ACTION_BUFFER`, experimental family-vmap flags | `DTYPE` is restricted to `float32` | `TICKS_PER_SEC` is present but not read by the current runtime |
| `GRID` | `W`, `H`, `HZ_OVERLAP_MODE`, `HZ_SUM_CLAMP`, `HZ_CLEAR_EACH_TICK` | overlap mode has a finite supported set | `EXPOSE_H_GRAD` is present but not shown to drive current rendering or world logic |
| `MAPGEN` | wall count and segment controls, H-zone count, size ratio, and heal rate | none beyond ordinary value sanity | none identified |
| `AGENTS` | `N`, `NO_STACKING` | `SPAWN_MODE` is validated to `uniform` | none identified |

### Respawn, overlays, traits, and evolution

| Section | Active surfaces | Guarded surfaces | Currently unread notes |
|---|---|---|---|
| `RESPAWN` core | cadence, floor, ceiling, extinction policy, placement, birth HP, mutation rates, anchor-parent selector | `MODE` is validated to `binary_parented`; anchor-parent selector accepts only a fixed set | `BRAIN_PARENT_SELECTOR`, `TRAIT_PARENT_SELECTOR`, `FLOOR_RECOVERY_REQUIRE_TWO_PARENTS`, and `ASSERT_BINARY_PARENTING` are present but not read by the current runtime |
| `RESPAWN.OVERLAYS` | crowding, cooldown, local-parent, viewer hotkeys and status exposure | doctrine behavior depends on explicit overlay enablement and policy settings | none identified for the active overlay path; `UNIFIED_UID_POLICY` is active in cooldown logic |
| `TRAITS` | active trait ranges, normalization, mutation, and derived initialization surfaces | `METAB_FORM` is validated to `affine_combo` | `TRAITS.INIT` remains a present public surface, but the current runtime comments and code indicate it is not read directly |
| `EVOL` | fitness decay, reward shaping, parent-ranking inputs used by active evolution code | none identified as value-set guards | `SELECTION` and `FITNESS_TEMP` are present but not read by the current runtime |

### Physics, perception, brains, and PPO

| Section | Active surfaces | Guarded surfaces | Currently unread notes |
|---|---|---|---|
| `PHYS` | movement cost, collision damage, metabolism, poison and contest parameters, tie breaker | `TIE_BREAKER` accepts only the documented supported set | `MOVE_FAIL_COST` is present but not used in the current physics path |
| `PERCEPT` | `NUM_RAYS`, canonical feature counts, `RETURN_CANONICAL_OBSERVATIONS` | canonical feature counts are load-bearing because they define observation shape | `RAY_FIELD_AGG`, `RAY_STEP_SAMPLER`, and `LEGACY_ADAPTER_MODE` are present but not read by the current runtime |
| `BRAIN` | family order, family specs, default family, fallback allowance, architecture parameters | topology-related fields are guarded by downstream shape and checkpoint compatibility | no dormant core field was relied on for this chapter; topology edits remain high risk |
| `PPO` | update cadence, batch size, reward form, bootstrap and validation flags, ownership mode checks | `OWNERSHIP_MODE` is validated to `uid_strict`; reward and gating modes are limited to the implemented sets | `TRACK_TRAINING_STATE` is present, but training state is tracked regardless of that field in the current implementation |

### Viewer, logging, telemetry, and migration

| Section | Active surfaces | Guarded surfaces | Currently unread notes |
|---|---|---|---|
| `VIEW` | window size, overlay defaults, catastrophe visibility, paint-rate step, many rendering toggles | none beyond ordinary viewer semantics | `CELL_SIZE` and `PAINT_BRUSH` are present, but the current viewer computes cell size dynamically and does not read the brush setting |
| `LOG` | directory, print cadence, snapshot cadence | none identified | none identified |
| `TELEMETRY` | deep-ledger toggles, summary cadence, parquet buffering, lineage export, viewer inspector enrichment, catastrophe exposure tracking | `LINEAGE_EXPORT_FORMAT` is currently constrained to `json` | none identified among the principal telemetry controls |
| `MIGRATION` | `LOG_LEGACY_SLOT_FIELDS`, `VIEWER_SHOW_SLOT_AND_UID`, and `VIEWER_SHOW_BLOODLINE` are active | none identified | `LOG_UID_FIELDS` and `REQUIRE_CANONICAL_UID_PATHS` are present in config and metadata posture, but the current runtime does not show a direct behavior branch on them |

### Identity, schema, checkpointing, validation, and catastrophes

| Section | Active surfaces | Guarded surfaces | Currently unread notes |
|---|---|---|---|
| `IDENTITY` | `ASSERT_BINDINGS`, `ASSERT_HISTORICAL_UIDS`, `MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS` | none beyond invariant semantics | `ENABLE_UID_SUBSTRATE`, `OWNERSHIP_MODE`, and `ASSERT_NO_SLOT_OWNERSHIP_LEAK` are present but not read as runtime branching controls in the current code |
| `SCHEMA` | all declared schema-version fields are active as recorded metadata and checkpoint-validation context | schema values are inherently migration-sensitive | none identified |
| `CHECKPOINT` | capture flags, strict validation flags, manifest controls, filename controls, retention, latest-pointer controls | manifest and latest-pointer settings are constrained by atomic-manifest publication requirements | none identified in the main checkpoint path |
| `VALIDATION` | enablement flags and tick budgets used by the final validation suite | probe semantics are guarded by the validation harness, not by ordinary runtime | `VALIDATION_STRICTNESS`, `SAVE_LOAD_SAVE_COMPARE_BUFFERS`, and `STRICT_TELEMETRY_SCHEMA_WRITES` are present but not read by the current validation code |
| `CATASTROPHE` | enablement, modes, scheduler controls, type enables, durations, weights, viewer controls, checkpoint persistence | mode and ordering fields are validated to supported enumerations | none identified among the main catastrophe controls |

## Explicit Guarded Surfaces

The following fields are active, but they should not be described as open-ended tuning knobs:

| Field | Current supported behavior |
|---|---|
| `cfg.SIM.DTYPE` | only `float32` is accepted |
| `cfg.AGENTS.SPAWN_MODE` | only `uniform` is accepted |
| `cfg.RESPAWN.MODE` | only `binary_parented` is accepted |
| `cfg.RESPAWN.ANCHOR_PARENT_SELECTOR` | only the documented selector set is accepted |
| `cfg.TRAITS.METAB_FORM` | only `affine_combo` is accepted |
| `cfg.PHYS.TIE_BREAKER` | only `strength_then_lowest_id` and `random_seeded` are accepted |
| `cfg.PPO.OWNERSHIP_MODE` | only `uid_strict` is accepted |
| `cfg.TELEMETRY.LINEAGE_EXPORT_FORMAT` | only `json` is accepted |
| `cfg.CATASTROPHE.DEFAULT_MODE` | only `off`, `manual_only`, `auto_dynamic`, and `auto_static` are accepted |
| `cfg.CATASTROPHE.AUTO_STATIC_ORDERING_POLICY` | only the documented ordering set is accepted |

## Checkpoint Dependency Constraints

Several checkpoint fields are active only in combination:

- `cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST` currently requires both `cfg.CHECKPOINT.MANIFEST_ENABLED` and `cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED`
- `cfg.CHECKPOINT.STRICT_MANIFEST_VALIDATION` requires manifests to be published on the active save path
- `cfg.CHECKPOINT.WRITE_LATEST_POINTER` likewise requires manifest publication on the active atomic path

These constraints are enforced in `tensor_crypt.app.runtime`, not merely described in prose.

## High-Risk Edit Classes

The following edits are technically possible to write in config but should be treated as architecture or migration work rather than routine tuning:

- canonical observation feature counts
- brain family order or family specs
- schema-version fields
- checkpoint strictness and manifest-publishing posture
- identity-substrate compatibility behavior
- catastrophe type tables and catastrophe parameter bundles

## Practical Guidance

For ordinary operator work, start with [D10](./10_operator_runbook_and_game_manual.md). For identity-sensitive interpretation, continue to [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md). For artifact-sensitive interpretation, continue to [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md) and [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md).
