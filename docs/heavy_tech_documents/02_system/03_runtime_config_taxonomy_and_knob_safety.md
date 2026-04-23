# Runtime Config Taxonomy and Knob Safety

> Scope: Organize the dataclass-driven runtime configuration surface by semantics and safety class instead of presenting it as one undifferentiated field listing.

## Who this document is for
Operators, maintainers, and auditors who need to know which knobs are active, guarded, experimental, or currently unread.

## What this document covers
- the main config sections
- safety classes for config fields
- selected high-impact knobs by section
- runtime-validated enumerations
- how to read guarded or currently unread fields conservatively

## What this document does not cover
- an exhaustive per-field explanation of every low-impact presentation toggle
- runtime values observed from a specific run

## Prerequisite reading
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Runtime assembly and session graph](02_runtime_assembly_launch_sequence_and_session_graph.md)
- [Config reference index](../07_reference/00_config_reference_index.md)

## 1. Why a taxonomy is necessary

The runtime configuration surface is large enough that a raw dataclass listing is misleading. The repository itself uses comment labels such as:
- active runtime knob
- guarded compatibility surface
- currently unread
- experimental runtime knob
- active safety knob with an explicit dependency constraint

A serious operator manual must preserve those distinctions instead of flattening them into one “settings” story.

## 2. Safety classes used in this corpus

| Safety class | Meaning |
| --- | --- |
| active runtime knob | A field directly intended to alter live runtime behavior |
| guarded compatibility surface | A field exists, but the runtime validation path currently accepts only a constrained subset of values |
| currently unread | A documented field exists, but the current implementation does not expose a direct read on the active path |
| experimental runtime knob | A field gates an optional, explicitly experimental path |
| active safety knob | A field enables or tightens validation or publication guarantees |

## 3. Section summary

| Section | Active-like fields | Guarded fields | Currently unread fields | Notes |
| --- | ---: | ---: | ---: | --- |
| `SimConfig` | 6 | 1 | 1 | semantic owner section |
| `GridConfig` | 3 | 0 | 1 | semantic owner section |
| `MapgenConfig` | 7 | 0 | 0 | semantic owner section |
| `AgentsConfig` | 12 | 7 | 0 | semantic owner section |
| `RespawnConfig` | 27 | 1 | 4 | semantic owner section |
| `TraitsConfig` | 3 | 1 | 1 | semantic owner section |
| `PhysicsConfig` | 6 | 0 | 1 | semantic owner section |
| `PerceptionConfig` | 13 | 0 | 2 | semantic owner section |
| `BrainConfig` | 10 | 0 | 0 | semantic owner section |
| `PPOConfig` | 22 | 1 | 1 | semantic owner section |
| `EvolutionConfig` | 10 | 0 | 2 | semantic owner section |
| `ViewerConfig` | 11 | 0 | 2 | semantic owner section |
| `LogConfig` | 5 | 0 | 0 | semantic owner section |
| `IdentityConfig` | 5 | 0 | 1 | semantic owner section |
| `SchemaConfig` | 8 | 0 | 0 | semantic owner section |
| `CheckpointConfig` | 27 | 0 | 0 | semantic owner section |
| `TelemetryConfig` | 17 | 0 | 0 | semantic owner section |
| `ValidationConfig` | 7 | 0 | 3 | semantic owner section |
| `MigrationConfig` | 5 | 0 | 0 | semantic owner section |
| `CatastropheConfig` | 25 | 0 | 0 | semantic owner section |

## 4. High-impact sections

### 4.1 Simulation and launch
`SimConfig` carries seed, device, dtype, and the experimental family-vmap inference gate. These settings affect determinism, device use, and inference-path eligibility.

### 4.2 Respawn and overlays
`RespawnConfig` and its overlay subconfigs govern population recovery, parent-role selection, floor-recovery behavior, doctrine overlays, and extinction policy. This is one of the most behavior-rich sections in the repository.

### 4.3 Perception and brain
`PerceptionConfig` fixes the canonical feature counts and ray layout. `BrainConfig` fixes family identity, action/value widths, family-order semantics, and family topology specs. These are checkpoint-sensitive surfaces.

### 4.4 PPO
`PPOConfig` governs update cadence, clipping, entropy, reward gating, ownership mode, and validation strictness around buffers.

### 4.5 Checkpoint and validation
`CheckpointConfig` and `ValidationConfig` control publication guarantees, restore strictness, RNG capture, and validation harness enablement.

### 4.6 Catastrophe and viewer
`CatastropheConfig` controls scheduler modes, overlap policy, duration policy, and per-type tables. `ViewerConfig` mostly controls presentation and interaction.

## 5. Runtime-validated enumerations enforced by the current implementation

| Validation surface | Supported values |
| --- | --- |
| `GRID.OVERLAP.MODES` | `last_wins`, `max_abs`, `sum_clamped` |
| `SPAWN.MODES` | `uniform` |
| `METAB.FORMS` | `affine_combo` |
| `INITIAL.FAMILY.ASSIGNMENTS` | `round_robin`, `weighted_random` |
| `RESPAWN.MODES` | `binary_parented` |
| `ANCHOR.PARENT.SELECTORS` | `brain_parent`, `fitter_of_two`, `random_parent`, `trait_parent` |
| `EXTINCTION.POLICIES` | `admin_spawn_defaults`, `fail_run`, `seed_bank_bootstrap` |
| `BIRTH.HP.MODES` | `fraction`, `full` |
| `PPO.OWNERSHIP.MODES` | `uid_strict` |
| `TIE.BREAKERS` | `random_seeded`, `strength_then_lowest_id` |
| `LINEAGE.EXPORT.FORMATS` | `json` |
| `CATASTROPHE.MODES` | `auto_dynamic`, `auto_static`, `manual_only`, `off` |
| `CATASTROPHE.STATIC.ORDERING` | `configured_sequence`, `fixed_priority`, `round_robin` |

## 6.1 SimConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `SEED` | active runtime knob | active runtime knob. Master deterministic seed. The runtime seeds `torch`, `random`, `numpy`, and CUDA seed-all from this value. Change it when you want a different but still re... |
| `DEVICE` | active runtime knob | active runtime knob. Primary execution device string. Typical values are `"cpu"`, `"cuda"`, or an explicit CUDA device such as `"cuda:0"`. Forcing CUDA on a machine without CUDA... |
| `DTYPE` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. Numeric dtype surface for the simulat... |
| `EXPERIMENTAL_FAMILY_VMAP_INFERENCE` | experimental runtime knob | experimental runtime knob. Enables same-family inference batching via torch.func. This remains opt-in because real ROI depends on family bucket sizes and the local PyTorch build... |
| `EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET` | experimental runtime knob | experimental runtime knob. Minimum same-family alive bucket size required before the engine attempts the torch.func fast path. Smaller buckets stay on the canonical per-brain loop. |

## 6.5 RespawnConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `RESPAWN_PERIOD` | active runtime knob | active runtime knob. Minimum tick gap between normal respawn cycles. Lower values make recovery more frequent. |
| `MAX_SPAWNS_PER_CYCLE` | active runtime knob | active runtime knob. Upper bound on births emitted in one respawn cycle. Raise for more aggressive population recovery. |
| `POPULATION_FLOOR` | active runtime knob | active runtime knob. Soft lower population threshold that triggers recovery behavior. If live population falls below this, the controller enters floor-recovery logic. |
| `POPULATION_CEILING` | active runtime knob | active runtime knob. Upper population ceiling for births. No births are emitted once live population is at or above this value. |
| `MODE` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted... |
| `ANCHOR_PARENT_SELECTOR` | active runtime knob | active runtime knob. Placement-anchor selection policy. Supported values in code are `"brain_parent"`, `"trait_parent"`, `"random_parent"`, and `"fitter_of_two"`. This changes w... |
| `EXTINCTION_POLICY` | active runtime knob | active runtime knob. What to do when live population drops below the minimum needed for binary reproduction. Supported policies are `"fail_run"`, `"seed_bank_bootstrap"`, and `"... |
| `BIRTH_HP_MODE` | active runtime knob | active runtime knob. Initial HP policy for newborn agents. Supported values are `"full"` and `"fraction"`. Use `fraction` when you want newborn fragility. |

## 6.8 PerceptionConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `NUM_RAYS` | active runtime knob | active runtime knob. Number of rays cast per observing agent. Increasing this improves directional coverage but raises observation and inference cost. |
| `CANONICAL_RAY_FEATURES` | active runtime knob | active runtime knob. Canonical per-ray feature count. This is a schema-critical tensor dimension; changing it requires synchronized changes throughout perception and brain code. |
| `CANONICAL_SELF_FEATURES` | active runtime knob | active runtime knob. Canonical self-feature count. High-risk schema knob: change only when the full observation pipeline is being migrated. |
| `CANONICAL_CONTEXT_FEATURES` | active runtime knob | active runtime knob. Canonical context-feature count. High-risk schema knob with full observation-contract implications. |
| `RETURN_CANONICAL_OBSERVATIONS` | active runtime knob | active runtime knob. Whether the perception system returns canonical observations by default. Disabling this would push more pressure onto legacy compatibility paths. |

## 6.9 BrainConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `ACTION_DIM` | active runtime knob | active runtime knob. Actor head output width. This must remain aligned with the action semantics expected by the engine. |
| `VALUE_DIM` | active runtime knob | active runtime knob. Critic head output width. The current architecture expects a scalar value head, so the default is `1`. |
| `FAMILY_ORDER` | active runtime knob | active runtime knob. Ordered list of valid bloodline families. Order matters for round-robin assignment and family-aware update ordering. |
| `DEFAULT_FAMILY` | active runtime knob | active runtime knob. Fallback family used when no explicit family is supplied. It must be present in `FAMILY_ORDER`. |
| `INITIAL_FAMILY_ASSIGNMENT` | active runtime knob | active runtime knob. Root-seed family assignment strategy. The code supports `"round_robin"` and `"weighted_random"`. This affects only initial/root assignment, not inherited fa... |
| `ALLOW_LEGACY_OBS_FALLBACK` | active runtime knob | active runtime knob. Whether the brain may adapt legacy observations into canonical form. Disable this only when you want strict canonical-only enforcement. |
| `FAMILY_SPECS` | active runtime knob | active runtime knob. Per-family topology specification bundle. This is one of the most shape-sensitive surfaces in the repository. |

## 6.10 PPOConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `BATCH_SZ` | active runtime knob | active runtime knob. Minimum trajectory length required before a UID buffer is eligible for update. Smaller values update more often but with noisier estimates. |
| `MINI_BATCHES` | active runtime knob | active runtime knob. Number of minibatches carved from each rollout update. Must remain positive and not exceed the effective batch size. |
| `EPOCHS` | active runtime knob | active runtime knob. Maximum number of optimization passes per update. More epochs extract more signal per rollout at the risk of overfitting stale data. |
| `UPDATE_EVERY_N_TICKS` | active runtime knob | active runtime knob. Global cadence used by `should_update()`. Lower values trigger optimizer work more frequently. |
| `OWNERSHIP_MODE` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. UID ownership semantics selector for ... |
| `REWARD_FORM` | active runtime knob | PPO reward surface: - REWARD_FORM selects the base reward shape. The default `sq_health_ratio` preserves the legacy behavior exactly: clamp(HP / max(HP_MAX, 1e-6), 0, 1)^2. - RE... |
| `REWARD_GATE_MODE` | active runtime knob | active runtime knob. Optional reward gating mode. Supported values are `"off"`, `"hp_ratio_min"`, and `"hp_abs_min"`. This controls whether reward is suppressed below a threshold. |
| `REWARD_GATE_THRESHOLD` | active runtime knob | active runtime knob. Inclusive threshold used by the configured reward gate mode. Interpretation depends on `REWARD_GATE_MODE`: normalized ratio vs absolute HP. |

## 6.12 ViewerConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `WINDOW_WIDTH` | active runtime knob | active runtime knob. Initial viewer window width in pixels. Pure presentation knob. |
| `WINDOW_HEIGHT` | active runtime knob | active runtime knob. Initial viewer window height in pixels. Pure presentation knob. |
| `FPS` | active runtime knob | active runtime knob. Viewer frame-rate target. Higher values make the UI smoother but demand more rendering work. |
| `SHOW_CATASTROPHE_PANEL` | active runtime knob | active runtime knob. Whether the viewer shows the catastrophe panel. Presentation-only. |
| `SHOW_CATASTROPHE_OVERLAY` | active runtime knob | active runtime knob. Whether catastrophe overlays are shown in the viewer. Presentation-only. |
| `SHOW_CATASTROPHE_STATUS_IN_HUD` | active runtime knob | active runtime knob. Whether catastrophe state appears in the HUD/status strip. Presentation-only. |

## 6.16 CheckpointConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `ENABLE_SUBSTRATE_CHECKPOINTS` | active runtime knob | active runtime knob. Master enablement flag for substrate-style runtime checkpoints. If disabled, higher-level checkpoint paths should be considered inactive by policy. |
| `SAVE_EVERY_TICKS` | active runtime knob | active runtime knob. Periodic runtime checkpoint cadence. Set `0` to disable scheduled runtime checkpoints. |
| `KEEP_LAST` | active runtime knob | active runtime knob. Retention count for scheduled runtime checkpoints. Use `<= 0` to keep every produced checkpoint. |
| `MANIFEST_ENABLED` | active runtime knob | active runtime knob. Whether manifests are part of the published checkpoint file set. Must remain enabled if strict manifest validation or latest-pointer writing is enabled. |
| `SAVE_CHECKPOINT_MANIFEST` | active runtime knob | active runtime knob. Whether a manifest file is emitted alongside checkpoint bundles. The atomic publish path uses this as a gate. |
| `WRITE_LATEST_POINTER` | active safety knob | active safety knob with an explicit dependency constraint. Whether a `latest` pointer JSON file is maintained. Requires `MANIFEST_ENABLED = True` in the current runtime. |
| `STRICT_MANIFEST_VALIDATION` | active safety knob | active safety knob with an explicit dependency constraint. Whether manifest metadata must validate during load. Requires `MANIFEST_ENABLED = True` in the current runtime. |
| `CAPTURE_RNG_STATE` | active runtime knob | active runtime knob. Whether Python / NumPy / Torch RNG states are captured. Keep enabled for deterministic resume. |
| `CAPTURE_OPTIMIZER_STATE` | active runtime knob | active runtime knob. Whether per-UID optimizer states are serialized. Disable only if you are willing to resume without optimizer continuity. |

## 6.18 ValidationConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `ENABLE_FINAL_AUDIT_HARNESS` | active runtime knob | active runtime knob. Master enablement for the bundled final audit harness. If disabled, the suite reports skipped checks. |
| `ENABLE_DETERMINISM_TESTS` | active runtime knob | active runtime knob. Whether determinism probes are run by the validation suite. Recommended for any serious change touching state evolution. |
| `ENABLE_RESUME_CONSISTENCY_TESTS` | active runtime knob | active runtime knob. Whether resume-consistency probes are run. Critical when checkpointing or identity semantics change. |
| `ENABLE_CATASTROPHE_REPRO_TESTS` | active runtime knob | active runtime knob. Whether catastrophe reproducibility probes are run. Useful when modifying scheduler or catastrophe state surfaces. |
| `ENABLE_SAVE_LOAD_SAVE_TESTS` | active runtime knob | active runtime knob. Whether save-load-save signature checks are run. Useful for checkpoint idempotence auditing. |

## 6.20 CatastropheConfig

| Field | Safety class | Repository meaning |
| --- | --- | --- |
| `ENABLED` | active runtime knob | active runtime knob. Master catastrophe enable switch. Turn this off to remove scheduler-driven world shocks entirely. |
| `DEFAULT_MODE` | active runtime knob | active runtime knob. Default catastrophe scheduler mode. Supported modes are `"off"`, `"manual_only"`, `"auto_dynamic"`, and `"auto_static"`. This determines how a fresh run beg... |
| `MANUAL_TRIGGER_ENABLED` | active runtime knob | active runtime knob. Whether operator-triggered catastrophes are allowed. Viewer/manual control surface. |
| `ALLOW_OVERLAP` | active runtime knob | active runtime knob. Whether multiple catastrophes may overlap in time. Overlapping shocks increase system complexity and severity. |
| `AUTO_DYNAMIC_GAP_MIN_TICKS` | active runtime knob | active runtime knob. Minimum scheduler gap for dynamic auto mode. Lower values create a busier catastrophe cadence. |
| `AUTO_DYNAMIC_GAP_MAX_TICKS` | active runtime knob | active runtime knob. Maximum scheduler gap for dynamic auto mode. Together with the minimum, this defines the random interval range. |
| `AUTO_STATIC_ORDERING_POLICY` | active runtime knob | active runtime knob. Ordering policy for static auto mode. Supported values are `"round_robin"`, `"configured_sequence"`, and `"fixed_priority"`. This changes only which enabled... |
| `TYPE_ENABLED` | active runtime knob | active runtime knob. Per-catastrophe enable table. Disable entries here to remove them from manual and auto selection. |

## 7. Reading rule

> **Do not infer activity from presence alone.**
> A field appearing in a dataclass is not proof that the current runtime reads it on the active path.

## 8. Operational consequences

- Any config touching observation shape, family topology, reward gating, checkpoint schema handling, or runtime validation should be treated as high risk.
- Guarded enum strings are part of the compatibility story, not evidence of fully implemented alternate semantics.
- Comment-labeled dead fields should remain in the reference layer for honesty, but should not dominate narrative chapters.
## Read next
- [Config reference index](../07_reference/00_config_reference_index.md)
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Related reference
- [Operator quickstart and common run modes](../05_operations/00_operator_quickstart_and_common_run_modes.md)
- [Checkpointing, manifests, restore, and latest pointer](../05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md)

## If debugging this, inspect…
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Terms introduced here
- `active runtime knob`
- `guarded compatibility surface`
- `experimental runtime knob`
- `active safety knob`
