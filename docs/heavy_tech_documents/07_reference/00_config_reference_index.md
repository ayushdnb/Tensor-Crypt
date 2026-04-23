# Config Reference Index

> Scope: Provide a lookup-first index of configuration sections, their safety classes, and selected field meanings derived from the runtime configuration dataclasses.

## Who this document is for
Operators, maintainers, and auditors looking up config surfaces directly rather than reading narrative docs first.

## What this document covers
- section-by-section config reference
- safety class summaries
- selected field notes
- how to interpret guarded and unread fields

## What this document does not cover
- long-form conceptual explanations already covered in the system layer

## Prerequisite reading
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Reading rule

Use this file for lookup, not for first exposure. If a field touches identity, observation shape, family topology, PPO ownership, or checkpoint publication, jump from here into the owning narrative document.

## 2. Section index

| Config class | Primary semantic role |
| --- | --- |
| `SimConfig` | seed, device, dtype, experimental inference path |
| `GridConfig` | field overlap and H-zone behavior |
| `MapgenConfig` | procedural walls and heal zones |
| `AgentsConfig` | population and spawn constraints |
| `RespawnConfig` | birth cadence, parent roles, extinction, overlays |
| `TraitsConfig` | latent budget and trait clamps |
| `PhysicsConfig` | damage constants and tie-breaker policy |
| `PerceptionConfig` | ray count and observation feature counts |
| `BrainConfig` | family order, colors, topology specs |
| `PPOConfig` | rollout/update settings and reward gate |
| `EvolutionConfig` | mutation and noise settings |
| `ViewerConfig` | window, panel, and presentation controls |
| `LogConfig` | log root, cadence, assertions, AMP |
| `IdentityConfig` | UID substrate and bridge visibility |
| `SchemaConfig` | schema version numbers |
| `CheckpointConfig` | capture, manifest, validation, latest pointer |
| `TelemetryConfig` | ledger cadence, export, buffering |
| `ValidationConfig` | audit harness toggles |
| `MigrationConfig` | legacy-versus-canonical visibility surfaces |
| `CatastropheConfig` | scheduler modes, durations, type tables |

<details>
<summary><strong>SimConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `SEED` | `int` | active runtime knob | active runtime knob. Master deterministic seed. The runtime seeds `torch`, `random`, `numpy`, and CUDA seed-all from this value. Change it when you want a di... |
| `DEVICE` | `str` | active runtime knob | active runtime knob. Primary execution device string. Typical values are `"cpu"`, `"cuda"`, or an explicit CUDA device such as `"cuda:0"`. Forcing CUDA on a ... |
| `DTYPE` | `str` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. Numeric dtype sur... |
| `TICKS_PER_SEC` | `int` | currently unread | currently unread. The current runtime does not consume this pacing hint. |
| `MAX_TICKS` | `int` | active runtime knob | active runtime knob. Optional automatic stop threshold. Set `0` to keep the session open until the operator exits. Set a positive integer to stop after that ... |
| `REUSE_ACTION_BUFFER` | `bool` | active runtime knob | active runtime knob. Controls reuse of the dense action scratch tensor. Keeping this enabled reduces per-tick allocations and is the efficient default. |
| `EXPERIMENTAL_FAMILY_VMAP_INFERENCE` | `bool` | experimental runtime knob | experimental runtime knob. Enables same-family inference batching via torch.func. This remains opt-in because real ROI depends on family bucket sizes and the... |
| `EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET` | `int` | experimental runtime knob | experimental runtime knob. Minimum same-family alive bucket size required before the engine attempts the torch.func fast path. Smaller buckets stay on the ca... |

</details>

<details>
<summary><strong>GridConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `HZ_OVERLAP_MODE` | `str` | active runtime knob | active runtime knob. Heal/harm zone overlap-composition mode. Documented options are `"max_abs"`, `"sum_clamped"`, and `"last_wins"`. Use caution: this chang... |
| `HZ_SUM_CLAMP` | `float` | active runtime knob | active runtime knob. Absolute clamp used when summed zone fields are limited. Increase it to allow stronger accumulated zone intensity; decrease it to keep t... |
| `HZ_CLEAR_EACH_TICK` | `bool` | active runtime knob | active runtime knob. Whether the zone field is rebuilt / cleared each tick before applying transient effects. Keeping this true is the safer baseline for rev... |
| `EXPOSE_H_GRAD` | `bool` | currently unread | currently unread. The current runtime does not read this diagnostic exposure switch on the live rendering or world path. |

</details>

<details>
<summary><strong>MapgenConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `RANDOM_WALLS` | `int` | active runtime knob | active runtime knob. Number of random wall segments requested during map generation. Higher values generally create a more obstructed arena. |
| `WALL_SEG_MIN` | `int` | active runtime knob | active runtime knob. Minimum wall segment length. Raise this for longer, more imposing wall runs. |
| `WALL_SEG_MAX` | `int` | active runtime knob | active runtime knob. Maximum wall segment length. Raise carefully: very large values can over-constrain the map, especially on smaller grids. |
| `WALL_AVOID_MARGIN` | `int` | active runtime knob | active runtime knob. Margin used when keeping walls away from protected regions / edges during generation. Larger values reserve more breathing room around t... |
| `HEAL_ZONE_COUNT` | `int` | active runtime knob | active runtime knob. Number of heal/harm zones requested during procedural generation. Higher counts create a busier field landscape. |
| `HEAL_ZONE_SIZE_RATIO` | `float` | active runtime knob | active runtime knob. Zone size as a fraction-like ratio of map size. Increase for larger individual zones; decrease for smaller patches. |
| `HEAL_RATE` | `float` | active runtime knob | active runtime knob. Base positive heal-zone rate used by generated zones. Higher values make positive zones more influential. |

</details>

<details>
<summary><strong>AgentsConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `SPAWN_MODE` | `str` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. Initial spawn pat... |
| `NO_STACKING` | `bool` | active runtime knob | active runtime knob. Whether multiple live agents may occupy the same tile. Keeping this true enforces single-occupancy invariants and grid consistency checks. |
| `ENABLED` | `bool` | active runtime knob | active runtime knob. Master enable for The Ashen Press. When enabled, anchor-local crowding can block or divert births. |
| `LOCAL_RADIUS` | `int` | active runtime knob | active runtime knob. Chebyshev radius used to count live neighbors around the anchor parent. |
| `MAX_NEIGHBORS` | `int` | active runtime knob | active runtime knob. Births are considered crowded once this many live non-anchor neighbors are present in the anchor neighborhood. |
| `POLICY_WHEN_CROWDED` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "block_birth" and "global_only". |
| `BELOW_FLOOR_POLICY` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "strict", "bypass", and "global_only". This is the floor-recovery softening surface for The Ashen Press. |
| `ENABLED` | `bool` | active runtime knob | active runtime knob. Master enable for The Widow Interval. |
| `DURATION_TICKS` | `int` | active runtime knob | active runtime knob. Tick duration for parent-UID refractory windows. |
| `APPLY_TO_BRAIN_PARENT` | `bool` | active runtime knob | active runtime knob. Whether the doctrine applies to the brain parent role. |
| `APPLY_TO_TRAIT_PARENT` | `bool` | active runtime knob | active runtime knob. Whether the doctrine applies to the trait parent role. |
| `APPLY_TO_ANCHOR_PARENT` | `bool` | active runtime knob | active runtime knob. Whether the doctrine applies to the anchor parent role. |
| `UNIFIED_UID_POLICY` | `bool` | active runtime knob | active runtime knob. When true, any cooled UID is treated as cooled for all enabled roles. When false, each role keeps its own refractory ledger. |
| `EMPTY_POOL_POLICY` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "allow_best_available" and "strict". |
| `BELOW_FLOOR_POLICY` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "allow_best_available", "bypass", and "strict". |
| `ENABLED` | `bool` | active runtime knob | active runtime knob. Master enable for The Bloodhold Radius. |
| `SELECTION_RADIUS` | `int` | active runtime knob | active runtime knob. Chebyshev radius around the dead slot used to build the local candidate pool. |
| `FALLBACK_BEHAVIOR` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "global" and "strict". |
| `BELOW_FLOOR_POLICY` | `str` | guarded compatibility surface | guarded compatibility surface. Supported values are "prefer_local_then_global", "bypass", and "strict". |
| `HOTKEYS_ENABLED` | `bool` | unspecified |  |
| `SHOW_STATUS_IN_HUD` | `bool` | unspecified |  |
| `SHOW_STATUS_IN_PANEL` | `bool` | unspecified |  |
| `SHOW_OVERRIDE_MARKERS` | `bool` | unspecified |  |
| `CROWDING` | `RespawnCrowdingOverlayConfig` | unspecified |  |
| `COOLDOWN` | `RespawnCooldownOverlayConfig` | unspecified |  |
| `LOCAL_PARENT` | `RespawnLocalParentOverlayConfig` | unspecified |  |
| `VIEWER` | `RespawnOverlayViewerConfig` | unspecified |  |

</details>

<details>
<summary><strong>RespawnConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `RESPAWN_PERIOD` | `int` | active runtime knob | active runtime knob. Minimum tick gap between normal respawn cycles. Lower values make recovery more frequent. |
| `MAX_SPAWNS_PER_CYCLE` | `int` | active runtime knob | active runtime knob. Upper bound on births emitted in one respawn cycle. Raise for more aggressive population recovery. |
| `POPULATION_FLOOR` | `int` | active runtime knob | active runtime knob. Soft lower population threshold that triggers recovery behavior. If live population falls below this, the controller enters floor-recove... |
| `POPULATION_CEILING` | `int` | active runtime knob | active runtime knob. Upper population ceiling for births. No births are emitted once live population is at or above this value. |
| `MODE` | `str` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than be... |
| `BRAIN_PARENT_SELECTOR` | `str` | currently unread | currently unread. Parent-role selection is governed by the active binary-parent path and overlay logic rather than this legacy selector field. |
| `TRAIT_PARENT_SELECTOR` | `str` | currently unread | currently unread. Parent-role selection is governed by the active binary-parent path and overlay logic rather than this legacy selector field. |
| `ANCHOR_PARENT_SELECTOR` | `str` | active runtime knob | active runtime knob. Placement-anchor selection policy. Supported values in code are `"brain_parent"`, `"trait_parent"`, `"random_parent"`, and `"fitter_of_t... |
| `BRAIN_PARENT_MIN_FITNESS` | `float` | active runtime knob | active runtime knob. Minimum fitness threshold for brain-parent eligibility under normal recovery. Lower it to widen parent eligibility; raise it to demand m... |
| `TRAIT_PARENT_MIN_HP_RATIO` | `float` | active runtime knob | active runtime knob. Minimum normalized HP ratio required for trait-parent eligibility under normal recovery. Higher values bias births toward healthier trai... |
| `TRAIT_PARENT_MIN_AGE_TICKS` | `int` | active runtime knob | active runtime knob. Minimum age in ticks required for trait-parent eligibility. Use this to prevent extremely young agents from donating traits. |
| `FLOOR_RECOVERY_SUSPEND_THRESHOLDS` | `bool` | active runtime knob | active runtime knob. Whether parent-eligibility thresholds are suspended during floor recovery. Keeping this true makes emergency recovery more permissive. |
| `FLOOR_RECOVERY_REQUIRE_TWO_PARENTS` | `bool` | currently unread | currently unread. Floor-recovery behavior does not branch on this field in the current controller. |
| `OVERLAYS` | `RespawnOverlayConfig` | active runtime knob | active runtime knob. Structured overlay doctrines layered on top of the binary reproduction substrate. These overlays do not replace the parent-role architec... |
| `EXTINCTION_POLICY` | `str` | active runtime knob | active runtime knob. What to do when live population drops below the minimum needed for binary reproduction. Supported policies are `"fail_run"`, `"seed_bank... |
| `EXTINCTION_BOOTSTRAP_SPAWNS` | `int` | active runtime knob | active runtime knob. How many bootstrap agents to spawn under extinction-recovery policies. Ignored when `EXTINCTION_POLICY` is `fail_run`. |
| `EXTINCTION_BOOTSTRAP_FAMILY` | `str` | active runtime knob | active runtime knob. Family assigned to bootstrap spawns created by extinction recovery. Choose a valid family name from `BRAIN.FAMILY_ORDER`. |
| `OFFSPRING_JITTER_RADIUS_MIN` | `int` | active runtime knob | active runtime knob. Minimum ring radius used when searching near the anchor parent for placement. Larger values push children farther from the anchor. |
| `OFFSPRING_JITTER_RADIUS_MAX` | `int` | active runtime knob | active runtime knob. Maximum ring radius used for local offspring placement search. Larger values widen the local search envelope. |
| `OFFSPRING_MAX_PLACEMENT_ATTEMPTS` | `int` | active runtime knob | active runtime knob. Hard cap on local placement attempts before fallback / failure. Higher values search harder but increase spawn-time cost. |
| `ALLOW_FALLBACK_GLOBAL_PLACEMENT` | `bool` | active runtime knob | active runtime knob. Whether to search globally if local anchor placement fails. Disabling this makes spawn locality stricter but can increase failed births. |
| `DISALLOW_SPAWN_ON_WALL` | `bool` | active runtime knob | active runtime knob. Whether births may occur on wall tiles. Keeping this true preserves physical validity. |
| `DISALLOW_SPAWN_ON_OCCUPIED` | `bool` | active runtime knob | active runtime knob. Whether births may occur on already occupied tiles. Keeping this true helps maintain no-stacking guarantees. |
| `DISALLOW_SPAWN_IN_HARM_ZONE` | `bool` | active runtime knob | active runtime knob. Whether births may occur in negative zone tiles. Keeping this true avoids immediately hostile spawn sites. |
| `BIRTH_HP_MODE` | `str` | active runtime knob | active runtime knob. Initial HP policy for newborn agents. Supported values are `"full"` and `"fraction"`. Use `fraction` when you want newborn fragility. |
| `BIRTH_HP_FRACTION` | `float` | active runtime knob | active runtime knob. Fraction of `hp_max` used when `BIRTH_HP_MODE` is `fraction`. Values are logically intended to live in `[0, 1]`. |
| `LOG_PLACEMENT_FAILURES` | `bool` | active runtime knob | active runtime knob. Whether failed placement attempts are emitted to telemetry/logging. Useful during debugging crowded maps or strict spawn constraints. |
| `ASSERT_BINARY_PARENTING` | `bool` | currently unread | currently unread. Binary parenting is enforced by the active reproduction path and validation logic rather than by this field. |
| `INIT_BUDGET` | `float` | active runtime knob | active runtime knob. TRAITS.INIT_BUDGET operator knob. |
| `INIT_LOGITS` | `List[float]` | active runtime knob | active runtime knob. TRAITS.INIT_LOGITS operator knob. |
| `MIN_BUDGET` | `float` | active runtime knob | active runtime knob. TRAITS.MIN_BUDGET operator knob. |
| `MAX_BUDGET` | `float` | active runtime knob | active runtime knob. TRAITS.MAX_BUDGET operator knob. |

</details>

<details>
<summary><strong>TraitsConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `INIT` | `TraitInit` | currently unread | currently unread. Legacy initialization container retained for compatibility and documentation; live births use latent decoding and active trait-budget surfaces. |
| `CLAMP` | `TraitClamp` | active runtime knob | active runtime knob. Hard trait clamp bundle. This is active and used when latent traits are decoded into realized trait values. |
| `BUDGET` | `TraitBudgetConfig` | active runtime knob | active runtime knob. Trait-budget control bundle. This is active and shapes both initialization and mutation of the latent trait budget. |
| `METAB_FORM` | `str` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. Metabolism formul... |
| `METAB_COEFFS` | `Dict[str, float]` | active runtime knob | active runtime knob. Coefficient dictionary used by the active affine metabolism formula. Changing these values changes the metabolic burden associated with ... |

</details>

<details>
<summary><strong>PhysicsConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `K_WALL_PENALTY` | `float` | active runtime knob | active runtime knob. Penalty or damage scale associated with wall interaction. Increase to punish wall contact more harshly. |
| `K_RAM_PENALTY` | `float` | active runtime knob | active runtime knob. Penalty scale associated with ram/collision events. Increase to make collision-heavy behavior more costly. |
| `K_IDLE_HIT_PENALTY` | `float` | active runtime knob | active runtime knob. Penalty applied when the relevant idle-hit condition is triggered. Raise for stronger discouragement of that condition. |
| `K_WINNER_DAMAGE` | `float` | active runtime knob | active runtime knob. Damage applied to the contest winner in asymmetric combat resolution. Higher values make even winning fights more expensive. |
| `K_LOSER_DAMAGE` | `float` | active runtime knob | active runtime knob. Damage applied to the contest loser. Higher values make losing engagements more punishing. |
| `MOVE_FAIL_COST` | `float` | currently unread | currently unread. The current physics path does not consume this compatibility penalty field. |
| `TIE_BREAKER` | `str` | active runtime knob | active runtime knob. Combat tie-break policy after primary strength ordering. The configured default is `"strength_then_lowest_id"`. Changing it changes dete... |

</details>

<details>
<summary><strong>PerceptionConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `NUM_RAYS` | `int` | active runtime knob | active runtime knob. Number of rays cast per observing agent. Increasing this improves directional coverage but raises observation and inference cost. |
| `RAY_FIELD_AGG` | `str` | currently unread | currently unread. The active perception path does not branch on this field. |
| `RAY_STEP_SAMPLER` | `str` | currently unread | currently unread. The active perception path does not branch on this field. |
| `CANONICAL_RAY_FEATURES` | `int` | active runtime knob | active runtime knob. Canonical per-ray feature count. This is a schema-critical tensor dimension; changing it requires synchronized changes throughout percep... |
| `CANONICAL_SELF_FEATURES` | `int` | active runtime knob | active runtime knob. Canonical self-feature count. High-risk schema knob: change only when the full observation pipeline is being migrated. |
| `CANONICAL_CONTEXT_FEATURES` | `int` | active runtime knob | active runtime knob. Canonical context-feature count. High-risk schema knob with full observation-contract implications. |
| `LEGACY_RAY_FEATURES` | `int` | active runtime knob | active runtime knob. Legacy bridge per-ray feature count. Used by the legacy-to-canonical adapter for compatibility. |
| `LEGACY_STATE_FEATURES` | `int` | active runtime knob | active runtime knob. Legacy state-vector width. Used only while bridging legacy observations into the canonical schema. |
| `LEGACY_GENOME_FEATURES` | `int` | active runtime knob | active runtime knob. Legacy genome-vector width used by the adapter. Changing it without changing the adapter will break legacy observation bridging. |
| `LEGACY_POSITION_FEATURES` | `int` | active runtime knob | active runtime knob. Legacy position-vector width used by the adapter. This is part of the compatibility surface. |
| `LEGACY_CONTEXT_FEATURES` | `int` | active runtime knob | active runtime knob. Legacy context-vector width used by the adapter. Keep aligned with any legacy observation producer still in use. |
| `LEGACY_ADAPTER_MODE` | `str` | active runtime knob | active runtime knob. Legacy adapter identity string. The configured default names the bridge implementation: `"canonical_bridge_v1"`. Treat this prim... |
| `ZONE_RATE_ABS_MAX` | `float` | active runtime knob | active runtime knob. Normalization ceiling for zone-rate features. Higher values make the normalized observation less sensitive to moderate field strengths. |
| `AGE_NORM_TICKS` | `int` | active runtime knob | active runtime knob. Age normalization denominator in ticks. Larger values make age-related features saturate more slowly. |
| `RETURN_CANONICAL_OBSERVATIONS` | `bool` | active runtime knob | active runtime knob. Whether the perception system returns canonical observations by default. Disabling this would push more pressure onto legacy compatibili... |

</details>

<details>
<summary><strong>BrainConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ACTION_DIM` | `int` | active runtime knob | active runtime knob. Actor head output width. This must remain aligned with the action semantics expected by the engine. |
| `VALUE_DIM` | `int` | active runtime knob | active runtime knob. Critic head output width. The current architecture expects a scalar value head, so the default is `1`. |
| `FAMILY_ORDER` | `List[str]` | active runtime knob | active runtime knob. Ordered list of valid bloodline families. Order matters for round-robin assignment and family-aware update ordering. |
| `DEFAULT_FAMILY` | `str` | active runtime knob | active runtime knob. Fallback family used when no explicit family is supplied. It must be present in `FAMILY_ORDER`. |
| `INITIAL_FAMILY_ASSIGNMENT` | `str` | active runtime knob | active runtime knob. Root-seed family assignment strategy. The code supports `"round_robin"` and `"weighted_random"`. This affects only initial/root assignme... |
| `INITIAL_FAMILY_WEIGHTS` | `Dict[str, float]` | active runtime knob | active runtime knob. Weight table used when `INITIAL_FAMILY_ASSIGNMENT` is `weighted_random`. Weights must sum to a positive value. |
| `LEGACY_TRANSFORMER_FALLBACK_ENABLED` | `bool` | active runtime knob | active runtime knob. Legacy transformer fallback toggle. Keep disabled unless you are deliberately resurrecting that compatibility surface. |
| `ALLOW_LEGACY_OBS_FALLBACK` | `bool` | active runtime knob | active runtime knob. Whether the brain may adapt legacy observations into canonical form. Disable this only when you want strict canonical-only enforcement. |
| `FAMILY_COLORS` | `Dict[str, List[int]]` | active runtime knob | active runtime knob. Viewer / UI color mapping per family. Each family name should map to an RGB triplet-like list of three integers. |
| `FAMILY_SPECS` | `Dict[str, BloodlineFamilySpec]` | active runtime knob | active runtime knob. Per-family topology specification bundle. This is one of the most shape-sensitive surfaces in the repository. |

</details>

<details>
<summary><strong>PPOConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `GAMMA` | `float` | active runtime knob | active runtime knob. Reward discount factor. Higher values value longer-horizon returns more heavily. |
| `LAMBDA` | `float` | active runtime knob | active runtime knob. GAE lambda. Higher values reduce bias and increase variance in the usual PPO tradeoff. |
| `CLIP_EPS` | `float` | active runtime knob | active runtime knob. PPO clipping epsilon. Larger values permit more aggressive policy movement per update. |
| `ENTROPY_COEF` | `float` | active runtime knob | active runtime knob. Entropy bonus coefficient. Raise it to encourage more exploration. |
| `VALUE_COEF` | `float` | active runtime knob | active runtime knob. Critic loss coefficient in the combined PPO objective. Higher values emphasize value fitting relative to policy loss. |
| `LR` | `float` | active runtime knob | active runtime knob. Adam learning rate for per-UID optimizers. Too high destabilizes updates; too low slows learning. |
| `BATCH_SZ` | `int` | active runtime knob | active runtime knob. Minimum trajectory length required before a UID buffer is eligible for update. Smaller values update more often but with noisier estimates. |
| `MINI_BATCHES` | `int` | active runtime knob | active runtime knob. Number of minibatches carved from each rollout update. Must remain positive and not exceed the effective batch size. |
| `EPOCHS` | `int` | active runtime knob | active runtime knob. Maximum number of optimization passes per update. More epochs extract more signal per rollout at the risk of overfitting stale data. |
| `TARGET_KL` | `float` | active runtime knob | active runtime knob. Early-stop KL threshold. Set positive to stop epochs early when policy drift exceeds this level. |
| `GRAD_NORM_CLIP` | `float` | active runtime knob | active runtime knob. Global gradient norm clip. Lower it for stricter update bounding. |
| `REWARD_FORM` | `str` | active runtime knob | PPO reward surface: - REWARD_FORM selects the base reward shape. The default `sq_health_ratio` preserves the legacy behavior exactly: clamp(HP / max(HP_MAX, ... |
| `REWARD_GATE_MODE` | `str` | active runtime knob | active runtime knob. Optional reward gating mode. Supported values are `"off"`, `"hp_ratio_min"`, and `"hp_abs_min"`. This controls whether reward is suppres... |
| `REWARD_GATE_THRESHOLD` | `float` | active runtime knob | active runtime knob. Inclusive threshold used by the configured reward gate mode. Interpretation depends on `REWARD_GATE_MODE`: normalized ratio vs absolute HP. |
| `REWARD_BELOW_GATE_VALUE` | `float` | active runtime knob | active runtime knob. Reward emitted when the gate condition is not met. Leave at `0.0` for pure threshold suppression below the gate. |
| `UPDATE_EVERY_N_TICKS` | `int` | active runtime knob | active runtime knob. Global cadence used by `should_update()`. Lower values trigger optimizer work more frequently. |
| `OWNERSHIP_MODE` | `str` | guarded compatibility surface | guarded compatibility surface. Non-supported non-default values are rejected during runtime validation rather than being silently accepted. UID ownership sem... |
| `BUFFER_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Schema version for serialized PPO buffers. Change only as part of a deliberate compatibility migration. |
| `TRACK_TRAINING_STATE` | `bool` | currently unread | currently unread. Training state is tracked by the live PPO implementation without consulting this flag. |
| `FAMILY_AWARE_UPDATE_ORDERING` | `bool` | active runtime knob | active runtime knob. Whether ready PPO updates are grouped / ordered by family. This changes scheduling order, not the per-UID ownership model. |
| `REQUIRE_BOOTSTRAP_FOR_ACTIVE_BUFFER` | `bool` | active runtime knob | active runtime knob. Whether non-terminal active buffers must carry staged bootstrap state. Keeping this true makes rollout finalization safer and stricter. |
| `COUNT_TRUNCATED_ROLLOUTS` | `bool` | active runtime knob | active runtime knob. Whether dropped non-terminal buffers increment truncated-rollout counters. Useful for diagnostics when buffers are cleared early. |
| `DROP_INACTIVE_UID_BUFFERS_AFTER_FINALIZATION` | `bool` | active runtime knob | active runtime knob. Whether finalized inactive UID buffers are removed from memory. Keeping this true avoids stale buffer buildup. |
| `STRICT_BUFFER_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether buffer structure / finiteness is validated before update. Disable only if you knowingly trade safety for speed. |

</details>

<details>
<summary><strong>EvolutionConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `SELECTION` | `str` | currently unread | currently unread. Current evolution logic does not branch on this selector field. |
| `FITNESS_DECAY` | `float` | active runtime knob | active runtime knob. Decay factor applied to stored fitness across death processing. Lower values forget history faster; higher values preserve it longer. |
| `POLICY_NOISE_SD` | `float` | active runtime knob | active runtime knob. Standard deviation of ordinary policy-parameter noise applied at birth. Raise for more exploration / lineage drift. |
| `FITNESS_TEMP` | `float` | currently unread | currently unread. Current evolution logic does not branch on this temperature field. |
| `TRAIT_LOGIT_MUTATION_SIGMA` | `float` | active runtime knob | active runtime knob. Standard deviation for ordinary latent trait-logit mutations. Higher values produce more aggressive trait drift. |
| `TRAIT_BUDGET_MUTATION_SIGMA` | `float` | active runtime knob | active runtime knob. Standard deviation for ordinary budget mutations. Higher values broaden budget drift across births. |
| `RARE_MUT_PROB` | `float` | active runtime knob | active runtime knob. Probability of entering the rare-mutation path for a birth. Raise cautiously: rare mutations are intentionally disruptive. |
| `RARE_TRAIT_LOGIT_MUTATION_SIGMA` | `float` | active runtime knob | active runtime knob. Rare-path trait-logit mutation strength. This is typically much larger than the ordinary sigma. |
| `RARE_TRAIT_BUDGET_MUTATION_SIGMA` | `float` | active runtime knob | active runtime knob. Rare-path budget mutation strength. Increase for stronger occasional budget shocks. |
| `RARE_POLICY_NOISE_SD` | `float` | active runtime knob | active runtime knob. Rare-path policy-noise standard deviation. This determines how disruptive rare policy mutation is. |
| `ENABLE_FAMILY_SHIFT_MUTATION` | `bool` | active runtime knob | active runtime knob. Whether births may mutate into a different family. Keep disabled if you want strict family inheritance. |
| `FAMILY_SHIFT_PROB` | `float` | active runtime knob | active runtime knob. Base probability of family-shift mutation when enabled. Usually kept very small. |

</details>

<details>
<summary><strong>ViewerConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `FPS` | `int` | active runtime knob | active runtime knob. Viewer frame-rate target. Higher values make the UI smoother but demand more rendering work. |
| `PAINT_BRUSH` | `List[int]` | currently unread | currently unread. The current viewer does not consume this brush-shape field. |
| `PAINT_RATE_STEP` | `float` | active runtime knob | active runtime knob. Step size used when adjusting paint rate interactively. Smaller steps give finer control. |
| `SHOW_OVERLAYS` | `Dict[str, bool]` | active runtime knob | active runtime knob. Default overlay on/off map for viewer startup. Keys in the default surface are `h_rate`, `h_grad`, and `rays`. |
| `WINDOW_WIDTH` | `int` | active runtime knob | active runtime knob. Initial viewer window width in pixels. Pure presentation knob. |
| `WINDOW_HEIGHT` | `int` | active runtime knob | active runtime knob. Initial viewer window height in pixels. Pure presentation knob. |
| `CELL_SIZE` | `int` | currently unread | currently unread. The current viewer computes cell size dynamically instead of reading this field. |
| `SHOW_BLOODLINE_LEGEND` | `bool` | active runtime knob | active runtime knob. Whether the viewer should display the bloodline legend by default. This affects UI density, not simulation state. |
| `BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether HP-based bloodline color modulation is active in the viewer. When disabled, rendered agents stay on their clean base family colo... |
| `BLOODLINE_LOW_HP_SHADE` | `float` | active runtime knob | active runtime knob. How strongly low-HP agents are shaded within bloodline coloring. Higher values generally make low-HP darkening more pronounced. |
| `SHOW_CATASTROPHE_PANEL` | `bool` | active runtime knob | active runtime knob. Whether the viewer shows the catastrophe panel. Presentation-only. |
| `SHOW_CATASTROPHE_OVERLAY` | `bool` | active runtime knob | active runtime knob. Whether catastrophe overlays are shown in the viewer. Presentation-only. |
| `SHOW_CATASTROPHE_STATUS_IN_HUD` | `bool` | active runtime knob | active runtime knob. Whether catastrophe state appears in the HUD/status strip. Presentation-only. |

</details>

<details>
<summary><strong>LogConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `DIR` | `str` | active runtime knob | active runtime knob. Base log directory. Most runs create a run subdirectory beneath this root. |
| `LOG_TICK_EVERY` | `int` | active runtime knob | active runtime knob. Primary telemetry/log cadence in ticks. Larger values reduce logging overhead and granularity. |
| `SNAPSHOT_EVERY` | `int` | active runtime knob | active runtime knob. Snapshot cadence in ticks. Higher values emit fewer snapshots. |
| `ASSERTIONS` | `bool` | active runtime knob | active runtime knob. Master assertion toggle used by invariant checks. Disabling this reduces safety diagnostics. |
| `AMP` | `bool` | active runtime knob | active runtime knob. Automatic mixed precision toggle. Typically enabled only on CUDA-capable runs. |

</details>

<details>
<summary><strong>IdentityConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ENABLE_UID_SUBSTRATE` | `bool` | active runtime knob | active runtime knob. Whether the UID identity substrate is conceptually enabled. This documents the canonical ownership model of the repository. |
| `OWNERSHIP_MODE` | `str` | active runtime knob | active runtime knob. Identity ownership-mode string. The configured default is `"uid_bridge"`. It describes the bridge between canonical UID identity and leg... |
| `ASSERT_BINDINGS` | `bool` | active runtime knob | active runtime knob. Whether binding invariants are asserted. Disable only for exceptional debugging/performance experiments. |
| `ASSERT_HISTORICAL_UIDS` | `bool` | active runtime knob | active runtime knob. Whether parent references are checked against the lifecycle ledger. Keeping this true protects lineage integrity. |
| `ASSERT_NO_SLOT_OWNERSHIP_LEAK` | `bool` | currently unread | currently unread. The current identity path does not branch on this assertion toggle. |
| `MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS` | `bool` | active runtime knob | active runtime knob. Whether canonical UIDs are mirrored into legacy float shadow columns. Useful for compatibility and viewer/log inspection. |

</details>

<details>
<summary><strong>SchemaConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `IDENTITY_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Identity schema version stamp. Bump only when the identity surface changes in a backward-incompatible way. |
| `OBS_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Observation schema version stamp. Changing this without a full migration will break compatibility expectations. |
| `PPO_STATE_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. PPO state schema version stamp. Used as a version marker; keep stable unless you deliberately migrate PPO checkpoint structure. |
| `CHECKPOINT_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Checkpoint schema version stamp. Critical for save/load compatibility. |
| `REPRODUCTION_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Reproduction schema version stamp. Documents compatibility for lineage / reproduction surfaces. |
| `CATASTROPHE_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Catastrophe schema version stamp. Used during catastrophe-state serialization/restore validation. |
| `TELEMETRY_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Telemetry schema version stamp. Protects downstream consumers from silent schema drift. |
| `LOGGING_SCHEMA_VERSION` | `int` | active runtime knob | active runtime knob. Logging schema version stamp. Bump only with deliberate ledger-format changes. |

</details>

<details>
<summary><strong>CheckpointConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ENABLE_SUBSTRATE_CHECKPOINTS` | `bool` | active runtime knob | active runtime knob. Master enablement flag for substrate-style runtime checkpoints. If disabled, higher-level checkpoint paths should be considered inactive... |
| `CAPTURE_RNG_STATE` | `bool` | active runtime knob | active runtime knob. Whether Python / NumPy / Torch RNG states are captured. Keep enabled for deterministic resume. |
| `CAPTURE_OPTIMIZER_STATE` | `bool` | active runtime knob | active runtime knob. Whether per-UID optimizer states are serialized. Disable only if you are willing to resume without optimizer continuity. |
| `CAPTURE_SCALER_STATE` | `bool` | active runtime knob | active runtime knob. Whether AMP scaler state is captured. Relevant mainly for CUDA AMP runs. |
| `CAPTURE_PPO_TRAINING_STATE` | `bool` | active runtime knob | active runtime knob. Whether PPO training counters / metadata are serialized. Useful for faithful training continuation. |
| `CAPTURE_BOOTSTRAP_STATE` | `bool` | active runtime knob | active runtime knob. Whether staged PPO bootstrap tails are checkpointed. Important for precise continuation of partially accumulated rollouts. |
| `STRICT_SCHEMA_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether checkpoint tensor/container shapes are validated strictly. Higher safety, slightly more validation work. |
| `STRICT_UID_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether checkpoint UID ownership consistency is validated strictly. Highly recommended for lineage-safe resume. |
| `STRICT_PPO_STATE_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether PPO-related checkpoint surfaces are validated strictly. Keep enabled unless you are debugging a migration. |
| `VALIDATE_OPTIMIZER_TENSOR_SHAPES` | `bool` | active runtime knob | active runtime knob. Whether optimizer tensor-state shapes are checked against live parameter shapes. Important when topology drift is a concern. |
| `VALIDATE_BUFFER_SCHEMA` | `bool` | active runtime knob | active runtime knob. Whether serialized PPO buffers are checked against the expected schema. Recommended for safety. |
| `SAVE_CHECKPOINT_MANIFEST` | `bool` | active runtime knob | active runtime knob. Whether a manifest file is emitted alongside checkpoint bundles. The atomic publish path uses this as a gate. |
| `SAVE_EVERY_TICKS` | `int` | active runtime knob | active runtime knob. Periodic runtime checkpoint cadence. Set `0` to disable scheduled runtime checkpoints. |
| `KEEP_LAST` | `int` | active runtime knob | active runtime knob. Retention count for scheduled runtime checkpoints. Use `<= 0` to keep every produced checkpoint. |
| `DIRECTORY_NAME` | `str` | active runtime knob | active runtime knob. Name of the subdirectory used for runtime checkpoints within a run directory. Pure pathing knob. |
| `FILENAME_PREFIX` | `str` | active runtime knob | active runtime knob. Stable filename prefix for scheduled checkpoint bundles. Pure pathing knob. |
| `ATOMIC_WRITE_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether checkpoint publish uses temp-file + atomic replace semantics. Recommended to k... |
| `MANIFEST_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether manifests are part of the published checkpoint file set. Must remain enabled if strict manifest validation or latest-pointer wri... |
| `CHECKSUM_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether bundle SHA-256 checksums are written/validated. Safety knob with modest extra I/O cost. |
| `STRICT_MANIFEST_VALIDATION` | `bool` | active safety knob | active safety knob with an explicit dependency constraint. Whether manifest metadata must validate during load. Requires `MANIFEST_ENABLED = True` in the cur... |
| `STRICT_DIRECTORY_STRUCTURE_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether manifest filenames must match the observed file structure exactly. Good for catching pathing / publish mistakes. |
| `STRICT_CONFIG_FINGERPRINT_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether manifest config fingerprint must match the bundle snapshot. Stricter but can block resumes across intentional config drift. |
| `WRITE_LATEST_POINTER` | `bool` | active safety knob | active safety knob with an explicit dependency constraint. Whether a `latest` pointer JSON file is maintained. Requires `MANIFEST_ENABLED = True` in the curr... |
| `LATEST_POINTER_FILENAME` | `str` | active runtime knob | active runtime knob. Filename used for the latest-checkpoint pointer. Pathing-only knob. |
| `BUNDLE_FILENAME_SUFFIX` | `str` | active runtime knob | active runtime knob. Filename suffix for checkpoint bundle files. Pathing-only knob. |
| `MANIFEST_FILENAME_SUFFIX` | `str` | active runtime knob | active runtime knob. Filename suffix appended to manifest files. Pathing-only knob. |
| `TEMPFILE_PREFIX` | `str` | active runtime knob | active runtime knob. Prefix used for temp files during atomic checkpoint publish. Pathing / hygiene knob. |

</details>

<details>
<summary><strong>TelemetryConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ENABLE_DEEP_LEDGERS` | `bool` | active runtime knob | active runtime knob. Master toggle for the richer ledger family. Disable to reduce telemetry volume. |
| `LOG_LIFE_LEDGER` | `bool` | active runtime knob | active runtime knob. Whether life-ledger rows are emitted. Useful for full lifecycle auditing. |
| `LOG_BIRTH_LEDGER` | `bool` | active runtime knob | active runtime knob. Whether birth events are logged. Important for lineage analysis. |
| `LOG_DEATH_LEDGER` | `bool` | active runtime knob | active runtime knob. Whether death events are logged. Important for mortality analysis. |
| `LOG_PPO_UPDATE_LEDGER` | `bool` | active runtime knob | active runtime knob. Whether PPO update summaries are logged. Useful for optimizer-trace analysis. |
| `LOG_CATASTROPHE_EVENT_LEDGER` | `bool` | active runtime knob | active runtime knob. Whether catastrophe start/end/clear events are logged. Useful for correlating shocks with outcomes. |
| `LOG_TICK_SUMMARY` | `bool` | active runtime knob | active runtime knob. Whether per-tick summaries are emitted. High-volume but very informative. |
| `LOG_FAMILY_SUMMARY` | `bool` | active runtime knob | active runtime knob. Whether family-level summaries are emitted. Good for bloodline-level analysis. |
| `FAMILY_SUMMARY_EVERY_TICKS` | `int` | active runtime knob | active runtime knob. Cadence for family-summary rows. Higher values reduce volume. |
| `SUMMARY_EXPORT_CADENCE_TICKS` | `int` | active runtime knob | active runtime knob. Cadence for exporting summary rows. Set above `1` to reduce per-tick I/O overhead. |
| `SUMMARY_SKIP_NON_EMIT_WORK` | `bool` | active runtime knob | active runtime knob. Whether summary aggregation work is skipped on non-emission ticks. Efficiency knob that reduces wasted computation. |
| `EXPORT_LINEAGE` | `bool` | active runtime knob | active runtime knob. Whether lineage graphs/structures are exported. Useful for ancestry reconstruction. |
| `LINEAGE_EXPORT_FORMAT` | `str` | active runtime knob | active runtime knob. Lineage export format selector. The comment in the repository states that the current runtime emits JSON lineage graphs. Treat alternate... |
| `PARQUET_BATCH_ROWS` | `int` | active runtime knob | active runtime knob. Buffered flush threshold per ledger. Larger values reduce write overhead but delay visibility. |
| `FLUSH_OPEN_LIVES_ON_CLOSE` | `bool` | active runtime knob | active runtime knob. Whether still-open life records are flushed on shutdown. Recommended for clean run closure. |
| `TRACK_CATASTROPHE_EXPOSURE` | `bool` | active runtime knob | active runtime knob. Whether catastrophe exposure is tracked for telemetry. Useful for shock attribution analyses. |
| `ENABLE_VIEWER_INSPECTOR_ENRICHMENT` | `bool` | active runtime knob | active runtime knob. Whether extended inspector detail is exposed to the viewer. Presentation/inspection enrichment only; it does not alter simulation semant... |

</details>

<details>
<summary><strong>ValidationConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ENABLE_FINAL_AUDIT_HARNESS` | `bool` | active runtime knob | active runtime knob. Master enablement for the bundled final audit harness. If disabled, the suite reports skipped checks. |
| `ENABLE_DETERMINISM_TESTS` | `bool` | active runtime knob | active runtime knob. Whether determinism probes are run by the validation suite. Recommended for any serious change touching state evolution. |
| `ENABLE_RESUME_CONSISTENCY_TESTS` | `bool` | active runtime knob | active runtime knob. Whether resume-consistency probes are run. Critical when checkpointing or identity semantics change. |
| `ENABLE_SAVE_LOAD_SAVE_TESTS` | `bool` | active runtime knob | active runtime knob. Whether save-load-save signature checks are run. Useful for checkpoint idempotence auditing. |
| `ENABLE_CATASTROPHE_REPRO_TESTS` | `bool` | active runtime knob | active runtime knob. Whether catastrophe reproducibility probes are run. Useful when modifying scheduler or catastrophe state surfaces. |
| `VALIDATION_STRICTNESS` | `str` | currently unread | currently unread. The current validation harness does not branch on this strictness string. |
| `AUDIT_DEFAULT_TICKS` | `int` | active runtime knob | active runtime knob. Default tick budget for audit harness runs. Longer runs increase confidence but cost more time. |
| `DETERMINISM_COMPARE_TICKS` | `int` | active runtime knob | active runtime knob. Tick count used by the determinism probe. Increase to make the comparison harsher. |
| `SAVE_LOAD_SAVE_COMPARE_BUFFERS` | `bool` | currently unread | currently unread. The current validation harness does not branch on this buffer-compare toggle. |
| `STRICT_TELEMETRY_SCHEMA_WRITES` | `bool` | currently unread | currently unread. The current validation harness does not branch on this telemetry-schema toggle. |

</details>

<details>
<summary><strong>MigrationConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `LOG_LEGACY_SLOT_FIELDS` | `bool` | active runtime knob | active runtime knob. Whether legacy slot fields are included in migration-era logging. Useful during transition periods. |
| `LOG_UID_FIELDS` | `bool` | active runtime knob | active runtime knob. Whether canonical UID fields are included in logs. Useful for lineage-safe auditing. |
| `VIEWER_SHOW_SLOT_AND_UID` | `bool` | active runtime knob | active runtime knob. Whether the viewer shows both slot and UID identity information. Inspection/UI only. |
| `VIEWER_SHOW_BLOODLINE` | `bool` | active runtime knob | active runtime knob. Whether bloodline information is shown in migration-era viewer surfaces. Inspection/UI only. |
| `REQUIRE_CANONICAL_UID_PATHS` | `bool` | active runtime knob | active runtime knob. Whether canonical UID paths are required. Safety / migration-hardening knob. |

</details>

<details>
<summary><strong>CatastropheConfig</strong></summary>

| Field | Type | Safety class | Note |
| --- | --- | --- | --- |
| `ENABLED` | `bool` | active runtime knob | active runtime knob. Master catastrophe enable switch. Turn this off to remove scheduler-driven world shocks entirely. |
| `DEFAULT_MODE` | `str` | active runtime knob | active runtime knob. Default catastrophe scheduler mode. Supported modes are `"off"`, `"manual_only"`, `"auto_dynamic"`, and `"auto_static"`. This determines... |
| `DEFAULT_SCHEDULER_ARMED` | `bool` | active runtime knob | active runtime knob. Whether a fresh run starts with the auto scheduler armed whenever the active mode is an auto mode. This does not change the selected mod... |
| `MANUAL_TRIGGER_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether operator-triggered catastrophes are allowed. Viewer/manual control surface. |
| `MANUAL_CLEAR_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether active catastrophes may be manually cleared. Viewer/manual control surface. |
| `ALLOW_OVERLAP` | `bool` | active runtime knob | active runtime knob. Whether multiple catastrophes may overlap in time. Overlapping shocks increase system complexity and severity. |
| `MAX_CONCURRENT` | `int` | active runtime knob | active runtime knob. Maximum number of simultaneously active catastrophes. Only relevant when overlap is allowed or manual triggering is aggressive. |
| `AUTO_DYNAMIC_GAP_MIN_TICKS` | `int` | active runtime knob | active runtime knob. Minimum scheduler gap for dynamic auto mode. Lower values create a busier catastrophe cadence. |
| `AUTO_DYNAMIC_GAP_MAX_TICKS` | `int` | active runtime knob | active runtime knob. Maximum scheduler gap for dynamic auto mode. Together with the minimum, this defines the random interval range. |
| `AUTO_DYNAMIC_SAMPLE_DURATION` | `bool` | active runtime knob | active runtime knob. Whether dynamic mode jitters duration around the configured base duration. Keeps repeated shocks less uniform. |
| `AUTO_STATIC_INTERVAL_TICKS` | `int` | active runtime knob | active runtime knob. Fixed interval between events in static auto mode. Lower values make the static schedule denser. |
| `AUTO_STATIC_ORDERING_POLICY` | `str` | active runtime knob | active runtime knob. Ordering policy for static auto mode. Supported values are `"round_robin"`, `"configured_sequence"`, and `"fixed_priority"`. This change... |
| `AUTO_STATIC_SEQUENCE` | `List[str]` | active runtime knob | active runtime knob. Explicit catastrophe sequence used when static ordering policy is `configured_sequence`. Only enabled catastrophe IDs from this list are... |
| `DEFAULT_DURATION_TICKS` | `int` | active runtime knob | active runtime knob. Fallback catastrophe duration in ticks. Used when a type-specific duration is not supplied. |
| `MIN_DURATION_TICKS` | `int` | active runtime knob | active runtime knob. Lower clamp for catastrophe duration. Protects against extremely short shocks. |
| `MAX_DURATION_TICKS` | `int` | active runtime knob | active runtime knob. Upper clamp for catastrophe duration. Protects against runaway-long shocks. |
| `PER_TYPE_DURATION_TICKS` | `Dict[str, int]` | active runtime knob | active runtime knob. Per-catastrophe duration overrides. Each key should be a valid catastrophe ID. |
| `TYPE_ENABLED` | `Dict[str, bool]` | active runtime knob | active runtime knob. Per-catastrophe enable table. Disable entries here to remove them from manual and auto selection. |
| `TYPE_SELECTION_WEIGHTS` | `Dict[str, float]` | active runtime knob | active runtime knob. Weighted-random selection table used by dynamic auto mode. Only relative magnitudes matter. |
| `TYPE_PARAMS` | `Dict[str, Dict[str, float]]` | active runtime knob | active runtime knob. Per-catastrophe parameter bundle. This is the main intensity-and-shape surface for ... |
| `VIEWER_CONTROLS_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether catastrophe viewer controls are enabled. Presentation/operator control only. |
| `VIEWER_OVERLAY_ENABLED` | `bool` | active runtime knob | active runtime knob. Whether catastrophe overlays are enabled in the viewer. Presentation only. |
| `PERSIST_STATE_IN_CHECKPOINTS` | `bool` | active runtime knob | active runtime knob. Whether active catastrophe state is serialized into checkpoints. Keep enabled for faithful resume. |
| `STRICT_CHECKPOINT_VALIDATION` | `bool` | active runtime knob | active runtime knob. Whether catastrophe checkpoint payloads are validated strictly. Useful safety flag when catastrophe state is part of save/load. |
| `RNG_STREAM_OFFSET` | `int` | active runtime knob | active runtime knob. Offset applied to the catastrophe RNG stream relative to the master seed. Keeps catastrophe scheduling deterministic while separating it... |

</details>
## Read next
- [Schema versions and compatibility surfaces](01_schema_versions_and_compatibility_surfaces.md)
- [Module reference index](02_module_reference_index.md)

## Related reference
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](../05_operations/08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `config reference`
- `safety class`
- `guarded surface`
