"""Canonical operational configuration for Tensor Crypt.

The dataclasses in this module define the repository's shared `cfg` object.
Runtime modules consume that singleton through `tensor_crypt.config_bridge` so
launch, simulation, checkpointing, telemetry, and viewer code all observe the
same configuration state.

Defaults favor deterministic runs, explicit validation, durable telemetry, and
checkpoint safety. Some fields are compatibility or schema surfaces whose
accepted values are narrower than their names imply; `validate_runtime_config`
in `tensor_crypt.app.runtime` is the authoritative startup gate.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class SimConfig:
    """Core session controls.

    This section decides the seed, device placement, and a few top-level runtime
    behaviors that affect the whole simulation session. These are the first knobs
    an operator should check when moving between local debugging, workstation
    training, or reproducibility-focused validation work.
    """
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "float32"  # Guarded compatibility surface; the runtime currently supports only float32.
    TICKS_PER_SEC: int = 30
    MAX_TICKS: int = 0  # Optional viewer/runtime auto-stop; 0 keeps the session open until the operator exits.
    REUSE_ACTION_BUFFER: bool = True  # Reuse the dense sparse-action tensor instead of reallocating it every tick.
    EXPERIMENTAL_FAMILY_VMAP_INFERENCE: bool = False
    EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET: int = 8


@dataclass
class GridConfig:
    """World-field and substrate controls.

    These knobs define the size of the world tensor and how heal / harm zones are
    combined into the field channel. They change the geometry and field behavior of
    the arena rather than agent learning logic directly.
    """
    W: int = 100
    H: int = 100
    HZ_OVERLAP_MODE: str = "max_abs"  # max_abs | sum_clamped | last_wins
    HZ_SUM_CLAMP: float = 5.0
    HZ_CLEAR_EACH_TICK: bool = True
    EXPOSE_H_GRAD: bool = False


@dataclass
class MapgenConfig:
    """Procedural map-generation controls.

    These values determine the density and scale of random walls and heal zones
    created when a new run is built from scratch.
    """
    RANDOM_WALLS: int = 6
    WALL_SEG_MIN: int = 20
    WALL_SEG_MAX: int = 52
    WALL_AVOID_MARGIN: int = 5
    HEAL_ZONE_COUNT: int = 28
    HEAL_ZONE_SIZE_RATIO: float = 0.05
    HEAL_RATE: float = 0.42


@dataclass
class AgentsConfig:
    """Population slot-capacity and spawn-surface controls.

    These knobs define how many dense runtime slots exist and whether the runtime
    permits multiple live agents to occupy the same tile.
    """
    N: int = 160
    SPAWN_MODE: str = "uniform"  # Guarded compatibility surface; only uniform spawn is currently implemented.
    NO_STACKING: bool = True


@dataclass
class RespawnCrowdingOverlayConfig:
    """The Ashen Press (crowding-gated reproduction overlay).

    This overlay evaluates local population density around the anchor parent
    before offspring placement is attempted.
    """

    ENABLED: bool = True
    LOCAL_RADIUS: int = 2
    MAX_NEIGHBORS: int = 6
    POLICY_WHEN_CROWDED: str = "global_only"  # block_birth | global_only
    BELOW_FLOOR_POLICY: str = "bypass"  # strict | bypass | global_only


@dataclass
class RespawnCooldownOverlayConfig:
    """The Widow Interval (parent refractory reproduction overlay).

    Cooldown is UID-scoped rather than slot-scoped so slot reuse does not
    corrupt parent eligibility semantics.
    """

    ENABLED: bool = True
    DURATION_TICKS: int = 48
    APPLY_TO_BRAIN_PARENT: bool = True
    APPLY_TO_TRAIT_PARENT: bool = True
    APPLY_TO_ANCHOR_PARENT: bool = False
    UNIFIED_UID_POLICY: bool = True
    EMPTY_POOL_POLICY: str = "allow_best_available"  # allow_best_available | strict
    BELOW_FLOOR_POLICY: str = "allow_best_available"  # allow_best_available | bypass | strict


@dataclass
class RespawnLocalParentOverlayConfig:
    """The Bloodhold Radius (local lineage parent-selection overlay)."""

    ENABLED: bool = True
    SELECTION_RADIUS: int = 10
    FALLBACK_BEHAVIOR: str = "global"  # global | strict
    BELOW_FLOOR_POLICY: str = "prefer_local_then_global"  # prefer_local_then_global | bypass | strict


@dataclass
class RespawnOverlayViewerConfig:
    """Viewer exposure controls for reproduction overlay doctrines."""

    HOTKEYS_ENABLED: bool = True
    SHOW_STATUS_IN_HUD: bool = True
    SHOW_STATUS_IN_PANEL: bool = True
    SHOW_OVERRIDE_MARKERS: bool = True


@dataclass
class RespawnOverlayConfig:
    """Structured overlay doctrine control surface for reproduction."""

    CROWDING: RespawnCrowdingOverlayConfig = field(default_factory=RespawnCrowdingOverlayConfig)
    COOLDOWN: RespawnCooldownOverlayConfig = field(default_factory=RespawnCooldownOverlayConfig)
    LOCAL_PARENT: RespawnLocalParentOverlayConfig = field(default_factory=RespawnLocalParentOverlayConfig)
    VIEWER: RespawnOverlayViewerConfig = field(default_factory=RespawnOverlayViewerConfig)


@dataclass
class RespawnConfig:
    """Binary reproduction, population recovery, and offspring placement controls.

    This section governs the post-death repopulation path: when births happen, how
    many can happen, which parent roles are selected, what floor-recovery means,
    how extinction is handled, and where offspring may be placed.
    """
    RESPAWN_PERIOD: int = 64
    MAX_SPAWNS_PER_CYCLE: int = 8
    POPULATION_FLOOR: int = 64
    POPULATION_CEILING: int = 160

    MODE: str = "binary_parented"  # Guarded compatibility surface; reproduction semantics remain binary parented.
    BRAIN_PARENT_SELECTOR: str = "fitness"
    TRAIT_PARENT_SELECTOR: str = "vitality"
    ANCHOR_PARENT_SELECTOR: str = "trait_parent"  # brain_parent | trait_parent | random_parent | fitter_of_two

    BRAIN_PARENT_MIN_FITNESS: float = 0.0
    TRAIT_PARENT_MIN_HP_RATIO: float = 0.10
    TRAIT_PARENT_MIN_AGE_TICKS: int = 0
    FLOOR_RECOVERY_SUSPEND_THRESHOLDS: bool = True
    FLOOR_RECOVERY_REQUIRE_TWO_PARENTS: bool = True
    OVERLAYS: RespawnOverlayConfig = field(default_factory=RespawnOverlayConfig)

    EXTINCTION_POLICY: str = "seed_bank_bootstrap"  # fail_run | seed_bank_bootstrap | admin_spawn_defaults
    EXTINCTION_BOOTSTRAP_SPAWNS: int = 8
    EXTINCTION_BOOTSTRAP_FAMILY: str = "House Nocthar"

    OFFSPRING_JITTER_RADIUS_MIN: int = 1
    OFFSPRING_JITTER_RADIUS_MAX: int = 4
    OFFSPRING_MAX_PLACEMENT_ATTEMPTS: int = 48
    ALLOW_FALLBACK_GLOBAL_PLACEMENT: bool = True
    DISALLOW_SPAWN_ON_WALL: bool = True
    DISALLOW_SPAWN_ON_OCCUPIED: bool = True
    DISALLOW_SPAWN_IN_HARM_ZONE: bool = True

    BIRTH_HP_MODE: str = "full"  # full | fraction
    BIRTH_HP_FRACTION: float = 1.0

    LOG_PLACEMENT_FAILURES: bool = True
    ASSERT_BINARY_PARENTING: bool = True


@dataclass
class TraitInit:
    """Legacy/default trait template.

    the live birth pipeline is driven by latent-budget reconstruction rather than
    direct reads from this template.
    """
    mass: float = 2.0
    vision: float = 8.0
    hp_max: float = 20.0
    metab: float = 0.005


@dataclass
class TraitClamp:
    """Trait clamp ranges.

    These are the hard lower/upper bounds used when latent allocations are decoded
    into physical trait values. Widening them expands the reachable biological
    space; tightening them compresses it.
    """
    mass: List[float] = field(default_factory=lambda: [0.5, 8.0])
    vision: List[float] = field(default_factory=lambda: [4.0, 16.0])
    hp_max: List[float] = field(default_factory=lambda: [5.0, 50.0])
    metab: List[float] = field(default_factory=lambda: [0.008, 0.28])


@dataclass
class TraitBudgetConfig:
    """Trait-allocation budget controls.

    These knobs define the latent budget used by the trait decoder and its allowed
    range. They shape the total amount of trait mass available before clamping.
    """
    INIT_BUDGET: float = 1.0
    INIT_LOGITS: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    MIN_BUDGET: float = 0.25
    MAX_BUDGET: float = 1.75


@dataclass
class TraitsConfig:
    """Trait decoding controls.

    The live repository uses a latent-budget system plus an affine metabolism
    formula. This section is where trait-space constraints and metabolism formula
    coefficients are documented.
    """
    INIT: TraitInit = field(default_factory=TraitInit)
    CLAMP: TraitClamp = field(default_factory=TraitClamp)
    BUDGET: TraitBudgetConfig = field(default_factory=TraitBudgetConfig)
    METAB_FORM: str = "affine_combo"  # Guarded compatibility surface; only affine_combo is currently implemented.
    METAB_COEFFS: Dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.00015,
            "per_mass": 0.00008,
            "per_vision": 0.000015,
        }
    )


@dataclass
class PhysicsConfig:
    """Combat / collision / movement cost controls.

    These constants shape deterministic world damage and penalties. They do not
    control learning directly; they change the environment the policies must solve.
    """
    K_WALL_PENALTY: float = 0.35
    K_RAM_PENALTY: float = 0.08
    K_IDLE_HIT_PENALTY: float = 0.45
    K_WINNER_DAMAGE: float = 0.12
    K_LOSER_DAMAGE: float = 0.36
    MOVE_FAIL_COST: float = -0.2
    TIE_BREAKER: str = "strength_then_lowest_id"  # Contest tie-break policy after strength sorting.


@dataclass
class PerceptionConfig:
    """Observation-schema controls.

    These values define the canonical observation layout, legacy bridge dimensions,
    and normalization constants consumed by the perception and brain subsystems.
    Changing schema counts here is high-risk because tensor shapes must remain
    consistent end-to-end.
    """
    NUM_RAYS: int = 32
    RAY_FIELD_AGG: str = "max_abs"
    RAY_STEP_SAMPLER: str = "dda_first_hit"

    OBS_MODE: str = "canonical_v2"
    RETURN_EXPERIMENTAL_OBSERVATIONS: bool = False

    CANONICAL_RAY_FEATURES: int = 8
    CANONICAL_SELF_FEATURES: int = 11
    CANONICAL_CONTEXT_FEATURES: int = 3

    LEGACY_RAY_FEATURES: int = 5
    LEGACY_STATE_FEATURES: int = 2
    LEGACY_GENOME_FEATURES: int = 4
    LEGACY_POSITION_FEATURES: int = 2
    LEGACY_CONTEXT_FEATURES: int = 3
    LEGACY_ADAPTER_MODE: str = "canonical_bridge_v1"

    ZONE_RATE_ABS_MAX: float = 1.0
    AGE_NORM_TICKS: int = 1024
    RETURN_CANONICAL_OBSERVATIONS: bool = True
    EXPERIMENTAL_RAY_FEATURES: int = 7
    EXPERIMENTAL_SELF_FEATURES: int = 11
    EXPERIMENTAL_CONTEXT_FEATURES: int = 1


@dataclass
class BloodlineFamilySpec:
    """Per-family MLP topology description.

    Each family owns a fixed architectural signature. Fields here determine widths,
    activation, normalization placement, residual/gating usage, and optional split
    ray/scalar encoding paths.
    """
    hidden_widths: List[int] = field(default_factory=lambda: [256, 256, 192])
    activation: str = "gelu"
    normalization: str = "pre"
    residual: bool = True
    gated: bool = False
    split_inputs: bool = False
    split_ray_width: int = 0
    split_scalar_width: int = 0
    dropout: float = 0.0
    observation_contract: str = "canonical_v2"


@dataclass
class BrainConfig:
    """Policy/value network family controls.

    This section defines the action/value head sizes, the set of valid bloodline
    families, their colors, and the exact architectural spec used to instantiate a
    brain for each family.
    """
    ACTION_DIM: int = 9
    VALUE_DIM: int = 1

    FAMILY_ORDER: List[str] = field(
        default_factory=lambda: [
            "House Nocthar",
            "House Vespera",
            "House Umbrael",
            "House Mourndveil",
            "House Somnyr",
        ]
    )
    DEFAULT_FAMILY: str = "House Nocthar"
    EXPERIMENTAL_BRANCH_PRESET: bool = False
    EXPERIMENTAL_BRANCH_FAMILY: str = "House Nocthar"
    EXPERIMENTAL_BRANCH_COLOR: List[int] = field(default_factory=lambda: [64, 224, 255])
    EXPERIMENTAL_BRANCH_SPEC: BloodlineFamilySpec = field(
        default_factory=lambda: BloodlineFamilySpec(
            hidden_widths=[96, 64, 64],
            activation="silu",
            normalization="pre",
            residual=True,
            gated=False,
            split_inputs=True,
            split_ray_width=64,
            split_scalar_width=32,
            dropout=0.00,
            observation_contract="experimental_selfcentric_v1",
        )
    )
    INITIAL_FAMILY_ASSIGNMENT: str = "round_robin"
    INITIAL_FAMILY_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "House Nocthar": 1.0,
            "House Vespera": 1.0,
            "House Umbrael": 1.0,
            "House Mourndveil": 1.0,
            "House Somnyr": 1.0,
        }
    )

    LEGACY_TRANSFORMER_FALLBACK_ENABLED: bool = False
    ALLOW_LEGACY_OBS_FALLBACK: bool = True

    FAMILY_COLORS: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "House Nocthar": [84, 138, 214],
            "House Vespera": [84, 160, 112],
            "House Umbrael": [220, 184, 76],
            "House Mourndveil": [208, 102, 102],
            "House Somnyr": [168, 112, 208],
        }
    )

    FAMILY_SPECS: Dict[str, BloodlineFamilySpec] = field(
        default_factory=lambda: {
            "House Nocthar": BloodlineFamilySpec(hidden_widths=[256, 256, 224, 192], activation="gelu", normalization="pre", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Vespera": BloodlineFamilySpec(hidden_widths=[160, 160, 160, 128, 128], activation="silu", normalization="pre", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Umbrael": BloodlineFamilySpec(hidden_widths=[320, 320, 224], activation="relu", normalization="post", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Mourndveil": BloodlineFamilySpec(hidden_widths=[224, 224, 192], activation="silu", normalization="pre", residual=True, gated=True, split_inputs=True, split_ray_width=160, split_scalar_width=96, dropout=0.00),
            "House Somnyr": BloodlineFamilySpec(hidden_widths=[256, 256, 256, 224, 192], activation="gelu", normalization="pre", residual=True, gated=True, split_inputs=True, split_ray_width=192, split_scalar_width=128, dropout=0.02),
        }
    )


@dataclass
class PPOConfig:
    """PPO optimization and reward-surface controls.

    These knobs govern rollout length, optimization strength, clipping, entropy,
    bootstrap strictness, and the configurable reward gate that the engine validates
    before runtime.
    """
    GAMMA: float = 0.99
    LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENTROPY_COEF: float = 0.001
    VALUE_COEF: float = 0.5
    LR: float = 3e-4
    BATCH_SZ: int = 16
    MINI_BATCHES: int = 4
    EPOCHS: int = 3
    TARGET_KL: float = 0.01
    GRAD_NORM_CLIP: float = 1.0
    REWARD_FORM: str = "sq_health_ratio"
    REWARD_GATE_MODE: str = "off"  # off | hp_ratio_min | hp_abs_min
    REWARD_GATE_THRESHOLD: float = 0.0
    REWARD_BELOW_GATE_VALUE: float = 0.0
    UPDATE_EVERY_N_TICKS: int = 64

    OWNERSHIP_MODE: str = "uid_strict"  # Guarded compatibility surface; PPO ownership remains canonical-UID based.
    BUFFER_SCHEMA_VERSION: int = 1
    TRACK_TRAINING_STATE: bool = True
    FAMILY_AWARE_UPDATE_ORDERING: bool = True
    REQUIRE_BOOTSTRAP_FOR_ACTIVE_BUFFER: bool = True
    COUNT_TRUNCATED_ROLLOUTS: bool = True
    DROP_INACTIVE_UID_BUFFERS_AFTER_FINALIZATION: bool = True
    STRICT_BUFFER_VALIDATION: bool = True


@dataclass
class EvolutionConfig:
    """Mutation and fitness carryover controls.

    These values shape how much policy / trait mutation is applied during births and
    how much historical fitness decays or persists across death cycles.
    """
    SELECTION: str = "softmax_fitness"
    FITNESS_DECAY: float = 0.99
    POLICY_NOISE_SD: float = 0.01
    FITNESS_TEMP: float = 1.0

    TRAIT_LOGIT_MUTATION_SIGMA: float = 0.012
    TRAIT_BUDGET_MUTATION_SIGMA: float = 0.05
    RARE_MUT_PROB: float = 0.0005
    RARE_TRAIT_LOGIT_MUTATION_SIGMA: float = 0.40
    RARE_TRAIT_BUDGET_MUTATION_SIGMA: float = 0.15
    RARE_POLICY_NOISE_SD: float = 0.03

    ENABLE_FAMILY_SHIFT_MUTATION: bool = False
    FAMILY_SHIFT_PROB: float = 0.0001


@dataclass
class ViewerConfig:
    """Viewer and HUD presentation controls.

    These knobs affect operator-facing rendering, overlay defaults, window size, and
    how catastrophe status is exposed in the UI. They do not change simulation
    mechanics.
    """
    FPS: int = 45
    PAINT_BRUSH: List[int] = field(default_factory=lambda: [3, 3])
    PAINT_RATE_STEP: float = 0.05
    SHOW_OVERLAYS: Dict[str, bool] = field(default_factory=lambda: {"h_rate": True, "h_grad": False, "rays": False})  # Viewer default overlay state.
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 800
    CELL_SIZE: int = 10
    SHOW_BLOODLINE_LEGEND: bool = True
    SHOW_OPERATOR_ACTION_BUTTONS: bool = True
    SHOW_OPERATOR_ACTION_STATUS: bool = True
    BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED: bool = True
    BLOODLINE_LOW_HP_SHADE: float = 0.35

    SHOW_CATASTROPHE_PANEL: bool = True
    SHOW_CATASTROPHE_OVERLAY: bool = True
    SHOW_CATASTROPHE_STATUS_IN_HUD: bool = True


@dataclass
class LogConfig:
    """Logging and runtime assertion controls.

    This section governs log directory placement, emission cadence, assertion
    hardness, and AMP enablement.
    """
    DIR: str = "logs"
    LOG_TICK_EVERY: int = 20000
    SNAPSHOT_EVERY: int = 1_000_000
    ASSERTIONS: bool = True
    AMP: bool = True


@dataclass
class IdentityConfig:
    """Canonical UID ownership controls.

    These knobs document the UID substrate and compatibility bridging between the
    canonical UID path and legacy float shadow columns used for visibility.
    """
    ENABLE_UID_SUBSTRATE: bool = True
    OWNERSHIP_MODE: str = "uid_bridge"
    ASSERT_BINDINGS: bool = True
    ASSERT_HISTORICAL_UIDS: bool = True
    ASSERT_NO_SLOT_OWNERSHIP_LEAK: bool = True
    MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS: bool = True


@dataclass
class SchemaConfig:
    """Schema version stamps.

    These values are written into checkpoint / telemetry surfaces to make schema
    drift explicit. They are not casual tuning knobs; changing them should happen
    only with a deliberate migration.
    """
    IDENTITY_SCHEMA_VERSION: int = 1
    OBS_SCHEMA_VERSION: int = 2
    PPO_STATE_SCHEMA_VERSION: int = 1
    CHECKPOINT_SCHEMA_VERSION: int = 6
    REPRODUCTION_SCHEMA_VERSION: int = 2
    CATASTROPHE_SCHEMA_VERSION: int = 1
    TELEMETRY_SCHEMA_VERSION: int = 4
    LOGGING_SCHEMA_VERSION: int = 5


@dataclass
class CheckpointConfig:
    """Checkpoint capture, validation, and publish controls.

    This section determines what state is saved, how strict checkpoint validation
    is, whether file publishing is atomic, and how runtime checkpoint scheduling and
    retention behave.
    """
    ENABLE_SUBSTRATE_CHECKPOINTS: bool = True
    CAPTURE_RNG_STATE: bool = True
    CAPTURE_OPTIMIZER_STATE: bool = True
    CAPTURE_SCALER_STATE: bool = True
    CAPTURE_PPO_TRAINING_STATE: bool = True
    CAPTURE_BOOTSTRAP_STATE: bool = True
    STRICT_SCHEMA_VALIDATION: bool = True
    STRICT_UID_VALIDATION: bool = True
    STRICT_PPO_STATE_VALIDATION: bool = True
    VALIDATE_OPTIMIZER_TENSOR_SHAPES: bool = True
    VALIDATE_BUFFER_SCHEMA: bool = True
    SAVE_CHECKPOINT_MANIFEST: bool = True  # Manifest emission gate used by atomic checkpoint publishing.
    SAVE_EVERY_TICKS: int = 100_000  # Positive value enables scheduled atomic runtime checkpoints.
    KEEP_LAST: int = 8  # Retention count for scheduler-produced runtime checkpoints; <=0 keeps every checkpoint.
    DIRECTORY_NAME: str = "checkpoints"  # Subdirectory below each run directory where scheduler checkpoints are published.
    FILENAME_PREFIX: str = "runtime_tick_"  # Stable filename prefix for periodic runtime checkpoints.
    ENABLE_WALLCLOCK_AUTOSAVE: bool = False
    WALLCLOCK_AUTOSAVE_INTERVAL_SECONDS: float = 900.0
    WALLCLOCK_AUTOSAVE_MIN_TICKS_ADVANCED: int = 1
    WALLCLOCK_AUTOSAVE_WHILE_PAUSED: bool = False
    ENABLE_SHUTDOWN_CHECKPOINT: bool = True
    SHUTDOWN_CHECKPOINT_BEST_EFFORT: bool = True
    SAVE_REASON_LOGGING_ENABLED: bool = True
    CONTINUE_TELEMETRY_ON_RESUME: bool = True
    RESUME_CONTINUATION_POLICY: str = "continue_lineage_root"
    FORK_TELEMETRY_POLICY: str = "new_lineage_root"
    RESTORE_CHECKPOINT_BOOKKEEPING_ON_RESUME: bool = True

    ATOMIC_WRITE_ENABLED: bool = True
    MANIFEST_ENABLED: bool = True
    CHECKSUM_ENABLED: bool = True
    STRICT_MANIFEST_VALIDATION: bool = True
    STRICT_DIRECTORY_STRUCTURE_VALIDATION: bool = True
    STRICT_CONFIG_FINGERPRINT_VALIDATION: bool = False
    WRITE_LATEST_POINTER: bool = True
    LATEST_POINTER_FILENAME: str = "latest_checkpoint.json"
    BUNDLE_FILENAME_SUFFIX: str = ".pt"
    MANIFEST_FILENAME_SUFFIX: str = ".manifest.json"
    TEMPFILE_PREFIX: str = ".tmp_ckpt_"
    LAUNCH_MODE: str = "fresh_run"
    LOAD_PATH: str = ""
    FORK_REASON: str = ""
    WRITE_COMPATIBILITY_REPORT: bool = True
    LEGACY_METADATA_POLICY: str = "infer_conservative"
    COMPATIBILITY_REPORT_FILENAME: str = "resume_compatibility_report.json"


@dataclass
class TelemetryConfig:
    """Ledger and export controls.

    These knobs determine which telemetry surfaces are emitted and how frequently
    summary/lineage data is flushed.
    """
    ENABLE_DEEP_LEDGERS: bool = True
    LOG_LIFE_LEDGER: bool = True
    LOG_BIRTH_LEDGER: bool = True
    LOG_DEATH_LEDGER: bool = True
    LOG_PPO_UPDATE_LEDGER: bool = True
    LOG_CATASTROPHE_EVENT_LEDGER: bool = True
    LOG_TICK_SUMMARY: bool = True
    LOG_FAMILY_SUMMARY: bool = True
    FAMILY_SUMMARY_EVERY_TICKS: int = 128
    SUMMARY_EXPORT_CADENCE_TICKS: int = 64  # Tick-summary export cadence for the overnight profile.
    SUMMARY_SKIP_NON_EMIT_WORK: bool = True  # Skip summary aggregation work on ticks that do not emit summary/family rows.
    EXPORT_LINEAGE: bool = True
    LINEAGE_EXPORT_FORMAT: str = "json"  # Export format gate; the runtime currently emits JSON lineage graphs.
    PARQUET_BATCH_ROWS: int = 4096  # Buffered parquet flush threshold per ledger for long unattended runs.
    FLUSH_OPEN_LIVES_ON_CLOSE: bool = True
    SESSION_SEGMENTATION_ENABLED: bool = True
    FINALIZE_OPEN_LIVES_ON_SESSION_CLOSE: bool = False
    FINALIZE_OPEN_LIVES_ON_TERMINAL_RUN_CLOSE: bool = True
    WRITE_SESSION_CATALOG: bool = True
    WRITE_SESSION_METADATA: bool = True
    TRACK_CATASTROPHE_EXPOSURE: bool = True
    ENABLE_VIEWER_INSPECTOR_ENRICHMENT: bool = True  # Toggle extended inspector details without affecting simulation semantics.
    SELECTED_BRAIN_EXPORT_DIRECTORY_NAME: str = "selected_exports"


@dataclass
class ValidationConfig:
    """Offline audit-harness controls.

    These values drive the bundled determinism, resume-consistency, catastrophe
    reproduction, and save-load-save validation helpers.
    """
    ENABLE_FINAL_AUDIT_HARNESS: bool = True
    ENABLE_DETERMINISM_TESTS: bool = True
    ENABLE_RESUME_CONSISTENCY_TESTS: bool = True
    ENABLE_SAVE_LOAD_SAVE_TESTS: bool = True
    ENABLE_CATASTROPHE_REPRO_TESTS: bool = True
    ENABLE_RESUME_POLICY_TESTS: bool = True
    ENABLE_RUNTIME_LIFECYCLE_TESTS: bool = True
    ENABLE_OPERATOR_ARTIFACT_TESTS: bool = True
    VALIDATION_STRICTNESS: str = "strict"  # permissive | strict
    AUDIT_DEFAULT_TICKS: int = 16
    DETERMINISM_COMPARE_TICKS: int = 8
    SAVE_LOAD_SAVE_COMPARE_BUFFERS: bool = True
    STRICT_TELEMETRY_SCHEMA_WRITES: bool = True


@dataclass
class MigrationConfig:
    """Migration / compatibility visibility controls.

    These knobs support the UID migration and tell logging / viewer surfaces what
    legacy-vs-canonical identity information should be shown.
    """
    LOG_LEGACY_SLOT_FIELDS: bool = True
    LOG_UID_FIELDS: bool = True
    VIEWER_SHOW_SLOT_AND_UID: bool = True
    VIEWER_SHOW_BLOODLINE: bool = True
    REQUIRE_CANONICAL_UID_PATHS: bool = True


@dataclass
class CatastropheConfig:
    """Catastrophe scheduler and world-shock controls.

    This section decides whether catastrophes are enabled, how the scheduler picks
    events, how long they last, which types are eligible, and what parameter bundle
    each catastrophe type uses when active.
    """
    ENABLED: bool = True
    DEFAULT_MODE: str = "manual_only"  # off | manual_only | auto_dynamic | auto_static
    DEFAULT_SCHEDULER_ARMED: bool = False
    MANUAL_TRIGGER_ENABLED: bool = True
    MANUAL_CLEAR_ENABLED: bool = True

    ALLOW_OVERLAP: bool = False
    MAX_CONCURRENT: int = 1

    AUTO_DYNAMIC_GAP_MIN_TICKS: int = 240
    AUTO_DYNAMIC_GAP_MAX_TICKS: int = 540
    AUTO_DYNAMIC_SAMPLE_DURATION: bool = True

    AUTO_STATIC_INTERVAL_TICKS: int = 420
    AUTO_STATIC_ORDERING_POLICY: str = "round_robin"  # round_robin | configured_sequence | fixed_priority
    AUTO_STATIC_SEQUENCE: List[str] = field(
        default_factory=lambda: [
            "ashfall_of_nocthar",
            "veil_of_somnyr",
            "the_hollow_fast",
            "graveweight",
            "glass_requiem",
            "the_witchstorm",
            "the_thorn_march",
            "the_barren_hymn",
        ]
    )

    DEFAULT_DURATION_TICKS: int = 180
    MIN_DURATION_TICKS: int = 60
    MAX_DURATION_TICKS: int = 480
    PER_TYPE_DURATION_TICKS: Dict[str, int] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": 160,
            "sanguine_bloom": 180,
            "the_woundtide": 180,
            "the_hollow_fast": 200,
            "mirror_of_thorns": 160,
            "veil_of_somnyr": 220,
            "graveweight": 200,
            "glass_requiem": 160,
            "the_witchstorm": 180,
            "the_thorn_march": 240,
            "the_barren_hymn": 120,
            "crimson_deluge": 180,
        }
    )

    TYPE_ENABLED: Dict[str, bool] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": True,
            "sanguine_bloom": True,
            "the_woundtide": True,
            "the_hollow_fast": True,
            "mirror_of_thorns": True,
            "veil_of_somnyr": True,
            "graveweight": True,
            "glass_requiem": True,
            "the_witchstorm": True,
            "the_thorn_march": True,
            "the_barren_hymn": True,
            "crimson_deluge": True,
        }
    )

    TYPE_SELECTION_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": 1.0,
            "sanguine_bloom": 1.0,
            "the_woundtide": 1.0,
            "the_hollow_fast": 1.0,
            "mirror_of_thorns": 1.0,
            "veil_of_somnyr": 1.2,
            "graveweight": 0.9,
            "glass_requiem": 0.9,
            "the_witchstorm": 0.8,
            "the_thorn_march": 0.7,
            "the_barren_hymn": 0.6,
            "crimson_deluge": 0.9,
        }
    )

    TYPE_PARAMS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": {"positive_zone_fraction": 0.65},
            "sanguine_bloom": {"zone_fraction": 0.45, "negative_rate": -1.5},
            "the_woundtide": {"front_half_width": 5.0, "negative_rate": -2.0},
            "the_hollow_fast": {"positive_scalar": 0.25},
            "mirror_of_thorns": {"zone_fraction": 0.50},
            "veil_of_somnyr": {"vision_scalar": 0.45},
            "graveweight": {"metabolism_scalar": 1.65, "mass_burden_scalar": 0.06},
            "glass_requiem": {"collision_damage_scalar": 1.8},
            "the_witchstorm": {"trait_sigma_scalar": 2.0, "budget_sigma_scalar": 2.0, "policy_noise_scalar": 2.0, "rare_prob_scalar": 3.0, "family_shift_scalar": 4.0},
            "the_thorn_march": {"negative_rate": -2.0, "max_shrink_fraction": 0.35},
            "the_barren_hymn": {"reproduction_enabled": 0.0},
            "crimson_deluge": {"patch_count": 3.0, "patch_size_fraction": 0.18, "negative_rate": -2.5},
        }
    )

    VIEWER_CONTROLS_ENABLED: bool = True
    VIEWER_OVERLAY_ENABLED: bool = True

    PERSIST_STATE_IN_CHECKPOINTS: bool = True
    STRICT_CHECKPOINT_VALIDATION: bool = True
    RNG_STREAM_OFFSET: int = 911


@dataclass
class Config:
    """Root aggregate configuration.

    This is the single object instantiated by repository root `config.py`, and the
    package consumes it through a bridge rather than maintaining a second source of
    truth.
    """
    SIM: SimConfig = field(default_factory=SimConfig)
    GRID: GridConfig = field(default_factory=GridConfig)
    MAPGEN: MapgenConfig = field(default_factory=MapgenConfig)
    AGENTS: AgentsConfig = field(default_factory=AgentsConfig)
    TRAITS: TraitsConfig = field(default_factory=TraitsConfig)
    PHYS: PhysicsConfig = field(default_factory=PhysicsConfig)
    PERCEPT: PerceptionConfig = field(default_factory=PerceptionConfig)
    BRAIN: BrainConfig = field(default_factory=BrainConfig)
    RESPAWN: RespawnConfig = field(default_factory=RespawnConfig)
    PPO: PPOConfig = field(default_factory=PPOConfig)
    EVOL: EvolutionConfig = field(default_factory=EvolutionConfig)
    VIEW: ViewerConfig = field(default_factory=ViewerConfig)
    LOG: LogConfig = field(default_factory=LogConfig)
    IDENTITY: IdentityConfig = field(default_factory=IdentityConfig)
    SCHEMA: SchemaConfig = field(default_factory=SchemaConfig)
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)
    TELEMETRY: TelemetryConfig = field(default_factory=TelemetryConfig)
    VALIDATION: ValidationConfig = field(default_factory=ValidationConfig)
    MIGRATION: MigrationConfig = field(default_factory=MigrationConfig)
    CATASTROPHE: CatastropheConfig = field(default_factory=CatastropheConfig)


cfg = Config()


def apply_experimental_single_family_launch_defaults() -> None:
    """
    Force the live app launch path onto the experimental self-centric preset.

    This intentionally mutates only the process-local runtime config object used
    by the repository entrypoints. Direct callers can still override these
    fields explicitly before startup if they need a different launch mode.
    """

    cfg.PERCEPT.OBS_MODE = "experimental_selfcentric_v1"
    cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = str(cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY or cfg.BRAIN.DEFAULT_FAMILY)
    cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION = False
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.LOG.AMP = False
