"""Root configuration surface for Tensor Crypt."""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class SimConfig:
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "float32"
    TICKS_PER_SEC: int = 30
    MAX_TICKS: int = 0


@dataclass
class GridConfig:
    W: int = 128
    H: int = 128
    HZ_OVERLAP_MODE: str = "max_abs"  # max_abs | sum_clamped | last_wins
    HZ_SUM_CLAMP: float = 5.0
    HZ_CLEAR_EACH_TICK: bool = True
    EXPOSE_H_GRAD: bool = False


@dataclass
class MapgenConfig:
    RANDOM_WALLS: int = 9
    WALL_SEG_MIN: int = 20
    WALL_SEG_MAX: int = 83
    WALL_AVOID_MARGIN: int = 4
    HEAL_ZONE_COUNT: int = 9
    HEAL_ZONE_SIZE_RATIO: float = 20 / 256
    HEAL_RATE: float = 0.5


@dataclass
class AgentsConfig:
    N: int = 100
    SPAWN_MODE: str = "uniform"  # uniform | poisson_clusters
    NO_STACKING: bool = True


@dataclass
class RespawnConfig:
    RESPAWN_PERIOD: int = 100
    MAX_SPAWNS_PER_CYCLE: int = 10
    POPULATION_FLOOR: int = 20
    POPULATION_CEILING: int = 150

    # Prompt 5 reproduction control surface.
    MODE: str = "binary_parented"
    BRAIN_PARENT_SELECTOR: str = "fitness"
    TRAIT_PARENT_SELECTOR: str = "vitality"
    ANCHOR_PARENT_SELECTOR: str = "trait_parent"  # brain_parent | trait_parent | random_parent | fitter_of_two

    BRAIN_PARENT_MIN_FITNESS: float = 0.0
    TRAIT_PARENT_MIN_HP_RATIO: float = 0.10
    TRAIT_PARENT_MIN_AGE_TICKS: int = 0
    FLOOR_RECOVERY_SUSPEND_THRESHOLDS: bool = True
    FLOOR_RECOVERY_REQUIRE_TWO_PARENTS: bool = True

    EXTINCTION_POLICY: str = "fail_run"  # fail_run | seed_bank_bootstrap | admin_spawn_defaults
    EXTINCTION_BOOTSTRAP_SPAWNS: int = 2
    EXTINCTION_BOOTSTRAP_FAMILY: str = "House Nocthar"

    OFFSPRING_JITTER_RADIUS_MIN: int = 1
    OFFSPRING_JITTER_RADIUS_MAX: int = 4
    OFFSPRING_MAX_PLACEMENT_ATTEMPTS: int = 32
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
    mass: float = 2.0
    vision: float = 8.0
    hp_max: float = 20.0
    metab: float = 0.05


@dataclass
class TraitClamp:
    mass: List[float] = field(default_factory=lambda: [0.5, 8.0])
    vision: List[float] = field(default_factory=lambda: [4.0, 16.0])
    hp_max: List[float] = field(default_factory=lambda: [5.0, 100.0])
    metab: List[float] = field(default_factory=lambda: [0.01, 0.4])


@dataclass
class TraitBudgetConfig:
    INIT_BUDGET: float = 1.0
    INIT_LOGITS: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    MIN_BUDGET: float = 0.25
    MAX_BUDGET: float = 1.75


@dataclass
class TraitsConfig:
    INIT: TraitInit = field(default_factory=TraitInit)
    CLAMP: TraitClamp = field(default_factory=TraitClamp)
    BUDGET: TraitBudgetConfig = field(default_factory=TraitBudgetConfig)
    METAB_FORM: str = "affine_combo"  # constant | affine_combo
    METAB_COEFFS: Dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.0002,
            "per_mass": 0.0001,
            "per_vision": 0.00002,
        }
    )


@dataclass
class PhysicsConfig:
    K_WALL_PENALTY: float = 0.5
    K_RAM_PENALTY: float = 0.1
    K_IDLE_HIT_PENALTY: float = 0.8
    K_WINNER_DAMAGE: float = 0.2
    K_LOSER_DAMAGE: float = 0.6
    MOVE_FAIL_COST: float = -0.2
    TIE_BREAKER: str = "strength_then_lowest_id"  # strength_then_lowest_id | random_seeded


@dataclass
class PerceptionConfig:
    NUM_RAYS: int = 32
    RAY_FIELD_AGG: str = "max_abs"
    RAY_STEP_SAMPLER: str = "dda_first_hit"

    CANONICAL_RAY_FEATURES: int = 8
    CANONICAL_SELF_FEATURES: int = 11
    CANONICAL_CONTEXT_FEATURES: int = 3

    LEGACY_RAY_FEATURES: int = 5
    LEGACY_STATE_FEATURES: int = 2
    LEGACY_GENOME_FEATURES: int = 4
    LEGACY_POSITION_FEATURES: int = 2
    LEGACY_CONTEXT_FEATURES: int = 3
    LEGACY_ADAPTER_MODE: str = "prompt2_canonical_bridge_v1"

    ZONE_RATE_ABS_MAX: float = 5.0
    AGE_NORM_TICKS: int = 1024
    RETURN_CANONICAL_OBSERVATIONS: bool = True


@dataclass
class BloodlineFamilySpec:
    hidden_widths: List[int] = field(default_factory=lambda: [256, 256, 192])
    activation: str = "gelu"
    normalization: str = "pre"
    residual: bool = True
    gated: bool = False
    split_inputs: bool = False
    split_ray_width: int = 0
    split_scalar_width: int = 0
    dropout: float = 0.0


@dataclass
class BrainConfig:
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
            "House Nocthar": [98, 84, 116],
            "House Vespera": [90, 78, 124],
            "House Umbrael": [70, 96, 112],
            "House Mourndveil": [120, 82, 90],
            "House Somnyr": [88, 98, 132],
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
    GAMMA: float = 0.99
    LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENTROPY_COEF: float = 0.001
    VALUE_COEF: float = 0.5
    LR: float = 3e-4
    BATCH_SZ: int = 8
    MINI_BATCHES: int = 8
    EPOCHS: int = 4
    TARGET_KL: float = 0.01
    GRAD_NORM_CLIP: float = 1.0
    REWARD_FORM: str = "sq_health_ratio"
    UPDATE_EVERY_N_TICKS: int = 256

    OWNERSHIP_MODE: str = "uid_strict"
    BUFFER_SCHEMA_VERSION: int = 1
    TRACK_TRAINING_STATE: bool = True
    FAMILY_AWARE_UPDATE_ORDERING: bool = True
    REQUIRE_BOOTSTRAP_FOR_ACTIVE_BUFFER: bool = True
    COUNT_TRUNCATED_ROLLOUTS: bool = True
    DROP_INACTIVE_UID_BUFFERS_AFTER_FINALIZATION: bool = True
    STRICT_BUFFER_VALIDATION: bool = True


@dataclass
class EvolutionConfig:
    SELECTION: str = "softmax_fitness"
    FITNESS_DECAY: float = 0.99
    POLICY_NOISE_SD: float = 0.01
    FITNESS_TEMP: float = 1.0

    # Prompt 5 mutation knobs.
    TRAIT_LOGIT_MUTATION_SIGMA: float = 0.12
    TRAIT_BUDGET_MUTATION_SIGMA: float = 0.05
    RARE_MUT_PROB: float = 0.0005
    RARE_TRAIT_LOGIT_MUTATION_SIGMA: float = 0.40
    RARE_TRAIT_BUDGET_MUTATION_SIGMA: float = 0.15
    RARE_POLICY_NOISE_SD: float = 0.03

    ENABLE_FAMILY_SHIFT_MUTATION: bool = False
    FAMILY_SHIFT_PROB: float = 0.0001


@dataclass
class ViewerConfig:
    FPS: int = 30
    PAINT_BRUSH: List[int] = field(default_factory=lambda: [3, 3])
    PAINT_RATE_STEP: float = 0.05
    SHOW_OVERLAYS: Dict[str, bool] = field(default_factory=lambda: {"h_rate": True, "h_grad": False, "rays": False})
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 800
    CELL_SIZE: int = 10
    SHOW_BLOODLINE_LEGEND: bool = True
    BLOODLINE_LOW_HP_SHADE: float = 0.35

    # Prompt 6 viewer catastrophe surfaces.
    SHOW_CATASTROPHE_PANEL: bool = True
    SHOW_CATASTROPHE_OVERLAY: bool = True
    SHOW_CATASTROPHE_STATUS_IN_HUD: bool = True


@dataclass
class LogConfig:
    DIR: str = "logs"
    LOG_TICK_EVERY: int = 250
    SNAPSHOT_EVERY: int = 500
    ASSERTIONS: bool = True
    AMP: bool = True


@dataclass
class IdentityConfig:
    ENABLE_UID_SUBSTRATE: bool = True
    OWNERSHIP_MODE: str = "uid_bridge"
    ASSERT_BINDINGS: bool = True
    ASSERT_HISTORICAL_UIDS: bool = True
    ASSERT_NO_SLOT_OWNERSHIP_LEAK: bool = True
    MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS: bool = True


@dataclass
class SchemaConfig:
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
    SAVE_CHECKPOINT_MANIFEST: bool = True

    # Prompt 7 atomic publish and corruption controls.
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


@dataclass
class TelemetryConfig:
    ENABLE_DEEP_LEDGERS: bool = True
    LOG_LIFE_LEDGER: bool = True
    LOG_BIRTH_LEDGER: bool = True
    LOG_DEATH_LEDGER: bool = True
    LOG_PPO_UPDATE_LEDGER: bool = True
    LOG_CATASTROPHE_EVENT_LEDGER: bool = True
    LOG_TICK_SUMMARY: bool = True
    LOG_FAMILY_SUMMARY: bool = True
    FAMILY_SUMMARY_EVERY_TICKS: int = 1
    SUMMARY_EXPORT_CADENCE_TICKS: int = 1
    EXPORT_LINEAGE: bool = True
    LINEAGE_EXPORT_FORMAT: str = "json"  # json
    FLUSH_OPEN_LIVES_ON_CLOSE: bool = True
    TRACK_CATASTROPHE_EXPOSURE: bool = True
    ENABLE_VIEWER_INSPECTOR_ENRICHMENT: bool = True


@dataclass
class ValidationConfig:
    ENABLE_FINAL_AUDIT_HARNESS: bool = True
    ENABLE_DETERMINISM_TESTS: bool = True
    ENABLE_RESUME_CONSISTENCY_TESTS: bool = True
    ENABLE_SAVE_LOAD_SAVE_TESTS: bool = True
    ENABLE_CATASTROPHE_REPRO_TESTS: bool = True
    VALIDATION_STRICTNESS: str = "strict"  # permissive | strict
    AUDIT_DEFAULT_TICKS: int = 16
    DETERMINISM_COMPARE_TICKS: int = 8
    SAVE_LOAD_SAVE_COMPARE_BUFFERS: bool = True
    STRICT_TELEMETRY_SCHEMA_WRITES: bool = True


@dataclass
class MigrationConfig:
    LOG_LEGACY_SLOT_FIELDS: bool = True
    LOG_UID_FIELDS: bool = True
    VIEWER_SHOW_SLOT_AND_UID: bool = True
    VIEWER_SHOW_BLOODLINE: bool = True
    REQUIRE_CANONICAL_UID_PATHS: bool = True


@dataclass
class CatastropheConfig:
    ENABLED: bool = True
    DEFAULT_MODE: str = "auto_dynamic"  # off | manual_only | auto_dynamic | auto_static
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

    # Prompt 6 catastrophe intensity and scheduler knobs.
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
