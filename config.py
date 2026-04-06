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
class TraitsConfig:
    INIT: TraitInit = field(default_factory=TraitInit)
    CLAMP: TraitClamp = field(default_factory=TraitClamp)
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
    RAY_FIELD_AGG: str = "max_abs"  # max_abs | max_pos_neg | mean | integral_kstep
    RAY_STEP_SAMPLER: str = "dda_first_hit"  # dda_first_hit | fixed_steps

    # Prompt 2 canonical observation contract.
    CANONICAL_RAY_FEATURES: int = 8
    CANONICAL_SELF_FEATURES: int = 11
    CANONICAL_CONTEXT_FEATURES: int = 3

    # Transitional Prompt 2 bridge. Prompt 3 brains no longer consume these
    # fields, but the emission path remains available for narrow compatibility.
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
    activation: str = "gelu"  # relu | gelu | silu | tanh
    normalization: str = "pre"  # pre | post | none
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
    INITIAL_FAMILY_ASSIGNMENT: str = "round_robin"  # round_robin | weighted_random
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
            "House Nocthar": BloodlineFamilySpec(
                hidden_widths=[256, 256, 224, 192],
                activation="gelu",
                normalization="pre",
                residual=True,
                gated=False,
                split_inputs=False,
                dropout=0.00,
            ),
            "House Vespera": BloodlineFamilySpec(
                hidden_widths=[160, 160, 160, 128, 128],
                activation="silu",
                normalization="pre",
                residual=True,
                gated=False,
                split_inputs=False,
                dropout=0.00,
            ),
            "House Umbrael": BloodlineFamilySpec(
                hidden_widths=[320, 320, 224],
                activation="relu",
                normalization="post",
                residual=True,
                gated=False,
                split_inputs=False,
                dropout=0.00,
            ),
            "House Mourndveil": BloodlineFamilySpec(
                hidden_widths=[224, 224, 192],
                activation="silu",
                normalization="pre",
                residual=True,
                gated=True,
                split_inputs=True,
                split_ray_width=160,
                split_scalar_width=96,
                dropout=0.00,
            ),
            "House Somnyr": BloodlineFamilySpec(
                hidden_widths=[256, 256, 256, 224, 192],
                activation="gelu",
                normalization="pre",
                residual=True,
                gated=True,
                split_inputs=True,
                split_ray_width=192,
                split_scalar_width=128,
                dropout=0.02,
            ),
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
    REWARD_FORM: str = "sq_health_ratio"  # sq_health_ratio | health_ratio | raw_health
    UPDATE_EVERY_N_TICKS: int = 256

    # Prompt 4 UID-owned training-state hardening.
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
    SELECTION: str = "softmax_fitness"  # softmax_fitness | rank
    FITNESS_DECAY: float = 0.99
    GENOME_NOISE_SD: Dict[str, float] = field(
        default_factory=lambda: {
            "mass": 0.05,
            "vision": 0.5,
            "hp_max": 0.5,
            "metab": 0.01,
        }
    )
    RARE_MUT_PROB: float = 0.0005
    RARE_MUT_SIGMA: float = 1.0
    TRAIT_COUPLING: Dict[str, float] = field(
        default_factory=lambda: {
            "mass_to_metab": 0.01,
            "vision_to_metab": 0.002,
        }
    )
    POLICY_NOISE_SD: float = 0.01
    FITNESS_TEMP: float = 1.0


@dataclass
class ViewerConfig:
    FPS: int = 30
    PAINT_BRUSH: List[int] = field(default_factory=lambda: [3, 3])
    PAINT_RATE_STEP: float = 0.05
    SHOW_OVERLAYS: Dict[str, bool] = field(
        default_factory=lambda: {
            "h_rate": True,
            "h_grad": False,
            "rays": False,
        }
    )
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 800
    CELL_SIZE: int = 10
    SHOW_BLOODLINE_LEGEND: bool = True
    BLOODLINE_LOW_HP_SHADE: float = 0.35


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
    CHECKPOINT_SCHEMA_VERSION: int = 3
    REPRODUCTION_SCHEMA_VERSION: int = 1
    TELEMETRY_SCHEMA_VERSION: int = 2
    LOGGING_SCHEMA_VERSION: int = 2


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


@dataclass
class MigrationConfig:
    LOG_LEGACY_SLOT_FIELDS: bool = True
    LOG_UID_FIELDS: bool = True
    VIEWER_SHOW_SLOT_AND_UID: bool = True
    VIEWER_SHOW_BLOODLINE: bool = True
    REQUIRE_CANONICAL_UID_PATHS: bool = True


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
    MIGRATION: MigrationConfig = field(default_factory=MigrationConfig)


cfg = Config()