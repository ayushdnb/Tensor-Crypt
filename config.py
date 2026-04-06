"""Root configuration surface for Tensor Crypt.

This module remains the canonical public knob surface for the project. Internal
package code reaches it through `tensor_crypt.config_bridge` so runtime imports
do not depend on the current working directory.
"""

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
    """Configuration for procedural map generation."""

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
    """Settings for population floor/ceiling control."""

    RESPAWN_PERIOD: int = 100
    MAX_SPAWNS_PER_CYCLE: int = 10
    POPULATION_FLOOR: int = 20
    POPULATION_CEILING: int = 150  # Must be <= AgentsConfig.N.


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

    # Temporary Prompt 2 bridge into the legacy transformer brain.
    LEGACY_RAY_FEATURES: int = 5
    LEGACY_STATE_FEATURES: int = 2
    LEGACY_GENOME_FEATURES: int = 4
    LEGACY_POSITION_FEATURES: int = 2
    LEGACY_CONTEXT_FEATURES: int = 3
    LEGACY_ADAPTER_MODE: str = "prompt2_canonical_bridge_v1"

    # Explicit bounded normalization knobs.
    ZONE_RATE_ABS_MAX: float = 5.0
    AGE_NORM_TICKS: int = 1024
    RETURN_CANONICAL_OBSERVATIONS: bool = True


@dataclass
class BrainConfig:
    D_MODEL: int = 16
    N_HEADS: int = 2
    FUSION_LAYERS: int = 2
    K_QUERIES: int = 2
    USE_GRU: bool = True
    GRU_HIDDEN: int = 64
    DROPOUT: float = 0.001


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
    CHECKPOINT_SCHEMA_VERSION: int = 1
    REPRODUCTION_SCHEMA_VERSION: int = 1
    TELEMETRY_SCHEMA_VERSION: int = 1
    LOGGING_SCHEMA_VERSION: int = 1


@dataclass
class CheckpointConfig:
    ENABLE_SUBSTRATE_CHECKPOINTS: bool = True
    CAPTURE_RNG_STATE: bool = True
    CAPTURE_OPTIMIZER_STATE: bool = True
    CAPTURE_SCALER_STATE: bool = True
    STRICT_SCHEMA_VALIDATION: bool = True
    STRICT_UID_VALIDATION: bool = True
    SAVE_CHECKPOINT_MANIFEST: bool = True


@dataclass
class MigrationConfig:
    LOG_LEGACY_SLOT_FIELDS: bool = True
    LOG_UID_FIELDS: bool = True
    VIEWER_SHOW_SLOT_AND_UID: bool = True
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

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "codes" / "evolution.txt"

IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    ".mypy_cache",
    ".pytest_cache",
    ".pytest_tmp",
    "build",
    "dist",
    "audit_tmp",
}


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    py_files = sorted(
        (
            path
            for path in BASE_DIR.rglob("*.py")
            if path.is_file() and not should_ignore(path.relative_to(BASE_DIR))
        ),
        key=lambda path: str(path.relative_to(BASE_DIR)).lower(),
    )

    with OUT_FILE.open("w", encoding="utf-8", errors="replace") as out:
        for index, path in enumerate(py_files):
            try:
                code = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                code = f"# [ERROR READING FILE: {exc}]\n"

            out.write(code)

            if index < len(py_files) - 1:
                if not code.endswith("\n"):
                    out.write("\n")
                out.write("\n")

    print(f"Done. Wrote raw code from {len(py_files)} .py files into:\n{OUT_FILE}")


if __name__ == "__main__":
    main()
