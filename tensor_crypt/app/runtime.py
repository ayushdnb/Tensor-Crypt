"""
Runtime assembly for Tensor Crypt.

The runtime builder wires together the stable subsystem graph for one
simulation run. This module is intentionally allowed to know about many
subsystems at once because assembly is its sole responsibility.

Critical invariant:
- the order of map generation, initial spawn, engine construction, and viewer
  construction must remain stable unless simulation semantics are explicitly
  being changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import torch

from ..config_bridge import cfg
from ..agents.state_registry import Registry
from ..learning.ppo import PPO
from ..population.evolution import Evolution
from ..simulation.engine import Engine, validate_ppo_reward_config
from ..telemetry.data_logger import DataLogger
from ..viewer.main import Viewer
from ..world.perception import Perception
from ..world.physics import Physics
from ..world.procedural_map import add_random_hzones, add_random_walls
from ..world.spatial_grid import Grid


SUPPORTED_GRID_OVERLAP_MODES = frozenset({"max_abs", "sum_clamped", "last_wins"})
SUPPORTED_SPAWN_MODES = frozenset({"uniform"})
SUPPORTED_METAB_FORMS = frozenset({"affine_combo"})
SUPPORTED_INITIAL_FAMILY_ASSIGNMENTS = frozenset({"round_robin", "weighted_random"})
SUPPORTED_RESPAWN_MODES = frozenset({"binary_parented"})
SUPPORTED_ANCHOR_PARENT_SELECTORS = frozenset({"brain_parent", "trait_parent", "random_parent", "fitter_of_two"})
SUPPORTED_EXTINCTION_POLICIES = frozenset({"fail_run", "seed_bank_bootstrap", "admin_spawn_defaults"})
SUPPORTED_BIRTH_HP_MODES = frozenset({"full", "fraction"})
SUPPORTED_PPO_OWNERSHIP_MODES = frozenset({"uid_strict"})
SUPPORTED_TIE_BREAKERS = frozenset({"strength_then_lowest_id", "random_seeded"})
SUPPORTED_LINEAGE_EXPORT_FORMATS = frozenset({"json"})
SUPPORTED_CATASTROPHE_MODES = frozenset({"off", "manual_only", "auto_dynamic", "auto_static"})
SUPPORTED_CATASTROPHE_STATIC_ORDERING = frozenset({"round_robin", "configured_sequence", "fixed_priority"})
SUPPORTED_RESPAWN_CROWDING_POLICIES = frozenset({"block_birth", "global_only"})
SUPPORTED_RESPAWN_CROWDING_BELOW_FLOOR_POLICIES = frozenset({"strict", "bypass", "global_only"})
SUPPORTED_RESPAWN_COOLDOWN_EMPTY_POOL_POLICIES = frozenset({"allow_best_available", "strict"})
SUPPORTED_RESPAWN_COOLDOWN_BELOW_FLOOR_POLICIES = frozenset({"allow_best_available", "bypass", "strict"})
SUPPORTED_RESPAWN_LOCAL_PARENT_FALLBACK_POLICIES = frozenset({"global", "strict"})
SUPPORTED_RESPAWN_LOCAL_PARENT_BELOW_FLOOR_POLICIES = frozenset({"prefer_local_then_global", "bypass", "strict"})


@dataclass
class SimulationRuntime:
    """
    Concrete runtime object for one launched session.

    Keeping the assembled graph in a dataclass makes the bootstrapping boundary
    explicit without introducing a new control abstraction into the simulation
    loop itself.
    """

    run_dir: str
    data_logger: DataLogger
    grid: Grid
    registry: Registry
    physics: Physics
    perception: Perception
    ppo: PPO
    evolution: Evolution
    engine: Engine
    viewer: Viewer


def _require_choice(label: str, value: object, supported: frozenset[str]) -> str:
    normalized = str(value).lower()
    if normalized not in supported:
        supported_text = ", ".join(sorted(supported))
        raise ValueError(f"{label} currently supports only {{{supported_text}}}, got {value!r}")
    return normalized


def validate_runtime_config() -> None:
    """Reject unsupported or misleading config combinations before runtime assembly."""
    _require_choice("SIM.DTYPE", cfg.SIM.DTYPE, frozenset({"float32"}))
    if cfg.SIM.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("SIM.DEVICE requests CUDA but torch.cuda.is_available() is False")

    _require_choice("GRID.HZ_OVERLAP_MODE", cfg.GRID.HZ_OVERLAP_MODE, SUPPORTED_GRID_OVERLAP_MODES)
    _require_choice("AGENTS.SPAWN_MODE", cfg.AGENTS.SPAWN_MODE, SUPPORTED_SPAWN_MODES)
    _require_choice("TRAITS.METAB_FORM", cfg.TRAITS.METAB_FORM, SUPPORTED_METAB_FORMS)
    _require_choice("BRAIN.INITIAL_FAMILY_ASSIGNMENT", cfg.BRAIN.INITIAL_FAMILY_ASSIGNMENT, SUPPORTED_INITIAL_FAMILY_ASSIGNMENTS)
    _require_choice("RESPAWN.MODE", cfg.RESPAWN.MODE, SUPPORTED_RESPAWN_MODES)
    _require_choice("RESPAWN.ANCHOR_PARENT_SELECTOR", cfg.RESPAWN.ANCHOR_PARENT_SELECTOR, SUPPORTED_ANCHOR_PARENT_SELECTORS)
    _require_choice("RESPAWN.EXTINCTION_POLICY", cfg.RESPAWN.EXTINCTION_POLICY, SUPPORTED_EXTINCTION_POLICIES)
    _require_choice("RESPAWN.BIRTH_HP_MODE", cfg.RESPAWN.BIRTH_HP_MODE, SUPPORTED_BIRTH_HP_MODES)
    _require_choice(
        "RESPAWN.OVERLAYS.CROWDING.POLICY_WHEN_CROWDED",
        cfg.RESPAWN.OVERLAYS.CROWDING.POLICY_WHEN_CROWDED,
        SUPPORTED_RESPAWN_CROWDING_POLICIES,
    )
    _require_choice(
        "RESPAWN.OVERLAYS.CROWDING.BELOW_FLOOR_POLICY",
        cfg.RESPAWN.OVERLAYS.CROWDING.BELOW_FLOOR_POLICY,
        SUPPORTED_RESPAWN_CROWDING_BELOW_FLOOR_POLICIES,
    )
    _require_choice(
        "RESPAWN.OVERLAYS.COOLDOWN.EMPTY_POOL_POLICY",
        cfg.RESPAWN.OVERLAYS.COOLDOWN.EMPTY_POOL_POLICY,
        SUPPORTED_RESPAWN_COOLDOWN_EMPTY_POOL_POLICIES,
    )
    _require_choice(
        "RESPAWN.OVERLAYS.COOLDOWN.BELOW_FLOOR_POLICY",
        cfg.RESPAWN.OVERLAYS.COOLDOWN.BELOW_FLOOR_POLICY,
        SUPPORTED_RESPAWN_COOLDOWN_BELOW_FLOOR_POLICIES,
    )
    _require_choice(
        "RESPAWN.OVERLAYS.LOCAL_PARENT.FALLBACK_BEHAVIOR",
        cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.FALLBACK_BEHAVIOR,
        SUPPORTED_RESPAWN_LOCAL_PARENT_FALLBACK_POLICIES,
    )
    _require_choice(
        "RESPAWN.OVERLAYS.LOCAL_PARENT.BELOW_FLOOR_POLICY",
        cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.BELOW_FLOOR_POLICY,
        SUPPORTED_RESPAWN_LOCAL_PARENT_BELOW_FLOOR_POLICIES,
    )
    _require_choice("PPO.OWNERSHIP_MODE", cfg.PPO.OWNERSHIP_MODE, SUPPORTED_PPO_OWNERSHIP_MODES)
    _require_choice("PHYS.TIE_BREAKER", cfg.PHYS.TIE_BREAKER, SUPPORTED_TIE_BREAKERS)
    _require_choice("TELEMETRY.LINEAGE_EXPORT_FORMAT", cfg.TELEMETRY.LINEAGE_EXPORT_FORMAT, SUPPORTED_LINEAGE_EXPORT_FORMATS)
    _require_choice("CATASTROPHE.DEFAULT_MODE", cfg.CATASTROPHE.DEFAULT_MODE, SUPPORTED_CATASTROPHE_MODES)
    _require_choice(
        "CATASTROPHE.AUTO_STATIC_ORDERING_POLICY",
        cfg.CATASTROPHE.AUTO_STATIC_ORDERING_POLICY,
        SUPPORTED_CATASTROPHE_STATIC_ORDERING,
    )

    validate_ppo_reward_config()

    if int(cfg.LOG.LOG_TICK_EVERY) <= 0:
        raise ValueError("LOG.LOG_TICK_EVERY must be positive")
    if int(cfg.LOG.SNAPSHOT_EVERY) <= 0:
        raise ValueError("LOG.SNAPSHOT_EVERY must be positive")
    if int(cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET) <= 0:
        raise ValueError("SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET must be positive")
    if int(cfg.PPO.UPDATE_EVERY_N_TICKS) <= 0:
        raise ValueError("PPO.UPDATE_EVERY_N_TICKS must be positive")
    if int(cfg.PPO.BATCH_SZ) <= 0:
        raise ValueError("PPO.BATCH_SZ must be positive")
    if int(cfg.PPO.MINI_BATCHES) <= 0:
        raise ValueError("PPO.MINI_BATCHES must be positive")
    if int(cfg.PPO.EPOCHS) <= 0:
        raise ValueError("PPO.EPOCHS must be positive")
    if int(cfg.CHECKPOINT.SAVE_EVERY_TICKS) < 0:
        raise ValueError("CHECKPOINT.SAVE_EVERY_TICKS must be >= 0")
    if int(cfg.RESPAWN.OVERLAYS.CROWDING.LOCAL_RADIUS) < 0:
        raise ValueError("RESPAWN.OVERLAYS.CROWDING.LOCAL_RADIUS must be >= 0")
    if int(cfg.RESPAWN.OVERLAYS.CROWDING.MAX_NEIGHBORS) < 0:
        raise ValueError("RESPAWN.OVERLAYS.CROWDING.MAX_NEIGHBORS must be >= 0")
    if int(cfg.RESPAWN.OVERLAYS.COOLDOWN.DURATION_TICKS) < 0:
        raise ValueError("RESPAWN.OVERLAYS.COOLDOWN.DURATION_TICKS must be >= 0")
    if int(cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.SELECTION_RADIUS) < 0:
        raise ValueError("RESPAWN.OVERLAYS.LOCAL_PARENT.SELECTION_RADIUS must be >= 0")
    if int(cfg.TELEMETRY.PARQUET_BATCH_ROWS) <= 0:
        raise ValueError("TELEMETRY.PARQUET_BATCH_ROWS must be positive")
    if int(cfg.CATASTROPHE.MAX_CONCURRENT) <= 0:
        raise ValueError("CATASTROPHE.MAX_CONCURRENT must be positive")
    if int(cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS) <= 0:
        raise ValueError("CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS must be positive")
    if int(cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS) < int(cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS):
        raise ValueError("CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS must be >= CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS")
    if int(cfg.CATASTROPHE.MIN_DURATION_TICKS) <= 0:
        raise ValueError("CATASTROPHE.MIN_DURATION_TICKS must be positive")
    if int(cfg.CATASTROPHE.MAX_DURATION_TICKS) < int(cfg.CATASTROPHE.MIN_DURATION_TICKS):
        raise ValueError("CATASTROPHE.MAX_DURATION_TICKS must be >= CATASTROPHE.MIN_DURATION_TICKS")
    if cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE:
        try:
            import torch.func  # noqa: F401
        except Exception as exc:
            raise ValueError(
                "SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE requires torch.func support in the current PyTorch build"
            ) from exc
    if cfg.RESPAWN.OVERLAYS.COOLDOWN.ENABLED and not any(
        [
            cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_BRAIN_PARENT,
            cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_TRAIT_PARENT,
            cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_ANCHOR_PARENT,
        ]
    ):
        raise ValueError(
            "RESPAWN.OVERLAYS.COOLDOWN.ENABLED requires at least one APPLY_TO_* flag to be True"
        )

    manifests_published = bool(
        cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED
        and cfg.CHECKPOINT.MANIFEST_ENABLED
        and cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST
    )
    if cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST and not cfg.CHECKPOINT.MANIFEST_ENABLED:
        raise ValueError("CHECKPOINT.SAVE_CHECKPOINT_MANIFEST requires CHECKPOINT.MANIFEST_ENABLED")
    if cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST and not cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED:
        raise ValueError(
            "CHECKPOINT.SAVE_CHECKPOINT_MANIFEST currently requires CHECKPOINT.ATOMIC_WRITE_ENABLED because manifest publication only exists on the atomic save path"
        )
    if cfg.CHECKPOINT.STRICT_MANIFEST_VALIDATION and not manifests_published:
        raise ValueError(
            "CHECKPOINT.STRICT_MANIFEST_VALIDATION requires manifest publication in the current runtime (ATOMIC_WRITE_ENABLED, MANIFEST_ENABLED, and SAVE_CHECKPOINT_MANIFEST)"
        )
    if cfg.CHECKPOINT.WRITE_LATEST_POINTER and not manifests_published:
        raise ValueError(
            "CHECKPOINT.WRITE_LATEST_POINTER requires manifest publication in the current runtime (ATOMIC_WRITE_ENABLED, MANIFEST_ENABLED, and SAVE_CHECKPOINT_MANIFEST)"
        )


def setup_determinism() -> None:
    """Set all random seeds for reproducibility."""
    validate_runtime_config()
    torch.manual_seed(cfg.SIM.SEED)
    random.seed(cfg.SIM.SEED)
    np.random.seed(cfg.SIM.SEED)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(cfg.SIM.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_runtime(run_dir: str) -> SimulationRuntime:
    """
    Assemble the simulation and viewer graph.

    This function preserves the original launch sequence after the logging path
    has been chosen by the outer entrypoint.
    """

    validate_runtime_config()
    data_logger = DataLogger(run_dir)

    grid = Grid()
    registry = Registry()
    physics = Physics(grid, registry)
    perception = Perception(grid, registry)
    ppo = PPO()
    evolution = Evolution(registry)

    print("Generating procedural map...")
    add_random_walls(grid)
    add_random_hzones(grid)
    physics.refresh_static_wall_cache()
    print("Map generation complete.")

    registry.spawn_initial_population(grid)

    engine = Engine(grid, registry, physics, perception, ppo, evolution, data_logger)
    viewer = Viewer(engine)

    return SimulationRuntime(
        run_dir=run_dir,
        data_logger=data_logger,
        grid=grid,
        registry=registry,
        physics=physics,
        perception=perception,
        ppo=ppo,
        evolution=evolution,
        engine=engine,
        viewer=viewer,
    )
