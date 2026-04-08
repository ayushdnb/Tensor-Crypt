"""
Runtime assembly for Tensor Crypt.

The runtime builder wires together the stable subsystem graph for one
simulation run. This module is intentionally allowed to know about many
subsystems at once because assembly is its sole responsibility.

Critical invariant:
- the order of map generation, initial spawn, engine construction, and viewer
  construction must remain stable unless simulation semantics are explicitly
  being changed. This refactor is not changing those semantics.
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


def validate_runtime_config() -> None:
    """Reject silent no-op config combinations on unsupported runtime surfaces."""
    if str(cfg.SIM.DTYPE).lower() != "float32":
        raise ValueError("SIM.DTYPE currently supports only 'float32'")
    if cfg.SIM.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("SIM.DEVICE requests CUDA but torch.cuda.is_available() is False")
    if str(cfg.AGENTS.SPAWN_MODE).lower() != "uniform":
        raise ValueError("AGENTS.SPAWN_MODE currently supports only 'uniform'")
    if str(cfg.TRAITS.METAB_FORM).lower() != "affine_combo":
        raise ValueError("TRAITS.METAB_FORM currently supports only 'affine_combo'")
    if str(cfg.RESPAWN.MODE).lower() != "binary_parented":
        raise ValueError("RESPAWN.MODE currently supports only 'binary_parented'")
    validate_ppo_reward_config()
    if str(cfg.PPO.OWNERSHIP_MODE).lower() != "uid_strict":
        raise ValueError("PPO.OWNERSHIP_MODE currently supports only 'uid_strict'")
    if cfg.CHECKPOINT.STRICT_MANIFEST_VALIDATION and not cfg.CHECKPOINT.MANIFEST_ENABLED:
        raise ValueError("STRICT_MANIFEST_VALIDATION requires MANIFEST_ENABLED")
    if cfg.CHECKPOINT.WRITE_LATEST_POINTER and not cfg.CHECKPOINT.MANIFEST_ENABLED:
        raise ValueError("WRITE_LATEST_POINTER requires MANIFEST_ENABLED")
    if int(cfg.CHECKPOINT.SAVE_EVERY_TICKS) < 0:
        raise ValueError("CHECKPOINT.SAVE_EVERY_TICKS must be >= 0")
    if int(cfg.TELEMETRY.PARQUET_BATCH_ROWS) <= 0:
        raise ValueError("TELEMETRY.PARQUET_BATCH_ROWS must be positive")


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
