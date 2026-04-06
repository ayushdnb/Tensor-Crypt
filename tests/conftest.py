# ruff: noqa: E402

"""Shared pytest fixtures for Tensor Crypt."""

import copy
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from tensor_crypt.app.runtime import build_runtime, setup_determinism
from tensor_crypt.config_bridge import cfg
from tensor_crypt.telemetry.run_paths import create_run_directory


def _restore_cfg(snapshot):
    for field in dataclasses.fields(snapshot):
        setattr(cfg, field.name, copy.deepcopy(getattr(snapshot, field.name)))


@pytest.fixture(autouse=True)
def reset_cfg():
    snapshot = copy.deepcopy(cfg)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    yield
    _restore_cfg(snapshot)


@pytest.fixture
def runtime_builder(tmp_path):
    runtimes = []

    def _build(
        *,
        seed=42,
        width=16,
        height=16,
        agents=8,
        walls=0,
        hzones=0,
        update_every=4,
        batch_size=4,
        mini_batches=2,
        max_ticks=999999,
        policy_noise=0.01,
    ):
        cfg.SIM.SEED = seed
        cfg.SIM.DEVICE = "cpu"
        cfg.LOG.AMP = False
        cfg.LOG.DIR = str(tmp_path / f"logs_seed_{seed}_{len(runtimes)}")
        cfg.GRID.W = width
        cfg.GRID.H = height
        cfg.AGENTS.N = agents
        cfg.MAPGEN.RANDOM_WALLS = walls
        cfg.MAPGEN.WALL_SEG_MIN = 2
        cfg.MAPGEN.WALL_SEG_MAX = 4
        cfg.MAPGEN.HEAL_ZONE_COUNT = hzones
        cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO = 0.2
        cfg.PERCEPT.NUM_RAYS = 8
        cfg.PPO.BATCH_SZ = batch_size
        cfg.PPO.MINI_BATCHES = mini_batches
        cfg.PPO.EPOCHS = 1
        cfg.PPO.UPDATE_EVERY_N_TICKS = update_every
        cfg.RESPAWN.POPULATION_FLOOR = max(1, agents // 4)
        cfg.RESPAWN.POPULATION_CEILING = agents
        cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = max(1, agents // 2)
        cfg.RESPAWN.RESPAWN_PERIOD = 1
        cfg.LOG.LOG_TICK_EVERY = max_ticks
        cfg.LOG.SNAPSHOT_EVERY = max_ticks
        cfg.EVOL.POLICY_NOISE_SD = policy_noise
        setup_determinism()
        run_dir = create_run_directory()
        runtime = build_runtime(run_dir)
        runtimes.append(runtime)
        return runtime

    yield _build

    for runtime in reversed(runtimes):
        try:
            runtime.data_logger.close()
        except Exception:
            pass

    try:
        import pygame

        if pygame.get_init():
            pygame.quit()
    except Exception:
        pass

