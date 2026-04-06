# ruff: noqa: E402

"""Headless soak runner for long-form invariant checks."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame
import torch

from tensor_crypt.app.runtime import build_runtime, setup_determinism
from tensor_crypt.checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    load_runtime_checkpoint,
    save_runtime_checkpoint,
    validate_runtime_checkpoint,
)
from tensor_crypt.config_bridge import cfg
from tensor_crypt.telemetry.run_paths import create_run_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible headless soak audit for Tensor Crypt.")
    parser.add_argument("--ticks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--walls", type=int, default=4)
    parser.add_argument("--hzones", type=int, default=2)
    parser.add_argument("--log-dir", default="audit_tmp/soak_logs")
    parser.add_argument("--checkpoint-validate-every", type=int, default=32)
    return parser.parse_args()


def _validate_runtime_state(runtime) -> None:
    runtime.registry.check_invariants(runtime.grid)
    if not torch.isfinite(runtime.registry.data).all():
        raise RuntimeError("Registry state became non-finite during soak run")
    if not torch.isfinite(runtime.grid.grid).all():
        raise RuntimeError("Grid state became non-finite during soak run")

    known_uids = set(runtime.registry.uid_lifecycle.keys())
    for uid, buffer in runtime.ppo.buffers_by_uid.items():
        if uid not in known_uids:
            raise RuntimeError(f"PPO buffer key {uid} is not a known lifecycle UID")
        buffer.validate()
    for uid in runtime.ppo.optimizers_by_uid:
        if uid not in known_uids:
            raise RuntimeError(f"PPO optimizer key {uid} is not a known lifecycle UID")

    for slot_idx, brain in enumerate(runtime.registry.brains):
        if brain is None:
            continue
        for name, param in brain.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(f"Brain parameter became non-finite during soak run: slot={slot_idx} param={name}")


def _validate_checkpoint_surfaces(runtime, checkpoint_path: Path) -> None:
    bundle = capture_runtime_checkpoint(runtime)
    validate_runtime_checkpoint(bundle, cfg)
    save_runtime_checkpoint(checkpoint_path, bundle)
    loaded = load_runtime_checkpoint(checkpoint_path)

    if int(loaded["engine_state"]["tick"]) != int(bundle["engine_state"]["tick"]):
        raise RuntimeError("Checkpoint save/load changed engine tick during soak validation")
    if loaded["registry_state"]["slot_uid"].tolist() != bundle["registry_state"]["slot_uid"].tolist():
        raise RuntimeError("Checkpoint save/load changed slot_uid bindings during soak validation")
    if loaded["grid_state"]["grid"].shape != bundle["grid_state"]["grid"].shape:
        raise RuntimeError("Checkpoint save/load changed grid tensor shape during soak validation")


def main():
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    cfg.SIM.SEED = args.seed
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.LOG.DIR = args.log_dir
    cfg.GRID.W = args.width
    cfg.GRID.H = args.height
    cfg.AGENTS.N = args.agents
    cfg.MAPGEN.RANDOM_WALLS = args.walls
    cfg.MAPGEN.WALL_SEG_MIN = 2
    cfg.MAPGEN.WALL_SEG_MAX = 6
    cfg.MAPGEN.HEAL_ZONE_COUNT = args.hzones
    cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO = 0.2
    cfg.PERCEPT.NUM_RAYS = 8
    cfg.PPO.BATCH_SZ = 8
    cfg.PPO.MINI_BATCHES = 2
    cfg.PPO.EPOCHS = 1
    cfg.PPO.UPDATE_EVERY_N_TICKS = 16
    cfg.RESPAWN.POPULATION_FLOOR = max(1, args.agents // 4)
    cfg.RESPAWN.POPULATION_CEILING = args.agents
    cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = max(1, args.agents // 2)
    cfg.RESPAWN.RESPAWN_PERIOD = 1
    cfg.LOG.LOG_TICK_EVERY = args.ticks + 1
    cfg.LOG.SNAPSHOT_EVERY = max(64, args.ticks + 1)

    setup_determinism()
    run_dir = create_run_directory()
    runtime = build_runtime(run_dir)
    checkpoint_path = Path(run_dir) / "soak_validation_checkpoint.pt"

    min_alive = runtime.registry.get_num_alive()
    max_alive = min_alive

    try:
        for _ in range(args.ticks):
            runtime.engine.step()
            _validate_runtime_state(runtime)

            if args.checkpoint_validate_every > 0 and runtime.engine.tick % int(args.checkpoint_validate_every) == 0:
                _validate_checkpoint_surfaces(runtime, checkpoint_path)

            alive = runtime.registry.get_num_alive()
            min_alive = min(min_alive, alive)
            max_alive = max(max_alive, alive)
    finally:
        runtime.data_logger.close()
        if pygame.get_init():
            pygame.quit()

    print(f"SOAK_OK ticks={runtime.engine.tick} alive_final={runtime.registry.get_num_alive()} min_alive={min_alive} max_alive={max_alive}")
    print(f"RUN_DIR {run_dir}")


if __name__ == "__main__":
    main()
