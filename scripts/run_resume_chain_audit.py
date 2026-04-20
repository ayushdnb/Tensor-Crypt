# ruff: noqa: E402

"""Headless repeated checkpoint/resume audit for Tensor Crypt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame

from tensor_crypt.app.runtime import build_runtime, setup_determinism
from tensor_crypt.audit.final_validation import run_resume_chain_probe
from tensor_crypt.config_bridge import cfg
from tensor_crypt.runtime_config import apply_experimental_single_family_launch_defaults
from tensor_crypt.telemetry.run_paths import create_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeated resume-chain audit for Tensor Crypt.")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--ticks-per-cycle", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--walls", type=int, default=4)
    parser.add_argument("--hzones", type=int, default=2)
    parser.add_argument("--log-dir", default="audit_tmp/resume_chain_logs")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _configure_runtime(args: argparse.Namespace) -> None:
    cfg.SIM.SEED = args.seed
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.LOG.DIR = args.log_dir
    apply_experimental_single_family_launch_defaults()
    cfg.SIM.DEVICE = "cpu"
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
    cfg.PPO.UPDATE_EVERY_N_TICKS = 8
    cfg.RESPAWN.POPULATION_FLOOR = max(1, args.agents // 4)
    cfg.RESPAWN.POPULATION_CEILING = args.agents
    cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = max(1, args.agents // 2)
    cfg.RESPAWN.RESPAWN_PERIOD = 1
    cfg.LOG.LOG_TICK_EVERY = args.cycles * args.ticks_per_cycle + 1000
    cfg.LOG.SNAPSHOT_EVERY = args.cycles * args.ticks_per_cycle + 1000


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    _configure_runtime(args)

    def factory():
        setup_determinism()
        run_dir = create_run_directory()
        return build_runtime(run_dir)

    checkpoint_dir = Path(args.log_dir) / "resume_chain_artifacts"
    report = run_resume_chain_probe(
        factory,
        checkpoint_dir,
        cycles=args.cycles,
        ticks_per_cycle=args.ticks_per_cycle,
    )
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")

    if pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    main()
