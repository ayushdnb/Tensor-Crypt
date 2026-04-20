# ruff: noqa: E402

"""Headless benchmark harness for Tensor Crypt."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
import time
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import pygame
import torch

from tensor_crypt.app.runtime import build_runtime, setup_determinism
from tensor_crypt.config_bridge import cfg
from tensor_crypt.runtime_config import apply_experimental_single_family_launch_defaults
from tensor_crypt.telemetry.run_paths import create_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible headless benchmark for Tensor Crypt.")
    parser.add_argument("--ticks", type=int, default=128)
    parser.add_argument("--warmup-ticks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--walls", type=int, default=4)
    parser.add_argument("--hzones", type=int, default=2)
    parser.add_argument("--num-rays", type=int, default=8)
    parser.add_argument("--update-every", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--summary-cadence", type=int, default=1)
    parser.add_argument("--parquet-batch-rows", type=int, default=64)
    parser.add_argument("--checkpoints-every", type=int, default=0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=3)
    parser.add_argument("--log-dir", default="audit_tmp/benchmark_logs")
    parser.add_argument("--output", default="")
    parser.add_argument("--profile-top", type=int, default=0)
    parser.add_argument("--experimental-family-vmap-inference", dest="experimental_family_vmap_inference", action="store_true")
    parser.add_argument("--disable-experimental-family-vmap-inference", dest="experimental_family_vmap_inference", action="store_false")
    parser.set_defaults(experimental_family_vmap_inference=True)
    parser.add_argument("--experimental-family-vmap-min-bucket", type=int, default=8)
    return parser.parse_args()


def _configure_runtime(args: argparse.Namespace) -> None:
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    cfg.SIM.SEED = args.seed
    cfg.SIM.DEVICE = device
    cfg.LOG.AMP = device == "cuda"
    cfg.LOG.DIR = args.log_dir
    apply_experimental_single_family_launch_defaults()
    cfg.SIM.DEVICE = device
    cfg.LOG.DIR = args.log_dir
    cfg.GRID.W = args.width
    cfg.GRID.H = args.height
    cfg.AGENTS.N = args.agents
    cfg.MAPGEN.RANDOM_WALLS = args.walls
    cfg.MAPGEN.WALL_SEG_MIN = 2
    cfg.MAPGEN.WALL_SEG_MAX = 6
    cfg.MAPGEN.HEAL_ZONE_COUNT = args.hzones
    cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO = 0.2
    cfg.PERCEPT.NUM_RAYS = args.num_rays
    cfg.PPO.BATCH_SZ = args.batch_size
    cfg.PPO.MINI_BATCHES = args.mini_batches
    cfg.PPO.EPOCHS = args.epochs
    cfg.PPO.UPDATE_EVERY_N_TICKS = args.update_every
    cfg.RESPAWN.POPULATION_FLOOR = max(1, args.agents // 4)
    cfg.RESPAWN.POPULATION_CEILING = args.agents
    cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = max(1, args.agents // 2)
    cfg.RESPAWN.RESPAWN_PERIOD = 1
    cfg.LOG.LOG_TICK_EVERY = args.ticks + args.warmup_ticks + 1000
    cfg.LOG.SNAPSHOT_EVERY = args.ticks + args.warmup_ticks + 1000
    cfg.CHECKPOINT.SAVE_EVERY_TICKS = args.checkpoints_every
    cfg.CHECKPOINT.KEEP_LAST = args.checkpoint_keep_last
    cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS = args.summary_cadence
    cfg.TELEMETRY.PARQUET_BATCH_ROWS = args.parquet_batch_rows
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = bool(args.experimental_family_vmap_inference)
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = int(args.experimental_family_vmap_min_bucket)


def _zero_inference_stats() -> dict[str, int]:
    return {
        "loop_slots": 0,
        "vmap_slots": 0,
        "family_loop_buckets": 0,
        "family_vmap_buckets": 0,
    }


def _accumulate_inference_stats(total: dict[str, int], current: dict | None) -> None:
    current = current or {}
    for key in total:
        total[key] += int(current.get(key, 0))


def _profiled_ticks(runtime, ticks: int, top_n: int) -> tuple[list[str], float, dict[str, int]]:
    profiler = cProfile.Profile()
    inference_stats = _zero_inference_stats()
    profiler.enable()
    started = time.perf_counter()
    for _ in range(ticks):
        runtime.engine.step()
        _accumulate_inference_stats(inference_stats, getattr(runtime.engine, "last_inference_path_stats", None))
    if cfg.SIM.DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    profiler.disable()

    capture = StringIO()
    import pstats

    stats = pstats.Stats(profiler, stream=capture).sort_stats("cumulative")
    stats.print_stats(top_n)
    lines = [line.rstrip() for line in capture.getvalue().splitlines() if line.strip()]
    return lines, elapsed, inference_stats


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    _configure_runtime(args)
    setup_determinism()
    run_dir = create_run_directory()
    runtime = build_runtime(run_dir)
    process = psutil.Process()

    try:
        for _ in range(max(0, args.warmup_ticks)):
            runtime.engine.step()

        rss_before = process.memory_info().rss
        if cfg.SIM.DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        profile_lines: list[str] = []
        inference_stats = _zero_inference_stats()
        if args.profile_top > 0:
            profile_lines, elapsed, inference_stats = _profiled_ticks(runtime, args.ticks, args.profile_top)
        else:
            started = time.perf_counter()
            for _ in range(args.ticks):
                runtime.engine.step()
                _accumulate_inference_stats(inference_stats, getattr(runtime.engine, "last_inference_path_stats", None))
            if cfg.SIM.DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

        rss_after = process.memory_info().rss
        result = {
            "device": cfg.SIM.DEVICE,
            "ticks": int(args.ticks),
            "warmup_ticks": int(args.warmup_ticks),
            "elapsed_sec": float(elapsed),
            "ticks_per_sec": float(args.ticks / elapsed),
            "rss_before_mb": float(rss_before / (1024 * 1024)),
            "rss_after_mb": float(rss_after / (1024 * 1024)),
            "rss_delta_mb": float((rss_after - rss_before) / (1024 * 1024)),
            "cuda_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if cfg.SIM.DEVICE == "cuda" else 0.0,
            "final_tick": int(runtime.engine.tick),
            "final_alive": int(runtime.registry.get_num_alive()),
            "last_runtime_checkpoint_tick": int(runtime.engine.last_runtime_checkpoint_tick),
            "last_runtime_checkpoint_path": runtime.engine.last_runtime_checkpoint_path,
            "buffered_parquet_rows": int(runtime.data_logger.get_buffered_row_count()),
            "run_dir": str(run_dir),
            "profile_top_cumulative": profile_lines,
            "experimental_family_vmap_inference": bool(cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE),
            "experimental_family_vmap_min_bucket": int(cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET),
            "inference_path_stats": inference_stats,
        }
    finally:
        runtime.data_logger.close(runtime.registry)
        if pygame.get_init():
            pygame.quit()

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
