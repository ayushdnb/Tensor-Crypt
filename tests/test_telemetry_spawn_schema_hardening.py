from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from tensor_crypt.app.runtime import build_runtime, setup_determinism
from tensor_crypt.config_bridge import cfg
from tensor_crypt.population.reproduction import default_trait_latent, trait_values_from_latent
from tensor_crypt.telemetry.data_logger import DataLogger
from tensor_crypt.telemetry.run_paths import create_run_directory


def _spawn_event_kwargs(
    *,
    tick: int,
    child_uid: int,
    child_slot: int,
    crowding_policy_applied,
    crowding_checked,
    crowding_neighbor_count,
):
    latent = default_trait_latent()
    traits = trait_values_from_latent(latent)
    return {
        "tick": int(tick),
        "child_slot": int(child_slot),
        "brain_parent_slot": 1,
        "trait_parent_slot": 2,
        "anchor_parent_slot": 1,
        "child_uid": int(child_uid),
        "brain_parent_uid": 11,
        "trait_parent_uid": 12,
        "anchor_parent_uid": 11,
        "child_family": "House Nocthar",
        "brain_parent_family": "House Nocthar",
        "trait_parent_family": "House Nocthar",
        "traits": dict(traits),
        "trait_latent": dict(latent),
        "mutation_flags": {
            "rare_mutation": False,
            "family_shift": False,
        },
        "placement": {
            "x": 5,
            "y": 6,
            "attempts": 1,
            "used_global_fallback": False,
            "failure_reason": "",
            "crowding_checked": crowding_checked,
            "crowding_neighbor_count": crowding_neighbor_count,
            "crowding_policy_applied": crowding_policy_applied,
        },
        "floor_recovery": False,
    }


def _build_runtime_for_spawn_schema(base: Path, *, seed: int):
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)

    cfg.SIM.SEED = seed
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.LOG.DIR = str(base / "logs")
    cfg.GRID.W = 12
    cfg.GRID.H = 12
    cfg.AGENTS.N = 6
    cfg.MAPGEN.RANDOM_WALLS = 0
    cfg.MAPGEN.WALL_SEG_MIN = 2
    cfg.MAPGEN.WALL_SEG_MAX = 4
    cfg.MAPGEN.HEAL_ZONE_COUNT = 0
    cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO = 0.2
    cfg.PERCEPT.NUM_RAYS = 8
    cfg.PPO.BATCH_SZ = 99
    cfg.PPO.MINI_BATCHES = 1
    cfg.PPO.EPOCHS = 1
    cfg.PPO.UPDATE_EVERY_N_TICKS = 99
    cfg.RESPAWN.POPULATION_FLOOR = 1
    cfg.RESPAWN.POPULATION_CEILING = 6
    cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = 3
    cfg.RESPAWN.RESPAWN_PERIOD = 1
    cfg.LOG.LOG_TICK_EVERY = 999999
    cfg.LOG.SNAPSHOT_EVERY = 999999
    cfg.EVOL.POLICY_NOISE_SD = 0.01
    setup_determinism()
    return build_runtime(create_run_directory())


def test_spawn_ledgers_survive_null_only_first_flush_then_non_null_values(monkeypatch):
    monkeypatch.setattr(cfg.TELEMETRY, "PARQUET_BATCH_ROWS", 1, raising=False)
    monkeypatch.setattr(cfg.TELEMETRY, "LOG_BIRTH_LEDGER", True, raising=False)

    base = Path(".pytest_tmp_spawn_schema_case")
    shutil.rmtree(base, ignore_errors=True)
    logger = DataLogger(str(base / "run"))
    try:
        logger.log_spawn_event(
            **_spawn_event_kwargs(
                tick=1,
                child_uid=101,
                child_slot=3,
                crowding_policy_applied=None,
                crowding_checked=None,
                crowding_neighbor_count=None,
            )
        )
        logger.log_spawn_event(
            **_spawn_event_kwargs(
                tick=2,
                child_uid=102,
                child_slot=4,
                crowding_policy_applied="block_birth",
                crowding_checked=True,
                crowding_neighbor_count=7,
            )
        )
    finally:
        logger.close()

    genealogy_df = pd.read_parquet(logger.genealogy_path).sort_values("child_uid").reset_index(drop=True)
    birth_df = pd.read_parquet(logger.birth_ledger_path).sort_values("child_uid").reset_index(drop=True)

    for df in (genealogy_df, birth_df):
        assert "placement_crowding_policy_applied" in df.columns
        assert "placement_crowding_checked" in df.columns
        assert "placement_crowding_neighbor_count" in df.columns

        assert pd.isna(df.loc[0, "placement_crowding_policy_applied"])
        assert pd.isna(df.loc[0, "placement_crowding_checked"])
        assert pd.isna(df.loc[0, "placement_crowding_neighbor_count"])

        assert df.loc[1, "placement_crowding_policy_applied"] == "block_birth"
        assert bool(df.loc[1, "placement_crowding_checked"]) is True
        assert int(df.loc[1, "placement_crowding_neighbor_count"]) == 7

    shutil.rmtree(base, ignore_errors=True)


def test_birth_ledger_bootstrap_rows_do_not_drop_later_overlay_columns(monkeypatch):
    monkeypatch.setattr(cfg.TELEMETRY, "PARQUET_BATCH_ROWS", 1, raising=False)
    monkeypatch.setattr(cfg.RESPAWN.OVERLAYS.CROWDING, "ENABLED", True, raising=False)
    monkeypatch.setattr(cfg.RESPAWN.OVERLAYS.CROWDING, "POLICY_WHEN_CROWDED", "block_birth", raising=False)

    base = Path(".pytest_tmp_spawn_schema_runtime_case")
    runtime = _build_runtime_for_spawn_schema(base, seed=701)
    try:
        monkeypatch.setattr(cfg.RESPAWN.OVERLAYS.CROWDING, "LOCAL_RADIUS", max(runtime.grid.W, runtime.grid.H), raising=False)
        monkeypatch.setattr(cfg.RESPAWN.OVERLAYS.CROWDING, "MAX_NEIGHBORS", 0, raising=False)

        registry = runtime.registry
        slot = int(registry.get_alive_indices()[0].item())
        registry.mark_dead(slot, runtime.grid)
        runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=3)
        runtime.engine.respawn_controller.step(3, registry, runtime.grid, runtime.data_logger)
        runtime.data_logger.close(runtime.registry)

        birth_df = pd.read_parquet(runtime.data_logger.birth_ledger_path)
        birth_df = birth_df.sort_values(["birth_tick", "child_slot"]).reset_index(drop=True)
        failed_row = birth_df[(birth_df["birth_tick"] == 3) & (birth_df["child_slot"] == slot)].iloc[-1]

        assert "placement_crowding_policy_applied" in birth_df.columns
        assert failed_row["placement_failure_reason"] == "crowding_blocked"
        assert failed_row["placement_crowding_policy_applied"] == "block_birth"
    finally:
        runtime.data_logger.close()
        try:
            import pygame

            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass
        shutil.rmtree(base, ignore_errors=True)
