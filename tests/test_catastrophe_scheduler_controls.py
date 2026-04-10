import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch

from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.population.evolution import Evolution
from tensor_crypt.simulation.engine import Engine
from tensor_crypt.telemetry.data_logger import DataLogger
from tensor_crypt.viewer.main import Viewer
from tensor_crypt.world.perception import Perception
from tensor_crypt.world.physics import Physics
from tensor_crypt.world.spatial_grid import Grid


def _make_engine(tmp_path, *, seed=42):
    cfg.SIM.SEED = seed
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.LOG.DIR = str(tmp_path)
    cfg.GRID.W = 18
    cfg.GRID.H = 18
    cfg.AGENTS.N = 8
    cfg.MAPGEN.HEAL_ZONE_COUNT = 2
    cfg.PERCEPT.NUM_RAYS = 8
    cfg.RESPAWN.RESPAWN_PERIOD = 1
    cfg.RESPAWN.POPULATION_FLOOR = 2
    cfg.RESPAWN.POPULATION_CEILING = 8
    cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE = 2

    grid = Grid()
    registry = Registry()
    physics = Physics(grid, registry)
    perception = Perception(grid, registry)
    ppo = PPO()
    evolution = Evolution(registry)
    logger = DataLogger(str(tmp_path))
    engine = Engine(grid, registry, physics, perception, ppo, evolution, logger)

    grid.add_hzone(2, 2, 4, 4, 0.8)
    grid.add_hzone(10, 10, 12, 12, 0.8)
    registry.spawn_agent(0, 6, 6, parent_uid=-1, grid=grid)
    registry.spawn_agent(1, 8, 8, parent_uid=-1, grid=grid)
    return engine, logger


_TEST_ROOT = Path("manual_debug_logs") / "catastrophe_scheduler_controls_pytest"
_CASE_COUNTER = 0


def _case_dir(name: str) -> Path:
    global _CASE_COUNTER
    _CASE_COUNTER += 1
    path = _TEST_ROOT / f"{name}_{_CASE_COUNTER}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_set_mode_plans_from_current_tick_and_disarms_non_auto_modes():
    cfg.CATASTROPHE.DEFAULT_MODE = "manual_only"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS = 4

    engine, logger = _make_engine(_case_dir("set_mode"))
    engine.catastrophes.set_mode("auto_static", current_tick=10)
    status = engine.catastrophes.build_status(10)
    assert status["mode"] == "auto_static"
    assert status["scheduler_armed"] is True
    assert status["next_auto_tick"] == 14

    engine.catastrophes.set_mode("manual_only", current_tick=11)
    status = engine.catastrophes.build_status(11)
    assert status["mode"] == "manual_only"
    assert status["scheduler_armed"] is False
    assert status["next_auto_tick"] is None
    logger.close()


def test_scheduler_toggle_disarms_without_mutating_auto_mode():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine, logger = _make_engine(_case_dir("toggle_disarms"))
    assert engine.catastrophes.build_status(0)["next_auto_tick"] == 5
    engine.catastrophes.toggle_scheduler_armed(current_tick=11)
    status = engine.catastrophes.build_status(11)
    assert status["mode"] == "auto_dynamic"
    assert status["scheduler_armed"] is False
    assert status["next_auto_tick"] is None
    assert status["auto_enabled"] is False
    logger.close()


def test_scheduler_toggle_from_manual_restores_last_auto_mode_and_plans_from_current_tick():
    cfg.CATASTROPHE.DEFAULT_MODE = "manual_only"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS = 4

    engine, logger = _make_engine(_case_dir("toggle_from_manual"))
    engine.catastrophes.last_auto_mode = "auto_static"
    armed = engine.catastrophes.toggle_scheduler_armed(current_tick=10)
    status = engine.catastrophes.build_status(10)
    assert armed is True
    assert status["mode"] == "auto_static"
    assert status["scheduler_armed"] is True
    assert status["next_auto_tick"] == 14
    logger.close()


def test_scheduler_resume_replans_when_next_tick_is_stale():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 7
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 7

    engine, logger = _make_engine(_case_dir("scheduler_resume"))
    engine.catastrophes.toggle_scheduler_pause(current_tick=0)
    engine.catastrophes._next_auto_tick = 5
    paused = engine.catastrophes.toggle_scheduler_pause(current_tick=20)
    status = engine.catastrophes.build_status(20)
    assert paused is False
    assert status["scheduler_armed"] is True
    assert status["next_auto_tick"] == 27
    logger.close()


def test_clear_active_catastrophes_restores_world_field_immediately():
    engine, logger = _make_engine(_case_dir("clear_active"))
    base = engine.grid.grid[1].clone()
    assert engine.catastrophes.manual_trigger_by_index(0, 1)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    shocked = engine.grid.grid[1].clone()
    assert not torch.equal(base, shocked)

    assert engine.catastrophes.clear_active_catastrophes(2) == 1
    restored = engine.grid.grid[1].clone()
    assert torch.equal(base, restored)
    assert engine.catastrophes.build_status(2)["active_count"] == 0
    logger.close()


def test_checkpoint_roundtrip_preserves_scheduler_armed_state():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine, logger = _make_engine(_case_dir("checkpoint_rt"))
    engine.catastrophes.toggle_scheduler_armed(current_tick=9)
    bundle = capture_runtime_checkpoint(
        type(
            "Runtime",
            (),
            {
                "registry": engine.registry,
                "grid": engine.grid,
                "ppo": engine.ppo,
                "engine": engine,
            },
        )()
    )
    engine2, logger2 = _make_engine(_case_dir("checkpoint_rt2"))
    restore_runtime_checkpoint(
        type(
            "Runtime",
            (),
            {
                "registry": engine2.registry,
                "grid": engine2.grid,
                "ppo": engine2.ppo,
                "engine": engine2,
            },
        )(),
        bundle,
    )
    status = engine2.catastrophes.build_status(9)
    assert status["mode"] == "auto_dynamic"
    assert status["scheduler_armed"] is False
    assert status["next_auto_tick"] is None
    logger.close()
    logger2.close()


def test_viewer_hotkeys_route_scheduler_toggle_pause_and_clear():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine, logger = _make_engine(_case_dir("viewer_hotkeys"))
    viewer = Viewer(engine)
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F1}))
        running, _ = viewer.input_handler.handle()
        assert running
        assert engine.catastrophes.build_status(engine.tick)["active_count"] == 1

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_u}))
        viewer.input_handler.handle()
        assert engine.catastrophes.build_status(engine.tick)["scheduler_armed"] is False

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_o}))
        viewer.input_handler.handle()
        assert engine.catastrophes.build_status(engine.tick)["scheduler_paused"] is False

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))
        viewer.input_handler.handle()
        assert engine.catastrophes.build_status(engine.tick)["active_count"] == 0
    finally:
        if pygame.get_init():
            pygame.quit()
        logger.close()

def test_default_scheduler_armed_config_controls_boot_state():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_static"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = False
    cfg.CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS = 4

    engine, logger = _make_engine(_case_dir("default_boot"))
    status = engine.catastrophes.build_status(0)
    assert status["mode"] == "auto_static"
    assert status["scheduler_armed"] is False
    assert status["next_auto_tick"] is None
    logger.close()


def test_restore_legacy_auto_enabled_payload_without_scheduler_armed():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine, logger = _make_engine(_case_dir("legacy_restore_src"))
    payload = engine.catastrophes.serialize()
    payload.pop("scheduler_armed", None)
    payload["auto_enabled"] = False
    payload["next_auto_tick"] = 99

    engine2, logger2 = _make_engine(_case_dir("legacy_restore_dst"))
    engine2.catastrophes.restore(payload)
    status = engine2.catastrophes.build_status(0)
    assert status["mode"] == "auto_dynamic"
    assert status["scheduler_armed"] is False
    assert status["next_auto_tick"] is None
    logger.close()
    logger2.close()


def test_status_and_help_text_describe_scheduler_truth():
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine, logger = _make_engine(_case_dir("status_help"))
    viewer = Viewer(engine)
    try:
        status = engine.catastrophes.build_status(0)
        line = viewer.hud_panel._compose_catastrophe_line(status)
        assert "armed/running" in line
        assert "active=0" in line
        assert "next=5" in line

        engine.catastrophes.toggle_scheduler_armed(current_tick=0)
        disarmed_line = viewer.hud_panel._compose_catastrophe_line(engine.catastrophes.build_status(0))
        assert "disarmed" in disarmed_line

        assert "Cata F1..F12: manual trigger" in viewer.side_panel.CONTROLS
        assert "Clear Active: C  Mode: Y" in viewer.side_panel.CONTROLS
        assert "Sched Arm: U  Panel: I  Sched Pause: O" in viewer.side_panel.CONTROLS
    finally:
        if pygame.get_init():
            pygame.quit()
        logger.close()