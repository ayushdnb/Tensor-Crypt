import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch

from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.config_bridge import cfg
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.population.evolution import Evolution
from tensor_crypt.simulation.engine import Engine
from tensor_crypt.telemetry.data_logger import DataLogger
from tensor_crypt.world.perception import Perception
from tensor_crypt.world.physics import Physics
from tensor_crypt.world.spatial_grid import Grid
from tensor_crypt.viewer.main import Viewer


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


def test_auto_dynamic_scheduler_is_deterministic_under_fixed_seed(tmp_path):
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 5
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 5

    engine_a, logger_a = _make_engine(tmp_path / "a", seed=123)
    engine_b, logger_b = _make_engine(tmp_path / "b", seed=123)

    for tick in range(20):
        engine_a.catastrophes.pre_tick(tick)
        engine_b.catastrophes.pre_tick(tick)

    assert engine_a.catastrophes.serialize()["active"] == engine_b.catastrophes.serialize()["active"]
    logger_a.close()
    logger_b.close()

def test_dynamic_scheduler_gap_bounds_are_respected(tmp_path):
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_dynamic"
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 7
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 9

    engine, logger = _make_engine(tmp_path)
    gaps = [engine.catastrophes._sample_dynamic_gap() for _ in range(50)]
    assert min(gaps) >= 7
    assert max(gaps) <= 9
    logger.close()

def test_static_scheduler_interval_and_round_robin_are_respected(tmp_path):
    cfg.CATASTROPHE.DEFAULT_MODE = "auto_static"
    cfg.CATASTROPHE.DEFAULT_SCHEDULER_ARMED = True
    cfg.CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS = 4
    cfg.CATASTROPHE.AUTO_STATIC_ORDERING_POLICY = "round_robin"

    engine, logger = _make_engine(tmp_path)
    first = engine.catastrophes._select_static_type()
    second = engine.catastrophes._select_static_type()
    assert first != second
    engine.catastrophes._plan_next_auto_tick(10)
    assert engine.catastrophes.build_status(10)["next_auto_tick"] == 14
    logger.close()

def test_manual_trigger_and_clear_path_works(tmp_path):
    engine, logger = _make_engine(tmp_path)
    assert engine.catastrophes.manual_trigger_by_index(0, 5)
    assert engine.catastrophes.build_status(5)["active_count"] == 1
    assert engine.catastrophes.manual_clear(6) == 1
    assert engine.catastrophes.build_status(6)["active_count"] == 0
    logger.close()

def test_overlap_policy_and_max_concurrent_are_respected(tmp_path):
    cfg.CATASTROPHE.ALLOW_OVERLAP = False
    cfg.CATASTROPHE.MAX_CONCURRENT = 1
    engine, logger = _make_engine(tmp_path)
    assert engine.catastrophes.manual_trigger_by_index(0, 1)
    assert not engine.catastrophes.manual_trigger_by_index(1, 1)
    logger.close()

def test_zone_catastrophe_apply_and_revert_without_leak(tmp_path):
    engine, logger = _make_engine(tmp_path)
    base = engine.grid.grid[1].clone()
    assert engine.catastrophes.manual_trigger_by_index(0, 1)  # Ashfall
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    shocked = engine.grid.grid[1].clone()
    assert not torch.equal(base, shocked)
    engine.catastrophes.manual_clear(2)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(2)
    restored = engine.grid.grid[1].clone()
    assert torch.equal(base, restored)
    logger.close()

def test_border_creep_and_woundtide_modify_world_field_coherently(tmp_path):
    cfg.CATASTROPHE.ALLOW_OVERLAP = True
    cfg.CATASTROPHE.MAX_CONCURRENT = 2
    engine, logger = _make_engine(tmp_path)
    assert engine.catastrophes._start_event("the_woundtide", 1, manual=True)
    assert engine.catastrophes._start_event("the_thorn_march", 1, manual=True)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(2)
    status = engine.catastrophes.build_status(2)
    assert status["woundtide_front_x"] is not None
    assert status["thorn_march_safe_rect"] is not None
    logger.close()

def test_barren_hymn_blocks_reproduction_only_while_active(tmp_path):
    engine, logger = _make_engine(tmp_path)
    engine.catastrophes._start_event("the_barren_hymn", 1, manual=True)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    assert not engine.respawn_controller.reproduction_enabled_override
    engine.catastrophes.manual_clear(2)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(2)
    assert engine.respawn_controller.reproduction_enabled_override
    logger.close()

def test_witchstorm_only_modifies_mutation_overrides_while_active(tmp_path):
    engine, logger = _make_engine(tmp_path)
    engine.catastrophes._start_event("the_witchstorm", 1, manual=True)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    assert engine.respawn_controller.mutation_overrides["policy_noise_scalar"] > 1.0
    engine.catastrophes.manual_clear(2)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(2)
    assert engine.respawn_controller.mutation_overrides == {}
    logger.close()

def test_veil_of_somnyr_scales_effective_vision_without_mutating_trait_tensor(tmp_path):
    engine, logger = _make_engine(tmp_path)
    slot = 0
    base_trait = float(engine.registry.data[engine.registry.VISION, slot].item())
    engine.catastrophes._start_event("veil_of_somnyr", 1, manual=True)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    assert engine.perception.get_effective_vision_for_slot(slot) < base_trait
    assert float(engine.registry.data[engine.registry.VISION, slot].item()) == base_trait
    logger.close()


def test_checkpoint_roundtrip_restores_catastrophe_state(tmp_path):
    from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint

    engine, logger = _make_engine(tmp_path / "rt")
    engine.catastrophes._start_event("glass_requiem", 3, manual=True)
    bundle = capture_runtime_checkpoint(type("Runtime", (), {
        "registry": engine.registry,
        "grid": engine.grid,
        "ppo": engine.ppo,
        "engine": engine,
    })())
    engine2, logger2 = _make_engine(tmp_path / "rt2")
    restore_runtime_checkpoint(type("Runtime", (), {
        "registry": engine2.registry,
        "grid": engine2.grid,
        "ppo": engine2.ppo,
        "engine": engine2,
    })(), bundle)
    assert engine2.catastrophes.build_status(3)["active_names"] == ["Glass Requiem"]
    logger.close()
    logger2.close()

def test_veil_of_somnyr_updates_canonical_vision_feature_only(tmp_path):
    engine, logger = _make_engine(tmp_path)
    slot = 0
    before = engine.perception.build_observations(torch.tensor([slot], dtype=torch.long))["canonical_self"][0].clone()
    engine.catastrophes._start_event("veil_of_somnyr", 1, manual=True)
    engine.grid.paint_hzones()
    engine.catastrophes.apply_world_overrides(1)
    after = engine.perception.build_observations(torch.tensor([slot], dtype=torch.long))["canonical_self"][0]

    assert after[4].item() < before[4].item()
    assert torch.isclose(after[3], before[3])
    logger.close()


def test_viewer_hotkeys_route_to_catastrophe_manager(tmp_path):
    engine, logger = _make_engine(tmp_path)
    viewer = Viewer(engine)
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F1}))
        running, advance = viewer.input_handler.handle()
        assert running
        assert engine.catastrophes.build_status(engine.tick)["active_count"] == 1
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))
        viewer.input_handler.handle()
        assert engine.catastrophes.build_status(engine.tick)["active_count"] == 0
    finally:
        if pygame.get_init():
            pygame.quit()
        logger.close()
