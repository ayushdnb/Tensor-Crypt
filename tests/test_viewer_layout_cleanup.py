
# ruff: noqa: E402

"""Viewer layout cleanup regression tests for Tensor Crypt."""

import copy
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame
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

    if pygame.get_init():
        pygame.quit()


def test_layout_rects_stay_inside_window_across_sizes(runtime_builder):
    runtime = runtime_builder(seed=701, width=12, height=12, agents=4, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    for window_size in ((760, 560), (920, 620), (1280, 720), (1600, 900)):
        viewer.handle_window_resize(*window_size)
        world_rect = viewer.layout.world_rect()
        side_rect = viewer.layout.side_rect()
        hud_rect = viewer.layout.hud_rect()

        assert world_rect.left >= viewer.layout.margin
        assert world_rect.top >= viewer.layout.margin
        assert world_rect.right <= side_rect.left - viewer.layout.margin
        assert side_rect.right <= viewer.Wpix - viewer.layout.margin
        assert hud_rect.left == world_rect.left
        assert hud_rect.right == world_rect.right
        assert world_rect.bottom <= hud_rect.top
        assert side_rect.top == viewer.layout.margin
        assert side_rect.bottom <= viewer.Hpix - viewer.layout.margin


def test_videoresize_event_updates_geometry_without_refitting(runtime_builder):
    runtime = runtime_builder(seed=702, width=12, height=12, agents=4, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    viewer.cam.zoom_at(1.5, 40, 40)
    old_cell_px = viewer.cam.cell_px

    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, {"w": 1400, "h": 900}))
    running, advance_tick = viewer.input_handler.handle()
    world_rect = viewer.layout.world_rect()

    assert running is True
    assert advance_tick is False
    assert viewer.cam.screen_width == world_rect.width
    assert viewer.cam.screen_height == world_rect.height
    assert viewer.cam.cell_px == old_cell_px


def test_side_panel_scrolls_on_mousewheel_when_content_overflows(runtime_builder, monkeypatch):
    cfg.TELEMETRY.ENABLE_VIEWER_INSPECTOR_ENRICHMENT = True
    runtime = runtime_builder(seed=703, width=12, height=12, agents=8, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    viewer.handle_window_resize(760, 560)
    viewer.selected_slot_id = int(runtime.registry.get_alive_indices()[0].item())
    viewer._last_state_data = viewer._prepare_state_data()
    viewer.side_panel.clamp_scroll_offset(viewer._last_state_data)

    side_rect = viewer.layout.side_rect()
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: side_rect.center)

    assert viewer.side_panel._content_height(viewer._last_state_data) > viewer.side_panel._content_rect(
        viewer.layout.side_rect(),
        viewer.side_panel._metrics(),
    ).height

    start_offset = viewer.side_panel.scroll_offset
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": -1}))
    viewer.input_handler.handle()

    assert viewer.side_panel.scroll_offset > start_offset


def test_small_window_draw_smoke_keeps_panels_renderable(runtime_builder):
    cfg.TELEMETRY.ENABLE_VIEWER_INSPECTOR_ENRICHMENT = True
    runtime = runtime_builder(seed=704, width=12, height=12, agents=6, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    viewer.handle_window_resize(760, 560)
    viewer.selected_slot_id = int(runtime.registry.get_alive_indices()[0].item())
    state_data = viewer._prepare_state_data()

    surface = pygame.Surface((viewer.Wpix, viewer.Hpix))
    viewer.world_renderer.draw(surface, state_data)
    viewer.hud_panel.draw(surface, state_data)
    viewer.side_panel.draw(surface, state_data)

    viewer.selected_slot_id = None
    viewer.selected_hzone_id = runtime.grid.hzones[0]["id"]
    viewer.side_panel.draw(surface, state_data)


def test_text_cache_wraps_long_inspector_lines():
    from tensor_crypt.viewer.text_cache import TextCache

    pygame.init()
    pygame.font.init()
    try:
        cache = TextCache()
        lines = cache.wrap_lines(
            "Parents B/T/A: 100200300 / 400500600 / 700800900",
            12,
            140,
        )
        assert len(lines) >= 2
        assert all(lines)
    finally:
        if pygame.get_init():
            pygame.quit()