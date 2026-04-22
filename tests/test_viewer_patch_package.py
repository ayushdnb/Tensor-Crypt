import math

import pygame

from tensor_crypt.config_bridge import cfg
from tensor_crypt.viewer.camera import Camera


def _offset_bounds(cam: Camera):
    visible_w = cam.screen_width / max(cam.cell_px, 0.01)
    visible_h = cam.screen_height / max(cam.cell_px, 0.01)
    return (
        -visible_w * 0.5,
        cam.grid_w - visible_w * 0.5,
        -visible_h * 0.5,
        cam.grid_h - visible_h * 0.5,
    )


def test_camera_fit_clamp_and_zoom_bounds():
    cam = Camera(screen_width=400, screen_height=300, grid_w=64, grid_h=48)

    assert cam.min_zoom <= cam.cell_px <= cam.max_zoom
    min_ox, max_ox, min_oy, max_oy = _offset_bounds(cam)
    assert min_ox <= cam.offset_x <= max_ox
    assert min_oy <= cam.offset_y <= max_oy

    cam.pan(-10_000, -10_000)
    min_ox, max_ox, min_oy, max_oy = _offset_bounds(cam)
    assert math.isclose(cam.offset_x, min_ox, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(cam.offset_y, min_oy, rel_tol=0.0, abs_tol=1e-6)

    cam.pan(20_000, 20_000)
    min_ox, max_ox, min_oy, max_oy = _offset_bounds(cam)
    assert math.isclose(cam.offset_x, max_ox, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(cam.offset_y, max_oy, rel_tol=0.0, abs_tol=1e-6)

    cam.zoom_at(1e6, 200, 150)
    assert cam.cell_px == cam.max_zoom
    cam.zoom_at(1e-6, 200, 150)
    assert cam.cell_px == cam.min_zoom

    wx, wy = cam.screen_to_world_float(123, 45)
    assert isinstance(wx, float)
    assert isinstance(wy, float)


def test_viewer_state_data_has_family_alive_counts_and_params(runtime_builder):
    runtime = runtime_builder(seed=601, width=14, height=14, agents=8, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    state_data = viewer._prepare_state_data()
    assert sum(state_data["family_alive_counts"].values()) == state_data["num_alive"]

    slot_id = runtime.registry.get_alive_indices().tolist()[0]
    joined = "\n".join(text for text, _ in viewer.side_panel._agent_detail_lines(slot_id))
    assert "Params:" in joined
    assert "Gen:" in joined


def test_input_fit_mousewheel_and_resize_preserves_zoom(runtime_builder, monkeypatch):
    runtime = runtime_builder(seed=602, width=16, height=16, agents=6, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    viewer.cam.pan(999, 999)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_f}))
    running, advance = viewer.input_handler.handle()
    assert running is True
    assert advance is False

    min_ox, max_ox, min_oy, max_oy = _offset_bounds(viewer.cam)
    assert min_ox <= viewer.cam.offset_x <= max_ox
    assert min_oy <= viewer.cam.offset_y <= max_oy

    wrect = viewer.layout.world_rect()
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (wrect.x + 10, wrect.y + 10))

    before_zoom = viewer.cam.cell_px
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": 1}))
    viewer.input_handler.handle()
    assert viewer.cam.cell_px >= before_zoom

    before_zoom_out = viewer.cam.cell_px
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": -1}))
    viewer.input_handler.handle()
    assert viewer.cam.cell_px <= before_zoom_out

    before_resize_zoom = viewer.cam.cell_px
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, {"w": 640, "h": 480}))
    viewer.input_handler.handle()
    world_rect = viewer.layout.world_rect()
    assert viewer.Wpix == 640
    assert viewer.Hpix == 480
    assert viewer.cam.screen_width == world_rect.width
    assert viewer.cam.screen_height == world_rect.height
    assert viewer.cam.cell_px == before_resize_zoom


def test_selection_prefers_agent_over_zone_with_proximity_pick(runtime_builder, monkeypatch):
    runtime = runtime_builder(seed=603, width=18, height=18, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    state_data = viewer._prepare_state_data()
    viewer._last_state_data = state_data

    slot_id = next(iter(state_data["agent_map"]))
    agent = state_data["agent_map"][slot_id]
    ax, ay = int(agent["x"]), int(agent["y"])

    fake_x = max(1, min(runtime.grid.W - 2, ax - 1))
    fake_y = max(1, min(runtime.grid.H - 2, ay))
    if runtime.grid.get_agent_at(fake_x, fake_y) >= 0:
        fake_x = max(1, min(runtime.grid.W - 2, ax + 1))

    runtime.grid.add_hzone(fake_x, fake_y, fake_x, fake_y, 0.5)

    wrect = viewer.layout.world_rect()
    sx, sy = viewer.cam.world_to_screen(agent["x"], agent["y"])
    click_pos = (int(wrect.x + sx + viewer.cam.cell_px * 0.5), int(wrect.y + sy + viewer.cam.cell_px * 0.5))

    monkeypatch.setattr(viewer.cam, "screen_to_world", lambda cx, cy: (fake_x, fake_y))

    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": click_pos, "button": 1}))
    viewer.input_handler.handle()

    assert viewer.selected_slot_id == slot_id
    assert viewer.selected_hzone_id is None


def test_zone_selection_and_empty_click_clear(runtime_builder):
    runtime = runtime_builder(seed=604, width=18, height=18, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    state_data = viewer._prepare_state_data()
    viewer._last_state_data = state_data
    occupied = {(int(a["x"]), int(a["y"])) for a in state_data["agent_map"].values()}

    def far_from_agents(x, y):
        return all(abs(x - ax) + abs(y - ay) > 2 for ax, ay in occupied)

    zone_cell = None
    clear_cell = None
    for y in range(1, runtime.grid.H - 1):
        for x in range(1, runtime.grid.W - 1):
            if (x, y) in occupied:
                continue
            if zone_cell is None and far_from_agents(x, y):
                zone_cell = (x, y)
            elif zone_cell is not None and clear_cell is None and far_from_agents(x, y) and (x, y) != zone_cell:
                clear_cell = (x, y)
            if zone_cell is not None and clear_cell is not None:
                break
        if zone_cell is not None and clear_cell is not None:
            break

    assert zone_cell is not None
    assert clear_cell is not None

    runtime.grid.add_hzone(zone_cell[0], zone_cell[1], zone_cell[0], zone_cell[1], 0.5)
    zone_id = runtime.grid.next_hzone_id - 1

    wrect = viewer.layout.world_rect()
    zx, zy = viewer.cam.world_to_screen(zone_cell[0], zone_cell[1])
    zone_click = (int(wrect.x + zx + viewer.cam.cell_px * 0.5), int(wrect.y + zy + viewer.cam.cell_px * 0.5))

    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": zone_click, "button": 1}))
    viewer.input_handler.handle()
    assert viewer.selected_hzone_id == zone_id
    assert viewer.selected_slot_id is None

    cx, cy = viewer.cam.world_to_screen(clear_cell[0], clear_cell[1])
    clear_click = (int(wrect.x + cx + viewer.cam.cell_px * 0.5), int(wrect.y + cy + viewer.cam.cell_px * 0.5))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": clear_click, "button": 1}))
    viewer.input_handler.handle()

    assert viewer.selected_hzone_id is None
    assert viewer.selected_slot_id is None


def test_side_panel_controls_and_legend_lines(runtime_builder, monkeypatch):
    runtime = runtime_builder(seed=605, width=14, height=14, agents=6, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    state_data = viewer._prepare_state_data()

    captured = []

    def fake_render(text, size, color, aa=True):
        captured.append(str(text))
        return pygame.Surface((max(1, len(str(text))), max(1, int(size))))

    monkeypatch.setattr(viewer.side_panel.text, "render", fake_render)
    surface = pygame.Surface((viewer.Wpix, viewer.Hpix))
    viewer.side_panel.draw(surface, state_data)

    assert "Fit world: F" in captured
    assert "Fullscreen: Alt+Enter" in captured
    assert "Quit: ESC" in captured

    for family_id in cfg.BRAIN.FAMILY_ORDER:
        expected = f"{family_id}  {state_data['family_alive_counts'].get(family_id, 0)}"
        assert expected in captured


def test_hotkeys_toggle_overlays_speed_and_catastrophe_controls(runtime_builder):
    runtime = runtime_builder(seed=606, width=16, height=16, agents=6, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    base_rays = viewer.show_rays
    base_hp_bars = viewer.show_hp_bars
    base_hzones = viewer.show_hzones
    base_grid = viewer.show_grid

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r}))
    viewer.input_handler.handle()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_b}))
    viewer.input_handler.handle()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_h}))
    viewer.input_handler.handle()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_g}))
    viewer.input_handler.handle()

    assert viewer.show_rays is (not base_rays)
    assert viewer.show_hp_bars is (not base_hp_bars)
    assert viewer.show_hzones is (not base_hzones)
    assert viewer.show_grid is (not base_grid)

    speed_before = viewer.speed_multiplier
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_EQUALS}))
    viewer.input_handler.handle()
    assert viewer.speed_multiplier > speed_before

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_MINUS}))
    viewer.input_handler.handle()
    assert viewer.speed_multiplier >= 0.125

    zone_id = runtime.grid.hzones[0]["id"]
    viewer.selected_hzone_id = zone_id
    rate_before = runtime.grid.get_hzone(zone_id)["rate"]
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_EQUALS}))
    viewer.input_handler.handle()
    rate_after = runtime.grid.get_hzone(zone_id)["rate"]
    assert rate_after > rate_before

    viewer.selected_hzone_id = None
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE}))
    _, advance = viewer.input_handler.handle()
    assert viewer.paused is True
    assert advance is False

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_PERIOD}))
    _, advance = viewer.input_handler.handle()
    assert advance is True

    panel_before = viewer.show_catastrophe_panel
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_i}))
    viewer.input_handler.handle()
    assert viewer.show_catastrophe_panel is (not panel_before)

    runtime.engine.catastrophes.set_mode("auto_dynamic", current_tick=runtime.engine.tick, arm_scheduler=True)
    scheduler_before = runtime.engine.catastrophes.scheduler_paused
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_o}))
    viewer.input_handler.handle()
    assert runtime.engine.catastrophes.scheduler_paused is (not scheduler_before)

    mode_before = runtime.engine.catastrophes.mode
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_y}))
    viewer.input_handler.handle()
    assert runtime.engine.catastrophes.mode != mode_before

    auto_before = runtime.engine.catastrophes.auto_enabled
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_u}))
    viewer.input_handler.handle()
    assert runtime.engine.catastrophes.auto_enabled != auto_before



def test_alt_enter_toggles_fullscreen(runtime_builder, monkeypatch):
    runtime = runtime_builder(seed=6061, width=16, height=16, agents=6, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    calls = []

    def fake_set_mode(size, flags=0):
        calls.append((tuple(size), flags))
        return pygame.Surface(size)

    monkeypatch.setattr(pygame.display, "set_mode", fake_set_mode)
    monkeypatch.setattr(pygame.display, "get_desktop_sizes", lambda: [(1600, 900)])

    start_size = (viewer.Wpix, viewer.Hpix)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": pygame.KMOD_ALT}))
    viewer.input_handler.handle()

    assert viewer.is_fullscreen is True
    assert (viewer.Wpix, viewer.Hpix) == (1600, 900)
    assert calls[-1] == ((1600, 900), pygame.FULLSCREEN)

    world_rect = viewer.layout.world_rect()
    assert viewer.cam.screen_width == world_rect.width
    assert viewer.cam.screen_height == world_rect.height

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": pygame.KMOD_ALT}))
    viewer.input_handler.handle()

    assert viewer.is_fullscreen is False
    assert (viewer.Wpix, viewer.Hpix) == start_size
    assert calls[-1] == (start_size, pygame.RESIZABLE)


def test_viewer_hotkeys_toggle_reproduction_overlay_doctrines(runtime_builder):
    runtime = runtime_builder(seed=607, width=16, height=16, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    controller = runtime.engine.respawn_controller

    assert controller.get_doctrine_effective_enabled("crowding") is True
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1, "mod": pygame.KMOD_SHIFT}))
    viewer.input_handler.handle()
    assert controller.get_doctrine_effective_enabled("crowding") is False

    assert controller.get_doctrine_effective_enabled("cooldown") is True
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_2, "mod": pygame.KMOD_SHIFT}))
    viewer.input_handler.handle()
    assert controller.get_doctrine_effective_enabled("cooldown") is False

    assert controller.get_doctrine_effective_enabled("local_parent") is True
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_3, "mod": pygame.KMOD_SHIFT}))
    viewer.input_handler.handle()
    assert controller.get_doctrine_effective_enabled("local_parent") is False

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_0, "mod": pygame.KMOD_SHIFT}))
    viewer.input_handler.handle()
    assert controller.doctrine_overrides == {"crowding": None, "cooldown": None, "local_parent": None}


def test_side_panel_controls_include_reproduction_overlay_hotkeys(runtime_builder, monkeypatch):
    runtime = runtime_builder(seed=608, width=14, height=14, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    state_data = viewer._prepare_state_data()

    captured = []

    def fake_render(text, size, color, aa=True):
        captured.append(str(text))
        return pygame.Surface((max(1, len(str(text))), max(1, int(size))))

    monkeypatch.setattr(viewer.side_panel.text, "render", fake_render)
    surface = pygame.Surface((viewer.Wpix, viewer.Hpix))
    viewer.side_panel.draw(surface, state_data)

    assert "Ashen Press: Shift+1" in captured
    assert "Widow Interval: Shift+2" in captured
    assert "Bloodhold Radius: Shift+3" in captured
    assert "Clear Doctrine Overrides: Shift+0" in captured


def test_prepare_state_data_exposes_overlay_status(runtime_builder):
    runtime = runtime_builder(seed=609, width=14, height=14, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    state_data = runtime.viewer._prepare_state_data()

    assert "respawn_overlay_state" in state_data
    assert "doctrines" in state_data["respawn_overlay_state"]
    assert {"crowding", "cooldown", "local_parent"}.issubset(state_data["respawn_overlay_state"]["doctrines"].keys())


def test_text_cache_lazily_builds_unconfigured_font_sizes(runtime_builder):
    runtime = runtime_builder(seed=610, width=14, height=14, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cache = runtime.viewer.text_cache

    rendered = cache.render("runtime override differs from config default", 11, (255, 255, 255))

    assert rendered.get_width() > 0
    assert 11 in cache.fonts
