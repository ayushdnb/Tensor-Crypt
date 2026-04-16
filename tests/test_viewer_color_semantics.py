from types import SimpleNamespace

import pygame

from tensor_crypt.config_bridge import cfg
import tensor_crypt.viewer.panels as panels
from tensor_crypt.viewer.colors import (
    COLORS,
    _blend_rgb,
    get_bloodline_agent_color,
    get_bloodline_base_color,
)


_EXPECTED_PALETTE = {
    "House Nocthar": (84, 138, 214),
    "House Vespera": (84, 160, 112),
    "House Umbrael": (220, 184, 76),
    "House Mourndveil": (208, 102, 102),
    "House Somnyr": (168, 112, 208),
}


class _FakeTextCache:
    def render(self, text, size, color, aa=True):
        return pygame.Surface((max(1, len(str(text))), max(1, int(size))))

    def line_height(self, size):
        return max(1, int(size))


class _FakeCam:
    cell_px = 10

    def world_to_screen(self, x, y):
        return (int(x) * self.cell_px, int(y) * self.cell_px)


def _srgb_channel_to_linear(channel: int) -> float:
    value = float(channel) / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    def rel_luminance(rgb: tuple[int, int, int]) -> float:
        r, g, b = (_srgb_channel_to_linear(channel) for channel in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lum_a = rel_luminance(a)
    lum_b = rel_luminance(b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def test_bloodline_palette_matches_expected_mapping():
    assert tuple(cfg.BRAIN.FAMILY_ORDER) == tuple(_EXPECTED_PALETTE.keys())
    for family_id, expected in _EXPECTED_PALETTE.items():
        assert tuple(cfg.BRAIN.FAMILY_COLORS[family_id]) == expected
        assert get_bloodline_base_color(family_id) == expected


def test_bloodline_palette_keeps_minimum_non_text_contrast_against_world_background():
    background = COLORS["empty"]
    for family_id in cfg.BRAIN.FAMILY_ORDER:
        ratio = _contrast_ratio(get_bloodline_base_color(family_id), background)
        assert ratio >= 3.0, (family_id, ratio)


def test_low_hp_color_modulation_off_keeps_base_color_across_hp_ratios():
    cfg.VIEW.BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED = False

    for family_id in cfg.BRAIN.FAMILY_ORDER:
        base = get_bloodline_base_color(family_id)
        for hp_ratio in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert get_bloodline_agent_color(family_id, hp_ratio) == base


def test_low_hp_color_modulation_on_preserves_existing_curve():
    cfg.VIEW.BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED = True
    cfg.VIEW.BLOODLINE_LOW_HP_SHADE = 0.35

    family_id = cfg.BRAIN.FAMILY_ORDER[0]
    base = get_bloodline_base_color(family_id)
    shaded = _blend_rgb(COLORS["bloodline_shadow"], base, 1.0 - cfg.VIEW.BLOODLINE_LOW_HP_SHADE)

    assert get_bloodline_agent_color(family_id, 0.0) == shaded
    assert get_bloodline_agent_color(family_id, 1.0) == base
    assert get_bloodline_agent_color(family_id, 0.5) == _blend_rgb(shaded, base, 0.5)


def test_experimental_branch_preset_overrides_only_target_family_base_color():
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = "House Nocthar"

    assert get_bloodline_base_color("House Nocthar") == tuple(cfg.BRAIN.EXPERIMENTAL_BRANCH_COLOR)
    assert get_bloodline_base_color("House Vespera") == _EXPECTED_PALETTE["House Vespera"]


def test_world_renderer_routes_agent_fill_through_canonical_color_helper(monkeypatch):
    fake_viewer = SimpleNamespace(
        cam=_FakeCam(),
        engine=SimpleNamespace(registry=None, grid=None),
        show_hzones=False,
        show_hp_bars=False,
        show_rays=False,
        selected_slot_id=None,
        show_grid=False,
        show_catastrophe_overlay=False,
    )
    renderer = panels.WorldRenderer(fake_viewer)
    state_data = {
        "agent_map": {
            0: {"x": 1, "y": 2, "hp": 3.0, "hp_max": 6.0, "family_id": "House Nocthar"},
            1: {"x": 3, "y": 4, "hp": 5.0, "hp_max": 10.0, "family_id": "House Somnyr"},
        }
    }

    helper_calls = []
    rect_calls = []
    sentinel = (9, 19, 29)

    def fake_get_bloodline_agent_color(family_id, hp_ratio):
        helper_calls.append((family_id, float(hp_ratio)))
        return sentinel

    def fake_draw_rect(surface, color, rect, width=0):
        rect_calls.append((tuple(color), tuple(rect), int(width)))
        return pygame.Rect(rect)

    monkeypatch.setattr(panels, "get_bloodline_agent_color", fake_get_bloodline_agent_color)
    monkeypatch.setattr(pygame.draw, "rect", fake_draw_rect)

    surface = pygame.Surface((120, 120))
    renderer._draw_agents(surface, pygame.Rect(0, 0, 120, 120), fake_viewer.cam.cell_px, state_data)

    assert helper_calls == [
        ("House Nocthar", 3.0 / (6.0 + 1e-6)),
        ("House Somnyr", 5.0 / (10.0 + 1e-6)),
    ]
    assert rect_calls == [
        ((9, 19, 29), (10, 20, 10, 10), 0),
        ((9, 19, 29), (30, 40, 10, 10), 0),
    ]


def test_bloodline_legend_uses_base_family_color_helper(monkeypatch):
    side_panel = panels.SidePanel(SimpleNamespace(text_cache=_FakeTextCache(), engine=SimpleNamespace(registry=None)))
    state_data = {"family_alive_counts": {family_id: 1 for family_id in cfg.BRAIN.FAMILY_ORDER}}
    seen = []

    def fake_get_bloodline_base_color(family_id):
        seen.append(family_id)
        return (40, 50, 60)

    monkeypatch.setattr(panels, "get_bloodline_base_color", fake_get_bloodline_base_color)

    surface = pygame.Surface((320, 180))
    side_panel._draw_bloodline_legend(
        surface,
        0,
        0,
        300,
        state_data,
        {"section_header_size": 12, "body_size": 10, "section_gap": 8},
    )

    assert seen == list(cfg.BRAIN.FAMILY_ORDER)
