from __future__ import annotations

from ..config_bridge import cfg


COLORS = {
    "bg": (20, 22, 27),
    "hud_bg": (12, 14, 18),
    "side_bg": (18, 20, 26),
    "grid": (40, 42, 48),
    "border": (70, 74, 82),
    "wall": (100, 104, 112),
    "empty": (24, 26, 32),
    "selection_marker": (241, 196, 15),
    "text": (230, 230, 230),
    "text_dim": (180, 186, 194),
    "text_dark": (100, 100, 100),
    "text_header": (255, 255, 255),
    "text_success": (46, 204, 113),
    "text_warn": (243, 156, 18),
    "text_harm": (155, 89, 182),
    "pause_text": (241, 196, 15),
    "bar_bg": (38, 42, 48),
    "bar_fg_hp": (46, 204, 113),
    "bloodline_shadow": (26, 28, 34),
    "hzone_heal": (46, 204, 113, 60),
    "hzone_harm": (155, 89, 182, 60),
    "selection_marker_hzone": (0, 255, 255),
    "ray_wall": (180, 180, 180),
    "ray_agent": (231, 76, 60),
    "ray_empty": (100, 100, 110),
}


def _blend_rgb(low: tuple[int, int, int], high: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(int(a * (1.0 - alpha) + b * alpha) for a, b in zip(low, high))


def get_bloodline_base_color(family_id: str) -> tuple[int, int, int]:
    raw = cfg.BRAIN.FAMILY_COLORS[family_id]
    return tuple(int(channel) for channel in raw)


def get_bloodline_agent_color(family_id: str, hp_ratio: float) -> tuple[int, int, int]:
    base = get_bloodline_base_color(family_id)
    if not cfg.VIEW.BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED:
        return base
    shaded = _blend_rgb(COLORS["bloodline_shadow"], base, 1.0 - cfg.VIEW.BLOODLINE_LOW_HP_SHADE)
    return _blend_rgb(shaded, base, hp_ratio)
