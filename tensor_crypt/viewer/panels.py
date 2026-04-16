
"""Viewer panels and world rendering helpers."""

import math

import pygame

from ..agents.state_registry import Registry
from ..config_bridge import cfg
from ..population.reproduction import trait_values_from_latent
from .colors import COLORS, get_bloodline_agent_color, get_bloodline_base_color


class WorldRenderer:
    def __init__(self, viewer):
        self.viewer = viewer
        self.cam = viewer.cam
        self.engine = viewer.engine
        self.registry = viewer.engine.registry
        self.grid = viewer.engine.grid
        self.static_surf = None

    def _build_static_cache(self, wrect):
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["empty"])

        occ_np = self.grid.grid[0].cpu().numpy()
        h_rate_np = self.grid.grid[1].cpu().numpy()
        h, w = occ_np.shape
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)

        for y in range(h):
            for x in range(w):
                rect = self.cam.cell_rect_px(x, y)
                if occ_np[y, x] == 1.0:
                    pygame.draw.rect(self.static_surf, COLORS["wall"], rect)
                if self.viewer.show_hzones:
                    h_rate = h_rate_np[y, x]
                    if h_rate > 0:
                        pygame.draw.rect(overlay, COLORS["hzone_heal"], rect)
                    elif h_rate < 0:
                        pygame.draw.rect(overlay, COLORS["hzone_harm"], rect)

        self.static_surf.blit(overlay, (0, 0))

    def draw(self, surf, state_data):
        wrect = self.viewer.layout.world_rect()
        c = self.cam.cell_px

        if self.static_surf is None or self.static_surf.get_size() != wrect.size:
            self._build_static_cache(wrect)

        previous_clip = surf.get_clip()
        surf.set_clip(wrect)
        surf.blit(self.static_surf, wrect.topleft)

        self._draw_agents(surf, wrect, c, state_data)
        if self.viewer.show_hp_bars:
            self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.show_rays and self.viewer.selected_slot_id is not None:
            self._draw_rays(surf, wrect, c, state_data)
        self._draw_selection_markers(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6:
            self._draw_grid_lines(surf, wrect, c)
        if self.viewer.show_catastrophe_overlay:
            self._draw_catastrophe_overlay(surf, wrect, state_data)
        surf.set_clip(previous_clip)
        pygame.draw.rect(surf, COLORS["border"], wrect, 2)

    def _draw_agents(self, surf, wrect, c, state_data):
        for slot_id, agent in state_data["agent_map"].items():
            hp_ratio = agent["hp"] / (agent["hp_max"] + 1e-6)
            color = get_bloodline_agent_color(agent["family_id"], hp_ratio)
            agent_rect = self.cam.cell_rect_px(agent["x"], agent["y"]).move(wrect.x, wrect.y)
            pygame.draw.rect(surf, color, agent_rect)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        if c < 8:
            return
        for slot_id, agent in state_data["agent_map"].items():
            hp_ratio = agent["hp"] / (agent["hp_max"] + 1e-6)
            cell_rect = self.cam.cell_rect_px(agent["x"], agent["y"])
            bar_w = cell_rect.width
            bar_h = max(1, cell_rect.height // 8)
            bar_y = wrect.y + cell_rect.top - bar_h - 2
            if wrect.y < bar_y < wrect.bottom:
                fg_width = max(0, min(bar_w, int(round(bar_w * hp_ratio))))
                bg_rect = pygame.Rect(wrect.x + cell_rect.left, bar_y, bar_w, bar_h)
                pygame.draw.rect(surf, COLORS["bar_bg"], bg_rect)
                if fg_width > 0:
                    fg_rect = pygame.Rect(wrect.x + cell_rect.left, bar_y, fg_width, bar_h)
                    pygame.draw.rect(surf, COLORS["bar_fg_hp"], fg_rect)

    def _draw_selection_markers(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id is not None and slot_id in state_data["agent_map"]:
            agent = state_data["agent_map"][slot_id]
            marker_rect = self.cam.cell_rect_px(agent["x"], agent["y"]).move(wrect.x, wrect.y)
            pygame.draw.rect(surf, COLORS["selection_marker"], marker_rect, max(1, int(c // 10)))

        hzone_id = self.viewer.selected_hzone_id
        if hzone_id is not None:
            zone = self.engine.grid.get_hzone(hzone_id)
            if zone:
                marker_rect = self.cam.world_rect_px(
                    zone["x1"],
                    zone["y1"],
                    zone["x2"] + 1,
                    zone["y2"] + 1,
                ).move(wrect.x, wrect.y)
                pygame.draw.rect(surf, COLORS["selection_marker_hzone"], marker_rect, max(1, int(c // 10)))

    def _draw_grid_lines(self, surf, wrect, c):
        for gx in range(self.grid.W + 1):
            sx = wrect.x + self.cam.edge_x_to_screen(gx)
            if wrect.left <= sx <= wrect.right:
                pygame.draw.line(surf, COLORS["grid"], (sx, wrect.y), (sx, wrect.bottom))
        for gy in range(self.grid.H + 1):
            sy = wrect.y + self.cam.edge_y_to_screen(gy)
            if wrect.top <= sy <= wrect.bottom:
                pygame.draw.line(surf, COLORS["grid"], (wrect.x, sy), (wrect.right, sy))

    def _draw_rays(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id not in state_data["agent_map"]:
            return

        agent = state_data["agent_map"][slot_id]
        agent_x = agent["x"]
        agent_y = agent["y"]
        start_cell_rect = self.cam.cell_rect_px(agent_x, agent_y)
        start_pos_screen = (
            wrect.x + start_cell_rect.centerx,
            wrect.y + start_cell_rect.centery,
        )
        vision_range = int(self.engine.perception.get_effective_vision_for_slot(slot_id))
        occ_grid = self.grid.grid[0]
        agent_grid = self.grid.grid[2]
        h, w = self.grid.H, self.grid.W
        num_rays = self.engine.perception.num_rays

        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            dx, dy = math.cos(angle), math.sin(angle)
            end_x, end_y, color = agent_x, agent_y, COLORS["ray_empty"]

            for step in range(1, vision_range + 1):
                tx, ty = int(round(agent_x + dx * step)), int(round(agent_y + dy * step))
                if not (0 <= tx < w and 0 <= ty < h):
                    end_x, end_y = tx, ty
                    color = COLORS["ray_wall"]
                    break
                if occ_grid[ty, tx] > 0.5:
                    end_x, end_y = tx, ty
                    color = COLORS["ray_wall"]
                    break
                if agent_grid[ty, tx] >= 0 and agent_grid[ty, tx] != slot_id:
                    end_x, end_y = tx, ty
                    color = COLORS["ray_agent"]
                    break
            else:
                end_x, end_y = agent_x + dx * vision_range, agent_y + dy * vision_range

            end_pos_world = self.cam.world_to_screen(end_x, end_y)
            end_pos_screen = (wrect.x + end_pos_world[0] + c // 2, wrect.y + end_pos_world[1] + c // 2)
            pygame.draw.line(surf, color, start_pos_screen, end_pos_screen, 1)

    def _draw_catastrophe_overlay(self, surf, wrect, state_data):
        catastrophe_state = state_data.get("catastrophe_state", {})
        if not catastrophe_state.get("active_count", 0):
            return

        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        overlay.fill((110, 20, 20, 22))
        surf.blit(overlay, wrect.topleft)

        border_rect = catastrophe_state.get("thorn_march_safe_rect")
        if border_rect:
            x1, y1, x2, y2 = border_rect
            rect = self.cam.world_rect_px(x1, y1, x2 + 1, y2 + 1)
            pygame.draw.rect(
                surf,
                (215, 80, 80),
                rect.move(wrect.x, wrect.y),
                2,
            )

        woundtide_x = catastrophe_state.get("woundtide_front_x")
        if woundtide_x is not None:
            sx = self.cam.edge_x_to_screen(woundtide_x)
            pygame.draw.line(surf, (220, 70, 120), (wrect.x + sx, wrect.y), (wrect.x + sx, wrect.bottom), 2)


class HudPanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.text = viewer.text_cache

    def _draw_family_counts(self, surf, x, y, max_right, size, state_data):
        line_height = max(14, self.text.line_height(size))
        cursor_x = x
        family_counts = state_data.get("family_alive_counts", {})
        for family_id in cfg.BRAIN.FAMILY_ORDER:
            count = family_counts.get(family_id, 0)
            color = get_bloodline_base_color(family_id)
            count_text = str(count)
            count_width, _ = self.text.measure(count_text, size)
            block_width = 10 + 4 + count_width + 16
            if cursor_x + block_width > max_right and cursor_x != x:
                cursor_x = x
                y += line_height
            pygame.draw.rect(surf, color, (cursor_x, y + 2, 10, 10))
            cursor_x += 14
            surf.blit(self.text.render(count_text, size, COLORS["text_dim"]), (cursor_x, y))
            cursor_x += count_width + 16
        return y + line_height

    def _compose_catastrophe_line(self, catastrophe_state: dict) -> str:
        mode = catastrophe_state.get("mode", "off")
        active_names = catastrophe_state.get("active_names", [])
        next_tick = catastrophe_state.get("next_auto_tick", None)
        global_enabled = bool(catastrophe_state.get("global_enabled", True))
        scheduler_armed = bool(catastrophe_state.get("scheduler_armed", False))
        scheduler_paused = bool(catastrophe_state.get("scheduler_paused", False))
        active_count = int(catastrophe_state.get("active_count", 0))

        if not global_enabled:
            scheduler_label = "disabled"
        elif mode == "off":
            scheduler_label = "off"
        elif mode == "manual_only":
            scheduler_label = "manual-only"
        elif scheduler_armed:
            scheduler_label = "armed/paused" if scheduler_paused else "armed/running"
        else:
            scheduler_label = "disarmed"

        line = f"Cata: {mode} | {scheduler_label} | active={active_count}"
        if next_tick is not None and next_tick >= 0:
            line += f" | next={next_tick}"
        if active_names:
            line += " | " + ", ".join(active_names[:2])
        return line

    def _compose_overlay_line(self, overlay_state: dict) -> str:
        doctrines = overlay_state.get("doctrines", {})

        def _fmt(key: str) -> str:
            item = doctrines.get(key, {})
            enabled = bool(item.get("effective_enabled", False))
            marker = "*" if (
                cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_OVERRIDE_MARKERS
                and item.get("override_differs", False)
            ) else ""
            label = item.get("short_name", key)
            return f"{label}:{'ON' if enabled else 'OFF'}{marker}"

        repro_gate = "EN" if overlay_state.get("reproduction_enabled", True) else "DIS"
        doctrine_line = (
            f"Repro:{repro_gate} | "
            f"{_fmt('crowding')} | "
            f"{_fmt('cooldown')} | "
            f"{_fmt('local_parent')}"
        )
        if overlay_state.get("below_floor_active", False):
            doctrine_line += " | floor-softened"
        return doctrine_line

    def draw(self, surf, state_data):
        hrect = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hrect)
        pygame.draw.rect(surf, COLORS["border"], hrect, 2)

        content = self.viewer.layout.content_rect(hrect)
        title_size = 16 if self.viewer.layout.is_dense() else 18
        badge_size = 14 if self.viewer.layout.is_dense() else 16
        text_size = 11 if self.viewer.layout.is_dense() else 12
        body_size = 12 if self.viewer.layout.is_dense() else 14

        if self.viewer.selected_hzone_id is not None:
            pause_str = "[ H-ZONE EDIT ]"
            pause_color = COLORS["selection_marker_hzone"]
        else:
            pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"
            pause_color = COLORS["pause_text"] if self.viewer.paused else COLORS["text"]

        previous_clip = surf.get_clip()
        surf.set_clip(content)

        x = content.x
        y = content.y

        tick_surface = self.text.render(f"Tick {self.viewer.engine.tick}", title_size, COLORS["text"])
        pause_surface = self.text.render(pause_str, badge_size, pause_color)
        surf.blit(tick_surface, (x, y))
        pause_x = content.right - pause_surface.get_width()
        min_pause_x = x + tick_surface.get_width() + 16
        pause_x = max(min_pause_x, pause_x)
        surf.blit(pause_surface, (pause_x, y + 1))

        y += max(tick_surface.get_height(), pause_surface.get_height()) + 6
        alive_str = f"Alive: {state_data['num_alive']} / {self.viewer.engine.registry.max_agents}"
        surf.blit(self.text.render(alive_str, body_size, COLORS["text_dim"]), (x, y))
        y += self.text.line_height(body_size)
        y = self._draw_family_counts(surf, x, y, content.right, text_size, state_data)

        wrap_width = max(1, content.width)
        catastrophe_state = state_data.get("catastrophe_state", {})
        if cfg.VIEW.SHOW_CATASTROPHE_STATUS_IN_HUD:
            for line in self.text.wrap_lines(self._compose_catastrophe_line(catastrophe_state), text_size, wrap_width):
                surf.blit(self.text.render(line, text_size, COLORS["text_warn"]), (x, y))
                y += self.text.line_height(text_size)

        overlay_state = state_data.get("respawn_overlay_state", {})
        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_STATUS_IN_HUD:
            for line in self.text.wrap_lines(self._compose_overlay_line(overlay_state), text_size, wrap_width):
                surf.blit(self.text.render(line, text_size, COLORS["text_dim"]), (x, y))
                y += self.text.line_height(text_size)

        surf.set_clip(previous_clip)


class SidePanel:
    CONTROLS = [
        "Pan: WASD / Arrows",
        "Zoom: Mouse Wheel",
        "Fit world: F",
        "Fullscreen: Alt+Enter",
        "Pause: SPACE",
        "Speed: +/- (no zone sel.)",
        "Step (Paused): .",
        "Quit: ESC",
        "---",
        "Edit H-Zone Rate: +/-",
        "Rays: R  HP Bars: B",
        "H-Zones: H  Grid: G",
        "---",
        "Ashen Press: Shift+1",
        "Widow Interval: Shift+2",
        "Bloodhold Radius: Shift+3",
        "Clear Doctrine Overrides: Shift+0",
        "---",
        "Cata F1..F12: manual trigger",
        "Clear Active: C  Mode: Y",
        "Sched Arm: U  Panel: I  Sched Pause: O",
    ]

    def __init__(self, viewer):
        self.viewer = viewer
        self.engine = viewer.engine
        self.registry = viewer.engine.registry
        self.text = viewer.text_cache
        self.line_height = 19
        self.scroll_offset = 0

    def _metrics(self) -> dict:
        dense = self.viewer.layout.is_dense()
        return {
            "header_size": 17 if dense else 18,
            "section_header_size": 13 if dense else 14,
            "body_size": 11 if dense else 12,
            "hint_size": 11 if dense else 13,
            "control_header_size": 15 if dense else 16,
            "control_size": 11 if dense else 12,
            "line_gap": 2 if dense else 3,
            "section_gap": 8 if dense else 10,
            "controls_step": 14 if dense else 16,
            "controls_separator_gap": 4,
            "padding": self.viewer.layout.panel_padding(),
        }

    def _blit_wrapped(self, surf, text, size, color, x, y, max_width, *, line_gap: int) -> int:
        wrapped = self.text.wrap_lines(text, size, max_width)
        line_height = self.text.line_height(size)
        for line in wrapped:
            if surf is not None:
                surf.blit(self.text.render(line, size, color), (x, y))
            y += line_height + line_gap
        return y

    def _draw_header_line(self, surf, text, size, color, x, y) -> int:
        if surf is not None:
            surf.blit(self.text.render(text, size, color), (x, y))
        return y + self.text.line_height(size)

    def _controls_height(self, metrics: dict) -> int:
        height = self.text.line_height(metrics["control_header_size"]) + 8
        for line in self.CONTROLS:
            if line == "---":
                height += metrics["controls_separator_gap"]
            else:
                height += metrics["controls_step"]
        return height + metrics["padding"]

    def _content_rect(self, srect: pygame.Rect, metrics: dict) -> pygame.Rect:
        header_top = srect.y + metrics["padding"] + self.text.line_height(metrics["header_size"]) + 10
        footer_height = self._controls_height(metrics)
        footer_top = max(header_top + 24, srect.bottom - footer_height)
        return pygame.Rect(
            srect.x + metrics["padding"],
            header_top,
            max(1, srect.width - (metrics["padding"] * 2)),
            max(1, footer_top - header_top - 8),
        )

    def _render_controls(self, surf, srect: pygame.Rect, metrics: dict) -> None:
        x = srect.x + metrics["padding"]
        y = srect.bottom - self._controls_height(metrics) + 4
        pygame.draw.line(surf, COLORS["border"], (srect.x, y - 6), (srect.right, y - 6), 1)
        surf.blit(self.text.render("Controls", metrics["control_header_size"], COLORS["text_header"]), (x, y))
        y += self.text.line_height(metrics["control_header_size"]) + 4

        for line in self.CONTROLS:
            if line == "---":
                y += metrics["controls_separator_gap"]
                continue
            surf.blit(self.text.render(line, metrics["control_size"], COLORS["text_dim"]), (x, y))
            y += metrics["controls_step"]

    def _compose_scroll_content(self, surf, x, y, max_width, state_data, metrics: dict) -> int:
        slot_id = self.viewer.selected_slot_id
        hzone_id = self.viewer.selected_hzone_id

        if slot_id is not None:
            if slot_id not in state_data["agent_map"]:
                y = self._blit_wrapped(
                    surf,
                    f"UID {self.viewer.last_selected_uid} (Dead)",
                    metrics["hint_size"],
                    COLORS["text_warn"],
                    x,
                    y,
                    max_width,
                    line_gap=metrics["line_gap"],
                )
            else:
                y = self._draw_agent_details(surf, x, y, max_width, slot_id, metrics)
        elif hzone_id is not None:
            y = self._draw_hzone_details(surf, x, y, max_width, hzone_id, metrics)
        else:
            y = self._blit_wrapped(
                surf,
                "Click an agent or H-Zone.",
                metrics["hint_size"],
                COLORS["text_dim"],
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )

        y += metrics["section_gap"] - 2
        y = self._draw_bloodline_legend(surf, x, y, max_width, state_data, metrics)
        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_STATUS_IN_PANEL:
            y = self._draw_reproduction_overlay_block(surf, x, y, max_width, state_data.get("respawn_overlay_state", {}), metrics)
        if self.viewer.show_catastrophe_panel:
            y = self._draw_catastrophe_block(surf, x, y, max_width, state_data.get("catastrophe_state", {}), metrics)
        return y

    def _content_height(self, state_data) -> int:
        srect = self.viewer.layout.side_rect()
        metrics = self._metrics()
        content_rect = self._content_rect(srect, metrics)
        y_end = self._compose_scroll_content(None, content_rect.x, content_rect.y, content_rect.width, state_data, metrics)
        return max(0, int(y_end - content_rect.y))

    def clamp_scroll_offset(self, state_data) -> None:
        if state_data is None:
            self.scroll_offset = 0
            return
        srect = self.viewer.layout.side_rect()
        metrics = self._metrics()
        content_rect = self._content_rect(srect, metrics)
        max_scroll = max(0, self._content_height(state_data) - content_rect.height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

    def scroll_by(self, direction_steps: int, state_data) -> None:
        metrics = self._metrics()
        step_px = max(12, self.text.line_height(metrics["body_size"]))
        self.scroll_offset += int(direction_steps) * step_px
        self.clamp_scroll_offset(state_data)

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        metrics = self._metrics()

        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)

        header_x = srect.x + metrics["padding"]
        header_y = srect.y + metrics["padding"]
        surf.blit(self.text.render("Inspector", metrics["header_size"], COLORS["text_header"]), (header_x, header_y))

        content_rect = self._content_rect(srect, metrics)
        self.clamp_scroll_offset(state_data)

        previous_clip = surf.get_clip()
        surf.set_clip(content_rect)
        y = content_rect.y - self.scroll_offset
        y = self._compose_scroll_content(surf, content_rect.x, y, content_rect.width, state_data, metrics)
        surf.set_clip(previous_clip)

        total_height = self._content_height(state_data)
        max_scroll = max(0, total_height - content_rect.height)
        if max_scroll > 0:
            track = pygame.Rect(srect.right - 8, content_rect.y, 4, content_rect.height)
            thumb_height = max(24, int((content_rect.height / max(total_height, 1)) * content_rect.height))
            thumb_travel = max(0, track.height - thumb_height)
            thumb_y = track.y + int((self.scroll_offset / max(max_scroll, 1)) * thumb_travel)
            pygame.draw.rect(surf, COLORS["bar_bg"], track)
            pygame.draw.rect(surf, COLORS["border"], (track.x, thumb_y, track.width, thumb_height))

        self._render_controls(surf, srect, metrics)

    def _draw_bloodline_legend(self, surf, x, y, max_width, state_data, metrics: dict):
        if not cfg.VIEW.SHOW_BLOODLINE_LEGEND:
            return y

        family_counts = state_data.get("family_alive_counts", {})
        total = max(1, sum(family_counts.values()))
        y = self._draw_header_line(surf, "Bloodlines", metrics["section_header_size"], COLORS["text_header"], x, y)
        y += 2

        bar_width = 60 if max_width >= 260 else 0
        bar_x = x + max(0, max_width - bar_width)
        for family_id in cfg.BRAIN.FAMILY_ORDER:
            color = get_bloodline_base_color(family_id)
            count = family_counts.get(family_id, 0)
            if surf is not None:
                pygame.draw.rect(surf, color, (x, y + 2, 10, 10))
                pygame.draw.rect(surf, COLORS["border"], (x, y + 2, 10, 10), 1)
                surf.blit(self.text.render(f"{family_id}  {count}", metrics["body_size"], COLORS["text_dim"]), (x + 16, y))
                if bar_width:
                    frac = count / total
                    pygame.draw.rect(surf, COLORS["bar_bg"], (bar_x, y + 3, bar_width, 8))
                    pygame.draw.rect(surf, color, (bar_x, y + 3, int(bar_width * frac), 8))
            y += self.text.line_height(metrics["body_size"]) + 1

        return y + metrics["section_gap"]

    def _draw_reproduction_overlay_block(self, surf, x, y, max_width, overlay_state: dict, metrics: dict):
        y = self._draw_header_line(surf, "Reproduction doctrines", metrics["section_header_size"], COLORS["text_header"], x, y)
        y += 2

        y = self._blit_wrapped(
            surf,
            f"Gate: {'enabled' if overlay_state.get('reproduction_enabled', True) else 'disabled'}  "
            f"Below floor: {overlay_state.get('below_floor_active', False)}",
            metrics["body_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )

        doctrines = overlay_state.get("doctrines", {})
        for key in ("crowding", "cooldown", "local_parent"):
            item = doctrines.get(key, {})
            enabled = bool(item.get("effective_enabled", False))
            marker = "*" if (
                cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_OVERRIDE_MARKERS
                and item.get("override_differs", False)
            ) else ""
            policy = item.get("active_policy", "-")
            label = item.get("short_name", key)
            line = f"{label}: {'ON' if enabled else 'OFF'}{marker} | policy={policy}"
            y = self._blit_wrapped(
                surf,
                line,
                metrics["body_size"],
                COLORS["text_dim"],
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )

        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_OVERRIDE_MARKERS:
            y = self._blit_wrapped(
                surf,
                "* runtime override differs from config default",
                max(10, metrics["body_size"] - 1),
                COLORS["text_warn"],
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )
        return y + metrics["section_gap"]

    def _draw_catastrophe_block(self, surf, x, y, max_width, catastrophe_state: dict, metrics: dict):
        y = self._draw_header_line(surf, "Catastrophes", metrics["section_header_size"], COLORS["text_header"], x, y)
        y += 2

        mode = catastrophe_state.get("mode", "off")
        global_enabled = bool(catastrophe_state.get("global_enabled", True))
        scheduler_armed = bool(catastrophe_state.get("scheduler_armed", False))
        scheduler_paused = bool(catastrophe_state.get("scheduler_paused", False))
        next_tick = catastrophe_state.get("next_auto_tick", None)
        manual_trigger_enabled = bool(catastrophe_state.get("manual_trigger_enabled", False))
        manual_clear_enabled = bool(catastrophe_state.get("manual_clear_enabled", False))

        if not global_enabled:
            scheduler_line = "Scheduler: globally disabled"
        elif mode in {"off", "manual_only"}:
            scheduler_line = "Scheduler: n/a for current mode"
        elif scheduler_armed:
            scheduler_state = "paused" if scheduler_paused else "running"
            scheduler_line = f"Scheduler: armed | {scheduler_state}"
        else:
            scheduler_line = "Scheduler: disarmed"

        y = self._blit_wrapped(
            surf,
            f"Mode: {mode}",
            metrics["body_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )
        y = self._blit_wrapped(
            surf,
            scheduler_line,
            metrics["body_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )
        y = self._blit_wrapped(
            surf,
            f"Manual trigger: {'ON' if manual_trigger_enabled else 'OFF'}  Clear: {'ON' if manual_clear_enabled else 'OFF'}",
            metrics["body_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )
        y = self._blit_wrapped(
            surf,
            f"Next auto tick: {next_tick}",
            metrics["body_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )

        active_names = catastrophe_state.get("active_names", [])
        if active_names:
            y = self._draw_header_line(surf, "Active:", metrics["body_size"], COLORS["text_warn"], x, y)
            for detail in catastrophe_state.get("active_details", [])[:3]:
                y = self._blit_wrapped(
                    surf,
                    f"{detail['display_name']} ({detail['remaining_ticks']}t)",
                    metrics["body_size"],
                    COLORS["text_harm"],
                    x + 8,
                    y,
                    max(1, max_width - 8),
                    line_gap=metrics["line_gap"],
                )
        else:
            y = self._blit_wrapped(
                surf,
                "Active: none",
                metrics["body_size"],
                COLORS["text_dim"],
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )
        return y + metrics["section_gap"]

    def _draw_hzone_details(self, surf, x, y, max_width, hzone_id, metrics: dict):
        zone = self.engine.grid.get_hzone(hzone_id)

        if not zone:
            return self._blit_wrapped(
                surf,
                f"H-Zone {hzone_id} (Error)",
                metrics["hint_size"],
                COLORS["text_warn"],
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )

        rate = zone["rate"]
        color = COLORS["text_success"] if rate >= 0 else COLORS["text_harm"]

        y = self._draw_header_line(surf, f"H-Zone ID: {hzone_id}", metrics["header_size"], color, x, y)
        y = self._blit_wrapped(
            surf,
            f"Coords: ({zone['x1']}, {zone['y1']}) to ({zone['x2']}, {zone['y2']})",
            metrics["hint_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )
        y = self._blit_wrapped(
            surf,
            f"Rate: {rate:.2f}",
            metrics["hint_size"],
            COLORS["text_dim"],
            x,
            y,
            max_width,
            line_gap=metrics["line_gap"],
        )
        return y

    def _agent_detail_lines(self, slot_id: int) -> list[tuple[str, tuple[int, int, int]]]:
        data = self.registry.data[:, slot_id]
        uid = self.registry.get_uid_for_slot(slot_id)
        family_id = self.registry.get_family_for_slot(slot_id)
        pos_x = int(data[Registry.X].item())
        pos_y = int(data[Registry.Y].item())
        hp = data[Registry.HP].item()
        hp_max = data[Registry.HP_MAX].item()
        mass = data[Registry.MASS].item()
        vision = data[Registry.VISION].item()
        metab = data[Registry.METABOLISM_RATE].item()

        lifecycle = self.registry.uid_lifecycle[uid]
        parent_roles = self.registry.get_parent_roles_for_uid(uid)
        birth_tick = int(lifecycle.birth_tick)
        age = int(self.engine.tick - birth_tick)
        lineage_depth = int(self.registry.uid_generation_depth.get(uid, 0))

        latent = self.registry.get_trait_latent_for_uid(uid)
        mapped = trait_values_from_latent(latent)

        training_state = self.engine.ppo.training_state_by_uid.get(uid)
        env_steps = 0 if training_state is None else int(training_state.env_steps)
        ppo_updates = 0 if training_state is None else int(training_state.ppo_updates)
        optimizer_steps = 0 if training_state is None else int(training_state.optimizer_steps)
        truncated_rollouts = 0 if training_state is None else int(training_state.truncated_rollouts)

        brain = self.registry.brains[slot_id]
        param_count = brain.get_param_count() if brain is not None else 0

        catastrophe_summary = {"survived_count": 0, "active_count": 0}
        if getattr(self.engine.logger, "get_catastrophe_exposure_summary", None) is not None:
            catastrophe_summary = self.engine.logger.get_catastrophe_exposure_summary(uid)

        lines = []
        if cfg.MIGRATION.VIEWER_SHOW_SLOT_AND_UID:
            lines.append((f"Slot: {slot_id}  UID: {uid}", COLORS["text_dim"]))
        else:
            lines.append((f"UID: {uid}", COLORS["text_success"]))
        if cfg.MIGRATION.VIEWER_SHOW_BLOODLINE:
            lines.append((f"Bloodline: {family_id}", get_bloodline_base_color(family_id)))
        lines.append((f"Age: {age}  Born: {birth_tick}  Gen: {lineage_depth}", COLORS["text_dim"]))
        lines.append((f"Parents B/T/A: {parent_roles['brain_parent_uid']} / {parent_roles['trait_parent_uid']} / {parent_roles['anchor_parent_uid']}", COLORS["text_dim"]))
        lines.append((f"HP: {hp:.1f} / {hp_max:.1f}  Pos: ({pos_x},{pos_y})", COLORS["text_dim"]))
        lines.append((f"Mass: {mass:.2f}  Vis: {vision:.1f}  Met: {metab:.4f}", COLORS["text_dim"]))
        lines.append((f"Params: {param_count:,}", COLORS["text_dim"]))
        if cfg.TELEMETRY.ENABLE_VIEWER_INSPECTOR_ENRICHMENT:
            lines.append((f"Budget: {mapped['budget']:.3f}  Alloc H/M/V/B: {mapped['alloc_hp']:.2f}/{mapped['alloc_mass']:.2f}/{mapped['alloc_vision']:.2f}/{mapped['alloc_metab']:.2f}", COLORS["text_dim"]))
            lines.append((f"PPO env/upd/opt: {env_steps}/{ppo_updates}/{optimizer_steps}  Trunc: {truncated_rollouts}", COLORS["text_dim"]))
            lines.append((f"Cata active/survived: {catastrophe_summary['active_count']}/{catastrophe_summary['survived_count']}", COLORS["text_dim"]))
        return lines

    def _draw_agent_details(self, surf, x, y, max_width, slot_id, metrics: dict):
        lines = self._agent_detail_lines(slot_id)

        self.viewer.last_selected_uid = self.registry.get_uid_for_slot(slot_id)
        hp = self.registry.data[self.registry.HP, slot_id].item()
        hp_max = self.registry.data[self.registry.HP_MAX, slot_id].item()
        hp_ratio = hp / (hp_max + 1e-6)

        for line, color in lines:
            y = self._blit_wrapped(
                surf,
                line,
                metrics["body_size"],
                color,
                x,
                y,
                max_width,
                line_gap=metrics["line_gap"],
            )

        bar_y = y + 1
        if surf is not None:
            pygame.draw.rect(surf, COLORS["bar_bg"], (x, bar_y, max_width, 8))
            pygame.draw.rect(surf, COLORS["bar_fg_hp"], (x, bar_y, int(max_width * hp_ratio), 8))
        return bar_y + 14
