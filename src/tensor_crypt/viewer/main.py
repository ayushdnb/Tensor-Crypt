"""Pygame viewer runtime for Tensor Crypt."""

import pygame
import torch

from ..config_bridge import cfg
from .camera import Camera
from .colors import COLORS
from .input import InputHandler
from .layout import LayoutManager
from .panels import HudPanel, SidePanel, WorldRenderer
from .text_cache import TextCache


class Viewer:
    """Own the interactive render loop and UI state."""

    def __init__(self, engine):
        pygame.init()
        pygame.font.init()

        self.engine = engine
        self.Wpix, self.Hpix = max(1024, cfg.VIEW.WINDOW_WIDTH), max(768, cfg.VIEW.WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)
        pygame.display.set_caption("Tensor Crypt")

        self.clock = pygame.time.Clock()
        self.text_cache = TextCache()
        self.layout = LayoutManager(self)
        self.cam = Camera(
            self.layout.world_rect().width,
            self.layout.world_rect().height,
            cfg.GRID.W,
            cfg.GRID.H,
        )
        self.paused = False
        self.speed_multiplier = 1.0
        self.frame_count = 0

        overlay_defaults = dict(cfg.VIEW.SHOW_OVERLAYS)
        self.show_rays = bool(overlay_defaults.get("rays", False))
        self.show_hp_bars = True
        self.show_hzones = bool(overlay_defaults.get("h_rate", True))
        self.show_grid = True

        self.show_catastrophe_panel = cfg.VIEW.SHOW_CATASTROPHE_PANEL
        self.show_catastrophe_overlay = bool(cfg.VIEW.SHOW_CATASTROPHE_OVERLAY and cfg.CATASTROPHE.VIEWER_OVERLAY_ENABLED)

        self.selected_slot_id = None
        self.selected_hzone_id = None
        self.last_selected_uid = -1
        self._last_catastrophe_visual_version = -1
        self._last_state_data = None

        self.world_renderer = WorldRenderer(self)
        self.hud_panel = HudPanel(self)
        self.side_panel = SidePanel(self)
        self.input_handler = InputHandler(self)

    def _prepare_state_data(self):
        with torch.no_grad():
            registry = self.engine.registry
            alive_indices = registry.get_alive_indices().cpu().tolist()
            agent_map = {}
            family_alive_counts = {fid: 0 for fid in cfg.BRAIN.FAMILY_ORDER}
            for idx in alive_indices:
                family_id = registry.get_family_for_slot(idx)
                agent_map[idx] = {
                    "x": registry.data[registry.X, idx].item(),
                    "y": registry.data[registry.Y, idx].item(),
                    "hp": registry.data[registry.HP, idx].item(),
                    "hp_max": registry.data[registry.HP_MAX, idx].item(),
                    "family_id": family_id,
                }
                if family_id not in family_alive_counts:
                    family_alive_counts[family_id] = 0
                family_alive_counts[family_id] += 1
            catastrophe_state = self.engine.catastrophes.build_status(self.engine.tick)
            return {
                "num_alive": len(alive_indices),
                "agent_map": agent_map,
                "catastrophe_state": catastrophe_state,
                "family_alive_counts": family_alive_counts,
            }

    def run(self):
        running = True
        render_every_n_frames = 2

        while running:
            running, advance_tick = self.input_handler.handle()

            num_ticks_this_frame = 0
            if not self.paused:
                if self.speed_multiplier >= 1.0:
                    num_ticks_this_frame = int(self.speed_multiplier * render_every_n_frames)
                else:
                    num_ticks_this_frame = 1 if self.frame_count % int(1.0 / self.speed_multiplier) == 0 else 0
            elif advance_tick:
                num_ticks_this_frame = 1

            for _ in range(num_ticks_this_frame):
                if cfg.SIM.MAX_TICKS > 0 and self.engine.tick >= int(cfg.SIM.MAX_TICKS):
                    running = False
                    break
                self.engine.step()

            state_data = self._prepare_state_data()
            self._last_state_data = state_data
            visual_version = state_data["catastrophe_state"].get("visual_state_version", 0)
            if visual_version != self._last_catastrophe_visual_version:
                self.world_renderer.static_surf = None
                self._last_catastrophe_visual_version = visual_version

            if self.frame_count % render_every_n_frames == 0:
                self.screen.fill(COLORS["bg"])
                self.world_renderer.draw(self.screen, state_data)
                self.hud_panel.draw(self.screen, state_data)
                self.side_panel.draw(self.screen, state_data)
                pygame.display.flip()

            self.clock.tick(cfg.VIEW.FPS)
            self.frame_count += 1

        pygame.quit()
