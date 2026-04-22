
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

    INITIAL_WINDOW_MIN_WIDTH = 800
    INITIAL_WINDOW_MIN_HEIGHT = 600

    def __init__(self, engine):
        pygame.init()
        pygame.font.init()

        self.engine = engine
        self.Wpix = max(self.INITIAL_WINDOW_MIN_WIDTH, int(cfg.VIEW.WINDOW_WIDTH))
        self.Hpix = max(self.INITIAL_WINDOW_MIN_HEIGHT, int(cfg.VIEW.WINDOW_HEIGHT))
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
        self.is_fullscreen = False
        self._windowed_size = (self.Wpix, self.Hpix)

        self.selected_slot_id = None
        self.selected_hzone_id = None
        self.last_selected_uid = -1
        self._last_catastrophe_visual_version = -1
        self._last_state_data = None
        self.finalize_callback = None

        self.world_renderer = WorldRenderer(self)
        self.hud_panel = HudPanel(self)
        self.side_panel = SidePanel(self)
        self.input_handler = InputHandler(self)

    def _refresh_view_geometry(self, *, refit_world: bool) -> None:
        wrect = self.layout.world_rect()
        self.cam.update_screen_size(wrect.width, wrect.height)
        if refit_world:
            self.cam.fit_to_world()
        else:
            self.cam._clamp_offsets()
        self.world_renderer.static_surf = None
        if self._last_state_data is not None:
            self.side_panel.clamp_scroll_offset(self._last_state_data)

    def _sync_surface_size(self, width: int, height: int, *, refit_world: bool, remember_windowed: bool) -> None:
        self.Wpix = max(1, int(width))
        self.Hpix = max(1, int(height))
        if remember_windowed and not self.is_fullscreen:
            self._windowed_size = (self.Wpix, self.Hpix)
        self._refresh_view_geometry(refit_world=refit_world)

    def handle_window_resize(self, width: int, height: int) -> None:
        """
        Update viewer geometry after a resizable-window event.

        In pygame 2, the display surface is already resized for VIDEORESIZE, so
        this method intentionally avoids calling set_mode() again.
        """
        self.screen = pygame.display.get_surface() or self.screen
        self._sync_surface_size(width, height, refit_world=False, remember_windowed=True)

    def resize_window_mode(self, width: int, height: int) -> None:
        self.screen = pygame.display.set_mode((max(1, int(width)), max(1, int(height))), pygame.RESIZABLE)
        width, height = self.screen.get_size()
        self._sync_surface_size(width, height, refit_world=False, remember_windowed=True)

    def _fullscreen_size(self) -> tuple[int, int]:
        get_desktop_sizes = getattr(pygame.display, "get_desktop_sizes", None)
        if callable(get_desktop_sizes):
            sizes = get_desktop_sizes()
            if sizes:
                width, height = sizes[0]
                if width > 0 and height > 0:
                    return int(width), int(height)

        info = pygame.display.Info()
        width = getattr(info, "current_w", 0) or self.Wpix
        height = getattr(info, "current_h", 0) or self.Hpix
        return max(1, int(width)), max(1, int(height))

    def toggle_fullscreen(self) -> None:
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            width, height = self.screen.get_size()
            self._sync_surface_size(width, height, refit_world=False, remember_windowed=False)
            return

        self._windowed_size = (self.Wpix, self.Hpix)
        self.is_fullscreen = True
        self.screen = pygame.display.set_mode(self._fullscreen_size(), pygame.FULLSCREEN)
        width, height = self.screen.get_size()
        self._sync_surface_size(width, height, refit_world=False, remember_windowed=False)

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
            respawn_overlay_state = self.engine.respawn_controller.build_overlay_status(len(alive_indices))
            return {
                "num_alive": len(alive_indices),
                "agent_map": agent_map,
                "catastrophe_state": catastrophe_state,
                "respawn_overlay_state": respawn_overlay_state,
                "family_alive_counts": family_alive_counts,
            }

    def run(self):
        running = True
        render_every_n_frames = 2

        try:
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
                if num_ticks_this_frame == 0 and hasattr(self.engine, "maybe_save_runtime_checkpoint_wallclock"):
                    self.engine.maybe_save_runtime_checkpoint_wallclock(paused=self.paused)

                state_data = self._prepare_state_data()
                self._last_state_data = state_data
                self.side_panel.clamp_scroll_offset(state_data)

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

        finally:
            try:
                if callable(self.finalize_callback):
                    self.finalize_callback()
            finally:
                pygame.quit()
