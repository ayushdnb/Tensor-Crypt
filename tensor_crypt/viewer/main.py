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
    def __init__(self, engine):
        pygame.init()
        pygame.font.init()

        self.engine = engine
        self.Wpix, self.Hpix = cfg.VIEW.WINDOW_WIDTH, cfg.VIEW.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)
        pygame.display.set_caption("Evolution Simulation (Advanced Viewer)")

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

        self.show_rays = False
        self.show_hp_bars = True
        self.show_hzones = True
        self.show_grid = True

        self.selected_slot_id = None
        self.selected_hzone_id = None
        self.last_selected_uid = -1

        self.world_renderer = WorldRenderer(self)
        self.hud_panel = HudPanel(self)
        self.side_panel = SidePanel(self)
        self.input_handler = InputHandler(self)

    def _prepare_state_data(self):
        """Create a snapshot of the current state for rendering."""
        with torch.no_grad():
            registry = self.engine.registry
            alive_indices = registry.get_alive_indices().cpu().tolist()
            agent_map = {}
            for idx in alive_indices:
                agent_map[idx] = (
                    registry.data[registry.X, idx].item(),
                    registry.data[registry.Y, idx].item(),
                    registry.data[registry.HP, idx].item(),
                    registry.data[registry.HP_MAX, idx].item(),
                )
            return {"num_alive": len(alive_indices), "agent_map": agent_map}

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
                self.engine.step()

            if self.frame_count % render_every_n_frames == 0:
                state_data = self._prepare_state_data()
                self.screen.fill(COLORS["bg"])
                self.world_renderer.draw(self.screen, state_data)
                self.hud_panel.draw(self.screen, state_data)
                self.side_panel.draw(self.screen, state_data)
                pygame.display.flip()

            self.clock.tick(cfg.VIEW.FPS)
            self.frame_count += 1

        pygame.quit()
