from ..config_bridge import cfg
import pygame


class InputHandler:
    def __init__(self, viewer):
        self.viewer = viewer
        self.cam = viewer.cam
        self.engine = viewer.engine

    def _handle_catastrophe_hotkey(self, ev) -> None:
        if not cfg.CATASTROPHE.VIEWER_CONTROLS_ENABLED:
            return

        key_to_idx = {
            pygame.K_F1: 0,
            pygame.K_F2: 1,
            pygame.K_F3: 2,
            pygame.K_F4: 3,
            pygame.K_F5: 4,
            pygame.K_F6: 5,
            pygame.K_F7: 6,
            pygame.K_F8: 7,
            pygame.K_F9: 8,
            pygame.K_F10: 9,
            pygame.K_F11: 10,
            pygame.K_F12: 11,
        }
        if ev.key in key_to_idx:
            self.engine.catastrophes.manual_trigger_by_index(key_to_idx[ev.key], self.engine.tick)
            self.viewer.world_renderer.static_surf = None
            return

        if ev.key == pygame.K_c:
            self.engine.catastrophes.manual_clear(self.engine.tick)
            self.viewer.world_renderer.static_surf = None
        elif ev.key == pygame.K_y:
            self.engine.catastrophes.cycle_mode()
        elif ev.key == pygame.K_u:
            self.engine.catastrophes.toggle_auto_enable()
        elif ev.key == pygame.K_i:
            self.viewer.show_catastrophe_panel = not self.viewer.show_catastrophe_panel
            self.viewer.show_catastrophe_overlay = self.viewer.show_catastrophe_panel
        elif ev.key == pygame.K_o:
            self.engine.catastrophes.toggle_scheduler_pause()

    def handle(self):
        running = True
        advance_tick = False

        keys = pygame.key.get_pressed()
        pan_speed = 10.0 / max(1.0, float(self.cam.cell_px))
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.cam.pan(-pan_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.cam.pan(pan_speed, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.cam.pan(0, -pan_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.cam.pan(0, pan_speed)

        wrect = self.viewer.layout.world_rect()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                self.viewer.Wpix, self.viewer.Hpix = max(1024, ev.w), max(768, ev.h)
                self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                self.cam.update_screen_size(wrect.width, wrect.height)
                self.viewer.world_renderer.static_surf = None
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    self.viewer.paused = not self.viewer.paused
                elif ev.key == pygame.K_PERIOD and self.viewer.paused:
                    advance_tick = True
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    if self.viewer.selected_hzone_id is not None:
                        zone = self.engine.grid.get_hzone(self.viewer.selected_hzone_id)
                        if zone:
                            new_rate = zone["rate"] + cfg.VIEW.PAINT_RATE_STEP
                            self.engine.grid.update_hzone_rate(self.viewer.selected_hzone_id, new_rate)
                            self.viewer.world_renderer.static_surf = None
                    else:
                        self.viewer.speed_multiplier = min(128, self.viewer.speed_multiplier * 2)
                elif ev.key == pygame.K_MINUS:
                    if self.viewer.selected_hzone_id is not None:
                        zone = self.engine.grid.get_hzone(self.viewer.selected_hzone_id)
                        if zone:
                            new_rate = zone["rate"] - cfg.VIEW.PAINT_RATE_STEP
                            self.engine.grid.update_hzone_rate(self.viewer.selected_hzone_id, new_rate)
                            self.viewer.world_renderer.static_surf = None
                    else:
                        self.viewer.speed_multiplier = max(0.125, self.viewer.speed_multiplier / 2)
                elif ev.key == pygame.K_r:
                    self.viewer.show_rays = not self.viewer.show_rays
                elif ev.key == pygame.K_b:
                    self.viewer.show_hp_bars = not self.viewer.show_hp_bars
                elif ev.key == pygame.K_h:
                    self.viewer.show_hzones = not self.viewer.show_hzones
                    self.viewer.world_renderer.static_surf = None
                elif ev.key == pygame.K_g:
                    self.viewer.show_grid = not self.viewer.show_grid
                else:
                    self._handle_catastrophe_hotkey(ev)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if wrect.collidepoint(ev.pos):
                    if ev.button == 1:
                        gx, gy = self.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                        slot_id = self.engine.grid.get_agent_at(gx, gy)
                        if slot_id >= 0:
                            self.viewer.selected_slot_id = slot_id
                            self.viewer.selected_hzone_id = None
                        else:
                            zone_id = self.engine.grid.find_hzone_at(gx, gy)
                            if zone_id is not None:
                                self.viewer.selected_hzone_id = zone_id
                                self.viewer.selected_slot_id = None
                            else:
                                self.viewer.selected_slot_id = None
                                self.viewer.selected_hzone_id = None
                    elif ev.button == 4:
                        self.cam.zoom_at(1.15, ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                        self.viewer.world_renderer.static_surf = None
                    elif ev.button == 5:
                        self.cam.zoom_at(1 / 1.15, ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                        self.viewer.world_renderer.static_surf = None

        return running, advance_tick