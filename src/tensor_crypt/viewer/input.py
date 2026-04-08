"""Viewer input routing and interaction semantics."""

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
            self.viewer.show_catastrophe_overlay = bool(
                self.viewer.show_catastrophe_panel and cfg.CATASTROPHE.VIEWER_OVERLAY_ENABLED
            )
        elif ev.key == pygame.K_o:
            self.engine.catastrophes.toggle_scheduler_pause()

    def _find_nearest_agent_screen(self, mx, my, wrect, state_data):
        """Pick an agent using screen-space proximity for stable zoomed interaction."""
        c = float(self.cam.cell_px)
        best_slot = None
        best_dist_sq = float("inf")
        tol = max(c * 0.75, 6.0)
        tol_sq = tol * tol

        for slot_id, agent in state_data["agent_map"].items():
            ax, ay = self.cam.world_to_screen(agent["x"], agent["y"])
            center_x = wrect.x + ax + c / 2.0
            center_y = wrect.y + ay + c / 2.0
            dx = mx - center_x
            dy = my - center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < tol_sq and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_slot = slot_id

        return best_slot

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
        events = pygame.event.get()
        has_mousewheel_event = any(ev.type == pygame.MOUSEWHEEL for ev in events)

        for ev in events:
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                self.viewer.Wpix, self.viewer.Hpix = max(1024, ev.w), max(768, ev.h)
                self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                wrect = self.viewer.layout.world_rect()
                self.cam.update_screen_size(wrect.width, wrect.height)
                self.cam.fit_to_world()
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
                elif ev.key == pygame.K_f:
                    self.cam.fit_to_world()
                    self.viewer.world_renderer.static_surf = None
                else:
                    self._handle_catastrophe_hotkey(ev)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if wrect.collidepoint(ev.pos):
                    if ev.button == 1:
                        state_data = self.viewer._last_state_data
                        agent_slot = None
                        if state_data is not None:
                            agent_slot = self._find_nearest_agent_screen(ev.pos[0], ev.pos[1], wrect, state_data)

                        if agent_slot is None:
                            gx, gy = self.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                            cell_slot = self.engine.grid.get_agent_at(gx, gy)
                            if cell_slot >= 0:
                                agent_slot = cell_slot

                        if agent_slot is not None:
                            self.viewer.selected_slot_id = agent_slot
                            self.viewer.selected_hzone_id = None
                        else:
                            gx, gy = self.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                            zone_id = self.engine.grid.find_hzone_at(gx, gy)
                            if zone_id is not None:
                                self.viewer.selected_hzone_id = zone_id
                                self.viewer.selected_slot_id = None
                            else:
                                self.viewer.selected_slot_id = None
                                self.viewer.selected_hzone_id = None
                    elif ev.button in (4, 5) and not has_mousewheel_event:
                        factor = 1.15 if ev.button == 4 else 1 / 1.15
                        self.cam.zoom_at(factor, ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                        self.viewer.world_renderer.static_surf = None
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if wrect.collidepoint(mx, my):
                    if ev.y > 0:
                        self.cam.zoom_at(1.15, mx - wrect.x, my - wrect.y)
                    elif ev.y < 0:
                        self.cam.zoom_at(1 / 1.15, mx - wrect.x, my - wrect.y)
                    self.viewer.world_renderer.static_surf = None

        return running, advance_tick
