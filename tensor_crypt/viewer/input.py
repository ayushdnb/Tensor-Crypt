
"""Viewer input routing and interaction semantics."""

from ..config_bridge import cfg
import pygame


class InputHandler:
    def __init__(self, viewer):
        self.viewer = viewer
        self.cam = viewer.cam
        self.engine = viewer.engine

    @staticmethod
    def _is_fullscreen_hotkey(ev) -> bool:
        mods = getattr(ev, "mod", 0)
        return bool(mods & pygame.KMOD_ALT) and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER)

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
            self.engine.catastrophes.clear_active_catastrophes(self.engine.tick)
            self.viewer.world_renderer.static_surf = None
        elif ev.key == pygame.K_y:
            self.engine.catastrophes.cycle_mode(self.engine.tick)
        elif ev.key == pygame.K_u:
            self.engine.catastrophes.toggle_scheduler_armed(self.engine.tick)
        elif ev.key == pygame.K_i:
            self.viewer.show_catastrophe_panel = not self.viewer.show_catastrophe_panel
            self.viewer.show_catastrophe_overlay = bool(
                self.viewer.show_catastrophe_panel and cfg.CATASTROPHE.VIEWER_OVERLAY_ENABLED
            )
        elif ev.key == pygame.K_o:
            self.engine.catastrophes.toggle_scheduler_pause(self.engine.tick)

    def _handle_reproduction_overlay_hotkey(self, ev) -> bool:
        if not cfg.RESPAWN.OVERLAYS.VIEWER.HOTKEYS_ENABLED:
            return False

        mods = getattr(ev, "mod", 0)
        if not (mods & pygame.KMOD_SHIFT):
            return False

        rc = self.engine.respawn_controller
        if ev.key == pygame.K_1:
            rc.toggle_doctrine_override("crowding")
            return True
        if ev.key == pygame.K_2:
            rc.toggle_doctrine_override("cooldown")
            return True
        if ev.key == pygame.K_3:
            rc.toggle_doctrine_override("local_parent")
            return True
        if ev.key == pygame.K_0:
            rc.clear_doctrine_overrides()
            return True
        return False

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

    def _scroll_side_panel(self, direction_steps: int) -> bool:
        state_data = self.viewer._last_state_data
        if state_data is None:
            return False
        start = self.viewer.side_panel.scroll_offset
        self.viewer.side_panel.scroll_by(direction_steps, state_data)
        return self.viewer.side_panel.scroll_offset != start

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
        srect = self.viewer.layout.side_rect()
        window_resized_event = getattr(pygame, "WINDOWRESIZED", None)
        events = pygame.event.get()
        has_mousewheel_event = any(ev.type == pygame.MOUSEWHEEL for ev in events)

        for ev in events:
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE or (window_resized_event is not None and ev.type == window_resized_event):
                if not self.viewer.is_fullscreen:
                    width = getattr(ev, "w", self.viewer.screen.get_width())
                    height = getattr(ev, "h", self.viewer.screen.get_height())
                    self.viewer.handle_window_resize(width, height)
                    wrect = self.viewer.layout.world_rect()
                    srect = self.viewer.layout.side_rect()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif self._is_fullscreen_hotkey(ev):
                    self.viewer.toggle_fullscreen()
                    wrect = self.viewer.layout.world_rect()
                    srect = self.viewer.layout.side_rect()
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
                elif self._handle_reproduction_overlay_hotkey(ev):
                    pass
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
                elif srect.collidepoint(ev.pos) and ev.button in (4, 5) and not has_mousewheel_event:
                    self._scroll_side_panel(-1 if ev.button == 4 else 1)
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if wrect.collidepoint(mx, my):
                    if ev.y > 0:
                        self.cam.zoom_at(1.15, mx - wrect.x, my - wrect.y)
                    elif ev.y < 0:
                        self.cam.zoom_at(1 / 1.15, mx - wrect.x, my - wrect.y)
                    self.viewer.world_renderer.static_surf = None
                elif srect.collidepoint(mx, my):
                    self._scroll_side_panel(-int(ev.y))

        return running, advance_tick