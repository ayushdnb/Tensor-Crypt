import math

import pygame

from ..agents.state_registry import Registry
from ..config_bridge import cfg
from .colors import COLORS


class WorldRenderer:
    def __init__(self, viewer):
        self.viewer = viewer
        self.cam = viewer.cam
        self.engine = viewer.engine
        self.registry = viewer.engine.registry
        self.grid = viewer.engine.grid
        self.static_surf = None

    def _build_static_cache(self, wrect):
        """Pre-render non-moving elements like walls and zones."""
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["empty"])

        occ_np = self.grid.grid[0].cpu().numpy()
        h_rate_np = self.grid.grid[1].cpu().numpy()
        H, W = occ_np.shape
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)

        for y in range(H):
            for x in range(W):
                cx, cy = self.cam.world_to_screen(x, y)
                c = self.cam.cell_px
                rect = (cx, cy, math.ceil(c), math.ceil(c))
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
        surf.blit(self.static_surf, wrect.topleft)

        self._draw_agents(surf, wrect, c, state_data)
        if self.viewer.show_hp_bars:
            self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.show_rays and self.viewer.selected_slot_id is not None:
            self._draw_rays(surf, wrect, c, state_data)
        self._draw_selection_markers(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6:
            self._draw_grid_lines(surf, wrect, c)
        pygame.draw.rect(surf, COLORS["border"], wrect, 2)

    def _draw_agents(self, surf, wrect, c, state_data):
        for slot_id, (x, y, hp, hp_max) in state_data["agent_map"].items():
            cx, cy = self.cam.world_to_screen(x, y)
            hp_ratio = hp / (hp_max + 1e-6)
            color = tuple(
                int(low * (1.0 - hp_ratio) + high * hp_ratio)
                for low, high in zip(COLORS["agent_low_hp"], COLORS["agent_high_hp"])
            )
            agent_rect = (wrect.x + cx, wrect.y + cy, math.ceil(c), math.ceil(c))
            pygame.draw.rect(surf, color, agent_rect)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        if c < 8:
            return
        for slot_id, (x, y, hp, hp_max) in state_data["agent_map"].items():
            hp_ratio = hp / (hp_max + 1e-6)
            cx, cy = self.cam.world_to_screen(x, y)
            bar_w, bar_h = c, max(1, c // 8)
            bar_y = wrect.y + cy - bar_h - 2
            if wrect.y < bar_y < wrect.bottom:
                bg_rect = (wrect.x + cx, bar_y, bar_w, bar_h)
                fg_rect = (wrect.x + cx, bar_y, bar_w * hp_ratio, bar_h)
                pygame.draw.rect(surf, COLORS["bar_bg"], bg_rect)
                pygame.draw.rect(surf, COLORS["bar_fg_hp"], fg_rect)

    def _draw_selection_markers(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id is not None and slot_id in state_data["agent_map"]:
            x, y, _, _ = state_data["agent_map"][slot_id]
            cx, cy = self.cam.world_to_screen(x, y)
            marker_rect = (wrect.x + cx, wrect.y + cy, math.ceil(c), math.ceil(c))
            pygame.draw.rect(surf, COLORS["selection_marker"], marker_rect, max(1, int(c // 10)))

        hzone_id = self.viewer.selected_hzone_id
        if hzone_id is not None:
            zone = self.engine.grid.get_hzone(hzone_id)
            if zone:
                cx1, cy1 = self.cam.world_to_screen(zone["x1"], zone["y1"])
                cx2, cy2 = self.cam.world_to_screen(zone["x2"] + 1, zone["y2"] + 1)
                screen_w = cx2 - cx1
                screen_h = cy2 - cy1
                marker_rect = (wrect.x + cx1, wrect.y + cy1, screen_w, screen_h)
                pygame.draw.rect(surf, COLORS["selection_marker_hzone"], marker_rect, max(1, int(c // 10)))

    def _draw_grid_lines(self, surf, wrect, c):
        ax, ay = self.cam.world_to_screen(0, 0)
        off_x, off_y = (c - (ax % c)) % c, (c - (ay % c)) % c
        x, y = wrect.x + off_x, wrect.y + off_y
        while x < wrect.right:
            pygame.draw.line(surf, COLORS["grid"], (x, wrect.y), (x, wrect.bottom))
            x += c
        while y < wrect.bottom:
            pygame.draw.line(surf, COLORS["grid"], (wrect.x, y), (wrect.right, y))
            y += c

    def _draw_rays(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id not in state_data["agent_map"]:
            return

        agent_x, agent_y, _, _ = state_data["agent_map"][slot_id]
        start_pos_screen = (
            wrect.x + self.cam.world_to_screen(agent_x, agent_y)[0] + c // 2,
            wrect.y + self.cam.world_to_screen(agent_x, agent_y)[1] + c // 2,
        )
        vision_range = int(self.registry.data[Registry.VISION, slot_id].item())
        occ_grid = self.grid.grid[0]
        agent_grid = self.grid.grid[2]
        H, W = self.grid.H, self.grid.W
        num_rays = self.engine.perception.num_rays

        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            dx, dy = math.cos(angle), math.sin(angle)
            end_x, end_y, color = agent_x, agent_y, COLORS["ray_empty"]

            for step in range(1, vision_range + 1):
                tx, ty = int(round(agent_x + dx * step)), int(round(agent_y + dy * step))
                if not (0 <= tx < W and 0 <= ty < H):
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


class HudPanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.text = viewer.text_cache

    def draw(self, surf, state_data):
        hrect = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hrect)
        pygame.draw.rect(surf, COLORS["border"], hrect, 2)

        pad, y = 12, hrect.y + 10
        x = hrect.x + pad

        if self.viewer.selected_hzone_id is not None:
            pause_str = "[ H-ZONE EDIT ]"
            pause_color = COLORS["selection_marker_hzone"]
        else:
            pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"
            pause_color = COLORS["pause_text"] if self.viewer.paused else COLORS["text"]

        surf.blit(self.text.render(f"Tick {self.viewer.engine.tick}", 18, COLORS["text"]), (x, y))
        surf.blit(self.text.render(pause_str, 16, pause_color), (x + 150, y + 2))

        y += 28
        alive_str = f"Alive: {state_data['num_alive']} / {self.viewer.engine.registry.max_agents}"
        surf.blit(self.text.render(alive_str, 16, COLORS["text_dim"]), (x, y))


class SidePanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.engine = viewer.engine
        self.registry = viewer.engine.registry
        self.text = viewer.text_cache
        self.line_height = 20

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)

        pad, y = 12, srect.y + 12
        x = srect.x + pad

        surf.blit(self.text.render("Inspector", 18, COLORS["text_header"]), (x, y))
        y += 30

        slot_id = self.viewer.selected_slot_id
        hzone_id = self.viewer.selected_hzone_id

        if slot_id is not None:
            if slot_id not in state_data["agent_map"]:
                surf.blit(self.text.render(f"UID {self.viewer.last_selected_uid} (Dead)", 13, COLORS["text_warn"]), (x, y))
            else:
                y = self._draw_agent_details(surf, x, y, slot_id, state_data)
        elif hzone_id is not None:
            y = self._draw_hzone_details(surf, x, y, hzone_id)
        else:
            surf.blit(self.text.render("Click an agent or H-Zone to inspect.", 13, COLORS["text_dim"]), (x, y))

        y = srect.bottom - 200
        pygame.draw.line(surf, COLORS["border"], (srect.x, y - 10), (srect.right, y - 10), 2)
        surf.blit(self.text.render("Controls", 18, COLORS["text_header"]), (x, y))
        y += 30

        controls = [
            "Pan: WASD / Arrows",
            "Zoom: Mouse Wheel",
            "Pause: SPACE",
            "Speed: +/- (when no zone selected)",
            "Step (Paused): . (Period)",
            "---",
            "Edit H-Zone Rate: +/- (when zone selected)",
            "Toggle Rays (R)",
            "Toggle HP Bars (B)",
            "Toggle H-Zones (H)",
            "Toggle Grid (G)",
        ]
        for line in controls:
            surf.blit(self.text.render(line, 13, COLORS["text_dim"]), (x, y))
            y += 18

    def _draw_hzone_details(self, surf, x, y, hzone_id):
        lh = self.line_height
        zone = self.engine.grid.get_hzone(hzone_id)

        if not zone:
            surf.blit(self.text.render(f"H-Zone {hzone_id} (Error)", 13, COLORS["text_warn"]), (x, y))
            return y + lh

        rate = zone["rate"]
        color = COLORS["text_success"] if rate >= 0 else COLORS["hzone_harm"]

        surf.blit(self.text.render(f"H-Zone ID: {hzone_id}", 16, color), (x, y))
        y += lh
        surf.blit(self.text.render(f"Coords: ({zone['x1']}, {zone['y1']}) to ({zone['x2']}, {zone['y2']})", 13, COLORS["text_dim"]), (x, y))
        y += lh

        rate_ratio = (rate + 2.0) / 4.0
        bar_w = self.viewer.layout.side_rect().width - 2 * 12
        pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 10))
        pygame.draw.rect(surf, color, (x, y, bar_w * rate_ratio, 10))

        zero_x = x + (bar_w * (2.0 / 4.0))
        pygame.draw.line(surf, COLORS["border"], (zero_x, y), (zero_x, y + 10), 1)
        y += 14

        surf.blit(self.text.render(f"Rate: {rate:.2f} (Range: -2.0 to 2.0)", 13, COLORS["text_dim"]), (x, y))
        y += lh + 10
        surf.blit(self.text.render("Use [ - ] and [ + ] keys", 13, COLORS["text_header"]), (x, y))
        y += lh
        surf.blit(self.text.render("to change the rate.", 13, COLORS["text_header"]), (x, y))
        y += lh
        return y

    def _draw_agent_details(self, surf, x, y, slot_id, state_data):
        lh = self.line_height
        data = self.registry.data[:, slot_id]
        uid = self.registry.get_uid_for_slot(slot_id)
        parent_uid = self.registry.get_parent_uid_for_slot(slot_id)
        pos_x = int(data[Registry.X].item())
        pos_y = int(data[Registry.Y].item())
        hp = data[Registry.HP].item()
        hp_max = data[Registry.HP_MAX].item()
        mass = data[Registry.MASS].item()
        vision = data[Registry.VISION].item()
        metab = data[Registry.METABOLISM_RATE].item()
        fitness = data[Registry.HP_GAINED].item()

        brain = self.registry.brains[slot_id]
        param_count_str = "N/A"
        if brain is not None:
            try:
                uncompiled_brain = getattr(brain, "_orig_mod", brain)
                if hasattr(uncompiled_brain, "get_param_count"):
                    param_count = uncompiled_brain.get_param_count()
                    param_count_str = f"{param_count:,}"
                else:
                    param_count_str = "N/A (no method)"
            except Exception:
                param_count_str = "Error"

        self.viewer.last_selected_uid = uid

        if cfg.MIGRATION.VIEWER_SHOW_SLOT_AND_UID:
            surf.blit(self.text.render(f"Slot: {slot_id}", 13, COLORS["text_dim"]), (x, y))
            y += lh
        surf.blit(self.text.render(f"UID: {uid}", 16, COLORS["text_success"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Parent UID: {parent_uid if parent_uid != -1 else 'N/A'}", 13, COLORS["text_dim"]), (x, y))
        y += lh

        hp_ratio = hp / (hp_max + 1e-6)
        bar_w = self.viewer.layout.side_rect().width - 2 * 12
        pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 10))
        pygame.draw.rect(surf, COLORS["bar_fg_hp"], (x, y, bar_w * hp_ratio, 10))
        y += 14
        surf.blit(self.text.render(f"HP: {hp:.2f} / {hp_max:.2f} ({hp_ratio * 100:.0f}%)", 13, COLORS["text_dim"]), (x, y))
        y += lh + 5

        surf.blit(self.text.render(f"Position: ({pos_x}, {pos_y})", 13, COLORS["text_dim"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Mass: {mass:.2f}", 13, COLORS["text_dim"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Vision: {vision:.1f}", 13, COLORS["text_dim"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Metabolism: {metab:.4f} (HP/tick)", 13, COLORS["text_dim"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Fitness (Score): {fitness:.2f}", 13, COLORS["text_dim"]), (x, y))
        y += lh

        y += 5
        surf.blit(self.text.render(f"Brain Params: {param_count_str}", 13, COLORS["text_dim"]), (x, y))
        y += lh
        return y

