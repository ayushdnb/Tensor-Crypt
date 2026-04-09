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
        if self.viewer.show_catastrophe_overlay:
            self._draw_catastrophe_overlay(surf, wrect, state_data)
        pygame.draw.rect(surf, COLORS["border"], wrect, 2)

    def _draw_agents(self, surf, wrect, c, state_data):
        for slot_id, agent in state_data["agent_map"].items():
            cx, cy = self.cam.world_to_screen(agent["x"], agent["y"])
            hp_ratio = agent["hp"] / (agent["hp_max"] + 1e-6)
            color = get_bloodline_agent_color(agent["family_id"], hp_ratio)
            agent_rect = (wrect.x + cx, wrect.y + cy, math.ceil(c), math.ceil(c))
            pygame.draw.rect(surf, color, agent_rect)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        if c < 8:
            return
        for slot_id, agent in state_data["agent_map"].items():
            hp_ratio = agent["hp"] / (agent["hp_max"] + 1e-6)
            cx, cy = self.cam.world_to_screen(agent["x"], agent["y"])
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
            agent = state_data["agent_map"][slot_id]
            cx, cy = self.cam.world_to_screen(agent["x"], agent["y"])
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

        agent = state_data["agent_map"][slot_id]
        agent_x = agent["x"]
        agent_y = agent["y"]
        start_pos_screen = (
            wrect.x + self.cam.world_to_screen(agent_x, agent_y)[0] + c // 2,
            wrect.y + self.cam.world_to_screen(agent_x, agent_y)[1] + c // 2,
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
            sx1, sy1 = self.cam.world_to_screen(x1, y1)
            sx2, sy2 = self.cam.world_to_screen(x2 + 1, y2 + 1)
            pygame.draw.rect(
                surf,
                (215, 80, 80),
                (wrect.x + sx1, wrect.y + sy1, max(1, sx2 - sx1), max(1, sy2 - sy1)),
                2,
            )

        woundtide_x = catastrophe_state.get("woundtide_front_x")
        if woundtide_x is not None:
            sx, _ = self.cam.world_to_screen(woundtide_x, 0)
            pygame.draw.line(surf, (220, 70, 120), (wrect.x + sx, wrect.y), (wrect.x + sx, wrect.bottom), 2)


class HudPanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.text = viewer.text_cache

    def draw(self, surf, state_data):
        hrect = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hrect)
        pygame.draw.rect(surf, COLORS["border"], hrect, 2)

        pad = 12
        x = hrect.x + pad
        y = hrect.y + 8

        if self.viewer.selected_hzone_id is not None:
            pause_str = "[ H-ZONE EDIT ]"
            pause_color = COLORS["selection_marker_hzone"]
        else:
            pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"
            pause_color = COLORS["pause_text"] if self.viewer.paused else COLORS["text"]

        surf.blit(self.text.render(f"Tick {self.viewer.engine.tick}", 18, COLORS["text"]), (x, y))
        surf.blit(self.text.render(pause_str, 16, pause_color), (x + 150, y + 2))

        y += 24
        alive_str = f"Alive: {state_data['num_alive']} / {self.viewer.engine.registry.max_agents}"
        surf.blit(self.text.render(alive_str, 14, COLORS["text_dim"]), (x, y))

        family_counts = state_data.get("family_alive_counts", {})
        bx = x + 220
        for family_id in cfg.BRAIN.FAMILY_ORDER:
            count = family_counts.get(family_id, 0)
            color = get_bloodline_base_color(family_id)
            pygame.draw.rect(surf, color, (bx, y + 2, 10, 10))
            bx += 14
            surf.blit(self.text.render(str(count), 13, COLORS["text_dim"]), (bx, y))
            bx += 28

        y += 22
        catastrophe_state = state_data.get("catastrophe_state", {})
        if cfg.VIEW.SHOW_CATASTROPHE_STATUS_IN_HUD:
            mode = catastrophe_state.get("mode", "off")
            active_names = catastrophe_state.get("active_names", [])
            next_tick = catastrophe_state.get("next_auto_tick", None)
            line = f"Cata: {mode}"
            if active_names:
                line += " | " + ", ".join(active_names[:2])
            if next_tick is not None and next_tick >= 0:
                line += f" | next={next_tick}"
            surf.blit(self.text.render(line, 12, COLORS["text_warn"]), (x, y))

        overlay_state = state_data.get("respawn_overlay_state", {})
        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_STATUS_IN_HUD:
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

            y += 16
            repro_gate = "EN" if overlay_state.get("reproduction_enabled", True) else "DIS"
            doctrine_line = (
                f"Repro:{repro_gate} | "
                f"{_fmt('crowding')} | "
                f"{_fmt('cooldown')} | "
                f"{_fmt('local_parent')}"
            )
            if overlay_state.get("below_floor_active", False):
                doctrine_line += " | floor-softened"
            surf.blit(self.text.render(doctrine_line, 12, COLORS["text_dim"]), (x, y))


class SidePanel:
    def __init__(self, viewer):
        self.viewer = viewer
        self.engine = viewer.engine
        self.registry = viewer.engine.registry
        self.text = viewer.text_cache
        self.line_height = 19

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)

        pad = 12
        y = srect.y + 10
        x = srect.x + pad

        surf.blit(self.text.render("Inspector", 18, COLORS["text_header"]), (x, y))
        y += 26

        slot_id = self.viewer.selected_slot_id
        hzone_id = self.viewer.selected_hzone_id

        if slot_id is not None:
            if slot_id not in state_data["agent_map"]:
                surf.blit(self.text.render(f"UID {self.viewer.last_selected_uid} (Dead)", 13, COLORS["text_warn"]), (x, y))
                y += self.line_height
            else:
                y = self._draw_agent_details(surf, x, y, slot_id)
        elif hzone_id is not None:
            y = self._draw_hzone_details(surf, x, y, hzone_id)
        else:
            surf.blit(self.text.render("Click an agent or H-Zone.", 13, COLORS["text_dim"]), (x, y))
            y += self.line_height

        y += 6
        y = self._draw_bloodline_legend(surf, x, y, state_data)
        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_STATUS_IN_PANEL:
            y = self._draw_reproduction_overlay_block(surf, x, y, state_data.get("respawn_overlay_state", {}))
        if self.viewer.show_catastrophe_panel:
            y = self._draw_catastrophe_block(surf, x, y, state_data.get("catastrophe_state", {}))

        y = max(y + 8, srect.bottom - 280)
        pygame.draw.line(surf, COLORS["border"], (srect.x, y - 6), (srect.right, y - 6), 1)
        surf.blit(self.text.render("Controls", 16, COLORS["text_header"]), (x, y))
        y += 22

        controls = [
            "Pan: WASD / Arrows",
            "Zoom: Mouse Wheel",
            "Fit world: F",
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
            "Cata F1..F12: trigger",
            "Clear: C  Mode: Y",
            "Auto: U  Panel: I  Pause: O",
        ]
        for line in controls:
            if line == "---":
                y += 4
                continue
            surf.blit(self.text.render(line, 12, COLORS["text_dim"]), (x, y))
            y += 16

    def _draw_bloodline_legend(self, surf, x, y, state_data):
        if not cfg.VIEW.SHOW_BLOODLINE_LEGEND:
            return y

        family_counts = state_data.get("family_alive_counts", {})
        total = max(1, sum(family_counts.values()))

        surf.blit(self.text.render("Bloodlines", 14, COLORS["text_header"]), (x, y))
        y += 18

        for family_id in cfg.BRAIN.FAMILY_ORDER:
            color = get_bloodline_base_color(family_id)
            count = family_counts.get(family_id, 0)
            pygame.draw.rect(surf, color, (x, y + 2, 10, 10))
            pygame.draw.rect(surf, COLORS["border"], (x, y + 2, 10, 10), 1)
            label = f"{family_id}  {count}"
            surf.blit(self.text.render(label, 12, COLORS["text_dim"]), (x + 16, y))
            bar_x = x + 220
            bar_w = 60
            frac = count / total
            pygame.draw.rect(surf, COLORS["bar_bg"], (bar_x, y + 3, bar_w, 8))
            pygame.draw.rect(surf, color, (bar_x, y + 3, int(bar_w * frac), 8))
            y += 17

        return y + 4

    def _draw_reproduction_overlay_block(self, surf, x, y, overlay_state: dict):
        surf.blit(self.text.render("Reproduction doctrines", 14, COLORS["text_header"]), (x, y))
        y += 18

        surf.blit(
            self.text.render(
                f"Gate: {'enabled' if overlay_state.get('reproduction_enabled', True) else 'disabled'}  "
                f"Below floor: {overlay_state.get('below_floor_active', False)}",
                12,
                COLORS["text_dim"],
            ),
            (x, y),
        )
        y += 16

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
            surf.blit(self.text.render(line, 12, COLORS["text_dim"]), (x, y))
            y += 16

        if cfg.RESPAWN.OVERLAYS.VIEWER.SHOW_OVERRIDE_MARKERS:
            surf.blit(self.text.render("* runtime override differs from config default", 11, COLORS["text_warn"]), (x, y))
            y += 16
        return y + 4

    def _draw_catastrophe_block(self, surf, x, y, catastrophe_state: dict):
        surf.blit(self.text.render("Catastrophes", 14, COLORS["text_header"]), (x, y))
        y += 18
        surf.blit(self.text.render(f"Mode: {catastrophe_state.get('mode', 'off')}", 12, COLORS["text_dim"]), (x, y))
        y += 16
        paused = catastrophe_state.get("scheduler_paused", False)
        next_tick = catastrophe_state.get("next_auto_tick", None)
        surf.blit(self.text.render(f"Paused: {paused}  Next: {next_tick}", 12, COLORS["text_dim"]), (x, y))
        y += 16

        active_names = catastrophe_state.get("active_names", [])
        if active_names:
            surf.blit(self.text.render("Active:", 12, COLORS["text_warn"]), (x, y))
            y += 16
            for detail in catastrophe_state.get("active_details", [])[:3]:
                surf.blit(
                    self.text.render(
                        f"  {detail['display_name']} ({detail['remaining_ticks']}t)",
                        12,
                        COLORS["text_harm"],
                    ),
                    (x, y),
                )
                y += 15
        else:
            surf.blit(self.text.render("Active: none", 12, COLORS["text_dim"]), (x, y))
            y += 16
        return y + 4

    def _draw_hzone_details(self, surf, x, y, hzone_id):
        lh = self.line_height
        zone = self.engine.grid.get_hzone(hzone_id)

        if not zone:
            surf.blit(self.text.render(f"H-Zone {hzone_id} (Error)", 13, COLORS["text_warn"]), (x, y))
            return y + lh

        rate = zone["rate"]
        color = COLORS["text_success"] if rate >= 0 else COLORS["text_harm"]

        surf.blit(self.text.render(f"H-Zone ID: {hzone_id}", 16, color), (x, y))
        y += lh
        surf.blit(self.text.render(f"Coords: ({zone['x1']}, {zone['y1']}) to ({zone['x2']}, {zone['y2']})", 13, COLORS["text_dim"]), (x, y))
        y += lh
        surf.blit(self.text.render(f"Rate: {rate:.2f}", 13, COLORS["text_dim"]), (x, y))
        y += lh
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

    def _draw_agent_details(self, surf, x, y, slot_id):
        lh = self.line_height
        lines = self._agent_detail_lines(slot_id)

        self.viewer.last_selected_uid = self.registry.get_uid_for_slot(slot_id)
        hp = self.registry.data[self.registry.HP, slot_id].item()
        hp_max = self.registry.data[self.registry.HP_MAX, slot_id].item()
        hp_ratio = hp / (hp_max + 1e-6)
        bar_w = self.viewer.layout.side_rect().width - 2 * 12

        for line, color in lines:
            surf.blit(self.text.render(line, 12, color), (x, y))
            y += lh

        pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 8))
        pygame.draw.rect(surf, COLORS["bar_fg_hp"], (x, y, bar_w * hp_ratio, 8))
        y += 14
        return y
