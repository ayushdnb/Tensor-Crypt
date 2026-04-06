"""Observation construction and ray-casting for live agents.

The canonical observation layout is shared by the brain, PPO storage, and
checkpoint surfaces, so feature ordering is a cross-module contract rather than
an implementation detail of this file.
"""

import math

import torch

from ..config_bridge import cfg
from .observation_schema import (
    build_empty_observation_batch,
    build_observation_bundle,
    normalize_from_bounds,
    normalize_signed,
)


class Perception:
    def __init__(self, grid, registry):
        self.grid = grid
        self.registry = registry
        self.num_rays = cfg.PERCEPT.NUM_RAYS

        angles = torch.linspace(0, 2 * math.pi, self.num_rays + 1, device=grid.device)[:-1]
        self.ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        self.vision_scale = 1.0

    def reset_runtime_modifiers(self) -> None:
        self.vision_scale = 1.0

    def set_runtime_modifiers(self, *, vision_scale: float = 1.0) -> None:
        self.vision_scale = max(0.0, float(vision_scale))

    def get_effective_vision_values(self, alive_indices: torch.Tensor) -> torch.Tensor:
        base = self.registry.data[self.registry.VISION, alive_indices]
        scaled = base * self.vision_scale
        return torch.clamp(scaled, min=0.0)

    def get_effective_vision_for_slot(self, slot_idx: int) -> float:
        base = float(self.registry.data[self.registry.VISION, slot_idx].item())
        return max(0.0, base * self.vision_scale)

    @torch.no_grad()
    def cast_rays_batched(
        self,
        positions: torch.Tensor,
        vision_ranges: torch.Tensor,
        self_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Vectorized ray casting for multiple agents.

        Prompt 2 canonical contract per ray:
        [hit_none, hit_agent, hit_wall, hit_distance_norm,
         path_zone_peak_rate_norm, terminal_zone_rate_norm,
         target_mass_norm, target_hp_ratio]
        """

        n_agents = positions.shape[0]
        if n_agents == 0:
            return torch.empty(
                0,
                self.num_rays,
                cfg.PERCEPT.CANONICAL_RAY_FEATURES,
                device=positions.device,
            )

        device = positions.device
        results = torch.zeros(
            n_agents,
            self.num_rays,
            cfg.PERCEPT.CANONICAL_RAY_FEATURES,
            device=device,
        )
        ray_dirs = self.ray_dirs.unsqueeze(0).expand(n_agents, -1, -1)
        safe_vision_ranges = vision_ranges.clamp_min(1e-6)

        if self_indices is None:
            self_indices = torch.full((n_agents,), -1, device=device, dtype=torch.long)
        self_indices = self_indices.to(device=device, dtype=torch.long)

        occ_grid = self.grid.grid[0]
        agent_grid = self.grid.grid[2]
        h_rate_grid = self.grid.grid[1]
        grid_h, grid_w = self.grid.H, self.grid.W

        max_vision = int(max(float(vision_ranges.max().item()), 0.0))
        if max_vision <= 0:
            return results

        path_zone_peak_raw = torch.zeros(n_agents, self.num_rays, device=device)
        terminal_zone_rate_raw = torch.zeros(n_agents, self.num_rays, device=device)

        mass_norm_by_slot = normalize_from_bounds(
            self.registry.data[self.registry.MASS],
            cfg.TRAITS.CLAMP.mass[0],
            cfg.TRAITS.CLAMP.mass[1],
        )
        hp_ratio_by_slot = torch.clamp(
            self.registry.data[self.registry.HP]
            / self.registry.data[self.registry.HP_MAX].clamp_min(1e-6),
            0.0,
            1.0,
        )

        for step in range(1, max_vision + 1):
            step_positions = positions.unsqueeze(1) + ray_dirs * step
            step_x = step_positions[..., 0].round().clamp(0, grid_w - 1).long()
            step_y = step_positions[..., 1].round().clamp(0, grid_h - 1).long()

            active = step <= vision_ranges.unsqueeze(1)
            if not active.any():
                continue

            occ_vals = occ_grid[step_y, step_x]
            agent_vals = agent_grid[step_y, step_x].long()
            h_rate_vals = h_rate_grid[step_y, step_x]

            path_update_mask = active & (torch.abs(h_rate_vals) > torch.abs(path_zone_peak_raw))
            path_zone_peak_raw = torch.where(path_update_mask, h_rate_vals, path_zone_peak_raw)
            terminal_zone_rate_raw = torch.where(active, h_rate_vals, terminal_zone_rate_raw)

            unresolved = (results[..., 0:3].sum(dim=-1) == 0.0) & active
            if not unresolved.any():
                continue

            norm_dist = torch.full((n_agents, self.num_rays), float(step), device=device) / safe_vision_ranges.unsqueeze(1)

            wall_hit = unresolved & (occ_vals > 0.5)
            if wall_hit.any():
                results[..., 2][wall_hit] = 1.0
                results[..., 3][wall_hit] = norm_dist[wall_hit]
                results[..., 5][wall_hit] = h_rate_vals[wall_hit]

            self_hits = agent_vals == self_indices.unsqueeze(1)
            agent_hit = unresolved & (agent_vals >= 0) & ~self_hits
            if agent_hit.any():
                hit_slots = agent_vals[agent_hit]
                results[..., 1][agent_hit] = 1.0
                results[..., 3][agent_hit] = norm_dist[agent_hit]
                results[..., 5][agent_hit] = h_rate_vals[agent_hit]
                results[..., 6][agent_hit] = mass_norm_by_slot[hit_slots]
                results[..., 7][agent_hit] = hp_ratio_by_slot[hit_slots]

        has_support = vision_ranges.unsqueeze(1) > 0.0
        no_hit = (results[..., 0:3].sum(dim=-1) == 0.0) & has_support
        if no_hit.any():
            results[..., 0][no_hit] = 1.0
            results[..., 3][no_hit] = 1.0
            results[..., 5][no_hit] = terminal_zone_rate_raw[no_hit]

        results[..., 4] = normalize_signed(path_zone_peak_raw, cfg.PERCEPT.ZONE_RATE_ABS_MAX)
        results[..., 5] = normalize_signed(results[..., 5], cfg.PERCEPT.ZONE_RATE_ABS_MAX)
        results[..., 6] = torch.where(results[..., 1] > 0.5, results[..., 6], torch.zeros_like(results[..., 6]))
        results[..., 7] = torch.where(results[..., 1] > 0.5, results[..., 7], torch.zeros_like(results[..., 7]))
        return results

    def build_observations(self, alive_indices: torch.Tensor) -> dict:
        if len(alive_indices) == 0:
            return build_empty_observation_batch(self.grid.device, self.num_rays)

        positions = self.registry.data[[self.registry.X, self.registry.Y], :][:, alive_indices].T
        effective_vision_ranges = self.get_effective_vision_values(alive_indices)
        canonical_rays = self.cast_rays_batched(positions, effective_vision_ranges, alive_indices)

        obs = build_observation_bundle(
            registry=self.registry,
            grid=self.grid,
            alive_indices=alive_indices,
            canonical_rays=canonical_rays,
        )

        # Prompt 6 fog must affect the intended perception pathway without
        # mutating the underlying inherited vision trait in registry storage.
        vision_norm = normalize_from_bounds(
            effective_vision_ranges,
            cfg.TRAITS.CLAMP.vision[0],
            cfg.TRAITS.CLAMP.vision[1],
        )
        obs["canonical_self"][:, 4] = vision_norm
        return obs
