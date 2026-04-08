"""Canonical observation schema helpers for Prompt 2."""

from __future__ import annotations

import torch

from ..config_bridge import cfg


def normalize_from_bounds(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    lower_f = float(lower)
    upper_f = float(upper)
    denom = max(upper_f - lower_f, 1e-6)
    return torch.clamp((values - lower_f) / denom, 0.0, 1.0)


def normalize_signed(values: torch.Tensor, max_abs: float) -> torch.Tensor:
    denom = max(float(max_abs), 1e-6)
    return torch.clamp(values / denom, -1.0, 1.0)


def distance_to_center_norm(positions: torch.Tensor, grid_w: int, grid_h: int) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.empty(0, device=positions.device, dtype=positions.dtype)

    center = torch.tensor(
        [(grid_w - 1) / 2.0, (grid_h - 1) / 2.0],
        device=positions.device,
        dtype=positions.dtype,
    )
    distances = torch.linalg.norm(positions - center, dim=1)
    max_distance = torch.linalg.norm(center).clamp_min(1e-6)
    return torch.clamp(distances / max_distance, 0.0, 1.0)


def build_empty_observation_batch(device: str, num_rays: int) -> dict:
    obs = {
        "rays": torch.empty(0, num_rays, cfg.PERCEPT.LEGACY_RAY_FEATURES, device=device),
        "state": torch.empty(0, cfg.PERCEPT.LEGACY_STATE_FEATURES, device=device),
        "genome": torch.empty(0, cfg.PERCEPT.LEGACY_GENOME_FEATURES, device=device),
        "position": torch.empty(0, cfg.PERCEPT.LEGACY_POSITION_FEATURES, device=device),
        "context": torch.empty(0, cfg.PERCEPT.LEGACY_CONTEXT_FEATURES, device=device),
    }
    if cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS:
        obs["canonical_rays"] = torch.empty(0, num_rays, cfg.PERCEPT.CANONICAL_RAY_FEATURES, device=device)
        obs["canonical_self"] = torch.empty(0, cfg.PERCEPT.CANONICAL_SELF_FEATURES, device=device)
        obs["canonical_context"] = torch.empty(0, cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES, device=device)
    return obs


def adapt_canonical_to_legacy(
    canonical_rays: torch.Tensor,
    canonical_self: torch.Tensor,
    canonical_context: torch.Tensor,
) -> dict:
    rays = torch.stack(
        [
            canonical_rays[..., 1],  # hit_agent
            canonical_rays[..., 2],  # hit_wall
            canonical_rays[..., 3],  # distance_norm
            canonical_rays[..., 6],  # target_mass_norm
            canonical_rays[..., 4],  # path_zone_peak_rate_norm
        ],
        dim=-1,
    )

    state = torch.stack(
        [
            canonical_self[:, 0],   # hp_ratio
            canonical_self[:, 10],  # current_zone_rate_norm
        ],
        dim=1,
    )

    genome = canonical_self[:, 2:6]
    position = canonical_self[:, 6:8]
    context = torch.stack(
        [
            canonical_self[:, 8],   # distance_to_center_norm
            canonical_self[:, 9],   # age_norm
            canonical_context[:, 0],  # alive_fraction
        ],
        dim=1,
    )

    return {
        "rays": rays,
        "state": state,
        "genome": genome,
        "position": position,
        "context": context,
    }


def build_observation_bundle(
    *,
    registry,
    grid,
    alive_indices: torch.Tensor,
    canonical_rays: torch.Tensor,
    vision_norm_override: torch.Tensor | None = None,
    global_alive_indices: torch.Tensor | None = None,
) -> dict:
    if len(alive_indices) == 0:
        return build_empty_observation_batch(grid.device, cfg.PERCEPT.NUM_RAYS)

    device = grid.device
    positions = registry.data[[registry.X, registry.Y], :][:, alive_indices].T
    hp = registry.data[registry.HP, alive_indices]
    hp_max = registry.data[registry.HP_MAX, alive_indices]
    hp_ratio = torch.clamp(hp / hp_max.clamp_min(1e-6), 0.0, 1.0)
    hp_deficit_ratio = 1.0 - hp_ratio

    mass_norm = normalize_from_bounds(
        registry.data[registry.MASS, alive_indices],
        cfg.TRAITS.CLAMP.mass[0],
        cfg.TRAITS.CLAMP.mass[1],
    )
    hp_max_norm = normalize_from_bounds(
        registry.data[registry.HP_MAX, alive_indices],
        cfg.TRAITS.CLAMP.hp_max[0],
        cfg.TRAITS.CLAMP.hp_max[1],
    )
    vision_norm = vision_norm_override
    if vision_norm is None:
        vision_norm = normalize_from_bounds(
            registry.data[registry.VISION, alive_indices],
            cfg.TRAITS.CLAMP.vision[0],
            cfg.TRAITS.CLAMP.vision[1],
        )
    metabolism_norm = normalize_from_bounds(
        registry.data[registry.METABOLISM_RATE, alive_indices],
        cfg.TRAITS.CLAMP.metab[0],
        cfg.TRAITS.CLAMP.metab[1],
    )

    x_norm = torch.clamp(positions[:, 0] / max(grid.W - 1, 1), 0.0, 1.0)
    y_norm = torch.clamp(positions[:, 1] / max(grid.H - 1, 1), 0.0, 1.0)
    dist_center_norm = distance_to_center_norm(positions, grid.W, grid.H)

    tick_born = registry.data[registry.TICK_BORN, alive_indices]
    age_ticks = torch.clamp(torch.tensor(float(registry.tick_counter), device=device) - tick_born, min=0.0)
    age_norm = torch.clamp(age_ticks / max(float(cfg.PERCEPT.AGE_NORM_TICKS), 1.0), 0.0, 1.0)

    x_int = positions[:, 0].long().clamp(0, grid.W - 1)
    y_int = positions[:, 1].long().clamp(0, grid.H - 1)
    current_zone_rate_norm = normalize_signed(
        grid.grid[1, y_int, x_int],
        cfg.PERCEPT.ZONE_RATE_ABS_MAX,
    )

    canonical_self = torch.stack(
        [
            hp_ratio,
            hp_deficit_ratio,
            mass_norm,
            hp_max_norm,
            vision_norm,
            metabolism_norm,
            x_norm,
            y_norm,
            dist_center_norm,
            age_norm,
            current_zone_rate_norm,
        ],
        dim=1,
    )

    global_alive = alive_indices if global_alive_indices is None else global_alive_indices
    if len(global_alive) == 0:
        alive_fraction = hp_ratio.new_zeros(len(alive_indices))
        mean_mass_norm = hp_ratio.new_zeros(len(alive_indices))
        mean_hp_ratio = hp_ratio.new_zeros(len(alive_indices))
    else:
        global_hp = registry.data[registry.HP, global_alive]
        global_hp_max = registry.data[registry.HP_MAX, global_alive]
        global_hp_ratio = torch.clamp(global_hp / global_hp_max.clamp_min(1e-6), 0.0, 1.0)
        global_mass_norm = normalize_from_bounds(
            registry.data[registry.MASS, global_alive],
            cfg.TRAITS.CLAMP.mass[0],
            cfg.TRAITS.CLAMP.mass[1],
        )
        alive_fraction = torch.full(
            (len(alive_indices),),
            float(len(global_alive)) / max(float(cfg.AGENTS.N), 1.0),
            device=device,
        )
        mean_mass_norm = global_mass_norm.mean().expand(len(alive_indices))
        mean_hp_ratio = global_hp_ratio.mean().expand(len(alive_indices))

    canonical_context = torch.stack(
        [
            alive_fraction,
            mean_mass_norm,
            mean_hp_ratio,
        ],
        dim=1,
    )

    obs = adapt_canonical_to_legacy(canonical_rays, canonical_self, canonical_context)
    if cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS:
        obs["canonical_rays"] = canonical_rays
        obs["canonical_self"] = canonical_self
        obs["canonical_context"] = canonical_context
    return obs

