from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from ..config_bridge import cfg
from ..agents.brain import get_bloodline_families


@dataclass(frozen=True)
class ParentRoles:
    brain_parent_uid: int
    trait_parent_uid: int
    anchor_parent_uid: int
    floor_recovery: bool


@dataclass(frozen=True)
class BirthMutationFlags:
    rare_mutation: bool
    family_shift: bool


@dataclass(frozen=True)
class PlacementResult:
    x: int | None
    y: int | None
    attempts: int
    used_global_fallback: bool
    failure_reason: str | None


def default_trait_latent() -> dict[str, float]:
    logits = list(cfg.TRAITS.BUDGET.INIT_LOGITS)
    return {
        "budget": float(cfg.TRAITS.BUDGET.INIT_BUDGET),
        "z_hp": float(logits[0]),
        "z_mass": float(logits[1]),
        "z_vision": float(logits[2]),
        "z_metab": float(logits[3]),
    }


def _lerp(lower: float, upper: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return float(lower) + (float(upper) - float(lower)) * alpha


def trait_values_from_latent(latent: dict[str, float]) -> dict[str, float]:
    budget = max(cfg.TRAITS.BUDGET.MIN_BUDGET, min(cfg.TRAITS.BUDGET.MAX_BUDGET, float(latent["budget"])))
    logits = torch.tensor([
        float(latent["z_hp"]),
        float(latent["z_mass"]),
        float(latent["z_vision"]),
        float(latent["z_metab"]),
    ], dtype=torch.float32)
    alloc = F.softmax(logits, dim=0)

    hp_alpha = min(1.0, budget * float(alloc[0]))
    mass_alpha = min(1.0, budget * float(alloc[1]))
    vision_alpha = min(1.0, budget * float(alloc[2]))
    metab_alpha = min(1.0, budget * float(alloc[3]))

    clamps = cfg.TRAITS.CLAMP
    hp_max = _lerp(clamps.hp_max[0], clamps.hp_max[1], hp_alpha)
    mass = _lerp(clamps.mass[0], clamps.mass[1], mass_alpha)
    vision = _lerp(clamps.vision[0], clamps.vision[1], vision_alpha)
    metab_base = _lerp(clamps.metab[0], clamps.metab[1], metab_alpha)
    metab = metab_base + cfg.TRAITS.METAB_COEFFS["base"] + mass * cfg.TRAITS.METAB_COEFFS["per_mass"] + vision * cfg.TRAITS.METAB_COEFFS["per_vision"]
    metab = max(clamps.metab[0], min(clamps.metab[1], metab))

    return {
        "hp_max": hp_max,
        "mass": mass,
        "vision": vision,
        "metab": metab,
        "alloc_hp": float(alloc[0]),
        "alloc_mass": float(alloc[1]),
        "alloc_vision": float(alloc[2]),
        "alloc_metab": float(alloc[3]),
        "budget": budget,
    }


def mutate_trait_latent(parent_latent: dict[str, float]) -> tuple[dict[str, float], BirthMutationFlags]:
    child = dict(parent_latent)
    rare = random.random() < float(cfg.EVOL.RARE_MUT_PROB)

    logit_sigma = cfg.EVOL.RARE_TRAIT_LOGIT_MUTATION_SIGMA if rare else cfg.EVOL.TRAIT_LOGIT_MUTATION_SIGMA
    budget_sigma = cfg.EVOL.RARE_TRAIT_BUDGET_MUTATION_SIGMA if rare else cfg.EVOL.TRAIT_BUDGET_MUTATION_SIGMA

    for key in ("z_hp", "z_mass", "z_vision", "z_metab"):
        child[key] = float(child[key]) + random.gauss(0.0, float(logit_sigma))

    child["budget"] = max(
        cfg.TRAITS.BUDGET.MIN_BUDGET,
        min(cfg.TRAITS.BUDGET.MAX_BUDGET, float(child["budget"]) + random.gauss(0.0, float(budget_sigma))),
    )

    family_shift = bool(cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION and random.random() < float(cfg.EVOL.FAMILY_SHIFT_PROB))
    return child, BirthMutationFlags(rare_mutation=rare, family_shift=family_shift)


def policy_noise_sigma(flags: BirthMutationFlags) -> float:
    return float(cfg.EVOL.RARE_POLICY_NOISE_SD if flags.rare_mutation else cfg.EVOL.POLICY_NOISE_SD)


def pick_shifted_family(current_family: str) -> str:
    families = [family for family in get_bloodline_families() if family != current_family]
    if not families:
        return current_family
    return random.choice(families)


def _age_norm(registry, slot_idx: int) -> float:
    age = max(0.0, float(registry.tick_counter) - float(registry.data[registry.TICK_BORN, slot_idx].item()))
    denom = max(1.0, float(cfg.PERCEPT.AGE_NORM_TICKS))
    return min(1.0, age / denom)


def _hp_ratio(registry, slot_idx: int) -> float:
    hp = float(registry.data[registry.HP, slot_idx].item())
    hp_max = max(1e-6, float(registry.data[registry.HP_MAX, slot_idx].item()))
    return max(0.0, min(1.0, hp / hp_max))


def _hp_max_norm(registry, slot_idx: int) -> float:
    hp_max = float(registry.data[registry.HP_MAX, slot_idx].item())
    lower, upper = cfg.TRAITS.CLAMP.hp_max
    return max(0.0, min(1.0, (hp_max - lower) / max(1e-6, upper - lower)))


def select_parent_roles(registry, *, floor_recovery: bool) -> ParentRoles:
    alive_slots = [int(idx) for idx in registry.get_alive_indices().tolist()]
    if len(alive_slots) < 2:
        raise RuntimeError("Binary reproduction requires at least two live agents")

    def eligible_brain(slot_idx: int) -> bool:
        if floor_recovery and cfg.RESPAWN.FLOOR_RECOVERY_SUSPEND_THRESHOLDS:
            return True
        return float(registry.fitness[slot_idx].item()) >= float(cfg.RESPAWN.BRAIN_PARENT_MIN_FITNESS)

    def eligible_trait(slot_idx: int) -> bool:
        if floor_recovery and cfg.RESPAWN.FLOOR_RECOVERY_SUSPEND_THRESHOLDS:
            return True
        if _hp_ratio(registry, slot_idx) < float(cfg.RESPAWN.TRAIT_PARENT_MIN_HP_RATIO):
            return False
        age_ticks = max(0.0, float(registry.tick_counter) - float(registry.data[registry.TICK_BORN, slot_idx].item()))
        return age_ticks >= float(cfg.RESPAWN.TRAIT_PARENT_MIN_AGE_TICKS)

    brain_candidates = [slot for slot in alive_slots if eligible_brain(slot)] or alive_slots
    trait_candidates = [slot for slot in alive_slots if eligible_trait(slot)] or alive_slots

    brain_slot = sorted(
        brain_candidates,
        key=lambda slot: (-float(registry.fitness[slot].item()), registry.get_uid_for_slot(slot)),
    )[0]

    trait_slot = sorted(
        trait_candidates,
        key=lambda slot: (-(_hp_ratio(registry, slot) + 0.25 * _hp_max_norm(registry, slot) + 0.10 * _age_norm(registry, slot)), registry.get_uid_for_slot(slot)),
    )[0]

    mode = str(cfg.RESPAWN.ANCHOR_PARENT_SELECTOR)
    if mode == "brain_parent":
        anchor_slot = brain_slot
    elif mode == "trait_parent":
        anchor_slot = trait_slot
    elif mode == "random_parent":
        anchor_slot = random.choice([brain_slot, trait_slot])
    elif mode == "fitter_of_two":
        anchor_slot = brain_slot if float(registry.fitness[brain_slot].item()) >= float(registry.fitness[trait_slot].item()) else trait_slot
    else:
        raise ValueError(f"Unsupported anchor parent selector: {mode}")

    return ParentRoles(
        brain_parent_uid=registry.get_uid_for_slot(brain_slot),
        trait_parent_uid=registry.get_uid_for_slot(trait_slot),
        anchor_parent_uid=registry.get_uid_for_slot(anchor_slot),
        floor_recovery=bool(floor_recovery),
    )


def birth_hp_value(traits: dict[str, float]) -> float:
    mode = str(cfg.RESPAWN.BIRTH_HP_MODE)
    if mode == "full":
        return float(traits["hp_max"])
    if mode == "fraction":
        frac = max(0.0, min(1.0, float(cfg.RESPAWN.BIRTH_HP_FRACTION)))
        return float(traits["hp_max"]) * frac
    raise ValueError(f"Unsupported birth HP mode: {mode}")


def _placement_candidates(ax: int, ay: int) -> Iterable[tuple[int, int]]:
    r_min = max(0, int(cfg.RESPAWN.OFFSPRING_JITTER_RADIUS_MIN))
    r_max = max(r_min, int(cfg.RESPAWN.OFFSPRING_JITTER_RADIUS_MAX))
    points: list[tuple[int, int]] = []
    for radius in range(r_min, r_max + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if max(abs(dx), abs(dy)) != radius:
                    continue
                points.append((ax + dx, ay + dy))
    random.shuffle(points)
    return points


def place_offspring_near_anchor(registry, grid, anchor_slot: int, evolution) -> PlacementResult:
    ax = int(registry.data[registry.X, anchor_slot].item())
    ay = int(registry.data[registry.Y, anchor_slot].item())
    attempts = 0

    for x, y in _placement_candidates(ax, ay):
        attempts += 1
        if attempts > int(cfg.RESPAWN.OFFSPRING_MAX_PLACEMENT_ATTEMPTS):
            break
        if grid.is_wall(x, y) and cfg.RESPAWN.DISALLOW_SPAWN_ON_WALL:
            continue
        if grid.get_agent_at(x, y) != -1 and cfg.RESPAWN.DISALLOW_SPAWN_ON_OCCUPIED:
            continue
        if cfg.RESPAWN.DISALLOW_SPAWN_IN_HARM_ZONE and grid.get_h_rate(x, y) < 0.0:
            continue
        return PlacementResult(x=x, y=y, attempts=attempts, used_global_fallback=False, failure_reason=None)

    if not cfg.RESPAWN.ALLOW_FALLBACK_GLOBAL_PLACEMENT:
        return PlacementResult(x=None, y=None, attempts=attempts, used_global_fallback=False, failure_reason="local_placement_failed")

    gx, gy = evolution._find_free_cell(grid)
    if gx is None:
        return PlacementResult(x=None, y=None, attempts=attempts, used_global_fallback=True, failure_reason="global_placement_failed")
    return PlacementResult(x=gx, y=gy, attempts=attempts, used_global_fallback=True, failure_reason=None)
