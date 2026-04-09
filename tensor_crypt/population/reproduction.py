"""Binary reproduction, mutation, and offspring-placement helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from ..agents.brain import get_bloodline_families
from ..config_bridge import cfg


class ParentSelectionError(RuntimeError):
    """Raised when overlay-constrained parent selection blocks one birth slot."""

    def __init__(
        self,
        failure_reason: str,
        *,
        local_parenting_enabled: bool = False,
        local_parenting_used_global_fallback: bool = False,
        local_parent_candidate_count: int = 0,
        cooldown_relaxed_brain: bool = False,
        cooldown_relaxed_trait: bool = False,
        cooldown_relaxed_anchor: bool = False,
    ) -> None:
        super().__init__(failure_reason)
        self.failure_reason = str(failure_reason)
        self.local_parenting_enabled = bool(local_parenting_enabled)
        self.local_parenting_used_global_fallback = bool(local_parenting_used_global_fallback)
        self.local_parent_candidate_count = int(local_parent_candidate_count)
        self.cooldown_relaxed_brain = bool(cooldown_relaxed_brain)
        self.cooldown_relaxed_trait = bool(cooldown_relaxed_trait)
        self.cooldown_relaxed_anchor = bool(cooldown_relaxed_anchor)


@dataclass(frozen=True)
class ParentRoles:
    brain_parent_uid: int
    trait_parent_uid: int
    anchor_parent_uid: int
    floor_recovery: bool
    local_parenting_enabled: bool = False
    local_parenting_used_global_fallback: bool = False
    local_parent_candidate_count: int = 0
    cooldown_relaxed_brain: bool = False
    cooldown_relaxed_trait: bool = False
    cooldown_relaxed_anchor: bool = False


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
    crowding_checked: bool = False
    crowding_neighbor_count: int = 0
    crowding_policy_applied: str | None = None


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
    logits = torch.tensor(
        [
            float(latent["z_hp"]),
            float(latent["z_mass"]),
            float(latent["z_vision"]),
            float(latent["z_metab"]),
        ],
        dtype=torch.float32,
    )
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


def mutate_trait_latent(
    parent_latent: dict[str, float],
    mutation_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, float], BirthMutationFlags]:
    child = dict(parent_latent)
    mutation_overrides = mutation_overrides or {}

    rare_prob_scale = float(mutation_overrides.get("rare_prob_scalar", 1.0))
    rare = random.random() < float(cfg.EVOL.RARE_MUT_PROB) * rare_prob_scale

    logit_sigma_scalar = float(mutation_overrides.get("trait_sigma_scalar", 1.0))
    budget_sigma_scalar = float(mutation_overrides.get("budget_sigma_scalar", 1.0))

    logit_sigma = (
        cfg.EVOL.RARE_TRAIT_LOGIT_MUTATION_SIGMA if rare else cfg.EVOL.TRAIT_LOGIT_MUTATION_SIGMA
    ) * logit_sigma_scalar
    budget_sigma = (
        cfg.EVOL.RARE_TRAIT_BUDGET_MUTATION_SIGMA if rare else cfg.EVOL.TRAIT_BUDGET_MUTATION_SIGMA
    ) * budget_sigma_scalar

    for key in ("z_hp", "z_mass", "z_vision", "z_metab"):
        child[key] = float(child[key]) + random.gauss(0.0, float(logit_sigma))

    child["budget"] = max(
        cfg.TRAITS.BUDGET.MIN_BUDGET,
        min(cfg.TRAITS.BUDGET.MAX_BUDGET, float(child["budget"]) + random.gauss(0.0, float(budget_sigma))),
    )

    family_shift_prob_scale = float(mutation_overrides.get("family_shift_scalar", 1.0))
    family_shift = bool(
        cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION
        and random.random() < float(cfg.EVOL.FAMILY_SHIFT_PROB) * family_shift_prob_scale
    )
    return child, BirthMutationFlags(rare_mutation=rare, family_shift=family_shift)


def policy_noise_sigma(
    flags: BirthMutationFlags,
    mutation_overrides: dict[str, float] | None = None,
) -> float:
    mutation_overrides = mutation_overrides or {}
    scalar = float(mutation_overrides.get("policy_noise_scalar", 1.0))
    return float((cfg.EVOL.RARE_POLICY_NOISE_SD if flags.rare_mutation else cfg.EVOL.POLICY_NOISE_SD) * scalar)


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


def _alive_slots_within_radius(registry, center_x: int, center_y: int, radius: int) -> list[int]:
    radius = max(0, int(radius))
    alive_slots = [int(idx) for idx in registry.get_alive_indices().tolist()]
    candidates = []
    for slot_idx in alive_slots:
        x = int(registry.data[registry.X, slot_idx].item())
        y = int(registry.data[registry.Y, slot_idx].item())
        if max(abs(x - center_x), abs(y - center_y)) <= radius:
            candidates.append(slot_idx)
    return candidates


def _count_anchor_neighbors(registry, anchor_slot: int, radius: int) -> int:
    radius = max(0, int(radius))
    ax = int(registry.data[registry.X, anchor_slot].item())
    ay = int(registry.data[registry.Y, anchor_slot].item())
    neighbor_count = 0
    for slot_idx in registry.get_alive_indices().tolist():
        slot_idx = int(slot_idx)
        if slot_idx == anchor_slot:
            continue
        x = int(registry.data[registry.X, slot_idx].item())
        y = int(registry.data[registry.Y, slot_idx].item())
        if max(abs(x - ax), abs(y - ay)) <= radius:
            neighbor_count += 1
    return neighbor_count


def _select_ranked_candidate(
    registry,
    candidates: list[int],
    *,
    role_name: str,
    floor_recovery: bool,
    tick: int | None,
    respawn_controller,
    key_fn,
    extra_roles: tuple[str, ...] = (),
) -> tuple[int, set[str]]:
    ordered = sorted(candidates, key=key_fn)
    if not ordered:
        raise ParentSelectionError(f"{role_name}_candidate_pool_empty")
    if respawn_controller is None or tick is None:
        return ordered[0], set()

    active_roles: dict[str, dict] = {}
    for role in (role_name, *extra_roles):
        runtime = respawn_controller.get_cooldown_role_runtime(role, floor_recovery=floor_recovery)
        if runtime["active"]:
            active_roles[role] = runtime
    if not active_roles:
        return ordered[0], set()

    eligible_slots = []
    for slot_idx in ordered:
        uid = registry.get_uid_for_slot(slot_idx)
        if uid == -1:
            continue
        cooled = False
        for role, runtime in active_roles.items():
            if respawn_controller.uid_on_cooldown(uid, role, tick, runtime=runtime):
                cooled = True
                break
        if not cooled:
            eligible_slots.append(slot_idx)
    if eligible_slots:
        return eligible_slots[0], set()

    if any(runtime["empty_pool_policy"] == "strict" for runtime in active_roles.values()):
        raise ParentSelectionError(f"cooldown_{role_name}_blocked")
    return ordered[0], set(active_roles.keys())


def _ordered_anchor_candidates(mode: str, registry, brain_slot: int, trait_slot: int) -> list[int]:
    if mode == "brain_parent":
        return [brain_slot]
    if mode == "trait_parent":
        return [trait_slot]
    if mode == "random_parent":
        if brain_slot == trait_slot:
            return [brain_slot]
        return [brain_slot, trait_slot]
    if mode == "fitter_of_two":
        unique = list(dict.fromkeys([brain_slot, trait_slot]))
        return sorted(
            unique,
            key=lambda slot: (-float(registry.fitness[slot].item()), registry.get_uid_for_slot(slot)),
        )
    raise ValueError(f"Unsupported anchor parent selector: {mode}")


def _choose_anchor_slot(mode: str, candidates: list[int]) -> int:
    if len(candidates) == 1:
        return candidates[0]
    if mode == "random_parent":
        return random.choice(candidates)
    return candidates[0]


def _select_anchor_slot(
    registry,
    *,
    brain_slot: int,
    trait_slot: int,
    floor_recovery: bool,
    tick: int | None,
    respawn_controller,
) -> tuple[int, set[str]]:
    mode = str(cfg.RESPAWN.ANCHOR_PARENT_SELECTOR)
    candidates = _ordered_anchor_candidates(mode, registry, brain_slot, trait_slot)
    if respawn_controller is None or tick is None:
        return _choose_anchor_slot(mode, candidates), set()

    runtime = respawn_controller.get_cooldown_role_runtime("anchor_parent", floor_recovery=floor_recovery)
    if not runtime["active"]:
        return _choose_anchor_slot(mode, candidates), set()

    eligible = []
    for slot_idx in candidates:
        uid = registry.get_uid_for_slot(slot_idx)
        if uid == -1:
            continue
        if not respawn_controller.uid_on_cooldown(uid, "anchor_parent", tick, runtime=runtime):
            eligible.append(slot_idx)
    if eligible:
        return _choose_anchor_slot(mode, eligible), set()
    if runtime["empty_pool_policy"] == "strict":
        raise ParentSelectionError("cooldown_anchor_parent_blocked")
    return _choose_anchor_slot(mode, candidates), {"anchor_parent"}


def select_parent_roles(
    registry,
    *,
    floor_recovery: bool,
    dead_slot: int | None = None,
    respawn_controller=None,
    tick: int | None = None,
) -> ParentRoles:
    alive_slots = [int(idx) for idx in registry.get_alive_indices().tolist()]
    if len(alive_slots) < 2:
        raise RuntimeError("Binary reproduction requires at least two live agents")

    local_parenting_enabled = False
    local_parenting_used_global_fallback = False
    local_parent_candidate_count = 0
    candidate_pool = list(alive_slots)

    if respawn_controller is not None:
        local_runtime = respawn_controller.get_local_parent_runtime(floor_recovery=floor_recovery)
        if local_runtime["active"]:
            if dead_slot is None:
                raise ParentSelectionError("local_parent_missing_dead_slot")
            center_x = int(registry.data[registry.X, dead_slot].item())
            center_y = int(registry.data[registry.Y, dead_slot].item())
            local_slots = _alive_slots_within_radius(registry, center_x, center_y, local_runtime["selection_radius"])
            local_parent_candidate_count = len(local_slots)
            local_parenting_enabled = True
            if local_slots:
                candidate_pool = local_slots
            elif local_runtime["fallback_behavior"] == "strict":
                raise ParentSelectionError(
                    "local_parent_candidates_empty",
                    local_parenting_enabled=True,
                    local_parent_candidate_count=0,
                )
            else:
                local_parenting_used_global_fallback = True

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

    brain_candidates = [slot for slot in candidate_pool if eligible_brain(slot)] or list(candidate_pool)
    trait_candidates = [slot for slot in candidate_pool if eligible_trait(slot)] or list(candidate_pool)

    selector_mode = str(cfg.RESPAWN.ANCHOR_PARENT_SELECTOR)
    brain_extra_roles = ("anchor_parent",) if selector_mode == "brain_parent" else ()
    trait_extra_roles = ("anchor_parent",) if selector_mode == "trait_parent" else ()

    brain_slot, brain_relaxed_roles = _select_ranked_candidate(
        registry,
        brain_candidates,
        role_name="brain_parent",
        floor_recovery=floor_recovery,
        tick=tick,
        respawn_controller=respawn_controller,
        key_fn=lambda slot: (-float(registry.fitness[slot].item()), registry.get_uid_for_slot(slot)),
        extra_roles=brain_extra_roles,
    )
    trait_slot, trait_relaxed_roles = _select_ranked_candidate(
        registry,
        trait_candidates,
        role_name="trait_parent",
        floor_recovery=floor_recovery,
        tick=tick,
        respawn_controller=respawn_controller,
        key_fn=lambda slot: (
            -(
                _hp_ratio(registry, slot)
                + 0.25 * _hp_max_norm(registry, slot)
                + 0.10 * _age_norm(registry, slot)
            ),
            registry.get_uid_for_slot(slot),
        ),
        extra_roles=trait_extra_roles,
    )

    if selector_mode == "brain_parent":
        anchor_slot = brain_slot
        anchor_relaxed_roles = {"anchor_parent"} if "anchor_parent" in brain_relaxed_roles else set()
    elif selector_mode == "trait_parent":
        anchor_slot = trait_slot
        anchor_relaxed_roles = {"anchor_parent"} if "anchor_parent" in trait_relaxed_roles else set()
    else:
        anchor_slot, anchor_relaxed_roles = _select_anchor_slot(
            registry,
            brain_slot=brain_slot,
            trait_slot=trait_slot,
            floor_recovery=floor_recovery,
            tick=tick,
            respawn_controller=respawn_controller,
        )

    return ParentRoles(
        brain_parent_uid=registry.get_uid_for_slot(brain_slot),
        trait_parent_uid=registry.get_uid_for_slot(trait_slot),
        anchor_parent_uid=registry.get_uid_for_slot(anchor_slot),
        floor_recovery=bool(floor_recovery),
        local_parenting_enabled=local_parenting_enabled,
        local_parenting_used_global_fallback=local_parenting_used_global_fallback,
        local_parent_candidate_count=local_parent_candidate_count,
        cooldown_relaxed_brain="brain_parent" in brain_relaxed_roles,
        cooldown_relaxed_trait="trait_parent" in trait_relaxed_roles,
        cooldown_relaxed_anchor=(
            "anchor_parent" in brain_relaxed_roles
            or "anchor_parent" in trait_relaxed_roles
            or "anchor_parent" in anchor_relaxed_roles
        ),
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
    # Placement expands in shuffled square rings so nearby cells are preferred
    # without introducing a directional bias across births.
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


def _attempt_global_fallback(
    grid,
    evolution,
    *,
    attempts: int,
    failure_reason: str,
    crowding_checked: bool,
    crowding_neighbor_count: int,
    crowding_policy_applied: str | None,
) -> PlacementResult:
    if not cfg.RESPAWN.ALLOW_FALLBACK_GLOBAL_PLACEMENT:
        return PlacementResult(
            x=None,
            y=None,
            attempts=attempts,
            used_global_fallback=False,
            failure_reason=failure_reason,
            crowding_checked=crowding_checked,
            crowding_neighbor_count=crowding_neighbor_count,
            crowding_policy_applied=crowding_policy_applied,
        )

    gx, gy = evolution.find_free_cell(grid)
    if gx is None:
        return PlacementResult(
            x=None,
            y=None,
            attempts=attempts,
            used_global_fallback=True,
            failure_reason="global_placement_failed",
            crowding_checked=crowding_checked,
            crowding_neighbor_count=crowding_neighbor_count,
            crowding_policy_applied=crowding_policy_applied,
        )
    return PlacementResult(
        x=gx,
        y=gy,
        attempts=attempts,
        used_global_fallback=True,
        failure_reason=None,
        crowding_checked=crowding_checked,
        crowding_neighbor_count=crowding_neighbor_count,
        crowding_policy_applied=crowding_policy_applied,
    )


def place_offspring_near_anchor(
    registry,
    grid,
    anchor_slot: int,
    evolution,
    *,
    floor_recovery: bool = False,
    respawn_controller=None,
) -> PlacementResult:
    """Resolve a spawn cell near the anchor parent or report why it failed."""
    ax = int(registry.data[registry.X, anchor_slot].item())
    ay = int(registry.data[registry.Y, anchor_slot].item())
    attempts = 0
    crowding_checked = False
    crowding_neighbor_count = 0
    crowding_policy_applied = None

    crowding_runtime = None
    if respawn_controller is not None:
        crowding_runtime = respawn_controller.get_crowding_runtime(floor_recovery=floor_recovery)
    if crowding_runtime and crowding_runtime["active"]:
        crowding_checked = True
        crowding_neighbor_count = _count_anchor_neighbors(
            registry,
            anchor_slot,
            crowding_runtime["local_radius"],
        )
        if crowding_neighbor_count >= crowding_runtime["max_neighbors"]:
            crowding_policy_applied = crowding_runtime["policy_when_crowded"]
            if crowding_policy_applied == "block_birth":
                return PlacementResult(
                    x=None,
                    y=None,
                    attempts=attempts,
                    used_global_fallback=False,
                    failure_reason="crowding_blocked",
                    crowding_checked=True,
                    crowding_neighbor_count=crowding_neighbor_count,
                    crowding_policy_applied=crowding_policy_applied,
                )
            return _attempt_global_fallback(
                grid,
                evolution,
                attempts=attempts,
                failure_reason="crowding_global_fallback_disabled",
                crowding_checked=True,
                crowding_neighbor_count=crowding_neighbor_count,
                crowding_policy_applied=crowding_policy_applied,
            )

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
        return PlacementResult(
            x=x,
            y=y,
            attempts=attempts,
            used_global_fallback=False,
            failure_reason=None,
            crowding_checked=crowding_checked,
            crowding_neighbor_count=crowding_neighbor_count,
            crowding_policy_applied=crowding_policy_applied,
        )

    return _attempt_global_fallback(
        grid,
        evolution,
        attempts=attempts,
        failure_reason="local_placement_failed",
        crowding_checked=crowding_checked,
        crowding_neighbor_count=crowding_neighbor_count,
        crowding_policy_applied=crowding_policy_applied,
    )
