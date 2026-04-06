from __future__ import annotations

from typing import Optional

import torch

from ..config_bridge import cfg
from ..telemetry.data_logger import DataLogger
from .reproduction import (
    birth_hp_value,
    default_trait_latent,
    mutate_trait_latent,
    pick_shifted_family,
    place_offspring_near_anchor,
    policy_noise_sigma,
    select_parent_roles,
    trait_values_from_latent,
)


class RespawnController:
    """Prompt 5 binary reproduction controller with Prompt 6 catastrophe gates."""

    def __init__(self, evolution):
        self.evolution = evolution
        self.respawn_period = cfg.RESPAWN.RESPAWN_PERIOD
        self.max_spawns_per_cycle = cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE
        self.last_respawn_tick = 0

        self.reproduction_enabled_override = True
        self.mutation_overrides: dict[str, float] = {}

    def reset_runtime_modifiers(self) -> None:
        self.reproduction_enabled_override = True
        self.mutation_overrides = {}

    def set_runtime_modifiers(
        self,
        *,
        reproduction_enabled: bool = True,
        mutation_overrides: dict[str, float] | None = None,
    ) -> None:
        self.reproduction_enabled_override = bool(reproduction_enabled)
        self.mutation_overrides = dict(mutation_overrides or {})

    def _handle_extinction(self, tick: int, registry, grid, dead_slots: list[int], logger: Optional[DataLogger]) -> int:
        policy = str(cfg.RESPAWN.EXTINCTION_POLICY)
        if policy == "fail_run":
            raise RuntimeError("Population dropped below two live agents; binary reproduction is impossible under EXTINCTION_POLICY=fail_run")
        if policy not in {"seed_bank_bootstrap", "admin_spawn_defaults"}:
            raise ValueError(f"Unsupported extinction policy: {policy}")

        spawns = 0
        for slot_idx in dead_slots[: int(cfg.RESPAWN.EXTINCTION_BOOTSTRAP_SPAWNS)]:
            latent = default_trait_latent()
            mapped = trait_values_from_latent(latent)
            traits = {"mass": mapped["mass"], "hp_max": mapped["hp_max"], "vision": mapped["vision"], "metab": mapped["metab"]}
            x, y = self.evolution._find_free_cell(grid)
            if x is None:
                continue
            child_uid = registry.spawn_agent(
                slot_idx,
                x,
                y,
                parent_uid=-1,
                grid=grid,
                traits=traits,
                tick_born=tick,
                family_id=cfg.RESPAWN.EXTINCTION_BOOTSTRAP_FAMILY,
                parent_roles={"brain_parent_uid": -1, "trait_parent_uid": -1, "anchor_parent_uid": -1},
                trait_latent=latent,
                birth_hp=birth_hp_value(traits),
            )
            if logger:
                logger.log_spawn_event(
                    tick=tick,
                    child_slot=slot_idx,
                    brain_parent_slot=-1,
                    trait_parent_slot=-1,
                    anchor_parent_slot=-1,
                    child_uid=child_uid,
                    brain_parent_uid=-1,
                    trait_parent_uid=-1,
                    anchor_parent_uid=-1,
                    child_family=cfg.RESPAWN.EXTINCTION_BOOTSTRAP_FAMILY,
                    brain_parent_family=None,
                    trait_parent_family=None,
                    traits=traits,
                    trait_latent=latent,
                    mutation_flags={"rare_mutation": False, "family_shift": False, "extinction_bootstrap": True},
                    placement={"x": x, "y": y, "attempts": 1, "used_global_fallback": True, "failure_reason": None},
                    floor_recovery=False,
                )
            spawns += 1
        return spawns

    def step(self, tick: int, registry, grid, logger: Optional[DataLogger] = None):
        if not self.reproduction_enabled_override:
            return

        num_alive = registry.get_num_alive()
        floor = cfg.RESPAWN.POPULATION_FLOOR
        ceiling = cfg.RESPAWN.POPULATION_CEILING
        if num_alive >= ceiling:
            return

        is_below_floor = num_alive < floor
        is_timer_ready = tick - self.last_respawn_tick >= self.respawn_period
        if not is_below_floor and not is_timer_ready:
            return

        pending_finalization = torch.nonzero((registry.data[registry.ALIVE] <= 0.5) & (registry.slot_uid >= 0), as_tuple=False).squeeze(-1)
        if pending_finalization.numel() > 0:
            raise AssertionError(f"Respawn cannot reuse slots before UID finalization: pending slots {pending_finalization.tolist()}")

        self.last_respawn_tick = tick
        dead_slots = [int(slot.item()) for slot in torch.nonzero((registry.data[registry.ALIVE] <= 0.5) & (registry.slot_uid < 0), as_tuple=False).squeeze(-1)]
        if not dead_slots:
            return

        num_room_before_ceil = ceiling - num_alive
        num_to_spawn = max(0, min(self.max_spawns_per_cycle, len(dead_slots), num_room_before_ceil))
        if num_to_spawn == 0:
            return

        if num_alive < 2:
            self._handle_extinction(tick, registry, grid, dead_slots[:num_to_spawn], logger)
            return

        spawns = 0
        for slot_idx in dead_slots:
            if spawns >= num_to_spawn:
                break
            parent_roles = select_parent_roles(registry, floor_recovery=is_below_floor)

            brain_parent_slot = registry.get_slot_for_uid(parent_roles.brain_parent_uid)
            trait_parent_slot = registry.get_slot_for_uid(parent_roles.trait_parent_uid)
            anchor_parent_slot = registry.get_slot_for_uid(parent_roles.anchor_parent_uid)
            if brain_parent_slot is None or trait_parent_slot is None or anchor_parent_slot is None:
                raise AssertionError("Parent-role selection returned an inactive UID")

            placement = place_offspring_near_anchor(registry, grid, anchor_parent_slot, self.evolution)
            if placement.x is None or placement.y is None:
                if logger and cfg.RESPAWN.LOG_PLACEMENT_FAILURES:
                    logger.log_spawn_event(
                        tick=tick,
                        child_slot=slot_idx,
                        brain_parent_slot=brain_parent_slot,
                        trait_parent_slot=trait_parent_slot,
                        anchor_parent_slot=anchor_parent_slot,
                        child_uid=-1,
                        brain_parent_uid=parent_roles.brain_parent_uid,
                        trait_parent_uid=parent_roles.trait_parent_uid,
                        anchor_parent_uid=parent_roles.anchor_parent_uid,
                        child_family=None,
                        brain_parent_family=registry.get_family_for_uid(parent_roles.brain_parent_uid),
                        trait_parent_family=registry.get_family_for_uid(parent_roles.trait_parent_uid),
                        traits={},
                        trait_latent={},
                        mutation_flags={"rare_mutation": False, "family_shift": False, "placement_failed": True},
                        placement={
                            "x": None,
                            "y": None,
                            "attempts": placement.attempts,
                            "used_global_fallback": placement.used_global_fallback,
                            "failure_reason": placement.failure_reason,
                        },
                        floor_recovery=is_below_floor,
                    )
                continue

            trait_parent_latent = registry.get_trait_latent_for_uid(parent_roles.trait_parent_uid)
            child_latent, mutation_flags = mutate_trait_latent(trait_parent_latent, mutation_overrides=self.mutation_overrides)
            mapped = trait_values_from_latent(child_latent)
            traits = {"mass": mapped["mass"], "hp_max": mapped["hp_max"], "vision": mapped["vision"], "metab": mapped["metab"]}

            brain_parent_family = registry.get_family_for_uid(parent_roles.brain_parent_uid)
            child_family = brain_parent_family
            if mutation_flags.family_shift:
                child_family = pick_shifted_family(brain_parent_family)

            child_brain = registry.ensure_slot_brain_family(slot_idx, child_family)
            if child_family == brain_parent_family:
                parent_brain = registry.brains[brain_parent_slot]
                if parent_brain is not None:
                    child_brain.load_state_dict(parent_brain.state_dict())
            self.evolution._apply_policy_noise(child_brain, sigma=policy_noise_sigma(mutation_flags, mutation_overrides=self.mutation_overrides))

            child_uid = registry.spawn_agent(
                slot_idx,
                placement.x,
                placement.y,
                parent_uid=parent_roles.brain_parent_uid,
                grid=grid,
                traits=traits,
                tick_born=tick,
                family_id=child_family,
                parent_roles={
                    "brain_parent_uid": parent_roles.brain_parent_uid,
                    "trait_parent_uid": parent_roles.trait_parent_uid,
                    "anchor_parent_uid": parent_roles.anchor_parent_uid,
                },
                trait_latent=child_latent,
                birth_hp=birth_hp_value(traits),
            )

            if logger:
                logger.log_spawn_event(
                    tick=tick,
                    child_slot=slot_idx,
                    brain_parent_slot=brain_parent_slot,
                    trait_parent_slot=trait_parent_slot,
                    anchor_parent_slot=anchor_parent_slot,
                    child_uid=child_uid,
                    brain_parent_uid=parent_roles.brain_parent_uid,
                    trait_parent_uid=parent_roles.trait_parent_uid,
                    anchor_parent_uid=parent_roles.anchor_parent_uid,
                    child_family=child_family,
                    brain_parent_family=brain_parent_family,
                    trait_parent_family=registry.get_family_for_uid(parent_roles.trait_parent_uid),
                    traits=traits,
                    trait_latent=child_latent,
                    mutation_flags={"rare_mutation": mutation_flags.rare_mutation, "family_shift": mutation_flags.family_shift},
                    placement={
                        "x": placement.x,
                        "y": placement.y,
                        "attempts": placement.attempts,
                        "used_global_fallback": placement.used_global_fallback,
                        "failure_reason": placement.failure_reason,
                    },
                    floor_recovery=is_below_floor,
                )
            spawns += 1
