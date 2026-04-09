"""Respawn controller for binary parented births and extinction recovery."""

from __future__ import annotations

from typing import Optional

import torch

from ..config_bridge import cfg
from ..telemetry.data_logger import DataLogger
from .reproduction import (
    ParentSelectionError,
    birth_hp_value,
    default_trait_latent,
    mutate_trait_latent,
    pick_shifted_family,
    place_offspring_near_anchor,
    policy_noise_sigma,
    select_parent_roles,
    trait_values_from_latent,
)


_DOCTRINE_METADATA = {
    "crowding": {
        "display_name": "The Ashen Press",
        "short_name": "Ashen Press",
        "technical_subtitle": "crowding-gated reproduction overlay",
    },
    "cooldown": {
        "display_name": "The Widow Interval",
        "short_name": "Widow Interval",
        "technical_subtitle": "parent refractory reproduction overlay",
    },
    "local_parent": {
        "display_name": "The Bloodhold Radius",
        "short_name": "Bloodhold Radius",
        "technical_subtitle": "local lineage parent-selection overlay",
    },
}
_COOLDOWN_LEDGER_NAMES = ("unified", "brain_parent", "trait_parent", "anchor_parent")


class RespawnController:
    """Binary reproduction controller with catastrophe-aware runtime gates."""

    def __init__(self, evolution):
        self.evolution = evolution
        self.respawn_period = cfg.RESPAWN.RESPAWN_PERIOD
        self.max_spawns_per_cycle = cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE
        self.last_respawn_tick = 0

        self.reproduction_enabled_override = True
        self.mutation_overrides: dict[str, float] = {}
        self.doctrine_overrides: dict[str, bool | None] = {key: None for key in _DOCTRINE_METADATA}
        self.cooldown_ledgers: dict[str, dict[int, int]] = {name: {} for name in _COOLDOWN_LEDGER_NAMES}

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

    def _overlay_default_enabled(self, doctrine: str) -> bool:
        if doctrine == "crowding":
            return bool(cfg.RESPAWN.OVERLAYS.CROWDING.ENABLED)
        if doctrine == "cooldown":
            return bool(cfg.RESPAWN.OVERLAYS.COOLDOWN.ENABLED)
        if doctrine == "local_parent":
            return bool(cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.ENABLED)
        raise KeyError(f"Unknown reproduction doctrine: {doctrine}")

    def get_doctrine_effective_enabled(self, doctrine: str) -> bool:
        override = self.doctrine_overrides[doctrine]
        if override is None:
            return self._overlay_default_enabled(doctrine)
        return bool(override)

    def toggle_doctrine_override(self, doctrine: str) -> bool:
        if doctrine not in self.doctrine_overrides:
            raise KeyError(f"Unknown reproduction doctrine: {doctrine}")
        self.doctrine_overrides[doctrine] = not self.get_doctrine_effective_enabled(doctrine)
        return self.get_doctrine_effective_enabled(doctrine)

    def clear_doctrine_overrides(self) -> None:
        for doctrine in self.doctrine_overrides:
            self.doctrine_overrides[doctrine] = None

    def get_crowding_runtime(self, *, floor_recovery: bool) -> dict[str, object]:
        overlay = cfg.RESPAWN.OVERLAYS.CROWDING
        enabled = self.get_doctrine_effective_enabled("crowding")
        if not enabled:
            return {
                "active": False,
                "policy_when_crowded": None,
                "active_policy": "disabled",
                "local_radius": int(overlay.LOCAL_RADIUS),
                "max_neighbors": int(overlay.MAX_NEIGHBORS),
            }
        policy = str(overlay.POLICY_WHEN_CROWDED)
        active = True
        if floor_recovery:
            below_policy = str(overlay.BELOW_FLOOR_POLICY)
            if below_policy == "bypass":
                active = False
                policy = "bypass"
            elif below_policy == "global_only":
                policy = "global_only"
            else:
                policy = str(overlay.POLICY_WHEN_CROWDED)
        return {
            "active": active,
            "policy_when_crowded": None if not active else policy,
            "active_policy": policy,
            "local_radius": int(overlay.LOCAL_RADIUS),
            "max_neighbors": int(overlay.MAX_NEIGHBORS),
        }

    def get_local_parent_runtime(self, *, floor_recovery: bool) -> dict[str, object]:
        overlay = cfg.RESPAWN.OVERLAYS.LOCAL_PARENT
        enabled = self.get_doctrine_effective_enabled("local_parent")
        if not enabled:
            return {
                "active": False,
                "selection_radius": int(overlay.SELECTION_RADIUS),
                "fallback_behavior": None,
                "active_policy": "disabled",
            }
        active = True
        fallback_behavior = str(overlay.FALLBACK_BEHAVIOR)
        if floor_recovery:
            below_policy = str(overlay.BELOW_FLOOR_POLICY)
            if below_policy == "bypass":
                active = False
                fallback_behavior = None
                active_policy = "bypass"
            elif below_policy == "prefer_local_then_global":
                fallback_behavior = "global"
                active_policy = "prefer_local_then_global"
            else:
                fallback_behavior = "strict"
                active_policy = "strict"
        else:
            active_policy = fallback_behavior
        return {
            "active": active,
            "selection_radius": int(overlay.SELECTION_RADIUS),
            "fallback_behavior": fallback_behavior,
            "active_policy": active_policy,
        }

    def get_cooldown_role_runtime(self, role_name: str, *, floor_recovery: bool) -> dict[str, object]:
        overlay = cfg.RESPAWN.OVERLAYS.COOLDOWN
        apply_flags = {
            "brain_parent": bool(overlay.APPLY_TO_BRAIN_PARENT),
            "trait_parent": bool(overlay.APPLY_TO_TRAIT_PARENT),
            "anchor_parent": bool(overlay.APPLY_TO_ANCHOR_PARENT),
        }
        enabled = self.get_doctrine_effective_enabled("cooldown")
        duration = max(0, int(overlay.DURATION_TICKS))
        if not enabled or not apply_flags.get(role_name, False) or duration <= 0:
            return {
                "active": False,
                "duration_ticks": duration,
                "unified_uid_policy": bool(overlay.UNIFIED_UID_POLICY),
                "empty_pool_policy": None,
                "active_policy": "disabled" if not enabled else "duration=0",
            }
        if floor_recovery:
            below_policy = str(overlay.BELOW_FLOOR_POLICY)
            if below_policy == "bypass":
                return {
                    "active": False,
                    "duration_ticks": duration,
                    "unified_uid_policy": bool(overlay.UNIFIED_UID_POLICY),
                    "empty_pool_policy": None,
                    "active_policy": "bypass",
                }
            if below_policy == "strict":
                empty_pool_policy = "strict"
                active_policy = "strict"
            else:
                empty_pool_policy = "allow_best_available"
                active_policy = "allow_best_available"
        else:
            empty_pool_policy = str(overlay.EMPTY_POOL_POLICY)
            active_policy = empty_pool_policy
        return {
            "active": True,
            "duration_ticks": duration,
            "unified_uid_policy": bool(overlay.UNIFIED_UID_POLICY),
            "empty_pool_policy": empty_pool_policy,
            "active_policy": active_policy,
        }

    def uid_on_cooldown(self, uid: int, role_name: str, tick: int, *, runtime: dict[str, object]) -> bool:
        if not runtime["active"]:
            return False
        ledger_name = "unified" if runtime["unified_uid_policy"] else role_name
        last_tick = self.cooldown_ledgers[ledger_name].get(int(uid))
        if last_tick is None:
            return False
        return int(tick) - int(last_tick) < int(runtime["duration_ticks"])

    def record_parent_role_usage(self, parent_roles, tick: int, *, floor_recovery: bool) -> None:
        role_to_uid = {
            "brain_parent": int(parent_roles.brain_parent_uid),
            "trait_parent": int(parent_roles.trait_parent_uid),
            "anchor_parent": int(parent_roles.anchor_parent_uid),
        }
        for role_name, uid in role_to_uid.items():
            if uid < 0:
                continue
            runtime = self.get_cooldown_role_runtime(role_name, floor_recovery=floor_recovery)
            if not runtime["active"]:
                continue
            ledger_name = "unified" if runtime["unified_uid_policy"] else role_name
            self.cooldown_ledgers[ledger_name][uid] = int(tick)

    def serialize_runtime_state(self) -> dict[str, object]:
        return {
            "doctrine_overrides": dict(self.doctrine_overrides),
            "cooldown_ledgers": {
                ledger_name: {int(uid): int(last_tick) for uid, last_tick in ledger.items()}
                for ledger_name, ledger in self.cooldown_ledgers.items()
            },
        }

    def restore_runtime_state(self, payload: dict | None) -> None:
        self.clear_doctrine_overrides()
        self.cooldown_ledgers = {name: {} for name in _COOLDOWN_LEDGER_NAMES}
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise ValueError("Respawn overlay runtime state must be a dict")

        overrides = payload.get("doctrine_overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError("Respawn doctrine overrides must be a dict")
        for doctrine in self.doctrine_overrides:
            value = overrides.get(doctrine)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"Respawn doctrine override for {doctrine} must be bool or None")
            self.doctrine_overrides[doctrine] = value

        ledgers = payload.get("cooldown_ledgers", {})
        if ledgers is None:
            ledgers = {}
        if not isinstance(ledgers, dict):
            raise ValueError("Respawn cooldown ledgers must be a dict")
        for ledger_name in _COOLDOWN_LEDGER_NAMES:
            ledger_payload = ledgers.get(ledger_name, {})
            if ledger_payload is None:
                ledger_payload = {}
            if not isinstance(ledger_payload, dict):
                raise ValueError(f"Respawn cooldown ledger {ledger_name} must be a dict")
            self.cooldown_ledgers[ledger_name] = {
                int(uid): int(last_tick) for uid, last_tick in ledger_payload.items()
            }

    def build_overlay_status(self, num_alive: int) -> dict[str, object]:
        below_floor_active = int(num_alive) < int(cfg.RESPAWN.POPULATION_FLOOR)
        crowding_runtime = self.get_crowding_runtime(floor_recovery=below_floor_active)
        local_runtime = self.get_local_parent_runtime(floor_recovery=below_floor_active)
        cooldown_overlay = cfg.RESPAWN.OVERLAYS.COOLDOWN
        if not self.get_doctrine_effective_enabled("cooldown"):
            cooldown_policy = "disabled"
        elif int(cooldown_overlay.DURATION_TICKS) <= 0:
            cooldown_policy = "duration=0"
        elif below_floor_active:
            cooldown_policy = str(cooldown_overlay.BELOW_FLOOR_POLICY)
        else:
            cooldown_policy = str(cooldown_overlay.EMPTY_POOL_POLICY)
        doctrines = {}
        for doctrine, meta in _DOCTRINE_METADATA.items():
            default_enabled = self._overlay_default_enabled(doctrine)
            override = self.doctrine_overrides[doctrine]
            effective_enabled = self.get_doctrine_effective_enabled(doctrine)
            if doctrine == "crowding":
                active_policy = crowding_runtime["active_policy"]
            elif doctrine == "cooldown":
                active_policy = cooldown_policy
            else:
                active_policy = local_runtime["active_policy"]
            doctrines[doctrine] = {
                **meta,
                "default_enabled": default_enabled,
                "override": override,
                "effective_enabled": effective_enabled,
                "override_differs": override is not None and bool(override) != bool(default_enabled),
                "active_policy": active_policy,
            }
        return {
            "reproduction_enabled": bool(self.reproduction_enabled_override),
            "below_floor_active": below_floor_active,
            "doctrines": doctrines,
        }

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
            x, y = self.evolution.find_free_cell(grid)
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
                    placement={
                        "x": x,
                        "y": y,
                        "attempts": 1,
                        "used_global_fallback": True,
                        "failure_reason": None,
                        "crowding_checked": False,
                        "crowding_neighbor_count": 0,
                        "crowding_policy_applied": None,
                    },
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

            try:
                parent_roles = select_parent_roles(
                    registry,
                    floor_recovery=is_below_floor,
                    dead_slot=slot_idx,
                    respawn_controller=self,
                    tick=tick,
                )
            except ParentSelectionError as exc:
                if logger and cfg.RESPAWN.LOG_PLACEMENT_FAILURES:
                    logger.log_spawn_event(
                        tick=tick,
                        child_slot=slot_idx,
                        brain_parent_slot=-1,
                        trait_parent_slot=-1,
                        anchor_parent_slot=-1,
                        child_uid=-1,
                        brain_parent_uid=-1,
                        trait_parent_uid=-1,
                        anchor_parent_uid=-1,
                        child_family=None,
                        brain_parent_family=None,
                        trait_parent_family=None,
                        traits={},
                        trait_latent={},
                        mutation_flags={
                            "rare_mutation": False,
                            "family_shift": False,
                            "parent_selection_blocked": True,
                            "local_parenting_enabled": exc.local_parenting_enabled,
                            "local_parenting_used_global_fallback": exc.local_parenting_used_global_fallback,
                            "local_parent_candidate_count": exc.local_parent_candidate_count,
                            "cooldown_relaxed_brain": exc.cooldown_relaxed_brain,
                            "cooldown_relaxed_trait": exc.cooldown_relaxed_trait,
                            "cooldown_relaxed_anchor": exc.cooldown_relaxed_anchor,
                        },
                        placement={
                            "x": None,
                            "y": None,
                            "attempts": 0,
                            "used_global_fallback": False,
                            "failure_reason": exc.failure_reason,
                            "crowding_checked": False,
                            "crowding_neighbor_count": 0,
                            "crowding_policy_applied": None,
                        },
                        floor_recovery=is_below_floor,
                    )
                continue

            brain_parent_slot = registry.get_slot_for_uid(parent_roles.brain_parent_uid)
            trait_parent_slot = registry.get_slot_for_uid(parent_roles.trait_parent_uid)
            anchor_parent_slot = registry.get_slot_for_uid(parent_roles.anchor_parent_uid)
            if brain_parent_slot is None or trait_parent_slot is None or anchor_parent_slot is None:
                raise AssertionError("Parent-role selection returned an inactive UID")

            placement = place_offspring_near_anchor(
                registry,
                grid,
                anchor_parent_slot,
                self.evolution,
                floor_recovery=is_below_floor,
                respawn_controller=self,
            )
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
                        mutation_flags={
                            "rare_mutation": False,
                            "family_shift": False,
                            "placement_failed": True,
                            "local_parenting_enabled": parent_roles.local_parenting_enabled,
                            "local_parenting_used_global_fallback": parent_roles.local_parenting_used_global_fallback,
                            "local_parent_candidate_count": parent_roles.local_parent_candidate_count,
                            "cooldown_relaxed_brain": parent_roles.cooldown_relaxed_brain,
                            "cooldown_relaxed_trait": parent_roles.cooldown_relaxed_trait,
                            "cooldown_relaxed_anchor": parent_roles.cooldown_relaxed_anchor,
                        },
                        placement={
                            "x": None,
                            "y": None,
                            "attempts": placement.attempts,
                            "used_global_fallback": placement.used_global_fallback,
                            "failure_reason": placement.failure_reason,
                            "crowding_checked": placement.crowding_checked,
                            "crowding_neighbor_count": placement.crowding_neighbor_count,
                            "crowding_policy_applied": placement.crowding_policy_applied,
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
            self.evolution.apply_policy_noise(
                child_brain,
                sigma=policy_noise_sigma(mutation_flags, mutation_overrides=self.mutation_overrides),
            )

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
            self.record_parent_role_usage(parent_roles, tick, floor_recovery=is_below_floor)

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
                    mutation_flags={
                        "rare_mutation": mutation_flags.rare_mutation,
                        "family_shift": mutation_flags.family_shift,
                        "local_parenting_enabled": parent_roles.local_parenting_enabled,
                        "local_parenting_used_global_fallback": parent_roles.local_parenting_used_global_fallback,
                        "local_parent_candidate_count": parent_roles.local_parent_candidate_count,
                        "cooldown_relaxed_brain": parent_roles.cooldown_relaxed_brain,
                        "cooldown_relaxed_trait": parent_roles.cooldown_relaxed_trait,
                        "cooldown_relaxed_anchor": parent_roles.cooldown_relaxed_anchor,
                    },
                    placement={
                        "x": placement.x,
                        "y": placement.y,
                        "attempts": placement.attempts,
                        "used_global_fallback": placement.used_global_fallback,
                        "failure_reason": placement.failure_reason,
                        "crowding_checked": placement.crowding_checked,
                        "crowding_neighbor_count": placement.crowding_neighbor_count,
                        "crowding_policy_applied": placement.crowding_policy_applied,
                    },
                    floor_recovery=is_below_floor,
                )
            spawns += 1
