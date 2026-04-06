"""
Handles all agent movement, collision detection, and environmental interactions.

This module is the referee of the simulation. It takes chosen actions from each
agent and determines the physically valid outcomes without owning higher-level
training or respawn policy.

Prompt 6 adds only temporary catastrophe-aware runtime modifiers. The stored
agent traits remain canonical; catastrophe effects are applied as reversible
multipliers during the active window only.

Prompt 7 adds explicit death-cause bookkeeping. The bookkeeping is diagnostic
only; it does not change any damage amount, movement resolution, or survival
semantics.
"""

from __future__ import annotations

import torch

from ..config_bridge import cfg


DEATH_REASON_PRIORITY = {
    "wall_collision": 100,
    "ram_damage": 90,
    "contest_damage": 80,
    "poison_zone": 70,
    "catastrophe": 60,
    "metabolism_death": 50,
    "birth_failure_cleanup": 40,
    "admin_kill": 30,
    "extinction_cleanup": 20,
    "unknown": 0,
}


class Physics:
    DIRECTIONS = [
        (0, 0),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]

    def __init__(self, grid, registry):
        self.grid = grid
        self.registry = registry
        self.collision_log = []

        self.collision_damage_multiplier = 1.0
        self.metabolism_multiplier = 1.0
        self.mass_metabolism_burden = 0.0

        self._pending_death_context_by_slot: dict[int, dict] = {}
        self._resolved_death_context_by_slot: dict[int, dict] = {}
        self._catastrophe_state: dict = {}

    def reset_runtime_modifiers(self) -> None:
        self.collision_damage_multiplier = 1.0
        self.metabolism_multiplier = 1.0
        self.mass_metabolism_burden = 0.0

    def set_runtime_modifiers(
        self,
        *,
        collision_damage_multiplier: float = 1.0,
        metabolism_multiplier: float = 1.0,
        mass_metabolism_burden: float = 0.0,
    ) -> None:
        self.collision_damage_multiplier = max(0.0, float(collision_damage_multiplier))
        self.metabolism_multiplier = max(0.0, float(metabolism_multiplier))
        self.mass_metabolism_burden = max(0.0, float(mass_metabolism_burden))

    def set_catastrophe_state(self, catastrophe_state: dict | None) -> None:
        self._catastrophe_state = dict(catastrophe_state or {})

    def _active_catastrophe_id(self) -> int | None:
        details = self._catastrophe_state.get("active_details", [])
        if not details:
            return None
        return int(details[0]["event_id"])

    def _active_catastrophe_name(self) -> str | None:
        details = self._catastrophe_state.get("active_details", [])
        if not details:
            return None
        return str(details[0]["catastrophe_id"])

    def _record_death_context(
        self,
        slot_idx: int,
        *,
        death_reason: str,
        killing_agent_uid: int | None = None,
        catastrophe_id: int | None = None,
        zone_id: int | None = None,
    ) -> None:
        death_reason = str(death_reason)
        existing = self._pending_death_context_by_slot.get(int(slot_idx))
        existing_priority = DEATH_REASON_PRIORITY.get(str(existing.get("death_reason", "unknown")), 0) if existing else -1
        new_priority = DEATH_REASON_PRIORITY.get(death_reason, 0)
        if existing is not None and existing_priority > new_priority:
            return

        self._pending_death_context_by_slot[int(slot_idx)] = {
            "death_reason": death_reason,
            "killing_agent_uid": killing_agent_uid,
            "catastrophe_id": catastrophe_id,
            "zone_id": zone_id,
        }

    def step(self, actions: torch.Tensor) -> dict:
        self.collision_log = []
        self._pending_death_context_by_slot = {}
        self._resolved_death_context_by_slot = {}
        stats = {"wall_collisions": 0, "rams": 0, "contests": 0}

        alive_indices = self.registry.get_alive_indices()
        if len(alive_indices) == 0:
            return stats

        intents = []
        for idx in alive_indices:
            idx_int = int(idx.item())
            action = int(actions[idx_int].item())
            x = int(self.registry.data[self.registry.X, idx_int].item())
            y = int(self.registry.data[self.registry.Y, idx_int].item())
            dx, dy = self.DIRECTIONS[action]
            tx, ty = x + dx, y + dy
            tx = max(0, min(tx, self.grid.W - 1))
            ty = max(0, min(ty, self.grid.H - 1))
            intents.append({"idx": idx_int, "action": action, "x": x, "y": y, "tx": tx, "ty": ty})

        non_movers = set()

        for intent in intents:
            idx = intent["idx"]
            tx, ty = intent["tx"], intent["ty"]

            if intent["action"] == 0:
                non_movers.add(idx)
                continue

            if self.grid.is_wall(tx, ty):
                self._handle_wall_collision(idx)
                stats["wall_collisions"] += 1
                non_movers.add(idx)
                continue

            target_agent = self.grid.get_agent_at(tx, ty)
            if target_agent >= 0 and target_agent != idx:
                target_intent = next((candidate for candidate in intents if candidate["idx"] == target_agent), None)
                if target_intent and target_intent["action"] == 0:
                    self._handle_ram(idx, target_agent)
                    stats["rams"] += 1
                    non_movers.add(idx)
                    continue

        move_approved = {}

        for intent in intents:
            idx = intent["idx"]
            tx, ty = intent["tx"], intent["ty"]

            if idx in non_movers:
                continue

            target_agent = self.grid.get_agent_at(tx, ty)
            if target_agent >= 0 and target_agent in non_movers:
                self._handle_ram(idx, target_agent)
                stats["rams"] += 1
                non_movers.add(idx)
                continue

            move_approved.setdefault((tx, ty), []).append(idx)

        for (tx, ty), contenders in move_approved.items():
            if len(contenders) == 1:
                idx = contenders[0]
                target_agent = self.grid.get_agent_at(tx, ty)
                if target_agent < 0:
                    self._approve_move(idx, tx, ty)
                else:
                    self._handle_ram(idx, target_agent)
                    stats["rams"] += 1
                    non_movers.add(idx)
            else:
                winner = self._resolve_contest(contenders)
                self._approve_move(winner, tx, ty)
                stats["contests"] += 1
                for idx in contenders:
                    if idx != winner:
                        non_movers.add(idx)

        return stats

    def _damage_scalar(self) -> float:
        return float(self.collision_damage_multiplier)

    def _handle_wall_collision(self, idx: int):
        mass = self.registry.data[self.registry.MASS, idx]
        damage = mass * cfg.PHYS.K_WALL_PENALTY * self._damage_scalar()
        self.registry.data[self.registry.HP, idx] -= damage
        self.registry.data[self.registry.HP_LOST_PHYSICS, idx] += damage
        self._record_death_context(
            idx,
            death_reason="wall_collision",
            catastrophe_id=self._active_catastrophe_id(),
        )
        self.collision_log.append(
            {
                "kind": "wall",
                "a": idx,
                "b": -1,
                "damage": damage.item(),
                "damage_a": float("nan"),
                "damage_b": float("nan"),
                "contenders": [],
                "winner": -1,
                "catastrophe_collision_scalar": self._damage_scalar(),
            }
        )

    def _handle_ram(self, rammer_idx: int, target_idx: int):
        mass_a = self.registry.data[self.registry.MASS, rammer_idx]
        scalar = self._damage_scalar()

        damage_a = mass_a * cfg.PHYS.K_RAM_PENALTY * scalar
        self.registry.data[self.registry.HP, rammer_idx] -= damage_a
        self.registry.data[self.registry.HP_LOST_PHYSICS, rammer_idx] += damage_a

        damage_b = mass_a * cfg.PHYS.K_IDLE_HIT_PENALTY * scalar
        self.registry.data[self.registry.HP, target_idx] -= damage_b
        self.registry.data[self.registry.HP_LOST_PHYSICS, target_idx] += damage_b

        rammer_uid = self.registry.get_uid_for_slot(rammer_idx)
        self._record_death_context(
            rammer_idx,
            death_reason="ram_damage",
            killing_agent_uid=self.registry.get_uid_for_slot(target_idx),
            catastrophe_id=self._active_catastrophe_id(),
        )
        self._record_death_context(
            target_idx,
            death_reason="ram_damage",
            killing_agent_uid=rammer_uid,
            catastrophe_id=self._active_catastrophe_id(),
        )

        self.collision_log.append(
            {
                "kind": "ram",
                "a": rammer_idx,
                "b": target_idx,
                "damage": float("nan"),
                "damage_a": damage_a.item(),
                "damage_b": damage_b.item(),
                "contenders": [],
                "winner": -1,
                "catastrophe_collision_scalar": scalar,
            }
        )

    def _resolve_contest(self, contenders: list[int]) -> int:
        strengths = []
        for idx in contenders:
            mass = self.registry.data[self.registry.MASS, idx]
            hp = self.registry.data[self.registry.HP, idx]
            hp_max = self.registry.data[self.registry.HP_MAX, idx]
            hp_ratio = hp / (hp_max + 1e-6)
            strength = mass * hp_ratio
            strengths.append((strength.item(), idx))

        strengths.sort(key=lambda value: (-value[0], value[1]))
        winner = strengths[0][1]
        scalar = self._damage_scalar()

        for _, idx in strengths:
            damage = (cfg.PHYS.K_WINNER_DAMAGE if idx == winner else cfg.PHYS.K_LOSER_DAMAGE) * scalar
            self.registry.data[self.registry.HP, idx] -= damage
            self.registry.data[self.registry.HP_LOST_PHYSICS, idx] += damage
            self._record_death_context(
                idx,
                death_reason="contest_damage",
                killing_agent_uid=self.registry.get_uid_for_slot(winner) if idx != winner else None,
                catastrophe_id=self._active_catastrophe_id(),
            )

        self.collision_log.append(
            {
                "kind": "contest",
                "a": -1,
                "b": -1,
                "damage": float("nan"),
                "damage_a": float("nan"),
                "damage_b": float("nan"),
                "contenders": contenders,
                "winner": winner,
                "catastrophe_collision_scalar": scalar,
            }
        )
        return winner

    def _approve_move(self, idx: int, tx: int, ty: int):
        old_x = int(self.registry.data[self.registry.X, idx].item())
        old_y = int(self.registry.data[self.registry.Y, idx].item())
        self.grid.clear_cell(old_x, old_y)
        self.registry.data[self.registry.X, idx] = float(tx)
        self.registry.data[self.registry.Y, idx] = float(ty)
        mass = self.registry.data[self.registry.MASS, idx].item()
        self.grid.set_cell(tx, ty, idx, mass)

    def apply_environment_effects(self):
        alive_indices = self.registry.get_alive_indices()
        for idx in alive_indices:
            idx_int = int(idx.item())
            x = int(self.registry.data[self.registry.X, idx_int].item())
            y = int(self.registry.data[self.registry.Y, idx_int].item())

            h_rate = self.grid.get_h_rate(x, y)
            self.registry.data[self.registry.HP, idx_int] += h_rate
            if h_rate > 0:
                self.registry.data[self.registry.HP_GAINED, idx_int] += h_rate
            elif h_rate < 0:
                self._record_death_context(
                    idx_int,
                    death_reason="poison_zone",
                    catastrophe_id=self._active_catastrophe_id(),
                )

            metab = self.registry.data[self.registry.METABOLISM_RATE, idx_int]
            mass = self.registry.data[self.registry.MASS, idx_int]
            effective_metab = (metab * self.metabolism_multiplier) + (mass * self.mass_metabolism_burden)
            self.registry.data[self.registry.HP, idx_int] -= effective_metab
            if float(effective_metab.item()) > 0.0:
                self._record_death_context(
                    idx_int,
                    death_reason="metabolism_death",
                    catastrophe_id=self._active_catastrophe_id(),
                )

    def process_deaths(self):
        alive_indices = self.registry.get_alive_indices()
        deaths = []

        for idx in alive_indices:
            idx_int = int(idx.item())
            hp = self.registry.data[self.registry.HP, idx_int]
            hp_max = self.registry.data[self.registry.HP_MAX, idx_int]
            hp_clamped = torch.clamp(hp, 0.0, hp_max)
            self.registry.data[self.registry.HP, idx_int] = hp_clamped
            if hp_clamped <= 0.0:
                context = dict(self._pending_death_context_by_slot.get(idx_int, {}))
                if not context:
                    context = {
                        "death_reason": "catastrophe" if self._active_catastrophe_id() is not None else "unknown",
                        "catastrophe_id": self._active_catastrophe_id(),
                        "zone_id": None,
                        "killing_agent_uid": None,
                    }
                context.setdefault("death_reason", "unknown")
                context.setdefault("catastrophe_id", self._active_catastrophe_id())
                context.setdefault("zone_id", None)
                context.setdefault("killing_agent_uid", None)
                self._resolved_death_context_by_slot[idx_int] = context
                self.registry.mark_dead(idx_int, self.grid)
                deaths.append(idx_int)

        return deaths

    def consume_death_context(self, slot_idx: int) -> dict:
        return dict(
            self._resolved_death_context_by_slot.pop(
                int(slot_idx),
                {
                    "death_reason": "unknown",
                    "catastrophe_id": self._active_catastrophe_id(),
                    "zone_id": None,
                    "killing_agent_uid": None,
                },
            )
        )
