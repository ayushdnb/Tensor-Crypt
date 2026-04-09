"""
Handles all agent movement, collision detection, and environmental interactions.

This module is the referee of the simulation. It takes chosen actions from each
agent and determines the physically valid outcomes without owning higher-level
training or respawn policy.

Catastrophe-aware runtime modifiers are temporary. The stored agent traits
remain canonical; catastrophe effects are applied as reversible multipliers
during the active window only.

Death-cause bookkeeping is diagnostic only; it does not change any damage
amount, movement resolution, or survival semantics.

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
        self._static_wall_grid_cpu: torch.Tensor | None = None

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

    def refresh_static_wall_cache(self) -> None:
        self._static_wall_grid_cpu = self.grid.grid[0].detach().cpu().clone()

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

    def _stage_resolution_inputs(self, actions: torch.Tensor, alive_indices: torch.Tensor):
        snapshot = torch.stack(
            (
                alive_indices.to(dtype=self.registry.data.dtype),
                actions[alive_indices].to(dtype=self.registry.data.dtype),
                self.registry.data[self.registry.X, alive_indices],
                self.registry.data[self.registry.Y, alive_indices],
                self.registry.data[self.registry.MASS, alive_indices],
            ),
            dim=1,
        ).detach().cpu()
        if self._static_wall_grid_cpu is None:
            self.refresh_static_wall_cache()
        wall_grid = self._static_wall_grid_cpu

        alive_list = snapshot[:, 0].to(torch.int64).tolist()
        action_values = snapshot[:, 1].to(torch.int64).tolist()
        x_values = snapshot[:, 2].to(torch.int64).tolist()
        y_values = snapshot[:, 3].to(torch.int64).tolist()
        mass_values = snapshot[:, 4].tolist()
        return alive_list, action_values, x_values, y_values, mass_values, wall_grid

    def step(self, actions: torch.Tensor) -> dict:
        self.collision_log = []
        self._pending_death_context_by_slot = {}
        self._resolved_death_context_by_slot = {}
        stats = {"wall_collisions": 0, "rams": 0, "contests": 0}

        alive_indices = self.registry.get_alive_indices()
        if len(alive_indices) == 0:
            return stats

        alive_list, action_values, x_values, y_values, mass_values, wall_grid = self._stage_resolution_inputs(
            actions,
            alive_indices,
        )
        mass_by_idx = {
            idx: float(value)
            for idx, value in zip(alive_list, mass_values)
        }

        intents = []
        intent_by_idx = {}
        start_occupancy = {}
        for idx_int, action, x, y in zip(alive_list, action_values, x_values, y_values):
            dx, dy = self.DIRECTIONS[action]
            tx, ty = x + dx, y + dy
            tx = max(0, min(tx, self.grid.W - 1))
            ty = max(0, min(ty, self.grid.H - 1))
            intent = {"idx": idx_int, "action": action, "x": x, "y": y, "tx": tx, "ty": ty}
            intents.append(intent)
            intent_by_idx[idx_int] = intent
            start_occupancy[(x, y)] = idx_int

        non_movers = set()
        contenders_by_cell = {}

        for intent in intents:
            idx = intent["idx"]
            tx, ty = intent["tx"], intent["ty"]

            if intent["action"] == 0:
                non_movers.add(idx)
                continue

            if bool(wall_grid[ty, tx] > 0.5):
                self._handle_wall_collision(idx)
                stats["wall_collisions"] += 1
                non_movers.add(idx)
                continue

            target_agent = start_occupancy.get((tx, ty), -1)
            if target_agent >= 0 and target_agent != idx:
                target_intent = intent_by_idx.get(target_agent)
                if target_intent is None or target_intent["action"] == 0:
                    self._handle_ram(idx, target_agent)
                    stats["rams"] += 1
                    non_movers.add(idx)
                    continue

            contenders_by_cell.setdefault((tx, ty), []).append(idx)

        proposed_moves = {}
        for (tx, ty), contenders in contenders_by_cell.items():
            active_contenders = [idx for idx in contenders if idx not in non_movers]
            if not active_contenders:
                continue
            if len(active_contenders) == 1:
                proposed_moves[active_contenders[0]] = (tx, ty)
            else:
                winner = self._resolve_contest(active_contenders)
                stats["contests"] += 1
                proposed_moves[winner] = (tx, ty)
                for idx in active_contenders:
                    if idx != winner:
                        non_movers.add(idx)

        resolution_cache = {}
        visiting = set()

        def can_move(idx: int) -> bool:
            if idx in resolution_cache:
                return resolution_cache[idx]
            if idx in visiting:
                return True

            visiting.add(idx)
            tx, ty = proposed_moves[idx]
            occupant = start_occupancy.get((tx, ty), -1)
            if occupant == -1 or occupant == idx:
                result = True
            elif occupant in non_movers or occupant not in proposed_moves:
                result = False
            else:
                result = can_move(occupant)
            visiting.remove(idx)
            resolution_cache[idx] = result
            return result

        successful_moves = sorted(idx for idx in proposed_moves if can_move(idx))
        successful_set = set(successful_moves)
        blocked_moves = sorted(idx for idx in proposed_moves if idx not in successful_set)

        for idx in blocked_moves:
            tx, ty = proposed_moves[idx]
            target_agent = start_occupancy.get((tx, ty), -1)
            if target_agent >= 0 and target_agent != idx:
                self._handle_ram(idx, target_agent)
                stats["rams"] += 1
                non_movers.add(idx)

        if successful_moves:
            move_slots = torch.tensor(successful_moves, device=self.registry.device, dtype=torch.long)
            old_x = torch.tensor(
                [intent_by_idx[idx]["x"] for idx in successful_moves],
                device=self.registry.device,
                dtype=torch.long,
            )
            old_y = torch.tensor(
                [intent_by_idx[idx]["y"] for idx in successful_moves],
                device=self.registry.device,
                dtype=torch.long,
            )
            new_x = torch.tensor(
                [proposed_moves[idx][0] for idx in successful_moves],
                device=self.registry.device,
                dtype=torch.long,
            )
            new_y = torch.tensor(
                [proposed_moves[idx][1] for idx in successful_moves],
                device=self.registry.device,
                dtype=torch.long,
            )
            new_mass = torch.tensor(
                [mass_by_idx[idx] for idx in successful_moves],
                device=self.registry.device,
                dtype=self.grid.grid.dtype,
            )

            self.grid.grid[2, old_y, old_x] = -1.0
            self.grid.grid[3, old_y, old_x] = 0.0
            self.registry.data[self.registry.X, move_slots] = new_x.to(dtype=self.registry.data.dtype)
            self.registry.data[self.registry.Y, move_slots] = new_y.to(dtype=self.registry.data.dtype)
            self.grid.grid[2, new_y, new_x] = move_slots.to(dtype=self.grid.grid.dtype)
            self.grid.grid[3, new_y, new_x] = new_mass

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
        """Resolve a contested cell deterministically, including deterministic seeded tie breaks."""
        strengths = []
        for idx in contenders:
            mass = self.registry.data[self.registry.MASS, idx]
            hp = self.registry.data[self.registry.HP, idx]
            hp_max = self.registry.data[self.registry.HP_MAX, idx]
            hp_ratio = hp / (hp_max + 1e-6)
            strength = mass * hp_ratio
            strengths.append((float(strength.item()), idx))

        strengths.sort(key=lambda value: (-value[0], value[1]))
        top_strength = strengths[0][0]
        tied = [idx for strength, idx in strengths if abs(strength - top_strength) <= 1e-9]
        tie_breaker = str(cfg.PHYS.TIE_BREAKER)
        if tie_breaker == "strength_then_lowest_id":
            winner = min(tied)
        elif tie_breaker == "random_seeded":
            ordered = sorted(tied)
            accumulator = int(cfg.SIM.SEED) + int(self.registry.tick_counter) * 2654435761
            for idx in ordered:
                accumulator += (idx + 1) * 2246822519
            winner = ordered[accumulator % len(ordered)]
        else:
            raise ValueError(f"Unsupported contest tie breaker: {cfg.PHYS.TIE_BREAKER}")

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
        """Apply heal/harm zones and metabolism in bulk before per-death context annotation."""
        alive_indices = self.registry.get_alive_indices()
        if len(alive_indices) == 0:
            return

        x_coords = self.registry.data[self.registry.X, alive_indices].long().clamp(0, self.grid.W - 1)
        y_coords = self.registry.data[self.registry.Y, alive_indices].long().clamp(0, self.grid.H - 1)
        h_rates = self.grid.grid[1, y_coords, x_coords]

        self.registry.data[self.registry.HP, alive_indices] += h_rates
        positive_h_mask = h_rates > 0.0
        if positive_h_mask.any():
            self.registry.data[self.registry.HP_GAINED, alive_indices[positive_h_mask]] += h_rates[positive_h_mask]

        negative_h_mask = h_rates < 0.0
        if negative_h_mask.any():
            for slot_idx, x, y in zip(
                alive_indices[negative_h_mask].tolist(),
                x_coords[negative_h_mask].tolist(),
                y_coords[negative_h_mask].tolist(),
            ):
                self._record_death_context(
                    int(slot_idx),
                    death_reason="poison_zone",
                    catastrophe_id=self._active_catastrophe_id(),
                    zone_id=self.grid.find_hzone_at(int(x), int(y)),
                )

        metab = self.registry.data[self.registry.METABOLISM_RATE, alive_indices]
        mass = self.registry.data[self.registry.MASS, alive_indices]
        effective_metab = (metab * self.metabolism_multiplier) + (mass * self.mass_metabolism_burden)
        self.registry.data[self.registry.HP, alive_indices] -= effective_metab

        positive_metab_mask = effective_metab > 0.0
        if positive_metab_mask.any():
            for slot_idx in alive_indices[positive_metab_mask].tolist():
                self._record_death_context(
                    int(slot_idx),
                    death_reason="metabolism_death",
                    catastrophe_id=self._active_catastrophe_id(),
                )

    def process_deaths(self):
        """Clamp HP into the legal range and retire any slots that have crossed the death boundary."""
        alive_indices = self.registry.get_alive_indices()
        if len(alive_indices) == 0:
            return []

        hp = self.registry.data[self.registry.HP, alive_indices]
        hp_max = self.registry.data[self.registry.HP_MAX, alive_indices]
        hp_clamped = torch.minimum(torch.clamp_min(hp, 0.0), hp_max)
        self.registry.data[self.registry.HP, alive_indices] = hp_clamped

        deaths: list[int] = []
        for idx_int in [int(idx) for idx in alive_indices[hp_clamped <= 0.0].tolist()]:
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

