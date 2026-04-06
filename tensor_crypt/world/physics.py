"""
Handles all agent movement, collision detection, and environmental interactions.

This module is the referee of the simulation. It takes chosen actions from each
agent and determines the physically valid outcomes without owning higher-level
training or respawn policy.

Critical sequencing boundary:
- this module only resolves movement and environment effects for agents already
  alive during the tick
- respawn happens later in the engine, after deaths are processed
"""

import torch

from ..config_bridge import cfg


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

    def step(self, actions: torch.Tensor) -> dict:
        self.collision_log = []
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

            if (tx, ty) not in move_approved:
                move_approved[(tx, ty)] = []
            move_approved[(tx, ty)].append(idx)

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

    def _handle_wall_collision(self, idx: int):
        mass = self.registry.data[self.registry.MASS, idx]
        damage = mass * cfg.PHYS.K_WALL_PENALTY
        self.registry.data[self.registry.HP, idx] -= damage
        self.registry.data[self.registry.HP_LOST_PHYSICS, idx] += damage
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
            }
        )

    def _handle_ram(self, rammer_idx: int, target_idx: int):
        mass_a = self.registry.data[self.registry.MASS, rammer_idx]

        damage_a = mass_a * cfg.PHYS.K_RAM_PENALTY
        self.registry.data[self.registry.HP, rammer_idx] -= damage_a
        self.registry.data[self.registry.HP_LOST_PHYSICS, rammer_idx] += damage_a

        damage_b = mass_a * cfg.PHYS.K_IDLE_HIT_PENALTY
        self.registry.data[self.registry.HP, target_idx] -= damage_b
        self.registry.data[self.registry.HP_LOST_PHYSICS, target_idx] += damage_b

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
            }
        )

    def _resolve_contest(self, contenders: list) -> int:
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

        for _, idx in strengths:
            damage = cfg.PHYS.K_WINNER_DAMAGE if idx == winner else cfg.PHYS.K_LOSER_DAMAGE
            self.registry.data[self.registry.HP, idx] -= damage
            self.registry.data[self.registry.HP_LOST_PHYSICS, idx] += damage

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
            metab = self.registry.data[self.registry.METABOLISM_RATE, idx_int]
            self.registry.data[self.registry.HP, idx_int] -= metab

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
                self.registry.mark_dead(idx_int, self.grid)
                deaths.append(idx_int)

        return deaths
