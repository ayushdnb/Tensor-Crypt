"""Evolution helpers used by death finalization and birth mutation paths."""

from __future__ import annotations

import random

import torch

from ..config_bridge import cfg
from ..learning.ppo import PPO


class Evolution:
    """Small coordination helpers shared by death and reproduction flows."""

    def __init__(self, registry):
        self.registry = registry

    def process_deaths(self, deaths, ppo: PPO, death_tick: int):
        if len(deaths) == 0:
            return
        for dead_idx in deaths:
            reward_sum = self.registry.data[self.registry.HP_GAINED, dead_idx]
            self.registry.fitness[dead_idx] = self.registry.fitness[dead_idx] * cfg.EVOL.FITNESS_DECAY + reward_sum
            dead_uid = self.registry.get_uid_for_slot(dead_idx)
            if dead_uid == -1:
                raise AssertionError(f"Dead slot {dead_idx} lost its UID binding before terminal finalization")
            ppo.clear_agent_state(dead_uid)
            self.registry.finalize_death(dead_idx, death_tick, assert_after=False)
        self.registry.assert_identity_invariants()

    def apply_policy_noise(self, brain: torch.nn.Module, sigma: float | None = None) -> None:
        noise_sd = float(cfg.EVOL.POLICY_NOISE_SD if sigma is None else sigma)
        if noise_sd <= 0.0:
            return
        with torch.no_grad():
            for param in brain.parameters():
                param.add_(torch.randn_like(param) * noise_sd)

    def _apply_policy_noise(self, brain: torch.nn.Module, sigma: float | None = None) -> None:
        """Compatibility alias for older call sites."""
        self.apply_policy_noise(brain, sigma=sigma)

    def find_free_cell(self, grid) -> tuple[int | None, int | None]:
        for _ in range(100):
            x = random.randint(1, grid.W - 2)
            y = random.randint(1, grid.H - 2)
            if not grid.is_wall(x, y) and grid.get_agent_at(x, y) == -1:
                return x, y
        return None, None

    def _find_free_cell(self, grid) -> tuple[int | None, int | None]:
        """Compatibility alias for older call sites."""
        return self.find_free_cell(grid)
