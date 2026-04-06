import random

import torch

from ..config_bridge import cfg
from ..learning.ppo import PPO


class Evolution:
    def __init__(self, registry):
        self.registry = registry

    def process_deaths(self, deaths, ppo: PPO, death_tick: int):
        """
        Process dead agents after terminal transitions have already been stored.

        Canonical UID finalization stays adjacent to PPO cleanup so slot reuse
        cannot leak optimizer or rollout ownership across organisms.
        """

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

    def _apply_policy_noise(self, brain: torch.nn.Module):
        if cfg.EVOL.POLICY_NOISE_SD <= 0:
            return
        with torch.no_grad():
            for param in brain.parameters():
                noise = torch.randn_like(param) * cfg.EVOL.POLICY_NOISE_SD
                param.add_(noise)

    def _mutate_traits(self, parent_idx):
        traits = {
            "mass": self.registry.data[self.registry.MASS, parent_idx].item(),
            "vision": self.registry.data[self.registry.VISION, parent_idx].item(),
            "hp_max": self.registry.data[self.registry.HP_MAX, parent_idx].item(),
            "metab": self.registry.data[self.registry.METABOLISM_RATE, parent_idx].item(),
        }

        noise_sd = cfg.EVOL.GENOME_NOISE_SD
        traits["mass"] += random.gauss(0, noise_sd["mass"])
        traits["vision"] += random.gauss(0, noise_sd["vision"])
        traits["hp_max"] += random.gauss(0, noise_sd["hp_max"])
        traits["metab"] += random.gauss(0, noise_sd["metab"])

        if random.random() < cfg.EVOL.RARE_MUT_PROB:
            mut_trait = random.choice(["mass", "vision", "hp_max"])
            mult = random.lognormvariate(0, cfg.EVOL.RARE_MUT_SIGMA)
            traits[mut_trait] *= mult

        clamp = cfg.TRAITS.CLAMP
        traits["mass"] = max(clamp.mass[0], min(clamp.mass[1], traits["mass"]))
        traits["vision"] = max(clamp.vision[0], min(clamp.vision[1], traits["vision"]))
        traits["hp_max"] = max(clamp.hp_max[0], min(clamp.hp_max[1], traits["hp_max"]))
        traits["metab"] = max(clamp.metab[0], min(clamp.metab[1], traits["metab"]))
        return traits

    def _find_free_cell(self, grid):
        for _ in range(100):
            x = random.randint(1, grid.W - 2)
            y = random.randint(1, grid.H - 2)
            if not grid.is_wall(x, y) and grid.get_agent_at(x, y) == -1:
                return x, y
        return None, None

