import copy
from typing import Optional

import torch

from ..config_bridge import cfg
from ..telemetry.data_logger import DataLogger


class RespawnController:
    """
    Manages timed respawning of new agents.

    This controller intentionally owns timing and slot reuse policy, while the
    `Evolution` module owns mutation helpers. Keeping that split prevents engine
    orchestration from needing to know mutation details.
    """

    def __init__(self, evolution):
        self.evolution = evolution
        self.respawn_period = cfg.RESPAWN.RESPAWN_PERIOD
        self.max_spawns_per_cycle = cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE
        self.last_respawn_tick = 0

    def step(self, tick: int, registry, grid, logger: Optional[DataLogger] = None):
        num_alive = registry.get_num_alive()
        floor = cfg.RESPAWN.POPULATION_FLOOR
        ceiling = cfg.RESPAWN.POPULATION_CEILING

        if num_alive >= ceiling:
            return

        is_below_floor = num_alive < floor
        is_timer_ready = tick - self.last_respawn_tick >= self.respawn_period
        if not is_below_floor and not is_timer_ready:
            return

        pending_finalization = torch.nonzero(
            (registry.data[registry.ALIVE] <= 0.5) & (registry.slot_uid >= 0),
            as_tuple=False,
        ).squeeze(-1)
        if pending_finalization.numel() > 0:
            raise AssertionError(
                f"Respawn cannot reuse slots before UID finalization: pending slots {pending_finalization.tolist()}"
            )

        self.last_respawn_tick = tick

        dead_slots = torch.nonzero(
            (registry.data[registry.ALIVE] <= 0.5) & (registry.slot_uid < 0),
            as_tuple=False,
        ).squeeze(-1)
        if dead_slots.numel() == 0:
            return

        num_room_before_ceil = ceiling - num_alive
        num_to_spawn = min(self.max_spawns_per_cycle, dead_slots.numel(), num_room_before_ceil)
        num_to_spawn = max(0, num_to_spawn)
        if num_to_spawn == 0:
            return

        alive_indices = registry.get_alive_indices()
        alive_positions = None
        if alive_indices.numel() > 0:
            alive_positions = registry.data[[registry.X, registry.Y]][:, alive_indices].T

        spawns = 0
        for slot_tensor in dead_slots:
            if spawns >= num_to_spawn:
                break

            slot_idx = int(slot_tensor.item())
            if registry.get_uid_for_slot(slot_idx) != -1:
                raise AssertionError(f"Respawn target slot {slot_idx} is still bound to a UID")

            spawn_x, spawn_y = self.evolution._find_free_cell(grid)
            if spawn_x is None:
                continue

            if alive_indices.numel() == 0:
                parent_slot_idx = -1
                parent_uid = -1
                traits = {
                    "mass": cfg.TRAITS.INIT.mass,
                    "vision": cfg.TRAITS.INIT.vision,
                    "hp_max": cfg.TRAITS.INIT.hp_max,
                    "metab": cfg.TRAITS.INIT.metab,
                }
            else:
                spawn_pos = torch.tensor([spawn_x, spawn_y], device=registry.device, dtype=torch.float32)
                distances = torch.norm(alive_positions - spawn_pos, dim=1)
                nearest_parent_idx_in_list = torch.argmin(distances)
                parent_slot_idx = int(alive_indices[nearest_parent_idx_in_list].item())
                parent_uid = registry.get_uid_for_slot(parent_slot_idx)
                traits = self.evolution._mutate_traits(parent_slot_idx)

                parent_brain = registry.brains[parent_slot_idx]
                child_brain = registry.brains[slot_idx]
                if parent_brain is not None:
                    if child_brain is None:
                        child_brain = copy.deepcopy(parent_brain).to(registry.device)
                        registry.brains[slot_idx] = child_brain
                    else:
                        child_brain.load_state_dict(parent_brain.state_dict())
                    self.evolution._apply_policy_noise(child_brain)

            child_uid = registry.spawn_agent(
                slot_idx,
                spawn_x,
                spawn_y,
                parent_uid,
                grid,
                traits,
                tick_born=tick,
                assign_new_identity=True,
            )
            if logger:
                logger.log_spawn_event(
                    tick=tick,
                    child_slot=slot_idx,
                    parent_slot=parent_slot_idx,
                    child_uid=child_uid,
                    parent_uid=parent_uid,
                    traits=traits,
                )
            spawns += 1
