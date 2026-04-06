"""Core simulation engine for Tensor Crypt.

This module is the sequencing authority for one simulation tick. It remains
behavior-sensitive because subtle reordering here can alter training data,
death timing, respawn timing, logging payloads, and viewer state.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from ..config_bridge import cfg
from ..population.respawn_controller import RespawnController


class Engine:
    """
    High-level orchestrator for the simulation.

    The engine holds stable references to the major subsystems and executes the
    tick pipeline in the exact order expected by the existing simulation.
    """

    def __init__(
        self,
        grid,
        registry,
        physics,
        perception,
        ppo,
        evolution,
        logger,
    ):
        self.grid = grid
        self.registry = registry
        self.physics = physics
        self.perception = perception
        self.ppo = ppo
        self.evolution = evolution
        self.logger = logger

        self.respawn_controller = RespawnController(self.evolution)
        self.tick = 0
        self.registry.tick_counter = 0

        self.last_obs_dict = {}
        self.last_dones_dict = {}

    def _batched_brain_forward(self, obs: dict, alive_indices: torch.Tensor):
        batch_size = len(alive_indices)
        all_logits = torch.zeros(batch_size, 9, device=cfg.SIM.DEVICE)
        all_values = torch.zeros(batch_size, 1, device=cfg.SIM.DEVICE)

        for i, agent_idx in enumerate(alive_indices):
            idx_int = int(agent_idx.item())
            brain = self.registry.brains[idx_int]

            if brain is None:
                all_logits[i, 0] = 1.0
                continue

            brain.eval()
            agent_obs = {key: value[i : i + 1] for key, value in obs.items()}

            logits, value = brain(agent_obs)
            all_logits[i] = logits.squeeze(0)
            all_values[i] = value.squeeze(0)

        return all_logits, all_values

    def _sample_actions(self, obs: dict, alive_indices: torch.Tensor):
        with torch.no_grad():
            if len(alive_indices) == 0:
                return None

            batched_logits, batched_values = self._batched_brain_forward(obs, alive_indices)
            dist = Categorical(logits=batched_logits)
            batched_actions = dist.sample()
            batched_log_probs = dist.log_prob(batched_actions)

        if batched_actions.numel() == 0:
            return None

        return batched_logits, batched_values, batched_actions, batched_log_probs

    def _store_transitions(
        self,
        alive_indices: torch.Tensor,
        obs: dict,
        actions_compact: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.last_obs_dict.clear()
        self.last_dones_dict.clear()

        for i, idx_int in enumerate(alive_indices.cpu().numpy()):
            uid = self.registry.get_uid_for_slot(int(idx_int))
            if uid == -1:
                raise AssertionError(f"Alive slot {idx_int} has no canonical UID during transition storage")

            agent_obs = {key: value[i] for key, value in obs.items()}

            self.ppo.store_transition_for_slot(
                self.registry,
                int(idx_int),
                agent_obs,
                actions_compact[i],
                log_probs[i],
                rewards[i],
                values[i],
                dones[i],
            )

            self.last_obs_dict[uid] = agent_obs
            self.last_dones_dict[uid] = dones[i]

    def _advance_empty_tick(self) -> None:
        self.tick += 1
        self.registry.tick_counter = self.tick
        self.respawn_controller.step(self.tick, self.registry, self.grid, self.logger)

    def _maybe_run_ppo_update(self) -> None:
        if not self.ppo.should_update(self.tick):
            return

        next_obs_dict = {}
        next_dones_dict = {}

        final_alive_indices = self.registry.get_alive_indices()
        if len(final_alive_indices) > 0:
            final_obs_batch = self.perception.build_observations(final_alive_indices)

            for i, idx_int in enumerate(final_alive_indices.cpu().numpy()):
                uid = self.registry.get_uid_for_slot(int(idx_int))
                if uid == -1:
                    raise AssertionError(f"Alive slot {idx_int} has no UID during PPO bootstrap capture")
                next_obs_dict[uid] = {key: value[i] for key, value in final_obs_batch.items()}
                next_dones_dict[uid] = torch.tensor(0.0, device=cfg.SIM.DEVICE)

        update_stats_list = self.ppo.update(
            self.registry,
            self.perception,
            next_obs_dict,
            next_dones_dict,
        )

        self.logger.log_ppo_update(self.tick, update_stats_list)

        if update_stats_list:
            print(f"Tick {self.tick}: PPO Update performed for {len(update_stats_list)} agents.")
            for stats in update_stats_list[:2]:
                print(
                    f"  UID {stats['agent_uid']} (slot {stats['agent_slot']}): "
                    f"PLoss={stats['policy_loss']:.3f}, "
                    f"VLoss={stats['value_loss']:.3f}"
                )

    def _maybe_save_snapshots(self) -> None:
        if self.tick > 0 and self.tick % cfg.LOG.SNAPSHOT_EVERY == 0:
            print(f"--- Tick {self.tick}: Saving snapshot ---")
            self.logger.log_agent_snapshot(self.tick, self.registry)
            self.logger.log_heatmap_snapshot(self.tick, self.grid)
            self.logger.log_brains(self.tick, self.registry)

    def _maybe_print_tick_progress(self) -> None:
        if self.tick % cfg.LOG.LOG_TICK_EVERY == 0:
            print(f"Tick {self.tick}: {self.registry.get_num_alive()} alive")

    def step(self) -> None:
        self.registry.tick_counter = self.tick
        self.grid.paint_hzones()
        alive_indices = self.registry.get_alive_indices()

        if len(alive_indices) == 0:
            self._advance_empty_tick()
            return

        obs = self.perception.build_observations(alive_indices)
        sampled = self._sample_actions(obs, alive_indices)

        if sampled is None:
            self._advance_empty_tick()
            return

        _, values, actions_compact, log_probs = sampled

        actions_sparse = torch.zeros(
            self.registry.max_agents,
            device=self.registry.device,
            dtype=torch.long,
        )
        actions_sparse[alive_indices] = actions_compact.long()
        self.registry.data[self.registry.LAST_ACTION, alive_indices] = actions_compact.float()

        physics_stats = self.physics.step(actions_sparse)
        self.logger.log_physics_events(self.tick, self.physics.collision_log)

        self.physics.apply_environment_effects()
        self.logger.log_tick_summary(self.tick, self.registry, physics_stats)

        current_hp = self.registry.data[self.registry.HP, alive_indices]
        max_hp = self.registry.data[self.registry.HP_MAX, alive_indices]
        rewards = (current_hp / (max_hp + 1e-6)) ** 2

        deaths = self.physics.process_deaths()
        dones = (self.registry.data[self.registry.ALIVE, alive_indices] == 0.0).float()

        self._store_transitions(
            alive_indices,
            obs,
            actions_compact,
            log_probs,
            rewards,
            values,
            dones,
        )

        self.evolution.process_deaths(deaths, self.ppo, death_tick=self.tick)
        self.respawn_controller.step(self.tick, self.registry, self.grid, self.logger)
        self.registry.check_invariants(self.grid)

        self.tick += 1
        self.registry.tick_counter = self.tick

        self._maybe_run_ppo_update()
        self._maybe_save_snapshots()
        self._maybe_print_tick_progress()
