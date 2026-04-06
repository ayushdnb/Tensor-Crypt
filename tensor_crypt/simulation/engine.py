"""Core simulation engine for Tensor Crypt."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from ..config_bridge import cfg
from ..population.respawn_controller import RespawnController
from .catastrophes import CatastropheManager


class Engine:
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
        self.catastrophes = CatastropheManager(
            grid=self.grid,
            registry=self.registry,
            physics=self.physics,
            perception=self.perception,
            respawn_controller=self.respawn_controller,
            logger=self.logger,
        )
        self.tick = 0
        self.registry.tick_counter = 0

    def _batched_brain_forward(self, obs: dict, alive_indices: torch.Tensor):
        batch_size = len(alive_indices)
        all_logits = torch.zeros(batch_size, cfg.BRAIN.ACTION_DIM, device=cfg.SIM.DEVICE)
        all_values = torch.zeros(batch_size, cfg.BRAIN.VALUE_DIM, device=cfg.SIM.DEVICE)

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
        for i, idx_int in enumerate(alive_indices.cpu().numpy()):
            slot_idx = int(idx_int)
            uid = self.registry.get_uid_for_slot(slot_idx)
            if uid == -1:
                raise AssertionError(f"Alive slot {slot_idx} has no canonical UID during transition storage")

            agent_obs = {key: value[i] for key, value in obs.items()}
            self.ppo.store_transition_for_slot(
                self.registry,
                slot_idx,
                agent_obs,
                actions_compact[i],
                log_probs[i],
                rewards[i],
                values[i],
                dones[i],
            )

            if float(dones[i].detach().item()) >= 0.5:
                self.ppo.finalize_terminal_uid(uid)

    def _stage_bootstrap_state_for_update(self) -> None:
        final_alive_indices = self.registry.get_alive_indices()
        if len(final_alive_indices) == 0:
            return

        final_obs_batch = self.perception.build_observations(final_alive_indices)
        for i, idx_int in enumerate(final_alive_indices.cpu().numpy()):
            slot_idx = int(idx_int)
            uid = self.registry.get_uid_for_slot(slot_idx)
            if uid == -1:
                raise AssertionError(f"Alive slot {slot_idx} has no UID during PPO bootstrap capture")
            self.ppo.stage_bootstrap_for_uid(
                uid,
                {key: value[i] for key, value in final_obs_batch.items()},
                torch.tensor(0.0, device=cfg.SIM.DEVICE),
                finalization_kind="active_bootstrap",
            )

    def _advance_empty_tick(self) -> None:
        self.tick += 1
        self.registry.tick_counter = self.tick
        self.respawn_controller.step(self.tick, self.registry, self.grid, self.logger)

    def _maybe_run_ppo_update(self) -> None:
        if not self.ppo.should_update(self.tick):
            return

        self._stage_bootstrap_state_for_update()
        update_stats_list = self.ppo.update(self.registry, tick=self.tick)
        self.logger.log_ppo_update(self.tick, update_stats_list)

    def _maybe_save_snapshots(self) -> None:
        if self.tick > 0 and self.tick % cfg.LOG.SNAPSHOT_EVERY == 0:
            self.logger.log_agent_snapshot(self.tick, self.registry)
            self.logger.log_heatmap_snapshot(self.tick, self.grid)
            self.logger.log_brains(self.tick, self.registry)

    def _maybe_print_tick_progress(self) -> None:
        if self.tick % cfg.LOG.LOG_TICK_EVERY == 0:
            print(f"Tick {self.tick}: {self.registry.get_num_alive()} alive")

    def step(self) -> None:
        self.registry.tick_counter = self.tick

        # Prompt 6 deterministic scheduling boundary:
        # 1) expire / trigger catastrophes
        # 2) repaint baseline h-zones
        # 3) layer reversible catastrophe field + runtime modifiers
        self.catastrophes.pre_tick(self.tick)
        self.grid.paint_hzones()
        self.catastrophes.apply_world_overrides(self.tick)

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
        self.logger.log_tick_summary(
            self.tick,
            self.registry,
            physics_stats,
            catastrophe_state=self.catastrophes.build_status(self.tick),
        )

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