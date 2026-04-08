"""Core simulation engine for Tensor Crypt.

The engine owns tick ordering, PPO transition capture boundaries, and the
point where checkpoint/telemetry side effects observe the simulation state.
Per-slot brains intentionally remain independent modules, so forward batching is
kept conservative here rather than forcing an unsafe family-stacking scheme
that could blur UID ownership or checkpoint topology semantics.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributions import Categorical

from ..checkpointing.atomic_checkpoint import manifest_path_for
from ..checkpointing.runtime_checkpoint import capture_runtime_checkpoint, save_runtime_checkpoint
from ..config_bridge import cfg
from ..population.respawn_controller import RespawnController
from .catastrophes import CatastropheManager


SUPPORTED_PPO_REWARD_FORMS = frozenset({"sq_health_ratio"})
SUPPORTED_PPO_REWARD_GATE_MODES = frozenset({"off", "hp_ratio_min", "hp_abs_min"})
HP_RATIO_DENOM_EPS = 1e-6


def _ppo_reward_form() -> str:
    return str(cfg.PPO.REWARD_FORM).lower()


def _ppo_reward_gate_mode() -> str:
    return str(cfg.PPO.REWARD_GATE_MODE).lower()


def validate_ppo_reward_config() -> None:
    reward_form = _ppo_reward_form()
    if reward_form not in SUPPORTED_PPO_REWARD_FORMS:
        supported = ", ".join(sorted(SUPPORTED_PPO_REWARD_FORMS))
        raise ValueError(f"PPO.REWARD_FORM must be one of {{{supported}}}, got {cfg.PPO.REWARD_FORM!r}")

    gate_mode = _ppo_reward_gate_mode()
    if gate_mode not in SUPPORTED_PPO_REWARD_GATE_MODES:
        supported = ", ".join(sorted(SUPPORTED_PPO_REWARD_GATE_MODES))
        raise ValueError(f"PPO.REWARD_GATE_MODE must be one of {{{supported}}}, got {cfg.PPO.REWARD_GATE_MODE!r}")

    threshold = float(cfg.PPO.REWARD_GATE_THRESHOLD)
    if not math.isfinite(threshold):
        raise ValueError("PPO.REWARD_GATE_THRESHOLD must be finite")

    below_gate_value = float(cfg.PPO.REWARD_BELOW_GATE_VALUE)
    if not math.isfinite(below_gate_value):
        raise ValueError("PPO.REWARD_BELOW_GATE_VALUE must be finite")

    if gate_mode == "hp_ratio_min":
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("PPO.REWARD_GATE_THRESHOLD must be within [0.0, 1.0] when PPO.REWARD_GATE_MODE='hp_ratio_min'")
        return

    if threshold < 0.0:
        raise ValueError(f"PPO.REWARD_GATE_THRESHOLD must be >= 0.0 when PPO.REWARD_GATE_MODE={cfg.PPO.REWARD_GATE_MODE!r}")


def compute_ppo_reward_tensor(hp: torch.Tensor, hp_max: torch.Tensor) -> torch.Tensor:
    reward_form = _ppo_reward_form()
    if reward_form == "sq_health_ratio":
        hp_ratio = torch.clamp(hp / hp_max.clamp_min(HP_RATIO_DENOM_EPS), 0.0, 1.0)
        base_reward = hp_ratio.square()
    else:
        supported = ", ".join(sorted(SUPPORTED_PPO_REWARD_FORMS))
        raise ValueError(f"Unsupported PPO.REWARD_FORM {cfg.PPO.REWARD_FORM!r}; expected one of {{{supported}}}")

    gate_mode = _ppo_reward_gate_mode()
    if gate_mode == "off":
        return base_reward

    threshold = float(cfg.PPO.REWARD_GATE_THRESHOLD)
    if gate_mode == "hp_ratio_min":
        gate_mask = hp_ratio >= threshold
    elif gate_mode == "hp_abs_min":
        gate_mask = hp >= threshold
    else:
        supported = ", ".join(sorted(SUPPORTED_PPO_REWARD_GATE_MODES))
        raise ValueError(f"Unsupported PPO.REWARD_GATE_MODE {cfg.PPO.REWARD_GATE_MODE!r}; expected one of {{{supported}}}")

    return torch.where(
        gate_mask,
        base_reward,
        base_reward.new_full(base_reward.shape, float(cfg.PPO.REWARD_BELOW_GATE_VALUE)),
    )


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
        self.last_runtime_checkpoint_tick = -1
        self.last_runtime_checkpoint_path: str | None = None
        self._runtime_checkpoint_view = SimpleNamespace(
            registry=self.registry,
            grid=self.grid,
            ppo=self.ppo,
            engine=self,
        )
        self._actions_sparse = torch.zeros(
            self.registry.max_agents,
            device=self.registry.device,
            dtype=torch.long,
        )
        self._bootstrap_not_done = torch.tensor(
            0.0,
            device=self.registry.device,
            dtype=self.registry.data.dtype,
        )

        if getattr(self.logger, "bootstrap_initial_population", None) is not None:
            self.logger.bootstrap_initial_population(self.registry)

    def _batched_brain_forward(self, obs: dict, alive_indices: torch.Tensor):
        """Run inference for the currently alive slots without crossing UID ownership boundaries."""
        batch_size = len(alive_indices)
        all_logits = torch.zeros(batch_size, cfg.BRAIN.ACTION_DIM, device=cfg.SIM.DEVICE)
        all_values = torch.zeros(batch_size, cfg.BRAIN.VALUE_DIM, device=cfg.SIM.DEVICE)

        for i, agent_idx in enumerate(alive_indices):
            idx_int = int(agent_idx.item())
            brain = self.registry.brains[idx_int]

            if brain is None:
                all_logits[i, 0] = 1.0
                continue

            # Each live UID owns a distinct module/optimizer pair, so inference stays
            # per-brain even though the surrounding tensors are batched by slot.
            agent_obs = {key: value[i : i + 1] for key, value in obs.items()}
            logits, value = brain(agent_obs)
            all_logits[i] = logits.squeeze(0)
            all_values[i] = value.squeeze(0)

        return all_logits, all_values

    def _sample_actions(self, obs: dict, alive_indices: torch.Tensor):
        with torch.inference_mode():
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
        alive_slots = [int(slot_idx) for slot_idx in alive_indices.detach().cpu().tolist()]
        done_values = dones.detach().cpu().tolist()

        for i, slot_idx in enumerate(alive_slots):
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

            if float(done_values[i]) >= 0.5:
                self.ppo.finalize_terminal_uid(uid)

    def _stage_bootstrap_state_for_update(self) -> None:
        final_alive_indices = self.registry.get_alive_indices()
        if len(final_alive_indices) == 0:
            return

        final_obs_batch = self.perception.build_observations(final_alive_indices)
        alive_slots = [int(slot_idx) for slot_idx in final_alive_indices.detach().cpu().tolist()]

        for i, slot_idx in enumerate(alive_slots):
            uid = self.registry.get_uid_for_slot(slot_idx)
            if uid == -1:
                raise AssertionError(f"Alive slot {slot_idx} has no UID during PPO bootstrap capture")
            self.ppo.stage_bootstrap_for_uid(
                uid,
                {key: value[i] for key, value in final_obs_batch.items()},
                self._bootstrap_not_done,
                finalization_kind="active_bootstrap",
            )

    def _compute_ppo_rewards(self, alive_indices: torch.Tensor) -> torch.Tensor:
        return compute_ppo_reward_tensor(
            self.registry.data[self.registry.HP, alive_indices],
            self.registry.data[self.registry.HP_MAX, alive_indices],
        )

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

    def _checkpoint_dir(self) -> Path:
        return Path(self.logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME

    def _checkpoint_path_for_tick(self, tick: int) -> Path:
        filename = f"{cfg.CHECKPOINT.FILENAME_PREFIX}{int(tick):08d}{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"
        return self._checkpoint_dir() / filename

    def _prune_old_runtime_checkpoints(self) -> None:
        keep_last = int(cfg.CHECKPOINT.KEEP_LAST)
        if keep_last <= 0:
            return

        checkpoint_dir = self._checkpoint_dir()
        if not checkpoint_dir.exists():
            return

        pattern = f"{cfg.CHECKPOINT.FILENAME_PREFIX}*{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"
        bundles = sorted(checkpoint_dir.glob(pattern))
        for bundle_path in bundles[:-keep_last]:
            bundle_path.unlink(missing_ok=True)
            manifest_path_for(bundle_path).unlink(missing_ok=True)

    def _maybe_save_runtime_checkpoint(self) -> None:
        """Publish a post-tick checkpoint only after physics, deaths, births, and PPO state settle."""
        interval = int(cfg.CHECKPOINT.SAVE_EVERY_TICKS)
        if not cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS or interval <= 0:
            return
        if self.tick <= 0 or self.tick % interval != 0:
            return

        checkpoint_path = self._checkpoint_path_for_tick(self.tick)
        bundle = capture_runtime_checkpoint(self._runtime_checkpoint_view)
        save_runtime_checkpoint(checkpoint_path, bundle)
        self.last_runtime_checkpoint_tick = int(self.tick)
        self.last_runtime_checkpoint_path = str(checkpoint_path)
        self._prune_old_runtime_checkpoints()

    def _maybe_print_tick_progress(self) -> None:
        if self.tick % cfg.LOG.LOG_TICK_EVERY == 0:
            print(f"Tick {self.tick}: {self.registry.get_num_alive()} alive")

    def _log_tick_summary(self, *, tick: int, catastrophe_state: dict, physics_stats: dict, births_this_tick: int, deaths_this_tick: int) -> None:
        self.logger.log_tick_summary(
            tick,
            self.registry,
            physics_stats,
            catastrophe_state=catastrophe_state,
            births_this_tick=births_this_tick,
            deaths_this_tick=deaths_this_tick,
            reproduction_disabled=not self.respawn_controller.reproduction_enabled_override,
            floor_recovery_active=self.registry.get_num_alive() < cfg.RESPAWN.POPULATION_FLOOR,
            ppo=self.ppo,
        )

    def _advance_empty_tick(self) -> None:
        catastrophe_state = self.catastrophes.build_status(self.tick)
        self.physics.set_catastrophe_state(catastrophe_state)
        self.respawn_controller.step(self.tick, self.registry, self.grid, self.logger)
        self._log_tick_summary(
            tick=self.tick,
            catastrophe_state=catastrophe_state,
            physics_stats={"wall_collisions": 0, "rams": 0, "contests": 0},
            births_this_tick=self.logger.get_tick_birth_count(self.tick),
            deaths_this_tick=self.logger.get_tick_death_count(self.tick),
        )
        self.tick += 1
        self.registry.tick_counter = self.tick

    def step(self) -> None:
        self.registry.tick_counter = self.tick

        # Prompt 6 deterministic scheduling boundary:
        # 1) expire / trigger catastrophes
        # 2) repaint baseline h-zones
        # 3) layer reversible catastrophe field + runtime modifiers
        self.catastrophes.pre_tick(self.tick)
        self.grid.paint_hzones()
        self.catastrophes.apply_world_overrides(self.tick)

        catastrophe_state = self.catastrophes.build_status(self.tick)
        self.physics.set_catastrophe_state(catastrophe_state)

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

        if cfg.SIM.REUSE_ACTION_BUFFER:
            actions_sparse = self._actions_sparse
            actions_sparse.zero_()
        else:
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
        self.logger.note_catastrophe_exposure(self.registry, catastrophe_state)

        rewards = self._compute_ppo_rewards(alive_indices)

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

        for dead_slot in deaths:
            self.logger.finalize_death(
                tick=self.tick,
                slot_idx=int(dead_slot),
                registry=self.registry,
                ppo=self.ppo,
                death_context=self.physics.consume_death_context(int(dead_slot)),
            )

        self.evolution.process_deaths(deaths, self.ppo, death_tick=self.tick)
        self.respawn_controller.step(self.tick, self.registry, self.grid, self.logger)
        self.registry.check_invariants(self.grid)

        self._log_tick_summary(
            tick=self.tick,
            catastrophe_state=catastrophe_state,
            physics_stats=physics_stats,
            births_this_tick=self.logger.get_tick_birth_count(self.tick),
            deaths_this_tick=self.logger.get_tick_death_count(self.tick),
        )

        self.tick += 1
        self.registry.tick_counter = self.tick

        self._maybe_run_ppo_update()
        self._maybe_save_snapshots()
        self._maybe_save_runtime_checkpoint()
        self._maybe_print_tick_progress()

