"""Core simulation engine for Tensor Crypt.

The engine owns tick ordering, PPO transition capture boundaries, and the
point where checkpoint/telemetry side effects observe the simulation state. The
canonical inference path remains a per-brain loop keyed to slot/UID ownership,
with an experimental opt-in same-family fast path that only stacks module state
ephemerally for inference.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.distributions import Categorical

try:
    from torch.func import functional_call, stack_module_state, vmap

    _HAS_TORCH_FUNC = True
except Exception:
    functional_call = None
    stack_module_state = None
    vmap = None
    _HAS_TORCH_FUNC = False

from ..checkpointing.atomic_checkpoint import manifest_path_for
from ..checkpointing.runtime_checkpoint import capture_runtime_checkpoint, save_runtime_checkpoint
from ..config_bridge import cfg
from ..population.respawn_controller import RespawnController
from .catastrophes import CatastropheManager


SUPPORTED_PPO_REWARD_FORMS = frozenset({"sq_health_ratio"})
SUPPORTED_PPO_REWARD_GATE_MODES = frozenset({"off", "hp_ratio_min", "hp_abs_min"})
HP_RATIO_DENOM_EPS = 1e-6
SAVE_REASON_SCHEDULED_TICK = "scheduled_tick"
SAVE_REASON_SCHEDULED_WALLCLOCK = "scheduled_wallclock"
SAVE_REASON_SHUTDOWN = "shutdown"
SAVE_REASON_MANUAL_RESERVED = "manual_future_reserved"


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
        *,
        bootstrap_initial_population: bool = True,
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
        self.last_runtime_checkpoint_reason: str | None = None
        self.last_runtime_checkpoint_wallclock_time: float | None = None
        self._wallclock_autosave_started_at = time.monotonic()
        self._checkpoint_capture_reason: str | None = None
        self._checkpoint_capture_path: str | None = None
        self._runtime_checkpoint_view = SimpleNamespace(
            registry=self.registry,
            grid=self.grid,
            ppo=self.ppo,
            engine=self,
            data_logger=self.logger,
            run_dir=str(getattr(self.logger, "run_dir", "")),
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
        self.last_inference_path_stats = {
            "loop_slots": 0,
            "vmap_slots": 0,
            "family_loop_buckets": 0,
            "family_vmap_buckets": 0,
        }

        if bootstrap_initial_population and getattr(self.logger, "bootstrap_initial_population", None) is not None:
            self.logger.bootstrap_initial_population(self.registry)

    @staticmethod
    def _alive_slots_from_indices(alive_indices: torch.Tensor) -> list[int]:
        if alive_indices.numel() == 0:
            return []
        return [int(slot_idx) for slot_idx in alive_indices.detach().cpu().tolist()]

    @staticmethod
    def _brain_topology_signature(brain) -> tuple | None:
        getter = getattr(brain, "get_topology_signature", None)
        if getter is None:
            return None
        return tuple(getter())

    @staticmethod
    def _slice_obs_batch(obs: dict, batch_positions: list[int]) -> dict:
        device = next(iter(obs.values())).device
        index = torch.tensor(batch_positions, device=device, dtype=torch.long)
        return {key: value.index_select(0, index) for key, value in obs.items()}

    def _family_bucket_is_vmap_eligible(self, slots: list[int]) -> bool:
        if not cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE:
            return False
        if not _HAS_TORCH_FUNC:
            raise RuntimeError("torch.func is unavailable but SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE=True")
        if len(slots) < int(cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET):
            return False

        exemplar = self.registry.brains[slots[0]]
        if exemplar is None or exemplar.training:
            return False

        exemplar_type = type(exemplar)
        exemplar_signature = self._brain_topology_signature(exemplar)

        for slot_idx in slots[1:]:
            brain = self.registry.brains[slot_idx]
            if brain is None or brain.training:
                return False
            if type(brain) is not exemplar_type:
                return False
            if self._brain_topology_signature(brain) != exemplar_signature:
                return False

        return True

    def _family_vmap_forward(
        self,
        obs: dict,
        batch_positions: list[int],
        slots: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_obs = self._slice_obs_batch(obs, batch_positions)
        brains = [self.registry.brains[slot_idx] for slot_idx in slots]
        exemplar = brains[0]
        params, buffers = stack_module_state(brains)

        def _single_forward(single_params, single_buffers, single_obs):
            single_obs_batch = {key: value.unsqueeze(0) for key, value in single_obs.items()}
            logits, value = functional_call(
                exemplar,
                (single_params, single_buffers),
                (single_obs_batch,),
                strict=True,
            )
            return logits.squeeze(0), value.squeeze(0)

        logits, values = vmap(
            _single_forward,
            in_dims=(0, 0, 0),
            randomness="error",
        )(params, buffers, bucket_obs)
        return logits, values

    def _loop_bucket_forward(
        self,
        obs: dict,
        entries: list[tuple[int, int]],
        all_logits: torch.Tensor,
        all_values: torch.Tensor,
    ) -> None:
        for batch_pos, slot_idx in entries:
            brain = self.registry.brains[slot_idx]

            if brain is None:
                all_logits[batch_pos, 0] = 1.0
                continue

            agent_obs = {key: value[batch_pos : batch_pos + 1] for key, value in obs.items()}
            logits, value = brain(agent_obs)
            all_logits[batch_pos] = logits.squeeze(0)
            all_values[batch_pos] = value.squeeze(0)

    def _batched_brain_forward(self, obs: dict, alive_slots: list[int]):
        """Run inference for the currently alive slots without crossing UID ownership boundaries."""
        batch_size = len(alive_slots)
        device = next(iter(obs.values())).device
        all_logits = torch.zeros(batch_size, cfg.BRAIN.ACTION_DIM, device=device)
        all_values = torch.zeros(batch_size, cfg.BRAIN.VALUE_DIM, device=device)
        self.last_inference_path_stats = {
            "loop_slots": 0,
            "vmap_slots": 0,
            "family_loop_buckets": 0,
            "family_vmap_buckets": 0,
        }

        buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for batch_pos, slot_idx in enumerate(alive_slots):
            family_id = self.registry.get_family_for_slot(slot_idx)
            buckets[str(family_id)].append((batch_pos, slot_idx))

        ordered_families = list(cfg.BRAIN.FAMILY_ORDER) + sorted(
            family_id
            for family_id in buckets
            if family_id not in cfg.BRAIN.FAMILY_ORDER
        )

        for family_id in ordered_families:
            entries = buckets.get(family_id, [])
            if not entries:
                continue

            batch_positions = [batch_pos for batch_pos, _ in entries]
            slots = [slot_idx for _, slot_idx in entries]

            if self._family_bucket_is_vmap_eligible(slots):
                logits, values = self._family_vmap_forward(obs, batch_positions, slots)
                index = torch.tensor(batch_positions, device=device, dtype=torch.long)
                all_logits.index_copy_(0, index, logits)
                all_values.index_copy_(0, index, values)
                self.last_inference_path_stats["vmap_slots"] += len(slots)
                self.last_inference_path_stats["family_vmap_buckets"] += 1
                continue

            self._loop_bucket_forward(obs, entries, all_logits, all_values)
            self.last_inference_path_stats["loop_slots"] += len(slots)
            self.last_inference_path_stats["family_loop_buckets"] += 1

        return all_logits, all_values

    def _sample_actions(self, obs: dict, alive_slots: list[int]):
        with torch.inference_mode():
            if len(alive_slots) == 0:
                return None

            batched_logits, batched_values = self._batched_brain_forward(obs, alive_slots)
            dist = Categorical(logits=batched_logits)
            batched_actions = dist.sample()
            batched_log_probs = dist.log_prob(batched_actions)

        if batched_actions.numel() == 0:
            return None

        return batched_logits, batched_values, batched_actions, batched_log_probs

    def _store_transitions(
        self,
        alive_slots: list[int],
        obs: dict,
        actions_compact: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        done_values = dones.detach().cpu().tolist() if dones.is_cuda else dones.tolist()

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
        final_alive_slots = self._alive_slots_from_indices(final_alive_indices)

        for i, slot_idx in enumerate(final_alive_slots):
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

    def _checkpoint_path_for_tick(self, tick: int, reason: str = SAVE_REASON_SCHEDULED_TICK) -> Path:
        if reason == SAVE_REASON_SCHEDULED_TICK:
            filename = f"{cfg.CHECKPOINT.FILENAME_PREFIX}{int(tick):08d}{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"
        else:
            safe_reason = "".join(ch if ch.isalnum() else "_" for ch in str(reason)).strip("_") or "checkpoint"
            filename = f"{cfg.CHECKPOINT.FILENAME_PREFIX}{int(tick):08d}_{safe_reason}{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"
        path = self._checkpoint_dir() / filename
        if not path.exists():
            return path
        stem = path.stem
        for idx in range(1, 1000):
            candidate = path.with_name(f"{stem}_{idx:02d}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Unable to allocate a unique checkpoint path for tick {tick} and reason {reason!r}")

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

    def flush_telemetry_for_checkpoint(self, reason: str) -> None:
        """Flush buffered telemetry rows before a checkpoint observes runtime state."""
        if getattr(self.logger, "_closed", False):
            return
        self.logger.flush_parquet_buffers()
        h5_file = getattr(self.logger, "h5_file", None)
        if h5_file is not None:
            h5_file.flush()

    def _publish_runtime_checkpoint(self, reason: str, *, force: bool = False) -> Path | None:
        """Publish one checkpoint after telemetry has reached a durable row boundary."""
        if not cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS:
            return None
        if not force and int(self.tick) <= 0:
            return None

        checkpoint_path = self._checkpoint_path_for_tick(self.tick, reason=reason)
        self.flush_telemetry_for_checkpoint(reason)
        previous_reason = self._checkpoint_capture_reason
        previous_path = self._checkpoint_capture_path
        self._checkpoint_capture_reason = str(reason)
        self._checkpoint_capture_path = str(checkpoint_path)
        try:
            bundle = capture_runtime_checkpoint(self._runtime_checkpoint_view)
        finally:
            self._checkpoint_capture_reason = previous_reason
            self._checkpoint_capture_path = previous_path
        save_runtime_checkpoint(checkpoint_path, bundle)
        self.last_runtime_checkpoint_tick = int(self.tick)
        self.last_runtime_checkpoint_path = str(checkpoint_path)
        self.last_runtime_checkpoint_reason = str(reason)
        if reason == SAVE_REASON_SCHEDULED_WALLCLOCK:
            self.last_runtime_checkpoint_wallclock_time = time.monotonic()
        if hasattr(self.logger, "record_checkpoint_published"):
            self.logger.record_checkpoint_published(tick=self.tick, path=checkpoint_path, reason=reason)
        self._prune_old_runtime_checkpoints()
        return checkpoint_path

    def _maybe_save_runtime_checkpoint_tick(self) -> Path | None:
        """Publish a post-tick checkpoint only after physics, deaths, births, and PPO state settle."""
        interval = int(cfg.CHECKPOINT.SAVE_EVERY_TICKS)
        if not cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS or interval <= 0:
            return None
        if self.tick <= 0 or self.tick % interval != 0:
            return None
        return self._publish_runtime_checkpoint(SAVE_REASON_SCHEDULED_TICK)

    def _maybe_save_runtime_checkpoint_wallclock(self, *, paused: bool = False) -> Path | None:
        if not cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS or not cfg.CHECKPOINT.ENABLE_WALLCLOCK_AUTOSAVE:
            return None
        if paused and not cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_WHILE_PAUSED:
            return None
        now = time.monotonic()
        last_wallclock = self.last_runtime_checkpoint_wallclock_time
        if last_wallclock is None:
            last_wallclock = self._wallclock_autosave_started_at
        if now - float(last_wallclock) < float(cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_INTERVAL_SECONDS):
            return None
        min_ticks = int(cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_MIN_TICKS_ADVANCED)
        if self.last_runtime_checkpoint_tick >= 0 and int(self.tick) - int(self.last_runtime_checkpoint_tick) < min_ticks:
            return None
        return self._publish_runtime_checkpoint(SAVE_REASON_SCHEDULED_WALLCLOCK)

    def maybe_save_runtime_checkpoint_wallclock(self, *, paused: bool = False) -> Path | None:
        return self._maybe_save_runtime_checkpoint_wallclock(paused=paused)

    def _maybe_save_runtime_checkpoint(self) -> Path | None:
        return self._maybe_save_runtime_checkpoint_tick()

    def publish_runtime_checkpoint(self, reason: str = SAVE_REASON_MANUAL_RESERVED, *, force: bool = False) -> Path | None:
        return self._publish_runtime_checkpoint(reason, force=force)

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

        # Deterministic catastrophe scheduling boundary:
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

        alive_slots = self._alive_slots_from_indices(alive_indices)
        obs = self.perception.build_observations(alive_indices)
        sampled = self._sample_actions(obs, alive_slots)

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
            alive_slots,
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
        self._maybe_save_runtime_checkpoint_tick()
        self._maybe_save_runtime_checkpoint_wallclock(paused=False)
        self._maybe_print_tick_progress()
