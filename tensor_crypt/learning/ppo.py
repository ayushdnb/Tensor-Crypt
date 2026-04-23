"""
Implements the Proximal Policy Optimization (PPO) algorithm.

Training ownership is anchored to canonical UIDs. Execution may still locate
the live brain through slot lookup, but optimizer state, rollout state,
bootstrap state, counters, and last-update summaries belong to `agent_uid`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ..agents.state_registry import Registry
from ..config_bridge import cfg


def _clone_obs_to_cpu(obs: dict | None) -> dict | None:
    if obs is None:
        return None
    return {key: value.detach().cpu().clone() for key, value in obs.items()}


def _clone_obs_to_device(obs: dict | None, device: str) -> dict | None:
    if obs is None:
        return None
    return {key: value.to(device) for key, value in obs.items()}


@dataclass
class AgentTrainingState:
    env_steps: int = 0
    ppo_updates: int = 0
    optimizer_steps: int = 0
    truncated_rollouts: int = 0
    last_kl: float = 0.0
    last_entropy: float = 0.0
    last_value_loss: float = 0.0
    last_policy_loss: float = 0.0
    last_grad_norm: float = 0.0
    last_buffer_size: int = 0
    last_update_tick: int = -1

    def serialize(self) -> dict:
        return {
            "env_steps": int(self.env_steps),
            "ppo_updates": int(self.ppo_updates),
            "optimizer_steps": int(self.optimizer_steps),
            "truncated_rollouts": int(self.truncated_rollouts),
            "last_kl": float(self.last_kl),
            "last_entropy": float(self.last_entropy),
            "last_value_loss": float(self.last_value_loss),
            "last_policy_loss": float(self.last_policy_loss),
            "last_grad_norm": float(self.last_grad_norm),
            "last_buffer_size": int(self.last_buffer_size),
            "last_update_tick": int(self.last_update_tick),
        }

    @classmethod
    def deserialize(cls, payload: dict) -> "AgentTrainingState":
        return cls(
            env_steps=int(payload.get("env_steps", 0)),
            ppo_updates=int(payload.get("ppo_updates", 0)),
            optimizer_steps=int(payload.get("optimizer_steps", 0)),
            truncated_rollouts=int(payload.get("truncated_rollouts", 0)),
            last_kl=float(payload.get("last_kl", 0.0)),
            last_entropy=float(payload.get("last_entropy", 0.0)),
            last_value_loss=float(payload.get("last_value_loss", 0.0)),
            last_policy_loss=float(payload.get("last_policy_loss", 0.0)),
            last_grad_norm=float(payload.get("last_grad_norm", 0.0)),
            last_buffer_size=int(payload.get("last_buffer_size", 0)),
            last_update_tick=int(payload.get("last_update_tick", -1)),
        )


class _AgentBuffer:
    """
    UID-owned rollout buffer.

    Invariants:
    - every tensor list length matches across all transition fields
    - bootstrap state belongs to the same UID as the trajectory it closes
    - terminal/death closure and active bootstrap closure are explicit
    """

    def __init__(self):
        self.observations: list[dict] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.bootstrap_obs: dict | None = None
        self.bootstrap_done: torch.Tensor | None = None
        self.finalization_kind: str = "none"

    def validate_structure(self) -> None:
        size = len(self.observations)
        if not all(
            len(values) == size
            for values in (
                self.actions,
                self.log_probs,
                self.rewards,
                self.values,
                self.dones,
            )
        ):
            raise ValueError("UID-owned PPO buffer contains ragged transition lists")
        if self.bootstrap_obs is None and self.bootstrap_done is not None and float(self.bootstrap_done.item()) < 0.5:
            raise ValueError("Active bootstrap buffer state is missing bootstrap observation payload")

    def validate_finite(self) -> None:
        for obs_idx, obs in enumerate(self.observations):
            for key, value in obs.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"UID-owned PPO buffer contains non-finite observation tensor '{key}' at index {obs_idx}")
        for field_name, values in (
            ("actions", self.actions),
            ("log_probs", self.log_probs),
            ("rewards", self.rewards),
            ("values", self.values),
            ("dones", self.dones),
        ):
            for value_idx, value in enumerate(values):
                if not torch.isfinite(value).all():
                    raise ValueError(f"UID-owned PPO buffer contains non-finite {field_name} tensor at index {value_idx}")

        if self.bootstrap_obs is not None:
            for key, value in self.bootstrap_obs.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"UID-owned PPO buffer contains non-finite bootstrap observation tensor '{key}'")
        if self.bootstrap_done is not None and not torch.isfinite(self.bootstrap_done).all():
            raise ValueError("UID-owned PPO buffer contains non-finite bootstrap_done state")

    def validate(self) -> None:
        self.validate_structure()
        self.validate_finite()

    def store_transition(
        self,
        obs: dict,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.bootstrap_obs = None
        self.bootstrap_done = None
        self.finalization_kind = "pending"

    def stage_bootstrap(self, obs: dict | None, done: torch.Tensor, *, finalization_kind: str) -> None:
        self.bootstrap_obs = obs
        self.bootstrap_done = done.detach().clone()
        self.finalization_kind = finalization_kind
        self.validate_structure()
        if self.bootstrap_obs is not None:
            for key, value in self.bootstrap_obs.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"UID-owned PPO buffer contains non-finite bootstrap observation tensor '{key}'")
        if not torch.isfinite(self.bootstrap_done).all():
            raise ValueError("UID-owned PPO buffer contains non-finite bootstrap_done state")

    def has_terminal_tail(self) -> bool:
        if not self.dones:
            return False
        return bool(float(self.dones[-1].detach().item()) >= 0.5)

    def is_truncated_if_dropped(self) -> bool:
        if len(self) == 0:
            return False
        if self.has_terminal_tail():
            return False
        if self.bootstrap_done is not None and float(self.bootstrap_done.item()) >= 0.5:
            return False
        return True

    def serialize(self) -> dict:
        self.validate()
        return {
            "buffer_schema_version": cfg.PPO.BUFFER_SCHEMA_VERSION,
            "observations": [_clone_obs_to_cpu(obs) for obs in self.observations],
            "actions": [value.detach().cpu().clone() for value in self.actions],
            "log_probs": [value.detach().cpu().clone() for value in self.log_probs],
            "rewards": [value.detach().cpu().clone() for value in self.rewards],
            "values": [value.detach().cpu().clone() for value in self.values],
            "dones": [value.detach().cpu().clone() for value in self.dones],
            "bootstrap_obs": _clone_obs_to_cpu(self.bootstrap_obs),
            "bootstrap_done": None if self.bootstrap_done is None else self.bootstrap_done.detach().cpu().clone(),
            "finalization_kind": self.finalization_kind,
        }

    @classmethod
    def deserialize(cls, payload: dict, device: str) -> "_AgentBuffer":
        expected_schema = cfg.PPO.BUFFER_SCHEMA_VERSION
        actual_schema = int(payload.get("buffer_schema_version", expected_schema))
        if actual_schema != expected_schema:
            raise ValueError(
                f"PPO buffer schema mismatch: expected {expected_schema}, got {actual_schema}"
            )

        buffer = cls()
        buffer.observations = [_clone_obs_to_device(obs, device) for obs in payload.get("observations", [])]
        buffer.actions = [value.to(device) for value in payload.get("actions", [])]
        buffer.log_probs = [value.to(device) for value in payload.get("log_probs", [])]
        buffer.rewards = [value.to(device) for value in payload.get("rewards", [])]
        buffer.values = [value.to(device) for value in payload.get("values", [])]
        buffer.dones = [value.to(device) for value in payload.get("dones", [])]
        buffer.bootstrap_obs = _clone_obs_to_device(payload.get("bootstrap_obs"), device)
        bootstrap_done = payload.get("bootstrap_done")
        buffer.bootstrap_done = None if bootstrap_done is None else bootstrap_done.to(device)
        buffer.finalization_kind = str(payload.get("finalization_kind", "none"))
        buffer.validate()
        return buffer

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.bootstrap_obs = None
        self.bootstrap_done = None
        self.finalization_kind = "none"

    def __len__(self) -> int:
        return len(self.observations)


class PPO:
    def __init__(self):
        self.optimizers_by_uid: Dict[int, optim.Adam] = {}
        self.scaler = torch.amp.GradScaler("cuda") if cfg.LOG.AMP and torch.cuda.is_available() else None
        self.buffers_by_uid: Dict[int, _AgentBuffer] = defaultdict(_AgentBuffer)
        self.training_state_by_uid: Dict[int, AgentTrainingState] = {}

    @staticmethod
    def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"PPO update produced non-finite tensor '{name}'")

    @staticmethod
    def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
        PPO._require_finite_tensor("advantages_input", advantages)
        if advantages.numel() == 0:
            return advantages

        mean = advantages.mean()
        PPO._require_finite_tensor("advantages_mean", mean)

        if advantages.numel() == 1:
            return torch.zeros_like(advantages)

        std = advantages.std(unbiased=False)
        PPO._require_finite_tensor("advantages_std", std)

        centered = advantages - mean
        if float(std.item()) <= 1e-8:
            return centered

        normalized = centered / (std + 1e-8)
        PPO._require_finite_tensor("advantages_normalized", normalized)
        return normalized

    @staticmethod
    def _summarize_nonfinite_gradients(brain: nn.Module, *, max_items: int = 6) -> str:
        names: list[str] = []
        for name, param in brain.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if torch.isfinite(grad).all():
                continue
            names.append(name)
            if len(names) >= max_items:
                break
        return ", ".join(names) if names else "<non-finite grad norm but no offending parameter names were isolated>"

    @staticmethod
    def validate_serialized_buffer_payload(uid: int, payload: dict) -> None:
        schema_version = int(payload.get("buffer_schema_version", cfg.PPO.BUFFER_SCHEMA_VERSION))
        if schema_version != cfg.PPO.BUFFER_SCHEMA_VERSION:
            raise ValueError(
                f"Serialized PPO buffer for UID {uid} has schema {schema_version}, expected {cfg.PPO.BUFFER_SCHEMA_VERSION}"
            )
        list_keys = ("observations", "actions", "log_probs", "rewards", "values", "dones")
        lengths = {key: len(payload.get(key, [])) for key in list_keys}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Serialized PPO buffer for UID {uid} has ragged list lengths: {lengths}")
        if payload.get("bootstrap_obs") is None and payload.get("bootstrap_done") is not None:
            bootstrap_done = payload["bootstrap_done"]
            if float(bootstrap_done.item()) < 0.5:
                raise ValueError(
                    f"Serialized PPO buffer for UID {uid} is missing bootstrap_obs for a non-terminal bootstrap tail"
                )

    def _get_training_state(self, uid: int) -> AgentTrainingState:
        if uid not in self.training_state_by_uid:
            self.training_state_by_uid[uid] = AgentTrainingState()
        return self.training_state_by_uid[uid]

    def serialize_training_state(self) -> dict:
        return {uid: state.serialize() for uid, state in self.training_state_by_uid.items()}

    def load_serialized_training_state(self, serialized_states: dict) -> None:
        self.training_state_by_uid = {
            int(uid): AgentTrainingState.deserialize(payload)
            for uid, payload in serialized_states.items()
        }

    def build_optimizer_metadata(self, brain: nn.Module, optimizer: optim.Optimizer) -> dict:
        named_params = list(brain.named_parameters())
        return {
            "param_names": [name for name, _ in named_params],
            "param_shapes": [list(param.shape) for _, param in named_params],
            "param_group_sizes": [len(group.get("params", [])) for group in optimizer.state_dict().get("param_groups", [])],
        }

    def clear_optimizer(self, uid: int) -> None:
        if uid in self.optimizers_by_uid:
            del self.optimizers_by_uid[uid]

    def clear_agent_state(self, uid: int, *, reason: str = "clear") -> None:
        self.clear_optimizer(uid)
        buffer = self.buffers_by_uid.get(uid)
        if buffer is not None:
            if cfg.PPO.COUNT_TRUNCATED_ROLLOUTS and buffer.is_truncated_if_dropped():
                self._get_training_state(uid).truncated_rollouts += 1
            if cfg.PPO.DROP_INACTIVE_UID_BUFFERS_AFTER_FINALIZATION or reason != "compatibility":
                del self.buffers_by_uid[uid]

    def clear_agent_state_for_slot(self, registry: Registry, slot_idx: int) -> None:
        uid = registry.get_uid_for_slot(slot_idx)
        if uid != -1:
            self.clear_agent_state(uid, reason="slot_clear")

    def _get_optimizer(self, uid: int, brain: nn.Module) -> optim.Adam:
        if uid not in self.optimizers_by_uid:
            self.optimizers_by_uid[uid] = optim.Adam(brain.parameters(), lr=cfg.PPO.LR)
        return self.optimizers_by_uid[uid]

    def store_transition(
        self,
        uid: int,
        obs: dict,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        buffer = self.buffers_by_uid[uid]
        buffer.store_transition(obs, action, log_prob, reward, value, done)
        self._get_training_state(uid).env_steps += 1

    def store_transition_for_slot(
        self,
        registry: Registry,
        slot_idx: int,
        obs: dict,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        uid = registry.get_uid_for_slot(slot_idx)
        if uid == -1:
            raise AssertionError(f"Slot {slot_idx} has no active UID to own this transition")
        self.store_transition(uid, obs, action, log_prob, reward, value, done)

    def stage_bootstrap_for_uid(
        self,
        uid: int,
        obs: dict | None,
        done: torch.Tensor,
        *,
        finalization_kind: str,
    ) -> None:
        buffer = self.buffers_by_uid.get(uid)
        if buffer is None or len(buffer) == 0:
            return
        if not cfg.CHECKPOINT.CAPTURE_BOOTSTRAP_STATE and finalization_kind == "checkpoint_restore":
            return
        buffer.stage_bootstrap(obs, done, finalization_kind=finalization_kind)

    def finalize_terminal_uid(self, uid: int) -> None:
        buffer = self.buffers_by_uid.get(uid)
        if buffer is None or len(buffer) == 0:
            return
        buffer.stage_bootstrap(None, torch.tensor(1.0, device=cfg.SIM.DEVICE), finalization_kind="terminal_death")

    def serialize_buffer(self, uid: int) -> dict | None:
        buffer = self.buffers_by_uid.get(uid)
        return None if buffer is None else buffer.serialize()

    def serialize_all_buffers(self) -> dict:
        return {uid: buffer.serialize() for uid, buffer in self.buffers_by_uid.items()}

    def load_serialized_buffers(self, serialized_buffers: dict, device: str | None = None) -> None:
        target_device = device or cfg.SIM.DEVICE
        self.buffers_by_uid = defaultdict(_AgentBuffer)
        for uid, payload in serialized_buffers.items():
            uid_int = int(uid)
            if cfg.CHECKPOINT.VALIDATE_BUFFER_SCHEMA:
                self.validate_serialized_buffer_payload(uid_int, payload)
            self.buffers_by_uid[uid_int] = _AgentBuffer.deserialize(payload, target_device)

    def _validate_uid_owner(self, registry: Registry, uid: int, brain: nn.Module) -> int:
        slot_idx = registry.get_slot_for_uid(uid)
        if slot_idx is None:
            raise ValueError(f"UID {uid} has PPO state but no active slot binding")
        if not registry.is_uid_active(uid):
            raise ValueError(f"UID {uid} has PPO state but is not active")
        if registry.brains[slot_idx] is not brain:
            raise ValueError(f"UID {uid} does not own the brain stored in slot {slot_idx}")
        return slot_idx

    def validate_optimizer_state(
        self,
        uid: int,
        brain: nn.Module,
        optimizer_state: dict,
        optimizer_metadata: dict | None = None,
    ) -> None:
        param_groups = optimizer_state.get("param_groups", [])
        if not param_groups:
            raise ValueError(f"Optimizer state for UID {uid} is missing parameter groups")

        named_params = list(brain.named_parameters())
        expected_param_count = len(named_params)
        saved_param_ids: list[int] = []
        for group in param_groups:
            saved_param_ids.extend(int(param_id) for param_id in group.get("params", []))

        if expected_param_count != len(saved_param_ids):
            raise ValueError(
                f"Optimizer state for UID {uid} does not match brain parameter topology: expected {expected_param_count}, got {len(saved_param_ids)}"
            )

        if len(set(saved_param_ids)) != len(saved_param_ids):
            raise ValueError(f"Optimizer state for UID {uid} contains duplicate parameter IDs")

        if optimizer_metadata is not None:
            expected_names = [name for name, _ in named_params]
            expected_shapes = [list(param.shape) for _, param in named_params]
            if optimizer_metadata.get("param_names") != expected_names:
                raise ValueError(f"Optimizer metadata for UID {uid} does not match current named parameter order")
            if optimizer_metadata.get("param_shapes") != expected_shapes:
                raise ValueError(f"Optimizer metadata for UID {uid} does not match current parameter shapes")
            expected_group_sizes = [len(group.get("params", [])) for group in param_groups]
            if optimizer_metadata.get("param_group_sizes") != expected_group_sizes:
                raise ValueError(f"Optimizer metadata for UID {uid} does not match optimizer param group sizes")

        if not cfg.CHECKPOINT.VALIDATE_OPTIMIZER_TENSOR_SHAPES:
            return

        state_by_param_id = optimizer_state.get("state", {})
        for param_id, (_, param) in zip(saved_param_ids, named_params):
            param_state = state_by_param_id.get(param_id, {})
            for state_key, value in param_state.items():
                if torch.is_tensor(value) and value.dim() > 0 and tuple(value.shape) != tuple(param.shape):
                    raise ValueError(
                        f"Optimizer tensor state '{state_key}' for UID {uid} does not match parameter shape {tuple(param.shape)}"
                    )

    def _collate_observations(self, obs_list: list[dict]) -> dict:
        keys = tuple(obs_list[0].keys())
        return {key: torch.stack([obs[key] for obs in obs_list], dim=0) for key in keys}

    @staticmethod
    def _approx_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        return (old_log_probs - new_log_probs).mean()

    def _compute_returns_and_advantages(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[torch.Tensor],
        last_value: torch.Tensor,
        last_done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE returns for one UID-owned rollout.

        The bootstrap gate must use the current transition's terminal flag, not
        the next transition's, otherwise advantages can leak across a death or
        other done boundary when iterating backward.
        """
        if len(rewards) == 0:
            empty = torch.empty(0, device=cfg.SIM.DEVICE)
            return empty, empty

        trajectory_len = len(rewards)
        gae = rewards[0].new_zeros(())
        returns_tensor = torch.empty(
            trajectory_len,
            device=cfg.SIM.DEVICE,
            dtype=rewards[0].dtype,
        )

        for t in reversed(range(trajectory_len)):
            if t == trajectory_len - 1:
                bootstrap_value = last_value
                bootstrap_done = last_done
            else:
                bootstrap_value = values[t + 1]
                bootstrap_done = dones[t]

            reward_t = rewards[t]
            value_t = values[t]
            delta = reward_t + cfg.PPO.GAMMA * bootstrap_value * (1 - bootstrap_done) - value_t
            gae = delta + cfg.PPO.GAMMA * cfg.PPO.LAMBDA * (1 - bootstrap_done) * gae
            returns_tensor[t] = gae + value_t

        values_tensor = torch.stack(values).reshape(-1)
        advantages_tensor = returns_tensor - values_tensor
        return returns_tensor.reshape(-1), advantages_tensor

    def _resolve_bootstrap(self, uid: int, brain: nn.Module, buffer: _AgentBuffer) -> tuple[torch.Tensor, torch.Tensor]:
        if len(buffer) == 0:
            return (
                torch.tensor(0.0, device=cfg.SIM.DEVICE),
                torch.tensor(1.0, device=cfg.SIM.DEVICE),
            )

        if buffer.bootstrap_done is not None:
            bootstrap_done = buffer.bootstrap_done.to(cfg.SIM.DEVICE).reshape(())
            if float(bootstrap_done.item()) >= 0.5:
                return (
                    torch.tensor(0.0, device=cfg.SIM.DEVICE),
                    bootstrap_done,
                )
            if buffer.bootstrap_obs is None:
                raise ValueError(f"UID {uid} has non-terminal bootstrap state without bootstrap observation payload")
            with torch.inference_mode():
                bootstrap_batch = {key: value.unsqueeze(0) for key, value in buffer.bootstrap_obs.items()}
                _, bootstrap_value = brain(bootstrap_batch)
            return bootstrap_value.reshape(()), bootstrap_done

        if buffer.has_terminal_tail():
            return (
                torch.tensor(0.0, device=cfg.SIM.DEVICE),
                torch.tensor(1.0, device=cfg.SIM.DEVICE),
            )

        if cfg.PPO.REQUIRE_BOOTSTRAP_FOR_ACTIVE_BUFFER:
            raise ValueError(
                f"UID {uid} has an active non-terminal buffer without staged bootstrap state"
            )

        return (
            torch.tensor(0.0, device=cfg.SIM.DEVICE),
            torch.tensor(1.0, device=cfg.SIM.DEVICE),
        )

    def _ordered_ready_uids(self, registry: Registry) -> list[int]:
        ready_by_family: dict[str, list[int]] = {family_id: [] for family_id in cfg.BRAIN.FAMILY_ORDER}
        ready_fallback: list[int] = []

        for uid, buffer in list(self.buffers_by_uid.items()):
            if len(buffer) < cfg.PPO.BATCH_SZ:
                continue
            if not registry.is_uid_active(uid):
                self.clear_agent_state(uid, reason="inactive_uid_drop")
                continue

            family_id = registry.get_family_for_uid(uid)
            if cfg.PPO.FAMILY_AWARE_UPDATE_ORDERING:
                ready_by_family.setdefault(family_id, []).append(uid)
            else:
                ready_fallback.append(uid)

        if not cfg.PPO.FAMILY_AWARE_UPDATE_ORDERING:
            return sorted(ready_fallback)

        ordered: list[int] = []
        for family_id in cfg.BRAIN.FAMILY_ORDER:
            ordered.extend(sorted(ready_by_family.get(family_id, [])))
        return ordered

    def update(
        self,
        registry: Registry,
        perception=None,
        last_obs_dict: dict | None = None,
        last_dones_dict: dict | None = None,
        *,
        tick: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict]:
        # Compatibility bridge for older call sites that still pass bootstrap
        # state directly into `update`.
        if last_obs_dict:
            done_lookup = last_dones_dict or {}
            for uid, obs in last_obs_dict.items():
                done = done_lookup.get(uid, torch.tensor(0.0, device=cfg.SIM.DEVICE))
                self.stage_bootstrap_for_uid(
                    int(uid),
                    obs,
                    done,
                    finalization_kind="compatibility_update_bridge",
                )

        stats_list = []
        agents_to_train = self._ordered_ready_uids(registry)

        for uid in agents_to_train:
            if should_stop is not None and should_stop():
                break

            buffer = self.buffers_by_uid[uid]
            if cfg.PPO.STRICT_BUFFER_VALIDATION:
                buffer.validate()

            slot_idx = registry.get_slot_for_uid(uid)
            if slot_idx is None:
                self.clear_agent_state(uid, reason="inactive_uid_drop")
                continue

            brain = registry.brains[slot_idx]
            if brain is None:
                self.clear_agent_state(uid, reason="missing_brain")
                continue

            self._validate_uid_owner(registry, uid, brain)
            brain.train()
            optimizer = self._get_optimizer(uid, brain)
            training_state = self._get_training_state(uid)

            last_value, last_done = self._resolve_bootstrap(uid, brain, buffer)
            returns, advantages = self._compute_returns_and_advantages(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                last_value,
                last_done,
            )

            batch_obs = self._collate_observations(buffer.observations)
            batch_actions = torch.stack(buffer.actions).reshape(-1)
            batch_log_probs = torch.stack(buffer.log_probs).reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = torch.stack(buffer.values).reshape(-1)

            batch_advantages = self._normalize_advantages(batch_advantages)
            self._require_finite_tensor("batch_returns", batch_returns)
            self._require_finite_tensor("batch_values", batch_values)

            batch_size = len(buffer)
            if cfg.PPO.MINI_BATCHES <= 0:
                raise ValueError("cfg.PPO.MINI_BATCHES must be positive")
            mini_batch_size = batch_size // cfg.PPO.MINI_BATCHES
            if mini_batch_size <= 0:
                raise ValueError(
                    f"cfg.PPO.MINI_BATCHES={cfg.PPO.MINI_BATCHES} cannot exceed trajectory batch size {batch_size}"
                )

            kl_div = 0.0
            grad_norm = 0.0
            optimizer_steps_this_update = 0
            amp_nonfinite_grad_events = 0
            completed_epochs = 0
            policy_loss = torch.tensor(0.0, device=cfg.SIM.DEVICE)
            value_loss = torch.tensor(0.0, device=cfg.SIM.DEVICE)
            entropy = torch.tensor(0.0, device=cfg.SIM.DEVICE)

            for epoch in range(cfg.PPO.EPOCHS):
                completed_epochs = epoch + 1
                indices = torch.randperm(batch_size, device=cfg.SIM.DEVICE)
                for start in range(0, batch_size, mini_batch_size):
                    end = min(start + mini_batch_size, batch_size)
                    mb_indices = indices[start:end]

                    mb_obs = {key: value[mb_indices] for key, value in batch_obs.items()}
                    mb_actions = batch_actions[mb_indices]
                    mb_log_probs = batch_log_probs[mb_indices]
                    mb_advantages = batch_advantages[mb_indices]
                    mb_returns = batch_returns[mb_indices]
                    mb_values = batch_values[mb_indices]

                    logits, new_values = brain(mb_obs)
                    new_values = new_values.squeeze(-1)
                    self._require_finite_tensor("logits", logits)
                    self._require_finite_tensor("new_values", new_values)

                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    self._require_finite_tensor("new_log_probs", new_log_probs)
                    self._require_finite_tensor("entropy", entropy)

                    log_ratio = new_log_probs - mb_log_probs
                    ratio = torch.exp(log_ratio)
                    clip_adv = torch.clamp(ratio, 1 - cfg.PPO.CLIP_EPS, 1 + cfg.PPO.CLIP_EPS) * mb_advantages
                    policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()

                    value_loss_unclipped = (new_values - mb_returns) ** 2
                    values_clipped = mb_values + (new_values - mb_values).clamp(-cfg.PPO.CLIP_EPS, cfg.PPO.CLIP_EPS)
                    value_loss_clipped = (values_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    loss = policy_loss + (cfg.PPO.VALUE_COEF * value_loss) - (cfg.PPO.ENTROPY_COEF * entropy)
                    self._require_finite_tensor("policy_loss", policy_loss)
                    self._require_finite_tensor("value_loss", value_loss)
                    self._require_finite_tensor("loss", loss)

                    optimizer.zero_grad(set_to_none=True)
                    if cfg.LOG.AMP and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.PPO.GRAD_NORM_CLIP)
                        grad_norm = float(grad_norm_tensor.detach().item()) if torch.is_tensor(grad_norm_tensor) else float(grad_norm_tensor)
                        if not math.isfinite(grad_norm):
                            bad_grad_summary = self._summarize_nonfinite_gradients(brain)
                            amp_nonfinite_grad_events += 1
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            print(
                                f"[ppo] Warning: skipped AMP optimizer step for UID {uid} at tick {int(-1 if tick is None else tick)} "
                                f"after non-finite gradient norm; offending grads: {bad_grad_summary}"
                            )
                            continue
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.PPO.GRAD_NORM_CLIP)
                        grad_norm = float(grad_norm_tensor.detach().item()) if torch.is_tensor(grad_norm_tensor) else float(grad_norm_tensor)
                        if not math.isfinite(grad_norm):
                            optimizer.zero_grad(set_to_none=True)
                            raise ValueError(f"PPO update produced non-finite gradient norm for UID {uid}")
                        optimizer.step()

                    optimizer_steps_this_update += 1
                    kl_div = float(self._approx_kl(mb_log_probs, new_log_probs).detach().item())

                    if cfg.PPO.TARGET_KL > 0 and kl_div > cfg.PPO.TARGET_KL:
                        break

                if cfg.PPO.TARGET_KL > 0 and kl_div > cfg.PPO.TARGET_KL:
                    break

            if optimizer_steps_this_update == 0 and amp_nonfinite_grad_events > 0:
                brain.eval()
                print(
                    f"[ppo] Warning: deferred UID {uid} PPO update at tick {int(-1 if tick is None else tick)} "
                    f"after {amp_nonfinite_grad_events} AMP overflow event(s); retaining buffer for retry."
                )
                continue

            training_state.ppo_updates += 1
            training_state.optimizer_steps += optimizer_steps_this_update
            training_state.last_kl = float(kl_div)
            training_state.last_entropy = float(entropy.detach().item())
            training_state.last_value_loss = float(value_loss.detach().item())
            training_state.last_policy_loss = float(policy_loss.detach().item())
            training_state.last_grad_norm = float(grad_norm)
            training_state.last_buffer_size = int(batch_size)
            training_state.last_update_tick = int(-1 if tick is None else tick)

            buffer.clear()
            brain.eval()
            stats_list.append(
                {
                    "agent_uid": uid,
                    "agent_slot": slot_idx,
                    "family_id": registry.get_family_for_uid(uid),
                    "policy_loss": training_state.last_policy_loss,
                    "value_loss": training_state.last_value_loss,
                    "entropy": training_state.last_entropy,
                    "kl_div": training_state.last_kl,
                    "grad_norm": training_state.last_grad_norm,
                    "buffer_size": training_state.last_buffer_size,
                    "ppo_updates": training_state.ppo_updates,
                    "optimizer_steps": training_state.optimizer_steps,
                    "env_steps": training_state.env_steps,
                    "truncated_rollouts": training_state.truncated_rollouts,
                    "update_epochs_completed": completed_epochs,
                }
            )

        return stats_list

    def should_update(self, tick: int) -> bool:
        return tick > 0 and tick % cfg.PPO.UPDATE_EVERY_N_TICKS == 0
