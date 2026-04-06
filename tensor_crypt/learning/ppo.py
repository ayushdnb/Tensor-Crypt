"""
Implements the Proximal Policy Optimization (PPO) algorithm.

Prompt 1 moves training ownership from slot indices to canonical UIDs. Slot
indices are still used to find the active execution-time brain object, but PPO
state itself is keyed only by organism UID.
"""

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ..agents.state_registry import Registry
from ..config_bridge import cfg


class _AgentBuffer:
    """A simple buffer to store one UID-owned trajectory."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store_transition(self, obs: dict, action: torch.Tensor, log_prob: torch.Tensor, reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def serialize(self) -> dict:
        return {
            "observations": [
                {key: value.detach().cpu().clone() for key, value in obs.items()}
                for obs in self.observations
            ],
            "actions": [value.detach().cpu().clone() for value in self.actions],
            "log_probs": [value.detach().cpu().clone() for value in self.log_probs],
            "rewards": [value.detach().cpu().clone() for value in self.rewards],
            "values": [value.detach().cpu().clone() for value in self.values],
            "dones": [value.detach().cpu().clone() for value in self.dones],
        }

    @classmethod
    def deserialize(cls, payload: dict, device: str):
        buffer = cls()
        buffer.observations = [
            {key: value.to(device) for key, value in obs.items()}
            for obs in payload.get("observations", [])
        ]
        buffer.actions = [value.to(device) for value in payload.get("actions", [])]
        buffer.log_probs = [value.to(device) for value in payload.get("log_probs", [])]
        buffer.rewards = [value.to(device) for value in payload.get("rewards", [])]
        buffer.values = [value.to(device) for value in payload.get("values", [])]
        buffer.dones = [value.to(device) for value in payload.get("dones", [])]
        return buffer

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)


class PPO:
    def __init__(self):
        self.optimizers_by_uid: Dict[int, optim.Adam] = {}
        self.scaler = torch.amp.GradScaler("cuda") if cfg.LOG.AMP and torch.cuda.is_available() else None
        self.buffers_by_uid: Dict[int, _AgentBuffer] = defaultdict(_AgentBuffer)

    def clear_optimizer(self, uid: int):
        if uid in self.optimizers_by_uid:
            del self.optimizers_by_uid[uid]

    def clear_agent_state(self, uid: int):
        self.clear_optimizer(uid)
        if uid in self.buffers_by_uid:
            del self.buffers_by_uid[uid]

    def clear_agent_state_for_slot(self, registry: Registry, slot_idx: int):
        uid = registry.get_uid_for_slot(slot_idx)
        if uid != -1:
            self.clear_agent_state(uid)

    def _get_optimizer(self, uid: int, brain: nn.Module) -> optim.Adam:
        if uid not in self.optimizers_by_uid:
            self.optimizers_by_uid[uid] = optim.Adam(brain.parameters(), lr=cfg.PPO.LR)
        return self.optimizers_by_uid[uid]

    def store_transition(self, uid: int, obs: dict, action: torch.Tensor, log_prob: torch.Tensor, reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        self.buffers_by_uid[uid].store_transition(obs, action, log_prob, reward, value, done)

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
    ):
        uid = registry.get_uid_for_slot(slot_idx)
        if uid == -1:
            raise AssertionError(f"Slot {slot_idx} has no active UID to own this transition")
        self.store_transition(uid, obs, action, log_prob, reward, value, done)

    def serialize_buffer(self, uid: int) -> dict | None:
        buffer = self.buffers_by_uid.get(uid)
        return None if buffer is None else buffer.serialize()

    def serialize_all_buffers(self) -> dict:
        return {uid: buffer.serialize() for uid, buffer in self.buffers_by_uid.items()}

    def load_serialized_buffers(self, serialized_buffers: dict, device: str | None = None):
        target_device = device or cfg.SIM.DEVICE
        self.buffers_by_uid = defaultdict(_AgentBuffer)
        for uid, payload in serialized_buffers.items():
            self.buffers_by_uid[int(uid)] = _AgentBuffer.deserialize(payload, target_device)

    def _validate_uid_owner(self, registry: Registry, uid: int, brain: nn.Module) -> int:
        slot_idx = registry.get_slot_for_uid(uid)
        if slot_idx is None:
            raise ValueError(f"UID {uid} has PPO state but no active slot binding")
        if not registry.is_uid_active(uid):
            raise ValueError(f"UID {uid} has PPO state but is not active")
        if registry.brains[slot_idx] is not brain:
            raise ValueError(f"UID {uid} does not own the brain stored in slot {slot_idx}")
        return slot_idx

    def validate_optimizer_state(self, uid: int, brain: nn.Module, optimizer_state: dict) -> None:
        param_groups = optimizer_state.get("param_groups", [])
        if not param_groups:
            raise ValueError(f"Optimizer state for UID {uid} is missing parameter groups")

        expected_param_count = sum(1 for _ in brain.parameters())
        saved_param_count = sum(len(group.get("params", [])) for group in param_groups)
        if expected_param_count != saved_param_count:
            raise ValueError(
                f"Optimizer state for UID {uid} does not match brain parameter topology: expected {expected_param_count}, got {saved_param_count}"
            )

    def _collate_observations(self, obs_list: list) -> dict:
        keys = obs_list[0].keys()
        return {key: torch.stack([obs[key] for obs in obs_list]).to(cfg.SIM.DEVICE) for key in keys}

    def _compute_returns_and_advantages(self, rewards: List, values: List, dones: List, last_value: torch.Tensor, last_done: torch.Tensor):
        gae = 0
        returns = []

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = last_value
                next_done = last_done
            else:
                next_val = values[t + 1]
                next_done = dones[t]

            delta = rewards[t] + cfg.PPO.GAMMA * next_val * (1 - next_done) - values[t]
            gae = delta + cfg.PPO.GAMMA * cfg.PPO.LAMBDA * (1 - next_done) * gae
            returns.insert(0, gae + values[t])

        advantages = [ret - val for ret, val in zip(returns, values)]
        return torch.stack(returns), torch.stack(advantages)

    def update(self, registry: Registry, perception, last_obs_dict: dict, last_dones_dict: dict):
        stats_list = []
        agents_to_train = list(self.buffers_by_uid.keys())

        for uid in agents_to_train:
            buffer = self.buffers_by_uid[uid]
            if len(buffer) < cfg.PPO.BATCH_SZ:
                continue

            slot_idx = registry.get_slot_for_uid(uid)
            if slot_idx is None:
                continue

            brain = registry.brains[slot_idx]
            if brain is None:
                continue

            self._validate_uid_owner(registry, uid, brain)
            brain.train()
            optimizer = self._get_optimizer(uid, brain)

            with torch.no_grad():
                if uid in last_obs_dict:
                    last_obs = last_obs_dict[uid]
                    last_obs_batch = {key: value.unsqueeze(0) for key, value in last_obs.items()}
                    _, last_value = brain(last_obs_batch)
                    last_value = last_value.squeeze()
                    last_done = last_dones_dict[uid]
                else:
                    last_value = torch.tensor(0.0, device=cfg.SIM.DEVICE)
                    last_done = torch.tensor(1.0, device=cfg.SIM.DEVICE)

            with torch.no_grad():
                returns, advantages = self._compute_returns_and_advantages(
                    buffer.rewards,
                    buffer.values,
                    buffer.dones,
                    last_value,
                    last_done,
                )

            batch_obs = self._collate_observations(buffer.observations)
            batch_actions = torch.stack(buffer.actions).to(cfg.SIM.DEVICE)
            batch_log_probs = torch.stack(buffer.log_probs).to(cfg.SIM.DEVICE)
            batch_advantages = advantages.to(cfg.SIM.DEVICE)
            batch_returns = returns.to(cfg.SIM.DEVICE)
            batch_values = torch.stack(buffer.values).to(cfg.SIM.DEVICE)

            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            batch_size = len(buffer)
            if cfg.PPO.MINI_BATCHES <= 0:
                raise ValueError("cfg.PPO.MINI_BATCHES must be positive")
            mini_batch_size = batch_size // cfg.PPO.MINI_BATCHES
            if mini_batch_size <= 0:
                raise ValueError(
                    f"cfg.PPO.MINI_BATCHES={cfg.PPO.MINI_BATCHES} cannot exceed trajectory batch size {batch_size}"
                )

            kl_div = 0.0
            policy_loss = torch.tensor(0.0, device=cfg.SIM.DEVICE)
            value_loss = torch.tensor(0.0, device=cfg.SIM.DEVICE)
            entropy = torch.tensor(0.0, device=cfg.SIM.DEVICE)

            for epoch in range(cfg.PPO.EPOCHS):
                indices = torch.randperm(batch_size)
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mb_indices = indices[start:end]

                    mb_obs = {key: value[mb_indices] for key, value in batch_obs.items()}
                    mb_actions = batch_actions[mb_indices]
                    mb_log_probs = batch_log_probs[mb_indices]
                    mb_advantages = batch_advantages[mb_indices]
                    mb_returns = batch_returns[mb_indices]
                    mb_values = batch_values[mb_indices]

                    logits, new_values = brain(mb_obs)
                    new_values = new_values.squeeze(-1)

                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    log_ratio = new_log_probs - mb_log_probs
                    ratio = torch.exp(log_ratio)
                    clip_adv = torch.clamp(ratio, 1 - cfg.PPO.CLIP_EPS, 1 + cfg.PPO.CLIP_EPS) * mb_advantages
                    policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()

                    value_loss_unclipped = (new_values - mb_returns) ** 2
                    values_clipped = mb_values + (new_values - mb_values).clamp(-cfg.PPO.CLIP_EPS, cfg.PPO.CLIP_EPS)
                    value_loss_clipped = (values_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    loss = policy_loss + (cfg.PPO.VALUE_COEF * value_loss) - (cfg.PPO.ENTROPY_COEF * entropy)

                    optimizer.zero_grad()
                    if cfg.LOG.AMP and self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.PPO.GRAD_NORM_CLIP)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.PPO.GRAD_NORM_CLIP)
                        optimizer.step()

                    with torch.no_grad():
                        kl_div = log_ratio.mean().item()
                    if cfg.PPO.TARGET_KL > 0 and kl_div > cfg.PPO.TARGET_KL:
                        break

                if cfg.PPO.TARGET_KL > 0 and kl_div > cfg.PPO.TARGET_KL:
                    break

            buffer.clear()
            stats_list.append(
                {
                    "agent_uid": uid,
                    "agent_slot": slot_idx,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "kl_div": kl_div,
                }
            )

        return stats_list

    def should_update(self, tick: int) -> bool:
        return tick > 0 and tick % cfg.PPO.UPDATE_EVERY_N_TICKS == 0
