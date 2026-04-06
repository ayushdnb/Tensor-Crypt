from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from ..agents.brain import Brain
from ..agents.state_registry import AgentLifecycleRecord, Registry
from ..config_bridge import cfg
from ..learning.ppo import PPO


def _schema_versions_dict(cfg_obj) -> dict:
    return {
        "IDENTITY_SCHEMA_VERSION": cfg_obj.SCHEMA.IDENTITY_SCHEMA_VERSION,
        "OBS_SCHEMA_VERSION": cfg_obj.SCHEMA.OBS_SCHEMA_VERSION,
        "CHECKPOINT_SCHEMA_VERSION": cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "REPRODUCTION_SCHEMA_VERSION": cfg_obj.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
        "TELEMETRY_SCHEMA_VERSION": cfg_obj.SCHEMA.TELEMETRY_SCHEMA_VERSION,
        "LOGGING_SCHEMA_VERSION": cfg_obj.SCHEMA.LOGGING_SCHEMA_VERSION,
    }


def _clone_nested_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_nested_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested_cpu(item) for item in value)
    return copy.deepcopy(value)


def serialize_agent_lifecycle(uid_lifecycle: dict[int, AgentLifecycleRecord]) -> dict[int, dict]:
    return {
        uid: {
            "uid": record.uid,
            "parent_uid": record.parent_uid,
            "birth_tick": record.birth_tick,
            "death_tick": record.death_tick,
            "current_slot": record.current_slot,
            "is_active": record.is_active,
        }
        for uid, record in uid_lifecycle.items()
    }


def deserialize_agent_lifecycle(serialized: dict[int, dict]) -> dict[int, AgentLifecycleRecord]:
    return {
        int(uid): AgentLifecycleRecord(
            uid=int(payload["uid"]),
            parent_uid=int(payload["parent_uid"]),
            birth_tick=int(payload["birth_tick"]),
            death_tick=None if payload["death_tick"] is None else int(payload["death_tick"]),
            current_slot=None if payload["current_slot"] is None else int(payload["current_slot"]),
            is_active=bool(payload["is_active"]),
        )
        for uid, payload in serialized.items()
    }


def capture_rng_state() -> dict:
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]

    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
        "torch_cuda_rng_state_all": cuda_state,
    }


def restore_rng_state(rng_state: dict) -> None:
    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])
    torch.set_rng_state(rng_state["torch_cpu_rng_state"].cpu())

    cuda_state = rng_state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def capture_runtime_checkpoint(runtime) -> dict:
    if not cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS:
        raise RuntimeError("Checkpoint substrate capture is disabled by configuration")

    registry = runtime.registry
    brain_state_by_uid = {}
    for uid, slot_idx in registry.active_uid_to_slot.items():
        brain = registry.brains[slot_idx]
        if brain is None:
            raise AssertionError(f"Active UID {uid} in slot {slot_idx} has no brain instance")
        brain_state_by_uid[uid] = _clone_nested_cpu(brain.state_dict())

    optimizer_state_by_uid = {}
    if cfg.CHECKPOINT.CAPTURE_OPTIMIZER_STATE:
        optimizer_state_by_uid = {
            uid: _clone_nested_cpu(optimizer.state_dict())
            for uid, optimizer in runtime.ppo.optimizers_by_uid.items()
        }

    scaler_state = None
    if cfg.CHECKPOINT.CAPTURE_SCALER_STATE and runtime.ppo.scaler is not None:
        scaler_state = _clone_nested_cpu(runtime.ppo.scaler.state_dict())

    return {
        "checkpoint_schema_version": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "schema_versions": _schema_versions_dict(cfg),
        "config_snapshot": asdict(cfg),
        "engine_state": {
            "tick": int(runtime.engine.tick),
            "respawn_last_tick": int(runtime.engine.respawn_controller.last_respawn_tick),
        },
        "registry_state": {
            "data": registry.data.detach().cpu().clone(),
            "slot_uid": registry.slot_uid.detach().cpu().clone(),
            "slot_parent_uid": registry.slot_parent_uid.detach().cpu().clone(),
            "next_agent_uid": int(registry.next_agent_uid),
            "fitness": registry.fitness.detach().cpu().clone(),
            "uid_lifecycle": serialize_agent_lifecycle(registry.uid_lifecycle),
        },
        "grid_state": {
            "grid": runtime.grid.grid.detach().cpu().clone(),
            "hzones": copy.deepcopy(runtime.grid.hzones),
            "next_hzone_id": int(runtime.grid.next_hzone_id),
        },
        "brain_state_by_uid": brain_state_by_uid,
        "ppo_state": {
            "optimizer_state_by_uid": optimizer_state_by_uid,
            "buffer_state_by_uid": runtime.ppo.serialize_all_buffers(),
            "scaler_state": scaler_state,
        },
        "rng_state": capture_rng_state() if cfg.CHECKPOINT.CAPTURE_RNG_STATE else None,
        "metadata": {
            "device": str(cfg.SIM.DEVICE),
            "amp_enabled": bool(cfg.LOG.AMP),
        },
    }


def validate_runtime_checkpoint(bundle: dict, cfg_obj) -> None:
    required_top_keys = {
        "checkpoint_schema_version",
        "schema_versions",
        "config_snapshot",
        "engine_state",
        "registry_state",
        "grid_state",
        "brain_state_by_uid",
        "ppo_state",
        "rng_state",
        "metadata",
    }
    missing = required_top_keys.difference(bundle.keys())
    if missing:
        raise ValueError(f"Checkpoint bundle is missing required keys: {sorted(missing)}")

    if cfg_obj.CHECKPOINT.STRICT_SCHEMA_VALIDATION:
        expected_checkpoint_version = cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION
        if int(bundle["checkpoint_schema_version"]) != expected_checkpoint_version:
            raise ValueError(
                f"Checkpoint schema mismatch: expected {expected_checkpoint_version}, got {bundle['checkpoint_schema_version']}"
            )
        expected_versions = _schema_versions_dict(cfg_obj)
        if bundle["schema_versions"] != expected_versions:
            raise ValueError("Checkpoint schema_versions payload does not match the current runtime schema contract")

    registry_state = bundle["registry_state"]
    data = registry_state["data"]
    slot_uid = registry_state["slot_uid"]
    slot_parent_uid = registry_state["slot_parent_uid"]
    uid_lifecycle = deserialize_agent_lifecycle(registry_state["uid_lifecycle"])

    if data.shape != (Registry.NUM_COLS, cfg_obj.AGENTS.N):
        raise ValueError(f"Registry tensor shape mismatch: expected {(Registry.NUM_COLS, cfg_obj.AGENTS.N)}, got {tuple(data.shape)}")
    if tuple(slot_uid.shape) != (cfg_obj.AGENTS.N,):
        raise ValueError(f"slot_uid shape mismatch: expected {(cfg_obj.AGENTS.N,)}, got {tuple(slot_uid.shape)}")
    if tuple(slot_parent_uid.shape) != (cfg_obj.AGENTS.N,):
        raise ValueError(f"slot_parent_uid shape mismatch: expected {(cfg_obj.AGENTS.N,)}, got {tuple(slot_parent_uid.shape)}")
    if slot_uid.dtype != torch.int64 or slot_parent_uid.dtype != torch.int64:
        raise ValueError("Canonical UID tensors must be int64")

    active_uids = set()
    for slot_idx in range(cfg_obj.AGENTS.N):
        uid = int(slot_uid[slot_idx].item())
        parent_uid = int(slot_parent_uid[slot_idx].item())
        alive = bool(data[Registry.ALIVE, slot_idx].item() > 0.5)

        if uid == -1:
            if alive:
                raise ValueError(f"Alive slot {slot_idx} is missing a canonical UID")
            if parent_uid != -1:
                raise ValueError(f"Unbound slot {slot_idx} still carries parent UID {parent_uid}")
            continue

        if uid in active_uids:
            raise ValueError(f"Duplicate active UID binding detected for UID {uid}")
        active_uids.add(uid)
        if not alive:
            raise ValueError(f"Active UID {uid} is bound to dead slot {slot_idx}")

        record = uid_lifecycle.get(uid)
        if record is None:
            raise ValueError(f"Active UID {uid} is missing from the lifecycle ledger")
        if not record.is_active or record.current_slot != slot_idx or record.death_tick is not None:
            raise ValueError(f"Lifecycle record for active UID {uid} does not match slot {slot_idx}")
        if parent_uid != record.parent_uid:
            raise ValueError(f"Parent UID shadow mismatch for slot {slot_idx}")

    if cfg_obj.CHECKPOINT.STRICT_UID_VALIDATION:
        for uid, record in uid_lifecycle.items():
            if record.parent_uid != -1 and record.parent_uid not in uid_lifecycle:
                raise ValueError(f"UID {uid} references unknown parent UID {record.parent_uid}")
            if record.is_active:
                if record.current_slot is None:
                    raise ValueError(f"Active UID {uid} is missing its current slot")
                if int(slot_uid[record.current_slot].item()) != uid:
                    raise ValueError(f"Active UID {uid} lifecycle points to the wrong slot")
                if record.death_tick is not None:
                    raise ValueError(f"Active UID {uid} incorrectly has a death tick")
            else:
                if record.current_slot is not None:
                    raise ValueError(f"Historical UID {uid} still points at slot {record.current_slot}")
                if record.death_tick is None:
                    raise ValueError(f"Historical UID {uid} is missing a death tick")

    brain_state_by_uid = {int(uid): state for uid, state in bundle["brain_state_by_uid"].items()}
    if set(brain_state_by_uid.keys()) != active_uids:
        raise ValueError("Active brain snapshots do not match the active UID set")

    optimizer_state_by_uid = {int(uid): state for uid, state in bundle["ppo_state"].get("optimizer_state_by_uid", {}).items()}
    for uid in optimizer_state_by_uid:
        if uid not in brain_state_by_uid:
            raise ValueError(f"Optimizer state exists for UID {uid} without a matching active brain snapshot")

    buffer_state_by_uid = {int(uid): state for uid, state in bundle["ppo_state"].get("buffer_state_by_uid", {}).items()}
    for uid in buffer_state_by_uid:
        if uid not in uid_lifecycle:
            raise ValueError(f"Serialized PPO buffer exists for unknown UID {uid}")

    if cfg_obj.IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS:
        expected_uid_shadow = torch.where(slot_uid >= 0, slot_uid, torch.full_like(slot_uid, -1)).to(torch.float32)
        expected_parent_shadow = torch.where(slot_parent_uid >= 0, slot_parent_uid, torch.full_like(slot_parent_uid, -1)).to(torch.float32)
        if not torch.equal(data[Registry.AGENT_UID_SHADOW], expected_uid_shadow):
            raise ValueError("Legacy AGENT_ID shadow column does not match canonical slot_uid")
        if not torch.equal(data[Registry.PARENT_UID_SHADOW], expected_parent_shadow):
            raise ValueError("Legacy PARENT_ID shadow column does not match canonical slot_parent_uid")


def _move_optimizer_state_to_device(optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_runtime_checkpoint(runtime, bundle: dict) -> None:
    validate_runtime_checkpoint(bundle, cfg)

    registry = runtime.registry
    registry_state = bundle["registry_state"]
    registry.data.copy_(registry_state["data"].to(registry.device))
    registry.slot_uid = registry_state["slot_uid"].to(registry.device, dtype=torch.int64).clone()
    registry.slot_parent_uid = registry_state["slot_parent_uid"].to(registry.device, dtype=torch.int64).clone()
    registry.next_agent_uid = int(registry_state["next_agent_uid"])
    registry.next_unique_id = registry.next_agent_uid
    registry.fitness = registry_state["fitness"].to(registry.device).clone()
    registry.uid_lifecycle = deserialize_agent_lifecycle(registry_state["uid_lifecycle"])
    registry.active_uid_to_slot = {
        uid: record.current_slot
        for uid, record in registry.uid_lifecycle.items()
        if record.is_active and record.current_slot is not None
    }

    registry.brains = [None] * registry.max_agents
    for uid, slot_idx in registry.active_uid_to_slot.items():
        brain = Brain().to(registry.device)
        brain.load_state_dict(bundle["brain_state_by_uid"][uid])
        registry.brains[slot_idx] = brain

    grid_state = bundle["grid_state"]
    runtime.grid.grid.copy_(grid_state["grid"].to(runtime.grid.device))
    runtime.grid.hzones = copy.deepcopy(grid_state["hzones"])
    runtime.grid.next_hzone_id = int(grid_state["next_hzone_id"])

    runtime.engine.tick = int(bundle["engine_state"]["tick"])
    runtime.engine.respawn_controller.last_respawn_tick = int(bundle["engine_state"]["respawn_last_tick"])
    runtime.engine.last_obs_dict.clear()
    runtime.engine.last_dones_dict.clear()

    runtime.ppo.optimizers_by_uid = {}
    runtime.ppo.load_serialized_buffers(bundle["ppo_state"].get("buffer_state_by_uid", {}), device=cfg.SIM.DEVICE)
    for uid, optimizer_state in bundle["ppo_state"].get("optimizer_state_by_uid", {}).items():
        uid_int = int(uid)
        slot_idx = registry.get_slot_for_uid(uid_int)
        if slot_idx is None:
            raise ValueError(f"Cannot restore optimizer state for inactive UID {uid_int}")
        brain = registry.brains[slot_idx]
        runtime.ppo.validate_optimizer_state(uid_int, brain, optimizer_state)
        optimizer = runtime.ppo._get_optimizer(uid_int, brain)
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, registry.device)

    scaler_state = bundle["ppo_state"].get("scaler_state")
    if scaler_state is not None and runtime.ppo.scaler is not None:
        runtime.ppo.scaler.load_state_dict(scaler_state)

    rng_state = bundle.get("rng_state")
    if rng_state is not None and cfg.CHECKPOINT.CAPTURE_RNG_STATE:
        restore_rng_state(rng_state)

    registry.sync_identity_shadow_columns()
    registry.assert_identity_invariants()
    registry.check_invariants(runtime.grid)


def save_runtime_checkpoint(path: str | Path, bundle: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_runtime_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
