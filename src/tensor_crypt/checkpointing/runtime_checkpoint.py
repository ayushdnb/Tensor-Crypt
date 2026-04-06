"""Runtime checkpoint capture, validation, and restore helpers.

Restore order is intentionally conservative: registry bindings are rebuilt
before brains, then PPO state is attached to the reconstituted active UIDs.
That ordering is part of the checkpoint ownership contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from ..agents.brain import create_brain, validate_bloodline_family
from ..agents.state_registry import AgentLifecycleRecord
from ..config_bridge import cfg
from ..learning.ppo import PPO
from .atomic_checkpoint import (
    atomic_save_checkpoint_files,
    load_checkpoint_bundle,
    validate_checkpoint_file_set,
)


def _schema_versions_dict(cfg_obj) -> dict:
    return {
        "IDENTITY_SCHEMA_VERSION": cfg_obj.SCHEMA.IDENTITY_SCHEMA_VERSION,
        "OBS_SCHEMA_VERSION": cfg_obj.SCHEMA.OBS_SCHEMA_VERSION,
        "PPO_STATE_SCHEMA_VERSION": cfg_obj.SCHEMA.PPO_STATE_SCHEMA_VERSION,
        "CHECKPOINT_SCHEMA_VERSION": cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "REPRODUCTION_SCHEMA_VERSION": cfg_obj.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
        "CATASTROPHE_SCHEMA_VERSION": cfg_obj.SCHEMA.CATASTROPHE_SCHEMA_VERSION,
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


def _config_fingerprint(bundle: dict) -> str:
    payload = json.dumps(bundle.get("config_snapshot", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    registry = runtime.registry
    brain_state_by_uid = {}
    brain_metadata_by_uid = {}
    for uid, slot_idx in registry.active_uid_to_slot.items():
        brain = registry.brains[slot_idx]
        brain_state_by_uid[uid] = _clone_nested_cpu(brain.state_dict())
        brain_metadata_by_uid[uid] = {
            "family_id": registry.get_family_for_uid(uid),
            "topology_signature": list(brain.get_topology_signature()),
        }

    optimizer_state_by_uid = {}
    optimizer_metadata_by_uid = {}
    if cfg.CHECKPOINT.CAPTURE_OPTIMIZER_STATE:
        for uid, optimizer in runtime.ppo.optimizers_by_uid.items():
            slot_idx = registry.get_slot_for_uid(uid)
            if slot_idx is None:
                raise ValueError(f"Cannot capture optimizer state for inactive UID {uid}")
            brain = registry.brains[slot_idx]
            optimizer_state_by_uid[uid] = _clone_nested_cpu(optimizer.state_dict())
            optimizer_metadata_by_uid[uid] = runtime.ppo.build_optimizer_metadata(brain, optimizer)

    scaler_state = None
    if cfg.CHECKPOINT.CAPTURE_SCALER_STATE and runtime.ppo.scaler is not None:
        scaler_state = _clone_nested_cpu(runtime.ppo.scaler.state_dict())

    catastrophe_state = None
    if cfg.CATASTROPHE.PERSIST_STATE_IN_CHECKPOINTS:
        catastrophe_state = runtime.engine.catastrophes.serialize()

    return {
        "checkpoint_schema_version": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "schema_versions": _schema_versions_dict(cfg),
        "config_snapshot": asdict(cfg),
        "engine_state": {
            "tick": int(runtime.engine.tick),
            "respawn_last_tick": int(runtime.engine.respawn_controller.last_respawn_tick),
            "catastrophe_state": catastrophe_state,
        },
        "registry_state": {
            "data": registry.data.detach().cpu().clone(),
            "slot_uid": registry.slot_uid.detach().cpu().clone(),
            "slot_parent_uid": registry.slot_parent_uid.detach().cpu().clone(),
            "next_agent_uid": int(registry.next_agent_uid),
            "fitness": registry.fitness.detach().cpu().clone(),
            "uid_lifecycle": serialize_agent_lifecycle(registry.uid_lifecycle),
            "uid_family": copy.deepcopy(registry.uid_family),
            "uid_parent_roles": copy.deepcopy(registry.uid_parent_roles),
            "uid_trait_latent": copy.deepcopy(registry.uid_trait_latent),
            "uid_generation_depth": copy.deepcopy(registry.uid_generation_depth),
        },
        "grid_state": {
            "grid": runtime.grid.grid.detach().cpu().clone(),
            "hzones": copy.deepcopy(runtime.grid.hzones),
            "next_hzone_id": int(runtime.grid.next_hzone_id),
        },
        "brain_state_by_uid": brain_state_by_uid,
        "brain_metadata_by_uid": brain_metadata_by_uid,
        "ppo_state": {
            "buffer_state_by_uid": runtime.ppo.serialize_all_buffers(),
            "training_state_by_uid": runtime.ppo.serialize_training_state(),
            "optimizer_state_by_uid": optimizer_state_by_uid,
            "optimizer_metadata_by_uid": optimizer_metadata_by_uid,
            "scaler_state": scaler_state,
        },
        "rng_state": capture_rng_state() if cfg.CHECKPOINT.CAPTURE_RNG_STATE else None,
        "metadata": {
            "device": str(cfg.SIM.DEVICE),
            "amp_enabled": bool(cfg.LOG.AMP),
            "config_fingerprint": _config_fingerprint({"config_snapshot": asdict(cfg)}),
        },
    }


def validate_runtime_checkpoint(bundle: dict, cfg_obj, *, manifest: dict | None = None) -> None:
    required_top_level = {
        "checkpoint_schema_version",
        "schema_versions",
        "config_snapshot",
        "engine_state",
        "registry_state",
        "grid_state",
        "brain_state_by_uid",
        "brain_metadata_by_uid",
        "ppo_state",
        "metadata",
    }
    missing_top_level = sorted(required_top_level - set(bundle.keys()))
    if missing_top_level:
        raise ValueError(f"Checkpoint is missing required top-level keys: {missing_top_level}")

    if int(bundle["checkpoint_schema_version"]) != int(cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION):
        raise ValueError("Checkpoint schema mismatch")

    schema_versions = bundle.get("schema_versions", {})
    if int(schema_versions.get("CATASTROPHE_SCHEMA_VERSION", cfg_obj.SCHEMA.CATASTROPHE_SCHEMA_VERSION)) != int(cfg_obj.SCHEMA.CATASTROPHE_SCHEMA_VERSION):
        raise ValueError("Checkpoint catastrophe schema mismatch")
    if int(schema_versions.get("TELEMETRY_SCHEMA_VERSION", cfg_obj.SCHEMA.TELEMETRY_SCHEMA_VERSION)) != int(cfg_obj.SCHEMA.TELEMETRY_SCHEMA_VERSION):
        raise ValueError("Checkpoint telemetry schema mismatch")

    registry_state = bundle["registry_state"]
    for key in ("data", "slot_uid", "slot_parent_uid", "fitness", "uid_lifecycle", "uid_family", "uid_parent_roles", "uid_trait_latent", "uid_generation_depth"):
        if key not in registry_state:
            raise ValueError(f"Checkpoint registry_state is missing '{key}'")

    uid_lifecycle = deserialize_agent_lifecycle(registry_state["uid_lifecycle"])
    uid_family = {int(uid): str(family_id) for uid, family_id in registry_state["uid_family"].items()}
    uid_parent_roles = {int(uid): {str(k): int(v) for k, v in payload.items()} for uid, payload in registry_state["uid_parent_roles"].items()}
    uid_trait_latent = {int(uid): {str(k): float(v) for k, v in payload.items()} for uid, payload in registry_state["uid_trait_latent"].items()}
    uid_generation_depth = {int(uid): int(depth) for uid, depth in registry_state["uid_generation_depth"].items()}

    if not (set(uid_lifecycle.keys()) == set(uid_family.keys()) == set(uid_parent_roles.keys())):
        raise ValueError("Checkpoint lineage ledgers are inconsistent")
    if set(uid_lifecycle.keys()) != set(uid_trait_latent.keys()) or set(uid_lifecycle.keys()) != set(uid_generation_depth.keys()):
        raise ValueError("Checkpoint trait or generation ledgers are inconsistent")

    slot_uid = registry_state["slot_uid"]
    active_uids = [int(uid) for uid in slot_uid.tolist() if int(uid) >= 0]
    if len(active_uids) != len(set(active_uids)):
        raise ValueError("Duplicate active UID in checkpoint slot bindings")

    active_uid_set = set(active_uids)
    known_uids = set(uid_lifecycle.keys())
    for uid in active_uid_set:
        if uid not in known_uids:
            raise ValueError(f"Checkpoint slot bindings reference unknown UID {uid}")

    for surface_name in ("brain_state_by_uid", "brain_metadata_by_uid"):
        surface_uids = {int(uid) for uid in bundle[surface_name].keys()}
        if surface_uids != active_uid_set:
            raise ValueError(f"Checkpoint {surface_name} does not match active UID bindings")

    for surface_name in ("training_state_by_uid", "buffer_state_by_uid", "optimizer_state_by_uid", "optimizer_metadata_by_uid"):
        surface = bundle["ppo_state"].get(surface_name, {})
        unknown_uids = sorted(int(uid) for uid in surface.keys() if int(uid) not in known_uids)
        if unknown_uids:
            raise ValueError(f"Checkpoint {surface_name} references unknown UID {unknown_uids[0]}")

    for uid in uid_lifecycle:
        validate_bloodline_family(uid_family[uid])
        roles = uid_parent_roles[uid]
        for key in ("brain_parent_uid", "trait_parent_uid", "anchor_parent_uid"):
            if key not in roles:
                raise ValueError(f"Checkpoint parent roles missing {key} for UID {uid}")

    expected_topology_by_family: dict[str, list[list | tuple]] = {}
    for uid, metadata in bundle["brain_metadata_by_uid"].items():
        family_id = validate_bloodline_family(metadata["family_id"])
        if family_id not in expected_topology_by_family:
            expected_topology_by_family[family_id] = [list(item) for item in create_brain(family_id).get_topology_signature()]
        actual_topology = [list(item) for item in metadata.get("topology_signature", [])]
        if actual_topology != expected_topology_by_family[family_id]:
            raise ValueError(f"Checkpoint brain topology signature mismatch for UID {uid}")

    for uid, payload in bundle["ppo_state"]["buffer_state_by_uid"].items():
        PPO.validate_serialized_buffer_payload(int(uid), payload)

    catastrophe_state = bundle["engine_state"].get("catastrophe_state")
    if catastrophe_state is not None and cfg_obj.CATASTROPHE.STRICT_CHECKPOINT_VALIDATION:
        if int(catastrophe_state.get("schema_version", cfg_obj.SCHEMA.CATASTROPHE_SCHEMA_VERSION)) != int(cfg_obj.SCHEMA.CATASTROPHE_SCHEMA_VERSION):
            raise ValueError("Checkpoint catastrophe state has wrong schema version")

    if manifest is not None:
        if int(manifest["checkpoint_schema_version"]) != int(bundle["checkpoint_schema_version"]):
            raise ValueError("Checkpoint manifest schema version does not match bundle")
        if int(manifest["tick"]) != int(bundle["engine_state"]["tick"]):
            raise ValueError("Checkpoint manifest tick does not match bundle")
        if int(manifest["active_uid_count"]) != int(len(bundle["brain_state_by_uid"])):
            raise ValueError("Checkpoint manifest active UID count does not match bundle")
        manifest_fingerprint = manifest.get("config_fingerprint")
        if manifest_fingerprint and cfg_obj.CHECKPOINT.STRICT_CONFIG_FINGERPRINT_VALIDATION:
            if manifest_fingerprint != _config_fingerprint(bundle):
                raise ValueError("Checkpoint manifest config fingerprint does not match bundle")


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
    registry.uid_family = {int(uid): str(family_id) for uid, family_id in registry_state["uid_family"].items()}
    registry.uid_parent_roles = {int(uid): {str(k): int(v) for k, v in payload.items()} for uid, payload in registry_state["uid_parent_roles"].items()}
    registry.uid_trait_latent = {int(uid): {str(k): float(v) for k, v in payload.items()} for uid, payload in registry_state["uid_trait_latent"].items()}
    registry.uid_generation_depth = {int(uid): int(depth) for uid, depth in registry_state["uid_generation_depth"].items()}
    registry.active_uid_to_slot = {
        uid: record.current_slot
        for uid, record in registry.uid_lifecycle.items()
        if record.is_active and record.current_slot is not None
    }
    registry.slot_family = [None] * registry.max_agents
    registry.brains = [None] * registry.max_agents

    brain_metadata_by_uid = {int(uid): payload for uid, payload in bundle["brain_metadata_by_uid"].items()}
    brain_state_by_uid = {int(uid): payload for uid, payload in bundle["brain_state_by_uid"].items()}
    for uid, slot_idx in registry.active_uid_to_slot.items():
        family_id = validate_bloodline_family(brain_metadata_by_uid[uid]["family_id"])
        registry.slot_family[slot_idx] = family_id
        brain = create_brain(family_id).to(registry.device)
        brain.load_state_dict(brain_state_by_uid[uid])
        registry.brains[slot_idx] = brain

    runtime.grid.grid.copy_(bundle["grid_state"]["grid"].to(runtime.grid.device))
    runtime.grid.hzones = copy.deepcopy(bundle["grid_state"]["hzones"])
    runtime.grid.next_hzone_id = int(bundle["grid_state"]["next_hzone_id"])
    runtime.engine.tick = int(bundle["engine_state"]["tick"])
    runtime.engine.respawn_controller.last_respawn_tick = int(bundle["engine_state"]["respawn_last_tick"])

    ppo_state = bundle["ppo_state"]
    runtime.ppo.optimizers_by_uid = {}
    runtime.ppo.load_serialized_buffers(ppo_state.get("buffer_state_by_uid", {}), device=cfg.SIM.DEVICE)
    runtime.ppo.load_serialized_training_state(ppo_state.get("training_state_by_uid", {}))
    optimizer_metadata_by_uid = {int(uid): payload for uid, payload in ppo_state.get("optimizer_metadata_by_uid", {}).items()}
    for uid, optimizer_state in ppo_state.get("optimizer_state_by_uid", {}).items():
        uid_int = int(uid)
        slot_idx = registry.get_slot_for_uid(uid_int)
        brain = registry.brains[slot_idx]
        runtime.ppo.validate_optimizer_state(uid_int, brain, optimizer_state, optimizer_metadata_by_uid.get(uid_int))
        optimizer = runtime.ppo._get_optimizer(uid_int, brain)
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, registry.device)

    scaler_state = ppo_state.get("scaler_state")
    if scaler_state is not None and runtime.ppo.scaler is not None:
        runtime.ppo.scaler.load_state_dict(scaler_state)

    rng_state = bundle.get("rng_state")
    if rng_state is not None and cfg.CHECKPOINT.CAPTURE_RNG_STATE:
        restore_rng_state(rng_state)

    catastrophe_state = bundle["engine_state"].get("catastrophe_state")
    if catastrophe_state is not None and cfg.CATASTROPHE.PERSIST_STATE_IN_CHECKPOINTS:
        runtime.engine.catastrophes.restore(catastrophe_state)
    else:
        runtime.engine.catastrophes.reset()

    registry.sync_identity_shadow_columns()
    registry.assert_identity_invariants()
    registry.check_invariants(runtime.grid)


def save_runtime_checkpoint(path: str | Path, bundle: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED and cfg.CHECKPOINT.MANIFEST_ENABLED:
        atomic_save_checkpoint_files(path, bundle)
        return
    torch.save(bundle, path)


def load_runtime_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if cfg.CHECKPOINT.STRICT_MANIFEST_VALIDATION and path.exists():
        try:
            bundle, manifest = load_checkpoint_bundle(path)
            validate_runtime_checkpoint(bundle, cfg, manifest=manifest)
            return bundle
        except FileNotFoundError:
            pass
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    validate_runtime_checkpoint(bundle, cfg)
    return bundle


def validate_checkpoint_artifacts(path: str | Path) -> dict:
    path = Path(path)
    manifest = validate_checkpoint_file_set(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    validate_runtime_checkpoint(bundle, cfg, manifest=manifest)
    return manifest

