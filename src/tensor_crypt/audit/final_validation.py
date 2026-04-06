"""Prompt 7 audit and validation harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch

from ..checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    load_runtime_checkpoint,
    restore_rng_state,
    restore_runtime_checkpoint,
    save_runtime_checkpoint,
)
from ..config_bridge import cfg


def _tensor_digest(tensor: torch.Tensor) -> str:
    tensor_cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor_cpu.numpy().tobytes()).hexdigest()


def _brain_state_digest(brain) -> str:
    digest = hashlib.sha256()
    for name, tensor in brain.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_digest(tensor).encode("utf-8"))
    return digest.hexdigest()


def _runtime_signature(runtime) -> dict:
    registry = runtime.registry
    active = sorted((int(uid), int(slot)) for uid, slot in registry.active_uid_to_slot.items())
    alive_slots = sorted(int(slot) for slot in registry.get_alive_indices().tolist())
    return {
        "tick": int(runtime.engine.tick),
        "active_uid_to_slot": active,
        "alive_slots": alive_slots,
        "slot_uid": [int(value) for value in registry.slot_uid.tolist()],
        "slot_parent_uid": [int(value) for value in registry.slot_parent_uid.tolist()],
        "uid_family": {int(uid): str(family) for uid, family in sorted(registry.uid_family.items())},
        "uid_generation_depth": {int(uid): int(depth) for uid, depth in sorted(registry.uid_generation_depth.items())},
        "catastrophe_mode": runtime.engine.catastrophes.mode,
        "catastrophe_next_auto_tick": runtime.engine.catastrophes.build_status(runtime.engine.tick).get("next_auto_tick"),
        "registry_data_sha": _tensor_digest(registry.data),
        "registry_fitness_sha": _tensor_digest(registry.fitness),
        "grid_sha": _tensor_digest(runtime.grid.grid),
        "brain_state_sha_by_uid": {
            int(uid): _brain_state_digest(runtime.registry.brains[slot])
            for uid, slot in active
        },
        "ppo_updates": {
            int(uid): int(state.ppo_updates)
            for uid, state in sorted(runtime.ppo.training_state_by_uid.items())
        },
        "ppo_buffer_sizes": {
            int(uid): int(len(buffer))
            for uid, buffer in sorted(runtime.ppo.buffers_by_uid.items())
        },
        "optimizer_uids": sorted(int(uid) for uid in runtime.ppo.optimizers_by_uid.keys()),
    }


def save_load_save_surface_signature(runtime_factory: Callable[[], object], checkpoint_path: str | Path) -> dict:
    runtime_a = runtime_factory()
    bundle_a = capture_runtime_checkpoint(runtime_a)
    save_runtime_checkpoint(checkpoint_path, bundle_a)
    loaded = load_runtime_checkpoint(checkpoint_path)

    runtime_b = runtime_factory()
    restore_runtime_checkpoint(runtime_b, loaded)
    bundle_b = capture_runtime_checkpoint(runtime_b)

    return {
        "bundle_a_tick": int(bundle_a["engine_state"]["tick"]),
        "bundle_b_tick": int(bundle_b["engine_state"]["tick"]),
        "bundle_a_active_uid_count": len(bundle_a["brain_state_by_uid"]),
        "bundle_b_active_uid_count": len(bundle_b["brain_state_by_uid"]),
        "slot_uid_equal": bundle_a["registry_state"]["slot_uid"].tolist() == bundle_b["registry_state"]["slot_uid"].tolist(),
        "slot_parent_uid_equal": bundle_a["registry_state"]["slot_parent_uid"].tolist() == bundle_b["registry_state"]["slot_parent_uid"].tolist(),
        "uid_family_equal": bundle_a["registry_state"]["uid_family"] == bundle_b["registry_state"]["uid_family"],
        "uid_generation_depth_equal": bundle_a["registry_state"]["uid_generation_depth"] == bundle_b["registry_state"]["uid_generation_depth"],
        "buffer_keys_equal": sorted(bundle_a["ppo_state"]["buffer_state_by_uid"].keys()) == sorted(bundle_b["ppo_state"]["buffer_state_by_uid"].keys()),
        "training_state_keys_equal": sorted(bundle_a["ppo_state"]["training_state_by_uid"].keys()) == sorted(bundle_b["ppo_state"]["training_state_by_uid"].keys()),
        "registry_data_equal": _tensor_digest(bundle_a["registry_state"]["data"]) == _tensor_digest(bundle_b["registry_state"]["data"]),
        "fitness_equal": _tensor_digest(bundle_a["registry_state"]["fitness"]) == _tensor_digest(bundle_b["registry_state"]["fitness"]),
        "grid_equal": _tensor_digest(bundle_a["grid_state"]["grid"]) == _tensor_digest(bundle_b["grid_state"]["grid"]),
    }


def run_determinism_probe(runtime_factory: Callable[[], object], *, ticks: int | None = None) -> dict:
    ticks = int(ticks or cfg.VALIDATION.DETERMINISM_COMPARE_TICKS)
    runtime_a = runtime_factory()

    trace_a = []
    for _ in range(ticks):
        runtime_a.engine.step()
        trace_a.append(_runtime_signature(runtime_a))

    runtime_b = runtime_factory()
    trace_b = []
    for _ in range(ticks):
        runtime_b.engine.step()
        trace_b.append(_runtime_signature(runtime_b))

    return {
        "ticks": ticks,
        "match": trace_a == trace_b,
        "trace_a": trace_a,
        "trace_b": trace_b,
    }


def run_resume_consistency_probe(
    runtime_factory: Callable[[], object],
    checkpoint_path: str | Path,
    *,
    pre_ticks: int = 4,
    post_ticks: int = 4,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    base = runtime_factory()
    for _ in range(pre_ticks):
        base.engine.step()

    bundle = capture_runtime_checkpoint(base)
    save_runtime_checkpoint(checkpoint_path, bundle)
    loaded = load_runtime_checkpoint(checkpoint_path)

    if bundle.get("rng_state") is not None:
        restore_rng_state(bundle["rng_state"])

    for _ in range(post_ticks):
        base.engine.step()
    base_signature = _runtime_signature(base)

    resumed = runtime_factory()
    restore_runtime_checkpoint(resumed, loaded)
    for _ in range(post_ticks):
        resumed.engine.step()
    resumed_signature = _runtime_signature(resumed)

    return {
        "pre_ticks": int(pre_ticks),
        "post_ticks": int(post_ticks),
        "match": base_signature == resumed_signature,
        "base": base_signature,
        "resumed": resumed_signature,
    }


def run_catastrophe_repro_probe(runtime_factory: Callable[[], object], *, ticks: int = 8) -> dict:
    runtime_a = runtime_factory()

    trace_a = []
    for _ in range(int(ticks)):
        trace_a.append(runtime_a.engine.catastrophes.build_status(runtime_a.engine.tick))
        runtime_a.engine.step()

    runtime_b = runtime_factory()
    trace_b = []
    for _ in range(int(ticks)):
        trace_b.append(runtime_b.engine.catastrophes.build_status(runtime_b.engine.tick))
        runtime_b.engine.step()

    return {
        "ticks": int(ticks),
        "match": trace_a == trace_b,
        "trace_a": trace_a,
        "trace_b": trace_b,
    }


def _skipped_check(reason: str) -> dict:
    return {"enabled": False, "skipped": True, "reason": reason}


def run_final_validation_suite(
    runtime_factory: Callable[[], object],
    checkpoint_path: str | Path,
    *,
    ticks: int | None = None,
) -> dict:
    ticks = int(ticks or cfg.VALIDATION.AUDIT_DEFAULT_TICKS)
    if not cfg.VALIDATION.ENABLE_FINAL_AUDIT_HARNESS:
        return {
            "determinism": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "resume": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "catastrophe": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "save_load_save": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "all_passed": True,
            "skipped": True,
        }

    report = {
        "determinism": _skipped_check("ENABLE_DETERMINISM_TESTS is disabled"),
        "resume": _skipped_check("ENABLE_RESUME_CONSISTENCY_TESTS is disabled"),
        "catastrophe": _skipped_check("ENABLE_CATASTROPHE_REPRO_TESTS is disabled"),
        "save_load_save": _skipped_check("ENABLE_SAVE_LOAD_SAVE_TESTS is disabled"),
    }

    if cfg.VALIDATION.ENABLE_DETERMINISM_TESTS:
        report["determinism"] = run_determinism_probe(
            runtime_factory,
            ticks=min(ticks, cfg.VALIDATION.DETERMINISM_COMPARE_TICKS),
        )
    if cfg.VALIDATION.ENABLE_RESUME_CONSISTENCY_TESTS:
        report["resume"] = run_resume_consistency_probe(
            runtime_factory,
            checkpoint_path,
            pre_ticks=max(1, ticks // 2),
            post_ticks=max(1, ticks // 2),
        )
    if cfg.VALIDATION.ENABLE_CATASTROPHE_REPRO_TESTS:
        report["catastrophe"] = run_catastrophe_repro_probe(runtime_factory, ticks=min(ticks, 8))
    if cfg.VALIDATION.ENABLE_SAVE_LOAD_SAVE_TESTS:
        report["save_load_save"] = save_load_save_surface_signature(runtime_factory, checkpoint_path)

    enabled_results = []
    if cfg.VALIDATION.ENABLE_DETERMINISM_TESTS:
        enabled_results.append(bool(report["determinism"]["match"]))
    if cfg.VALIDATION.ENABLE_RESUME_CONSISTENCY_TESTS:
        enabled_results.append(bool(report["resume"]["match"]))
    if cfg.VALIDATION.ENABLE_CATASTROPHE_REPRO_TESTS:
        enabled_results.append(bool(report["catastrophe"]["match"]))
    if cfg.VALIDATION.ENABLE_SAVE_LOAD_SAVE_TESTS:
        enabled_results.extend(
            [
                bool(report["save_load_save"]["slot_uid_equal"]),
                bool(report["save_load_save"]["slot_parent_uid_equal"]),
                bool(report["save_load_save"]["uid_family_equal"]),
                bool(report["save_load_save"]["registry_data_equal"]),
                bool(report["save_load_save"]["fitness_equal"]),
                bool(report["save_load_save"]["grid_equal"]),
            ]
        )
    report["all_passed"] = all(enabled_results) if enabled_results else True
    return report




