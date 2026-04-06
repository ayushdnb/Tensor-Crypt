"""Prompt 7 audit and validation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    load_runtime_checkpoint,
    restore_runtime_checkpoint,
    save_runtime_checkpoint,
)
from ..config_bridge import cfg


def _runtime_signature(runtime) -> dict:
    registry = runtime.registry
    active = sorted((int(uid), int(slot)) for uid, slot in registry.active_uid_to_slot.items())
    return {
        "tick": int(runtime.engine.tick),
        "active_uid_to_slot": active,
        "slot_uid": [int(value) for value in registry.slot_uid.tolist()],
        "slot_parent_uid": [int(value) for value in registry.slot_parent_uid.tolist()],
        "uid_family": {int(uid): str(family) for uid, family in sorted(registry.uid_family.items())},
        "uid_generation_depth": {int(uid): int(depth) for uid, depth in sorted(registry.uid_generation_depth.items())},
        "catastrophe_mode": runtime.engine.catastrophes.mode,
        "catastrophe_next_auto_tick": runtime.engine.catastrophes.build_status(runtime.engine.tick).get("next_auto_tick"),
        "ppo_updates": {
            int(uid): int(state.ppo_updates)
            for uid, state in sorted(runtime.ppo.training_state_by_uid.items())
        },
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
    }


def run_determinism_probe(runtime_factory: Callable[[], object], *, ticks: int | None = None) -> dict:
    ticks = int(ticks or cfg.VALIDATION.DETERMINISM_COMPARE_TICKS)
    runtime_a = runtime_factory()
    runtime_b = runtime_factory()

    trace_a = []
    trace_b = []
    for _ in range(ticks):
        runtime_a.engine.step()
        runtime_b.engine.step()
        trace_a.append(_runtime_signature(runtime_a))
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
    resumed = runtime_factory()

    for _ in range(pre_ticks):
        base.engine.step()

    bundle = capture_runtime_checkpoint(base)
    save_runtime_checkpoint(checkpoint_path, bundle)
    loaded = load_runtime_checkpoint(checkpoint_path)
    restore_runtime_checkpoint(resumed, loaded)

    for _ in range(post_ticks):
        base.engine.step()
        resumed.engine.step()

    return {
        "pre_ticks": int(pre_ticks),
        "post_ticks": int(post_ticks),
        "match": _runtime_signature(base) == _runtime_signature(resumed),
        "base": _runtime_signature(base),
        "resumed": _runtime_signature(resumed),
    }


def run_catastrophe_repro_probe(runtime_factory: Callable[[], object], *, ticks: int = 8) -> dict:
    runtime_a = runtime_factory()
    runtime_b = runtime_factory()

    trace_a = []
    trace_b = []
    for _ in range(int(ticks)):
        trace_a.append(runtime_a.engine.catastrophes.build_status(runtime_a.engine.tick))
        trace_b.append(runtime_b.engine.catastrophes.build_status(runtime_b.engine.tick))
        runtime_a.engine.step()
        runtime_b.engine.step()

    return {
        "ticks": int(ticks),
        "match": trace_a == trace_b,
        "trace_a": trace_a,
        "trace_b": trace_b,
    }


def run_final_validation_suite(
    runtime_factory: Callable[[], object],
    checkpoint_path: str | Path,
    *,
    ticks: int | None = None,
) -> dict:
    ticks = int(ticks or cfg.VALIDATION.AUDIT_DEFAULT_TICKS)
    report = {
        "determinism": run_determinism_probe(runtime_factory, ticks=min(ticks, cfg.VALIDATION.DETERMINISM_COMPARE_TICKS)),
        "resume": run_resume_consistency_probe(runtime_factory, checkpoint_path, pre_ticks=max(1, ticks // 2), post_ticks=max(1, ticks // 2)),
        "catastrophe": run_catastrophe_repro_probe(runtime_factory, ticks=min(ticks, 8)),
        "save_load_save": save_load_save_surface_signature(runtime_factory, checkpoint_path),
    }
    report["all_passed"] = all(
        [
            bool(report["determinism"]["match"]),
            bool(report["resume"]["match"]),
            bool(report["catastrophe"]["match"]),
            bool(report["save_load_save"]["slot_uid_equal"]),
            bool(report["save_load_save"]["slot_parent_uid_equal"]),
            bool(report["save_load_save"]["uid_family_equal"]),
        ]
    )
    return report
