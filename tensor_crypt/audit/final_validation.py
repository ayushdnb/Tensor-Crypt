"""Determinism, resume, and checkpoint validation harness."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import h5py
import pandas as pd
import torch

from ..app.lifecycle import finalize_runtime
from ..app.runtime import build_resume_runtime
from ..checkpointing.resume_policy import resolve_resume_request
from ..checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    load_runtime_checkpoint,
    restore_rng_state,
    restore_runtime_checkpoint,
    save_runtime_checkpoint,
)
from ..config_bridge import cfg
from ..simulation.engine import (
    SAVE_REASON_MANUAL_OPERATOR,
    SAVE_REASON_SCHEDULED_TICK,
    SAVE_REASON_SCHEDULED_WALLCLOCK,
    SAVE_REASON_SHUTDOWN,
)
from ..telemetry.run_paths import prepare_checkpoint_backed_session_plan, session_metadata_path_for


def _restore_cfg(snapshot) -> None:
    for field in dataclasses.fields(snapshot):
        setattr(cfg, field.name, copy.deepcopy(getattr(snapshot, field.name)))


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
        "respawn_overlay_runtime_state": runtime.engine.respawn_controller.serialize_runtime_state(),
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


def run_resume_chain_probe(
    runtime_factory: Callable[[], object],
    checkpoint_dir: str | Path,
    *,
    cycles: int = 3,
    ticks_per_cycle: int = 4,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resumed = runtime_factory()
    cycle_reports = []
    elapsed_ticks = 0

    for cycle in range(1, int(cycles) + 1):
        for _ in range(int(ticks_per_cycle)):
            resumed.engine.step()
        elapsed_ticks += int(ticks_per_cycle)

        checkpoint_path = checkpoint_dir / f"resume_chain_cycle_{cycle:02d}.pt"
        bundle = capture_runtime_checkpoint(resumed)
        save_runtime_checkpoint(checkpoint_path, bundle)
        loaded = load_runtime_checkpoint(checkpoint_path)

        replacement = runtime_factory()
        restore_runtime_checkpoint(replacement, loaded)
        reference = runtime_factory()
        for _ in range(elapsed_ticks):
            reference.engine.step()

        control_signature = _runtime_signature(reference)
        resumed_signature = _runtime_signature(replacement)
        cycle_reports.append(
            {
                "cycle": int(cycle),
                "elapsed_ticks": int(elapsed_ticks),
                "checkpoint_path": str(checkpoint_path),
                "match": control_signature == resumed_signature,
                "control": control_signature,
                "resumed": resumed_signature,
            }
        )
        resumed = replacement

    return {
        "cycles": int(cycles),
        "ticks_per_cycle": int(ticks_per_cycle),
        "total_ticks": int(cycles) * int(ticks_per_cycle),
        "match": all(report["match"] for report in cycle_reports),
        "cycle_reports": cycle_reports,
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


def run_resume_policy_probe(runtime_factory: Callable[[], object], checkpoint_path: str | Path) -> dict:
    snapshot = copy.deepcopy(cfg)
    checkpoint_path = Path(checkpoint_path)
    try:
        runtime = runtime_factory()
        bundle = capture_runtime_checkpoint(runtime)
        save_runtime_checkpoint(checkpoint_path, bundle)

        exact_report = resolve_resume_request(
            requested_mode="resume_exact",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )

        original_log_cadence = int(cfg.LOG.LOG_TICK_EVERY)
        cfg.LOG.LOG_TICK_EVERY = original_log_cadence + 1
        exact_with_drift_report = resolve_resume_request(
            requested_mode="resume_exact",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )
        drift_report = resolve_resume_request(
            requested_mode="resume_with_drift",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )

        cfg.LOG.LOG_TICK_EVERY = original_log_cadence
        cfg.PPO.LR = float(cfg.PPO.LR) * 1.5
        drift_with_fork_delta_report = resolve_resume_request(
            requested_mode="resume_with_drift",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )
        fork_report = resolve_resume_request(
            requested_mode="fork_from_checkpoint",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )

        _restore_cfg(snapshot)
        legacy_bundle = copy.deepcopy(bundle)
        legacy_bundle.get("metadata", {}).pop("resume_contract", None)
        legacy_bundle["rng_state"] = None
        legacy_report = resolve_resume_request(
            requested_mode="resume_exact",
            bundle=legacy_bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )

        checks = {
            "exact_accepts_clean_checkpoint": exact_report["allowed"] and exact_report["resolved_mode"] == "resume_exact",
            "exact_rejects_drift_surface": not exact_with_drift_report["allowed"]
            and exact_with_drift_report["failure_class"] == "drift_acknowledgment_required",
            "drift_accepts_drift_surface": drift_report["allowed"] and drift_report["resolved_mode"] == "resume_with_drift",
            "drift_rejects_fork_only_surface": not drift_with_fork_delta_report["allowed"]
            and drift_with_fork_delta_report["failure_class"] == "fork_semantics_required",
            "fork_accepts_fork_only_surface": fork_report["allowed"] and fork_report["resolved_mode"] == "fork_from_checkpoint",
            "legacy_inference_blocks_exact_when_rng_missing": not legacy_report["allowed"]
            and legacy_report["legacy_contract_inference_used"]
            and "rng_state_missing" in legacy_report["exact_resume_completeness_deficits"],
        }
        return {
            "match": all(checks.values()),
            "checks": checks,
            "reports": {
                "exact": exact_report,
                "exact_with_drift": exact_with_drift_report,
                "drift": drift_report,
                "drift_with_fork_delta": drift_with_fork_delta_report,
                "fork": fork_report,
                "legacy": legacy_report,
            },
        }
    finally:
        _restore_cfg(snapshot)


def _checkpoint_manifests(run_dir: str | Path) -> list[dict]:
    checkpoint_dir = Path(run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME
    manifests = []
    for path in sorted(checkpoint_dir.glob(f"{cfg.CHECKPOINT.FILENAME_PREFIX}*{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}{cfg.CHECKPOINT.MANIFEST_FILENAME_SUFFIX}")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_manifest_path"] = str(path)
        manifests.append(payload)
    return manifests


def _path_is_under(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def run_manual_checkpoint_probe(runtime_factory: Callable[[], object]) -> dict:
    snapshot = copy.deepcopy(cfg)
    try:
        cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
        cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS = True
        runtime = runtime_factory()
        checkpoint_path = runtime.engine.publish_runtime_checkpoint(SAVE_REASON_MANUAL_OPERATOR, force=True)
        checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        manifests = _checkpoint_manifests(runtime.data_logger.run_dir)
        bundle = load_runtime_checkpoint(checkpoint_path) if checkpoint_path is not None else {}
        lifecycle = bundle.get("metadata", {}).get("runtime_lifecycle", {})
        metadata_path = session_metadata_path_for(runtime.session_plan)
        session_metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                session_metadata = json.load(handle)

        matching_manifest = None
        if checkpoint_path is not None:
            for manifest in manifests:
                if Path(manifest.get("_manifest_path", "")).name.startswith(checkpoint_path.name):
                    matching_manifest = manifest
                    break
                if manifest.get("bundle_path") and Path(manifest["bundle_path"]).name == checkpoint_path.name:
                    matching_manifest = manifest
                    break
        manifest_reason = None if matching_manifest is None else matching_manifest.get("save_reason")
        return {
            "match": (
                checkpoint_path is not None
                and checkpoint_path.exists()
                and lifecycle.get("save_reason") == SAVE_REASON_MANUAL_OPERATOR
                and manifest_reason == SAVE_REASON_MANUAL_OPERATOR
                and session_metadata.get("last_checkpoint_reason") == SAVE_REASON_MANUAL_OPERATOR
            ),
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "lifecycle_save_reason": lifecycle.get("save_reason"),
            "manifest_save_reason": manifest_reason,
            "session_last_checkpoint_reason": session_metadata.get("last_checkpoint_reason"),
            "manifest_count": len(manifests),
        }
    finally:
        _restore_cfg(snapshot)


def run_selected_brain_export_probe(runtime_factory: Callable[[], object]) -> dict:
    runtime = runtime_factory()
    alive_slots = [int(slot_idx) for slot_idx in runtime.registry.get_alive_indices().detach().cpu().tolist()]
    if not alive_slots:
        return {
            "match": False,
            "reason": "runtime has no live selected-agent candidate",
        }

    slot_idx = alive_slots[0]
    uid = int(runtime.registry.get_uid_for_slot(slot_idx))
    family_id = runtime.registry.get_family_for_uid(uid)
    brain = runtime.registry.brains[slot_idx]
    result = runtime.data_logger.export_selected_brain(
        registry=runtime.registry,
        ppo=runtime.ppo,
        slot_idx=slot_idx,
        tick=int(runtime.engine.tick),
    )
    pt_path = Path(result["path"])
    json_path = Path(result["metadata_path"])
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)

    expected_topology = [
        [str(name), [int(dim) for dim in shape]]
        for name, shape in brain.get_topology_signature()
    ]
    return {
        "match": (
            pt_path.exists()
            and json_path.exists()
            and _path_is_under(pt_path, runtime.data_logger.brains_dir)
            and _path_is_under(json_path, runtime.data_logger.brains_dir)
            and metadata.get("uid") == uid
            and metadata.get("slot_at_export") == slot_idx
            and metadata.get("family_id") == family_id
            and metadata.get("parameter_count") == int(brain.get_param_count())
            and metadata.get("topology_signature") == expected_topology
            and payload.get("metadata", {}).get("uid") == uid
            and "state_dict" in payload
        ),
        "uid": uid,
        "slot_idx": slot_idx,
        "family_id": family_id,
        "export_path": str(pt_path),
        "metadata_path": str(json_path),
        "under_brains_dir": _path_is_under(pt_path, runtime.data_logger.brains_dir),
        "metadata": metadata,
    }


def run_wallclock_autosave_probe(runtime_factory: Callable[[], object]) -> dict:
    snapshot = copy.deepcopy(cfg)
    try:
        cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
        cfg.CHECKPOINT.ENABLE_WALLCLOCK_AUTOSAVE = True
        cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_INTERVAL_SECONDS = 0.001
        cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_MIN_TICKS_ADVANCED = 1
        cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_WHILE_PAUSED = False
        runtime = runtime_factory()
        runtime.engine._wallclock_autosave_started_at = time.monotonic() - 1.0
        runtime.engine.step()
        first_count = len(_checkpoint_manifests(runtime.data_logger.run_dir))
        runtime.engine.last_runtime_checkpoint_wallclock_time = time.monotonic() - 1.0
        paused_result = runtime.engine.maybe_save_runtime_checkpoint_wallclock(paused=True)
        manifests = _checkpoint_manifests(runtime.data_logger.run_dir)
        reasons = [manifest.get("save_reason") for manifest in manifests]
        return {
            "match": (
                SAVE_REASON_SCHEDULED_WALLCLOCK in reasons
                and first_count == 1
                and paused_result is None
                and len(manifests) == first_count
            ),
            "reasons": reasons,
            "manifest_count": len(manifests),
            "paused_result": None if paused_result is None else str(paused_result),
        }
    finally:
        _restore_cfg(snapshot)


def run_shutdown_checkpoint_probe(runtime_factory: Callable[[], object]) -> dict:
    snapshot = copy.deepcopy(cfg)
    try:
        cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
        cfg.CHECKPOINT.ENABLE_SHUTDOWN_CHECKPOINT = True
        runtime = runtime_factory()
        runtime.engine.step()
        result = finalize_runtime(runtime, close_reason="probe_shutdown")
        second = finalize_runtime(runtime, close_reason="probe_shutdown_second")
        manifests = _checkpoint_manifests(runtime.data_logger.run_dir)
        reasons = [manifest.get("save_reason") for manifest in manifests]
        return {
            "match": (
                SAVE_REASON_SHUTDOWN in reasons
                and result.checkpoint_path is not None
                and result.logger_closed
                and second.already_finalized
                and getattr(runtime.data_logger, "_closed", False)
            ),
            "first_result": result.__dict__,
            "second_result": second.__dict__,
            "reasons": reasons,
        }
    finally:
        _restore_cfg(snapshot)


def run_logger_close_once_probe(runtime_factory: Callable[[], object]) -> dict:
    runtime = runtime_factory()
    runtime.engine.step()
    runtime.data_logger.close(
        runtime.registry,
        finalize_open_lives=False,
        close_reason="probe_close_once",
        close_tick=runtime.engine.tick,
    )
    runtime.data_logger.close(
        runtime.registry,
        finalize_open_lives=False,
        close_reason="probe_close_once_second",
        close_tick=runtime.engine.tick,
    )
    return {
        "match": bool(getattr(runtime.data_logger, "_closed", False)),
        "closed": bool(getattr(runtime.data_logger, "_closed", False)),
    }


def _save_source_checkpoint_for_lifecycle_probe(runtime_factory: Callable[[], object], checkpoint_name: str):
    cfg.CHECKPOINT.LAUNCH_MODE = "fresh_run"
    cfg.CHECKPOINT.LOAD_PATH = ""
    cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
    runtime = runtime_factory()
    for _ in range(2):
        runtime.engine.step()
    checkpoint_path = Path(runtime.data_logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME / checkpoint_name
    runtime.engine.publish_runtime_checkpoint(SAVE_REASON_SCHEDULED_TICK, force=True)
    checkpoint_paths = sorted((Path(runtime.data_logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME).glob(f"{cfg.CHECKPOINT.FILENAME_PREFIX}*{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"))
    checkpoint_path = checkpoint_paths[-1]
    runtime.data_logger.close(
        runtime.registry,
        finalize_open_lives=False,
        close_reason="probe_source_session_close",
        close_tick=runtime.engine.tick,
    )
    bundle = load_runtime_checkpoint(checkpoint_path)
    return runtime, checkpoint_path, bundle


def run_resume_telemetry_continuation_probe(runtime_factory: Callable[[], object]) -> dict:
    snapshot = copy.deepcopy(cfg)
    try:
        cfg.TELEMETRY.LOG_TICK_SUMMARY = True
        cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS = 1
        cfg.TELEMETRY.PARQUET_BATCH_ROWS = 1
        source_runtime, checkpoint_path, bundle = _save_source_checkpoint_for_lifecycle_probe(
            runtime_factory,
            "lifecycle_source.pt",
        )
        root_hdf_path = Path(source_runtime.data_logger.run_dir) / "simulation_data.hdf5"
        root_hdf_mtime_before = root_hdf_path.stat().st_mtime_ns
        report = resolve_resume_request(
            requested_mode="resume_exact",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )
        plan = prepare_checkpoint_backed_session_plan(
            report=report,
            bundle=bundle,
            source_checkpoint_path=checkpoint_path,
        )
        cfg.CHECKPOINT.LAUNCH_MODE = "resume_exact"
        cfg.CHECKPOINT.LOAD_PATH = str(checkpoint_path)
        resumed = build_resume_runtime(plan.lineage_root_dir, bundle, session_plan=plan)
        resumed.engine.step()
        resumed.data_logger.close(
            resumed.registry,
            finalize_open_lives=False,
            close_reason="probe_resume_session_close",
            close_tick=resumed.engine.tick,
        )

        session_tick_path = Path(plan.telemetry_dir) / "tick_summary.parquet"
        session_df = pd.read_parquet(session_tick_path)
        with h5py.File(root_hdf_path, "r") as handle:
            hdf5_session_group_exists = f"sessions/{plan.session_label}/agent_snapshots" in handle

        return {
            "match": (
                Path(plan.lineage_root_dir) == Path(source_runtime.data_logger.run_dir)
                and int(plan.session_id) >= 2
                and session_tick_path.exists()
                and hdf5_session_group_exists
                and session_df["tick"].duplicated().sum() == 0
                and root_hdf_path.stat().st_mtime_ns >= root_hdf_mtime_before
            ),
            "lineage_root_reused": str(plan.lineage_root_dir) == str(source_runtime.data_logger.run_dir),
            "session_id": int(plan.session_id),
            "session_tick_path": str(session_tick_path),
            "hdf5_session_group_exists": hdf5_session_group_exists,
            "duplicate_session_tick_rows": int(session_df["tick"].duplicated().sum()),
        }
    finally:
        _restore_cfg(snapshot)


def run_fork_vs_continue_telemetry_policy_probe(runtime_factory: Callable[[], object]) -> dict:
    snapshot = copy.deepcopy(cfg)
    try:
        source_runtime, checkpoint_path, bundle = _save_source_checkpoint_for_lifecycle_probe(
            runtime_factory,
            "lifecycle_fork_source.pt",
        )
        exact_report = resolve_resume_request(
            requested_mode="resume_exact",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )
        exact_plan = prepare_checkpoint_backed_session_plan(
            report=exact_report,
            bundle=bundle,
            source_checkpoint_path=checkpoint_path,
        )
        fork_report = resolve_resume_request(
            requested_mode="fork_from_checkpoint",
            bundle=bundle,
            cfg_obj=cfg,
            source_checkpoint_path=checkpoint_path,
        )
        fork_plan = prepare_checkpoint_backed_session_plan(
            report=fork_report,
            bundle=bundle,
            source_checkpoint_path=checkpoint_path,
        )
        return {
            "match": (
                Path(exact_plan.lineage_root_dir) == Path(source_runtime.data_logger.run_dir)
                and Path(fork_plan.lineage_root_dir) != Path(source_runtime.data_logger.run_dir)
                and fork_plan.parent_lineage_root_dir == str(source_runtime.data_logger.run_dir)
                and exact_plan.is_continuation
                and fork_plan.is_fork
            ),
            "source_root": str(source_runtime.data_logger.run_dir),
            "exact_root": str(exact_plan.lineage_root_dir),
            "fork_root": str(fork_plan.lineage_root_dir),
            "fork_parent_lineage_root": fork_plan.parent_lineage_root_dir,
        }
    finally:
        _restore_cfg(snapshot)


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
            "resume_policy": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "runtime_lifecycle": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "operator_artifacts": _skipped_check("ENABLE_FINAL_AUDIT_HARNESS is disabled"),
            "all_passed": True,
            "skipped": True,
        }

    report = {
        "determinism": _skipped_check("ENABLE_DETERMINISM_TESTS is disabled"),
        "resume": _skipped_check("ENABLE_RESUME_CONSISTENCY_TESTS is disabled"),
        "catastrophe": _skipped_check("ENABLE_CATASTROPHE_REPRO_TESTS is disabled"),
        "save_load_save": _skipped_check("ENABLE_SAVE_LOAD_SAVE_TESTS is disabled"),
        "resume_policy": _skipped_check("ENABLE_RESUME_POLICY_TESTS is disabled"),
        "runtime_lifecycle": _skipped_check("ENABLE_RUNTIME_LIFECYCLE_TESTS is disabled"),
        "operator_artifacts": _skipped_check("ENABLE_OPERATOR_ARTIFACT_TESTS is disabled"),
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
    if cfg.VALIDATION.ENABLE_RESUME_POLICY_TESTS:
        report["resume_policy"] = run_resume_policy_probe(runtime_factory, checkpoint_path)
    if cfg.VALIDATION.ENABLE_RUNTIME_LIFECYCLE_TESTS:
        wallclock = run_wallclock_autosave_probe(runtime_factory)
        shutdown = run_shutdown_checkpoint_probe(runtime_factory)
        close_once = run_logger_close_once_probe(runtime_factory)
        continuation = run_resume_telemetry_continuation_probe(runtime_factory)
        fork_policy = run_fork_vs_continue_telemetry_policy_probe(runtime_factory)
        report["runtime_lifecycle"] = {
            "match": all(
                [
                    wallclock["match"],
                    shutdown["match"],
                    close_once["match"],
                    continuation["match"],
                    fork_policy["match"],
                ]
            ),
            "wallclock_autosave": wallclock,
            "shutdown_checkpoint": shutdown,
            "logger_close_once": close_once,
            "resume_telemetry_continuation": continuation,
            "fork_vs_continue_policy": fork_policy,
        }
    if cfg.VALIDATION.ENABLE_OPERATOR_ARTIFACT_TESTS:
        manual_checkpoint = run_manual_checkpoint_probe(runtime_factory)
        selected_brain_export = run_selected_brain_export_probe(runtime_factory)
        report["operator_artifacts"] = {
            "match": bool(manual_checkpoint["match"] and selected_brain_export["match"]),
            "manual_checkpoint": manual_checkpoint,
            "selected_brain_export": selected_brain_export,
        }

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
    if cfg.VALIDATION.ENABLE_RESUME_POLICY_TESTS:
        enabled_results.append(bool(report["resume_policy"]["match"]))
    if cfg.VALIDATION.ENABLE_RUNTIME_LIFECYCLE_TESTS:
        enabled_results.append(bool(report["runtime_lifecycle"]["match"]))
    if cfg.VALIDATION.ENABLE_OPERATOR_ARTIFACT_TESTS:
        enabled_results.append(bool(report["operator_artifacts"]["match"]))
    report["all_passed"] = all(enabled_results) if enabled_results else True
    return report




