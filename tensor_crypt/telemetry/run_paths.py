"""Run-directory creation utilities."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from ..config_bridge import cfg


FIXED_IDENTITY_RUNTIME = {
    "substrate": "canonical_uid",
    "ppo_ownership_mode": "uid_strict",
}
FIXED_OBSERVATION_COMPAT = {
    "legacy_adapter": "canonical_bridge_v1",
}
FIXED_BRAIN_RUNTIME = {
    "mode": "bloodline_mlp_families_v1",
}


def schema_versions_dict() -> dict:
    return {
        "IDENTITY_SCHEMA_VERSION": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
        "OBS_SCHEMA_VERSION": cfg.SCHEMA.OBS_SCHEMA_VERSION,
        "CHECKPOINT_SCHEMA_VERSION": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "REPRODUCTION_SCHEMA_VERSION": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
        "CATASTROPHE_SCHEMA_VERSION": cfg.SCHEMA.CATASTROPHE_SCHEMA_VERSION,
        "TELEMETRY_SCHEMA_VERSION": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
        "LOGGING_SCHEMA_VERSION": cfg.SCHEMA.LOGGING_SCHEMA_VERSION,
    }


def build_run_metadata(session_metadata: dict | None = None) -> dict:
    manifests_published = bool(
        cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED
        and cfg.CHECKPOINT.MANIFEST_ENABLED
        and cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST
    )
    launch_mode = str(getattr(cfg.CHECKPOINT, "LAUNCH_MODE", "fresh_run"))
    session = {
        "launch_mode_requested": launch_mode,
        "launch_mode_resolved": launch_mode,
        "session_kind": "fresh" if launch_mode == "fresh_run" else launch_mode,
        "source_checkpoint_path": None,
        "source_manifest_path": None,
        "source_checkpoint_tick": None,
        "source_checkpoint_schema_version": None,
        "legacy_contract_inference_used": False,
        "compatibility_report_path": None,
        "fork_reason": "",
        "ancestor_session_kind": None,
        "compatibility_failure_class": None,
    }
    if session_metadata:
        session.update(session_metadata)
    return {
        "schema_versions": schema_versions_dict(),
        "config_snapshot": "config.json",
        "session": session,
        "identity": {
            **FIXED_IDENTITY_RUNTIME,
            "mirror_legacy_float_columns": cfg.IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS,
        },
        "checkpoint": {
            "enabled": cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS,
            "capture_rng_state": cfg.CHECKPOINT.CAPTURE_RNG_STATE,
            "capture_optimizer_state": cfg.CHECKPOINT.CAPTURE_OPTIMIZER_STATE,
            "capture_scaler_state": cfg.CHECKPOINT.CAPTURE_SCALER_STATE,
            "save_every_ticks": cfg.CHECKPOINT.SAVE_EVERY_TICKS,
            "keep_last": cfg.CHECKPOINT.KEEP_LAST,
            "directory_name": cfg.CHECKPOINT.DIRECTORY_NAME,
            "publishes_manifest": manifests_published,
            "writes_latest_pointer": bool(cfg.CHECKPOINT.WRITE_LATEST_POINTER and manifests_published),
        },
        "migration": {
            "log_legacy_slot_fields": cfg.MIGRATION.LOG_LEGACY_SLOT_FIELDS,
            "viewer_show_slot_and_uid": cfg.MIGRATION.VIEWER_SHOW_SLOT_AND_UID,
            "viewer_show_bloodline": cfg.MIGRATION.VIEWER_SHOW_BLOODLINE,
        },
        "observation": {
            **FIXED_OBSERVATION_COMPAT,
            "return_canonical_observations": cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS,
            "canonical_ray_features": cfg.PERCEPT.CANONICAL_RAY_FEATURES,
            "canonical_self_features": cfg.PERCEPT.CANONICAL_SELF_FEATURES,
            "canonical_context_features": cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES,
        },
        "brain_runtime": {
            **FIXED_BRAIN_RUNTIME,
            "default_family": cfg.BRAIN.DEFAULT_FAMILY,
            "family_order": list(cfg.BRAIN.FAMILY_ORDER),
            "legacy_obs_fallback_enabled": cfg.BRAIN.ALLOW_LEGACY_OBS_FALLBACK,
        },
        "catastrophe_runtime": {
            "enabled": cfg.CATASTROPHE.ENABLED,
            "default_mode": cfg.CATASTROPHE.DEFAULT_MODE,
            "allow_overlap": cfg.CATASTROPHE.ALLOW_OVERLAP,
            "max_concurrent": cfg.CATASTROPHE.MAX_CONCURRENT,
            "manual_trigger_enabled": cfg.CATASTROPHE.MANUAL_TRIGGER_ENABLED,
            "manual_clear_enabled": cfg.CATASTROPHE.MANUAL_CLEAR_ENABLED,
        },
        "telemetry": {
            "summary_export_cadence_ticks": cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS,
            "parquet_batch_rows": cfg.TELEMETRY.PARQUET_BATCH_ROWS,
            "lineage_export_format": cfg.TELEMETRY.LINEAGE_EXPORT_FORMAT,
        },
        "viewer": {
            "show_bloodline_legend": cfg.VIEW.SHOW_BLOODLINE_LEGEND,
            "show_catastrophe_panel": cfg.VIEW.SHOW_CATASTROPHE_PANEL,
            "show_catastrophe_overlay": cfg.VIEW.SHOW_CATASTROPHE_OVERLAY,
        },
    }


def create_run_directory(session_metadata: dict | None = None) -> str:
    base_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    root_dir = Path(cfg.LOG.DIR)

    suffix = 0
    while True:
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        run_dir = root_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1

    for subdir in ("snapshots", "brains", "heatmaps"):
        (run_dir / subdir).mkdir(exist_ok=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    run_metadata_path = run_dir / "run_metadata.json"
    with run_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(build_run_metadata(session_metadata=session_metadata), handle, indent=2)

    return str(run_dir)
