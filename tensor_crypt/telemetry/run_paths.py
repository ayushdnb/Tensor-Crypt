"""
Run-directory creation utilities.

Path handling lives here so launch code does not scatter filesystem policy
throughout the project. All paths are resolved through `pathlib` to avoid
machine-specific separator assumptions.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from ..config_bridge import cfg


def schema_versions_dict() -> dict:
    return {
        "IDENTITY_SCHEMA_VERSION": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
        "OBS_SCHEMA_VERSION": cfg.SCHEMA.OBS_SCHEMA_VERSION,
        "CHECKPOINT_SCHEMA_VERSION": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
        "REPRODUCTION_SCHEMA_VERSION": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
        "TELEMETRY_SCHEMA_VERSION": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
        "LOGGING_SCHEMA_VERSION": cfg.SCHEMA.LOGGING_SCHEMA_VERSION,
    }


def build_run_metadata() -> dict:
    return {
        "schema_versions": schema_versions_dict(),
        "ownership_mode": cfg.IDENTITY.OWNERSHIP_MODE,
        "config_snapshot": "config.json",
        "identity": {
            "enable_uid_substrate": cfg.IDENTITY.ENABLE_UID_SUBSTRATE,
            "mirror_legacy_float_columns": cfg.IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS,
        },
        "checkpoint": {
            "enabled": cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS,
            "capture_rng_state": cfg.CHECKPOINT.CAPTURE_RNG_STATE,
            "capture_optimizer_state": cfg.CHECKPOINT.CAPTURE_OPTIMIZER_STATE,
            "capture_scaler_state": cfg.CHECKPOINT.CAPTURE_SCALER_STATE,
        },
        "migration": {
            "log_legacy_slot_fields": cfg.MIGRATION.LOG_LEGACY_SLOT_FIELDS,
            "log_uid_fields": cfg.MIGRATION.LOG_UID_FIELDS,
            "viewer_show_slot_and_uid": cfg.MIGRATION.VIEWER_SHOW_SLOT_AND_UID,
            "require_canonical_uid_paths": cfg.MIGRATION.REQUIRE_CANONICAL_UID_PATHS,
        },
        "observation": {
            "legacy_adapter_mode": cfg.PERCEPT.LEGACY_ADAPTER_MODE,
            "return_canonical_observations": cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS,
            "canonical_ray_features": cfg.PERCEPT.CANONICAL_RAY_FEATURES,
            "canonical_self_features": cfg.PERCEPT.CANONICAL_SELF_FEATURES,
            "canonical_context_features": cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES,
        },
    }


def create_run_directory() -> str:
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
        json.dump(build_run_metadata(), handle, indent=2)

    return str(run_dir)
