"""Run-directory and session-lineage creation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from ..config_bridge import cfg


SESSION_CATALOG_FILENAME = "session_catalog.json"
SESSION_METADATA_FILENAME = "session_metadata.json"
SESSION_DIR_NAME = "sessions"
SESSION_LABEL_RE = re.compile(r"^session_(\d{4,})$")

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


@dataclass(frozen=True)
class SessionPlan:
    """Concrete telemetry/checkpoint lineage plan for one process session."""

    lineage_root_dir: str
    session_id: int
    session_label: str
    session_dir: str
    telemetry_dir: str
    launch_mode_requested: str
    launch_mode_resolved: str
    session_kind: str
    is_continuation: bool
    is_fork: bool
    source_checkpoint_path: str | None = None
    source_manifest_path: str | None = None
    source_checkpoint_tick: int | None = None
    source_checkpoint_schema_version: int | None = None
    continued_from_session_id: int | None = None
    parent_lineage_root_dir: str | None = None
    compatibility_report_path: str | None = None

    @property
    def uses_root_telemetry_layout(self) -> bool:
        return int(self.session_id) == 1 and not bool(self.is_continuation)

    def to_metadata(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "session_id": int(self.session_id),
            "session_label": self.session_label,
            "session_dir": _relpath_or_str(self.session_dir, self.lineage_root_dir),
            "telemetry_dir": _relpath_or_str(self.telemetry_dir, self.lineage_root_dir),
            "lineage_root_dir": str(Path(self.lineage_root_dir)),
            "lineage_root_identifier": Path(self.lineage_root_dir).name,
            "launch_mode_requested": self.launch_mode_requested,
            "launch_mode_resolved": self.launch_mode_resolved,
            "session_kind": self.session_kind,
            "is_continuation": bool(self.is_continuation),
            "is_fork": bool(self.is_fork),
            "source_checkpoint_path": self.source_checkpoint_path,
            "source_manifest_path": self.source_manifest_path,
            "source_checkpoint_tick": self.source_checkpoint_tick,
            "source_checkpoint_schema_version": self.source_checkpoint_schema_version,
            "continued_from_session_id": self.continued_from_session_id,
            "parent_lineage_root_dir": self.parent_lineage_root_dir,
            "compatibility_report_path": self.compatibility_report_path,
        }
        if extra:
            payload.update(extra)
        return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(temp_path, path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relpath_or_str(path: str | Path, root: str | Path) -> str:
    try:
        return os.path.relpath(Path(path), start=Path(root))
    except ValueError:
        return str(path)


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


def _default_session_metadata() -> dict:
    launch_mode = str(getattr(cfg.CHECKPOINT, "LAUNCH_MODE", "fresh_run"))
    return {
        "launch_mode_requested": launch_mode,
        "launch_mode_resolved": launch_mode,
        "session_kind": "fresh" if launch_mode == "fresh_run" else launch_mode,
        "session_id": 1,
        "session_label": session_label_for(1),
        "lineage_root_identifier": None,
        "source_checkpoint_path": None,
        "source_manifest_path": None,
        "source_checkpoint_tick": None,
        "source_checkpoint_schema_version": None,
        "legacy_contract_inference_used": False,
        "compatibility_report_path": None,
        "fork_reason": "",
        "ancestor_session_kind": None,
        "compatibility_failure_class": None,
        "is_continuation": False,
        "is_fork": launch_mode == "fork_from_checkpoint",
        "continued_from_session_id": None,
        "parent_lineage_root_dir": None,
    }


def build_run_metadata(session_metadata: dict | None = None) -> dict:
    manifests_published = bool(
        cfg.CHECKPOINT.ATOMIC_WRITE_ENABLED
        and cfg.CHECKPOINT.MANIFEST_ENABLED
        and cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST
    )
    session = _default_session_metadata()
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
            "wallclock_autosave_enabled": bool(cfg.CHECKPOINT.ENABLE_WALLCLOCK_AUTOSAVE),
            "wallclock_autosave_interval_seconds": float(cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_INTERVAL_SECONDS),
            "shutdown_checkpoint_enabled": bool(cfg.CHECKPOINT.ENABLE_SHUTDOWN_CHECKPOINT),
            "save_reason_logging_enabled": bool(cfg.CHECKPOINT.SAVE_REASON_LOGGING_ENABLED),
            "resume_continuation_policy": str(cfg.CHECKPOINT.RESUME_CONTINUATION_POLICY),
            "fork_telemetry_policy": str(cfg.CHECKPOINT.FORK_TELEMETRY_POLICY),
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
            "session_segmentation_enabled": bool(cfg.TELEMETRY.SESSION_SEGMENTATION_ENABLED),
            "finalize_open_lives_on_session_close": bool(cfg.TELEMETRY.FINALIZE_OPEN_LIVES_ON_SESSION_CLOSE),
            "write_session_catalog": bool(cfg.TELEMETRY.WRITE_SESSION_CATALOG),
            "write_session_metadata": bool(cfg.TELEMETRY.WRITE_SESSION_METADATA),
            "selected_brain_export_directory_name": str(cfg.TELEMETRY.SELECTED_BRAIN_EXPORT_DIRECTORY_NAME),
        },
        "viewer": {
            "show_bloodline_legend": cfg.VIEW.SHOW_BLOODLINE_LEGEND,
            "show_operator_action_buttons": bool(cfg.VIEW.SHOW_OPERATOR_ACTION_BUTTONS),
            "show_operator_action_status": bool(cfg.VIEW.SHOW_OPERATOR_ACTION_STATUS),
            "show_catastrophe_panel": cfg.VIEW.SHOW_CATASTROPHE_PANEL,
            "show_catastrophe_overlay": cfg.VIEW.SHOW_CATASTROPHE_OVERLAY,
        },
    }


def session_label_for(session_id: int) -> str:
    return f"session_{int(session_id):04d}"


def session_dir_for(run_dir: str | Path, session_id: int) -> Path:
    return Path(run_dir) / SESSION_DIR_NAME / session_label_for(session_id)


def session_metadata_path_for(plan: SessionPlan) -> Path:
    return Path(plan.session_dir) / SESSION_METADATA_FILENAME


def session_catalog_path_for(run_dir: str | Path) -> Path:
    return Path(run_dir) / SESSION_CATALOG_FILENAME


def _empty_session_catalog(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    return {
        "catalog_version": 1,
        "lineage_root_dir": str(run_dir),
        "lineage_root_identifier": run_dir.name,
        "sessions": [],
    }


def load_session_catalog(run_dir: str | Path) -> dict:
    catalog_path = session_catalog_path_for(run_dir)
    if not catalog_path.exists():
        return _empty_session_catalog(run_dir)
    catalog = _read_json(catalog_path)
    catalog.setdefault("catalog_version", 1)
    catalog.setdefault("lineage_root_dir", str(Path(run_dir)))
    catalog.setdefault("lineage_root_identifier", Path(run_dir).name)
    catalog.setdefault("sessions", [])
    return catalog


def _infer_session_one_record(run_dir: str | Path) -> dict | None:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = _read_json(metadata_path)
    session = dict(metadata.get("session", {}))
    session.update(
        {
            "session_id": int(session.get("session_id") or 1),
            "session_label": session.get("session_label") or session_label_for(1),
            "session_dir": _relpath_or_str(session_dir_for(run_dir, 1), run_dir),
            "telemetry_dir": ".",
            "lineage_root_dir": str(run_dir),
            "lineage_root_identifier": run_dir.name,
            "is_continuation": bool(session.get("is_continuation", False)),
            "is_fork": bool(session.get("is_fork", session.get("launch_mode_resolved") == "fork_from_checkpoint")),
        }
    )
    return session


def allocate_next_session_id(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    ids: set[int] = set()
    catalog = load_session_catalog(run_dir)
    for record in catalog.get("sessions", []):
        raw_session_id = record.get("session_id")
        if raw_session_id is not None:
            ids.add(int(raw_session_id))
    sessions_root = run_dir / SESSION_DIR_NAME
    if sessions_root.exists():
        for child in sessions_root.iterdir():
            if child.is_dir():
                match = SESSION_LABEL_RE.match(child.name)
                if match:
                    ids.add(int(match.group(1)))
    if not ids and (run_dir / "run_metadata.json").exists():
        ids.add(1)
    return max(ids, default=0) + 1


def write_or_update_session_catalog(run_dir: str | Path, session_record: dict) -> dict:
    run_dir = Path(run_dir)
    catalog = load_session_catalog(run_dir)
    session_id = int(session_record["session_id"])
    sessions = [record for record in catalog.get("sessions", []) if int(record.get("session_id", -1)) != session_id]
    sessions.append(dict(session_record))
    sessions.sort(key=lambda record: int(record.get("session_id", 0)))
    catalog["sessions"] = sessions
    catalog["lineage_root_dir"] = str(run_dir)
    catalog["lineage_root_identifier"] = run_dir.name
    if cfg.TELEMETRY.WRITE_SESSION_CATALOG:
        _write_json(session_catalog_path_for(run_dir), catalog)
    return catalog


def write_session_metadata(plan: SessionPlan, metadata: dict | None = None) -> Path:
    payload = dict(metadata or plan.to_metadata())
    path = session_metadata_path_for(plan)
    if cfg.TELEMETRY.WRITE_SESSION_METADATA:
        _write_json(path, payload)
    return path


def update_session_metadata(plan: SessionPlan, **updates: Any) -> dict:
    metadata_path = session_metadata_path_for(plan)
    if metadata_path.exists():
        payload = _read_json(metadata_path)
    else:
        payload = plan.to_metadata()
    payload.update(updates)
    write_session_metadata(plan, payload)
    write_or_update_session_catalog(plan.lineage_root_dir, payload)
    return payload


def _session_plan_from_metadata(run_dir: Path, session: dict) -> SessionPlan:
    session_id = int(session.get("session_id") or 1)
    session_label = session.get("session_label") or session_label_for(session_id)
    session_dir = session_dir_for(run_dir, session_id)
    raw_telemetry_dir = session.get("telemetry_dir")
    if raw_telemetry_dir in (None, ".", ""):
        telemetry_dir = run_dir
    else:
        telemetry_dir = run_dir / raw_telemetry_dir
    return SessionPlan(
        lineage_root_dir=str(run_dir),
        session_id=session_id,
        session_label=session_label,
        session_dir=str(session_dir),
        telemetry_dir=str(telemetry_dir),
        launch_mode_requested=str(session.get("launch_mode_requested", "fresh_run")),
        launch_mode_resolved=str(session.get("launch_mode_resolved", session.get("launch_mode_requested", "fresh_run"))),
        session_kind=str(session.get("session_kind", "fresh")),
        is_continuation=bool(session.get("is_continuation", False)),
        is_fork=bool(session.get("is_fork", False)),
        source_checkpoint_path=session.get("source_checkpoint_path"),
        source_manifest_path=session.get("source_manifest_path"),
        source_checkpoint_tick=session.get("source_checkpoint_tick"),
        source_checkpoint_schema_version=session.get("source_checkpoint_schema_version"),
        continued_from_session_id=session.get("continued_from_session_id"),
        parent_lineage_root_dir=session.get("parent_lineage_root_dir"),
        compatibility_report_path=session.get("compatibility_report_path"),
    )


def session_plan_from_run_directory(run_dir: str | Path) -> SessionPlan:
    run_dir = Path(run_dir)
    catalog = load_session_catalog(run_dir)
    records = catalog.get("sessions", [])
    if records:
        latest = max(records, key=lambda record: int(record.get("session_id", 0)))
        return _session_plan_from_metadata(run_dir, latest)
    inferred = _infer_session_one_record(run_dir) or _default_session_metadata()
    return _session_plan_from_metadata(run_dir, inferred)


def create_fresh_run_directory(session_metadata: dict | None = None) -> tuple[str, SessionPlan]:
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

    session = _default_session_metadata()
    if session_metadata:
        session.update(session_metadata)
    session["session_id"] = int(session.get("session_id") or 1)
    session["session_label"] = session.get("session_label") or session_label_for(session["session_id"])
    session["lineage_root_identifier"] = run_dir.name
    session["lineage_root_dir"] = str(run_dir)
    session["telemetry_dir"] = "."
    session["session_dir"] = _relpath_or_str(session_dir_for(run_dir, session["session_id"]), run_dir)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    run_metadata_path = run_dir / "run_metadata.json"
    with run_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(build_run_metadata(session_metadata=session), handle, indent=2)

    plan = _session_plan_from_metadata(run_dir, session)
    Path(plan.session_dir).mkdir(parents=True, exist_ok=True)
    write_session_metadata(plan, session)
    write_or_update_session_catalog(run_dir, session)
    return str(run_dir), plan


def create_run_directory(session_metadata: dict | None = None) -> str:
    run_dir, _plan = create_fresh_run_directory(session_metadata=session_metadata)
    return run_dir


def _source_metadata_from_report(report: dict) -> dict:
    source_identity = report.get("source_checkpoint_identity", {})
    resolved_mode = report.get("resolved_mode") or report.get("requested_mode")
    return {
        "launch_mode_requested": report.get("requested_mode"),
        "launch_mode_resolved": resolved_mode,
        "session_kind": resolved_mode,
        "source_checkpoint_path": source_identity.get("checkpoint_path"),
        "source_manifest_path": source_identity.get("manifest_path"),
        "source_checkpoint_tick": source_identity.get("tick"),
        "source_checkpoint_schema_version": source_identity.get("checkpoint_schema_version"),
        "legacy_contract_inference_used": bool(report.get("legacy_contract_inference_used", False)),
        "compatibility_report_path": None,
        "fork_reason": report.get("ancestry", {}).get("fork_reason", ""),
        "ancestor_session_kind": "checkpointed_session",
        "compatibility_failure_class": report.get("failure_class"),
    }


def infer_lineage_root_from_checkpoint(bundle: dict, source_checkpoint_path: str | Path) -> Path:
    lifecycle = bundle.get("metadata", {}).get("runtime_lifecycle", {})
    candidate = lifecycle.get("lineage_root_run_dir")
    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    checkpoint_path = Path(source_checkpoint_path).resolve()
    if checkpoint_path.parent.name == cfg.CHECKPOINT.DIRECTORY_NAME:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def prepare_checkpoint_backed_session_plan(
    *,
    report: dict,
    bundle: dict,
    source_checkpoint_path: str | Path,
    source_manifest_path: str | Path | None = None,
) -> SessionPlan:
    resolved_mode = str(report.get("resolved_mode") or report.get("requested_mode"))
    source_checkpoint_path = Path(source_checkpoint_path)
    source_manifest_text = None if source_manifest_path is None else str(source_manifest_path)
    source_root = infer_lineage_root_from_checkpoint(bundle, source_checkpoint_path)
    source_lifecycle = bundle.get("metadata", {}).get("runtime_lifecycle", {})
    source_session_id = source_lifecycle.get("session_id") or 1

    if (
        resolved_mode in {"resume_exact", "resume_with_drift"}
        and cfg.CHECKPOINT.CONTINUE_TELEMETRY_ON_RESUME
        and str(cfg.CHECKPOINT.RESUME_CONTINUATION_POLICY) == "continue_lineage_root"
    ):
        session_id = allocate_next_session_id(source_root)
        session_label = session_label_for(session_id)
        session_dir = session_dir_for(source_root, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)
        metadata = _source_metadata_from_report(report)
        metadata.update(
            {
                "session_id": session_id,
                "session_label": session_label,
                "session_dir": _relpath_or_str(session_dir, source_root),
                "telemetry_dir": _relpath_or_str(session_dir, source_root),
                "lineage_root_dir": str(source_root),
                "lineage_root_identifier": source_root.name,
                "source_checkpoint_path": str(source_checkpoint_path),
                "source_manifest_path": source_manifest_text,
                "source_checkpoint_tick": int(bundle.get("engine_state", {}).get("tick", -1)),
                "source_checkpoint_schema_version": int(bundle.get("checkpoint_schema_version", -1)),
                "continued_from_session_id": int(source_session_id),
                "is_continuation": True,
                "is_fork": False,
            }
        )
        plan = _session_plan_from_metadata(source_root, metadata)
        write_session_metadata(plan, metadata)
        write_or_update_session_catalog(source_root, metadata)
        return plan

    fork_metadata = _source_metadata_from_report(report)
    fork_metadata.update(
        {
            "is_continuation": False,
            "is_fork": resolved_mode == "fork_from_checkpoint",
            "parent_lineage_root_dir": str(source_root),
            "source_checkpoint_path": str(source_checkpoint_path),
            "source_manifest_path": source_manifest_text,
            "source_checkpoint_tick": int(bundle.get("engine_state", {}).get("tick", -1)),
            "source_checkpoint_schema_version": int(bundle.get("checkpoint_schema_version", -1)),
        }
    )
    run_dir, plan = create_fresh_run_directory(session_metadata=fork_metadata)
    return session_plan_from_run_directory(run_dir)


__all__ = [
    "SESSION_CATALOG_FILENAME",
    "SESSION_METADATA_FILENAME",
    "SessionPlan",
    "allocate_next_session_id",
    "build_run_metadata",
    "create_fresh_run_directory",
    "create_run_directory",
    "infer_lineage_root_from_checkpoint",
    "load_session_catalog",
    "prepare_checkpoint_backed_session_plan",
    "schema_versions_dict",
    "session_catalog_path_for",
    "session_dir_for",
    "session_label_for",
    "session_metadata_path_for",
    "session_plan_from_run_directory",
    "update_session_metadata",
    "write_or_update_session_catalog",
    "write_session_metadata",
]
