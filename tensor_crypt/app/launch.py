"""Repository launch entrypoint for Tensor Crypt.

This module keeps user-facing startup logic separate from the simulation
subsystems. It owns only launch-time concerns:
- determinism setup
- run-directory creation
- startup diagnostics
- viewer launch

It intentionally does not own simulation rules.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..checkpointing.atomic_checkpoint import manifest_path_for, resolve_latest_checkpoint_bundle
from ..checkpointing.resume_policy import (
    CHECKPOINT_BACKED_MODES,
    DEFAULT_COMPATIBILITY_REPORT_FILENAME,
    FRESH_RUN,
    normalize_launch_mode,
    resolve_resume_request,
    session_metadata_from_report,
    write_resume_compatibility_report,
)
from ..checkpointing.runtime_checkpoint import load_runtime_checkpoint
from ..config_bridge import cfg
from ..runtime_config import apply_experimental_single_family_launch_defaults
from ..telemetry.run_paths import (
    create_run_directory,
    prepare_checkpoint_backed_session_plan,
    update_session_metadata,
)
from .runtime import build_resume_runtime, build_runtime, setup_determinism

__all__ = ["main"]


def main() -> None:
    apply_experimental_single_family_launch_defaults()
    setup_determinism()
    requested_mode = normalize_launch_mode(cfg.CHECKPOINT.LAUNCH_MODE)

    print(f"Device: {cfg.SIM.DEVICE}")
    print(
        "Startup mode: experimental self-centric single-family preset "
        f"({cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY})"
    )

    if requested_mode == FRESH_RUN:
        run_dir = create_run_directory()
        print(f"Launch mode: {requested_mode}")
        print(f"Grid: {cfg.GRID.W}x{cfg.GRID.H}")
        print(f"Initial agents: {cfg.AGENTS.N}")
        print(f"Run directory: {run_dir}")
        runtime = build_runtime(run_dir)
        runtime.viewer.run()
        return

    if requested_mode not in CHECKPOINT_BACKED_MODES:
        raise ValueError(f"Unsupported checkpoint launch mode: {requested_mode}")

    load_path = Path(cfg.CHECKPOINT.LOAD_PATH)
    source_checkpoint_path = (
        resolve_latest_checkpoint_bundle(load_path)
        if load_path.is_dir() or load_path.name == cfg.CHECKPOINT.LATEST_POINTER_FILENAME
        else load_path
    )
    source_manifest_path = manifest_path_for(source_checkpoint_path)
    source_manifest_for_report = source_manifest_path if source_manifest_path.exists() else None
    bundle = load_runtime_checkpoint(source_checkpoint_path)
    report = resolve_resume_request(
        requested_mode=requested_mode,
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=source_checkpoint_path,
        source_manifest_path=source_manifest_for_report,
    )
    if not report["allowed"]:
        run_dir = create_run_directory(session_metadata=session_metadata_from_report(report))
        report_path = Path(run_dir) / getattr(
            cfg.CHECKPOINT,
            "COMPATIBILITY_REPORT_FILENAME",
            DEFAULT_COMPATIBILITY_REPORT_FILENAME,
        )
        if cfg.CHECKPOINT.WRITE_COMPATIBILITY_REPORT:
            write_resume_compatibility_report(report_path, report)
        raise RuntimeError(
            "Checkpoint launch rejected by resume policy: "
            f"{report.get('failure_class')} (report: {report_path})"
        )

    session_plan = prepare_checkpoint_backed_session_plan(
        report=report,
        bundle=bundle,
        source_checkpoint_path=source_checkpoint_path,
        source_manifest_path=source_manifest_for_report,
    )
    run_dir = session_plan.lineage_root_dir
    report_path = Path(session_plan.session_dir) / getattr(
        cfg.CHECKPOINT,
        "COMPATIBILITY_REPORT_FILENAME",
        DEFAULT_COMPATIBILITY_REPORT_FILENAME,
    )
    if cfg.CHECKPOINT.WRITE_COMPATIBILITY_REPORT:
        write_resume_compatibility_report(report_path, report)
        update_session_metadata(
            session_plan,
            compatibility_report_path=os.path.relpath(report_path, start=Path(run_dir)),
        )

    print(f"Launch mode requested: {requested_mode}")
    print(f"Launch mode resolved: {report.get('resolved_mode')}")
    print(f"Source checkpoint: {source_checkpoint_path}")
    print(f"Source checkpoint tick: {bundle.get('engine_state', {}).get('tick')}")
    print(f"Legacy contract inference: {report.get('legacy_contract_inference_used')}")
    print(f"Run directory: {run_dir}")

    runtime = build_resume_runtime(run_dir, bundle, session_plan=session_plan)
    runtime.viewer.run()
