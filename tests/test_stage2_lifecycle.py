import json
from pathlib import Path
import time

import h5py
import pandas as pd

from tensor_crypt.app.lifecycle import finalize_runtime
from tensor_crypt.app.runtime import build_resume_runtime
from tensor_crypt.checkpointing.atomic_checkpoint import manifest_path_for, resolve_latest_checkpoint_bundle
from tensor_crypt.checkpointing.resume_policy import resolve_resume_request
from tensor_crypt.checkpointing.runtime_checkpoint import load_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.simulation.engine import (
    SAVE_REASON_SCHEDULED_TICK,
    SAVE_REASON_SCHEDULED_WALLCLOCK,
    SAVE_REASON_SHUTDOWN,
)
from tensor_crypt.telemetry.run_paths import (
    load_session_catalog,
    prepare_checkpoint_backed_session_plan,
)


def _manifest_for(bundle_path: Path) -> dict:
    with manifest_path_for(bundle_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_source_checkpoint(runtime_builder, *, seed: int = 901):
    cfg.CHECKPOINT.LAUNCH_MODE = "fresh_run"
    cfg.CHECKPOINT.LOAD_PATH = ""
    cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
    runtime = runtime_builder(seed=seed, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(2):
        runtime.engine.step()
    checkpoint_path = runtime.engine.publish_runtime_checkpoint(SAVE_REASON_SCHEDULED_TICK, force=True)
    runtime.data_logger.close(
        runtime.registry,
        finalize_open_lives=False,
        close_reason="test_source_close",
        close_tick=runtime.engine.tick,
    )
    bundle = load_runtime_checkpoint(checkpoint_path)
    return runtime, Path(checkpoint_path), bundle


def _resume_plan(bundle: dict, checkpoint_path: Path, *, requested_mode: str):
    report = resolve_resume_request(
        requested_mode=requested_mode,
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=checkpoint_path,
        source_manifest_path=manifest_path_for(checkpoint_path),
    )
    return prepare_checkpoint_backed_session_plan(
        report=report,
        bundle=bundle,
        source_checkpoint_path=checkpoint_path,
        source_manifest_path=manifest_path_for(checkpoint_path),
    )


def test_tick_wallclock_and_shutdown_checkpoints_record_save_reasons(runtime_builder):
    cfg.CHECKPOINT.SAVE_EVERY_TICKS = 2
    cfg.CHECKPOINT.KEEP_LAST = 5
    runtime = runtime_builder(seed=902, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(2):
        runtime.engine.step()

    tick_checkpoint = Path(runtime.engine.last_runtime_checkpoint_path)
    tick_manifest = _manifest_for(tick_checkpoint)
    assert tick_manifest["save_reason"] == SAVE_REASON_SCHEDULED_TICK
    assert tick_manifest["session_id"] == 1

    cfg.CHECKPOINT.SAVE_EVERY_TICKS = 0
    cfg.CHECKPOINT.ENABLE_WALLCLOCK_AUTOSAVE = True
    cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_INTERVAL_SECONDS = 0.001
    cfg.CHECKPOINT.WALLCLOCK_AUTOSAVE_MIN_TICKS_ADVANCED = 1
    runtime.engine.last_runtime_checkpoint_wallclock_time = time.monotonic() - 1.0
    runtime.engine.step()
    wallclock_checkpoint = Path(runtime.engine.last_runtime_checkpoint_path)
    wallclock_manifest = _manifest_for(wallclock_checkpoint)
    assert wallclock_manifest["save_reason"] == SAVE_REASON_SCHEDULED_WALLCLOCK

    runtime.engine.last_runtime_checkpoint_wallclock_time = time.monotonic() - 1.0
    before = sorted((Path(runtime.data_logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME).glob("*.pt"))
    assert runtime.engine.maybe_save_runtime_checkpoint_wallclock(paused=True) is None
    after = sorted((Path(runtime.data_logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME).glob("*.pt"))
    assert after == before

    result = finalize_runtime(runtime, close_reason="test_shutdown")
    second = finalize_runtime(runtime, close_reason="test_shutdown_again")
    assert result.checkpoint_path is not None
    assert second.already_finalized is True
    assert runtime.data_logger._closed is True
    shutdown_manifest = _manifest_for(Path(result.checkpoint_path))
    assert shutdown_manifest["save_reason"] == SAVE_REASON_SHUTDOWN


def test_resume_exact_reuses_lineage_root_and_writes_session_segment(runtime_builder):
    source, checkpoint_path, bundle = _build_source_checkpoint(runtime_builder, seed=903)
    source_root = Path(source.data_logger.run_dir)
    root_hdf_path = source_root / "simulation_data.hdf5"

    cfg.CHECKPOINT.LAUNCH_MODE = "resume_exact"
    cfg.CHECKPOINT.LOAD_PATH = str(checkpoint_path)
    plan = _resume_plan(bundle, checkpoint_path, requested_mode="resume_exact")
    resumed = build_resume_runtime(plan.lineage_root_dir, bundle, session_plan=plan)
    resumed.engine.step()
    resumed.data_logger.close(
        resumed.registry,
        finalize_open_lives=False,
        close_reason="test_resume_close",
        close_tick=resumed.engine.tick,
    )

    session_tick_path = Path(plan.telemetry_dir) / "tick_summary.parquet"
    session_df = pd.read_parquet(session_tick_path)
    catalog = load_session_catalog(source_root)

    assert Path(plan.lineage_root_dir) == source_root
    assert int(plan.session_id) == 2
    assert session_tick_path.exists()
    assert session_df["tick"].duplicated().sum() == 0
    assert (source_root / "birth_ledger.parquet").exists()
    assert any(int(record["session_id"]) == 2 for record in catalog["sessions"])
    with h5py.File(root_hdf_path, "r") as handle:
        assert f"sessions/{plan.session_label}/agent_snapshots" in handle


def test_fork_creates_new_lineage_root_and_continue_reuses_source(runtime_builder):
    source, checkpoint_path, bundle = _build_source_checkpoint(runtime_builder, seed=904)
    source_root = Path(source.data_logger.run_dir)

    exact_plan = _resume_plan(bundle, checkpoint_path, requested_mode="resume_exact")
    fork_plan = _resume_plan(bundle, checkpoint_path, requested_mode="fork_from_checkpoint")

    assert Path(exact_plan.lineage_root_dir) == source_root
    assert exact_plan.is_continuation is True
    assert Path(fork_plan.lineage_root_dir) != source_root
    assert fork_plan.is_fork is True
    assert fork_plan.parent_lineage_root_dir == str(source_root)


def test_latest_pointer_advances_within_continued_lineage_root(runtime_builder):
    source, checkpoint_path, bundle = _build_source_checkpoint(runtime_builder, seed=905)
    source_root = Path(source.data_logger.run_dir)

    cfg.CHECKPOINT.LAUNCH_MODE = "resume_exact"
    cfg.CHECKPOINT.LOAD_PATH = str(checkpoint_path)
    plan = _resume_plan(bundle, checkpoint_path, requested_mode="resume_exact")
    resumed = build_resume_runtime(plan.lineage_root_dir, bundle, session_plan=plan)
    resumed.engine.step()
    resumed_checkpoint = resumed.engine.publish_runtime_checkpoint(SAVE_REASON_SCHEDULED_TICK, force=True)
    resumed.data_logger.close(
        resumed.registry,
        finalize_open_lives=False,
        close_reason="test_latest_pointer_close",
        close_tick=resumed.engine.tick,
    )

    resolved = resolve_latest_checkpoint_bundle(source_root / cfg.CHECKPOINT.DIRECTORY_NAME)
    manifest = _manifest_for(Path(resumed_checkpoint))

    assert resolved == Path(resumed_checkpoint).resolve()
    assert manifest["session_id"] == int(plan.session_id)
    assert manifest["continued_from_session_id"] == 1
