import copy
from pathlib import Path

from tensor_crypt.app.runtime import build_resume_runtime
from tensor_crypt.checkpointing.resume_policy import (
    build_resume_compatibility_report,
    resolve_resume_request,
)
from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.telemetry.run_paths import create_run_directory


def test_resume_policy_accepts_exact_and_classifies_drift_and_fork(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=311, width=10, height=10, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)

    contract = bundle["metadata"]["resume_contract"]
    assert contract["resume_taxonomy_version"] == 1
    assert contract["exact_resume_capability"]["exact_resume_capable"] is True
    assert {"substrate_shape_contract", "observation_contract", "family_contract_set"}.issubset(
        contract["contract_hashes"].keys()
    )

    exact = resolve_resume_request(
        requested_mode="resume_exact",
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "source.pt",
    )
    assert exact["allowed"] is True
    assert exact["resolved_mode"] == "resume_exact"

    cfg.LOG.LOG_TICK_EVERY += 1
    exact_with_drift = resolve_resume_request(
        requested_mode="resume_exact",
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "source.pt",
    )
    assert exact_with_drift["allowed"] is False
    assert exact_with_drift["failure_class"] == "drift_acknowledgment_required"
    assert any(delta["surface"] == "LOG.LOG_TICK_EVERY" for delta in exact_with_drift["drift_surfaces"])

    drift = resolve_resume_request(
        requested_mode="resume_with_drift",
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "source.pt",
    )
    assert drift["allowed"] is True
    assert drift["resolved_mode"] == "resume_with_drift"

    cfg.LOG.LOG_TICK_EVERY -= 1
    cfg.PPO.LR = float(cfg.PPO.LR) * 2.0
    drift_with_fork_delta = resolve_resume_request(
        requested_mode="resume_with_drift",
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "source.pt",
    )
    assert drift_with_fork_delta["allowed"] is False
    assert drift_with_fork_delta["failure_class"] == "fork_semantics_required"
    assert any(delta["surface"] == "PPO.LR" for delta in drift_with_fork_delta["fork_required_surfaces"])

    cfg.CHECKPOINT.FORK_REASON = "policy test fork"
    fork = resolve_resume_request(
        requested_mode="fork_from_checkpoint",
        bundle=bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "source.pt",
    )
    assert fork["allowed"] is True
    assert fork["resolved_mode"] == "fork_from_checkpoint"
    assert fork["ancestry"]["fork_reason"] == "policy test fork"


def test_legacy_contract_inference_blocks_false_exact_when_continuity_missing(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=312, width=10, height=10, agents=4, walls=0, hzones=0)
    legacy_bundle = copy.deepcopy(capture_runtime_checkpoint(runtime))
    legacy_bundle["metadata"].pop("resume_contract", None)
    legacy_bundle["rng_state"] = None

    report = build_resume_compatibility_report(
        requested_mode="resume_exact",
        bundle=legacy_bundle,
        cfg_obj=cfg,
        source_checkpoint_path=tmp_path / "legacy.pt",
    )

    assert report["allowed"] is False
    assert report["legacy_contract_inference_used"] is True
    assert report["failure_class"] == "exact_resume_completeness_deficit"
    assert "resume_contract_metadata_missing" in report["exact_resume_completeness_deficits"]
    assert "rng_state_missing" in report["exact_resume_completeness_deficits"]


def test_resume_runtime_does_not_regenerate_or_bootstrap_root_telemetry(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=313, width=10, height=10, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    source_grid = bundle["grid_state"]["grid"].clone()

    cfg.CHECKPOINT.LAUNCH_MODE = "resume_exact"
    cfg.CHECKPOINT.LOAD_PATH = str(tmp_path / "source.pt")
    cfg.LOG.DIR = str(tmp_path / "resume_logs")
    run_dir = create_run_directory(
        session_metadata={
            "launch_mode_requested": "resume_exact",
            "launch_mode_resolved": "resume_exact",
            "session_kind": "resume_exact",
            "source_checkpoint_path": str(tmp_path / "source.pt"),
            "source_checkpoint_tick": int(bundle["engine_state"]["tick"]),
            "compatibility_report_path": cfg.CHECKPOINT.COMPATIBILITY_REPORT_FILENAME,
        }
    )

    restored = build_resume_runtime(run_dir, bundle)
    try:
        assert restored.registry.active_uid_to_slot == runtime.registry.active_uid_to_slot
        assert restored.grid.grid.cpu().equal(source_grid)
        assert restored.data_logger._initial_population_bootstrapped is False
        assert dict(restored.data_logger.birth_counts_by_tick) == {}
        assert restored.data_logger.open_lives_by_uid == {}
    finally:
        restored.data_logger.close()

    metadata_path = Path(run_dir) / "run_metadata.json"
    assert metadata_path.exists()
