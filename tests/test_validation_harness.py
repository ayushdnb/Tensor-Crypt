from tensor_crypt.audit.final_validation import (
    run_catastrophe_repro_probe,
    run_determinism_probe,
    run_final_validation_suite,
    run_resume_chain_probe,
    run_resume_consistency_probe,
    run_stage1_resume_policy_probe,
    save_load_save_surface_signature,
)
from tensor_crypt.config_bridge import cfg


def test_determinism_probe_passes(runtime_builder):
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 3
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 3

    def factory():
        return runtime_builder(seed=31, agents=6, walls=0, hzones=0)

    report = run_determinism_probe(factory, ticks=4)
    assert report["match"] is True


def test_resume_consistency_probe_passes(runtime_builder, tmp_path):
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 3
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 3

    def factory():
        return runtime_builder(seed=32, agents=6, walls=0, hzones=0)

    report = run_resume_consistency_probe(factory, tmp_path / "resume_probe.pt", pre_ticks=3, post_ticks=3)
    assert report["match"] is True


def test_catastrophe_repro_probe_passes(runtime_builder):
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 2
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 2

    def factory():
        return runtime_builder(seed=33, agents=6, walls=0, hzones=0)

    report = run_catastrophe_repro_probe(factory, ticks=4)
    assert report["match"] is True


def test_resume_chain_probe_passes(runtime_builder, tmp_path):
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 3
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 3

    def factory():
        return runtime_builder(seed=320, agents=6, walls=0, hzones=0)

    report = run_resume_chain_probe(factory, tmp_path / "resume_chain", cycles=3, ticks_per_cycle=2)
    assert report["match"] is True
    assert len(report["cycle_reports"]) == 3


def test_save_load_save_surface_signature_passes(runtime_builder, tmp_path):
    def factory():
        return runtime_builder(seed=34, agents=6, walls=0, hzones=0)

    report = save_load_save_surface_signature(factory, tmp_path / "save_load_save.pt")
    assert report["slot_uid_equal"] is True
    assert report["slot_parent_uid_equal"] is True
    assert report["uid_family_equal"] is True
    assert report["registry_data_equal"] is True
    assert report["fitness_equal"] is True
    assert report["grid_equal"] is True


def test_stage1_resume_policy_probe_passes(runtime_builder, tmp_path):
    def factory():
        return runtime_builder(seed=340, agents=6, walls=0, hzones=0)

    report = run_stage1_resume_policy_probe(factory, tmp_path / "policy_probe.pt")
    assert report["match"] is True


def test_full_validation_suite_passes(runtime_builder, tmp_path):
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS = 3
    cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS = 3

    def factory():
        return runtime_builder(seed=35, agents=6, walls=0, hzones=0)

    report = run_final_validation_suite(factory, tmp_path / "suite.pt", ticks=6)
    assert report["all_passed"] is True


def test_validation_suite_respects_config_flags(runtime_builder, tmp_path):
    cfg.VALIDATION.ENABLE_DETERMINISM_TESTS = False
    cfg.VALIDATION.ENABLE_RESUME_CONSISTENCY_TESTS = False
    cfg.VALIDATION.ENABLE_CATASTROPHE_REPRO_TESTS = False
    cfg.VALIDATION.ENABLE_SAVE_LOAD_SAVE_TESTS = True

    def factory():
        return runtime_builder(seed=36, agents=6, walls=0, hzones=0)

    report = run_final_validation_suite(factory, tmp_path / "suite_flags.pt", ticks=6)
    assert report["determinism"]["skipped"] is True
    assert report["resume"]["skipped"] is True
    assert report["catastrophe"]["skipped"] is True
    assert report["save_load_save"]["grid_equal"] is True
    assert report["all_passed"] is True


def test_validation_runtime_signature_includes_overlay_state(runtime_builder):
    from tensor_crypt.audit.final_validation import _runtime_signature

    runtime = runtime_builder(seed=37, agents=6, walls=0, hzones=0)
    before = _runtime_signature(runtime)

    runtime.engine.respawn_controller.toggle_doctrine_override("crowding")
    after = _runtime_signature(runtime)

    assert before["respawn_overlay_runtime_state"] != after["respawn_overlay_runtime_state"]
