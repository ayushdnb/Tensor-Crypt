import importlib

import config as root_config_module
import main as root_main_module
import run as root_run_module

from tensor_crypt.app.launch import main as launch_main
from tensor_crypt.config_bridge import cfg as bridged_cfg
from tensor_crypt.world.physics import Physics
from tensor_crypt.world.spatial_grid import Grid
from tensor_crypt.agents.brain import Brain
from tensor_crypt.simulation.engine import Engine
from tensor_crypt.viewer.main import Viewer


def test_public_and_compatibility_imports_resolve():
    names = [
        "config",
        "main",
        "run",
        "engine.brain",
        "engine.grid",
        "engine.physics",
        "engine.simulation",
        "viewer.main",
        "viewer.camera",
        "viewer.layout",
    ]
    modules = {name: importlib.import_module(name) for name in names}

    assert modules["main"].main is launch_main
    assert modules["run"].main is launch_main
    assert modules["engine.brain"].Brain is Brain
    assert modules["engine.grid"].Grid is Grid
    assert modules["engine.physics"].Physics is Physics
    assert modules["engine.simulation"].Engine is Engine
    assert modules["viewer.main"].Viewer is Viewer


def test_config_bridge_shares_root_cfg_instance():
    root_config = importlib.import_module("config")

    assert bridged_cfg is root_config.cfg


def test_launch_entrypoints_apply_single_brain_vmap_preset_and_run_viewer(monkeypatch):
    captured = {}

    class _DummyViewer:
        def run(self):
            captured["viewer_ran"] = True

    class _DummyRuntime:
        viewer = _DummyViewer()

    def fake_setup_determinism():
        captured["setup_called"] = True

    def fake_create_run_directory():
        captured["run_dir_created"] = "fake_run_dir"
        return "fake_run_dir"

    def fake_build_runtime(run_dir):
        captured["run_dir_used"] = run_dir
        captured["obs_mode"] = bridged_cfg.PERCEPT.OBS_MODE
        captured["return_experimental_observations"] = bridged_cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS
        captured["experimental_branch_preset"] = bridged_cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET
        captured["family_shift_mutation"] = bridged_cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION
        captured["family_vmap_inference"] = bridged_cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE
        return _DummyRuntime()

    monkeypatch.setattr("tensor_crypt.app.launch.setup_determinism", fake_setup_determinism)
    monkeypatch.setattr("tensor_crypt.app.launch.create_run_directory", fake_create_run_directory)
    monkeypatch.setattr("tensor_crypt.app.launch.build_runtime", fake_build_runtime)

    root_run_module.main()

    assert captured["setup_called"] is True
    assert captured["run_dir_created"] == "fake_run_dir"
    assert captured["run_dir_used"] == "fake_run_dir"
    assert captured["viewer_ran"] is True
    assert captured["obs_mode"] == "experimental_selfcentric_v1"
    assert captured["return_experimental_observations"] is True
    assert captured["experimental_branch_preset"] is True
    assert captured["family_shift_mutation"] is False
    assert captured["family_vmap_inference"] is True
    assert root_main_module.main is launch_main
    assert root_config_module.cfg is bridged_cfg
