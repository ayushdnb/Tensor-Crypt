import importlib

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
