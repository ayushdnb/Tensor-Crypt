from pathlib import Path

import h5py
import pandas as pd
import torch

from tensor_crypt.config_bridge import cfg
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.telemetry.data_logger import DataLogger
from tensor_crypt.telemetry import run_paths
from tensor_crypt.world.spatial_grid import Grid


def _create_logger_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir / "run"
    (run_dir / "brains").mkdir(parents=True, exist_ok=True)
    (run_dir / "heatmaps").mkdir(exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    return run_dir


def test_create_run_directory_deduplicates_same_timestamp(monkeypatch, tmp_path):
    cfg.LOG.DIR = str(tmp_path / "logs")
    monkeypatch.setattr(run_paths.time, "strftime", lambda _: "20260404_233100")

    first = Path(run_paths.create_run_directory())
    second = Path(run_paths.create_run_directory())

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.name == "run_20260404_233100"
    assert second.name == "run_20260404_233100_01"
    assert (first / "config.json").exists()
    assert (second / "config.json").exists()
    assert (first / "run_metadata.json").exists()
    metadata = run_paths.build_run_metadata()
    assert metadata["schema_versions"]["IDENTITY_SCHEMA_VERSION"] == cfg.SCHEMA.IDENTITY_SCHEMA_VERSION


def test_collision_parquet_schema_handles_empty_then_nonempty_contenders(tmp_path):
    run_dir = _create_logger_run_dir(tmp_path)
    logger = DataLogger(str(run_dir))

    logger.log_physics_events(
        0,
        [
            {
                "kind": "wall",
                "a": 1,
                "b": -1,
                "damage": 1.2,
                "damage_a": float("nan"),
                "damage_b": float("nan"),
                "contenders": [],
                "winner": -1,
            }
        ],
    )
    logger.log_physics_events(
        1,
        [
            {
                "kind": "contest",
                "a": -1,
                "b": -1,
                "damage": float("nan"),
                "damage_a": float("nan"),
                "damage_b": float("nan"),
                "contenders": [1, 2],
                "winner": 1,
            }
        ],
    )
    logger.close()

    rows = pd.read_parquet(run_dir / "collisions.parquet").to_dict("records")
    assert len(rows) == 2
    assert list(rows[0]["contenders"]) == []
    assert list(rows[1]["contenders"]) == [1, 2]


def test_logger_writes_snapshots_heatmaps_and_brains(tmp_path):
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.AGENTS.N = 2

    run_dir = _create_logger_run_dir(tmp_path)
    logger = DataLogger(str(run_dir))
    grid = Grid()
    registry = Registry()
    uid = registry.spawn_agent(0, 2, 2, -1, grid)

    logger.log_agent_snapshot(3, registry)
    logger.log_heatmap_snapshot(3, grid)
    logger.log_brains(3, registry)
    logger.close()

    assert (run_dir / "simulation_data.hdf5").exists()
    assert (run_dir / "brains" / "brains_tick_3.pt").exists()

    payload = torch.load(run_dir / "brains" / "brains_tick_3.pt", map_location="cpu", weights_only=False)
    assert payload["tick"] == 3
    assert payload["uid_to_slot"][uid] == 0
    assert uid in payload["by_uid"]
    assert payload["schema_versions"]["identity"] == cfg.SCHEMA.IDENTITY_SCHEMA_VERSION

    with h5py.File(run_dir / "simulation_data.hdf5", "r") as handle:
        assert "agent_snapshots/tick_3" in handle
        assert "heatmaps/density_tick_3" in handle
        assert "heatmaps/mass_tick_3" in handle
        assert "agent_identity/slot_uid_tick_3" in handle
        assert "agent_identity/slot_parent_uid_tick_3" in handle
