import torch
import pytest

from tensor_crypt.app.runtime import setup_determinism
from tensor_crypt.config_bridge import cfg
from tensor_crypt.world.procedural_map import add_random_hzones, add_random_walls
from tensor_crypt.world.spatial_grid import Grid


def test_grid_border_walls_and_cell_helpers():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 8
    cfg.GRID.H = 6

    grid = Grid()

    for x in range(grid.W):
        assert grid.is_wall(x, 0)
        assert grid.is_wall(x, grid.H - 1)
    for y in range(grid.H):
        assert grid.is_wall(0, y)
        assert grid.is_wall(grid.W - 1, y)

    grid.set_cell(3, 3, 7, 1.5)
    assert grid.get_agent_at(3, 3) == 7
    grid.clear_cell(3, 3)
    assert grid.get_agent_at(3, 3) == -1
    assert grid.get_agent_at(-1, 0) == -1
    assert grid.get_h_rate(-1, 0) == 0.0


@pytest.mark.parametrize(
    ("mode", "expected_overlap"),
    [("max_abs", -2.0), ("sum_clamped", -1.0), ("last_wins", -2.0)],
)
def test_heal_zone_overlap_modes_and_gradient(mode, expected_overlap):
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.GRID.HZ_OVERLAP_MODE = mode
    cfg.GRID.HZ_SUM_CLAMP = 1.5

    grid = Grid()
    grid.add_hzone(2, 2, 4, 4, 1.0)
    grid.add_hzone(3, 3, 5, 5, -2.0)

    assert grid.get_h_rate(3, 3) == pytest.approx(expected_overlap)
    grad_x, grad_y = grid.compute_h_gradient()
    assert grad_x.shape == (grid.H, grid.W)
    assert grad_y.shape == (grid.H, grid.W)
    assert torch.isfinite(grad_x).all()
    assert torch.isfinite(grad_y).all()


def test_mapgen_stays_in_bounds_and_preserves_wall_cells_empty():
    cfg.SIM.DEVICE = "cpu"
    cfg.SIM.SEED = 11
    cfg.GRID.W = 12
    cfg.GRID.H = 10
    cfg.MAPGEN.RANDOM_WALLS = 5
    cfg.MAPGEN.WALL_SEG_MIN = 2
    cfg.MAPGEN.WALL_SEG_MAX = 4
    cfg.MAPGEN.HEAL_ZONE_COUNT = 3
    cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO = 0.25
    cfg.MAPGEN.HEAL_RATE = 0.5

    setup_determinism()
    grid = Grid()
    add_random_walls(grid)
    add_random_hzones(grid)

    assert torch.all(grid.grid[2][grid.grid[0] > 0.5] == -1)
    for zone in grid.hzones:
        assert 0 <= zone["x1"] <= zone["x2"] < grid.W
        assert 0 <= zone["y1"] <= zone["y2"] < grid.H
