import random

import torch

from ..config_bridge import cfg
from .spatial_grid import Grid


@torch.no_grad()
def add_random_walls(grid: Grid):
    """Carve random, one-cell-thick wall segments onto the grid."""
    n_segments = cfg.MAPGEN.RANDOM_WALLS
    seg_min = cfg.MAPGEN.WALL_SEG_MIN
    seg_max = cfg.MAPGEN.WALL_SEG_MAX
    avoid_margin = cfg.MAPGEN.WALL_AVOID_MARGIN

    if n_segments <= 0:
        return

    H, W = grid.H, grid.W
    dirs8 = torch.tensor(
        [
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=torch.long,
        device=grid.device,
    )

    def _place_wall_cell(x: int, y: int):
        if 0 <= x < W and 0 <= y < H:
            if grid.grid[0, y, x].item() == 1.0:
                return
            grid.grid[0, y, x] = 1.0
            grid.grid[1, y, x] = 0.0
            grid.grid[2, y, x] = -1.0

    x0_min = max(1, avoid_margin)
    x0_max = W - max(1, avoid_margin) - 1
    y0_min = max(1, avoid_margin)
    y0_max = H - max(1, avoid_margin) - 1

    if x0_min >= x0_max or y0_min >= y0_max:
        return

    for _ in range(n_segments):
        x = random.randint(x0_min, x0_max)
        y = random.randint(y0_min, y0_max)
        length = random.randint(seg_min, seg_max)

        _place_wall_cell(x, y)
        last_dir_idx = random.randrange(8)

        for _ in range(length):
            if random.random() < 0.70:
                d_idx = last_dir_idx
            else:
                d_idx = (last_dir_idx + random.choice([-2, -1, 1, 2])) % 8
            last_dir_idx = d_idx

            dx, dy = dirs8[d_idx].tolist()
            x = max(1, min(W - 2, x + dx))
            y = max(1, min(H - 2, y + dy))
            _place_wall_cell(x, y)


@torch.no_grad()
def add_random_hzones(grid: Grid):
    """Add random rectangular heal zones to the grid."""
    heal_count = cfg.MAPGEN.HEAL_ZONE_COUNT
    heal_ratio = cfg.MAPGEN.HEAL_ZONE_SIZE_RATIO
    heal_rate = cfg.MAPGEN.HEAL_RATE
    H, W = grid.H, grid.W

    if heal_count <= 0 or heal_ratio <= 0.0:
        return

    h_side = max(1, int(round(heal_ratio * H)))
    w_side = max(1, int(round(heal_ratio * W)))

    for _ in range(heal_count):
        x1 = random.randint(1, max(1, W - w_side - 2))
        y1 = random.randint(1, max(1, H - h_side - 2))
        x2 = x1 + w_side
        y2 = y1 + h_side
        grid.add_hzone(x1, y1, x2, y2, heal_rate)
