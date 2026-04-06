from typing import List, Optional, Tuple

import torch

from ..config_bridge import cfg


class Grid:
    """
    Dense tensor-backed world grid.

    Channel contract for `self.grid`:
    - channel 0: occupancy (0 empty, 1 wall)
    - channel 1: H-zone rate field
    - channel 2: agent slot id (-1 empty)
    - channel 3: agent mass
    """

    def __init__(self):
        self.device = cfg.SIM.DEVICE
        self.W = cfg.GRID.W
        self.H = cfg.GRID.H

        self.grid = torch.zeros(4, self.H, self.W, device=self.device, dtype=torch.float32)
        self.grid[2] = -1.0

        self.hzones: List[dict] = []
        self.next_hzone_id = 0

        self._create_border_walls()

    def _create_border_walls(self):
        self.grid[0, 0, :] = 1.0
        self.grid[0, -1, :] = 1.0
        self.grid[0, :, 0] = 1.0
        self.grid[0, :, -1] = 1.0

    def add_hzone(self, x1: int, y1: int, x2: int, y2: int, rate: float):
        zone = {
            "id": self.next_hzone_id,
            "x1": min(x1, x2),
            "y1": min(y1, y2),
            "x2": max(x1, x2),
            "y2": max(y1, y2),
            "rate": rate,
            "active": True,
        }
        self.next_hzone_id += 1
        self.hzones.append(zone)
        self.paint_hzones()

    def paint_hzones(self):
        if cfg.GRID.HZ_CLEAR_EACH_TICK:
            self.grid[1] = 0.0

        for zone in self.hzones:
            if not zone["active"]:
                continue

            x1, y1 = zone["x1"], zone["y1"]
            x2, y2 = zone["x2"], zone["y2"]
            rate = zone["rate"]

            x1 = max(0, min(x1, self.W - 1))
            x2 = max(0, min(x2, self.W - 1))
            y1 = max(0, min(y1, self.H - 1))
            y2 = max(0, min(y2, self.H - 1))

            if cfg.GRID.HZ_OVERLAP_MODE == "max_abs":
                current = self.grid[1, y1 : y2 + 1, x1 : x2 + 1]
                new_val = torch.where(
                    abs(rate) > torch.abs(current),
                    torch.tensor(rate, device=self.device),
                    current,
                )
                self.grid[1, y1 : y2 + 1, x1 : x2 + 1] = new_val
            elif cfg.GRID.HZ_OVERLAP_MODE == "sum_clamped":
                self.grid[1, y1 : y2 + 1, x1 : x2 + 1] += rate
            elif cfg.GRID.HZ_OVERLAP_MODE == "last_wins":
                self.grid[1, y1 : y2 + 1, x1 : x2 + 1] = rate

        if cfg.GRID.HZ_OVERLAP_MODE == "sum_clamped":
            self.grid[1] = torch.clamp(self.grid[1], -cfg.GRID.HZ_SUM_CLAMP, cfg.GRID.HZ_SUM_CLAMP)

    def find_hzone_at(self, x: int, y: int) -> Optional[int]:
        for zone in reversed(self.hzones):
            if zone["active"] and zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
                return zone["id"]
        return None

    def get_hzone(self, zone_id: int) -> Optional[dict]:
        for zone in self.hzones:
            if zone["id"] == zone_id:
                return zone
        return None

    def update_hzone_rate(self, zone_id: int, new_rate: float):
        zone = self.get_hzone(zone_id)
        if zone:
            zone["rate"] = max(-2.0, min(2.0, new_rate))
            self.paint_hzones()

    def is_wall(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return True
        return self.grid[0, y, x].item() > 0.5

    def get_h_rate(self, x: int, y: int) -> float:
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return 0.0
        return self.grid[1, y, x].item()

    def set_cell(self, x: int, y: int, slot_idx: float, mass: float):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.grid[2, y, x] = slot_idx
            self.grid[3, y, x] = mass

    def clear_cell(self, x: int, y: int):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.grid[2, y, x] = -1.0
            self.grid[3, y, x] = 0.0

    def get_agent_at(self, x: int, y: int) -> int:
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return -1
        slot_idx = self.grid[2, y, x].item()
        return int(slot_idx) if slot_idx >= 0 else -1

    def compute_h_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.grid[1]
        grad_x = torch.zeros_like(h)
        grad_y = torch.zeros_like(h)
        grad_x[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0
        grad_y[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0
        return grad_x, grad_y
