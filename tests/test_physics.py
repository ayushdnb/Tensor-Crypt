import torch

from tensor_crypt.config_bridge import cfg
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.world.physics import Physics
from tensor_crypt.world.spatial_grid import Grid


def _spawn(registry, grid, idx, x, y, *, mass=2.0, hp=10.0, hp_max=10.0):
    registry.spawn_agent(
        idx,
        x,
        y,
        -1,
        grid,
        traits={"mass": mass, "vision": 8.0, "hp_max": hp_max, "metab": 0.0},
    )
    registry.data[registry.HP, idx] = hp
    registry.data[registry.HP_MAX, idx] = hp_max


def test_wall_collision_keeps_position_and_logs_event():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 1

    grid = Grid()
    registry = Registry()
    _spawn(registry, grid, 0, 1, 1)
    physics = Physics(grid, registry)

    actions = torch.zeros(registry.max_agents, dtype=torch.long)
    actions[0] = 7
    stats = physics.step(actions)

    assert stats["wall_collisions"] == 1
    assert physics.collision_log[0]["kind"] == "wall"
    assert int(registry.data[registry.X, 0].item()) == 1
    assert int(registry.data[registry.Y, 0].item()) == 1
    assert registry.data[registry.HP, 0].item() < 10.0


def test_ram_damages_both_agents_without_corrupting_occupancy():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 7
    cfg.GRID.H = 7
    cfg.AGENTS.N = 2

    grid = Grid()
    registry = Registry()
    _spawn(registry, grid, 0, 2, 3, mass=2.0)
    _spawn(registry, grid, 1, 3, 3, mass=1.0)
    physics = Physics(grid, registry)

    actions = torch.zeros(registry.max_agents, dtype=torch.long)
    actions[0] = 3
    stats = physics.step(actions)

    assert stats["rams"] == 1
    assert physics.collision_log[0]["kind"] == "ram"
    assert registry.data[registry.HP, 0].item() < 10.0
    assert registry.data[registry.HP, 1].item() < 10.0
    assert grid.get_agent_at(2, 3) == 0
    assert grid.get_agent_at(3, 3) == 1


def test_contest_resolution_moves_winner_and_leaves_loser_in_place():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.AGENTS.N = 2

    grid = Grid()
    registry = Registry()
    _spawn(registry, grid, 0, 2, 3, mass=3.0)
    _spawn(registry, grid, 1, 4, 3, mass=1.0)
    physics = Physics(grid, registry)

    actions = torch.zeros(registry.max_agents, dtype=torch.long)
    actions[0] = 3
    actions[1] = 7
    stats = physics.step(actions)

    assert stats["contests"] == 1
    assert grid.get_agent_at(3, 3) == 0
    assert grid.get_agent_at(4, 3) == 1
    assert physics.collision_log[0]["winner"] == 0


def test_process_deaths_clears_grid_cells():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 1

    grid = Grid()
    registry = Registry()
    _spawn(registry, grid, 0, 2, 2)
    physics = Physics(grid, registry)
    registry.data[registry.HP, 0] = 0.0

    deaths = physics.process_deaths()

    assert deaths == [0]
    assert registry.data[registry.ALIVE, 0].item() == 0.0
    assert grid.get_agent_at(2, 2) == -1
