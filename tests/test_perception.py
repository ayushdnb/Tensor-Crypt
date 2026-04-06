import math

import torch

from tensor_crypt.agents.brain import Brain
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.config_bridge import cfg
from tensor_crypt.world.perception import Perception
from tensor_crypt.world.spatial_grid import Grid


def test_build_observations_empty_batch_shapes():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.AGENTS.N = 1
    cfg.PERCEPT.NUM_RAYS = 8
    cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS = True

    grid = Grid()
    registry = Registry()
    perception = Perception(grid, registry)

    obs = perception.build_observations(torch.empty(0, dtype=torch.long))

    assert obs["rays"].shape == (0, 8, 5)
    assert obs["state"].shape == (0, 2)
    assert obs["genome"].shape == (0, 4)
    assert obs["position"].shape == (0, 2)
    assert obs["context"].shape == (0, 3)
    assert obs["canonical_rays"].shape == (0, 8, 8)
    assert obs["canonical_self"].shape == (0, 11)
    assert obs["canonical_context"].shape == (0, 3)


def test_perception_self_exclusion_and_cardinal_walls_use_one_hot():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 16
    cfg.GRID.H = 16
    cfg.AGENTS.N = 1
    cfg.PERCEPT.NUM_RAYS = 8

    grid = Grid()
    registry = Registry()
    registry.spawn_agent(0, 8, 8, -1, grid)
    perception = Perception(grid, registry)

    obs = perception.build_observations(registry.get_alive_indices())
    canonical_rays = obs["canonical_rays"][0]

    assert torch.allclose(canonical_rays[:, 0:3].sum(dim=-1), torch.ones(cfg.PERCEPT.NUM_RAYS))
    assert canonical_rays[:, 1].sum().item() == 0.0

    for ray_idx in (0, 2, 4, 6):
        assert canonical_rays[ray_idx, 2].item() == 1.0
        assert 0.0 < canonical_rays[ray_idx, 3].item() <= 1.0

    assert torch.isfinite(canonical_rays).all()
    assert obs["rays"].shape[-1] == 5
    assert torch.isfinite(obs["rays"]).all()


def test_zero_vision_is_stable_and_finite():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 10
    cfg.GRID.H = 10
    cfg.AGENTS.N = 1
    cfg.PERCEPT.NUM_RAYS = 8

    grid = Grid()
    registry = Registry()
    registry.spawn_agent(0, 5, 5, -1, grid)
    registry.data[registry.VISION, 0] = 0.0
    perception = Perception(grid, registry)

    obs = perception.build_observations(registry.get_alive_indices())

    assert torch.isfinite(obs["canonical_rays"]).all()
    assert torch.isfinite(obs["rays"]).all()
    assert torch.equal(obs["canonical_rays"], torch.zeros_like(obs["canonical_rays"]))
    assert torch.equal(obs["rays"], torch.zeros_like(obs["rays"]))


def test_agent_hit_masks_target_fields_and_adapter_is_bounded():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 16
    cfg.GRID.H = 16
    cfg.AGENTS.N = 2
    cfg.PERCEPT.NUM_RAYS = 8

    grid = Grid()
    registry = Registry()
    registry.spawn_agent(
        0,
        5,
        5,
        -1,
        grid,
        traits={"mass": 2.0, "vision": 8.0, "hp_max": 10.0, "metab": 0.0},
    )
    registry.spawn_agent(
        1,
        8,
        5,
        -1,
        grid,
        traits={"mass": 7.0, "vision": 8.0, "hp_max": 20.0, "metab": 0.0},
    )
    registry.data[registry.HP, 1] = 10.0
    perception = Perception(grid, registry)

    obs = perception.build_observations(torch.tensor([0], dtype=torch.long))
    canonical_rays = obs["canonical_rays"][0]

    east_ray = canonical_rays[0]
    assert east_ray[1].item() == 1.0
    assert east_ray[2].item() == 0.0
    assert 0.0 < east_ray[3].item() <= 1.0
    assert east_ray[6].item() > 0.0
    assert east_ray[7].item() == 0.5

    non_agent_mask = canonical_rays[:, 1] < 0.5
    assert torch.equal(canonical_rays[:, 6][non_agent_mask], torch.zeros_like(canonical_rays[:, 6][non_agent_mask]))
    assert torch.equal(canonical_rays[:, 7][non_agent_mask], torch.zeros_like(canonical_rays[:, 7][non_agent_mask]))

    assert torch.isfinite(obs["rays"]).all()
    assert torch.max(torch.abs(obs["rays"])).item() <= 1.0


def test_distance_to_center_and_bridge_fields_are_correct():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 11
    cfg.GRID.H = 11
    cfg.AGENTS.N = 1
    cfg.PERCEPT.NUM_RAYS = 8

    grid = Grid()
    registry = Registry()
    registry.spawn_agent(0, 5, 5, -1, grid)
    registry.tick_counter = 7
    registry.data[registry.TICK_BORN, 0] = 2.0
    perception = Perception(grid, registry)

    obs = perception.build_observations(registry.get_alive_indices())
    canonical_self = obs["canonical_self"][0]

    assert canonical_self[8].item() == 0.0
    assert obs["context"][0, 0].item() == canonical_self[8].item()
    assert obs["context"][0, 1].item() == canonical_self[9].item()
    assert obs["state"][0, 0].item() == canonical_self[0].item()
    assert obs["state"][0, 1].item() == canonical_self[10].item()

    brain = Brain().to("cpu")
    logits, value = brain(obs)
    assert logits.shape == (1, 9)
    assert value.shape == (1, 1)
