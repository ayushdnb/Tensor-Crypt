import pytest
import torch

from tensor_crypt.config_bridge import cfg
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.population.evolution import Evolution
from tensor_crypt.world.spatial_grid import Grid


def _make_obs():
    return {
        "rays": torch.zeros(cfg.PERCEPT.NUM_RAYS, 5),
        "state": torch.zeros(2),
        "genome": torch.zeros(4),
        "position": torch.zeros(2),
        "context": torch.zeros(3),
    }


def test_compute_returns_respect_done_boundaries():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False

    ppo = PPO()
    returns, advantages = ppo._compute_returns_and_advantages(
        rewards=[torch.tensor(1.0), torch.tensor(2.0)],
        values=[torch.tensor(10.0), torch.tensor(20.0)],
        dones=[torch.tensor(1.0), torch.tensor(0.0)],
        last_value=torch.tensor(30.0),
        last_done=torch.tensor(0.0),
    )

    assert returns.tolist() == pytest.approx([1.0, 31.7], rel=1e-5)
    assert advantages.tolist() == pytest.approx([-9.0, 11.7], rel=1e-5)


def test_process_deaths_clears_agent_state():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 2

    ppo = PPO()
    grid = Grid()
    registry = Registry()
    evolution = Evolution(registry)
    uid = registry.spawn_agent(1, 2, 2, -1, grid)
    obs = _make_obs()
    ppo.store_transition(uid, obs, torch.tensor(0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0))
    ppo._get_optimizer(uid, registry.brains[1])
    registry.mark_dead(1, grid)

    evolution.process_deaths([1], ppo, death_tick=3)

    assert uid not in ppo.buffers_by_uid
    assert uid not in ppo.optimizers_by_uid
    assert registry.get_uid_for_slot(1) == -1


def test_invalid_minibatch_configuration_raises_clear_error():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 1
    cfg.PPO.BATCH_SZ = 2
    cfg.PPO.MINI_BATCHES = 4
    cfg.PPO.EPOCHS = 1

    ppo = PPO()
    grid = Grid()
    registry = Registry()
    uid = registry.spawn_agent(0, 2, 2, -1, grid)

    obs = _make_obs()
    for _ in range(2):
        ppo.store_transition(uid, obs, torch.tensor(0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))

    with pytest.raises(ValueError, match="cannot exceed trajectory batch size"):
        ppo.update(registry, None, {uid: obs}, {uid: torch.tensor(0.0)})


def test_update_trains_and_clears_buffer():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 1
    cfg.PPO.BATCH_SZ = 2
    cfg.PPO.MINI_BATCHES = 1
    cfg.PPO.EPOCHS = 1
    cfg.PPO.TARGET_KL = 0.0

    ppo = PPO()
    grid = Grid()
    registry = Registry()
    uid = registry.spawn_agent(0, 2, 2, -1, grid)

    obs = _make_obs()
    for _ in range(2):
        ppo.store_transition(uid, obs, torch.tensor(0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))

    stats = ppo.update(registry, None, {uid: obs}, {uid: torch.tensor(0.0)})

    assert len(stats) == 1
    assert stats[0]["agent_uid"] == uid
    assert stats[0]["agent_slot"] == 0
    assert len(ppo.buffers_by_uid[uid]) == 0
