import math

import torch

from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.config_bridge import cfg
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.world.spatial_grid import Grid


def _make_obs():
    return {
        "canonical_rays": torch.zeros(cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.CANONICAL_RAY_FEATURES),
        "canonical_self": torch.zeros(cfg.PERCEPT.CANONICAL_SELF_FEATURES),
        "canonical_context": torch.zeros(cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES),
    }


def test_normalize_advantages_handles_singleton_without_nan():
    normalized = PPO._normalize_advantages(torch.tensor([3.5]))
    assert torch.allclose(normalized, torch.tensor([0.0]))


def test_singleton_batch_update_does_not_nan_during_advantage_normalization():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 6
    cfg.GRID.H = 6
    cfg.AGENTS.N = 1
    cfg.PPO.BATCH_SZ = 1
    cfg.PPO.MINI_BATCHES = 1
    cfg.PPO.EPOCHS = 1
    cfg.PPO.TARGET_KL = 0.0

    grid = Grid()
    registry = Registry()
    ppo = PPO()
    uid = registry.spawn_agent(0, 2, 2, -1, grid)

    obs = _make_obs()
    ppo.store_transition(
        uid,
        obs,
        torch.tensor(0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )

    stats = ppo.update(registry, None, {uid: obs}, {uid: torch.tensor(0.0)}, tick=1)

    assert len(stats) == 1
    assert stats[0]["agent_uid"] == uid
    for key in ("policy_loss", "value_loss", "entropy", "kl_div", "grad_norm"):
        assert math.isfinite(stats[0][key]), key
    assert ppo.training_state_by_uid[uid].ppo_updates == 1
