import pytest
import torch

from tensor_crypt.app.runtime import validate_runtime_config
from tensor_crypt.config_bridge import cfg
from tensor_crypt.simulation.engine import compute_ppo_reward_tensor


def test_compute_ppo_reward_tensor_preserves_legacy_sq_health_ratio_behavior():
    cfg.PPO.REWARD_FORM = "sq_health_ratio"
    cfg.PPO.REWARD_GATE_MODE = "off"
    cfg.PPO.REWARD_GATE_THRESHOLD = 0.0
    cfg.PPO.REWARD_BELOW_GATE_VALUE = 0.0

    hp = torch.tensor([-2.0, 5.0, 12.0], dtype=torch.float32)
    hp_max = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)

    rewards = compute_ppo_reward_tensor(hp, hp_max)
    expected = torch.clamp(hp / hp_max.clamp_min(1e-6), 0.0, 1.0).square()

    assert torch.allclose(rewards, expected)
    assert rewards.shape == hp.shape
    assert rewards.device == hp.device
    assert rewards.dtype == hp.dtype


def test_compute_ppo_reward_tensor_applies_hp_ratio_gate_inclusively_at_threshold():
    cfg.PPO.REWARD_GATE_MODE = "hp_ratio_min"
    cfg.PPO.REWARD_GATE_THRESHOLD = 0.5
    cfg.PPO.REWARD_BELOW_GATE_VALUE = 0.0

    hp = torch.tensor([4.0, 5.0, 8.0], dtype=torch.float32)
    hp_max = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)

    rewards = compute_ppo_reward_tensor(hp, hp_max)

    assert torch.allclose(rewards, torch.tensor([0.0, 0.25, 0.64], dtype=torch.float32))


def test_compute_ppo_reward_tensor_uses_configured_below_gate_value():
    cfg.PPO.REWARD_GATE_MODE = "hp_ratio_min"
    cfg.PPO.REWARD_GATE_THRESHOLD = 0.7
    cfg.PPO.REWARD_BELOW_GATE_VALUE = -0.25

    hp = torch.tensor([6.0, 8.0], dtype=torch.float32)
    hp_max = torch.tensor([10.0, 10.0], dtype=torch.float32)

    rewards = compute_ppo_reward_tensor(hp, hp_max)

    assert torch.allclose(rewards, torch.tensor([-0.25, 0.64], dtype=torch.float32))


def test_compute_ppo_reward_tensor_supports_absolute_hp_gate_mode():
    cfg.PPO.REWARD_GATE_MODE = "hp_abs_min"
    cfg.PPO.REWARD_GATE_THRESHOLD = 6.0
    cfg.PPO.REWARD_BELOW_GATE_VALUE = 0.0

    hp = torch.tensor([5.0, 6.0], dtype=torch.float32)
    hp_max = torch.tensor([10.0, 10.0], dtype=torch.float32)

    rewards = compute_ppo_reward_tensor(hp, hp_max)

    assert torch.allclose(rewards, torch.tensor([0.0, 0.36], dtype=torch.float32), atol=1e-6)


@pytest.mark.parametrize(
    ("reward_form", "gate_mode", "threshold", "below_gate_value", "match"),
    [
        ("linear_health", "off", 0.0, 0.0, "PPO.REWARD_FORM must be one of"),
        ("sq_health_ratio", "unknown", 0.0, 0.0, "PPO.REWARD_GATE_MODE must be one of"),
        ("sq_health_ratio", "hp_ratio_min", 1.1, 0.0, r"within \[0\.0, 1\.0\]"),
        ("sq_health_ratio", "hp_abs_min", -0.1, 0.0, ">= 0.0"),
        ("sq_health_ratio", "off", 0.0, float("nan"), "PPO.REWARD_BELOW_GATE_VALUE must be finite"),
    ],
)
def test_validate_runtime_config_rejects_invalid_reward_surface(reward_form, gate_mode, threshold, below_gate_value, match):
    cfg.SIM.DEVICE = "cpu"
    cfg.PPO.REWARD_FORM = reward_form
    cfg.PPO.REWARD_GATE_MODE = gate_mode
    cfg.PPO.REWARD_GATE_THRESHOLD = threshold
    cfg.PPO.REWARD_BELOW_GATE_VALUE = below_gate_value

    with pytest.raises(ValueError, match=match):
        validate_runtime_config()


def test_engine_step_applies_threshold_gate_to_live_reward_path(runtime_builder):
    cfg.PPO.REWARD_GATE_MODE = "hp_ratio_min"
    cfg.PPO.REWARD_GATE_THRESHOLD = 0.5
    cfg.PPO.REWARD_BELOW_GATE_VALUE = 0.0

    runtime = runtime_builder(seed=51, width=10, height=10, agents=2, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    slot = int(runtime.registry.get_alive_indices()[0].item())
    hp_max = float(runtime.registry.data[runtime.registry.HP_MAX, slot].item())
    captured = {}

    original_store = runtime.ppo.store_transition_for_slot

    def wrapped_store(registry, slot_idx, obs, action, log_prob, reward, value, done):
        if slot_idx == slot:
            captured["reward"] = float(reward.item())
        return original_store(registry, slot_idx, obs, action, log_prob, reward, value, done)

    runtime.ppo.store_transition_for_slot = wrapped_store
    runtime.physics.step = lambda actions: {"wall_collisions": 0, "rams": 0, "contests": 0}

    def force_half_health():
        runtime.registry.data[runtime.registry.HP, slot] = 0.5 * hp_max

    runtime.physics.apply_environment_effects = force_half_health
    runtime.engine.step()

    assert captured["reward"] == pytest.approx(0.25)


def test_threshold_gated_reward_mode_keeps_ppo_update_flow_running(runtime_builder):
    cfg.PPO.REWARD_GATE_MODE = "hp_ratio_min"
    cfg.PPO.REWARD_GATE_THRESHOLD = 0.5
    cfg.PPO.REWARD_BELOW_GATE_VALUE = 0.0

    runtime = runtime_builder(seed=52, width=10, height=10, agents=2, walls=0, hzones=0, update_every=2, batch_size=2, mini_batches=1)
    runtime.physics.step = lambda actions: {"wall_collisions": 0, "rams": 0, "contests": 0}
    runtime.physics.apply_environment_effects = lambda: None

    tracked_uid = int(runtime.registry.get_uid_for_slot(int(runtime.registry.get_alive_indices()[0].item())))
    for _ in range(3):
        runtime.engine.step()

    assert runtime.ppo.training_state_by_uid[tracked_uid].ppo_updates >= 1
