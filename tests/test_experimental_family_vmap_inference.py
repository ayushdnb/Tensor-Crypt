import pytest
import torch

from tensor_crypt.app import runtime as runtime_module
from tensor_crypt.app.runtime import validate_runtime_config
from tensor_crypt.config_bridge import cfg
from tensor_crypt.learning.ppo import PPO

try:
    import torch.func  # noqa: F401

    HAS_TORCH_FUNC = True
except Exception:
    HAS_TORCH_FUNC = False


def _force_single_family_roots():
    target = cfg.BRAIN.FAMILY_ORDER[0]
    cfg.BRAIN.INITIAL_FAMILY_ASSIGNMENT = "weighted_random"
    cfg.BRAIN.INITIAL_FAMILY_WEIGHTS = {
        family_id: (1.0 if family_id == target else 0.0)
        for family_id in cfg.BRAIN.FAMILY_ORDER
    }


def _enable_experimental_branch(family_id: str = "House Nocthar"):
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = family_id
    cfg.PERCEPT.OBS_MODE = "experimental_selfcentric_v1"
    cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True
    cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION = False


def _live_obs_and_slots(runtime):
    alive_indices = runtime.registry.get_alive_indices()
    alive_slots = [int(slot) for slot in alive_indices.detach().cpu().tolist()]
    obs = runtime.perception.build_observations(alive_indices)
    return obs, alive_slots


def _reference_gae(rewards, values, dones, last_value, last_done):
    if len(rewards) == 0:
        empty = torch.empty(0, device=cfg.SIM.DEVICE)
        return empty, empty

    gae = rewards[0].new_zeros(())
    returns = [rewards[0].new_zeros(()) for _ in rewards]

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            bootstrap_value = last_value
            bootstrap_done = last_done
        else:
            bootstrap_value = values[t + 1]
            bootstrap_done = dones[t]

        reward_t = rewards[t]
        value_t = values[t]
        delta = reward_t + cfg.PPO.GAMMA * bootstrap_value * (1 - bootstrap_done) - value_t
        gae = delta + cfg.PPO.GAMMA * cfg.PPO.LAMBDA * (1 - bootstrap_done) * gae
        returns[t] = gae + value_t

    returns_tensor = torch.stack(returns).reshape(-1)
    values_tensor = torch.stack(values).reshape(-1)
    advantages_tensor = returns_tensor - values_tensor
    return returns_tensor, advantages_tensor


def test_validate_runtime_config_allows_missing_torch_func_when_experimental_path_is_disabled(monkeypatch):
    real_import_module = runtime_module.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "torch.func":
            raise ImportError("torch.func unavailable for test")
        return real_import_module(name, package)

    cfg.SIM.DEVICE = "cpu"
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    monkeypatch.setattr(runtime_module.importlib, "import_module", fake_import_module)

    validate_runtime_config()

def test_validate_runtime_config_allows_cuda_device_check_without_local_torch_shadow(monkeypatch):
    cfg.SIM.DEVICE = "cuda:0"
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    validate_runtime_config()


def test_validate_runtime_config_rejects_missing_torch_func_when_experimental_path_is_enabled(monkeypatch):
    real_import_module = runtime_module.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "torch.func":
            raise ImportError("torch.func unavailable for test")
        return real_import_module(name, package)

    cfg.SIM.DEVICE = "cpu"
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 2
    monkeypatch.setattr(runtime_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ValueError, match="requires torch.func support"):
        validate_runtime_config()


def test_validate_runtime_config_rejects_experimental_branch_without_experimental_obs_mode():
    cfg.SIM.DEVICE = "cpu"
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = "House Nocthar"
    cfg.PERCEPT.OBS_MODE = "canonical_v2"
    cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True
    cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION = False

    with pytest.raises(ValueError, match="requires PERCEPT.OBS_MODE='experimental_selfcentric_v1'"):
        validate_runtime_config()


def test_validate_runtime_config_rejects_experimental_branch_with_family_shift_enabled():
    cfg.SIM.DEVICE = "cpu"
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = "House Nocthar"
    cfg.PERCEPT.OBS_MODE = "experimental_selfcentric_v1"
    cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True
    cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION = True

    with pytest.raises(ValueError, match="requires EVOL.ENABLE_FAMILY_SHIFT_MUTATION=False"):
        validate_runtime_config()

@pytest.mark.skipif(not HAS_TORCH_FUNC, reason="torch.func unavailable")
def test_experimental_family_vmap_forward_matches_loop_for_same_family_bucket(runtime_builder):
    _force_single_family_roots()
    runtime = runtime_builder(
        seed=701,
        agents=10,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    obs, alive_slots = _live_obs_and_slots(runtime)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    loop_logits, loop_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 2
    fast_logits, fast_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    torch.testing.assert_close(loop_logits, fast_logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loop_values, fast_values, rtol=1e-6, atol=1e-6)
    assert runtime.engine.last_inference_path_stats["vmap_slots"] == len(alive_slots)
    assert runtime.engine.last_inference_path_stats["family_vmap_buckets"] == 1


@pytest.mark.skipif(not HAS_TORCH_FUNC, reason="torch.func unavailable")
def test_experimental_family_vmap_threshold_keeps_small_buckets_on_loop(runtime_builder):
    _force_single_family_roots()
    runtime = runtime_builder(
        seed=702,
        agents=4,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    obs, alive_slots = _live_obs_and_slots(runtime)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    loop_logits, loop_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 8
    fast_logits, fast_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    torch.testing.assert_close(loop_logits, fast_logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loop_values, fast_values, rtol=1e-6, atol=1e-6)
    assert runtime.engine.last_inference_path_stats["vmap_slots"] == 0
    assert runtime.engine.last_inference_path_stats["loop_slots"] == len(alive_slots)


@pytest.mark.skipif(not HAS_TORCH_FUNC, reason="torch.func unavailable")
def test_experimental_family_vmap_does_not_replace_owned_modules(runtime_builder):
    _force_single_family_roots()
    runtime = runtime_builder(
        seed=703,
        agents=8,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    obs, alive_slots = _live_obs_and_slots(runtime)
    original_ids = {slot_idx: id(runtime.registry.brains[slot_idx]) for slot_idx in alive_slots}

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 2
    runtime.engine._batched_brain_forward(obs, alive_slots)

    assert {slot_idx: id(runtime.registry.brains[slot_idx]) for slot_idx in alive_slots} == original_ids


@pytest.mark.skipif(not HAS_TORCH_FUNC, reason="torch.func unavailable")
def test_experimental_family_vmap_skips_mixed_singleton_families(runtime_builder):
    cfg.BRAIN.INITIAL_FAMILY_ASSIGNMENT = "round_robin"
    runtime = runtime_builder(
        seed=704,
        agents=min(5, len(cfg.BRAIN.FAMILY_ORDER)),
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    obs, alive_slots = _live_obs_and_slots(runtime)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    loop_logits, loop_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 2
    fast_logits, fast_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    torch.testing.assert_close(loop_logits, fast_logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loop_values, fast_values, rtol=1e-6, atol=1e-6)
    assert runtime.engine.last_inference_path_stats["vmap_slots"] == 0
    assert runtime.engine.last_inference_path_stats["loop_slots"] == len(alive_slots)


@pytest.mark.skipif(not HAS_TORCH_FUNC, reason="torch.func unavailable")
def test_experimental_branch_family_vmap_forward_matches_loop(runtime_builder):
    _enable_experimental_branch("House Nocthar")
    runtime = runtime_builder(
        seed=705,
        agents=10,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    obs, alive_slots = _live_obs_and_slots(runtime)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = False
    loop_logits, loop_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET = 2
    fast_logits, fast_values = runtime.engine._batched_brain_forward(obs, alive_slots)

    torch.testing.assert_close(loop_logits, fast_logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loop_values, fast_values, rtol=1e-6, atol=1e-6)
    assert runtime.engine.last_inference_path_stats["vmap_slots"] == len(alive_slots)
    assert runtime.engine.last_inference_path_stats["family_vmap_buckets"] == 1


def test_ppo_gae_preallocation_matches_reference():
    ppo = PPO()
    rewards = [
        torch.tensor(1.0, device=cfg.SIM.DEVICE),
        torch.tensor(0.5, device=cfg.SIM.DEVICE),
        torch.tensor(0.25, device=cfg.SIM.DEVICE),
    ]
    values = [
        torch.tensor(0.8, device=cfg.SIM.DEVICE),
        torch.tensor(0.3, device=cfg.SIM.DEVICE),
        torch.tensor(0.1, device=cfg.SIM.DEVICE),
    ]
    dones = [
        torch.tensor(0.0, device=cfg.SIM.DEVICE),
        torch.tensor(0.0, device=cfg.SIM.DEVICE),
        torch.tensor(1.0, device=cfg.SIM.DEVICE),
    ]
    last_value = torch.tensor(0.0, device=cfg.SIM.DEVICE)
    last_done = torch.tensor(1.0, device=cfg.SIM.DEVICE)

    ref_returns, ref_advantages = _reference_gae(rewards, values, dones, last_value, last_done)
    got_returns, got_advantages = ppo._compute_returns_and_advantages(
        rewards,
        values,
        dones,
        last_value,
        last_done,
    )

    torch.testing.assert_close(got_returns, ref_returns, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got_advantages, ref_advantages, rtol=0.0, atol=0.0)


