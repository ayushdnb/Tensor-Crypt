import pytest
import torch

from tensor_crypt.checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    restore_runtime_checkpoint,
    validate_runtime_checkpoint,
)
from tensor_crypt.config_bridge import cfg
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.world.spatial_grid import Grid


def _make_obs():
    return {
        "canonical_rays": torch.zeros(cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.CANONICAL_RAY_FEATURES),
        "canonical_self": torch.zeros(cfg.PERCEPT.CANONICAL_SELF_FEATURES),
        "canonical_context": torch.zeros(cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES),
    }


def test_training_counters_increment_and_remain_uid_owned(runtime_builder):
    runtime = runtime_builder(
        seed=401,
        width=10,
        height=10,
        agents=1,
        walls=0,
        hzones=0,
        update_every=1,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )
    uid = runtime.registry.get_uid_for_slot(0)
    runtime.engine.step()
    runtime.engine.step()

    state = runtime.ppo.training_state_by_uid[uid]
    assert state.env_steps >= 2
    assert state.ppo_updates >= 1
    assert state.optimizer_steps >= 1
    assert state.last_buffer_size >= 2
    assert uid in runtime.ppo.training_state_by_uid
    assert set(runtime.ppo.training_state_by_uid.keys()) == {uid}


def test_inactive_uid_buffer_is_counted_as_truncated_and_dropped(runtime_builder):
    runtime = runtime_builder(
        seed=402,
        width=10,
        height=10,
        agents=4,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    uid = registry.get_uid_for_slot(slot)

    runtime.ppo.store_transition_for_slot(
        registry,
        slot,
        _make_obs(),
        torch.tensor(0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=7)

    assert uid not in runtime.ppo.buffers_by_uid
    assert runtime.ppo.training_state_by_uid[uid].truncated_rollouts == 1


def test_checkpoint_restore_preserves_training_state_optimizer_and_buffer_schema(runtime_builder):
    runtime = runtime_builder(
        seed=403,
        width=10,
        height=10,
        agents=2,
        walls=0,
        hzones=0,
        update_every=1,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )

    runtime.engine.step()
    runtime.engine.step()
    uid = runtime.registry.get_uid_for_slot(int(runtime.registry.get_alive_indices()[0].item()))
    bundle = capture_runtime_checkpoint(runtime)

    restored = runtime_builder(
        seed=404,
        width=10,
        height=10,
        agents=2,
        walls=0,
        hzones=0,
        update_every=1,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )
    restore_runtime_checkpoint(restored, bundle)

    assert restored.ppo.training_state_by_uid[uid].serialize() == runtime.ppo.training_state_by_uid[uid].serialize()
    assert uid in restored.ppo.optimizers_by_uid
    assert bundle["ppo_state"]["optimizer_metadata_by_uid"][uid]["param_names"]


def test_validate_runtime_checkpoint_rejects_unknown_training_state_uid(runtime_builder):
    runtime = runtime_builder(seed=405, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    bundle = capture_runtime_checkpoint(runtime)
    bundle["ppo_state"]["training_state_by_uid"][999999] = {
        "env_steps": 1,
        "ppo_updates": 0,
        "optimizer_steps": 0,
        "truncated_rollouts": 0,
        "last_kl": 0.0,
        "last_entropy": 0.0,
        "last_value_loss": 0.0,
        "last_policy_loss": 0.0,
        "last_grad_norm": 0.0,
        "last_buffer_size": 0,
        "last_update_tick": -1,
    }

    with pytest.raises(ValueError, match="unknown UID"):
        validate_runtime_checkpoint(bundle, cfg)


def test_restore_rejects_optimizer_topology_mismatch(runtime_builder):
    runtime = runtime_builder(
        seed=406,
        width=10,
        height=10,
        agents=1,
        walls=0,
        hzones=0,
        update_every=1,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )
    runtime.engine.step()
    runtime.engine.step()
    bundle = capture_runtime_checkpoint(runtime)

    uid = next(iter(bundle["ppo_state"]["optimizer_metadata_by_uid"].keys()))
    bundle["ppo_state"]["optimizer_metadata_by_uid"][uid]["param_shapes"][0] = [999]

    restored = runtime_builder(
        seed=407,
        width=10,
        height=10,
        agents=1,
        walls=0,
        hzones=0,
        update_every=1,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )
    with pytest.raises(ValueError, match="parameter shapes"):
        restore_runtime_checkpoint(restored, bundle)


def test_same_family_agents_keep_distinct_optimizers_even_under_family_ordering(runtime_builder):
    runtime = runtime_builder(
        seed=408,
        width=12,
        height=12,
        agents=10,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=99,
        mini_batches=1,
        policy_noise=0.0,
    )
    registry = runtime.registry

    by_family = {}
    for slot in registry.get_alive_indices().tolist():
        family_id = registry.get_family_for_slot(slot)
        by_family.setdefault(family_id, []).append(slot)

    family_slots = next(slots for slots in by_family.values() if len(slots) >= 2)
    slot_a, slot_b = family_slots[:2]
    uid_a = registry.get_uid_for_slot(slot_a)
    uid_b = registry.get_uid_for_slot(slot_b)

    opt_a = runtime.ppo._get_optimizer(uid_a, registry.brains[slot_a])
    opt_b = runtime.ppo._get_optimizer(uid_b, registry.brains[slot_b])

    assert uid_a != uid_b
    assert opt_a is not opt_b
    assert runtime.ppo.optimizers_by_uid[uid_a] is opt_a
    assert runtime.ppo.optimizers_by_uid[uid_b] is opt_b


def test_buffer_round_trip_preserves_non_terminal_bootstrap_state():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.AGENTS.N = 1

    grid = Grid()
    registry = Registry()
    ppo = PPO()
    uid = registry.spawn_agent(0, 3, 3, -1, grid)

    ppo.store_transition(uid, _make_obs(), torch.tensor(0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))
    ppo.stage_bootstrap_for_uid(uid, _make_obs(), torch.tensor(0.0), finalization_kind="unit_test_active_bootstrap")
    payload = ppo.serialize_all_buffers()

    restored = PPO()
    restored.load_serialized_buffers(payload, device="cpu")
    restored_buffer = restored.buffers_by_uid[uid]

    assert restored_buffer.bootstrap_obs is not None
    assert float(restored_buffer.bootstrap_done.item()) == 0.0
    assert restored_buffer.finalization_kind == "unit_test_active_bootstrap"


def test_compatibility_update_bridge_still_trains(runtime_builder):
    runtime = runtime_builder(
        seed=409,
        width=8,
        height=8,
        agents=1,
        walls=0,
        hzones=0,
        update_every=99,
        batch_size=2,
        mini_batches=1,
        policy_noise=0.0,
    )
    registry = runtime.registry
    uid = registry.get_uid_for_slot(0)

    obs = _make_obs()
    for _ in range(2):
        runtime.ppo.store_transition(uid, obs, torch.tensor(0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))

    stats = runtime.ppo.update(
        registry,
        None,
        {uid: obs},
        {uid: torch.tensor(0.0)},
        tick=11,
    )

    assert len(stats) == 1
    assert stats[0]["agent_uid"] == uid
    assert runtime.ppo.training_state_by_uid[uid].ppo_updates == 1