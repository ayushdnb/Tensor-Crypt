import pandas as pd
import pytest
import torch

from tensor_crypt.config_bridge import cfg
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.learning.ppo import PPO
from tensor_crypt.world.spatial_grid import Grid


def _make_obs():
    return {
        "rays": torch.zeros(cfg.PERCEPT.NUM_RAYS, 5),
        "state": torch.zeros(2),
        "genome": torch.zeros(4),
        "position": torch.zeros(2),
        "context": torch.zeros(3),
    }


def test_uid_is_monotonic_across_slot_reuse(runtime_builder):
    runtime = runtime_builder(seed=101, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    initial_uid = registry.get_uid_for_slot(slot)

    child_uids = []
    for tick in (1, 2):
        registry.mark_dead(slot, runtime.grid)
        runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=tick)
        runtime.engine.respawn_controller.step(tick, registry, runtime.grid, runtime.data_logger)
        child_uids.append(registry.get_uid_for_slot(slot))

    assert initial_uid < child_uids[0] < child_uids[1]
    assert len({initial_uid, *child_uids}) == 3


def test_dead_uid_becomes_historical_and_unbound(runtime_builder):
    runtime = runtime_builder(seed=102, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
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
        torch.tensor(1.0),
    )
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=5)

    record = registry.uid_lifecycle[uid]
    assert record.is_active is False
    assert record.current_slot is None
    assert record.death_tick == 5
    assert registry.get_uid_for_slot(slot) == -1
    assert uid in registry.uid_lifecycle


def test_ppo_ownership_is_uid_keyed_not_slot_keyed():
    cfg.SIM.DEVICE = "cpu"
    cfg.LOG.AMP = False
    cfg.GRID.W = 8
    cfg.GRID.H = 8
    cfg.AGENTS.N = 4

    grid = Grid()
    registry = Registry()
    ppo = PPO()
    uid = registry.spawn_agent(3, 4, 4, -1, grid)

    ppo.store_transition_for_slot(
        registry,
        3,
        _make_obs(),
        torch.tensor(0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )

    assert uid in ppo.buffers_by_uid
    assert 3 not in ppo.buffers_by_uid


def test_new_child_uid_does_not_inherit_dead_uid_optimizer_state(runtime_builder):
    runtime = runtime_builder(seed=103, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    dead_uid = registry.get_uid_for_slot(slot)
    runtime.ppo._get_optimizer(dead_uid, registry.brains[slot])

    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=4)
    runtime.engine.respawn_controller.step(4, registry, runtime.grid, runtime.data_logger)
    child_uid = registry.get_uid_for_slot(slot)

    assert child_uid != dead_uid
    assert dead_uid not in runtime.ppo.optimizers_by_uid
    assert child_uid not in runtime.ppo.optimizers_by_uid


def test_spawn_event_logs_uid_and_slot_fields(runtime_builder):
    runtime = runtime_builder(seed=104, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())

    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=6)
    runtime.engine.respawn_controller.step(6, registry, runtime.grid, runtime.data_logger)
    runtime.data_logger.close()

    df = pd.read_parquet(runtime.data_logger.genealogy_path)
    row = df.iloc[0]
    assert {"child_idx", "parent_idx", "child_slot", "parent_slot", "child_uid", "parent_uid", "identity_schema_version", "telemetry_schema_version"}.issubset(df.columns)
    assert row["child_slot"] == slot
    assert row["child_uid"] == registry.get_uid_for_slot(slot)


def test_shadow_columns_match_canonical_uid_surfaces(runtime_builder):
    runtime = runtime_builder(seed=105, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    registry = runtime.registry

    expected_uid_shadow = torch.where(registry.slot_uid >= 0, registry.slot_uid, torch.full_like(registry.slot_uid, -1)).to(torch.float32)
    expected_parent_shadow = torch.where(registry.slot_parent_uid >= 0, registry.slot_parent_uid, torch.full_like(registry.slot_parent_uid, -1)).to(torch.float32)

    assert torch.equal(registry.data[registry.AGENT_UID_SHADOW], expected_uid_shadow)
    assert torch.equal(registry.data[registry.PARENT_UID_SHADOW], expected_parent_shadow)


def test_terminal_transition_is_stored_before_uid_finalization(runtime_builder):
    runtime = runtime_builder(seed=106, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    uid = registry.get_uid_for_slot(slot)
    registry.data[registry.HP, slot] = 0.0

    seen = {}
    original_process_deaths = runtime.evolution.process_deaths

    def wrapped_process_deaths(deaths, ppo, death_tick):
        seen["buffer_present_before_clear"] = uid in ppo.buffers_by_uid
        return original_process_deaths(deaths, ppo, death_tick)

    runtime.evolution.process_deaths = wrapped_process_deaths
    runtime.engine.step()

    assert seen["buffer_present_before_clear"] is True
    assert uid not in runtime.ppo.buffers_by_uid
