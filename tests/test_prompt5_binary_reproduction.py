import pytest
import torch

from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.population.reproduction import trait_values_from_latent


def test_binary_birth_assigns_distinct_parent_roles(runtime_builder):
    runtime = runtime_builder(seed=501, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=3)
    runtime.engine.respawn_controller.step(3, registry, runtime.grid, runtime.data_logger)

    child_uid = registry.get_uid_for_slot(slot)
    roles = registry.get_parent_roles_for_uid(child_uid)
    assert roles["brain_parent_uid"] != -1
    assert roles["trait_parent_uid"] != -1
    assert roles["anchor_parent_uid"] != -1


def test_no_parentless_normal_births(runtime_builder):
    runtime = runtime_builder(seed=502, width=10, height=10, agents=2, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    registry = runtime.registry
    cfg.RESPAWN.EXTINCTION_POLICY = "fail_run"

    dead_slots = registry.get_alive_indices().tolist()
    registry.mark_dead(dead_slots[0], runtime.grid)
    runtime.evolution.process_deaths([dead_slots[0]], runtime.ppo, death_tick=0)

    with pytest.raises(RuntimeError, match="Binary reproduction requires at least two live agents|Population dropped below two live agents"):
        runtime.engine.respawn_controller.step(1, registry, runtime.grid, runtime.data_logger)


def test_child_uid_is_new_and_freshly_owned(runtime_builder):
    runtime = runtime_builder(seed=503, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    old_uid = registry.get_uid_for_slot(slot)

    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=5)
    runtime.engine.respawn_controller.step(5, registry, runtime.grid, runtime.data_logger)

    new_uid = registry.get_uid_for_slot(slot)
    assert new_uid != old_uid
    assert new_uid not in runtime.ppo.optimizers_by_uid
    assert new_uid not in runtime.ppo.buffers_by_uid


def test_child_inherits_brain_from_brain_parent(runtime_builder):
    runtime = runtime_builder(seed=504, width=8, height=8, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = 0
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=4)
    runtime.engine.respawn_controller.step(4, registry, runtime.grid, runtime.data_logger)

    child_uid = registry.get_uid_for_slot(slot)
    brain_parent_uid = registry.get_parent_roles_for_uid(child_uid)["brain_parent_uid"]
    brain_parent_slot = registry.get_slot_for_uid(brain_parent_uid)
    child_state = registry.brains[slot].state_dict()
    parent_state = registry.brains[brain_parent_slot].state_dict()
    max_diff = max((child_state[k] - parent_state[k]).abs().max().item() for k in child_state)
    assert max_diff == 0.0


def test_child_trait_latent_inherits_from_trait_parent_then_mutates(runtime_builder):
    runtime = runtime_builder(seed=505, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = 0
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=6)
    runtime.engine.respawn_controller.step(6, registry, runtime.grid, runtime.data_logger)

    child_uid = registry.get_uid_for_slot(slot)
    roles = registry.get_parent_roles_for_uid(child_uid)
    trait_parent_latent = registry.get_trait_latent_for_uid(roles["trait_parent_uid"])
    child_latent = registry.get_trait_latent_for_uid(child_uid)
    assert set(child_latent.keys()) == set(trait_parent_latent.keys())


def test_trait_budget_transform_respects_clamps():
    latent = {"budget": 10.0, "z_hp": 4.0, "z_mass": 1.0, "z_vision": -1.0, "z_metab": -2.0}
    traits = trait_values_from_latent(latent)
    assert cfg.TRAITS.CLAMP.hp_max[0] <= traits["hp_max"] <= cfg.TRAITS.CLAMP.hp_max[1]
    assert cfg.TRAITS.CLAMP.mass[0] <= traits["mass"] <= cfg.TRAITS.CLAMP.mass[1]
    assert cfg.TRAITS.CLAMP.vision[0] <= traits["vision"] <= cfg.TRAITS.CLAMP.vision[1]
    assert cfg.TRAITS.CLAMP.metab[0] <= traits["metab"] <= cfg.TRAITS.CLAMP.metab[1]


def test_checkpoint_roundtrip_preserves_prompt5_reproduction_state(runtime_builder):
    runtime = runtime_builder(seed=506, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry
    slot = 0
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=8)
    runtime.engine.respawn_controller.step(8, registry, runtime.grid, runtime.data_logger)
    bundle = capture_runtime_checkpoint(runtime)

    restored = runtime_builder(seed=507, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    restore_runtime_checkpoint(restored, bundle)

    child_uid = runtime.registry.get_uid_for_slot(slot)
    assert restored.registry.get_parent_roles_for_uid(child_uid) == runtime.registry.get_parent_roles_for_uid(child_uid)
    assert restored.registry.get_trait_latent_for_uid(child_uid) == runtime.registry.get_trait_latent_for_uid(child_uid)
