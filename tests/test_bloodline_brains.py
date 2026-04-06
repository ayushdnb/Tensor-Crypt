import torch

from tensor_crypt.agents.brain import Brain, create_brain, get_bloodline_families
from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.viewer.colors import get_bloodline_base_color


def _make_canonical_obs(batch_size: int = 2):
    return {
        "canonical_rays": torch.zeros(batch_size, cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.CANONICAL_RAY_FEATURES),
        "canonical_self": torch.zeros(batch_size, cfg.PERCEPT.CANONICAL_SELF_FEATURES),
        "canonical_context": torch.zeros(batch_size, cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES),
    }


def test_all_bloodline_families_instantiate_and_forward():
    cfg.SIM.DEVICE = "cpu"
    obs = _make_canonical_obs(batch_size=3)

    for family_id in get_bloodline_families():
        brain = create_brain(family_id).to("cpu")
        logits, value = brain(obs)
        assert brain.family_id == family_id
        assert logits.shape == (3, cfg.BRAIN.ACTION_DIM)
        assert value.shape == (3, cfg.BRAIN.VALUE_DIM)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()


def test_within_family_topology_is_invariant_and_across_families_can_differ():
    signatures = {}
    for family_id in get_bloodline_families():
        brain_a = create_brain(family_id)
        brain_b = create_brain(family_id)
        sig_a = brain_a.get_topology_signature()
        sig_b = brain_b.get_topology_signature()
        assert sig_a == sig_b
        signatures[family_id] = sig_a

    unique_signatures = {tuple(signature) for signature in signatures.values()}
    assert len(unique_signatures) >= 2


def test_registry_assigns_every_active_uid_a_bloodline(runtime_builder):
    runtime = runtime_builder(seed=301, width=14, height=14, agents=10, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)

    for slot_idx in runtime.registry.get_alive_indices().tolist():
        uid = runtime.registry.get_uid_for_slot(slot_idx)
        family_id = runtime.registry.get_family_for_slot(slot_idx)
        brain = runtime.registry.brains[slot_idx]

        assert uid in runtime.registry.uid_family
        assert family_id in cfg.BRAIN.FAMILY_ORDER
        assert runtime.registry.uid_family[uid] == family_id
        assert brain.family_id == family_id


def test_respawn_preserves_parent_bloodline(runtime_builder):
    runtime = runtime_builder(seed=302, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry

    slot = int(registry.get_alive_indices()[0].item())
    old_uid = registry.get_uid_for_slot(slot)

    registry.data[registry.HP, slot] = 0.0
    deaths = runtime.physics.process_deaths()
    runtime.evolution.process_deaths(deaths, runtime.ppo, death_tick=7)
    runtime.engine.respawn_controller.step(7, registry, runtime.grid, runtime.data_logger)

    child_uid = registry.get_uid_for_slot(slot)
    parent_uid = registry.get_parent_uid_for_slot(slot)
    parent_family = registry.get_family_for_uid(parent_uid)
    child_family = registry.get_family_for_slot(slot)
    assert child_uid != old_uid
    assert parent_uid != -1
    assert child_family == parent_family
    assert registry.brains[slot].family_id == parent_family

def test_checkpoint_restore_recreates_correct_bloodline_families(runtime_builder):
    runtime = runtime_builder(seed=303, width=14, height=14, agents=8, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    bundle = capture_runtime_checkpoint(runtime)

    restored = runtime_builder(seed=404, width=14, height=14, agents=8, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    restore_runtime_checkpoint(restored, bundle)

    assert restored.registry.uid_family == runtime.registry.uid_family
    for uid, slot_idx in restored.registry.active_uid_to_slot.items():
        expected_family = runtime.registry.get_family_for_uid(uid)
        assert restored.registry.get_family_for_slot(slot_idx) == expected_family
        assert restored.registry.brains[slot_idx].family_id == expected_family


def test_viewer_color_mapping_and_inspector_family_label(runtime_builder):
    runtime = runtime_builder(seed=304, width=12, height=12, agents=5, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer
    state_data = viewer._prepare_state_data()
    slot_id = runtime.registry.get_alive_indices().tolist()[0]
    family_id = runtime.registry.get_family_for_slot(slot_id)

    base_color = get_bloodline_base_color(family_id)
    lines = viewer.side_panel._agent_detail_lines(slot_id)
    joined = "\n".join(line for line, _ in lines)

    assert base_color == tuple(cfg.BRAIN.FAMILY_COLORS[family_id])
    assert f"Bloodline: {family_id}" in joined