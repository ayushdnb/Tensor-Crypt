import pytest

from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint
from tensor_crypt.config_bridge import cfg
from tensor_crypt.population.reproduction import ParentRoles, trait_values_from_latent


def _latest_birth_row_for_slot(runtime, slot: int) -> dict:
    rows = [row for row in runtime.data_logger._row_buffers["birth"] if int(row.get("child_slot", -1)) == int(slot)]
    assert rows, f"No birth rows recorded for slot {slot}"
    return rows[-1]


def _move_alive_agents_far_from_slot(runtime, dead_slot: int, *, min_radius: int = 2) -> None:
    registry = runtime.registry
    grid = runtime.grid
    center_x = int(registry.data[registry.X, dead_slot].item())
    center_y = int(registry.data[registry.Y, dead_slot].item())
    alive_slots = [int(slot) for slot in registry.get_alive_indices().tolist()]

    for slot_idx in alive_slots:
        x = int(registry.data[registry.X, slot_idx].item())
        y = int(registry.data[registry.Y, slot_idx].item())
        grid.clear_cell(x, y)

    candidates = []
    for y in range(grid.H - 2, 0, -1):
        for x in range(grid.W - 2, 0, -1):
            if max(abs(x - center_x), abs(y - center_y)) <= int(min_radius):
                continue
            candidates.append((x, y))
    assert len(candidates) >= len(alive_slots)

    for slot_idx, (x, y) in zip(alive_slots, candidates):
        registry.data[registry.X, slot_idx] = float(x)
        registry.data[registry.Y, slot_idx] = float(y)
        grid.set_cell(x, y, slot_idx, registry.data[registry.MASS, slot_idx].item())


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


def test_checkpoint_roundtrip_preserves_binary_reproduction_state(runtime_builder):
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


def test_crowding_overlay_can_block_birth_above_floor(runtime_builder):
    runtime = runtime_builder(seed=507, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.OVERLAYS.CROWDING.ENABLED = True
    cfg.RESPAWN.OVERLAYS.CROWDING.POLICY_WHEN_CROWDED = "block_birth"
    cfg.RESPAWN.OVERLAYS.CROWDING.LOCAL_RADIUS = max(runtime.grid.W, runtime.grid.H)
    cfg.RESPAWN.OVERLAYS.CROWDING.MAX_NEIGHBORS = 0

    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=3)
    runtime.engine.respawn_controller.step(3, registry, runtime.grid, runtime.data_logger)

    assert registry.get_uid_for_slot(slot) == -1
    row = _latest_birth_row_for_slot(runtime, slot)
    assert row["placement_failure_reason"] == "crowding_blocked"
    assert row.get("placement_crowding_policy_applied") == "block_birth"


def test_crowding_overlay_bypasses_blocking_below_floor_when_configured(runtime_builder):
    runtime = runtime_builder(seed=508, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.POPULATION_FLOOR = runtime.registry.get_num_alive()
    cfg.RESPAWN.OVERLAYS.CROWDING.ENABLED = True
    cfg.RESPAWN.OVERLAYS.CROWDING.POLICY_WHEN_CROWDED = "block_birth"
    cfg.RESPAWN.OVERLAYS.CROWDING.BELOW_FLOOR_POLICY = "bypass"
    cfg.RESPAWN.OVERLAYS.CROWDING.LOCAL_RADIUS = max(runtime.grid.W, runtime.grid.H)
    cfg.RESPAWN.OVERLAYS.CROWDING.MAX_NEIGHBORS = 0

    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=4)
    runtime.engine.respawn_controller.step(4, registry, runtime.grid, runtime.data_logger)

    assert registry.get_uid_for_slot(slot) != -1


def test_cooldown_overlay_rotates_brain_parent_when_alternative_exists(runtime_builder):
    runtime = runtime_builder(seed=509, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    cfg.RESPAWN.OVERLAYS.COOLDOWN.ENABLED = True
    cfg.RESPAWN.OVERLAYS.COOLDOWN.DURATION_TICKS = 16
    cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_BRAIN_PARENT = True
    cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_TRAIT_PARENT = False
    cfg.RESPAWN.OVERLAYS.COOLDOWN.APPLY_TO_ANCHOR_PARENT = False

    registry = runtime.registry
    alive_slots = [int(slot) for slot in registry.get_alive_indices().tolist()]
    for rank, slot_idx in enumerate(alive_slots):
        registry.fitness[slot_idx] = float(len(alive_slots) - rank)

    first_slot = alive_slots[-1]
    second_slot = alive_slots[-2]

    registry.mark_dead(first_slot, runtime.grid)
    runtime.evolution.process_deaths([first_slot], runtime.ppo, death_tick=5)
    runtime.engine.respawn_controller.step(5, registry, runtime.grid, runtime.data_logger)
    first_child_uid = registry.get_uid_for_slot(first_slot)
    first_roles = registry.get_parent_roles_for_uid(first_child_uid)

    registry.mark_dead(second_slot, runtime.grid)
    runtime.evolution.process_deaths([second_slot], runtime.ppo, death_tick=6)
    runtime.engine.respawn_controller.step(6, registry, runtime.grid, runtime.data_logger)
    second_child_uid = registry.get_uid_for_slot(second_slot)
    second_roles = registry.get_parent_roles_for_uid(second_child_uid)

    assert second_roles["brain_parent_uid"] != first_roles["brain_parent_uid"]


def test_cooldown_overlay_does_not_deadlock_recovery_below_floor(runtime_builder):
    runtime = runtime_builder(seed=510, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.POPULATION_FLOOR = runtime.registry.get_num_alive()
    cfg.RESPAWN.OVERLAYS.COOLDOWN.ENABLED = True
    cfg.RESPAWN.OVERLAYS.COOLDOWN.DURATION_TICKS = 32
    cfg.RESPAWN.OVERLAYS.COOLDOWN.BELOW_FLOOR_POLICY = "allow_best_available"

    registry = runtime.registry
    controller = runtime.engine.respawn_controller
    for uid in [registry.get_uid_for_slot(int(slot)) for slot in registry.get_alive_indices().tolist()]:
        controller.record_parent_role_usage(
            ParentRoles(uid, uid, uid, False),
            tick=1,
            floor_recovery=False,
        )

    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=2)
    runtime.engine.respawn_controller.step(2, registry, runtime.grid, runtime.data_logger)

    assert registry.get_uid_for_slot(slot) != -1
    row = _latest_birth_row_for_slot(runtime, slot)
    assert row.get("mutation_cooldown_relaxed_brain") is True


def test_local_parent_overlay_global_fallback_is_safe(runtime_builder):
    runtime = runtime_builder(seed=511, width=20, height=20, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.ENABLED = True
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.SELECTION_RADIUS = 1
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.FALLBACK_BEHAVIOR = "global"

    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=3)
    _move_alive_agents_far_from_slot(runtime, slot, min_radius=2)
    runtime.engine.respawn_controller.step(3, registry, runtime.grid, runtime.data_logger)

    assert registry.get_uid_for_slot(slot) != -1
    row = _latest_birth_row_for_slot(runtime, slot)
    assert row.get("mutation_local_parenting_used_global_fallback") is True
    assert row.get("mutation_local_parent_candidate_count") == 0


def test_local_parent_overlay_strict_can_block_only_that_birth_slot(runtime_builder):
    runtime = runtime_builder(seed=512, width=20, height=20, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.ENABLED = True
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.SELECTION_RADIUS = 1
    cfg.RESPAWN.OVERLAYS.LOCAL_PARENT.FALLBACK_BEHAVIOR = "strict"

    registry = runtime.registry
    slot = int(registry.get_alive_indices()[0].item())
    registry.mark_dead(slot, runtime.grid)
    runtime.evolution.process_deaths([slot], runtime.ppo, death_tick=4)
    _move_alive_agents_far_from_slot(runtime, slot, min_radius=2)
    runtime.engine.respawn_controller.step(4, registry, runtime.grid, runtime.data_logger)

    assert registry.get_uid_for_slot(slot) == -1
    row = _latest_birth_row_for_slot(runtime, slot)
    assert row["placement_failure_reason"] == "local_parent_candidates_empty"


def test_checkpoint_roundtrip_preserves_reproduction_overlay_runtime_state(runtime_builder):
    runtime = runtime_builder(seed=513, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    cfg.RESPAWN.OVERLAYS.COOLDOWN.ENABLED = True
    controller = runtime.engine.respawn_controller
    controller.toggle_doctrine_override("crowding")
    controller.toggle_doctrine_override("cooldown")
    controller.toggle_doctrine_override("local_parent")

    alive_uids = [runtime.registry.get_uid_for_slot(int(slot)) for slot in runtime.registry.get_alive_indices().tolist()[:2]]
    controller.record_parent_role_usage(
        ParentRoles(alive_uids[0], alive_uids[1], alive_uids[0], False),
        tick=9,
        floor_recovery=False,
    )

    bundle = capture_runtime_checkpoint(runtime)
    restored = runtime_builder(seed=514, width=12, height=12, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    restore_runtime_checkpoint(restored, bundle)

    assert restored.engine.respawn_controller.serialize_runtime_state() == runtime.engine.respawn_controller.serialize_runtime_state()
