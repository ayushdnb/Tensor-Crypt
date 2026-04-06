from tensor_crypt.config_bridge import cfg
from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.world.spatial_grid import Grid


def test_spawn_initial_population_has_no_overlap_and_passes_invariants():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 12
    cfg.GRID.H = 12
    cfg.AGENTS.N = 8

    grid = Grid()
    registry = Registry()
    registry.spawn_initial_population(grid)
    registry.check_invariants(grid)

    assert registry.get_num_alive() == 8
    positions = {
        (int(registry.data[registry.X, idx].item()), int(registry.data[registry.Y, idx].item()))
        for idx in registry.get_alive_indices().tolist()
    }
    assert len(positions) == 8


def test_respawn_assigns_birth_tick_new_identity_and_parent_identity(runtime_builder):
    runtime = runtime_builder(seed=21, width=12, height=12, agents=6, walls=0, hzones=0, policy_noise=0.0)
    registry = runtime.registry

    slot = int(registry.get_alive_indices()[0].item())
    old_uid = registry.get_uid_for_slot(slot)
    registry.data[registry.HP, slot] = 0.0
    deaths = runtime.physics.process_deaths()
    runtime.evolution.process_deaths(deaths, runtime.ppo, death_tick=7)
    runtime.engine.respawn_controller.step(7, registry, runtime.grid, runtime.data_logger)

    assert registry.data[registry.ALIVE, slot].item() == 1.0
    assert int(registry.data[registry.TICK_BORN, slot].item()) == 7
    assert registry.get_uid_for_slot(slot) != old_uid
    parent_uid = registry.get_parent_uid_for_slot(slot)
    live_uids = {registry.get_uid_for_slot(idx) for idx in registry.get_alive_indices().tolist()}
    assert parent_uid in live_uids


def test_respawn_inherits_parent_brain_for_slot_without_existing_brain(runtime_builder):
    runtime = runtime_builder(seed=22, width=8, height=8, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1, policy_noise=0.0)
    registry = runtime.registry

    registry.mark_dead(5, runtime.grid)
    runtime.evolution.process_deaths([5], runtime.ppo, death_tick=3)
    registry.brains[5] = None
    runtime.engine.respawn_controller.step(3, registry, runtime.grid, runtime.data_logger)

    parent_uid = registry.get_parent_uid_for_slot(5)
    parent_slot = registry.get_slot_for_uid(parent_uid)

    child_state = registry.brains[5].state_dict()
    parent_state = registry.brains[parent_slot].state_dict()
    max_diff = max((child_state[key] - parent_state[key]).abs().max().item() for key in child_state)

    assert registry.data[registry.ALIVE, 5].item() == 1.0
    assert max_diff == 0.0


def test_extinction_path_respawns_from_default_traits(runtime_builder):
    runtime = runtime_builder(seed=23, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    registry = runtime.registry

    dead_slots = registry.get_alive_indices().tolist()
    for idx in dead_slots:
        registry.mark_dead(idx, runtime.grid)
    runtime.evolution.process_deaths(dead_slots, runtime.ppo, death_tick=0)

    runtime.engine.step()

    alive = registry.get_alive_indices().tolist()
    assert len(alive) == cfg.RESPAWN.MAX_SPAWNS_PER_CYCLE
    for idx in alive:
        assert int(registry.data[registry.TICK_BORN, idx].item()) == 1
        assert registry.get_parent_uid_for_slot(idx) == -1
        assert registry.data[registry.MASS, idx].item() == cfg.TRAITS.INIT.mass
        assert registry.data[registry.HP_MAX, idx].item() == cfg.TRAITS.INIT.hp_max
