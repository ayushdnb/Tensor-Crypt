import pytest
import torch

from tensor_crypt.agents.state_registry import Registry
from tensor_crypt.config_bridge import cfg
from tensor_crypt.world.observation_schema import build_observation_bundle, normalize_from_bounds
from tensor_crypt.world.spatial_grid import Grid


def _spawn(registry, grid, idx, x, y, *, mass, hp, hp_max, vision, metab):
    registry.spawn_agent(
        idx,
        x,
        y,
        -1,
        grid,
        traits={"mass": mass, "vision": vision, "hp_max": hp_max, "metab": metab},
    )
    registry.data[registry.HP, idx] = hp
    registry.data[registry.HP_MAX, idx] = hp_max



def test_build_observation_bundle_uses_explicit_global_alive_indices_for_context():
    cfg.SIM.DEVICE = "cpu"
    cfg.GRID.W = 12
    cfg.GRID.H = 12
    cfg.AGENTS.N = 4
    cfg.PERCEPT.NUM_RAYS = 8
    cfg.PERCEPT.RETURN_CANONICAL_OBSERVATIONS = True

    grid = Grid()
    registry = Registry()
    _spawn(registry, grid, 0, 3, 3, mass=2.0, hp=5.0, hp_max=10.0, vision=6.0, metab=0.10)
    _spawn(registry, grid, 1, 8, 8, mass=6.0, hp=10.0, hp_max=20.0, vision=10.0, metab=0.20)

    full_alive = registry.get_alive_indices()
    subset_alive = torch.tensor([0], dtype=torch.long)
    canonical_rays = torch.zeros(1, cfg.PERCEPT.NUM_RAYS, cfg.PERCEPT.CANONICAL_RAY_FEATURES, dtype=torch.float32)

    default_obs = build_observation_bundle(
        registry=registry,
        grid=grid,
        alive_indices=subset_alive,
        canonical_rays=canonical_rays,
    )
    explicit_obs = build_observation_bundle(
        registry=registry,
        grid=grid,
        alive_indices=subset_alive,
        canonical_rays=canonical_rays,
        global_alive_indices=full_alive,
    )

    global_hp_ratio = torch.clamp(
        registry.data[registry.HP, full_alive] / registry.data[registry.HP_MAX, full_alive].clamp_min(1e-6),
        0.0,
        1.0,
    )
    global_mass_norm = normalize_from_bounds(
        registry.data[registry.MASS, full_alive],
        cfg.TRAITS.CLAMP.mass[0],
        cfg.TRAITS.CLAMP.mass[1],
    )

    assert default_obs["canonical_context"][0, 0].item() == pytest.approx(1.0 / cfg.AGENTS.N)
    assert explicit_obs["canonical_context"][0, 0].item() == pytest.approx(len(full_alive) / cfg.AGENTS.N)
    assert explicit_obs["canonical_context"][0, 1].item() == pytest.approx(float(global_mass_norm.mean().item()))
    assert explicit_obs["canonical_context"][0, 2].item() == pytest.approx(float(global_hp_ratio.mean().item()))


@pytest.mark.parametrize("reuse_action_buffer", [True, False])
def test_engine_sparse_action_buffer_respects_reuse_knob_without_stale_actions(runtime_builder, reuse_action_buffer):
    cfg.SIM.REUSE_ACTION_BUFFER = reuse_action_buffer

    runtime = runtime_builder(seed=61 if reuse_action_buffer else 62, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    alive_slots = [int(slot.item()) for slot in runtime.registry.get_alive_indices()]
    live_slot = alive_slots[0]
    for dead_slot in alive_slots[1:]:
        runtime.registry.mark_dead(dead_slot, runtime.grid)
        runtime.evolution.process_deaths([dead_slot], runtime.ppo, death_tick=runtime.engine.tick)

    runtime.engine._actions_sparse.fill_(7)
    reused_buffer_ptr = runtime.engine._actions_sparse.data_ptr()

    logits = torch.zeros(1, cfg.BRAIN.ACTION_DIM, device=cfg.SIM.DEVICE)
    values = torch.zeros(1, cfg.BRAIN.VALUE_DIM, device=cfg.SIM.DEVICE)
    sampled_actions = torch.tensor([3], device=cfg.SIM.DEVICE, dtype=torch.long)
    log_probs = torch.zeros(1, device=cfg.SIM.DEVICE)
    runtime.engine._sample_actions = lambda obs, alive_indices: (logits, values, sampled_actions, log_probs)

    captured = {}

    def fake_step(actions_sparse):
        captured["actions"] = actions_sparse.detach().cpu().clone()
        captured["ptr"] = actions_sparse.data_ptr()
        return {"wall_collisions": 0, "rams": 0, "contests": 0}

    runtime.physics.step = fake_step
    runtime.physics.apply_environment_effects = lambda: None
    runtime.physics.process_deaths = lambda: []

    runtime.engine.step()

    assert int(captured["actions"][live_slot].item()) == 3
    assert torch.count_nonzero(captured["actions"]).item() == 1
    if reuse_action_buffer:
        assert captured["ptr"] == reused_buffer_ptr
    else:
        assert captured["ptr"] != reused_buffer_ptr



def test_log_tick_summary_skip_knob_avoids_non_emit_work(runtime_builder):
    cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS = 4
    cfg.TELEMETRY.FAMILY_SUMMARY_EVERY_TICKS = 5
    cfg.TELEMETRY.SUMMARY_SKIP_NON_EMIT_WORK = True

    runtime = runtime_builder(seed=63, agents=4, walls=0, hzones=0)

    def explode():
        raise AssertionError("summary aggregation should be skipped")

    runtime.registry.get_alive_indices = explode
    stats = {"wall_collisions": 0, "rams": 0, "contests": 0}
    buffered_before = runtime.data_logger.get_buffered_row_count()

    runtime.data_logger.log_tick_summary(1, runtime.registry, stats, ppo=runtime.ppo)
    assert runtime.data_logger.get_buffered_row_count() == buffered_before

    cfg.TELEMETRY.SUMMARY_SKIP_NON_EMIT_WORK = False
    with pytest.raises(AssertionError, match="summary aggregation should be skipped"):
        runtime.data_logger.log_tick_summary(1, runtime.registry, stats, ppo=runtime.ppo)
