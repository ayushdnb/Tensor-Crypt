import pandas as pd
import pygame
import torch

from tensor_crypt.checkpointing.runtime_checkpoint import load_runtime_checkpoint
from tensor_crypt.simulation.engine import SAVE_REASON_MANUAL_OPERATOR


def test_engine_end_to_end_writes_artifacts_and_keeps_invariants(runtime_builder):
    runtime = runtime_builder(seed=31, width=16, height=16, agents=8, walls=2, hzones=2, update_every=4)

    for _ in range(6):
        runtime.engine.step()

    runtime.registry.check_invariants(runtime.grid)
    runtime.data_logger.close()

    summary = pd.read_parquet(runtime.data_logger.tick_summary_path)
    assert runtime.engine.tick == 6
    assert len(summary) == 6
    assert runtime.data_logger.hdf_path.exists()
    assert runtime.data_logger.tick_summary_path.exists()



def test_seeded_runtime_is_deterministic(runtime_builder):
    runtime_a = runtime_builder(seed=77, width=14, height=14, agents=6, walls=1, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(5):
        runtime_a.engine.step()
    data_a = runtime_a.registry.data.cpu().clone()
    grid_a = runtime_a.grid.grid.cpu().clone()

    runtime_b = runtime_builder(seed=77, width=14, height=14, agents=6, walls=1, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(5):
        runtime_b.engine.step()
    data_b = runtime_b.registry.data.cpu().clone()
    grid_b = runtime_b.grid.grid.cpu().clone()

    assert torch.equal(data_a, data_b)
    assert torch.equal(grid_a, grid_b)



def test_viewer_draw_smoke_handles_dead_selection_and_hzone_selection(runtime_builder):
    runtime = runtime_builder(seed=41, width=12, height=12, agents=4, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    selected_slot = runtime.registry.get_alive_indices().tolist()[0]
    runtime.registry.mark_dead(selected_slot, runtime.grid)
    viewer.selected_slot_id = selected_slot
    viewer.last_selected_uid = int(runtime.registry.data[runtime.registry.AGENT_ID, selected_slot].item())
    state_data = viewer._prepare_state_data()

    surface = pygame.Surface((viewer.Wpix, viewer.Hpix))
    viewer.world_renderer.draw(surface, state_data)
    viewer.hud_panel.draw(surface, state_data)
    viewer.side_panel.draw(surface, state_data)

    viewer.selected_slot_id = None
    viewer.selected_hzone_id = runtime.grid.hzones[0]["id"]
    viewer.world_renderer.draw(surface, state_data)
    viewer.side_panel.draw(surface, state_data)



def test_engine_reward_clamps_negative_health_before_squaring(runtime_builder):
    runtime = runtime_builder(seed=43, width=10, height=10, agents=2, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    slot = int(runtime.registry.get_alive_indices()[0].item())
    captured = {}

    original_store = runtime.ppo.store_transition_for_slot

    def wrapped_store(registry, slot_idx, obs, action, log_prob, reward, value, done):
        if slot_idx == slot:
            captured["reward"] = float(reward.item())
        return original_store(registry, slot_idx, obs, action, log_prob, reward, value, done)

    runtime.ppo.store_transition_for_slot = wrapped_store
    runtime.physics.step = lambda actions: {"wall_collisions": 0, "rams": 0, "contests": 0}

    def force_negative_hp():
        runtime.registry.data[runtime.registry.HP, slot] = -5.0

    runtime.physics.apply_environment_effects = force_negative_hp
    runtime.engine.step()

    assert captured["reward"] == 0.0



def test_viewer_resize_event_updates_camera_world_rect(runtime_builder):
    runtime = runtime_builder(seed=42, width=12, height=12, agents=4, walls=0, hzones=1, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, {"w": 1400, "h": 900}))
    running, advance_tick = viewer.input_handler.handle()
    world_rect = viewer.layout.world_rect()

    assert running is True
    assert advance_tick is False
    assert viewer.cam.screen_width == world_rect.width
    assert viewer.cam.screen_height == world_rect.height


def test_escape_requests_graceful_shutdown(runtime_builder):
    runtime = runtime_builder(seed=44, width=12, height=12, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    viewer = runtime.viewer

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0}))
    running, advance_tick = viewer.input_handler.handle()

    assert running is False
    assert advance_tick is False
    assert viewer.shutdown_requested is True
    assert viewer.shutdown_reason == "viewer_escape"
    assert runtime.engine.is_graceful_shutdown_requested() is True
    assert runtime.engine.graceful_shutdown_reason == "viewer_escape"


def test_forced_checkpoint_stages_active_ppo_bootstrap(runtime_builder):
    runtime = runtime_builder(seed=45, width=12, height=12, agents=2, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    runtime.engine.step()

    uid = int(runtime.registry.get_uid_for_slot(int(runtime.registry.get_alive_indices()[0].item())))
    checkpoint_path = runtime.engine.publish_runtime_checkpoint(SAVE_REASON_MANUAL_OPERATOR, force=True)
    bundle = load_runtime_checkpoint(checkpoint_path)
    buffer_payload = bundle["ppo_state"]["buffer_state_by_uid"][uid]

    assert buffer_payload["bootstrap_obs"] is not None
    assert float(buffer_payload["bootstrap_done"].item()) == 0.0
    assert buffer_payload["finalization_kind"] == "checkpoint_manual_operator"
