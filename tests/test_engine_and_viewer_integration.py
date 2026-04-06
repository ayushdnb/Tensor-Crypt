import pandas as pd
import pygame
import torch


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