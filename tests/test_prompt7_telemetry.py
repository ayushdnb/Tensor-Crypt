import json

import pandas as pd

from tensor_crypt.population.reproduction import default_trait_latent, trait_values_from_latent


def test_prompt7_initial_population_bootstraps_birth_and_open_life_ledgers(runtime_builder):
    runtime = runtime_builder(seed=11, agents=6, walls=0, hzones=0)
    runtime.data_logger.close(runtime.registry)

    birth_df = pd.read_parquet(runtime.data_logger.birth_ledger_path)
    life_df = pd.read_parquet(runtime.data_logger.life_ledger_path)

    assert len(birth_df) == 6
    assert len(life_df) == 6
    assert {"child_uid", "birth_tick", "brain_parent_uid", "trait_parent_uid", "anchor_parent_uid", "child_family"}.issubset(birth_df.columns)
    assert {"agent_uid", "brain_family", "birth_tick", "trait_budget", "alloc_hp", "alloc_mass", "alloc_vision", "alloc_metab"}.issubset(life_df.columns)
    assert life_df["death_tick"].isna().all()

def test_prompt7_death_finalization_emits_taxonomy_and_life_row(runtime_builder):
    runtime = runtime_builder(seed=12, agents=4, walls=0, hzones=0)
    slot_idx = int(runtime.registry.get_alive_indices()[0].item())

    runtime.physics._resolved_death_context_by_slot[slot_idx] = {
        "death_reason": "wall_collision",
        "catastrophe_id": None,
        "zone_id": None,
        "killing_agent_uid": None,
    }
    runtime.registry.data[runtime.registry.HP, slot_idx] = 0.0
    runtime.registry.mark_dead(slot_idx, runtime.grid)

    runtime.data_logger.finalize_death(
        tick=runtime.engine.tick,
        slot_idx=slot_idx,
        registry=runtime.registry,
        ppo=runtime.ppo,
        death_context=runtime.physics.consume_death_context(slot_idx),
    )
    runtime.evolution.process_deaths([slot_idx], runtime.ppo, death_tick=runtime.engine.tick)
    runtime.data_logger.close(runtime.registry)

    death_df = pd.read_parquet(runtime.data_logger.death_ledger_path)
    life_df = pd.read_parquet(runtime.data_logger.life_ledger_path)

    assert "wall_collision" in set(death_df["death_reason"].tolist())
    life_row = life_df.loc[life_df["death_reason"] == "wall_collision"].iloc[0]
    assert int(life_row["death_slot"]) == slot_idx
    assert int(life_row["age_at_death"]) >= 0

def test_prompt7_lineage_export_contains_parent_role_edges(runtime_builder):
    runtime = runtime_builder(seed=13, agents=6, walls=0, hzones=0)
    runtime.engine.step()
    runtime.data_logger.close(runtime.registry)

    with open(runtime.data_logger.lineage_path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)

    assert graph["node_count"] >= runtime.registry.next_agent_uid
    assert {"nodes", "edges"} <= set(graph.keys())

    edge_types = {edge["edge_type"] for edge in graph["edges"]}
    assert {"brain_parent", "trait_parent", "anchor_parent"} & edge_types or len(graph["edges"]) == 0

def test_prompt7_tick_and_family_summaries_include_prompt7_fields(runtime_builder):
    runtime = runtime_builder(seed=14, agents=6, walls=0, hzones=0)
    runtime.engine.step()
    runtime.data_logger.close(runtime.registry)

    tick_df = pd.read_parquet(runtime.data_logger.tick_summary_path)
    family_df = pd.read_parquet(runtime.data_logger.family_summary_path)

    assert {"live_population", "mean_hp_ratio", "mean_mass", "mean_vision", "mean_metabolism", "births_this_tick", "deaths_this_tick", "reproduction_disabled_flag", "floor_recovery_active_flag"}.issubset(tick_df.columns)
    assert {"family_id", "active_count", "births_this_tick", "deaths_this_tick", "mean_hp_ratio", "mean_lineage_depth", "ppo_update_count_cumulative"}.issubset(family_df.columns)

def test_prompt7_viewer_trait_surface_matches_trait_latent(runtime_builder):
    runtime = runtime_builder(seed=15, agents=4, walls=0, hzones=0)
    uid = next(iter(runtime.registry.active_uid_to_slot.keys()))
    latent = runtime.registry.get_trait_latent_for_uid(uid)
    mapped = trait_values_from_latent(latent)
    baseline = default_trait_latent()

    assert set(baseline.keys()) == {"budget", "z_hp", "z_mass", "z_vision", "z_metab"}
    assert {"budget", "alloc_hp", "alloc_mass", "alloc_vision", "alloc_metab"} <= set(mapped.keys())
