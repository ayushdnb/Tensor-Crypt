import pandas as pd
import torch



def test_poison_zone_death_records_zone_id(runtime_builder):
    runtime = runtime_builder(seed=501, width=10, height=10, agents=2, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    slot = int(runtime.registry.get_alive_indices()[0].item())
    x = int(runtime.registry.data[runtime.registry.X, slot].item())
    y = int(runtime.registry.data[runtime.registry.Y, slot].item())

    runtime.grid.add_hzone(x, y, x, y, -5.0)
    zone_id = runtime.grid.find_hzone_at(x, y)
    runtime.registry.data[runtime.registry.HP, slot] = 1.0

    def idle_sample(obs, alive_indices):
        batch = len(alive_indices)
        return (
            torch.zeros(batch, 9),
            torch.zeros(batch, 1),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch),
        )

    runtime.engine._sample_actions = idle_sample
    runtime.engine.step()
    runtime.data_logger.close(runtime.registry)

    death_df = pd.read_parquet(runtime.data_logger.death_ledger_path)
    assert death_df.iloc[0]["death_reason"] == "poison_zone"
    assert int(death_df.iloc[0]["zone_id"]) == zone_id
