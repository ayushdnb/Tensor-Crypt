from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..config_bridge import cfg


class DataLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.hdf_path = self.run_dir / "simulation_data.hdf5"
        self.h5_file = h5py.File(str(self.hdf_path), "w")
        self.h5_snapshots = self.h5_file.create_group("agent_snapshots")
        self.h5_heatmaps = self.h5_file.create_group("heatmaps")
        self.h5_identity = self.h5_file.create_group("agent_identity")

        self.genealogy_path = self.run_dir / "genealogy.parquet"
        self.collisions_path = self.run_dir / "collisions.parquet"
        self.ppo_path = self.run_dir / "ppo_events.parquet"
        self.tick_summary_path = self.run_dir / "tick_summary.parquet"

        self.genealogy_writer: Optional[pq.ParquetWriter] = None
        self.collisions_writer: Optional[pq.ParquetWriter] = None
        self.ppo_writer: Optional[pq.ParquetWriter] = None
        self.tick_summary_writer: Optional[pq.ParquetWriter] = None

        self.genealogy_schema: Optional[pa.Schema] = None
        self.collisions_schema: Optional[pa.Schema] = None
        self.ppo_schema: Optional[pa.Schema] = None
        self.tick_summary_schema: Optional[pa.Schema] = None

    def _write_parquet(self, df: pd.DataFrame, path: Path, writer_attr: str, schema_attr: str):
        writer = getattr(self, writer_attr)
        schema = getattr(self, schema_attr)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        if writer is None:
            setattr(self, schema_attr, table.schema)
            writer = pq.ParquetWriter(str(path), table.schema, compression="gzip")
            setattr(self, writer_attr, writer)
        writer.write_table(table)

    def close(self):
        self.h5_file.close()
        if self.genealogy_writer:
            self.genealogy_writer.close()
        if self.collisions_writer:
            self.collisions_writer.close()
        if self.ppo_writer:
            self.ppo_writer.close()
        if self.tick_summary_writer:
            self.tick_summary_writer.close()

    def _schema_versions(self) -> dict:
        return {
            "identity": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
            "observation": cfg.SCHEMA.OBS_SCHEMA_VERSION,
            "checkpoint": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
            "reproduction": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
            "telemetry": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
            "logging": cfg.SCHEMA.LOGGING_SCHEMA_VERSION,
        }

    def log_agent_snapshot(self, tick: int, registry):
        self.h5_snapshots.create_dataset(f"tick_{tick}", data=registry.data.cpu().numpy(), compression="gzip")
        self.h5_identity.create_dataset(f"slot_uid_tick_{tick}", data=registry.slot_uid.cpu().numpy(), compression="gzip")
        self.h5_identity.create_dataset(f"slot_parent_uid_tick_{tick}", data=registry.slot_parent_uid.cpu().numpy(), compression="gzip")

    def log_heatmap_snapshot(self, tick: int, grid):
        self.h5_heatmaps.create_dataset(f"density_tick_{tick}", data=grid.grid[2].cpu().numpy(), compression="gzip")
        self.h5_heatmaps.create_dataset(f"mass_tick_{tick}", data=grid.grid[3].cpu().numpy(), compression="gzip")

    def log_brains(self, tick: int, registry):
        by_uid = {}
        uid_to_slot = {}
        family_by_uid = {}
        for uid, slot_idx in sorted(registry.active_uid_to_slot.items()):
            brain = registry.brains[slot_idx]
            if brain is None:
                continue
            by_uid[uid] = brain.state_dict()
            uid_to_slot[uid] = slot_idx
            family_by_uid[uid] = registry.get_family_for_uid(uid)
        payload = {"by_uid": by_uid, "uid_to_slot": uid_to_slot, "family_by_uid": family_by_uid, "tick": tick, "schema_versions": self._schema_versions()}
        torch.save(payload, str(self.run_dir / "brains" / f"brains_tick_{tick}.pt"))

    def log_spawn_event(
        self,
        *,
        tick: int,
        child_slot: int,
        brain_parent_slot: int,
        trait_parent_slot: int,
        anchor_parent_slot: int,
        child_uid: int,
        brain_parent_uid: int,
        trait_parent_uid: int,
        anchor_parent_uid: int,
        child_family: str | None,
        brain_parent_family: str | None,
        trait_parent_family: str | None,
        traits: dict,
        trait_latent: dict,
        mutation_flags: dict,
        placement: dict,
        floor_recovery: bool,
    ):
        payload = {
            "tick": tick,
            "child_slot": child_slot,
            "parent_slot": brain_parent_slot,
            "brain_parent_slot": brain_parent_slot,
            "trait_parent_slot": trait_parent_slot,
            "anchor_parent_slot": anchor_parent_slot,
            "child_uid": child_uid,
            "parent_uid": brain_parent_uid,
            "brain_parent_uid": brain_parent_uid,
            "trait_parent_uid": trait_parent_uid,
            "anchor_parent_uid": anchor_parent_uid,
            "child_family": child_family,
            "brain_parent_family": brain_parent_family,
            "trait_parent_family": trait_parent_family,
            "floor_recovery": floor_recovery,
            "identity_schema_version": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
            "telemetry_schema_version": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
            "reproduction_schema_version": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
            **{f"trait_{k}": v for k, v in trait_latent.items()},
            **{f"value_{k}": v for k, v in traits.items()},
            **{f"mutation_{k}": v for k, v in mutation_flags.items()},
            **{f"placement_{k}": v for k, v in placement.items()},
        }
        if cfg.MIGRATION.LOG_LEGACY_SLOT_FIELDS:
            payload["parent_idx"] = brain_parent_slot
            payload["child_idx"] = child_slot
        df = pd.DataFrame([payload])
        self._write_parquet(df, self.genealogy_path, "genealogy_writer", "genealogy_schema")

    def log_physics_events(self, tick: int, collision_log: list[dict]):
        if not collision_log:
            return
        df = pd.DataFrame(collision_log)
        df["tick"] = tick
        self._write_parquet(df, self.collisions_path, "collisions_writer", "collisions_schema")

    def log_ppo_update(self, tick: int, ppo_stats_list: list[dict]):
        if not ppo_stats_list:
            return
        df = pd.DataFrame(ppo_stats_list)
        df["tick"] = tick
        self._write_parquet(df, self.ppo_path, "ppo_writer", "ppo_schema")

    def log_tick_summary(self, tick: int, registry, physics_stats: dict):
        alive_mask = registry.get_alive_mask()
        num_alive = int(alive_mask.sum().item())
        if num_alive == 0:
            stats = {"tick": tick, "num_alive": 0}
        else:
            stats = {
                "tick": tick,
                "num_alive": num_alive,
                "avg_hp": registry.data[registry.HP, alive_mask].mean().item(),
                "avg_mass": registry.data[registry.MASS, alive_mask].mean().item(),
                "avg_vision": registry.data[registry.VISION, alive_mask].mean().item(),
                "total_hp_gained": registry.data[registry.HP_GAINED, alive_mask].sum().item(),
                "total_physics_dmg": registry.data[registry.HP_LOST_PHYSICS, alive_mask].sum().item(),
                **physics_stats,
            }
        self._write_parquet(pd.DataFrame([stats]), self.tick_summary_path, "tick_summary_writer", "tick_summary_schema")
