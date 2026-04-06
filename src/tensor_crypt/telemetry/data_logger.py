"""Logging and artifact persistence for Tensor Crypt."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..config_bridge import cfg
from ..population.reproduction import default_trait_latent, trait_values_from_latent
from .lineage_export import export_lineage_json


class DataLogger:
    """
    Research-grade run logger.

    Prompt 7 adds durable permanent ledgers without changing the simulation
    semantics:
    - births remain logged at spawn time
    - deaths are finalized before UID retirement
    - life rows are emitted exactly once per UID at death or close
    - lineage export is derived from the canonical UID/parent-role substrate
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "brains").mkdir(parents=True, exist_ok=True)

        self.hdf_path = self.run_dir / "simulation_data.hdf5"
        self.h5_file = h5py.File(str(self.hdf_path), "w")
        self.h5_snapshots = self.h5_file.create_group("agent_snapshots")
        self.h5_heatmaps = self.h5_file.create_group("heatmaps")
        self.h5_identity = self.h5_file.create_group("agent_identity")

        self.birth_ledger_path = self.run_dir / "birth_ledger.parquet"
        self.genealogy_path = self.run_dir / "genealogy.parquet"  # backward-compatible alias surface
        self.life_ledger_path = self.run_dir / "life_ledger.parquet"
        self.death_ledger_path = self.run_dir / "death_ledger.parquet"
        self.collisions_path = self.run_dir / "collisions.parquet"
        self.ppo_path = self.run_dir / "ppo_events.parquet"
        self.tick_summary_path = self.run_dir / "tick_summary.parquet"
        self.family_summary_path = self.run_dir / "family_summary.parquet"
        self.catastrophes_path = self.run_dir / "catastrophes.parquet"
        self.lineage_path = self.run_dir / "lineage_graph.json"

        self.birth_writer: Optional[pq.ParquetWriter] = None
        self.genealogy_writer: Optional[pq.ParquetWriter] = None
        self.life_writer: Optional[pq.ParquetWriter] = None
        self.death_writer: Optional[pq.ParquetWriter] = None
        self.collisions_writer: Optional[pq.ParquetWriter] = None
        self.ppo_writer: Optional[pq.ParquetWriter] = None
        self.tick_summary_writer: Optional[pq.ParquetWriter] = None
        self.family_summary_writer: Optional[pq.ParquetWriter] = None
        self.catastrophes_writer: Optional[pq.ParquetWriter] = None

        self.birth_schema: Optional[pa.Schema] = None
        self.genealogy_schema: Optional[pa.Schema] = None
        self.life_schema: Optional[pa.Schema] = None
        self.death_schema: Optional[pa.Schema] = pa.schema(
            [
                pa.field("agent_uid", pa.int64()),
                pa.field("tick", pa.int64()),
                pa.field("death_reason", pa.string()),
                pa.field("catastrophe_id", pa.int64()),
                pa.field("killing_agent_uid", pa.int64()),
                pa.field("zone_id", pa.int64()),
                pa.field("position_x", pa.int64()),
                pa.field("position_y", pa.int64()),
                pa.field("final_hp", pa.float64()),
                pa.field("family", pa.string()),
                pa.field("lineage_depth", pa.int64()),
                pa.field("telemetry_schema_version", pa.int64()),
            ]
        )
        self.collisions_schema: Optional[pa.Schema] = None
        self.ppo_schema: Optional[pa.Schema] = None
        self.tick_summary_schema: Optional[pa.Schema] = None
        self.family_summary_schema: Optional[pa.Schema] = None
        self.catastrophes_schema: Optional[pa.Schema] = None

        self.open_lives_by_uid: dict[int, dict] = {}
        self.finalized_lives_by_uid: dict[int, dict] = {}
        self.birth_counts_by_tick: dict[int, int] = {}
        self.death_counts_by_tick: dict[int, int] = {}
        self.birth_counts_by_family_and_tick: dict[tuple[int, str], int] = {}
        self.death_counts_by_family_and_tick: dict[tuple[int, str], int] = {}
        self.exposed_catastrophes_by_uid: dict[int, set[int]] = {}
        self.survived_catastrophes_by_uid: dict[int, set[int]] = {}
        self.just_closed_catastrophe_ids: set[int] = set()
        self.active_catastrophe_ids: set[int] = set()
        self.current_tick: int = 0
        self._initial_population_bootstrapped = False
        self._closed = False

    def _write_parquet(self, df: pd.DataFrame, path: Path, writer_attr: str, schema_attr: str):
        writer = getattr(self, writer_attr)
        schema = getattr(self, schema_attr)
        df = self._align_dataframe_to_schema(df.copy(), schema)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        if writer is None:
            setattr(self, schema_attr, table.schema)
            writer = pq.ParquetWriter(str(path), table.schema, compression="gzip")
            setattr(self, writer_attr, writer)
        writer.write_table(table)

    def _align_dataframe_to_schema(self, df: pd.DataFrame, schema: pa.Schema | None) -> pd.DataFrame:
        if schema is None:
            return df
        for column_name in schema.names:
            if column_name not in df.columns:
                df[column_name] = None
        return df

    def _schema_versions(self) -> dict:
        return {
            "identity": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
            "observation": cfg.SCHEMA.OBS_SCHEMA_VERSION,
            "checkpoint": cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION,
            "reproduction": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
            "catastrophe": cfg.SCHEMA.CATASTROPHE_SCHEMA_VERSION,
            "telemetry": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
            "logging": cfg.SCHEMA.LOGGING_SCHEMA_VERSION,
        }

    def _increment_tick_counter(self, counter: dict[int, int], tick: int, amount: int = 1) -> None:
        counter[int(tick)] = counter.get(int(tick), 0) + int(amount)

    def _increment_family_tick_counter(self, counter: dict[tuple[int, str], int], tick: int, family_id: str | None, amount: int = 1) -> None:
        if family_id is None:
            return
        key = (int(tick), str(family_id))
        counter[key] = counter.get(key, 0) + int(amount)

    def get_tick_birth_count(self, tick: int) -> int:
        return int(self.birth_counts_by_tick.get(int(tick), 0))

    def get_tick_death_count(self, tick: int) -> int:
        return int(self.death_counts_by_tick.get(int(tick), 0))

    def _training_state_fields(self, ppo, uid: int) -> dict:
        state = ppo.training_state_by_uid.get(uid)
        if state is None:
            return {
                "optimizer_steps": 0,
                "ppo_updates": 0,
                "env_steps": 0,
                "truncated_rollouts": 0,
            }
        return {
            "optimizer_steps": int(state.optimizer_steps),
            "ppo_updates": int(state.ppo_updates),
            "env_steps": int(state.env_steps),
            "truncated_rollouts": int(state.truncated_rollouts),
        }

    def _trait_fields_for_uid(self, registry, uid: int) -> dict:
        latent = registry.get_trait_latent_for_uid(uid)
        mapped = trait_values_from_latent(latent)
        return {
            "trait_budget": float(mapped["budget"]),
            "alloc_hp": float(mapped["alloc_hp"]),
            "alloc_mass": float(mapped["alloc_mass"]),
            "alloc_vision": float(mapped["alloc_vision"]),
            "alloc_metab": float(mapped["alloc_metab"]),
        }

    def _spawn_and_trait_surface(self, registry, slot_idx: int, uid: int) -> dict:
        data = registry.data[:, slot_idx]
        return {
            "birth_slot": int(slot_idx),
            "spawn_x": int(data[registry.X].item()),
            "spawn_y": int(data[registry.Y].item()),
            "hp_max": float(data[registry.HP_MAX].item()),
            "mass": float(data[registry.MASS].item()),
            "vision": float(data[registry.VISION].item()),
            "metabolism": float(data[registry.METABOLISM_RATE].item()),
            "lineage_depth": int(registry.uid_generation_depth.get(uid, 0)),
            **self._trait_fields_for_uid(registry, uid),
        }

    def _open_life_record(self, *, registry, slot_idx: int, uid: int, tick: int, mutation_flags: dict | None = None) -> None:
        if uid in self.open_lives_by_uid:
            return

        family_id = registry.get_family_for_uid(uid)
        parent_roles = registry.get_parent_roles_for_uid(uid)
        record = {
            "agent_uid": int(uid),
            "brain_family": family_id,
            "brain_parent_uid": int(parent_roles["brain_parent_uid"]),
            "trait_parent_uid": int(parent_roles["trait_parent_uid"]),
            "anchor_parent_uid": int(parent_roles["anchor_parent_uid"]),
            "birth_tick": int(tick),
            "death_tick": None,
            "age_at_death": None,
            "birth_slot": int(slot_idx),
            "death_slot": None,
            "spawn_x": int(registry.data[registry.X, slot_idx].item()),
            "spawn_y": int(registry.data[registry.Y, slot_idx].item()),
            "death_x": None,
            "death_y": None,
            "death_reason": None,
            "final_hp": None,
            "hp_max": float(registry.data[registry.HP_MAX, slot_idx].item()),
            "mass": float(registry.data[registry.MASS, slot_idx].item()),
            "vision": float(registry.data[registry.VISION, slot_idx].item()),
            "metabolism": float(registry.data[registry.METABOLISM_RATE, slot_idx].item()),
            "optimizer_steps": 0,
            "ppo_updates": 0,
            "env_steps": 0,
            "truncated_rollouts": 0,
            "catastrophes_survived_count": 0,
            "rare_mutation_flag": bool((mutation_flags or {}).get("rare_mutation", False)),
            "family_shift_mutation_flag": bool((mutation_flags or {}).get("family_shift", False)),
            "lineage_depth": int(registry.uid_generation_depth.get(uid, 0)),
            **self._trait_fields_for_uid(registry, uid),
        }
        self.open_lives_by_uid[int(uid)] = record
        self.exposed_catastrophes_by_uid.setdefault(int(uid), set())
        self.survived_catastrophes_by_uid.setdefault(int(uid), set())

    def bootstrap_initial_population(self, registry) -> None:
        if self._initial_population_bootstrapped or not cfg.TELEMETRY.ENABLE_DEEP_LEDGERS:
            return

        for uid, slot_idx in sorted(registry.active_uid_to_slot.items()):
            tick = int(registry.uid_lifecycle[uid].birth_tick)
            self._open_life_record(
                registry=registry,
                slot_idx=slot_idx,
                uid=uid,
                tick=tick,
                mutation_flags={"rare_mutation": False, "family_shift": False},
            )
            latent = registry.get_trait_latent_for_uid(uid)
            traits = trait_values_from_latent(latent)
            spawn_x = int(registry.data[registry.X, slot_idx].item())
            spawn_y = int(registry.data[registry.Y, slot_idx].item())
            root_row = {
                "child_uid": int(uid),
                "birth_tick": tick,
                "brain_parent_uid": -1,
                "trait_parent_uid": -1,
                "anchor_parent_uid": -1,
                "parent_uid": -1,
                "child_family": registry.get_family_for_uid(uid),
                "inherited_family_source": "root_seed",
                "spawn_x": spawn_x,
                "spawn_y": spawn_y,
                "used_global_fallback": False,
                "floor_recovery_flag": False,
                "thresholds_suspended_flag": False,
                "rare_mutation_flag": False,
                "family_shift_flag": False,
                "mutation_sigma_policy": 0.0,
                "mutation_sigma_traits": 0.0,
                "child_slot": int(slot_idx),
                "birth_slot": int(slot_idx),
                "brain_parent_slot": -1,
                "trait_parent_slot": -1,
                "anchor_parent_slot": -1,
                "parent_slot": -1,
                "brain_parent_family": "root_seed",
                "trait_parent_family": "root_seed",
                "parent_idx": -1,
                "child_idx": int(slot_idx),
                **self._spawn_and_trait_surface(registry, slot_idx, uid),
                **{f"trait_{k}": float(v) for k, v in latent.items()},
                **{f"value_{k}": float(v) for k, v in traits.items()},
                "mutation_rare_mutation": False,
                "mutation_family_shift": False,
                "mutation_placement_failed": False,
                "placement_x": spawn_x,
                "placement_y": spawn_y,
                "placement_attempts": 1,
                "placement_used_global_fallback": False,
                "placement_failure_reason": "",
                "identity_schema_version": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
                "telemetry_schema_version": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
                "reproduction_schema_version": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
            }
            frame = pd.DataFrame([root_row])
            self._write_parquet(frame, self.birth_ledger_path, "birth_writer", "birth_schema")
            self._increment_tick_counter(self.birth_counts_by_tick, tick, 1)
            self._increment_family_tick_counter(self.birth_counts_by_family_and_tick, tick, registry.get_family_for_uid(uid), 1)

        self._initial_population_bootstrapped = True

    def log_agent_snapshot(self, tick: int, registry):
        self.h5_snapshots.create_dataset(f"tick_{tick}", data=registry.data.cpu().numpy(), compression="gzip")
        self.h5_identity.create_dataset(f"slot_uid_tick_{tick}", data=registry.slot_uid.cpu().numpy(), compression="gzip")
        self.h5_identity.create_dataset(f"slot_parent_uid_tick_{tick}", data=registry.slot_parent_uid.cpu().numpy(), compression="gzip")

    def log_heatmap_snapshot(self, tick: int, grid):
        self.h5_heatmaps.create_dataset(f"density_tick_{tick}", data=grid.grid[2].cpu().numpy(), compression="gzip")
        self.h5_heatmaps.create_dataset(f"mass_tick_{tick}", data=grid.grid[3].cpu().numpy(), compression="gzip")
        self.h5_heatmaps.create_dataset(f"h_rate_tick_{tick}", data=grid.grid[1].cpu().numpy(), compression="gzip")

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
        payload = {
            "by_uid": by_uid,
            "uid_to_slot": uid_to_slot,
            "family_by_uid": family_by_uid,
            "tick": tick,
            "schema_versions": self._schema_versions(),
        }
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
        latent_defaults = default_trait_latent()
        value_defaults = trait_values_from_latent(latent_defaults)
        canonical_traits = trait_values_from_latent(trait_latent) if trait_latent else value_defaults
        payload = {
            "child_uid": int(child_uid),
            "birth_tick": int(tick),
            "brain_parent_uid": int(brain_parent_uid),
            "trait_parent_uid": int(trait_parent_uid),
            "anchor_parent_uid": int(anchor_parent_uid),
            "parent_uid": int(brain_parent_uid),
            "child_family": child_family,
            "inherited_family_source": brain_parent_family if brain_parent_family is not None else "root_seed",
            "spawn_x": int(placement.get("x", placement.get("spawn_x", -1))) if placement.get("x", placement.get("spawn_x")) is not None else None,
            "spawn_y": int(placement.get("y", placement.get("spawn_y", -1))) if placement.get("y", placement.get("spawn_y")) is not None else None,
            "used_global_fallback": bool(placement.get("used_global_fallback", False)),
            "floor_recovery_flag": bool(floor_recovery),
            "thresholds_suspended_flag": bool(floor_recovery and cfg.RESPAWN.FLOOR_RECOVERY_SUSPEND_THRESHOLDS),
            "rare_mutation_flag": bool(mutation_flags.get("rare_mutation", False)),
            "family_shift_flag": bool(mutation_flags.get("family_shift", False)),
            "mutation_sigma_policy": float(cfg.EVOL.RARE_POLICY_NOISE_SD if mutation_flags.get("rare_mutation", False) else cfg.EVOL.POLICY_NOISE_SD),
            "mutation_sigma_traits": float(cfg.EVOL.RARE_TRAIT_LOGIT_MUTATION_SIGMA if mutation_flags.get("rare_mutation", False) else cfg.EVOL.TRAIT_LOGIT_MUTATION_SIGMA),
            "child_slot": int(child_slot),
            "birth_slot": int(child_slot),
            "brain_parent_slot": int(brain_parent_slot),
            "trait_parent_slot": int(trait_parent_slot),
            "anchor_parent_slot": int(anchor_parent_slot),
            "parent_slot": int(brain_parent_slot),
            "brain_parent_family": brain_parent_family if brain_parent_family is not None else "root_seed",
            "trait_parent_family": trait_parent_family if trait_parent_family is not None else "root_seed",
            "hp_max": float(canonical_traits["hp_max"]),
            "mass": float(canonical_traits["mass"]),
            "vision": float(canonical_traits["vision"]),
            "metabolism": float(canonical_traits["metab"]),
            "trait_budget": float(canonical_traits["budget"]),
            "alloc_hp": float(canonical_traits["alloc_hp"]),
            "alloc_mass": float(canonical_traits["alloc_mass"]),
            "alloc_vision": float(canonical_traits["alloc_vision"]),
            "alloc_metab": float(canonical_traits["alloc_metab"]),
            **{f"trait_{k}": float(v) for k, v in latent_defaults.items()},
            **{f"value_{k}": float(v) for k, v in value_defaults.items()},
            "mutation_rare_mutation": False,
            "mutation_family_shift": False,
            "mutation_placement_failed": False,
            "placement_x": placement.get("x", placement.get("spawn_x")),
            "placement_y": placement.get("y", placement.get("spawn_y")),
            "placement_attempts": int(placement.get("attempts", 0)),
            "placement_used_global_fallback": bool(placement.get("used_global_fallback", False)),
            "placement_failure_reason": placement.get("failure_reason") or "",
            **{f"trait_{k}": float(v) for k, v in trait_latent.items()},
            **{f"value_{k}": float(v) for k, v in traits.items()},
            **{f"mutation_{k}": v for k, v in mutation_flags.items()},
            **{f"placement_{k}": v for k, v in placement.items()},
            "identity_schema_version": cfg.SCHEMA.IDENTITY_SCHEMA_VERSION,
            "telemetry_schema_version": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
            "reproduction_schema_version": cfg.SCHEMA.REPRODUCTION_SCHEMA_VERSION,
        }
        if cfg.MIGRATION.LOG_LEGACY_SLOT_FIELDS:
            payload["parent_idx"] = brain_parent_slot
            payload["child_idx"] = child_slot

        if child_uid != -1 and child_family is not None:
            self._increment_tick_counter(self.birth_counts_by_tick, tick, 1)
            self._increment_family_tick_counter(self.birth_counts_by_family_and_tick, tick, child_family, 1)

        frame = pd.DataFrame([payload])
        birth_frame = self._align_dataframe_to_schema(frame.copy(), self.birth_schema)
        genealogy_frame = self._align_dataframe_to_schema(frame.copy(), self.genealogy_schema)
        self._write_parquet(birth_frame, self.birth_ledger_path, "birth_writer", "birth_schema")
        self._write_parquet(genealogy_frame, self.genealogy_path, "genealogy_writer", "genealogy_schema")

    def log_physics_events(self, tick: int, collision_log: list[dict]):
        if not collision_log:
            return
        df = pd.DataFrame(collision_log)
        if "contenders" in df.columns:
            df["contenders"] = df["contenders"].apply(
                lambda value: [int(item) for item in value] if isinstance(value, (list, tuple)) else []
            )
        if "catastrophe_collision_scalar" not in df.columns:
            df["catastrophe_collision_scalar"] = 1.0
        df["tick"] = tick
        if self.collisions_schema is None:
            self.collisions_schema = pa.schema(
                [
                    pa.field("kind", pa.string()),
                    pa.field("a", pa.int64()),
                    pa.field("b", pa.int64()),
                    pa.field("damage", pa.float64()),
                    pa.field("damage_a", pa.float64()),
                    pa.field("damage_b", pa.float64()),
                    pa.field("contenders", pa.list_(pa.int64())),
                    pa.field("winner", pa.int64()),
                    pa.field("catastrophe_collision_scalar", pa.float64()),
                    pa.field("tick", pa.int64()),
                ]
            )
        self._write_parquet(df, self.collisions_path, "collisions_writer", "collisions_schema")

    def log_ppo_update(self, tick: int, ppo_stats_list: list[dict]):
        if not ppo_stats_list or not cfg.TELEMETRY.LOG_PPO_UPDATE_LEDGER:
            return
        df = pd.DataFrame(ppo_stats_list)
        df["tick"] = tick
        df["telemetry_schema_version"] = cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION
        self._write_parquet(df, self.ppo_path, "ppo_writer", "ppo_schema")

    def log_catastrophe_event(self, payload: dict):
        if not payload or not cfg.TELEMETRY.LOG_CATASTROPHE_EVENT_LEDGER:
            return
        payload = dict(payload)
        payload["catastrophe_schema_version"] = cfg.SCHEMA.CATASTROPHE_SCHEMA_VERSION
        payload["telemetry_schema_version"] = cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION

        event_id = int(payload.get("event_id", -1))
        kind = str(payload.get("kind", "unknown"))
        if kind == "start" and event_id != -1:
            self.active_catastrophe_ids.add(event_id)
        elif kind in {"end", "clear"} and event_id != -1:
            self.active_catastrophe_ids.discard(event_id)
            self.just_closed_catastrophe_ids.add(event_id)

        self._write_parquet(
            pd.DataFrame([payload]),
            self.catastrophes_path,
            "catastrophes_writer",
            "catastrophes_schema",
        )

    def note_catastrophe_exposure(self, registry, catastrophe_state: dict | None = None) -> None:
        if not cfg.TELEMETRY.TRACK_CATASTROPHE_EXPOSURE:
            return

        catastrophe_state = catastrophe_state or {}
        active_ids = {int(detail["event_id"]) for detail in catastrophe_state.get("active_details", [])}
        for uid in registry.active_uid_to_slot.keys():
            if active_ids:
                self.exposed_catastrophes_by_uid.setdefault(int(uid), set()).update(active_ids)

        if self.just_closed_catastrophe_ids:
            for uid in registry.active_uid_to_slot.keys():
                seen = self.exposed_catastrophes_by_uid.setdefault(int(uid), set())
                survived = self.survived_catastrophes_by_uid.setdefault(int(uid), set())
                survived.update(event_id for event_id in self.just_closed_catastrophe_ids if event_id in seen)
            self.just_closed_catastrophe_ids.clear()

    def get_catastrophe_exposure_summary(self, uid: int) -> dict:
        active_seen = sorted(self.exposed_catastrophes_by_uid.get(int(uid), set()) & self.active_catastrophe_ids)
        survived = sorted(self.survived_catastrophes_by_uid.get(int(uid), set()))
        return {
            "active_event_ids": active_seen,
            "active_count": len(active_seen),
            "survived_count": len(survived),
        }

    def finalize_death(self, *, tick: int, slot_idx: int, registry, ppo, death_context: dict | None = None) -> None:
        uid = registry.get_uid_for_slot(slot_idx)
        if uid == -1:
            raise AssertionError(f"Cannot finalize a life ledger row for an unbound slot {slot_idx}")

        death_context = dict(death_context or {})
        family_id = registry.get_family_for_uid(uid)
        life_row = dict(self.open_lives_by_uid.get(uid, {}))
        if not life_row:
            self._open_life_record(
                registry=registry,
                slot_idx=slot_idx,
                uid=uid,
                tick=int(registry.uid_lifecycle[uid].birth_tick),
                mutation_flags={"rare_mutation": False, "family_shift": False},
            )
            life_row = dict(self.open_lives_by_uid[uid])

        life_row.update(
            {
                "death_tick": int(tick),
                "age_at_death": int(tick) - int(life_row["birth_tick"]),
                "death_slot": int(slot_idx),
                "death_x": int(registry.data[registry.X, slot_idx].item()),
                "death_y": int(registry.data[registry.Y, slot_idx].item()),
                "death_reason": str(death_context.get("death_reason", "unknown")),
                "final_hp": float(registry.data[registry.HP, slot_idx].item()),
                "catastrophes_survived_count": int(len(self.survived_catastrophes_by_uid.get(uid, set()))),
                **self._training_state_fields(ppo, uid),
            }
        )

        death_row = {
            "agent_uid": int(uid),
            "tick": int(tick),
            "death_reason": str(death_context.get("death_reason", "unknown")),
            "catastrophe_id": death_context.get("catastrophe_id"),
            "killing_agent_uid": death_context.get("killing_agent_uid"),
            "zone_id": death_context.get("zone_id"),
            "position_x": int(registry.data[registry.X, slot_idx].item()),
            "position_y": int(registry.data[registry.Y, slot_idx].item()),
            "final_hp": float(registry.data[registry.HP, slot_idx].item()),
            "family": family_id,
            "lineage_depth": int(registry.uid_generation_depth.get(uid, 0)),
            "telemetry_schema_version": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
        }

        self._write_parquet(pd.DataFrame([death_row]), self.death_ledger_path, "death_writer", "death_schema")
        self._write_parquet(pd.DataFrame([life_row]), self.life_ledger_path, "life_writer", "life_schema")
        self.finalized_lives_by_uid[int(uid)] = dict(life_row)
        self.open_lives_by_uid.pop(int(uid), None)

        self._increment_tick_counter(self.death_counts_by_tick, tick, 1)
        self._increment_family_tick_counter(self.death_counts_by_family_and_tick, tick, family_id, 1)

    def log_tick_summary(
        self,
        tick: int,
        registry,
        physics_stats: dict,
        catastrophe_state: dict | None = None,
        *,
        births_this_tick: int = 0,
        deaths_this_tick: int = 0,
        reproduction_disabled: bool = False,
        floor_recovery_active: bool = False,
        ppo=None,
    ):
        if not cfg.TELEMETRY.LOG_TICK_SUMMARY:
            return

        self.current_tick = int(tick)
        alive_mask = registry.get_alive_mask()
        num_alive = int(alive_mask.sum().item())
        catastrophe_state = catastrophe_state or {}
        family_counts = {family_id: 0 for family_id in cfg.BRAIN.FAMILY_ORDER}
        family_mean_hp_ratio = {family_id: None for family_id in cfg.BRAIN.FAMILY_ORDER}

        if num_alive > 0:
            for slot_idx in registry.get_alive_indices().tolist():
                family_id = registry.get_family_for_slot(int(slot_idx))
                if family_id is not None:
                    family_counts[family_id] += 1

            for family_id in cfg.BRAIN.FAMILY_ORDER:
                slots = [
                    int(slot_idx)
                    for slot_idx in registry.get_alive_indices().tolist()
                    if registry.get_family_for_slot(int(slot_idx)) == family_id
                ]
                if slots:
                    hp = registry.data[registry.HP, slots]
                    hp_max = registry.data[registry.HP_MAX, slots]
                    family_mean_hp_ratio[family_id] = float((hp / (hp_max + 1e-6)).mean().item())

        if num_alive == 0:
            stats = {
                "tick": int(tick),
                "live_population": 0,
                "births_this_tick": int(births_this_tick),
                "deaths_this_tick": int(deaths_this_tick),
                "reproduction_disabled_flag": bool(reproduction_disabled),
                "floor_recovery_active_flag": bool(floor_recovery_active),
                "catastrophe_active_count": int(catastrophe_state.get("active_count", 0)),
            }
        else:
            hp = registry.data[registry.HP, alive_mask]
            hp_max = registry.data[registry.HP_MAX, alive_mask]
            stats = {
                "tick": int(tick),
                "live_population": int(num_alive),
                "mean_hp_ratio": float((hp / (hp_max + 1e-6)).mean().item()),
                "mean_mass": float(registry.data[registry.MASS, alive_mask].mean().item()),
                "mean_vision": float(registry.data[registry.VISION, alive_mask].mean().item()),
                "mean_metabolism": float(registry.data[registry.METABOLISM_RATE, alive_mask].mean().item()),
                "births_this_tick": int(births_this_tick),
                "deaths_this_tick": int(deaths_this_tick),
                "reproduction_disabled_flag": bool(reproduction_disabled),
                "floor_recovery_active_flag": bool(floor_recovery_active),
                "catastrophe_mode": catastrophe_state.get("mode"),
                "catastrophe_active_count": int(catastrophe_state.get("active_count", 0)),
                "catastrophe_active_names": "|".join(catastrophe_state.get("active_names", [])),
                "catastrophe_next_tick": catastrophe_state.get("next_auto_tick", -1),
                "catastrophe_paused": bool(catastrophe_state.get("scheduler_paused", False)),
                **physics_stats,
            }
            for family_id, count in family_counts.items():
                slug = family_id.lower().replace(" ", "_")
                stats[f"family_count__{slug}"] = int(count)

        stats["telemetry_schema_version"] = cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION
        self._write_parquet(pd.DataFrame([stats]), self.tick_summary_path, "tick_summary_writer", "tick_summary_schema")

        if cfg.TELEMETRY.LOG_FAMILY_SUMMARY and tick % max(1, cfg.TELEMETRY.FAMILY_SUMMARY_EVERY_TICKS) == 0:
            rows = []
            training_state_by_uid = {} if ppo is None else ppo.training_state_by_uid
            for family_id in cfg.BRAIN.FAMILY_ORDER:
                update_count = sum(
                    int(training_state_by_uid[uid].ppo_updates)
                    for uid, uid_family in registry.uid_family.items()
                    if uid_family == family_id and uid in training_state_by_uid
                )
                rows.append(
                    {
                        "tick": int(tick),
                        "family_id": family_id,
                        "active_count": int(family_counts.get(family_id, 0)),
                        "births_this_tick": int(self.birth_counts_by_family_and_tick.get((int(tick), family_id), 0)),
                        "deaths_this_tick": int(self.death_counts_by_family_and_tick.get((int(tick), family_id), 0)),
                        "mean_hp_ratio": family_mean_hp_ratio.get(family_id),
                        "mean_lineage_depth": self._mean_lineage_depth_for_family(registry, family_id),
                        "ppo_update_count_cumulative": int(update_count),
                        "telemetry_schema_version": cfg.SCHEMA.TELEMETRY_SCHEMA_VERSION,
                    }
                )
            self._write_parquet(
                pd.DataFrame(rows),
                self.family_summary_path,
                "family_summary_writer",
                "family_summary_schema",
            )

    def _mean_lineage_depth_for_family(self, registry, family_id: str) -> float | None:
        depths = [
            int(registry.uid_generation_depth.get(uid, 0))
            for uid, uid_family in registry.uid_family.items()
            if uid_family == family_id and registry.is_uid_active(uid)
        ]
        if not depths:
            return None
        return float(sum(depths) / len(depths))

    def export_lineage(self, registry) -> dict | None:
        if not cfg.TELEMETRY.EXPORT_LINEAGE:
            return None
        rows_by_uid = {**self.finalized_lives_by_uid, **self.open_lives_by_uid}
        return export_lineage_json(self.lineage_path, registry, life_rows_by_uid=rows_by_uid)

    def close(self, registry=None):
        if self._closed:
            return

        if registry is not None and cfg.TELEMETRY.FLUSH_OPEN_LIVES_ON_CLOSE:
            for uid, slot_idx in sorted(registry.active_uid_to_slot.items()):
                life_row = dict(self.open_lives_by_uid.get(uid, {}))
                if not life_row:
                    continue
                life_row.update(
                    {
                        "optimizer_steps": int(life_row.get("optimizer_steps", 0)),
                        "ppo_updates": int(life_row.get("ppo_updates", 0)),
                        "env_steps": int(life_row.get("env_steps", 0)),
                        "truncated_rollouts": int(life_row.get("truncated_rollouts", 0)),
                        "catastrophes_survived_count": int(len(self.survived_catastrophes_by_uid.get(uid, set()))),
                    }
                )
                self._write_parquet(pd.DataFrame([life_row]), self.life_ledger_path, "life_writer", "life_schema")
                self.finalized_lives_by_uid[int(uid)] = dict(life_row)
            self.export_lineage(registry)

        self.h5_file.close()
        for attr in (
            "birth_writer",
            "genealogy_writer",
            "life_writer",
            "death_writer",
            "collisions_writer",
            "ppo_writer",
            "tick_summary_writer",
            "family_summary_writer",
            "catastrophes_writer",
        ):
            writer = getattr(self, attr)
            if writer is not None:
                writer.close()

        self._closed = True

