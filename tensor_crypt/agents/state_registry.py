from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config_bridge import cfg
from .brain import Brain


@dataclass
class AgentLifecycleRecord:
    uid: int
    parent_uid: int
    birth_tick: int
    death_tick: int | None
    current_slot: int | None
    is_active: bool


class Registry:
    # Registry column indices
    ALIVE = 0
    X = 1
    Y = 2
    HP = 3
    LAST_ACTION = 4
    MASS = 5
    HP_MAX = 6
    VISION = 7
    METABOLISM_RATE = 8
    AGENT_UID_SHADOW = 9
    AGENT_ID = AGENT_UID_SHADOW
    TICK_BORN = 10
    PARENT_UID_SHADOW = 11
    PARENT_ID = PARENT_UID_SHADOW
    HP_GAINED = 12
    HP_LOST_PHYSICS = 13
    OPTIMIZATION_CYCLE = 14

    NUM_COLS = 15

    def __init__(self):
        self.device = cfg.SIM.DEVICE
        self.max_agents = cfg.AGENTS.N

        # Structure-of-arrays state tensor. Column order is a hard contract with
        # physics, perception, evolution, logging, and the viewer.
        self.data = torch.zeros(self.NUM_COLS, self.max_agents, device=self.device, dtype=torch.float32)

        # Slot-resident brains remain the execution objects for Prompt 1, but the
        # active UID bound to the slot is the canonical owner.
        self.brains: List[Optional[nn.Module]] = [None] * self.max_agents

        self.slot_uid = torch.full((self.max_agents,), -1, device=self.device, dtype=torch.int64)
        self.slot_parent_uid = torch.full((self.max_agents,), -1, device=self.device, dtype=torch.int64)
        self.active_uid_to_slot: Dict[int, int] = {}
        self.uid_lifecycle: Dict[int, AgentLifecycleRecord] = {}
        self.next_agent_uid = 0
        self.next_unique_id = 0
        self.tick_counter = 0
        self.fitness = torch.zeros(self.max_agents, device=self.device, dtype=torch.float32)

        self.data[self.AGENT_UID_SHADOW] = -1.0
        self.data[self.PARENT_UID_SHADOW] = -1.0

    def allocate_uid(self, parent_uid: int, birth_tick: int) -> int:
        if parent_uid != -1 and cfg.IDENTITY.ASSERT_HISTORICAL_UIDS and parent_uid not in self.uid_lifecycle:
            raise AssertionError(f"Parent UID {parent_uid} is not present in the lifecycle ledger")

        uid = self.next_agent_uid
        if uid in self.uid_lifecycle:
            raise AssertionError(f"UID {uid} is already present in the lifecycle ledger")

        self.uid_lifecycle[uid] = AgentLifecycleRecord(
            uid=uid,
            parent_uid=parent_uid,
            birth_tick=birth_tick,
            death_tick=None,
            current_slot=None,
            is_active=True,
        )
        self.next_agent_uid += 1
        self.next_unique_id = self.next_agent_uid
        return uid

    def bind_uid_to_slot(self, uid: int, slot_idx: int) -> None:
        if uid not in self.uid_lifecycle:
            raise KeyError(f"Cannot bind unknown UID {uid}")
        if self.get_uid_for_slot(slot_idx) != -1:
            raise AssertionError(f"Slot {slot_idx} is already bound to UID {self.get_uid_for_slot(slot_idx)}")
        if uid in self.active_uid_to_slot:
            raise AssertionError(f"UID {uid} is already active in slot {self.active_uid_to_slot[uid]}")

        record = self.uid_lifecycle[uid]
        if record.death_tick is not None:
            raise AssertionError(f"Historical UID {uid} cannot be rebound")
        if record.current_slot is not None:
            raise AssertionError(f"UID {uid} already tracks current slot {record.current_slot}")

        self.slot_uid[slot_idx] = uid
        self.slot_parent_uid[slot_idx] = record.parent_uid
        self.active_uid_to_slot[uid] = slot_idx
        record.current_slot = slot_idx
        record.is_active = True

    def finalize_death(self, slot_idx: int, death_tick: int, *, assert_after: bool = True) -> int:
        uid = self.get_uid_for_slot(slot_idx)
        if uid == -1:
            raise AssertionError(f"Slot {slot_idx} has no bound UID to finalize")

        record = self.uid_lifecycle[uid]
        if not record.is_active:
            raise AssertionError(f"UID {uid} is already historical")

        record.death_tick = death_tick
        record.current_slot = None
        record.is_active = False
        self.active_uid_to_slot.pop(uid, None)
        self.slot_uid[slot_idx] = -1
        self.slot_parent_uid[slot_idx] = -1
        self.sync_identity_shadow_columns()
        if assert_after:
            self.assert_identity_invariants()
        return uid

    def get_uid_for_slot(self, slot_idx: int) -> int:
        return int(self.slot_uid[slot_idx].item())

    def get_slot_for_uid(self, uid: int) -> int | None:
        return self.active_uid_to_slot.get(uid)

    def is_uid_active(self, uid: int) -> bool:
        record = self.uid_lifecycle.get(uid)
        return bool(record and record.is_active)

    def get_parent_uid_for_slot(self, slot_idx: int) -> int:
        return int(self.slot_parent_uid[slot_idx].item())

    def sync_identity_shadow_columns(self) -> None:
        if not cfg.IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS:
            return

        self.data[self.AGENT_UID_SHADOW] = -1.0
        self.data[self.PARENT_UID_SHADOW] = -1.0
        bound_mask = self.slot_uid >= 0
        if bound_mask.any():
            self.data[self.AGENT_UID_SHADOW, bound_mask] = self.slot_uid[bound_mask].to(torch.float32)
            self.data[self.PARENT_UID_SHADOW, bound_mask] = self.slot_parent_uid[bound_mask].to(torch.float32)

    def assert_identity_invariants(self) -> None:
        if not cfg.IDENTITY.ASSERT_BINDINGS:
            return

        expected_uids = set(range(self.next_agent_uid))
        actual_uids = set(self.uid_lifecycle.keys())
        if expected_uids != actual_uids:
            raise AssertionError("UID lifecycle ledger is missing allocated UIDs")

        seen_active_uids = set()
        for slot_idx in range(self.max_agents):
            uid = self.get_uid_for_slot(slot_idx)
            alive = bool(self.data[self.ALIVE, slot_idx].item() > 0.5)

            if uid == -1:
                if alive:
                    raise AssertionError(f"Alive slot {slot_idx} has no canonical UID binding")
                if int(self.slot_parent_uid[slot_idx].item()) != -1:
                    raise AssertionError(f"Unbound slot {slot_idx} still has a parent UID binding")
                continue

            if not alive:
                raise AssertionError(f"Dead slot {slot_idx} still owns active UID {uid}")
            if uid in seen_active_uids:
                raise AssertionError(f"UID {uid} is bound to more than one slot")
            seen_active_uids.add(uid)

            record = self.uid_lifecycle.get(uid)
            if record is None:
                raise AssertionError(f"Active UID {uid} is missing from the lifecycle ledger")
            if not record.is_active or record.current_slot != slot_idx or record.death_tick is not None:
                raise AssertionError(f"Lifecycle record for UID {uid} disagrees with slot binding {slot_idx}")
            if self.active_uid_to_slot.get(uid) != slot_idx:
                raise AssertionError(f"UID {uid} is missing from the active UID-to-slot map")

        if set(self.active_uid_to_slot.keys()) != seen_active_uids:
            raise AssertionError("Active UID-to-slot map disagrees with slot bindings")

        if cfg.IDENTITY.ASSERT_HISTORICAL_UIDS:
            for uid, record in self.uid_lifecycle.items():
                if record.parent_uid != -1 and record.parent_uid not in self.uid_lifecycle:
                    raise AssertionError(f"UID {uid} references unknown parent UID {record.parent_uid}")
                if record.is_active:
                    if record.current_slot is None or record.death_tick is not None:
                        raise AssertionError(f"Active UID {uid} has invalid lifecycle state")
                else:
                    if record.current_slot is not None or record.death_tick is None:
                        raise AssertionError(f"Historical UID {uid} has invalid lifecycle state")

        if cfg.IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS:
            expected_uid_shadow = torch.where(self.slot_uid >= 0, self.slot_uid, torch.full_like(self.slot_uid, -1)).to(torch.float32)
            expected_parent_shadow = torch.where(
                self.slot_parent_uid >= 0,
                self.slot_parent_uid,
                torch.full_like(self.slot_parent_uid, -1),
            ).to(torch.float32)
            if not torch.equal(self.data[self.AGENT_UID_SHADOW], expected_uid_shadow):
                raise AssertionError("Legacy AGENT_ID shadow column diverged from canonical slot UID state")
            if not torch.equal(self.data[self.PARENT_UID_SHADOW], expected_parent_shadow):
                raise AssertionError("Legacy PARENT_ID shadow column diverged from canonical parent UID state")

    def spawn_initial_population(self, grid):
        """Spawn initial agents uniformly across the grid."""
        alive_count = 0
        attempts = 0
        max_attempts = self.max_agents * 10

        while alive_count < self.max_agents and attempts < max_attempts:
            x = random.randint(1, grid.W - 2)
            y = random.randint(1, grid.H - 2)

            if not grid.is_wall(x, y) and grid.get_agent_at(x, y) == -1:
                self.spawn_agent(alive_count, x, y, parent_uid=-1, grid=grid)
                alive_count += 1

            attempts += 1

        print(f"Spawned {alive_count} agents")

    def spawn_agent(
        self,
        idx: int,
        x: int,
        y: int,
        parent_uid: int,
        grid,
        traits=None,
        *,
        tick_born: int | None = None,
        assign_new_identity: bool = True,
    ) -> int:
        """
        Spawn a single agent into an unbound slot.

        The float AGENT_ID/PARENT_ID columns remain compatibility mirrors only.
        Canonical identity ownership is always allocated through the UID ledger.
        """
        _ = assign_new_identity

        if idx < 0 or idx >= self.max_agents:
            raise IndexError(f"Slot index {idx} is out of bounds for {self.max_agents} agents")
        if self.get_uid_for_slot(idx) != -1:
            raise AssertionError(f"Slot {idx} must be unbound before a new UID can be attached")
        if self.data[self.ALIVE, idx] > 0.5:
            raise AssertionError(f"Slot {idx} is already alive")

        if tick_born is None:
            tick_born = self.tick_counter
        uid = self.allocate_uid(int(parent_uid), int(tick_born))
        self.bind_uid_to_slot(uid, idx)

        self.data[self.ALIVE, idx] = 1.0
        self.data[self.X, idx] = float(x)
        self.data[self.Y, idx] = float(y)

        if traits is None:
            self.data[self.MASS, idx] = cfg.TRAITS.INIT.mass
            self.data[self.HP_MAX, idx] = cfg.TRAITS.INIT.hp_max
            self.data[self.VISION, idx] = cfg.TRAITS.INIT.vision
            self.data[self.METABOLISM_RATE, idx] = cfg.TRAITS.INIT.metab
        else:
            self.data[self.MASS, idx] = traits["mass"]
            self.data[self.HP_MAX, idx] = traits["hp_max"]
            self.data[self.VISION, idx] = traits["vision"]
            self.data[self.METABOLISM_RATE, idx] = traits["metab"]

        self.data[self.HP, idx] = self.data[self.HP_MAX, idx]
        self.data[self.LAST_ACTION, idx] = 0.0
        self.data[self.TICK_BORN, idx] = float(tick_born)
        self.data[self.HP_GAINED, idx] = 0.0
        self.data[self.HP_LOST_PHYSICS, idx] = 0.0
        self.data[self.OPTIMIZATION_CYCLE, idx] = 0.0

        if self.brains[idx] is None:
            self.brains[idx] = Brain().to(self.device)

        grid.set_cell(x, y, idx, self.data[self.MASS, idx].item())
        self.fitness[idx] = 0.0
        self.sync_identity_shadow_columns()
        self.assert_identity_invariants()
        return uid

    def get_alive_mask(self) -> torch.Tensor:
        return self.data[self.ALIVE] > 0.5

    def get_alive_indices(self) -> torch.Tensor:
        return torch.nonzero(self.data[self.ALIVE] > 0.5, as_tuple=False).squeeze(-1)

    def get_num_alive(self) -> int:
        return int(self.data[self.ALIVE].sum().item())

    def mark_dead(self, idx: int, grid):
        """Phase-A physical death mark. UID finalization happens later."""
        if self.data[self.ALIVE, idx] > 0.5:
            x = int(self.data[self.X, idx].item())
            y = int(self.data[self.Y, idx].item())
            grid.clear_cell(x, y)
            self.data[self.ALIVE, idx] = 0.0

    def check_invariants(self, grid):
        if not cfg.LOG.ASSERTIONS:
            return

        self.assert_identity_invariants()
        alive_mask = self.get_alive_mask()
        assert not torch.isnan(self.data).any(), "NaN detected in registry"

        alive_hp = self.data[self.HP, alive_mask]
        alive_hp_max = self.data[self.HP_MAX, alive_mask]
        assert (alive_hp >= 0).all(), "Negative HP detected"
        assert (alive_hp <= alive_hp_max + 1e-3).all(), "HP exceeds HP_MAX"

        if cfg.AGENTS.NO_STACKING:
            positions = set()
            for idx in self.get_alive_indices():
                slot_idx = int(idx.item())
                x = int(self.data[self.X, slot_idx].item())
                y = int(self.data[self.Y, slot_idx].item())
                pos = (x, y)
                assert pos not in positions, f"Multiple agents at {pos}"
                positions.add(pos)
                assert grid.get_agent_at(x, y) == slot_idx, f"Grid occupancy lost slot binding for slot {slot_idx}"




