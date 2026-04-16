"""Slot-backed agent registry with canonical UID lifecycle ownership.

The simulation still stores dense per-slot tensors for runtime speed, but
identity, lineage, and PPO ownership semantics are defined in terms of
monotonic UIDs. Slot reuse must never recycle canonical ownership state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config_bridge import cfg
from ..population.reproduction import default_trait_latent, trait_values_from_latent
from .brain import create_brain, get_bloodline_families, validate_bloodline_family


@dataclass
class AgentLifecycleRecord:
    uid: int
    parent_uid: int
    birth_tick: int
    death_tick: int | None
    current_slot: int | None
    is_active: bool


class Registry:
    """Own the slot tensors and the UID-to-slot lifecycle contract."""

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

        self.data = torch.zeros(self.NUM_COLS, self.max_agents, device=self.device, dtype=torch.float32)
        self.brains: List[Optional[nn.Module]] = [None] * self.max_agents

        self.slot_uid = torch.full((self.max_agents,), -1, device=self.device, dtype=torch.int64)
        self.slot_parent_uid = torch.full((self.max_agents,), -1, device=self.device, dtype=torch.int64)
        self.active_uid_to_slot: Dict[int, int] = {}
        self.uid_lifecycle: Dict[int, AgentLifecycleRecord] = {}
        self.uid_family: Dict[int, str] = {}
        self.uid_parent_roles: Dict[int, dict[str, int]] = {}
        self.uid_trait_latent: Dict[int, dict[str, float]] = {}
        self.uid_generation_depth: Dict[int, int] = {}
        self.slot_family: List[Optional[str]] = [None] * self.max_agents

        self.next_agent_uid = 0
        self.next_unique_id = 0
        self.tick_counter = 0
        self.fitness = torch.zeros(self.max_agents, device=self.device, dtype=torch.float32)
        self._initial_family_cursor = 0

        self.data[self.AGENT_UID_SHADOW] = -1.0
        self.data[self.PARENT_UID_SHADOW] = -1.0

    def allocate_uid(self, parent_uid: int, birth_tick: int) -> int:
        if parent_uid != -1 and cfg.IDENTITY.ASSERT_HISTORICAL_UIDS and parent_uid not in self.uid_lifecycle:
            raise AssertionError(f"Parent UID {parent_uid} is not present in the lifecycle ledger")
        uid = self.next_agent_uid
        if uid in self.uid_lifecycle:
            raise AssertionError(f"UID {uid} is already present in the lifecycle ledger")
        self.uid_lifecycle[uid] = AgentLifecycleRecord(uid=uid, parent_uid=parent_uid, birth_tick=birth_tick, death_tick=None, current_slot=None, is_active=True)
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
        self.slot_family[slot_idx] = None
        self.sync_identity_shadow_columns()
        if assert_after:
            self.assert_identity_invariants()
        return uid

    def _select_root_family(self) -> str:
        if cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET:
            return validate_bloodline_family(str(cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY))

        families = list(get_bloodline_families())
        mode = str(cfg.BRAIN.INITIAL_FAMILY_ASSIGNMENT).lower()
        if mode == "round_robin":
            family_id = families[self._initial_family_cursor % len(families)]
            self._initial_family_cursor += 1
            return family_id
        if mode == "weighted_random":
            weights = [float(cfg.BRAIN.INITIAL_FAMILY_WEIGHTS.get(family, 0.0)) for family in families]
            total = sum(weights)
            if total <= 0.0:
                raise ValueError("INITIAL_FAMILY_WEIGHTS must contain a positive total weight")
            return random.choices(families, weights=weights, k=1)[0]
        raise ValueError(f"Unsupported initial family assignment mode: {cfg.BRAIN.INITIAL_FAMILY_ASSIGNMENT}")

    def ensure_slot_brain_family(self, slot_idx: int, family_id: str):
        family_id = validate_bloodline_family(family_id)
        brain = self.brains[slot_idx]
        if brain is None or getattr(brain, "family_id", None) != family_id:
            brain = create_brain(family_id).to(self.device)
            brain.eval()
            self.brains[slot_idx] = brain
        return brain

    def get_uid_for_slot(self, slot_idx: int) -> int:
        return int(self.slot_uid[slot_idx].item())

    def get_slot_for_uid(self, uid: int) -> int | None:
        return self.active_uid_to_slot.get(uid)

    def is_uid_active(self, uid: int) -> bool:
        record = self.uid_lifecycle.get(uid)
        return bool(record and record.is_active)

    def get_parent_uid_for_slot(self, slot_idx: int) -> int:
        return int(self.slot_parent_uid[slot_idx].item())

    def get_parent_roles_for_uid(self, uid: int) -> dict[str, int]:
        return dict(self.uid_parent_roles.get(uid, {
            "brain_parent_uid": self.uid_lifecycle[uid].parent_uid,
            "trait_parent_uid": self.uid_lifecycle[uid].parent_uid,
            "anchor_parent_uid": self.uid_lifecycle[uid].parent_uid,
        }))

    def get_trait_latent_for_uid(self, uid: int) -> dict[str, float]:
        return dict(self.uid_trait_latent[uid])

    def get_family_for_uid(self, uid: int) -> str:
        family_id = self.uid_family.get(uid)
        if family_id is None:
            # Root seeds choose from the configured family assignment policy.
            # Parented births inherit the brain parent's family unless an
            # explicit family override has already been resolved upstream.
            raise KeyError(f"UID {uid} has no bloodline family binding")
        return family_id

    def get_family_for_slot(self, slot_idx: int) -> str | None:
        uid = self.get_uid_for_slot(slot_idx)
        if uid == -1:
            return None
        family_id = self.slot_family[slot_idx]
        if family_id is None:
            family_id = self.get_family_for_uid(uid)
            self.slot_family[slot_idx] = family_id
        return family_id

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
        if set(self.uid_family.keys()) != actual_uids:
            raise AssertionError("Bloodline ledger is missing allocated UIDs")
        if set(self.uid_parent_roles.keys()) != actual_uids:
            raise AssertionError("Parent-role ledger is missing allocated UIDs")
        if set(self.uid_trait_latent.keys()) != actual_uids:
            raise AssertionError("Trait-latent ledger is missing allocated UIDs")

        seen_active_uids = set()
        for slot_idx in range(self.max_agents):
            uid = self.get_uid_for_slot(slot_idx)
            alive = bool(self.data[self.ALIVE, slot_idx].item() > 0.5)
            if uid == -1:
                if alive:
                    raise AssertionError(f"Alive slot {slot_idx} has no canonical UID binding")
                if int(self.slot_parent_uid[slot_idx].item()) != -1:
                    raise AssertionError(f"Unbound slot {slot_idx} still has a parent UID binding")
                if self.slot_family[slot_idx] is not None:
                    raise AssertionError(f"Unbound slot {slot_idx} still has bloodline family state")
                continue
            if not alive:
                raise AssertionError(f"Dead slot {slot_idx} still owns active UID {uid}")
            if uid in seen_active_uids:
                raise AssertionError(f"UID {uid} is bound to more than one slot")
            seen_active_uids.add(uid)

            record = self.uid_lifecycle[uid]
            if not record.is_active or record.current_slot != slot_idx or record.death_tick is not None:
                raise AssertionError(f"Lifecycle record for UID {uid} disagrees with slot binding {slot_idx}")
            if self.active_uid_to_slot.get(uid) != slot_idx:
                raise AssertionError(f"UID {uid} is missing from the active UID-to-slot map")

            family_id = self.get_family_for_slot(slot_idx)
            if family_id != self.uid_family.get(uid):
                raise AssertionError(f"Bloodline family ledger disagrees for UID {uid} in slot {slot_idx}")
            brain = self.brains[slot_idx]
            if brain is None:
                raise AssertionError(f"Alive slot {slot_idx} has no instantiated brain")
            if getattr(brain, "family_id", None) != family_id:
                raise AssertionError(f"Alive slot {slot_idx} brain family mismatch for UID {uid}")

        if set(self.active_uid_to_slot.keys()) != seen_active_uids:
            raise AssertionError("Active UID-to-slot map disagrees with slot bindings")

    def spawn_initial_population(self, grid):
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
        family_id: str | None = None,
        parent_roles: dict[str, int] | None = None,
        trait_latent: dict[str, float] | None = None,
        birth_hp: float | None = None,
    ) -> int:
        _ = assign_new_identity
        if idx < 0 or idx >= self.max_agents:
            raise IndexError(f"Slot index {idx} is out of bounds for {self.max_agents} agents")
        if self.get_uid_for_slot(idx) != -1:
            raise AssertionError(f"Slot {idx} must be unbound before a new UID can be attached")
        if self.data[self.ALIVE, idx] > 0.5:
            raise AssertionError(f"Slot {idx} is already alive")

        if tick_born is None:
            tick_born = self.tick_counter

        if parent_roles is None:
            parent_roles = {
                "brain_parent_uid": int(parent_uid),
                "trait_parent_uid": int(parent_uid),
                "anchor_parent_uid": int(parent_uid),
            }
        canonical_parent_uid = int(parent_roles["brain_parent_uid"])

        uid = self.allocate_uid(canonical_parent_uid, int(tick_born))
        self.bind_uid_to_slot(uid, idx)

        if family_id is None:
            if canonical_parent_uid != -1 and canonical_parent_uid in self.uid_family:
                family_id = self.uid_family[canonical_parent_uid]
            else:
                family_id = self._select_root_family()
        family_id = validate_bloodline_family(family_id)
        self.uid_family[uid] = family_id
        self.slot_family[idx] = family_id
        self.uid_parent_roles[uid] = {
            "brain_parent_uid": int(parent_roles["brain_parent_uid"]),
            "trait_parent_uid": int(parent_roles["trait_parent_uid"]),
            "anchor_parent_uid": int(parent_roles["anchor_parent_uid"]),
        }

        if trait_latent is None:
            trait_latent = default_trait_latent()
        self.uid_trait_latent[uid] = dict(trait_latent)

        generation_depth = 0
        brain_parent_uid = int(parent_roles["brain_parent_uid"])
        if brain_parent_uid != -1 and brain_parent_uid in self.uid_generation_depth:
            generation_depth = self.uid_generation_depth[brain_parent_uid] + 1
        self.uid_generation_depth[uid] = generation_depth

        if traits is None:
            mapped = trait_values_from_latent(trait_latent)
            traits = {
                "mass": mapped["mass"],
                "hp_max": mapped["hp_max"],
                "vision": mapped["vision"],
                "metab": mapped["metab"],
            }

        self.data[self.ALIVE, idx] = 1.0
        self.data[self.X, idx] = float(x)
        self.data[self.Y, idx] = float(y)
        self.data[self.MASS, idx] = float(traits["mass"])
        self.data[self.HP_MAX, idx] = float(traits["hp_max"])
        self.data[self.VISION, idx] = float(traits["vision"])
        self.data[self.METABOLISM_RATE, idx] = float(traits["metab"])
        self.data[self.HP, idx] = float(self.data[self.HP_MAX, idx].item() if birth_hp is None else birth_hp)
        self.data[self.LAST_ACTION, idx] = 0.0
        self.data[self.TICK_BORN, idx] = float(tick_born)
        self.data[self.HP_GAINED, idx] = 0.0
        self.data[self.HP_LOST_PHYSICS, idx] = 0.0
        self.data[self.OPTIMIZATION_CYCLE, idx] = 0.0

        self.ensure_slot_brain_family(idx, family_id)
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
