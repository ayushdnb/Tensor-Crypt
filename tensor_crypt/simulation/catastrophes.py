"""
Prompt 6 catastrophe scheduler and reversible world-shock system.

Design doctrine:
- catastrophes are explicit bounded events
- baseline zone definitions remain canonical and are never permanently
  mutated by temporary world shocks
- apply/update/revert is implemented by rebuilding transient field and
  runtime multipliers every tick from active event state
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from ..config_bridge import cfg


_CATASTROPHE_ORDER = [
    "ashfall_of_nocthar",
    "sanguine_bloom",
    "the_woundtide",
    "the_hollow_fast",
    "mirror_of_thorns",
    "veil_of_somnyr",
    "graveweight",
    "glass_requiem",
    "the_witchstorm",
    "the_thorn_march",
    "the_barren_hymn",
    "crimson_deluge",
]

_DISPLAY_NAMES = {
    "ashfall_of_nocthar": "Ashfall of Nocthar",
    "sanguine_bloom": "Sanguine Bloom",
    "the_woundtide": "The Woundtide",
    "the_hollow_fast": "The Hollow Fast",
    "mirror_of_thorns": "Mirror of Thorns",
    "veil_of_somnyr": "Veil of Somnyr",
    "graveweight": "Graveweight",
    "glass_requiem": "Glass Requiem",
    "the_witchstorm": "The Witchstorm",
    "the_thorn_march": "The Thorn March",
    "the_barren_hymn": "The Barren Hymn",
    "crimson_deluge": "Crimson Deluge",
}

_TECHNICAL_CLASSES = {
    "ashfall_of_nocthar": "zone_collapse",
    "sanguine_bloom": "poison_bloom",
    "the_woundtide": "blight_wave",
    "the_hollow_fast": "healing_drought",
    "mirror_of_thorns": "zone_inversion",
    "veil_of_somnyr": "fog_epoch",
    "graveweight": "heavy_world",
    "glass_requiem": "glass_world",
    "the_witchstorm": "mutation_storm",
    "the_thorn_march": "border_creep",
    "the_barren_hymn": "sterility_window",
    "crimson_deluge": "scarlet_flood",
}


@dataclass
class ActiveCatastrophe:
    event_id: int
    catastrophe_id: str
    display_name: str
    technical_class: str
    start_tick: int
    end_tick: int
    manual: bool
    params: dict[str, Any]

    @property
    def remaining_ticks(self) -> int:
        return max(0, self.end_tick - self.start_tick)


class CatastropheManager:
    def __init__(self, *, grid, registry, physics, perception, respawn_controller, logger=None):
        self.grid = grid
        self.registry = registry
        self.physics = physics
        self.perception = perception
        self.respawn_controller = respawn_controller
        self.logger = logger

        self.mode = str(cfg.CATASTROPHE.DEFAULT_MODE)
        self.last_auto_mode = self.mode if self.mode.startswith("auto_") else "auto_dynamic"
        self.scheduler_paused = False
        self.auto_enabled = self.mode in {"auto_dynamic", "auto_static"}

        self._rng = random.Random(int(cfg.SIM.SEED) + int(cfg.CATASTROPHE.RNG_STREAM_OFFSET))
        self._next_auto_tick: int | None = None
        self._static_cursor: int = 0
        self._event_counter: int = 0
        self._visual_state_version: int = 0
        self._active: list[ActiveCatastrophe] = []
        self._last_status_cache_tick: int | None = None
        self._last_status_cache: dict[str, Any] | None = None

        self._plan_next_auto_tick(0)

    def _invalidate_status_cache(self) -> None:
        self._last_status_cache_tick = None
        self._last_status_cache = None

    def reset(self) -> None:
        self.mode = str(cfg.CATASTROPHE.DEFAULT_MODE)
        self.last_auto_mode = self.mode if self.mode.startswith("auto_") else "auto_dynamic"
        self.scheduler_paused = False
        self.auto_enabled = self.mode in {"auto_dynamic", "auto_static"}
        self._rng = random.Random(int(cfg.SIM.SEED) + int(cfg.CATASTROPHE.RNG_STREAM_OFFSET))
        self._next_auto_tick = None
        self._static_cursor = 0
        self._event_counter = 0
        self._active = []
        self._visual_state_version += 1
        self._invalidate_status_cache()
        self._plan_next_auto_tick(0)

    @property
    def roster_ids(self) -> list[str]:
        return list(_CATASTROPHE_ORDER)

    def cycle_mode(self) -> str:
        modes = ["off", "manual_only", "auto_dynamic", "auto_static"]
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        self.auto_enabled = self.mode in {"auto_dynamic", "auto_static"}
        if self.auto_enabled:
            self.last_auto_mode = self.mode
        self._invalidate_status_cache()
        self._plan_next_auto_tick(0)
        return self.mode

    def toggle_auto_enable(self) -> str:
        if self.mode in {"auto_dynamic", "auto_static"}:
            self.last_auto_mode = self.mode
            self.mode = "manual_only"
            self.auto_enabled = False
        else:
            self.mode = self.last_auto_mode
            self.auto_enabled = True
        self._invalidate_status_cache()
        if self.auto_enabled:
            self._plan_next_auto_tick(0)
        return self.mode

    def toggle_scheduler_pause(self) -> bool:
        self.scheduler_paused = not self.scheduler_paused
        self._invalidate_status_cache()
        return self.scheduler_paused

    def set_mode(self, mode: str) -> None:
        if mode not in {"off", "manual_only", "auto_dynamic", "auto_static"}:
            raise ValueError(f"Unsupported catastrophe mode: {mode}")
        self.mode = str(mode)
        self.auto_enabled = self.mode in {"auto_dynamic", "auto_static"}
        if self.auto_enabled:
            self.last_auto_mode = self.mode
        self._invalidate_status_cache()
        self._plan_next_auto_tick(0)

    def _enabled_ids(self) -> list[str]:
        return [cat_id for cat_id in _CATASTROPHE_ORDER if cfg.CATASTROPHE.TYPE_ENABLED.get(cat_id, False)]

    def _duration_for(self, catastrophe_id: str) -> int:
        duration = cfg.CATASTROPHE.PER_TYPE_DURATION_TICKS.get(catastrophe_id, cfg.CATASTROPHE.DEFAULT_DURATION_TICKS)
        duration = max(cfg.CATASTROPHE.MIN_DURATION_TICKS, min(cfg.CATASTROPHE.MAX_DURATION_TICKS, int(duration)))
        if self.mode == "auto_dynamic" and cfg.CATASTROPHE.AUTO_DYNAMIC_SAMPLE_DURATION:
            jitter = self._rng.randint(-max(1, duration // 5), max(1, duration // 5))
            duration = max(cfg.CATASTROPHE.MIN_DURATION_TICKS, min(cfg.CATASTROPHE.MAX_DURATION_TICKS, duration + jitter))
        return int(duration)

    def _sample_dynamic_gap(self) -> int:
        low = int(cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS)
        high = int(cfg.CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS)
        if high < low:
            raise ValueError("AUTO_DYNAMIC_GAP_MAX_TICKS must be >= AUTO_DYNAMIC_GAP_MIN_TICKS")
        return self._rng.randint(low, high)

    def _select_dynamic_type(self) -> str | None:
        enabled = self._enabled_ids()
        if not enabled:
            return None
        weights = [float(cfg.CATASTROPHE.TYPE_SELECTION_WEIGHTS.get(cat_id, 0.0)) for cat_id in enabled]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(enabled)
        return self._rng.choices(enabled, weights=weights, k=1)[0]

    def _select_static_type(self) -> str | None:
        enabled = self._enabled_ids()
        if not enabled:
            return None

        policy = str(cfg.CATASTROPHE.AUTO_STATIC_ORDERING_POLICY)
        if policy == "configured_sequence":
            sequence = [cat_id for cat_id in cfg.CATASTROPHE.AUTO_STATIC_SEQUENCE if cat_id in enabled]
            if not sequence:
                sequence = enabled
            cat_id = sequence[self._static_cursor % len(sequence)]
            self._static_cursor += 1
            return cat_id
        if policy == "fixed_priority":
            return enabled[0]

        cat_id = enabled[self._static_cursor % len(enabled)]
        self._static_cursor += 1
        return cat_id

    def _plan_next_auto_tick(self, current_tick: int) -> None:
        if self.mode == "auto_dynamic":
            self._next_auto_tick = int(current_tick) + self._sample_dynamic_gap()
        elif self.mode == "auto_static":
            self._next_auto_tick = int(current_tick) + int(cfg.CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS)
        else:
            self._next_auto_tick = None

    def _candidate_zone_ids(self, *, positive_only: bool = False) -> list[int]:
        zone_ids = []
        for zone in self.grid.hzones:
            if not zone.get("active", True):
                continue
            if positive_only and float(zone["rate"]) <= 0.0:
                continue
            zone_ids.append(int(zone["id"]))
        return zone_ids

    def _pick_zone_ids(self, fraction: float, *, positive_only: bool = False) -> list[int]:
        zone_ids = self._candidate_zone_ids(positive_only=positive_only)
        if not zone_ids:
            return []
        count = max(1, min(len(zone_ids), int(round(len(zone_ids) * max(0.0, min(1.0, fraction))))))
        self._rng.shuffle(zone_ids)
        return zone_ids[:count]

    def _make_crimson_patches(self) -> list[tuple[int, int, int, int]]:
        params = cfg.CATASTROPHE.TYPE_PARAMS["crimson_deluge"]
        patch_count = max(1, int(params.get("patch_count", 3.0)))
        patch_frac = max(0.05, float(params.get("patch_size_fraction", 0.18)))
        patch_w = max(2, int(round(self.grid.W * patch_frac)))
        patch_h = max(2, int(round(self.grid.H * patch_frac)))
        patches = []
        for _ in range(patch_count):
            x1 = self._rng.randint(1, max(1, self.grid.W - patch_w - 2))
            y1 = self._rng.randint(1, max(1, self.grid.H - patch_h - 2))
            patches.append((x1, y1, min(self.grid.W - 2, x1 + patch_w), min(self.grid.H - 2, y1 + patch_h)))
        return patches

    def _make_event_params(self, catastrophe_id: str) -> dict[str, Any]:
        params = dict(cfg.CATASTROPHE.TYPE_PARAMS.get(catastrophe_id, {}))

        if catastrophe_id == "ashfall_of_nocthar":
            params["zone_ids"] = self._pick_zone_ids(float(params.get("positive_zone_fraction", 0.65)), positive_only=True)
        elif catastrophe_id == "sanguine_bloom":
            params["zone_ids"] = self._pick_zone_ids(float(params.get("zone_fraction", 0.45)), positive_only=True)
        elif catastrophe_id == "mirror_of_thorns":
            params["zone_ids"] = self._pick_zone_ids(float(params.get("zone_fraction", 0.50)), positive_only=False)
        elif catastrophe_id == "the_woundtide":
            params["direction"] = self._rng.choice(["left_to_right", "right_to_left"])
        elif catastrophe_id == "the_thorn_march":
            params["center_x"] = (self.grid.W - 1) / 2.0
            params["center_y"] = (self.grid.H - 1) / 2.0
        elif catastrophe_id == "crimson_deluge":
            params["patches"] = self._make_crimson_patches()

        return params

    def _emit_event(self, payload: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.log_catastrophe_event(payload)

    def _start_event(self, catastrophe_id: str, tick: int, *, manual: bool) -> bool:
        if catastrophe_id not in _CATASTROPHE_ORDER:
            raise ValueError(f"Unknown catastrophe id: {catastrophe_id}")
        if not cfg.CATASTROPHE.ENABLED:
            return False
        if not cfg.CATASTROPHE.TYPE_ENABLED.get(catastrophe_id, False):
            return False
        if self.mode == "off":
            return False
        if manual and not cfg.CATASTROPHE.MANUAL_TRIGGER_ENABLED:
            return False
        if (not cfg.CATASTROPHE.ALLOW_OVERLAP) and self._active:
            return False
        if len(self._active) >= int(cfg.CATASTROPHE.MAX_CONCURRENT):
            return False

        duration = self._duration_for(catastrophe_id)
        self._event_counter += 1
        event = ActiveCatastrophe(
            event_id=self._event_counter,
            catastrophe_id=catastrophe_id,
            display_name=_DISPLAY_NAMES[catastrophe_id],
            technical_class=_TECHNICAL_CLASSES[catastrophe_id],
            start_tick=int(tick),
            end_tick=int(tick) + duration,
            manual=bool(manual),
            params=self._make_event_params(catastrophe_id),
        )
        self._active.append(event)
        self._visual_state_version += 1
        self._invalidate_status_cache()

        self._emit_event(
            {
                "tick": int(tick),
                "kind": "start",
                "event_id": event.event_id,
                "catastrophe_id": event.catastrophe_id,
                "display_name": event.display_name,
                "technical_class": event.technical_class,
                "manual": event.manual,
                "start_tick": event.start_tick,
                "end_tick": event.end_tick,
            }
        )

        if self.mode in {"auto_dynamic", "auto_static"}:
            self._plan_next_auto_tick(int(tick))
        return True

    def manual_trigger_by_index(self, roster_index: int, tick: int) -> bool:
        if roster_index < 0 or roster_index >= len(_CATASTROPHE_ORDER):
            return False
        return self._start_event(_CATASTROPHE_ORDER[roster_index], tick, manual=True)

    def manual_clear(self, tick: int) -> int:
        if not cfg.CATASTROPHE.MANUAL_CLEAR_ENABLED:
            return 0
        cleared = 0
        for event in list(self._active):
            self._emit_event(
                {
                    "tick": int(tick),
                    "kind": "clear",
                    "event_id": event.event_id,
                    "catastrophe_id": event.catastrophe_id,
                    "display_name": event.display_name,
                    "technical_class": event.technical_class,
                    "manual": event.manual,
                    "start_tick": event.start_tick,
                    "end_tick": event.end_tick,
                }
            )
            self._active.remove(event)
            cleared += 1
        if cleared:
            self._visual_state_version += 1
            self._invalidate_status_cache()
        return cleared

    def pre_tick(self, tick: int) -> None:
        expired = [event for event in self._active if int(tick) >= event.end_tick]
        if expired:
            for event in expired:
                self._emit_event(
                    {
                        "tick": int(tick),
                        "kind": "end",
                        "event_id": event.event_id,
                        "catastrophe_id": event.catastrophe_id,
                        "display_name": event.display_name,
                        "technical_class": event.technical_class,
                        "manual": event.manual,
                        "start_tick": event.start_tick,
                        "end_tick": event.end_tick,
                    }
                )
                self._active.remove(event)
            self._visual_state_version += 1
            self._invalidate_status_cache()

        if not cfg.CATASTROPHE.ENABLED:
            return
        if self.mode not in {"auto_dynamic", "auto_static"}:
            return
        if self.scheduler_paused:
            return
        if self._next_auto_tick is None or int(tick) < int(self._next_auto_tick):
            return
        if (not cfg.CATASTROPHE.ALLOW_OVERLAP) and self._active:
            return
        if len(self._active) >= int(cfg.CATASTROPHE.MAX_CONCURRENT):
            return

        catastrophe_id = self._select_dynamic_type() if self.mode == "auto_dynamic" else self._select_static_type()
        if catastrophe_id is not None:
            self._start_event(catastrophe_id, tick, manual=False)

    def apply_world_overrides(self, tick: int) -> None:
        self.physics.reset_runtime_modifiers()
        self.perception.reset_runtime_modifiers()
        self.respawn_controller.reset_runtime_modifiers()

        field = self.grid.grid[1].clone()
        vision_scale = 1.0
        collision_scalar = 1.0
        metabolism_scalar = 1.0
        mass_burden = 0.0
        reproduction_enabled = True
        mutation_overrides: dict[str, float] = {}


        for event in self._active:
            params = event.params
            progress = 0.0
            if event.end_tick > event.start_tick:
                progress = max(0.0, min(1.0, (int(tick) - event.start_tick) / (event.end_tick - event.start_tick)))

            if event.catastrophe_id == "ashfall_of_nocthar":
                for zone_id in params.get("zone_ids", []):
                    zone = self.grid.get_hzone(zone_id)
                    if zone and float(zone["rate"]) > 0.0:
                        field[zone["y1"] : zone["y2"] + 1, zone["x1"] : zone["x2"] + 1] = 0.0

            elif event.catastrophe_id == "sanguine_bloom":
                neg_rate = float(params.get("negative_rate", -1.5))
                for zone_id in params.get("zone_ids", []):
                    zone = self.grid.get_hzone(zone_id)
                    if zone:
                        field[zone["y1"] : zone["y2"] + 1, zone["x1"] : zone["x2"] + 1] = neg_rate

            elif event.catastrophe_id == "the_woundtide":
                half_width = max(1, int(round(float(params.get("front_half_width", 5.0)))))
                neg_rate = float(params.get("negative_rate", -2.0))
                if params.get("direction") == "right_to_left":
                    center_x = int(round((self.grid.W - 1) * (1.0 - progress)))
                else:
                    center_x = int(round((self.grid.W - 1) * progress))
                x1 = max(1, center_x - half_width)
                x2 = min(self.grid.W - 2, center_x + half_width)
                field[:, x1 : x2 + 1] = neg_rate

            elif event.catastrophe_id == "the_hollow_fast":
                positive_scalar = float(params.get("positive_scalar", 0.25))
                field = torch.where(field > 0.0, field * positive_scalar, field)

            elif event.catastrophe_id == "mirror_of_thorns":
                for zone_id in params.get("zone_ids", []):
                    zone = self.grid.get_hzone(zone_id)
                    if zone:
                        region = field[zone["y1"] : zone["y2"] + 1, zone["x1"] : zone["x2"] + 1]
                        field[zone["y1"] : zone["y2"] + 1, zone["x1"] : zone["x2"] + 1] = -region

            elif event.catastrophe_id == "veil_of_somnyr":
                vision_scale = min(vision_scale, float(params.get("vision_scalar", 0.45)))

            elif event.catastrophe_id == "graveweight":
                metabolism_scalar = max(metabolism_scalar, float(params.get("metabolism_scalar", 1.65)))
                mass_burden = max(mass_burden, float(params.get("mass_burden_scalar", 0.06)))

            elif event.catastrophe_id == "glass_requiem":
                collision_scalar = max(collision_scalar, float(params.get("collision_damage_scalar", 1.8)))

            elif event.catastrophe_id == "the_witchstorm":
                for key in ("trait_sigma_scalar", "budget_sigma_scalar", "policy_noise_scalar", "rare_prob_scalar", "family_shift_scalar"):
                    mutation_overrides[key] = max(float(mutation_overrides.get(key, 1.0)), float(params.get(key, 1.0)))

            elif event.catastrophe_id == "the_thorn_march":
                max_shrink_fraction = max(0.0, min(0.45, float(params.get("max_shrink_fraction", 0.35))))
                shrink = max_shrink_fraction * progress
                x_margin = int(round((self.grid.W - 2) * shrink / 2.0))
                y_margin = int(round((self.grid.H - 2) * shrink / 2.0))
                x1 = 1 + x_margin
                x2 = self.grid.W - 2 - x_margin
                y1 = 1 + y_margin
                y2 = self.grid.H - 2 - y_margin
                mask = torch.ones_like(field, dtype=torch.bool)
                mask[y1 : y2 + 1, x1 : x2 + 1] = False
                field = torch.where(mask, torch.tensor(float(params.get("negative_rate", -2.0)), device=field.device), field)

            elif event.catastrophe_id == "the_barren_hymn":
                reproduction_enabled = False

            elif event.catastrophe_id == "crimson_deluge":
                neg_rate = float(params.get("negative_rate", -2.5))
                for x1, y1, x2, y2 in params.get("patches", []):
                    field[y1 : y2 + 1, x1 : x2 + 1] = neg_rate

        field = torch.clamp(field, -cfg.GRID.HZ_SUM_CLAMP, cfg.GRID.HZ_SUM_CLAMP)
        self.grid.grid[1] = field

        self.physics.set_runtime_modifiers(
            collision_damage_multiplier=collision_scalar,
            metabolism_multiplier=metabolism_scalar,
            mass_metabolism_burden=mass_burden,
        )
        self.perception.set_runtime_modifiers(vision_scale=vision_scale)
        self.respawn_controller.set_runtime_modifiers(
            reproduction_enabled=reproduction_enabled,
            mutation_overrides=mutation_overrides,
        )

        self._last_status_cache = None
        self._last_status_cache_tick = None

    def build_status(self, tick: int) -> dict[str, Any]:
        if self._last_status_cache_tick == int(tick) and self._last_status_cache is not None:
            return dict(self._last_status_cache)

        active_details = []
        for event in self._active:
            active_details.append(
                {
                    "event_id": event.event_id,
                    "catastrophe_id": event.catastrophe_id,
                    "display_name": event.display_name,
                    "technical_class": event.technical_class,
                    "remaining_ticks": max(0, event.end_tick - int(tick)),
                    "end_tick": event.end_tick,
                }
            )

        thorn_march_safe_rect = None
        woundtide_front_x = None
        for event in self._active:
            if event.catastrophe_id == "the_thorn_march":
                progress = max(0.0, min(1.0, (int(tick) - event.start_tick) / max(1, event.end_tick - event.start_tick)))
                max_shrink_fraction = max(0.0, min(0.45, float(event.params.get("max_shrink_fraction", 0.35))))
                shrink = max_shrink_fraction * progress
                x_margin = int(round((self.grid.W - 2) * shrink / 2.0))
                y_margin = int(round((self.grid.H - 2) * shrink / 2.0))
                thorn_march_safe_rect = [1 + x_margin, 1 + y_margin, self.grid.W - 2 - x_margin, self.grid.H - 2 - y_margin]
            elif event.catastrophe_id == "the_woundtide":
                progress = max(0.0, min(1.0, (int(tick) - event.start_tick) / max(1, event.end_tick - event.start_tick)))
                if event.params.get("direction") == "right_to_left":
                    woundtide_front_x = int(round((self.grid.W - 1) * (1.0 - progress)))
                else:
                    woundtide_front_x = int(round((self.grid.W - 1) * progress))

        status = {
            "mode": self.mode,
            "scheduler_paused": self.scheduler_paused,
            "auto_enabled": self.auto_enabled,
            "next_auto_tick": self._next_auto_tick,
            "active_count": len(self._active),
            "active_names": [event.display_name for event in self._active],
            "active_details": active_details,
            "visual_state_version": self._visual_state_version,
            "thorn_march_safe_rect": thorn_march_safe_rect,
            "woundtide_front_x": woundtide_front_x,
        }
        self._last_status_cache_tick = int(tick)
        self._last_status_cache = dict(status)
        return status

    def serialize(self) -> dict[str, Any]:
        return {
            "schema_version": cfg.SCHEMA.CATASTROPHE_SCHEMA_VERSION,
            "mode": self.mode,
            "last_auto_mode": self.last_auto_mode,
            "scheduler_paused": self.scheduler_paused,
            "auto_enabled": self.auto_enabled,
            "next_auto_tick": self._next_auto_tick,
            "static_cursor": self._static_cursor,
            "event_counter": self._event_counter,
            "visual_state_version": self._visual_state_version,
            "active": [
                {
                    "event_id": event.event_id,
                    "catastrophe_id": event.catastrophe_id,
                    "display_name": event.display_name,
                    "technical_class": event.technical_class,
                    "start_tick": event.start_tick,
                    "end_tick": event.end_tick,
                    "manual": event.manual,
                    "params": event.params,
                }
                for event in self._active
            ],
            "rng_state": self._rng.getstate(),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.mode = str(payload["mode"])
        self.last_auto_mode = str(payload.get("last_auto_mode", self.mode))
        self.scheduler_paused = bool(payload.get("scheduler_paused", False))
        self.auto_enabled = bool(payload.get("auto_enabled", self.mode in {"auto_dynamic", "auto_static"}))
        self._next_auto_tick = payload.get("next_auto_tick")
        self._static_cursor = int(payload.get("static_cursor", 0))
        self._event_counter = int(payload.get("event_counter", 0))
        self._visual_state_version = int(payload.get("visual_state_version", 0))
        self._active = [
            ActiveCatastrophe(
                event_id=int(item["event_id"]),
                catastrophe_id=str(item["catastrophe_id"]),
                display_name=str(item["display_name"]),
                technical_class=str(item["technical_class"]),
                start_tick=int(item["start_tick"]),
                end_tick=int(item["end_tick"]),
                manual=bool(item["manual"]),
                params=dict(item.get("params", {})),
            )
            for item in payload.get("active", [])
        ]
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._invalidate_status_cache()