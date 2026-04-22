"""Canonical operational configuration for Tensor Crypt.

This file is the single authoritative control surface for simulation semantics,
training cadence, checkpointing, telemetry, validation, and viewer behavior.

This rewrite preserves the original defaults and the original public class /
field names, but expands the file into a much deeper operator manual.

Audit legend used throughout this file
--------------------------------------
- ACTIVE RUNTIME KNOB:
  The uploaded repository dump contains direct reads of this surface.
- GUARDED COMPATIBILITY SURFACE:
  The surface exists publicly, but current runtime validation only accepts a
  restricted subset of values. Unsupported values are rejected.
- CURRENTLY UNREAD / EFFECTIVELY DEAD:
  No direct `.<SECTION>.<FIELD>` runtime read was found in the uploaded dump.
  This is an honest repository-grounded marker, not a stylistic opinion.
  Such knobs may still exist for future work, historical reasons, or external
  tooling, but they do not appear to drive the current in-repo runtime.
- HIGH-RISK SCHEMA KNOB:
  These are shape / schema / checkpoint / compatibility surfaces. Change them
  only as part of a deliberate migration.

Operational policy
------------------
- Defaults favor determinism, observability, checkpoint safety, and explicit
  validation over maximum raw throughput.
- This root module remains the authoritative configuration source, and package
  modules consume it through the repository's config bridge.
- Comments below explain what each knob does, what values are meaningful, and
  what practical effect a change is expected to have in the current codebase.

IMPORTANT
---------
This file is documentation-heavy on purpose. The comments are long because this
module is intended to double as a deep experiment manual for operators.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class SimConfig:
    """Core session controls.

    This section decides the seed, device placement, and a few top-level runtime
    behaviors that affect the whole simulation session. These are the first knobs
    an operator should check when moving between laptop debugging, workstation
    training, or reproducibility-focused validation work.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Master deterministic seed.
    # The runtime seeds `torch`, `random`, `numpy`, and CUDA seed-all from this value.
    # Change it when you want a different but still reproducible world history.
    SEED: int = 42
    #
    # CURRENT STATUS: active runtime knob.
    # Primary execution device string.
    # Typical values are `"cpu"`, `"cuda"`, or an explicit CUDA device such as `"cuda:0"`.
    # Forcing CUDA on a machine without CUDA availability is rejected during runtime validation.
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Non-supported non-default values are rejected during runtime validation rather than being silently accepted.
    # Numeric dtype surface for the simulation.
    # In the current repository, runtime validation accepts only `"float32"`.
    # Treat this as a compatibility marker rather than a free tuning knob.
    DTYPE: str = "float32"  # Guarded compatibility surface; the runtime currently supports only float32.
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.SIM.TICKS_PER_SEC` runtime read was found in the code dump.
    # Viewer cadence / pacing hint.
    # In the uploaded code dump no live runtime read was found, so changing this currently has no
    # observed effect.
    TICKS_PER_SEC: int = 30
    #
    # CURRENT STATUS: active runtime knob.
    # Optional automatic stop threshold.
    # Set `0` to keep the session open until the operator exits.
    # Set a positive integer to stop after that many engine ticks if the viewer / runtime path reads
    # it.
    MAX_TICKS: int = 0  # Optional viewer/runtime auto-stop; 0 keeps the session open until the operator exits.
    #
    # CURRENT STATUS: active runtime knob.
    # Controls reuse of the dense action scratch tensor.
    # Keeping this enabled reduces per-tick allocations and is the efficient default.
    REUSE_ACTION_BUFFER: bool = True  # Reuse the dense sparse-action tensor instead of reallocating it every tick.
    #
    # CURRENT STATUS: experimental runtime knob.
    # Enables same-family inference batching via torch.func.
    # This remains opt-in because real ROI depends on family bucket sizes and the
    # local PyTorch build. The canonical loop remains the default.
    EXPERIMENTAL_FAMILY_VMAP_INFERENCE: bool = False
    #
    # CURRENT STATUS: experimental runtime knob.
    # Minimum same-family alive bucket size required before the engine attempts
    # the torch.func fast path. Smaller buckets stay on the canonical per-brain loop.
    EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET: int = 8


@dataclass
class GridConfig:
    """World-field and substrate controls.

    These knobs define the size of the world tensor and how heal / harm zones are
    combined into the field channel. They change the geometry and field behavior of
    the arena rather than agent learning logic directly.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # World width in cells.
    # Larger values widen the arena, usually reducing encounter density and increasing memory /
    # compute cost.
    W: int = 100
    #
    # CURRENT STATUS: active runtime knob.
    # World height in cells.
    # Larger values deepen the arena, with similar sparsity and cost implications as increasing
    # width.
    H: int = 100
    #
    # CURRENT STATUS: active runtime knob.
    # Heal/harm zone overlap-composition mode.
    # Documented options are `"max_abs"`, `"sum_clamped"`, and `"last_wins"`.
    # Use caution: this changes how overlapping zone fields combine.
    HZ_OVERLAP_MODE: str = "max_abs"  # max_abs | sum_clamped | last_wins
    #
    # CURRENT STATUS: active runtime knob.
    # Absolute clamp used when summed zone fields are limited.
    # Increase it to allow stronger accumulated zone intensity; decrease it to keep the field milder.
    HZ_SUM_CLAMP: float = 5.0
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the zone field is rebuilt / cleared each tick before applying transient effects.
    # Keeping this true is the safer baseline for reversible catastrophe overlays.
    HZ_CLEAR_EACH_TICK: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.GRID.EXPOSE_H_GRAD` runtime read was found in the code dump.
    # Gradient/diagnostic field exposure switch.
    # No live runtime read was found in the uploaded dump, so this currently behaves like an unread
    # compatibility surface.
    EXPOSE_H_GRAD: bool = False


@dataclass
class MapgenConfig:
    """Procedural map-generation controls.

    These values determine the density and scale of random walls and heal zones
    created when a new run is built from scratch.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Number of random wall segments requested during map generation.
    # Overnight profile choice: reduce this from 9 to 6 so the arena is less maze-like.
    # The default was rejected because the current one-cell-thick wandering segments can consume a surprising
    # amount of traversable space on a 100x100 map and increase accidental wall attrition before policies have
    # learned anything useful.
    # More aggressive reductions were rejected because an almost empty map tends to collapse into low-contact
    # wandering rather than visible interaction.
    RANDOM_WALLS: int = 6
    #
    # CURRENT STATUS: active runtime knob.
    # Minimum wall segment length.
    # Lowered from 34 to 20 to keep obstacles present but less dominating.
    # Expected effect: fewer long dead-end corridors, more recoverable movement paths, and fewer incidental wall
    # hits from immature policies.
    # Tradeoff: slightly less positional pressure and slightly less forced pathing variety.
    WALL_SEG_MIN: int = 20
    #
    # CURRENT STATUS: active runtime knob.
    # Maximum wall segment length.
    # Lowered from 83 to 52 for the same reason: the default upper tail can over-constrain the map in a way that
    # amplifies collision damage and local starvation.
    # A more conservative value near the original default was rejected because it preserved too much of the
    # obstructive regime; a much smaller value was rejected because it would make the map overly open.
    WALL_SEG_MAX: int = 52
    #
    # CURRENT STATUS: active runtime knob.
    # Margin used when keeping walls away from protected regions / edges during generation.
    # Nudged from 4 to 5 so the reduced wall set also leaves slightly cleaner breathing room near boundaries.
    # This is a mild operator-comfort and movement-stability adjustment, not a semantic rewrite.
    WALL_AVOID_MARGIN: int = 5
    #
    # CURRENT STATUS: active runtime knob.
    # Number of heal zones requested during procedural generation.
    # Raised from 20 to 28 so the baseline world contains more survivable refugia and fewer sterile deserts.
    # Expected effect: better overnight persistence, more local ecological niches, and less dependence on rare
    # lucky pathing for first-generation survival.
    # More aggressive zone counts were rejected because they risk turning the map into a near-everywhere safe
    # field with weak selection pressure.
    HEAL_ZONE_COUNT: int = 28
    #
    # CURRENT STATUS: active runtime knob.
    # Zone size as a fraction-like ratio of map size.
    # Raised from 10/256 (~0.039) to 0.05 so each refuge is slightly easier to rediscover and exploit.
    # On a 100x100 map this moves the nominal generated patch size from roughly 4x4 to roughly 5x5 cells.
    # Tradeoff: larger zones reduce starvation pressure; the compensating choice here is to keep the heal rate
    # below the original 0.5 rather than making zones both larger and equally strong.
    HEAL_ZONE_SIZE_RATIO: float = 0.05
    #
    # CURRENT STATUS: active runtime knob.
    # Base positive heal-zone rate used by generated zones.
    # Lowered slightly from 0.5 to 0.42 while zone coverage is increased.
    # This specific combination was chosen to widen refuge availability without making a small number of tiles so
    # strong that they trivialize survival once found.
    # A more conservative 0.5 was rejected because larger and more numerous zones at full strength risked overly
    # safe camping; a much lower rate was rejected because it would not offset overnight metabolism/physics loss.
    HEAL_RATE: float = 0.42


@dataclass
class AgentsConfig:
    """Population slot-capacity and spawn-surface controls.

    These knobs define how many dense runtime slots exist and whether the runtime
    permits multiple live agents to occupy the same tile.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Dense slot capacity and initial seed-population target.
    # Raised from 100 to 160 because this repository uses `AGENTS.N` as both the registry slot count and the
    # initial seed-population target.
    # The default was rejected because it underuses the available runtime headroom of the supplied machine and,
    # more importantly, keeps the live interaction surface thin on a 100x100 map.
    # Expected effect: more concurrent contact opportunities, more lineage turnover, and more visible overnight
    # dynamics without moving into reckless crowding.
    # Tradeoff: more per-tick compute, larger optimizer/checkpoint state, and reduced compatibility with old runs
    # that were created under a different slot width.
    N: int = 160
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Non-supported non-default values are rejected during runtime validation rather than being silently accepted.
    # Initial spawn pattern selector.
    # The current runtime validates only `"uniform"`; alternate values are rejected instead of
    # silently ignored.
    SPAWN_MODE: str = "uniform"  # Guarded compatibility surface; only uniform spawn is currently implemented.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether multiple live agents may occupy the same tile.
    # Keeping this true enforces single-occupancy invariants and grid consistency checks.
    NO_STACKING: bool = True


@dataclass
class RespawnCrowdingOverlayConfig:
    """The Ashen Press (crowding-gated reproduction overlay).

    This overlay evaluates local population density around the anchor parent
    before offspring placement is attempted.
    """

    #
    # CURRENT STATUS: active runtime knob.
    # Master enable for The Ashen Press.
    # Enabled for the overnight profile because local anti-clumping is useful once slot capacity is increased and
    # floor recovery becomes more capable.
    # The disabled default was rejected because it allows repeated anchor-local births to produce avoidable dense
    # piles that magnify collision losses and placement failures.
    # Tradeoff: some births that would have stayed local are diverted to global fallback instead.
    ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Chebyshev radius used to count live neighbors around the anchor parent.
    # Kept at 2 because the doctrine should react to genuinely local crowding rather than broad regional density.
    LOCAL_RADIUS: int = 2
    #
    # CURRENT STATUS: active runtime knob.
    # Births are considered crowded once this many live non-anchor neighbors
    # are present in the anchor neighborhood.
    # Raised from 5 to 6 so the doctrine smooths only the densest local clusters rather than overreacting to mild
    # healthy neighborhoods.
    # A stricter threshold was rejected because it would suppress too much local lineage growth.
    MAX_NEIGHBORS: int = 6
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "block_birth" and "global_only".
    POLICY_WHEN_CROWDED: str = "global_only"  # block_birth | global_only
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "strict", "bypass", and "global_only".
    # This is the floor-recovery softening surface for The Ashen Press.
    BELOW_FLOOR_POLICY: str = "bypass"  # strict | bypass | global_only


@dataclass
class RespawnCooldownOverlayConfig:
    """The Widow Interval (parent refractory reproduction overlay).

    Cooldown is UID-scoped rather than slot-scoped so slot reuse does not
    corrupt parent eligibility semantics.
    """

    #
    # CURRENT STATUS: active runtime knob.
    # Master enable for The Widow Interval.
    # Enabled because the current live brain-parent surface does not carry a rich fitness signal for alive agents,
    # so a light refractory doctrine is one of the few config-only tools available to prevent the same surviving
    # UID from dominating consecutive birth cycles.
    # The disabled default was rejected because it amplifies repetitive parent reuse in a way that reduces local
    # diversity.
    ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Tick duration for parent-UID refractory windows.
    # Raised from 32 to 48 to span a meaningful fraction of the new 64-tick normal respawn cadence without making
    # the parent pool brittle.
    # More aggressive durations were rejected because they can strand the controller when the alive set is small;
    # shorter windows were rejected because they do not meaningfully break repetitive parent reuse.
    DURATION_TICKS: int = 48
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the doctrine applies to the brain parent role.
    APPLY_TO_BRAIN_PARENT: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the doctrine applies to the trait parent role.
    APPLY_TO_TRAIT_PARENT: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the doctrine applies to the anchor parent role.
    APPLY_TO_ANCHOR_PARENT: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # When true, any cooled UID is treated as cooled for all enabled roles.
    # When false, each role keeps its own refractory ledger.
    UNIFIED_UID_POLICY: bool = True
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "allow_best_available" and "strict".
    EMPTY_POOL_POLICY: str = "allow_best_available"  # allow_best_available | strict
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "allow_best_available", "bypass", and "strict".
    BELOW_FLOOR_POLICY: str = "allow_best_available"  # allow_best_available | bypass | strict


@dataclass
class RespawnLocalParentOverlayConfig:
    """The Bloodhold Radius (local lineage parent-selection overlay)."""

    #
    # CURRENT STATUS: active runtime knob.
    # Master enable for The Bloodhold Radius.
    # Enabled because the repository launch path is single-family and the alive-slot fitness surface is too weak to
    # trust as a global brain-parent discriminator.
    # This local doctrine is therefore a practical config-only way to preserve spatially grounded lineage dynamics.
    ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Chebyshev radius around the dead slot used to build the local candidate pool.
    # Raised from 8 to 10 so sparse regions still find a usable local pool while remaining meaningfully local.
    # A smaller radius was rejected because it risks empty pools on a 100x100 map; a much larger radius was
    # rejected because it collapses back toward near-global parenting.
    SELECTION_RADIUS: int = 10
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "global" and "strict".
    FALLBACK_BEHAVIOR: str = "global"  # global | strict
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Supported values are "prefer_local_then_global", "bypass", and "strict".
    BELOW_FLOOR_POLICY: str = "prefer_local_then_global"  # prefer_local_then_global | bypass | strict


@dataclass
class RespawnOverlayViewerConfig:
    """Viewer exposure controls for reproduction overlay doctrines."""

    HOTKEYS_ENABLED: bool = True
    SHOW_STATUS_IN_HUD: bool = True
    SHOW_STATUS_IN_PANEL: bool = True
    SHOW_OVERRIDE_MARKERS: bool = True


@dataclass
class RespawnOverlayConfig:
    """Structured overlay doctrine control surface for reproduction."""

    CROWDING: RespawnCrowdingOverlayConfig = field(default_factory=RespawnCrowdingOverlayConfig)
    COOLDOWN: RespawnCooldownOverlayConfig = field(default_factory=RespawnCooldownOverlayConfig)
    LOCAL_PARENT: RespawnLocalParentOverlayConfig = field(default_factory=RespawnLocalParentOverlayConfig)
    VIEWER: RespawnOverlayViewerConfig = field(default_factory=RespawnOverlayViewerConfig)


@dataclass
class RespawnConfig:
    """Binary reproduction, population recovery, and offspring placement controls.

    This section governs the post-death repopulation path: when births happen, how
    many can happen, which parent roles are selected, what floor-recovery means,
    how extinction is handled, and where offspring may be placed.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Minimum tick gap between normal respawn cycles.
    # Lowered from 100 to 64 so normal recovery starts sooner after attrition, while still leaving enough time for
    # deaths and selection pressure to matter between scheduled recovery sweeps.
    # The default was rejected because dead slots can remain empty for too long relative to the observed health and
    # metabolism scales, which makes moderate downturns snowball into thin populations.
    # More aggressive values were rejected because they would make the controller feel like a constant birth pump.
    RESPAWN_PERIOD: int = 64
    #
    # CURRENT STATUS: active runtime knob.
    # Upper bound on births emitted in one respawn cycle.
    # Raised from 3 to 8 because this same cap also constrains the practical effect of extinction recovery.
    # The default was rejected as too weak for overnight floor repair on a 160-slot run.
    # Expected effect: materially better recovery after localized die-offs without allowing unlimited burst births.
    # Tradeoff: more spawn work on recovery ticks and a higher chance of rapid repopulation if the environment turns
    # temporarily very safe.
    MAX_SPAWNS_PER_CYCLE: int = 8
    #
    # CURRENT STATUS: active runtime knob.
    # Soft lower population threshold that triggers recovery behavior.
    # Raised from 20 to 64 so the controller intervenes before the population becomes functionally non-interactive.
    # The old floor was rejected because it waits until the run is already close to ecological failure on a 160-slot
    # substrate.
    # A much higher floor was rejected because it would keep the system in near-permanent recovery mode.
    POPULATION_FLOOR: int = 64
    #
    # CURRENT STATUS: active runtime knob.
    # Upper population ceiling for births.
    # Set equal to `AGENTS.N` rather than left at the misleading old value of 350.
    # Repository fact: births can only occupy dead slots within the fixed registry width, so a ceiling above the slot
    # count does not create real headroom; it only obscures the effective limit.
    # Expected effect: the public control surface now matches the actual runtime cap seen by the controller.
    POPULATION_CEILING: int = 160

    # Prompt 5 reproduction control surface.
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Non-supported non-default values are rejected during runtime validation rather than being silently accepted.
    # Reproduction-mode selector.
    # The current runtime validates only `"binary_parented"`.
    # Treat alternate values as unsupported until code is expanded.
    MODE: str = "binary_parented"  # Guarded compatibility surface; reproduction semantics remain binary parented.
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.RESPAWN.BRAIN_PARENT_SELECTOR` runtime read was found in the code dump.
    # Documented brain-parent selector identity.
    # No direct runtime read was found; the live code currently selects by explicit logic rather than
    # this string knob.
    BRAIN_PARENT_SELECTOR: str = "fitness"
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.RESPAWN.TRAIT_PARENT_SELECTOR` runtime read was found in the code dump.
    # Documented trait-parent selector identity.
    # No direct runtime read was found in the uploaded dump.
    TRAIT_PARENT_SELECTOR: str = "vitality"
    #
    # CURRENT STATUS: active runtime knob.
    # Placement-anchor selection policy.
    # Supported values in code are `"brain_parent"`, `"trait_parent"`, `"random_parent"`, and
    # `"fitter_of_two"`.
    # This changes where offspring are placed relative to parents.
    ANCHOR_PARENT_SELECTOR: str = "trait_parent"  # brain_parent | trait_parent | random_parent | fitter_of_two

    #
    # CURRENT STATUS: active runtime knob.
    # Minimum fitness threshold for brain-parent eligibility under normal recovery.
    # Lower it to widen parent eligibility; raise it to demand more proven performers.
    BRAIN_PARENT_MIN_FITNESS: float = 0.0
    #
    # CURRENT STATUS: active runtime knob.
    # Minimum normalized HP ratio required for trait-parent eligibility under normal recovery.
    # Higher values bias births toward healthier trait donors.
    TRAIT_PARENT_MIN_HP_RATIO: float = 0.10
    #
    # CURRENT STATUS: active runtime knob.
    # Minimum age in ticks required for trait-parent eligibility.
    # Use this to prevent extremely young agents from donating traits.
    TRAIT_PARENT_MIN_AGE_TICKS: int = 0
    #
    # CURRENT STATUS: active runtime knob.
    # Whether parent-eligibility thresholds are suspended during floor recovery.
    # Keeping this true makes emergency recovery more permissive.
    FLOOR_RECOVERY_SUSPEND_THRESHOLDS: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.RESPAWN.FLOOR_RECOVERY_REQUIRE_TWO_PARENTS` runtime read was found in the code dump.
    # Documented emergency two-parent requirement knob.
    # No direct runtime read was found in the uploaded dump.
    FLOOR_RECOVERY_REQUIRE_TWO_PARENTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Structured overlay doctrines layered on top of the binary reproduction
    # substrate. These overlays do not replace the parent-role architecture;
    # they constrain it.
    OVERLAYS: RespawnOverlayConfig = field(default_factory=RespawnOverlayConfig)

    #
    # CURRENT STATUS: active runtime knob.
    # What to do when live population drops below the minimum needed for binary reproduction.
    # Switched from `fail_run` to `seed_bank_bootstrap` because premature overnight extinction is a first-class
    # failure mode for this profile.
    # The default was rejected because it optimizes for operator disappointment: one bad hour can terminate the
    # entire unattended run.
    # Tradeoff: recovery after true collapse reintroduces default-latent seeds rather than preserving a pure
    # no-rescue ecology.
    EXTINCTION_POLICY: str = "seed_bank_bootstrap"  # fail_run | seed_bank_bootstrap | admin_spawn_defaults
    #
    # CURRENT STATUS: active runtime knob.
    # How many bootstrap agents to spawn under extinction-recovery policies.
    # Raised from 2 to 8 so extinction recovery is materially capable once triggered.
    # This value was coordinated with `MAX_SPAWNS_PER_CYCLE`, which also caps practical bootstrap throughput.
    # A smaller value was rejected because it produces a fragile near-zero restart; a larger value was rejected to
    # avoid turning collapse recovery into an almost instantaneous full reset.
    EXTINCTION_BOOTSTRAP_SPAWNS: int = 8
    #
    # CURRENT STATUS: active runtime knob.
    # Family assigned to bootstrap spawns created by extinction recovery.
    # Choose a valid family name from `BRAIN.FAMILY_ORDER`.
    EXTINCTION_BOOTSTRAP_FAMILY: str = "House Nocthar"

    #
    # CURRENT STATUS: active runtime knob.
    # Minimum ring radius used when searching near the anchor parent for placement.
    # Larger values push children farther from the anchor.
    OFFSPRING_JITTER_RADIUS_MIN: int = 1
    #
    # CURRENT STATUS: active runtime knob.
    # Maximum ring radius used for local offspring placement search.
    # Raised from 3 to 4 so births have a slightly wider anchor-local envelope before giving up.
    # The default was rejected because local placement becomes brittle once crowding and cooldown doctrines are
    # enabled together.
    # A much larger radius was rejected because it would blur local lineage structure.
    OFFSPRING_JITTER_RADIUS_MAX: int = 4
    #
    # CURRENT STATUS: active runtime knob.
    # Hard cap on local placement attempts before fallback / failure.
    # Raised from 32 to 48 so the controller searches a little harder before declaring local placement failure.
    # Expected effect: fewer lost births during recovery without resorting immediately to global fallback.
    # Tradeoff: somewhat more work on spawn-heavy ticks.
    OFFSPRING_MAX_PLACEMENT_ATTEMPTS: int = 48
    #
    # CURRENT STATUS: active runtime knob.
    # Whether to search globally if local anchor placement fails.
    # Disabling this makes spawn locality stricter but can increase failed births.
    ALLOW_FALLBACK_GLOBAL_PLACEMENT: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether births may occur on wall tiles.
    # Keeping this true preserves physical validity.
    DISALLOW_SPAWN_ON_WALL: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether births may occur on already occupied tiles.
    # Keeping this true helps maintain no-stacking guarantees.
    DISALLOW_SPAWN_ON_OCCUPIED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether births may occur in negative zone tiles.
    # Keeping this true avoids immediately hostile spawn sites.
    DISALLOW_SPAWN_IN_HARM_ZONE: bool = True

    #
    # CURRENT STATUS: active runtime knob.
    # Initial HP policy for newborn agents.
    # Supported values are `"full"` and `"fraction"`.
    # Use `fraction` when you want newborn fragility.
    BIRTH_HP_MODE: str = "full"  # full | fraction
    #
    # CURRENT STATUS: active runtime knob.
    # Fraction of `hp_max` used when `BIRTH_HP_MODE` is `fraction`.
    # Values are logically intended to live in `[0, 1]`.
    BIRTH_HP_FRACTION: float = 1.0

    #
    # CURRENT STATUS: active runtime knob.
    # Whether failed placement attempts are emitted to telemetry/logging.
    # Useful during debugging crowded maps or strict spawn constraints.
    LOG_PLACEMENT_FAILURES: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.RESPAWN.ASSERT_BINARY_PARENTING` runtime read was found in the code dump.
    # Documented binary-parenting assertion knob.
    # No direct runtime read was found in the uploaded dump.
    ASSERT_BINARY_PARENTING: bool = True


@dataclass
class TraitInit:
    """Legacy/default trait template.

    This structure exists in the root config surface, but in the uploaded code dump
    the live birth pipeline is driven by latent-budget reconstruction rather than
    direct reads from this template.
    """
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # This value lives inside the legacy/default trait template container, but the live
    # birth pipeline in the uploaded code uses latent-budget decoding instead of
    # directly reading `TRAITS.INIT` fields.
    mass: float = 2.0
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # This value lives inside the legacy/default trait template container, but the live
    # birth pipeline in the uploaded code uses latent-budget decoding instead of
    # directly reading `TRAITS.INIT` fields.
    vision: float = 8.0
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # This value lives inside the legacy/default trait template container, but the live
    # birth pipeline in the uploaded code uses latent-budget decoding instead of
    # directly reading `TRAITS.INIT` fields.
    hp_max: float = 20.0
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # This value lives inside the legacy/default trait template container, but the live
    # birth pipeline in the uploaded code uses latent-budget decoding instead of
    # directly reading `TRAITS.INIT` fields.
    metab: float = 0.005


@dataclass
class TraitClamp:
    """Trait clamp ranges.

    These are the hard lower/upper bounds used when latent allocations are decoded
    into physical trait values. Widening them expands the reachable biological
    space; tightening them compresses it.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.mass operator knob.
    mass: List[float] = field(default_factory=lambda: [0.5, 8.0])
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.vision operator knob.
    vision: List[float] = field(default_factory=lambda: [4.0, 16.0])
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.hp_max operator knob.
    hp_max: List[float] = field(default_factory=lambda: [5.0, 50.0])
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.metab operator knob.
    # Tightened from [0.01, 0.4] to [0.008, 0.28].
    # The default upper bound was rejected because the affine metabolism formula plus high-vision/high-mass samples
    # can become punishing enough to erase young lineages before learning has any real chance to matter.
    # Expected effect: fewer biologically doomed high-metabolism draws while still preserving meaningful variation.
    # Tradeoff: slightly softer environmental pressure and a somewhat narrower trait-space frontier.
    metab: List[float] = field(default_factory=lambda: [0.008, 0.28])


@dataclass
class TraitBudgetConfig:
    """Trait-allocation budget controls.

    These knobs define the latent budget used by the trait decoder and its allowed
    range. They shape the total amount of trait mass available before clamping.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.INIT_BUDGET operator knob.
    INIT_BUDGET: float = 1.0
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.INIT_LOGITS operator knob.
    INIT_LOGITS: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.MIN_BUDGET operator knob.
    MIN_BUDGET: float = 0.25
    #
    # CURRENT STATUS: active runtime knob.
    # TRAITS.MAX_BUDGET operator knob.
    MAX_BUDGET: float = 1.75


@dataclass
class TraitsConfig:
    """Trait decoding controls.

    The live repository uses a latent-budget system plus an affine metabolism
    formula. This section is where trait-space constraints and metabolism formula
    coefficients are documented.
    """
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.TRAITS.INIT` runtime read was found in the code dump.
    # Legacy initial trait template container.
    # The current birth pipeline uses latent decoding rather than reading this block directly, so the
    # container itself is currently unread.
    INIT: TraitInit = field(default_factory=TraitInit)
    #
    # CURRENT STATUS: active runtime knob.
    # Hard trait clamp bundle.
    # This is active and used when latent traits are decoded into realized trait values.
    CLAMP: TraitClamp = field(default_factory=TraitClamp)
    #
    # CURRENT STATUS: active runtime knob.
    # Trait-budget control bundle.
    # This is active and shapes both initialization and mutation of the latent trait budget.
    BUDGET: TraitBudgetConfig = field(default_factory=TraitBudgetConfig)
    #
    # CURRENT STATUS: guarded compatibility surface.
    # Non-supported non-default values are rejected during runtime validation rather than being silently accepted.
    # Metabolism formula selector.
    # The current runtime validates only `"affine_combo"`.
    # Treat this as a guarded compatibility surface.
    METAB_FORM: str = "affine_combo"  # Guarded compatibility surface; only affine_combo is currently implemented.
    #
    # CURRENT STATUS: active runtime knob.
    # Coefficient dictionary used by the active affine metabolism formula.
    # Reduced modestly from {base: 0.0002, per_mass: 0.0001, per_vision: 0.00002} to a slightly gentler burden.
    # Repository-grounded rationale: the default latent decode already yields modest HP budgets, so keeping both the
    # clamp ceiling and the affine coefficients too high biases the world toward fast attrition rather than overnight
    # persistence.
    # Expected effect: a longer median pre-learning survival window, especially for non-zone pathing mistakes.
    # Tradeoff: agents can coast longer without solving the environment, so the profile compensates with higher
    # concurrency and active reproduction doctrines rather than making survival free.
    METAB_COEFFS: Dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.00015,
            "per_mass": 0.00008,
            "per_vision": 0.000015,
        }
    )


@dataclass
class PhysicsConfig:
    """Combat / collision / movement cost controls.

    These constants shape deterministic world damage and penalties. They do not
    control learning directly; they change the environment the policies must solve.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Penalty or damage scale associated with wall interaction.
    # Lowered from 0.60 to 0.35 because the overnight goal is not to let first-generation random movement die almost
    # immediately on map geometry.
    # The default was rejected as too punishing once combined with procedural walls and metabolism.
    # Tradeoff: walls still matter, but they become more of a navigational tax than a frequent execution mechanism.
    K_WALL_PENALTY: float = 0.35
    #
    # CURRENT STATUS: active runtime knob.
    # Penalty scale associated with ram/collision events.
    # Lowered slightly from 0.1 to 0.08 so exploratory contact is not over-taxed while collisions remain meaningful.
    # A much lower value was rejected because it would make body-contact almost free.
    K_RAM_PENALTY: float = 0.08
    #
    # CURRENT STATUS: active runtime knob.
    # Penalty applied when the relevant idle-hit condition is triggered.
    # Lowered from 0.8 to 0.45 because the default passive-hit punishment is very large relative to the decoded early
    # HP scales.
    # Expected effect: fewer abrupt deaths from immature spacing behavior.
    # Tradeoff: somewhat softer deterrence against bad collision etiquette.
    K_IDLE_HIT_PENALTY: float = 0.45
    #
    # CURRENT STATUS: active runtime knob.
    # Damage applied to the contest winner in asymmetric combat resolution.
    # Lowered from 0.2 to 0.12 so successful contests remain costly but not self-nullifying.
    # The more conservative default was rejected because it drains even advantaged agents too quickly in aggregate.
    K_WINNER_DAMAGE: float = 0.12
    #
    # CURRENT STATUS: active runtime knob.
    # Damage applied to the contest loser.
    # Lowered from 0.6 to 0.36 to preserve selection pressure while reducing the chance that routine contact wipes out
    # thin local populations before reproduction can refill them.
    # A much softer value was rejected because it would dull competitive turnover too far.
    K_LOSER_DAMAGE: float = 0.36
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.PHYS.MOVE_FAIL_COST` runtime read was found in the code dump.
    # Configured cost for failed movement.
    # No direct runtime read was found in the uploaded dump, so this currently appears unread.
    MOVE_FAIL_COST: float = -0.2
    #
    # CURRENT STATUS: active runtime knob.
    # Combat tie-break policy after primary strength ordering.
    # The configured default is `"strength_then_lowest_id"`.
    # Changing it changes deterministic contest resolution semantics.
    TIE_BREAKER: str = "strength_then_lowest_id"  # Contest tie-break policy after strength sorting.


@dataclass
class PerceptionConfig:
    """Observation-schema controls.

    These values define the canonical observation layout, legacy bridge dimensions,
    and normalization constants consumed by the perception and brain subsystems.
    Changing schema counts here is high-risk because tensor shapes must remain
    consistent end-to-end.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Number of rays cast per observing agent.
    # Increasing this improves directional coverage but raises observation and inference cost.
    NUM_RAYS: int = 32
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.PERCEPT.RAY_FIELD_AGG` runtime read was found in the code dump.
    # Ray-field aggregation mode surface.
    # No direct runtime read was found in the uploaded dump, so this currently appears to be an
    # unread compatibility knob.
    RAY_FIELD_AGG: str = "max_abs"
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.PERCEPT.RAY_STEP_SAMPLER` runtime read was found in the code dump.
    # Ray stepping / sampler mode surface.
    # No direct runtime read was found in the uploaded dump, so this currently appears unread.
    RAY_STEP_SAMPLER: str = "dda_first_hit"

    #
    # CURRENT STATUS: active runtime knob.
    # Observation-contract selector for additive perception branches.
    # Supported values are `"canonical_v2"` and `"experimental_selfcentric_v1"`.
    OBS_MODE: str = "canonical_v2"
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the perception system should emit the additive experimental observation bundle.
    # This may be enabled independently for debugging, but the experimental branch preset requires it.
    RETURN_EXPERIMENTAL_OBSERVATIONS: bool = False

    #
    # CURRENT STATUS: active runtime knob.
    # Canonical per-ray feature count.
    # This is a schema-critical tensor dimension; changing it requires synchronized changes
    # throughout perception and brain code.
    CANONICAL_RAY_FEATURES: int = 8
    #
    # CURRENT STATUS: active runtime knob.
    # Canonical self-feature count.
    # High-risk schema knob: change only when the full observation pipeline is being migrated.
    CANONICAL_SELF_FEATURES: int = 11
    #
    # CURRENT STATUS: active runtime knob.
    # Canonical context-feature count.
    # High-risk schema knob with full observation-contract implications.
    CANONICAL_CONTEXT_FEATURES: int = 3

    #
    # CURRENT STATUS: active runtime knob.
    # Legacy bridge per-ray feature count.
    # Used by the legacy-to-canonical adapter for compatibility.
    LEGACY_RAY_FEATURES: int = 5
    #
    # CURRENT STATUS: active runtime knob.
    # Legacy state-vector width.
    # Used only while bridging legacy observations into the canonical schema.
    LEGACY_STATE_FEATURES: int = 2
    #
    # CURRENT STATUS: active runtime knob.
    # Legacy genome-vector width used by the adapter.
    # Changing it without changing the adapter will break legacy observation bridging.
    LEGACY_GENOME_FEATURES: int = 4
    #
    # CURRENT STATUS: active runtime knob.
    # Legacy position-vector width used by the adapter.
    # This is part of the compatibility surface.
    LEGACY_POSITION_FEATURES: int = 2
    #
    # CURRENT STATUS: active runtime knob.
    # Legacy context-vector width used by the adapter.
    # Keep aligned with any legacy observation producer still in use.
    LEGACY_CONTEXT_FEATURES: int = 3
    #
    # CURRENT STATUS: active runtime knob.
    # Legacy adapter identity string.
    # The configured default names the bridge implementation: `"prompt2_canonical_bridge_v1"`.
    # Treat this primarily as a compatibility/documentation surface.
    LEGACY_ADAPTER_MODE: str = "prompt2_canonical_bridge_v1"

    #
    # CURRENT STATUS: active runtime knob.
    # Normalization ceiling for zone-rate features.
    # Higher values make the normalized observation less sensitive to moderate field strengths.
    ZONE_RATE_ABS_MAX: float = 1.0
    #
    # CURRENT STATUS: active runtime knob.
    # Age normalization denominator in ticks.
    # Larger values make age-related features saturate more slowly.
    AGE_NORM_TICKS: int = 1024
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the perception system returns canonical observations by default.
    # Disabling this would push more pressure onto legacy compatibility paths.
    RETURN_CANONICAL_OBSERVATIONS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Experimental per-ray feature count.
    # This is the additive self-centric branch ray width; keep aligned with the experimental brain contract.
    EXPERIMENTAL_RAY_FEATURES: int = 7
    #
    # CURRENT STATUS: active runtime knob.
    # Experimental self-feature count.
    EXPERIMENTAL_SELF_FEATURES: int = 11
    #
    # CURRENT STATUS: active runtime knob.
    # Experimental context-feature count.
    EXPERIMENTAL_CONTEXT_FEATURES: int = 1


@dataclass
class BloodlineFamilySpec:
    """Per-family MLP topology description.

    Each family owns a fixed architectural signature. Fields here determine widths,
    activation, normalization placement, residual/gating usage, and optional split
    ray/scalar encoding paths.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.hidden_widths operator knob.
    hidden_widths: List[int] = field(default_factory=lambda: [256, 256, 192])
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.activation operator knob.
    activation: str = "gelu"
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.normalization operator knob.
    normalization: str = "pre"
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.residual operator knob.
    residual: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.gated operator knob.
    gated: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.split_inputs operator knob.
    split_inputs: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.split_ray_width operator knob.
    split_ray_width: int = 0
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.split_scalar_width operator knob.
    split_scalar_width: int = 0
    #
    # CURRENT STATUS: active runtime knob.
    # BRAIN.dropout operator knob.
    dropout: float = 0.0
    #
    # CURRENT STATUS: active runtime knob.
    # Observation contract consumed by this family topology.
    # Supported values are `"canonical_v2"` and `"experimental_selfcentric_v1"`.
    observation_contract: str = "canonical_v2"


@dataclass
class BrainConfig:
    """Policy/value network family controls.

    This section defines the action/value head sizes, the set of valid bloodline
    families, their colors, and the exact architectural spec used to instantiate a
    brain for each family.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Actor head output width.
    # This must remain aligned with the action semantics expected by the engine.
    ACTION_DIM: int = 9
    #
    # CURRENT STATUS: active runtime knob.
    # Critic head output width.
    # The current architecture expects a scalar value head, so the default is `1`.
    VALUE_DIM: int = 1

    #
    # CURRENT STATUS: active runtime knob.
    # Ordered list of valid bloodline families.
    # Order matters for round-robin assignment and family-aware update ordering.
    FAMILY_ORDER: List[str] = field(
        default_factory=lambda: [
            "House Nocthar",
            "House Vespera",
            "House Umbrael",
            "House Mourndveil",
            "House Somnyr",
        ]
    )
    #
    # CURRENT STATUS: active runtime knob.
    # Fallback family used when no explicit family is supplied.
    # It must be present in `FAMILY_ORDER`.
    DEFAULT_FAMILY: str = "House Nocthar"
    #
    # CURRENT STATUS: active runtime knob.
    # Experimental single-family preset gate.
    # When enabled, root seeds are forced onto `EXPERIMENTAL_BRANCH_FAMILY`, that family switches to
    # the experimental self-centric observation contract, and viewer color is remapped to cyan.
    EXPERIMENTAL_BRANCH_PRESET: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # Existing family slot that is repurposed by the experimental branch preset.
    # It must already exist in `FAMILY_ORDER` so family-aware telemetry and viewer surfaces remain stable.
    EXPERIMENTAL_BRANCH_FAMILY: str = "House Nocthar"
    #
    # CURRENT STATUS: active runtime knob.
    # Cyan family color used when the experimental branch preset is active.
    EXPERIMENTAL_BRANCH_COLOR: List[int] = field(default_factory=lambda: [64, 224, 255])
    #
    # CURRENT STATUS: active runtime knob.
    # Lightweight split-input topology used by the experimental branch preset.
    EXPERIMENTAL_BRANCH_SPEC: BloodlineFamilySpec = field(
        default_factory=lambda: BloodlineFamilySpec(
            hidden_widths=[96, 64, 64],
            activation="silu",
            normalization="pre",
            residual=True,
            gated=False,
            split_inputs=True,
            split_ray_width=64,
            split_scalar_width=32,
            dropout=0.00,
            observation_contract="experimental_selfcentric_v1",
        )
    )
    #
    # CURRENT STATUS: active runtime knob.
    # Root-seed family assignment strategy.
    # The code supports `"round_robin"` and `"weighted_random"`.
    # This affects only initial/root assignment, not inherited family selection.
    INITIAL_FAMILY_ASSIGNMENT: str = "round_robin"
    #
    # CURRENT STATUS: active runtime knob.
    # Weight table used when `INITIAL_FAMILY_ASSIGNMENT` is `weighted_random`.
    # Weights must sum to a positive value.
    INITIAL_FAMILY_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "House Nocthar": 1.0,
            "House Vespera": 1.0,
            "House Umbrael": 1.0,
            "House Mourndveil": 1.0,
            "House Somnyr": 1.0,
        }
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Legacy transformer fallback toggle.
    # Keep disabled unless you are deliberately resurrecting that compatibility surface.
    LEGACY_TRANSFORMER_FALLBACK_ENABLED: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the brain may adapt legacy observations into canonical form.
    # Disable this only when you want strict canonical-only enforcement.
    ALLOW_LEGACY_OBS_FALLBACK: bool = True

    #
    # CURRENT STATUS: active runtime knob.
    # Viewer / UI color mapping per family.
    # Each family name should map to an RGB triplet-like list of three integers.
    FAMILY_COLORS: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "House Nocthar": [84, 138, 214],
            "House Vespera": [84, 160, 112],
            "House Umbrael": [220, 184, 76],
            "House Mourndveil": [208, 102, 102],
            "House Somnyr": [168, 112, 208],
        }
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Per-family topology specification bundle.
    # This is one of the most shape-sensitive surfaces in the repository.
    FAMILY_SPECS: Dict[str, BloodlineFamilySpec] = field(
        default_factory=lambda: {
            "House Nocthar": BloodlineFamilySpec(hidden_widths=[256, 256, 224, 192], activation="gelu", normalization="pre", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Vespera": BloodlineFamilySpec(hidden_widths=[160, 160, 160, 128, 128], activation="silu", normalization="pre", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Umbrael": BloodlineFamilySpec(hidden_widths=[320, 320, 224], activation="relu", normalization="post", residual=True, gated=False, split_inputs=False, dropout=0.00),
            "House Mourndveil": BloodlineFamilySpec(hidden_widths=[224, 224, 192], activation="silu", normalization="pre", residual=True, gated=True, split_inputs=True, split_ray_width=160, split_scalar_width=96, dropout=0.00),
            "House Somnyr": BloodlineFamilySpec(hidden_widths=[256, 256, 256, 224, 192], activation="gelu", normalization="pre", residual=True, gated=True, split_inputs=True, split_ray_width=192, split_scalar_width=128, dropout=0.02),
        }
    )


@dataclass
class PPOConfig:
    """PPO optimization and reward-surface controls.

    These knobs govern rollout length, optimization strength, clipping, entropy,
    bootstrap strictness, and the configurable reward gate that the engine validates
    before runtime.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Reward discount factor.
    # Higher values value longer-horizon returns more heavily.
    GAMMA: float = 0.99
    #
    # CURRENT STATUS: active runtime knob.
    # GAE lambda.
    # Higher values reduce bias and increase variance in the usual PPO tradeoff.
    LAMBDA: float = 0.95
    #
    # CURRENT STATUS: active runtime knob.
    # PPO clipping epsilon.
    # Larger values permit more aggressive policy movement per update.
    CLIP_EPS: float = 0.2
    #
    # CURRENT STATUS: active runtime knob.
    # Entropy bonus coefficient.
    # Raise it to encourage more exploration.
    ENTROPY_COEF: float = 0.001
    #
    # CURRENT STATUS: active runtime knob.
    # Critic loss coefficient in the combined PPO objective.
    # Higher values emphasize value fitting relative to policy loss.
    VALUE_COEF: float = 0.5
    #
    # CURRENT STATUS: active runtime knob.
    # Adam learning rate for per-UID optimizers.
    # Too high destabilizes updates; too low slows learning.
    LR: float = 3e-4
    #
    # CURRENT STATUS: active runtime knob.
    # Minimum trajectory length required before a UID buffer is eligible for update.
    # Raised from 8 to 16 so per-UID updates are not driven by almost degenerate micro-rollouts.
    # The smaller default was rejected because, together with 8 minibatches, it yields one-sample minibatches right
    # at eligibility and amplifies gradient noise.
    # Tradeoff: some short-lived UIDs will still die before first update; the compensating change is to shorten the
    # update cadence rather than to keep ultra-tiny rollout thresholds.
    BATCH_SZ: int = 16
    #
    # CURRENT STATUS: active runtime knob.
    # Number of minibatches carved from each rollout update.
    # Lowered from 8 to 4 so a minimally eligible 16-step rollout still produces usable minibatches instead of
    # mostly singleton slices.
    # A more aggressive reduction was rejected because it would throw away too much shuffling; the old value was
    # rejected because it was too fragmented for the new batch threshold.
    MINI_BATCHES: int = 4
    #
    # CURRENT STATUS: active runtime knob.
    # Maximum number of optimization passes per update.
    # Lowered from 4 to 3 because updates are made more frequent in this profile, so the stale-data reuse budget can
    # come down slightly without starving learning.
    # Tradeoff: each individual update extracts a little less signal, but aggregate overnight freshness improves.
    EPOCHS: int = 3
    #
    # CURRENT STATUS: active runtime knob.
    # Early-stop KL threshold.
    # Set positive to stop epochs early when policy drift exceeds this level.
    TARGET_KL: float = 0.01
    #
    # CURRENT STATUS: active runtime knob.
    # Global gradient norm clip.
    # Lower it for stricter update bounding.
    GRAD_NORM_CLIP: float = 1.0
    # PPO reward surface:
    # - REWARD_FORM selects the base reward shape. The default `sq_health_ratio`
    #   preserves the legacy behavior exactly: clamp(HP / max(HP_MAX, 1e-6), 0, 1)^2.
    # - REWARD_GATE_MODE optionally gates reward accumulation. `off` preserves
    #   legacy behavior. `hp_ratio_min` is preferred because HP_MAX varies across
    #   agents, so its threshold is expressed in normalized health ratio units in
    #   [0, 1]. `hp_abs_min` uses absolute HP units instead.
    # - REWARD_GATE_THRESHOLD is an inclusive minimum threshold in the units of
    #   the active gate mode.
    # - REWARD_BELOW_GATE_VALUE is the reward emitted when the gate is not met.
    #   Leaving it at 0.0 yields pure threshold-gated accumulation below the gate.
    #
    # CURRENT STATUS: active runtime knob.
    # Base reward-shape selector.
    # The current engine validates only `"sq_health_ratio"`.
    # This is a guarded reward-surface mode.
    REWARD_FORM: str = "sq_health_ratio"
    #
    # CURRENT STATUS: active runtime knob.
    # Optional reward gating mode.
    # Supported values are `"off"`, `"hp_ratio_min"`, and `"hp_abs_min"`.
    # This controls whether reward is suppressed below a threshold.
    REWARD_GATE_MODE: str = "off"  # off | hp_ratio_min | hp_abs_min
    #
    # CURRENT STATUS: active runtime knob.
    # Inclusive threshold used by the configured reward gate mode.
    # Interpretation depends on `REWARD_GATE_MODE`: normalized ratio vs absolute HP.
    REWARD_GATE_THRESHOLD: float = 0.0
    #
    # CURRENT STATUS: active runtime knob.
    # Reward emitted when the gate condition is not met.
    # Leave at `0.0` for pure threshold suppression below the gate.
    REWARD_BELOW_GATE_VALUE: float = 0.0
    #
    # CURRENT STATUS: active runtime knob.
    # Global cadence used by `should_update()`.
    # Lowered from 256 to 64 because dead UIDs have their buffers cleared at finalization, so waiting too long can
    # cause a large fraction of the population to die before ever becoming update-eligible.
    # The default was rejected as too slow for the observed health/metabolism scales.
    # A much lower value was rejected because it would drive very frequent optimizer traffic with little additional
    # benefit once the batch threshold is already modest.
    UPDATE_EVERY_N_TICKS: int = 64

    #
    # CURRENT STATUS: guarded compatibility surface.
    # Non-supported non-default values are rejected during runtime validation rather than being silently accepted.
    # UID ownership semantics selector for PPO state.
    # The current runtime validates only `"uid_strict"`.
    # Treat this as a guarded compatibility surface.
    OWNERSHIP_MODE: str = "uid_strict"  # Guarded compatibility surface; PPO ownership remains canonical-UID based.
    #
    # CURRENT STATUS: active runtime knob.
    # Schema version for serialized PPO buffers.
    # Change only as part of a deliberate compatibility migration.
    BUFFER_SCHEMA_VERSION: int = 1
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.PPO.TRACK_TRAINING_STATE` runtime read was found in the code dump.
    # Documented training-state tracking toggle.
    # No direct runtime read was found; training state is currently tracked regardless in the
    # uploaded dump.
    TRACK_TRAINING_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether ready PPO updates are grouped / ordered by family.
    # This changes scheduling order, not the per-UID ownership model.
    FAMILY_AWARE_UPDATE_ORDERING: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether non-terminal active buffers must carry staged bootstrap state.
    # Keeping this true makes rollout finalization safer and stricter.
    REQUIRE_BOOTSTRAP_FOR_ACTIVE_BUFFER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether dropped non-terminal buffers increment truncated-rollout counters.
    # Useful for diagnostics when buffers are cleared early.
    COUNT_TRUNCATED_ROLLOUTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether finalized inactive UID buffers are removed from memory.
    # Keeping this true avoids stale buffer buildup.
    DROP_INACTIVE_UID_BUFFERS_AFTER_FINALIZATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether buffer structure / finiteness is validated before update.
    # Disable only if you knowingly trade safety for speed.
    STRICT_BUFFER_VALIDATION: bool = True


@dataclass
class EvolutionConfig:
    """Mutation and fitness carryover controls.

    These values shape how much policy / trait mutation is applied during births and
    how much historical fitness decays or persists across death cycles.
    """
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.EVOL.SELECTION` runtime read was found in the code dump.
    # Documented selection-mode identity.
    # No direct runtime read was found in the uploaded dump.
    SELECTION: str = "softmax_fitness"
    #
    # CURRENT STATUS: active runtime knob.
    # Decay factor applied to stored fitness across death processing.
    # Lower values forget history faster; higher values preserve it longer.
    FITNESS_DECAY: float = 0.99
    #
    # CURRENT STATUS: active runtime knob.
    # Standard deviation of ordinary policy-parameter noise applied at birth.
    # Raise for more exploration / lineage drift.
    POLICY_NOISE_SD: float = 0.01
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.EVOL.FITNESS_TEMP` runtime read was found in the code dump.
    # Documented fitness-temperature surface.
    # No direct runtime read was found in the uploaded dump.
    FITNESS_TEMP: float = 1.0

    # Prompt 5 mutation knobs.
    #
    # CURRENT STATUS: active runtime knob.
    # Standard deviation for ordinary latent trait-logit mutations.
    # Higher values produce more aggressive trait drift.
    TRAIT_LOGIT_MUTATION_SIGMA: float = 0.012
    #
    # CURRENT STATUS: active runtime knob.
    # Standard deviation for ordinary budget mutations.
    # Higher values broaden budget drift across births.
    TRAIT_BUDGET_MUTATION_SIGMA: float = 0.05
    #
    # CURRENT STATUS: active runtime knob.
    # Probability of entering the rare-mutation path for a birth.
    # Raise cautiously: rare mutations are intentionally disruptive.
    RARE_MUT_PROB: float = 0.0005
    #
    # CURRENT STATUS: active runtime knob.
    # Rare-path trait-logit mutation strength.
    # This is typically much larger than the ordinary sigma.
    RARE_TRAIT_LOGIT_MUTATION_SIGMA: float = 0.40
    #
    # CURRENT STATUS: active runtime knob.
    # Rare-path budget mutation strength.
    # Increase for stronger occasional budget shocks.
    RARE_TRAIT_BUDGET_MUTATION_SIGMA: float = 0.15
    #
    # CURRENT STATUS: active runtime knob.
    # Rare-path policy-noise standard deviation.
    # This determines how disruptive rare policy mutation is.
    RARE_POLICY_NOISE_SD: float = 0.03

    #
    # CURRENT STATUS: active runtime knob.
    # Whether births may mutate into a different family.
    # Keep disabled if you want strict family inheritance.
    ENABLE_FAMILY_SHIFT_MUTATION: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # Base probability of family-shift mutation when enabled.
    # Usually kept very small.
    FAMILY_SHIFT_PROB: float = 0.0001


@dataclass
class ViewerConfig:
    """Viewer and HUD presentation controls.

    These knobs affect operator-facing rendering, overlay defaults, window size, and
    how catastrophe status is exposed in the UI. They do not change simulation
    mechanics.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Viewer frame-rate target.
    # Raised from 30 to 45 because the viewer advances roughly two engine ticks per rendered frame at default speed,
    # so this also increases overnight progress throughput.
    # The supplied machine appears strong enough to justify a moderate increase, but an extreme value was rejected
    # because unattended laptop thermals and CPU fallback scenarios remain uncertain from the supplied evidence.
    # Tradeoff: higher render load and potentially more heat/noise during a long run.
    FPS: int = 45
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.VIEW.PAINT_BRUSH` runtime read was found in the code dump.
    # Documented paint-brush shape/size for interactive painting tools.
    # No direct runtime read was found in the uploaded dump.
    PAINT_BRUSH: List[int] = field(default_factory=lambda: [3, 3])
    #
    # CURRENT STATUS: active runtime knob.
    # Step size used when adjusting paint rate interactively.
    # Smaller steps give finer control.
    PAINT_RATE_STEP: float = 0.05
    #
    # CURRENT STATUS: active runtime knob.
    # Default overlay on/off map for viewer startup.
    # Keys in the default surface are `h_rate`, `h_grad`, and `rays`.
    SHOW_OVERLAYS: Dict[str, bool] = field(default_factory=lambda: {"h_rate": True, "h_grad": False, "rays": False})  # Viewer default overlay state.
    #
    # CURRENT STATUS: active runtime knob.
    # Initial viewer window width in pixels.
    # Pure presentation knob.
    WINDOW_WIDTH: int = 800
    #
    # CURRENT STATUS: active runtime knob.
    # Initial viewer window height in pixels.
    # Pure presentation knob.
    WINDOW_HEIGHT: int = 800
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.VIEW.CELL_SIZE` runtime read was found in the code dump.
    # Documented default cell-size control.
    # No direct runtime read was found in the uploaded dump.
    CELL_SIZE: int = 10
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the viewer should display the bloodline legend by default.
    # This affects UI density, not simulation state.
    SHOW_BLOODLINE_LEGEND: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether HP-based bloodline color modulation is active in the viewer.
    # When disabled, rendered agents stay on their clean base family color across all HP ratios.
    BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # How strongly low-HP agents are shaded within bloodline coloring.
    # Higher values generally make low-HP darkening more pronounced.
    BLOODLINE_LOW_HP_SHADE: float = 0.35

    # Prompt 6 viewer catastrophe surfaces.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the viewer shows the catastrophe panel.
    # Presentation-only.
    SHOW_CATASTROPHE_PANEL: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe overlays are shown in the viewer.
    # Presentation-only.
    SHOW_CATASTROPHE_OVERLAY: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe state appears in the HUD/status strip.
    # Presentation-only.
    SHOW_CATASTROPHE_STATUS_IN_HUD: bool = True


@dataclass
class LogConfig:
    """Logging and runtime assertion controls.

    This section governs log directory placement, emission cadence, assertion
    hardness, and AMP enablement.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Base log directory.
    # Most runs create a run subdirectory beneath this root.
    DIR: str = "logs"
    #
    # CURRENT STATUS: active runtime knob.
    # Primary telemetry/log cadence in ticks.
    # Raised from 250 to 20000 because the default is far too chatty for an unattended overnight run.
    # Expected effect: dramatically lower console spam while still giving occasional heartbeat prints.
    # Tradeoff: less immediate textual visibility into short-timescale swings.
    LOG_TICK_EVERY: int = 20000
    #
    # CURRENT STATUS: active runtime knob.
    # Snapshot cadence in ticks.
    # Raised from 500 to 1_000_000 because each snapshot path writes dense registry data, heatmaps, and a full brain
    # bundle.
    # The default was rejected as operationally absurd for a long unattended run; scheduled runtime checkpoints are
    # the primary safety surface here, so snapshots are intentionally rare.
    # Tradeoff: fewer intermediate standalone snapshot artifacts.
    SNAPSHOT_EVERY: int = 1_000_000
    #
    # CURRENT STATUS: active runtime knob.
    # Master assertion toggle used by invariant checks.
    # Disabling this reduces safety diagnostics.
    ASSERTIONS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Automatic mixed precision toggle.
    # Typically enabled only on CUDA-capable runs.
    AMP: bool = True


@dataclass
class IdentityConfig:
    """Canonical UID ownership controls.

    These knobs document the UID substrate and compatibility bridging between the
    canonical UID path and legacy float shadow columns used for visibility.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the UID identity substrate is conceptually enabled.
    # This documents the canonical ownership model of the repository.
    ENABLE_UID_SUBSTRATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Identity ownership-mode string.
    # The configured default is `"uid_bridge"`.
    # It describes the bridge between canonical UID identity and legacy surfaces.
    OWNERSHIP_MODE: str = "uid_bridge"
    #
    # CURRENT STATUS: active runtime knob.
    # Whether binding invariants are asserted.
    # Disable only for exceptional debugging/performance experiments.
    ASSERT_BINDINGS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether parent references are checked against the lifecycle ledger.
    # Keeping this true protects lineage integrity.
    ASSERT_HISTORICAL_UIDS: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.IDENTITY.ASSERT_NO_SLOT_OWNERSHIP_LEAK` runtime read was found in the code dump.
    # Documented slot-leak assertion surface.
    # No direct runtime read was found in the uploaded dump.
    ASSERT_NO_SLOT_OWNERSHIP_LEAK: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether canonical UIDs are mirrored into legacy float shadow columns.
    # Useful for compatibility and viewer/log inspection.
    MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS: bool = True


@dataclass
class SchemaConfig:
    """Schema version stamps.

    These values are written into checkpoint / telemetry surfaces to make schema
    drift explicit. They are not casual tuning knobs; changing them should happen
    only with a deliberate migration.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Identity schema version stamp.
    # Bump only when the identity surface changes in a backward-incompatible way.
    IDENTITY_SCHEMA_VERSION: int = 1
    #
    # CURRENT STATUS: active runtime knob.
    # Observation schema version stamp.
    # Changing this without a full migration will break compatibility expectations.
    OBS_SCHEMA_VERSION: int = 2
    #
    # CURRENT STATUS: active runtime knob.
    # PPO state schema version stamp.
    # Used as a version marker; keep stable unless you deliberately migrate PPO checkpoint structure.
    PPO_STATE_SCHEMA_VERSION: int = 1
    #
    # CURRENT STATUS: active runtime knob.
    # Checkpoint schema version stamp.
    # Critical for save/load compatibility.
    CHECKPOINT_SCHEMA_VERSION: int = 6
    #
    # CURRENT STATUS: active runtime knob.
    # Reproduction schema version stamp.
    # Documents compatibility for lineage / reproduction surfaces.
    REPRODUCTION_SCHEMA_VERSION: int = 2
    #
    # CURRENT STATUS: active runtime knob.
    # Catastrophe schema version stamp.
    # Used during catastrophe-state serialization/restore validation.
    CATASTROPHE_SCHEMA_VERSION: int = 1
    #
    # CURRENT STATUS: active runtime knob.
    # Telemetry schema version stamp.
    # Protects downstream consumers from silent schema drift.
    TELEMETRY_SCHEMA_VERSION: int = 4
    #
    # CURRENT STATUS: active runtime knob.
    # Logging schema version stamp.
    # Bump only with deliberate ledger-format changes.
    LOGGING_SCHEMA_VERSION: int = 5


@dataclass
class CheckpointConfig:
    """Checkpoint capture, validation, and publish controls.

    This section determines what state is saved, how strict checkpoint validation
    is, whether file publishing is atomic, and how runtime checkpoint scheduling and
    retention behave.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Master enablement flag for substrate-style runtime checkpoints.
    # If disabled, higher-level checkpoint paths should be considered inactive by policy.
    ENABLE_SUBSTRATE_CHECKPOINTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether Python / NumPy / Torch RNG states are captured.
    # Keep enabled for deterministic resume.
    CAPTURE_RNG_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether per-UID optimizer states are serialized.
    # Disable only if you are willing to resume without optimizer continuity.
    CAPTURE_OPTIMIZER_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether AMP scaler state is captured.
    # Relevant mainly for CUDA AMP runs.
    CAPTURE_SCALER_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether PPO training counters / metadata are serialized.
    # Useful for faithful training continuation.
    CAPTURE_PPO_TRAINING_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether staged PPO bootstrap tails are checkpointed.
    # Important for precise continuation of partially accumulated rollouts.
    CAPTURE_BOOTSTRAP_STATE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether checkpoint tensor/container shapes are validated strictly.
    # Higher safety, slightly more validation work.
    STRICT_SCHEMA_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether checkpoint UID ownership consistency is validated strictly.
    # Highly recommended for lineage-safe resume.
    STRICT_UID_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether PPO-related checkpoint surfaces are validated strictly.
    # Keep enabled unless you are debugging a migration.
    STRICT_PPO_STATE_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether optimizer tensor-state shapes are checked against live parameter shapes.
    # Important when topology drift is a concern.
    VALIDATE_OPTIMIZER_TENSOR_SHAPES: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether serialized PPO buffers are checked against the expected schema.
    # Recommended for safety.
    VALIDATE_BUFFER_SCHEMA: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether a manifest file is emitted alongside checkpoint bundles.
    # The atomic publish path uses this as a gate.
    SAVE_CHECKPOINT_MANIFEST: bool = True  # Manifest emission gate used by atomic checkpoint publishing.
    #
    # CURRENT STATUS: active runtime knob.
    # Periodic runtime checkpoint cadence.
    # Enabled at 100_000 ticks because an overnight profile should not rely on a single uninterrupted process for all
    # progress preservation.
    # The disabled default was rejected because it offers no periodic safety net against crashes, power loss, or
    # late-run ecological collapse that the operator may wish to rewind.
    # Tradeoff: occasional heavier I/O and checkpoint serialization work.
    SAVE_EVERY_TICKS: int = 100_000  # Positive value enables scheduled atomic runtime checkpoints.
    #
    # CURRENT STATUS: active runtime knob.
    # Retention count for scheduled runtime checkpoints.
    # Raised from 3 to 8 so the overnight run keeps a useful rollback window instead of only the most recent tail.
    # A much larger value was rejected because full substrate checkpoints include optimizer state and can grow bulky.
    KEEP_LAST: int = 8  # Retention count for scheduler-produced runtime checkpoints; <=0 keeps every checkpoint.
    #
    # CURRENT STATUS: active runtime knob.
    # Name of the subdirectory used for runtime checkpoints within a run directory.
    # Pure pathing knob.
    DIRECTORY_NAME: str = "checkpoints"  # Subdirectory below each run directory where scheduler checkpoints are published.
    #
    # CURRENT STATUS: active runtime knob.
    # Stable filename prefix for scheduled checkpoint bundles.
    # Pure pathing knob.
    FILENAME_PREFIX: str = "runtime_tick_"  # Stable filename prefix for periodic runtime checkpoints.

    # Prompt 7 atomic publish and corruption controls.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether checkpoint publish uses temp-file + atomic replace semantics.
    # Recommended to keep enabled for corruption safety.
    ATOMIC_WRITE_ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether manifests are part of the published checkpoint file set.
    # Must remain enabled if strict manifest validation or latest-pointer writing is enabled.
    MANIFEST_ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether bundle SHA-256 checksums are written/validated.
    # Safety knob with modest extra I/O cost.
    CHECKSUM_ENABLED: bool = True
    #
    # CURRENT STATUS: active safety knob with an explicit dependency constraint.
    # Whether manifest metadata must validate during load.
    # Requires `MANIFEST_ENABLED = True` in the current runtime.
    STRICT_MANIFEST_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether manifest filenames must match the observed file structure exactly.
    # Good for catching pathing / publish mistakes.
    STRICT_DIRECTORY_STRUCTURE_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether manifest config fingerprint must match the bundle snapshot.
    # Stricter but can block resumes across intentional config drift.
    STRICT_CONFIG_FINGERPRINT_VALIDATION: bool = False
    #
    # CURRENT STATUS: active safety knob with an explicit dependency constraint.
    # Whether a `latest` pointer JSON file is maintained.
    # Requires `MANIFEST_ENABLED = True` in the current runtime.
    WRITE_LATEST_POINTER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Filename used for the latest-checkpoint pointer.
    # Pathing-only knob.
    LATEST_POINTER_FILENAME: str = "latest_checkpoint.json"
    #
    # CURRENT STATUS: active runtime knob.
    # Filename suffix for checkpoint bundle files.
    # Pathing-only knob.
    BUNDLE_FILENAME_SUFFIX: str = ".pt"
    #
    # CURRENT STATUS: active runtime knob.
    # Filename suffix appended to manifest files.
    # Pathing-only knob.
    MANIFEST_FILENAME_SUFFIX: str = ".manifest.json"
    #
    # CURRENT STATUS: active runtime knob.
    # Prefix used for temp files during atomic checkpoint publish.
    # Pathing / hygiene knob.
    TEMPFILE_PREFIX: str = ".tmp_ckpt_"
    #
    # CURRENT STATUS: active launch/resume knob.
    # Canonical launch intent. Supported values are:
    # - "fresh_run": build a new world from config
    # - "resume_exact": deterministic continuation intent; drift/fork deltas reject
    # - "resume_with_drift": restore substrate while explicitly accepting narrow operator/cadence drift
    # - "fork_from_checkpoint": restore a checkpoint as the ancestor of a deliberately changed run
    LAUNCH_MODE: str = "fresh_run"
    #
    # CURRENT STATUS: active launch/resume knob.
    # Explicit checkpoint bundle, checkpoint directory, or latest-pointer path used by checkpoint-backed launch modes.
    LOAD_PATH: str = ""
    #
    # CURRENT STATUS: active launch/resume knob.
    # Operator-visible reason recorded when `LAUNCH_MODE="fork_from_checkpoint"`.
    FORK_REASON: str = ""
    #
    # CURRENT STATUS: active launch/resume reporting knob.
    # Writes `resume_compatibility_report.json` into the new session directory for checkpoint-backed launches.
    WRITE_COMPATIBILITY_REPORT: bool = True
    #
    # CURRENT STATUS: active compatibility knob.
    # Policy for old checkpoints without stage-1 resume-contract metadata. The conservative default infers what it can
    # and labels the report as legacy-inferred; "reject" blocks such checkpoints at resume-policy resolution.
    LEGACY_METADATA_POLICY: str = "infer_conservative"
    #
    # Compatibility surface categories used by the stage-1 resume policy:
    # hard-fixed, allowed drift, fork-only, ignored-on-resume, blocked.
    COMPATIBILITY_REPORT_FILENAME: str = "resume_compatibility_report.json"


@dataclass
class TelemetryConfig:
    """Ledger and export controls.

    These knobs determine which telemetry surfaces are emitted and how frequently
    summary/lineage data is flushed.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Master toggle for the richer ledger family.
    # Disable to reduce telemetry volume.
    ENABLE_DEEP_LEDGERS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether life-ledger rows are emitted.
    # Useful for full lifecycle auditing.
    LOG_LIFE_LEDGER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether birth events are logged.
    # Important for lineage analysis.
    LOG_BIRTH_LEDGER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether death events are logged.
    # Important for mortality analysis.
    LOG_DEATH_LEDGER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether PPO update summaries are logged.
    # Useful for optimizer-trace analysis.
    LOG_PPO_UPDATE_LEDGER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe start/end/clear events are logged.
    # Useful for correlating shocks with outcomes.
    LOG_CATASTROPHE_EVENT_LEDGER: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether per-tick summaries are emitted.
    # High-volume but very informative.
    LOG_TICK_SUMMARY: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether family-level summaries are emitted.
    # Good for bloodline-level analysis.
    LOG_FAMILY_SUMMARY: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Cadence for family-summary rows.
    # Raised from 1 to 128 because the standard launch path is single-family and per-tick family rows are therefore
    # much less information-dense than they would be in a true multi-family run.
    # Expected effect: much lower telemetry volume with little loss of overnight interpretability.
    FAMILY_SUMMARY_EVERY_TICKS: int = 128
    #
    # CURRENT STATUS: active runtime knob.
    # Cadence for exporting summary rows.
    # Raised from 1 to 64 so the run still produces rich summaries, but not at an every-tick I/O cost.
    # The per-tick default was rejected because it is disproportionately expensive for long unattended horizons.
    # Tradeoff: summary plots become slightly coarser in time.
    SUMMARY_EXPORT_CADENCE_TICKS: int = 64  # Tick-summary export cadence for the overnight profile.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether summary aggregation work is skipped on non-emission ticks.
    # Efficiency knob that reduces wasted computation.
    SUMMARY_SKIP_NON_EMIT_WORK: bool = True  # Skip summary aggregation work on ticks that do not emit summary/family rows.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether lineage graphs/structures are exported.
    # Useful for ancestry reconstruction.
    EXPORT_LINEAGE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Lineage export format selector.
    # The comment in the repository states that the current runtime emits JSON lineage graphs.
    # Treat alternate formats as unsupported until implemented.
    LINEAGE_EXPORT_FORMAT: str = "json"  # Export format gate; the runtime currently emits JSON lineage graphs.
    #
    # CURRENT STATUS: active runtime knob.
    # Buffered flush threshold per ledger.
    # Raised from 64 to 4096 so summary/event exports amortize disk writes over larger batches.
    # The default was rejected because it flushes far too eagerly for an overnight run with enabled ledgers.
    # Tradeoff: rows remain buffered longer before they appear on disk.
    PARQUET_BATCH_ROWS: int = 4096  # Buffered parquet flush threshold per ledger for long unattended runs.
    #
    # CURRENT STATUS: active runtime knob.
    # Whether still-open life records are flushed on shutdown.
    # Recommended for clean run closure.
    FLUSH_OPEN_LIVES_ON_CLOSE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe exposure is tracked for telemetry.
    # Useful for shock attribution analyses.
    TRACK_CATASTROPHE_EXPOSURE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether extended inspector detail is exposed to the viewer.
    # Presentation/inspection enrichment only; it does not alter simulation semantics.
    ENABLE_VIEWER_INSPECTOR_ENRICHMENT: bool = True  # Toggle extended inspector details without affecting simulation semantics.


@dataclass
class ValidationConfig:
    """Offline audit-harness controls.

    These values drive the bundled determinism, resume-consistency, catastrophe
    reproduction, and save-load-save validation helpers.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Master enablement for the bundled final audit harness.
    # If disabled, the suite reports skipped checks.
    ENABLE_FINAL_AUDIT_HARNESS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether determinism probes are run by the validation suite.
    # Recommended for any serious change touching state evolution.
    ENABLE_DETERMINISM_TESTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether resume-consistency probes are run.
    # Critical when checkpointing or identity semantics change.
    ENABLE_RESUME_CONSISTENCY_TESTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether save-load-save signature checks are run.
    # Useful for checkpoint idempotence auditing.
    ENABLE_SAVE_LOAD_SAVE_TESTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe reproducibility probes are run.
    # Useful when modifying scheduler or catastrophe state surfaces.
    ENABLE_CATASTROPHE_REPRO_TESTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether stage-1 checkpoint launch-mode compatibility probes are run.
    # These verify exact/drift/fork classification without changing simulation rules.
    ENABLE_RESUME_POLICY_TESTS: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.VALIDATION.VALIDATION_STRICTNESS` runtime read was found in the code dump.
    # Documented strictness label.
    # No direct runtime branch on this label was found in the uploaded dump.
    VALIDATION_STRICTNESS: str = "strict"  # permissive | strict
    #
    # CURRENT STATUS: active runtime knob.
    # Default tick budget for audit harness runs.
    # Longer runs increase confidence but cost more time.
    AUDIT_DEFAULT_TICKS: int = 16
    #
    # CURRENT STATUS: active runtime knob.
    # Tick count used by the determinism probe.
    # Increase to make the comparison harsher.
    DETERMINISM_COMPARE_TICKS: int = 8
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.VALIDATION.SAVE_LOAD_SAVE_COMPARE_BUFFERS` runtime read was found in the code dump.
    # Documented save-load-save buffer comparison flag.
    # No direct runtime read was found in the uploaded dump.
    SAVE_LOAD_SAVE_COMPARE_BUFFERS: bool = True
    #
    # CURRENT STATUS: currently unread / effectively dead in the uploaded repository dump.
    # Audit basis: no direct `.VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES` runtime read was found in the code dump.
    # Documented strict telemetry-write flag.
    # No direct runtime read was found in the uploaded dump.
    STRICT_TELEMETRY_SCHEMA_WRITES: bool = True


@dataclass
class MigrationConfig:
    """Migration / compatibility visibility controls.

    These knobs support the UID migration and tell logging / viewer surfaces what
    legacy-vs-canonical identity information should be shown.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Whether legacy slot fields are included in migration-era logging.
    # Useful during transition periods.
    LOG_LEGACY_SLOT_FIELDS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether canonical UID fields are included in logs.
    # Useful for lineage-safe auditing.
    LOG_UID_FIELDS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether the viewer shows both slot and UID identity information.
    # Inspection/UI only.
    VIEWER_SHOW_SLOT_AND_UID: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether bloodline information is shown in migration-era viewer surfaces.
    # Inspection/UI only.
    VIEWER_SHOW_BLOODLINE: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether canonical UID paths are required.
    # Safety / migration-hardening knob.
    REQUIRE_CANONICAL_UID_PATHS: bool = True


@dataclass
class CatastropheConfig:
    """Catastrophe scheduler and world-shock controls.

    This section decides whether catastrophes are enabled, how the scheduler picks
    events, how long they last, which types are eligible, and what parameter bundle
    each catastrophe type uses when active.
    """
    #
    # CURRENT STATUS: active runtime knob.
    # Master catastrophe enable switch.
    # Turn this off to remove scheduler-driven world shocks entirely.
    ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Default catastrophe scheduler mode.
    # Switched from `auto_dynamic` to `manual_only` for the overnight profile.
    # The automatic scheduler default was rejected because several catastrophe types directly increase damage,
    # suppress healing, or disable reproduction, which is the opposite of what a first-pass stability-and-emergence
    # profile should optimize for.
    # Manual mode preserves the full control surface for later interactive experiments without injecting unattended
    # night-time shocks.
    DEFAULT_MODE: str = "manual_only"  # off | manual_only | auto_dynamic | auto_static
    #
    # CURRENT STATUS: active runtime knob.
    # Whether a fresh run starts with the auto scheduler armed whenever the active mode is an auto mode.
    # Set to False so no hidden scheduler activation survives mode changes at boot.
    # This is intentionally conservative: the overnight run should not be ambushed by automatic catastrophes.
    DEFAULT_SCHEDULER_ARMED: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # Whether operator-triggered catastrophes are allowed.
    # Viewer/manual control surface.
    MANUAL_TRIGGER_ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether active catastrophes may be manually cleared.
    # Viewer/manual control surface.
    MANUAL_CLEAR_ENABLED: bool = True

    #
    # CURRENT STATUS: active runtime knob.
    # Whether multiple catastrophes may overlap in time.
    # Overlapping shocks increase system complexity and severity.
    ALLOW_OVERLAP: bool = False
    #
    # CURRENT STATUS: active runtime knob.
    # Maximum number of simultaneously active catastrophes.
    # Only relevant when overlap is allowed or manual triggering is aggressive.
    MAX_CONCURRENT: int = 1

    #
    # CURRENT STATUS: active runtime knob.
    # Minimum scheduler gap for dynamic auto mode.
    # Lower values create a busier catastrophe cadence.
    AUTO_DYNAMIC_GAP_MIN_TICKS: int = 240
    #
    # CURRENT STATUS: active runtime knob.
    # Maximum scheduler gap for dynamic auto mode.
    # Together with the minimum, this defines the random interval range.
    AUTO_DYNAMIC_GAP_MAX_TICKS: int = 540
    #
    # CURRENT STATUS: active runtime knob.
    # Whether dynamic mode jitters duration around the configured base duration.
    # Keeps repeated shocks less uniform.
    AUTO_DYNAMIC_SAMPLE_DURATION: bool = True

    #
    # CURRENT STATUS: active runtime knob.
    # Fixed interval between events in static auto mode.
    # Lower values make the static schedule denser.
    AUTO_STATIC_INTERVAL_TICKS: int = 420
    #
    # CURRENT STATUS: active runtime knob.
    # Ordering policy for static auto mode.
    # Supported values are `"round_robin"`, `"configured_sequence"`, and `"fixed_priority"`.
    # This changes only which enabled type is chosen next.
    AUTO_STATIC_ORDERING_POLICY: str = "round_robin"  # round_robin | configured_sequence | fixed_priority
    #
    # CURRENT STATUS: active runtime knob.
    # Explicit catastrophe sequence used when static ordering policy is `configured_sequence`.
    # Only enabled catastrophe IDs from this list are respected.
    AUTO_STATIC_SEQUENCE: List[str] = field(
        default_factory=lambda: [
            "ashfall_of_nocthar",
            "veil_of_somnyr",
            "the_hollow_fast",
            "graveweight",
            "glass_requiem",
            "the_witchstorm",
            "the_thorn_march",
            "the_barren_hymn",
        ]
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Fallback catastrophe duration in ticks.
    # Used when a type-specific duration is not supplied.
    DEFAULT_DURATION_TICKS: int = 180
    #
    # CURRENT STATUS: active runtime knob.
    # Lower clamp for catastrophe duration.
    # Protects against extremely short shocks.
    MIN_DURATION_TICKS: int = 60
    #
    # CURRENT STATUS: active runtime knob.
    # Upper clamp for catastrophe duration.
    # Protects against runaway-long shocks.
    MAX_DURATION_TICKS: int = 480
    #
    # CURRENT STATUS: active runtime knob.
    # Per-catastrophe duration overrides.
    # Each key should be a valid catastrophe ID.
    PER_TYPE_DURATION_TICKS: Dict[str, int] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": 160,
            "sanguine_bloom": 180,
            "the_woundtide": 180,
            "the_hollow_fast": 200,
            "mirror_of_thorns": 160,
            "veil_of_somnyr": 220,
            "graveweight": 200,
            "glass_requiem": 160,
            "the_witchstorm": 180,
            "the_thorn_march": 240,
            "the_barren_hymn": 120,
            "crimson_deluge": 180,
        }
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Per-catastrophe enable table.
    # Disable entries here to remove them from manual and auto selection.
    TYPE_ENABLED: Dict[str, bool] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": True,
            "sanguine_bloom": True,
            "the_woundtide": True,
            "the_hollow_fast": True,
            "mirror_of_thorns": True,
            "veil_of_somnyr": True,
            "graveweight": True,
            "glass_requiem": True,
            "the_witchstorm": True,
            "the_thorn_march": True,
            "the_barren_hymn": True,
            "crimson_deluge": True,
        }
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Weighted-random selection table used by dynamic auto mode.
    # Only relative magnitudes matter.
    TYPE_SELECTION_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": 1.0,
            "sanguine_bloom": 1.0,
            "the_woundtide": 1.0,
            "the_hollow_fast": 1.0,
            "mirror_of_thorns": 1.0,
            "veil_of_somnyr": 1.2,
            "graveweight": 0.9,
            "glass_requiem": 0.9,
            "the_witchstorm": 0.8,
            "the_thorn_march": 0.7,
            "the_barren_hymn": 0.6,
            "crimson_deluge": 0.9,
        }
    )

    # Prompt 6 catastrophe intensity and scheduler knobs.
    #
    # CURRENT STATUS: active runtime knob.
    # Per-catastrophe parameter bundle.
    # This is the main intensity-and-shape surface for catastrophe behavior.
    TYPE_PARAMS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "ashfall_of_nocthar": {"positive_zone_fraction": 0.65},
            "sanguine_bloom": {"zone_fraction": 0.45, "negative_rate": -1.5},
            "the_woundtide": {"front_half_width": 5.0, "negative_rate": -2.0},
            "the_hollow_fast": {"positive_scalar": 0.25},
            "mirror_of_thorns": {"zone_fraction": 0.50},
            "veil_of_somnyr": {"vision_scalar": 0.45},
            "graveweight": {"metabolism_scalar": 1.65, "mass_burden_scalar": 0.06},
            "glass_requiem": {"collision_damage_scalar": 1.8},
            "the_witchstorm": {"trait_sigma_scalar": 2.0, "budget_sigma_scalar": 2.0, "policy_noise_scalar": 2.0, "rare_prob_scalar": 3.0, "family_shift_scalar": 4.0},
            "the_thorn_march": {"negative_rate": -2.0, "max_shrink_fraction": 0.35},
            "the_barren_hymn": {"reproduction_enabled": 0.0},
            "crimson_deluge": {"patch_count": 3.0, "patch_size_fraction": 0.18, "negative_rate": -2.5},
        }
    )

    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe viewer controls are enabled.
    # Presentation/operator control only.
    VIEWER_CONTROLS_ENABLED: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe overlays are enabled in the viewer.
    # Presentation only.
    VIEWER_OVERLAY_ENABLED: bool = True

    #
    # CURRENT STATUS: active runtime knob.
    # Whether active catastrophe state is serialized into checkpoints.
    # Keep enabled for faithful resume.
    PERSIST_STATE_IN_CHECKPOINTS: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Whether catastrophe checkpoint payloads are validated strictly.
    # Useful safety flag when catastrophe state is part of save/load.
    STRICT_CHECKPOINT_VALIDATION: bool = True
    #
    # CURRENT STATUS: active runtime knob.
    # Offset applied to the catastrophe RNG stream relative to the master seed.
    # Keeps catastrophe scheduling deterministic while separating its random stream from other
    # subsystems.
    RNG_STREAM_OFFSET: int = 911


@dataclass
class Config:
    """Root aggregate configuration.

    This is the single object instantiated by repository root `config.py`, and the
    package consumes it through a bridge rather than maintaining a second source of
    truth.
    """
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    SIM: SimConfig = field(default_factory=SimConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    GRID: GridConfig = field(default_factory=GridConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    MAPGEN: MapgenConfig = field(default_factory=MapgenConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    AGENTS: AgentsConfig = field(default_factory=AgentsConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    TRAITS: TraitsConfig = field(default_factory=TraitsConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    PHYS: PhysicsConfig = field(default_factory=PhysicsConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    PERCEPT: PerceptionConfig = field(default_factory=PerceptionConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    BRAIN: BrainConfig = field(default_factory=BrainConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    RESPAWN: RespawnConfig = field(default_factory=RespawnConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    PPO: PPOConfig = field(default_factory=PPOConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    EVOL: EvolutionConfig = field(default_factory=EvolutionConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    VIEW: ViewerConfig = field(default_factory=ViewerConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    LOG: LogConfig = field(default_factory=LogConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    IDENTITY: IdentityConfig = field(default_factory=IdentityConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    SCHEMA: SchemaConfig = field(default_factory=SchemaConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    TELEMETRY: TelemetryConfig = field(default_factory=TelemetryConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    VALIDATION: ValidationConfig = field(default_factory=ValidationConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    MIGRATION: MigrationConfig = field(default_factory=MigrationConfig)
    #
    # Root section handle.
    # This keeps the top-level aggregate object explicit and stable so the rest of the
    # repository can access `cfg.<SECTION>` with no semantic ambiguity.
    CATASTROPHE: CatastropheConfig = field(default_factory=CatastropheConfig)


cfg = Config()


def apply_experimental_single_family_launch_defaults() -> None:
    """
    Force the live app launch path onto the experimental self-centric preset.

    This intentionally mutates only the process-local runtime config object used
    by the repository entrypoints. Direct callers can still override these
    fields explicitly before startup if they need a different launch mode.
    """

    cfg.PERCEPT.OBS_MODE = "experimental_selfcentric_v1"
    cfg.PERCEPT.RETURN_EXPERIMENTAL_OBSERVATIONS = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_PRESET = True
    cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY = str(cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY or cfg.BRAIN.DEFAULT_FAMILY)
    cfg.EVOL.ENABLE_FAMILY_SHIFT_MUTATION = False
    cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE = True
    # The current repository enables GradScaler when LOG.AMP is True, but PPO training does not
    # use torch.autocast(...) in the update path. For the forced experimental live preset, disable
    # scaler-only AMP so recoverable CUDA scaling overflows do not become an avoidable failure mode.
    cfg.LOG.AMP = False
