# The Control Room Handbook

This file is the working reference for running, inspecting, configuring, validating, checkpointing, and extending Tensor Crypt without drifting its semantic, checkpoint, or ownership contracts. It assumes the reader already understands the repository at a conceptual level and now needs the operational truth: which knobs are live, which surfaces are guarded, what the viewer can and cannot change, what telemetry actually records, how runtime checkpoints are published and restored, and where a change should be routed so that adjacent invariants remain intact.

## What this file teaches

- how the repository treats `config.py` as the single operator-facing control surface
- how to distinguish semantic knobs from presentation, telemetry, checkpoint, and validation knobs
- how runtime, viewer, telemetry, checkpoint, and validation surfaces fit together during real work
- what a runtime checkpoint must contain to be trustworthy here
- what the bundled validation probes actually compare
- where to implement a change so that it lands in the right subsystem and does not silently damage neighboring contracts

## What this file deliberately leaves for later

This file does not try to re-teach the game world, PPO mathematics, or folder-by-folder architecture from first principles. It also does not serve as a raw dataclass field dump. Those tasks belong to the earlier documents in the suite. Here, the question is not “what is the project?” but “how do I operate it and modify it safely?”

## How to use this handbook

Use the document in this order:

1. Read the control-surface taxonomy first. That tells you whether your intended change is semantic, observational, persistence-sensitive, or validation-sensitive.
2. Use the configuration-group map to decide which `cfg.<SECTION>` owns the decision.
3. Before touching persistence-sensitive code, read the checkpoint and validation sections together.
4. Before editing runtime logic, use the change-routing table to identify the primary ownership surface and the collateral surfaces that must be rechecked.
5. Treat the failure-mode section and maintainer checklist as the minimum bar before committing a change.

## `config.py` as the authoritative control surface

Repository-root `config.py` is the user-facing configuration entry surface. It does not define a separate second configuration object; it re-exports the canonical package configuration from `tensor_crypt.runtime_config`. Internal modules do not import the root file directly. They import `cfg` through `tensor_crypt.config_bridge`, which re-exposes the same `Config` dataclass aggregate and the same singleton instance.

The practical consequence is simple: there is one authoritative configuration object, but it is bridged into the package so runtime imports do not depend on the current working directory.

```text
repository root config.py
        │
        ▼
tensor_crypt.runtime_config.cfg
        │
        ▼
tensor_crypt.config_bridge.cfg
        │
        ├── runtime assembly
        ├── engine / physics / perception / PPO / respawn
        ├── viewer
        ├── telemetry
        ├── checkpointing
        └── validation
```

The repository also persists this same surface in two places:

- `run_dir/config.json` contains `asdict(cfg)` at run creation time.
- every runtime checkpoint bundle stores a full `config_snapshot`, and manifests may additionally record a configuration fingerprint.

### What “active knob” means here

An active knob is a field that the runtime actually reads and uses to alter behavior, output, or strictness in the uploaded codebase.

Examples:
- `SIM.SEED`
- `GRID.W`, `GRID.H`
- `RESPAWN.POPULATION_FLOOR`
- `PPO.UPDATE_EVERY_N_TICKS`
- `VIEW.SHOW_CATASTROPHE_PANEL`
- `CHECKPOINT.SAVE_EVERY_TICKS`
- `TELEMETRY.PARQUET_BATCH_ROWS`

### What “guarded compatibility surface” means here

A guarded compatibility surface is present in the config schema, but the runtime currently accepts only one supported mode or a very narrow supported set. Unsupported non-default values are rejected during validation rather than being silently ignored.

The repository explicitly validates several such surfaces:

- `SIM.DTYPE` -> only `"float32"`
- `AGENTS.SPAWN_MODE` -> only `"uniform"`
- `TRAITS.METAB_FORM` -> only `"affine_combo"`
- `RESPAWN.MODE` -> only `"binary_parented"`
- `PPO.OWNERSHIP_MODE` -> only `"uid_strict"`
- checkpoint dependency constraints:
  - `STRICT_MANIFEST_VALIDATION` requires `MANIFEST_ENABLED`
  - `WRITE_LATEST_POINTER` requires `MANIFEST_ENABLED`
- `PPO.REWARD_FORM` and `PPO.REWARD_GATE_MODE` are also validated against explicit supported sets

This is a deliberate design choice. The repository would rather reject an unsupported mode than let an operator believe a value is live when it is not.

### What to do with documented-but-unread surfaces

The runtime config itself marks some fields as “currently unread / effectively dead” in the uploaded dump. Treat those as documentary residue or compatibility placeholders until you verify a live runtime read. Do not assume that because a field exists, it is a safe experimental control.

Representative examples the config comments themselves flag this way include:

- `SIM.TICKS_PER_SEC`
- `GRID.EXPOSE_H_GRAD`
- `RESPAWN.BRAIN_PARENT_SELECTOR`
- `RESPAWN.TRAIT_PARENT_SELECTOR`
- `RESPAWN.FLOOR_RECOVERY_REQUIRE_TWO_PARENTS`
- `RESPAWN.ASSERT_BINARY_PARENTING`
- `TRAITS.INIT`
- `PHYS.MOVE_FAIL_COST`
- `PERCEPT.RAY_FIELD_AGG`
- `PERCEPT.RAY_STEP_SAMPLER`
- `PPO.TRACK_TRAINING_STATE`
- `EVOL.SELECTION`
- `EVOL.FITNESS_TEMP`
- `VIEW.PAINT_BRUSH`
- `VIEW.CELL_SIZE`
- `IDENTITY.ASSERT_NO_SLOT_OWNERSHIP_LEAK`
- `VALIDATION.VALIDATION_STRICTNESS`
- `VALIDATION.SAVE_LOAD_SAVE_COMPARE_BUFFERS`
- `VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES`

That distinction matters. In this repository, the mere existence of a config field does not prove it is part of the live operational surface.

## A taxonomy of knobs and control surfaces

The most important operational distinction in this repository is not “which section owns the knob?” but “what kind of consequence does changing this knob have?”

| Control class | Primary sections | What changes when you touch it | Typical risk |
|---|---|---|---|
| Session substrate | `SIM`, parts of `LOG` | seed, device, AMP, run length, assertion posture | reproducibility drift, hardware-specific behavior |
| World semantics | `GRID`, `MAPGEN`, `AGENTS`, `PHYS`, `RESPAWN`, `TRAITS`, `EVOL`, `CATASTROPHE` | the arena, field behavior, reproduction, mutation, combat, catastrophe pressure | the simulation becomes a different world |
| Observation and learning surfaces | `PERCEPT`, `BRAIN`, `PPO` | feature tensors, network topology, reward shape, optimizer behavior | checkpoint breakage, schema drift, PPO divergence |
| Viewer/operator presentation | `VIEW`, parts of `MIGRATION`, `TELEMETRY.ENABLE_VIEWER_INSPECTOR_ENRICHMENT` | what the operator sees and how much inspector detail is exposed | mistaken belief that a presentational edit is semantics-free |
| Logging and export | `LOG`, `TELEMETRY`, run metadata surfaces | cadence, ledgers, summaries, lineage export, snapshot frequency | hidden hot-path cost, schema drift in output artifacts |
| Checkpoint publication and restore | `CHECKPOINT`, `SCHEMA`, parts of `IDENTITY`, `BRAIN`, `PPO`, `CATASTROPHE` | what is serialized, how strict restore is, what file set is published | untrustworthy resumes, corrupted or incompatible bundles |
| Validation and audit | `VALIDATION`, `LOG.ASSERTIONS`, strict checkpoint flags | what the repository proves about itself and how harshly it checks | false confidence, skipped safety nets |
| Migration and compatibility | `IDENTITY`, `MIGRATION`, legacy observation bridge surfaces | legacy-vs-canonical visibility, UID bridging, compatibility display | mixing slot-era assumptions into canonical UID ownership |

Two practical rules follow:

1. A knob that changes what the simulation is must not be treated like a viewer convenience.
2. A knob that changes checkpoint, schema, or ownership meaning must be assumed to affect restore and validation even if the immediate local code change looks small.

## Configuration groups and what they really control

The table below is the compact working map of the root config surface. The “high-risk” column lists representative fields whose changes carry cross-subsystem consequences. It is not exhaustive.

### Config-surface map

| Section | Primary role | Representative high-risk knobs | Mostly operational / presentation knobs | Working note |
|---|---|---|---|---|
| `SIM` | session seed, device, top-level runtime posture | `SEED`, `DEVICE`, `DTYPE` | `MAX_TICKS`, `REUSE_ACTION_BUFFER` | `DTYPE` is guarded; CUDA availability is validated |
| `GRID` | world tensor size and heal/harm field composition | `W`, `H`, `HZ_OVERLAP_MODE`, `HZ_SUM_CLAMP`, `HZ_CLEAR_EACH_TICK` | — | catastrophe overlays depend on predictable field rebuild semantics |
| `MAPGEN` | random walls and baseline zone generation | wall count/length knobs, zone count/size/rate | — | changes the initial substrate of every new run |
| `AGENTS` | slot capacity and spawn substrate | `N`, `SPAWN_MODE`, `NO_STACKING` | — | `SPAWN_MODE` is guarded; `N` also affects telemetry and checkpoint width |
| `TRAITS` | latent trait budget, clamp ranges, metabolism coefficients | clamp ranges, budget bounds, `METAB_COEFFS`, `METAB_FORM` | — | live birth pipeline uses latent decoding, not `TRAITS.INIT` |
| `PHYS` | deterministic combat and environment cost model | damage/penalty constants, `TIE_BREAKER` | — | changes environment difficulty and death patterns directly |
| `PERCEPT` | canonical observation contract and normalization constants | canonical feature counts, `NUM_RAYS`, legacy bridge widths, normalization constants | `RETURN_CANONICAL_OBSERVATIONS` | schema-sensitive; shapes must stay aligned end-to-end |
| `BRAIN` | bloodline family set, topology, colors, head dimensions | `ACTION_DIM`, `VALUE_DIM`, `FAMILY_ORDER`, `FAMILY_SPECS`, `ALLOW_LEGACY_OBS_FALLBACK` | `FAMILY_COLORS` | family topology is checkpoint-visible |
| `RESPAWN` | binary reproduction, extinction handling, placement, newborn HP | thresholds, selector policy, extinction policy, placement rules, birth HP mode | `LOG_PLACEMENT_FAILURES` | changes lineage and population recovery semantics |
| `PPO` | reward surface, rollout cadence, update strength, ownership rules | reward form/gating, batch/epoch/clip/LR, `OWNERSHIP_MODE`, bootstrap strictness | family-aware update ordering | slot-based ownership is not supported; UID ownership is canonical |
| `EVOL` | mutation and fitness carryover | fitness decay, policy noise, trait mutation sigmas, rare mutation path, family shift | — | births and long-horizon lineage drift live here |
| `VIEW` | viewer startup defaults and rendering presentation | catastrophe panel/overlay flags | FPS, overlays, legend visibility, shade strength, window size | mostly presentational, but the viewer also contains semantic controls |
| `LOG` | run directory, progress cadence, snapshot cadence, assertions, AMP | `ASSERTIONS`, `AMP` | `DIR`, `LOG_TICK_EVERY`, `SNAPSHOT_EVERY` | snapshot cadence affects artifact volume, not world rules |
| `IDENTITY` | canonical UID substrate and shadow-column bridge | `ASSERT_BINDINGS`, `ASSERT_HISTORICAL_UIDS`, `MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS` | ownership mode label | UID ownership is part of checkpoint and telemetry truth |
| `SCHEMA` | version stamps written into persisted artifacts | all schema version fields | — | bump only with deliberate migrations |
| `CHECKPOINT` | runtime checkpoint capture, strictness, publishing, retention | strict validation flags, capture flags, atomic/manifest flags, `SAVE_EVERY_TICKS` | file/path naming knobs, `KEEP_LAST` | one of the most invariant-sensitive surfaces in the repo |
| `TELEMETRY` | deep ledgers, summaries, lineage export, batching | ledger enablement flags, export cadence, batch rows, catastrophe exposure tracking | inspector enrichment | observe-only by intent, but can impose hot-path cost |
| `VALIDATION` | final audit harness and probe enablement | harness/probe flags, tick budgets | — | determines what proof surfaces are run, not simulation rules |
| `MIGRATION` | legacy/canonical visibility and UID-era hardening | `REQUIRE_CANONICAL_UID_PATHS` | log/viewer visibility switches | operationally important during compatibility work |
| `CATASTROPHE` | scheduler mode, allowed types, durations, per-type parameters, persistence | scheduler mode/gaps, overlap, type enables, durations, `TYPE_PARAMS`, checkpoint persistence | viewer controls/overlay toggles | active shocks change world field, physics, perception, and reproduction |

### A practical grouping that matches real work

#### 1. World and lineage semantics

The sections that most directly change “what world is being simulated” are:

- `GRID`
- `MAPGEN`
- `AGENTS`
- `TRAITS`
- `PHYS`
- `RESPAWN`
- `EVOL`
- `CATASTROPHE`

When you change these, you are not decorating the existing system. You are changing the environment, the population dynamics, or both.

Examples:
- raising `MAPGEN.RANDOM_WALLS` changes pathing pressure
- changing `RESPAWN.EXTINCTION_POLICY` changes what a near-extinction event means
- changing `EVOL.RARE_MUT_PROB` changes lineage shock frequency
- changing `CATASTROPHE.TYPE_PARAMS` changes active world overrides

#### 2. Observation, policy, and training surfaces

The sections that most directly change what the policies can observe and how they are optimized are:

- `PERCEPT`
- `BRAIN`
- `PPO`

These are shape-sensitive and checkpoint-sensitive. A small local change here can invalidate brain topology signatures, optimizer metadata, serialized buffer payloads, and determinism assumptions.

The highest-risk fields are the canonical observation dimensions, family topology specs, action/value head sizes, and PPO ownership/reward/update surfaces.

#### 3. Operator, logging, and visibility surfaces

The sections that mostly change what an operator sees or what artifacts are emitted are:

- `VIEW`
- `LOG`
- `TELEMETRY`
- parts of `MIGRATION`

These are not automatically low-risk. A viewer toggle is low-risk if it only changes rendering, but telemetry and logging knobs alter cadence and output schema, and some viewer controls are not read-only.

#### 4. Persistence, schema, and safety-hardening surfaces

The sections that decide whether a run can be resumed faithfully and validated honestly are:

- `IDENTITY`
- `SCHEMA`
- `CHECKPOINT`
- `VALIDATION`
- parts of `MIGRATION`

Treat these as contract surfaces, not casual tuning knobs.

### Live guarded surfaces worth remembering

The following surfaces are intentionally narrow. Do not treat them as general mode selectors unless you are prepared to implement the missing branches and extend validation accordingly.

| Surface | Observed support |
|---|---|
| `SIM.DTYPE` | only `"float32"` |
| `AGENTS.SPAWN_MODE` | only `"uniform"` |
| `TRAITS.METAB_FORM` | only `"affine_combo"` |
| `RESPAWN.MODE` | only `"binary_parented"` |
| `PPO.OWNERSHIP_MODE` | only `"uid_strict"` |
| `PPO.REWARD_FORM` | only `"sq_health_ratio"` |
| `PPO.REWARD_GATE_MODE` | `"off"`, `"hp_ratio_min"`, `"hp_abs_min"` |
| `TELEMETRY.LINEAGE_EXPORT_FORMAT` | JSON export only in the observed code |

## Viewer and operator-facing surfaces

The viewer is not merely a renderer. It is the live operator console for stepping the engine, inspecting agents and zones, watching catastrophe state, and, in limited cases, changing runtime state.

### What the viewer is for

In practice, the viewer is the place where an operator:

- watches the current world state
- inspects a selected agent or H-Zone
- observes bloodline population counts and catastrophe state
- pauses, single-steps, or changes stepping speed
- toggles overlays
- manually triggers or clears catastrophes
- edits the rate of a selected H-Zone

That last point matters. The viewer is not purely observational.

### The main viewer surfaces

| Surface | What it shows | What it can change |
|---|---|---|
| world pane | walls, heal/harm zones, agents, HP bars, rays, selection markers, catastrophe overlay | selection, zoom, pan |
| HUD | tick, pause/speed state, alive count, per-family counts, catastrophe status line | none directly |
| side inspector | selected agent details or selected H-Zone details, bloodline legend, catastrophe block, controls cheat sheet | none directly |
| catastrophe controls | mode, active shocks, next tick, pause state | manual trigger, clear, mode cycle, auto enable, scheduler pause |
| selected H-Zone editor | selected zone bounds and rate | `+` / `-` mutates zone rate |

### What the viewer shows for an agent

The side inspector reads from canonical runtime state, not from a separate cached mirror. For a selected live agent it can show:

- slot and UID together when migration visibility is enabled
- bloodline family
- age, birth tick, lineage depth
- brain/trait/anchor parent UIDs
- HP, position, mass, vision, metabolism
- parameter count of the selected brain
- when inspector enrichment is enabled:
  - trait budget and trait allocation breakdown
  - PPO env/update/optimizer/truncated-rollout counters
  - catastrophe active/survived counts for that UID

That means the inspector is a real operational surface for identity, lineage, PPO, and catastrophe debugging.

### What the catastrophe overlay actually draws

When catastrophe overlay display is enabled, the viewer can add:

- a general active-catastrophe tint
- a safe rectangle for **The Thorn March**
- a front line marker for **The Woundtide**

This overlay is presentation-only. The actual shock semantics come from the catastrophe manager and its runtime modifiers, not from the renderer.

### What the viewer is not allowed to affect

The viewer does not directly edit policy parameters, PPO buffers, registry ownership maps, or checkpoint bundles.

It does, however, affect run state in three limited ways:

1. stepping control changes when engine ticks occur
2. catastrophe hotkeys route to `CatastropheManager`
3. selected H-Zone editing changes a zone’s rate on the grid

That distinction is important. Many UI toggles are visibility-only, but the viewer is not a fully read-only instrument panel.

### Operator controls that matter in practice

The UI exposes many keys, but the working distinction is this:

- **navigation controls**: pan, zoom, fit world
- **tick controls**: pause, single-step, speed multiplier
- **visibility controls**: rays, HP bars, H-Zones, grid, catastrophe panel
- **semantic controls**:
  - selected H-Zone `+` / `-` changes zone rate
  - `F1..F12` trigger catastrophe roster entries by roster index
  - `C` clears active catastrophe state
  - `Y`, `U`, `O` alter catastrophe scheduling mode/auto/pause

If you are auditing “why did this run diverge?”, remember that the viewer contains both non-semantic toggles and semantic controls.

## Telemetry, ledgers, and export surfaces

Telemetry in this repository is richer than a simple scalar logger. The logger maintains durable ledgers, snapshots, lineage export, catastrophe exposure tracking, and buffered parquet writes.

### What telemetry writes

At run creation time, the run directory also receives:

- `config.json`
- `run_metadata.json`

The `DataLogger` then manages these main artifact surfaces:

| Artifact | Format | Practical purpose |
|---|---|---|
| `simulation_data.hdf5` | HDF5 | dense snapshots and heatmaps |
| `brains/brains_tick_<tick>.pt` | Torch bundle | active brains by UID plus UID-to-slot and family mapping |
| `birth_ledger.parquet` | Parquet | birth events |
| `genealogy.parquet` | Parquet | backward-compatible alias surface for birth/genealogy-style data |
| `life_ledger.parquet` | Parquet | one life row per UID at death or close |
| `death_ledger.parquet` | Parquet | terminal death records with cause context |
| `collisions.parquet` | Parquet | physics event stream |
| `ppo_events.parquet` | Parquet | per-UID PPO update summaries |
| `tick_summary.parquet` | Parquet | one operator-facing summary row per emitted tick |
| `family_summary.parquet` | Parquet | per-family summary rows |
| `catastrophes.parquet` | Parquet | catastrophe start/end/clear event stream |
| `lineage_graph.json` | JSON | exported lineage graph derived from canonical UID/parent-role state |

### What the HDF5 snapshots contain

The HDF5 file includes groups for:

- `agent_snapshots`
- `heatmaps`
- `agent_identity`

Snapshot methods write:

- dense registry data
- `slot_uid` and `slot_parent_uid` identity surfaces
- density, mass, and `h_rate` heatmaps

These are snapshot artifacts, not ledgers.

### What the ledgers are trying to preserve

The code explicitly documents four telemetry invariants:

- births are logged at spawn time
- deaths are finalized before UID retirement
- life rows are emitted exactly once per UID at death or close
- lineage export is derived from the canonical UID/parent-role substrate

That is the right way to think about the logger: it is not collecting loosely related metrics. It is maintaining a durable history indexed by canonical identity.

### Summary and family rows

`log_tick_summary()` does more than print counters. It can emit:

- live population
- births and deaths this tick
- reproduction-disabled and floor-recovery flags
- catastrophe mode, active count, names, next tick, pause state
- PPO buffer and optimizer counts
- max alive lineage depth
- buffered parquet row count
- physics summary counts
- mean HP ratio, mass, vision, metabolism
- family counts
- per-family births, deaths, mean HP ratio, mean lineage depth, cumulative PPO update count

Two cadence controls matter operationally:

- `TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS`
- `TELEMETRY.FAMILY_SUMMARY_EVERY_TICKS`

And one cost control matters more than it first appears:

- `TELEMETRY.SUMMARY_SKIP_NON_EMIT_WORK`

That flag prevents summary aggregation work on ticks that will not emit summary rows.

### Buffered parquet behavior

Ledger rows are buffered in memory and flushed per-ledger once the buffer reaches `TELEMETRY.PARQUET_BATCH_ROWS`. This is a runtime cost/visibility tradeoff:

- lower values -> faster visibility, more file I/O
- higher values -> fewer writes, more delayed visibility

`flush_parquet_buffers()` flushes all outstanding ledger buffers, and `close()` finalizes open life rows, optionally exports lineage, flushes buffers, closes HDF5, then closes all parquet writers exactly once.

### Catastrophe exposure tracking

Telemetry also tracks catastrophe exposure and survival by UID.

The logger:

- records active catastrophe IDs seen by alive UIDs
- converts recently closed catastrophe IDs into “survived” counts for still-alive UIDs
- exposes this summary back to the viewer inspector

This is observability, not control. It does not alter catastrophe semantics; it annotates them.

### What telemetry does not control

The logger observes engine, registry, PPO, and catastrophe state after the simulation has evolved. It is not part of policy selection, combat resolution, or checkpoint ownership logic.

That said, telemetry is still operationally significant because:

- it can add hot-path cost
- it writes schema-bearing artifacts
- it influences how much evidence you retain after a run

## Checkpoint publication, restore, and validation boundaries

Runtime checkpoints are treated here as a substrate-level persistence surface, not as a casual “save model weights” shortcut. The bundle captures enough state to reconstruct identity ownership, active brains, PPO state, grid field, catastrophe state, RNG state, and engine counters.

### When scheduled runtime checkpoints are published

The engine publishes scheduled runtime checkpoints only after the tick has settled.

In the observed step order, checkpoint publication occurs after:

- catastrophe scheduling and application
- physics step
- environment effects
- death processing
- telemetry death finalization
- evolution death processing
- respawn step
- invariant checks
- tick summary logging
- tick increment
- PPO update attempt
- snapshot attempt

Then `_maybe_save_runtime_checkpoint()` publishes the runtime checkpoint for the post-settle tick.

That ordering is deliberate. The code comment on the scheduler says the checkpoint is meant to observe the runtime only after physics, deaths, births, and PPO state settle.

### Checkpoint file-set layout

```text
<run_dir>/
├── config.json
├── run_metadata.json
└── checkpoints/
    ├── runtime_tick_00000128.pt
    ├── runtime_tick_00000128.pt.manifest.json
    └── latest_checkpoint.json
```

The filename pieces come from:

- `CHECKPOINT.DIRECTORY_NAME`
- `CHECKPOINT.FILENAME_PREFIX`
- `CHECKPOINT.BUNDLE_FILENAME_SUFFIX`
- `CHECKPOINT.MANIFEST_FILENAME_SUFFIX`
- `CHECKPOINT.LATEST_POINTER_FILENAME`

### What a runtime checkpoint bundle contains

The captured bundle includes these top-level surfaces:

- `checkpoint_schema_version`
- `schema_versions`
- `config_snapshot`
- `engine_state`
- `registry_state`
- `grid_state`
- `brain_state_by_uid`
- `brain_metadata_by_uid`
- `ppo_state`
- `rng_state` when enabled
- `metadata`

At a practical level, those surfaces contain:

- engine tick and respawn-controller last tick
- optional serialized catastrophe state
- dense registry tensor state
- canonical `slot_uid` and `slot_parent_uid`
- `next_agent_uid`
- fitness tensor
- serialized UID lifecycle ledger
- UID family bindings
- parent-role ledger
- trait-latent ledger
- UID generation depth ledger
- grid tensor plus H-Zones and next H-Zone id
- per-active-UID brain state dicts
- per-active-UID brain family/topology metadata
- per-UID PPO buffers
- per-UID PPO training state
- per-UID optimizer states and optimizer metadata
- AMP scaler state when present
- Python / NumPy / Torch RNG state when enabled
- metadata including device, AMP enablement, and a config fingerprint

This is why checkpoint safety here is more than “model save/load”.

### What the manifest and latest pointer add

When manifest-enabled atomic publishing is used, the published file set adds:

- bundle size
- bundle SHA-256
- config fingerprint
- tick
- active UID count
- schema version
- flags for catastrophe/RNG/optimizer/buffer presence

The latest pointer adds a second operator-facing resolution surface. It records:

- checkpoint path
- manifest path
- tick
- checkpoint schema version
- active UID count
- bundle size
- bundle checksum
- config fingerprint

`load_runtime_checkpoint()` can resolve from either a bundle path, a checkpoint directory, or the latest-pointer filename.

### Atomic publish behavior

If all of the following are true:

- `ATOMIC_WRITE_ENABLED`
- `MANIFEST_ENABLED`
- `SAVE_CHECKPOINT_MANIFEST`

then checkpoint publication uses temp files in the target directory and promotes them with `os.replace`. The publish order is:

1. temp bundle write
2. temp manifest write
3. atomic replace of bundle
4. atomic replace of manifest
5. optional atomic replace of latest pointer

If those gates are not satisfied, `save_runtime_checkpoint()` falls back to plain `torch.save`.

### Restore order, and why it matters

Restore is intentionally conservative. In order, it rebuilds:

1. registry dense state and canonical UID bindings
2. active UID map and lineage ledgers
3. per-slot family bindings and brains reconstructed from saved family IDs
4. grid tensor and H-Zone state
5. engine tick and respawn-controller last tick
6. PPO buffers and training state
7. per-UID optimizers, validated against live brain topology
8. scaler state
9. RNG state
10. catastrophe state or manager reset
11. identity shadow-column sync and invariant checks

The reason this order matters is that later stages depend on earlier ownership truth.

Examples:
- optimizer state cannot be trusted until the correct UID-owned brain exists
- brains cannot be rebuilt correctly until slot-to-UID and UID-to-family bindings are restored
- catastrophe state cannot be safely reattached until the engine substrate exists

### What validation checks before restore

`validate_runtime_checkpoint()` is strict by design. Depending on flags, it checks:

- required top-level keys
- checkpoint schema version
- catastrophe and telemetry schema versions
- required registry subkeys
- rank and shape constraints on registry tensors
- consistency of lifecycle, family, parent-role, trait-latent, and generation-depth ledgers
- absence of duplicate active UIDs in slot bindings
- that active slot bindings reference known UIDs
- that `next_agent_uid` advances beyond known allocated UIDs
- active lifecycle records match active slot bindings
- `brain_state_by_uid` and `brain_metadata_by_uid` exactly match active UID bindings
- PPO state surfaces do not reference unknown UIDs
- every UID has required parent-role keys
- every saved brain topology signature matches the topology for its family in the current code
- serialized PPO buffers satisfy the expected payload schema
- catastrophe state schema matches when strict catastrophe validation is enabled
- manifest tick, active UID count, schema version, and optional config fingerprint match the bundle

This is the checkpoint trust boundary. If you change a persistence-visible surface, this validator is part of the change.

### Retention and pruning

Scheduled checkpoint retention is controlled by `CHECKPOINT.KEEP_LAST`. Pruning removes older bundle files and their matching manifest files. A non-positive keep count disables pruning.

That means retention is bundle-centric. It is not a general garbage collector for all run artifacts.

## Validation harnesses and what they prove

The repository ships a small but meaningful family of validation probes. They do not prove everything, but they do test the surfaces that most often break when ownership, checkpoint, scheduler, or schema changes drift.

### Validation matrix

| Probe / surface | Controlled by | What it actually compares or asserts | Good at catching | It does not prove |
|---|---|---|---|---|
| determinism probe | `VALIDATION.ENABLE_DETERMINISM_TESTS`, `DETERMINISM_COMPARE_TICKS` | two fresh runtimes stepped for the same tick count; compares runtime signatures including UID-slot map, registry/grid digests, brain state digests, catastrophe schedule status, PPO counters | hidden nondeterminism, RNG misuse, scheduler drift | long-run learning quality or performance |
| resume consistency probe | `VALIDATION.ENABLE_RESUME_CONSISTENCY_TESTS` | run `pre_ticks`, save/load, restore RNG, continue both baseline and resumed paths for `post_ticks`, then compare runtime signatures | incomplete restore, bad RNG continuation, missing PPO/catastrophe state | all possible resume durations or all artifact consumers |
| catastrophe repro probe | `VALIDATION.ENABLE_CATASTROPHE_REPRO_TESTS` | compares catastrophe status traces across fresh seeded runs | scheduler nondeterminism, catastrophe planning drift | broader world determinism outside catastrophe state |
| save-load-save signature | `VALIDATION.ENABLE_SAVE_LOAD_SAVE_TESTS` | compares two checkpoint captures around a save/load roundtrip: tick, slot UID bindings, parent bindings, family ledger, generation depth, registry data, fitness, grid, buffer/training-state key sets | persistence drift, broken checkpoint idempotence, ledger loss | exact optimizer tensor equality or training quality |
| final validation suite | `VALIDATION.ENABLE_FINAL_AUDIT_HARNESS` plus individual probe flags | orchestrates the enabled probes and returns skipped vs active results | regression gating for the major persistence/determinism surfaces | exhaustive testing |
| runtime assertions | `LOG.ASSERTIONS`, identity assertion knobs | registry invariants, NaN checks, HP bounds, no-stacking, grid occupancy consistency | immediate local invariant breaks | subtle semantic drift that still satisfies local invariants |
| soak runner | command-line harness | repeated steps plus invariant checks, finite checks, PPO/optimizer UID ownership checks, optional periodic checkpoint surface validation | long-form state corruption, accumulation bugs, non-finite tensors | throughput benchmarking |
| benchmark harness | command-line harness | timed headless stepping with memory/throughput accounting and optional profiling | runtime cost regressions, obvious memory drift | semantic correctness |

### What the determinism probe compares

The determinism probe is stronger than a simple “same final tick” test. Its runtime signature includes:

- tick
- active UID to slot map
- alive slots
- full `slot_uid` and `slot_parent_uid`
- `uid_family`
- `uid_generation_depth`
- catastrophe mode and next auto tick
- digests of registry data, fitness, grid, and active brain states
- PPO update counts
- PPO buffer sizes
- optimizer UID set

That makes it a good first line of defense for ownership drift and hidden scheduling nondeterminism.

### What the resume probe protects

The resume probe protects the exact place where many systems quietly fail: the seam between saved state and resumed state.

In this repository, it is specifically protecting:

- RNG continuity
- slot/UID ownership continuity
- brain reconstruction continuity
- PPO state continuity
- catastrophe state continuity

If a resume bug exists but the immediate local code still “loads”, this probe is designed to catch the divergence.

### What the save-load-save signature protects

The save-load-save check is about surface idempotence, not about stepping after resume. It is especially useful when you are changing:

- checkpoint bundle structure
- lineage ledgers
- UID ownership plumbing
- grid state
- schema versions
- serialized buffer payloads

### What the validation suite does when disabled

The final suite does not pretend to have run when disabled. It returns explicit skipped-check payloads when the master harness or individual probes are off.

That behavior is operationally useful because it makes it obvious whether a green result means “passed” or merely “not executed”.

## Benchmark, soak, and audit surfaces

The repository contains three distinct audit-style surfaces, and they answer different questions.

### 1. Headless benchmark harness

The benchmark harness runs a reproducible headless session with dummy SDL drivers, applies a compact config override set, optionally warms up, optionally profiles, and emits a JSON result.

It lets the operator choose or override:

- tick count and warmup count
- seed
- device (`auto`, `cpu`, `cuda`)
- world size
- agent count
- walls and H-Zones
- ray count
- PPO update cadence / batch / minibatches / epochs
- summary cadence
- parquet batch size
- checkpoint cadence and retention
- output path
- `cProfile` top-N cumulative output

Its JSON payload includes:

- device
- ticks and warmup ticks
- elapsed seconds
- ticks per second
- RSS before/after/delta
- CUDA peak memory when relevant
- final tick and alive count
- last runtime checkpoint tick and path
- buffered parquet row count
- run directory
- optional profiler lines

Use it to measure cost and throughput under a controlled miniature workload. Do not confuse it with a semantic correctness harness.

### 2. Headless soak runner

The soak runner is a longer-form invariant check. It also runs headlessly, but its purpose is not timing. Its purpose is to expose gradual corruption.

It repeatedly checks:

- registry invariants
- finiteness of registry tensors
- finiteness of grid tensors
- PPO buffer UID keys are known lifecycle UIDs
- optimizer UID keys are known lifecycle UIDs
- buffer structural validity
- finiteness of all brain parameters

At a configurable cadence it also validates checkpoint surfaces by:

- capturing a bundle
- validating it
- saving it
- loading it
- checking that engine tick, `slot_uid`, and grid shape survive roundtrip

That makes the soak runner the right surface for “does this slowly go bad?” rather than “how fast is it?”

### 3. Final validation harness

The final validation harness is the library-level composition of determinism, resume, catastrophe reproducibility, and save-load-save checks. It is closer to an integration proof surface than to a benchmark or soak.

### Test-backed audit signals also present in the repository

The observed tests include direct checks for:

- checkpoint roundtrip restoring catastrophe state
- Veil of Somnyr affecting the canonical vision feature without mutating unrelated canonical self features
- viewer catastrophe hotkeys routing to the catastrophe manager
- final validation-suite behavior under config flags

These tests are not a replacement for the main probes, but they are evidence that those surfaces are expected to remain stable.

## Where to modify what safely

The table below is the shortest safe route from change intent to ownership surface.

### Change intent -> where to modify

| Change intent | Primary ownership surface | Also re-check before committing |
|---|---|---|
| I want to change survival pressure | `tensor_crypt.world.physics`, `tensor_crypt.runtime_config.PHYS`, sometimes `GRID`/`CATASTROPHE` | death reasons, tick summaries, catastrophe multipliers, no-stacking and HP-bound assertions |
| I want to change map density or baseline field layout | procedural map generation plus `MAPGEN` / `GRID` config | initial spawn viability, H-Zone overlays, floor-recovery behavior, benchmark assumptions |
| I want to add a new observation feature | `tensor_crypt.world.observation_schema`, `tensor_crypt.world.perception` | `PERCEPT` dimensions, `Brain.extract_canonical_observation`, checkpoint compatibility, validation probes, tests |
| I want to alter brain-family topology | `tensor_crypt.agents.brain` plus `BRAIN.FAMILY_SPECS` and related config | topology signatures, optimizer metadata validation, checkpoint restore, family colors/order if family set changes |
| I want to change reward gating or reward shape | `tensor_crypt.simulation.engine` reward helpers plus `PPO` config | reward validation, PPO update stability, determinism, telemetry interpretation |
| I want to change PPO state or update logic | `tensor_crypt.learning.ppo` | serialized buffer payloads, optimizer validation, training-state serialization, checkpoint restore, final validation suite |
| I want to change reproduction or mutation behavior | `tensor_crypt.population.respawn_controller`, `tensor_crypt.population.reproduction`, `tensor_crypt.population.evolution` | lineage ledgers, parent-role semantics, extinction behavior, birth/death telemetry, catastrophe mutation overrides |
| I want to add or change a catastrophe type | `tensor_crypt.simulation.catastrophes` plus `CATASTROPHE.TYPE_*` config | reversible override semantics, checkpoint serialize/restore, catastrophe repro probe, viewer overlay/status, catastrophe event ledger |
| I want to change viewer overlays or inspector content | `tensor_crypt.viewer.panels`, `tensor_crypt.viewer.main`, `tensor_crypt.viewer.input` | whether the change is presentation-only or semantic, migration visibility flags, inspector enrichment, tests |
| I want more telemetry detail | `tensor_crypt.telemetry.data_logger`, `tensor_crypt.telemetry.lineage_export`, run metadata helpers | parquet schema stability, close/flush behavior, batch cost, schema version stamps, downstream consumers |
| I want stronger checkpoint safety | `tensor_crypt.checkpointing.atomic_checkpoint`, `tensor_crypt.checkpointing.runtime_checkpoint`, `CHECKPOINT` config | manifest/pointer load path, strict validation flags, retention, save/load tests, run metadata |
| I want a new validation probe | `tensor_crypt.audit.final_validation` and associated tests | runtime signature scope, factory determinism, checkpoint path hygiene, suite flag behavior |
| I want to change compatibility or migration behavior | `IDENTITY`, `MIGRATION`, legacy observation adapter surfaces, compatibility wrappers | UID ownership invariants, shadow columns, viewer/log visibility, checkpoint ledgers, restore assumptions |

### Routing rules that keep changes contained

#### If the change is semantic, start near the subsystem that owns the rule

Examples:
- collision damage: physics
- catastrophe front geometry: catastrophe manager
- birth placement search: reproduction / respawn
- observation feature ordering: observation schema

Do not start by changing telemetry or the viewer to “simulate” a semantic change.

#### If the change is checkpoint-visible, start by deciding whether compatibility is supposed to break

This is especially true for:

- observation dimensions
- brain topology
- schema version bumps
- optimizer state structure
- UID ownership meaning
- catastrophe state structure

If compatibility is intentionally broken, say so in schema and validation surfaces. If it is not intentionally broken, the checkpoint and validation code must be updated together.

#### If the change is viewer-facing, decide whether it is read-only or a control

This repository contains both.

- overlays and legend visibility are read-only presentation
- catastrophe hotkeys and H-Zone rate editing are runtime controls

Treat those classes differently.

## Common failure modes

### 1. Changing semantics when you only meant to change visibility

This is a real risk in the viewer.

Examples:
- using selected H-Zone `+` / `-` edits the live zone rate
- catastrophe hotkeys alter the catastrophe manager, not just the panel
- changing how catastrophe state is displayed is safe only if you stay in the renderer and panels

### 2. Drifting config, schema, and checkpoint surfaces out of agreement

The repository writes schema versions into run metadata, telemetry, and checkpoint bundles. It also validates topology and ownership assumptions during restore.

If you change:
- canonical observation width
- family topology
- serialized PPO payload structure
- catastrophe serialized state

then config comments alone are not enough. The schema-bearing surfaces must agree.

### 3. Breaking canonical UID ownership by smuggling slot-era assumptions back in

PPO buffers, optimizers, lifecycle records, and lineage export are all UID-owned here.

Common bad moves:
- keying new persisted state by slot instead of UID
- reusing a slot before the old UID has been finalized
- assuming a slot identity is stable across death/birth cycles

The registry and checkpoint code are explicitly hardened against this.

### 4. Changing brain topology without treating it as checkpoint-visible

Brain family topology is not a private implementation detail. The code captures topology signatures per family and validates optimizer metadata against live parameter names and shapes.

Changing topology without checkpoint consideration is one of the quickest ways to make resumes silently untrustworthy.

### 5. Mutating temporary catastrophe effects as though they were baseline world state

The catastrophe design resets runtime modifiers and repaints baseline H-Zones before layering temporary overrides each tick. That is the intended reversible-shock contract.

A dangerous change is to mutate baseline field or trait state permanently when the design expects temporary override semantics.

### 6. Adding telemetry that quietly changes runtime cadence assumptions

Telemetry is buffered on purpose. A naive new write in a hot path can change the cost profile of the simulation, especially if it bypasses batching or adds repeated CPU transfers.

When adding telemetry, always decide:

- is this per-event or per-summary?
- can it be buffered?
- does it need a schema version implication?
- does it belong in parquet, HDF5, or lineage JSON?

### 7. Treating guarded surfaces as though they were fully flexible mode knobs

Several config fields exist specifically to preserve explicit compatibility surfaces. They are not invitations to assume multi-mode support.

The right pattern is:
- implement the new branch
- validate it
- test it
- then relax the guard

not the reverse.

### 8. Relaxing strict checkpoint validation without understanding what proof you are losing

Turning off strict schema, UID, PPO, or manifest validation can be useful during controlled migration work, but it also removes evidence that a resume is faithful.

Do not disable strictness merely to make a new checkpoint load.

### 9. Changing catastrophe scheduling without rerunning catastrophe reproducibility checks

Catastrophe state depends on its own seeded RNG stream, schedule planning, type selection, and duration logic. Any change there needs catastrophe repro coverage, and often determinism coverage as well.

### 10. Forgetting that run artifacts are part of the operational contract

`config.json`, `run_metadata.json`, checkpoint manifests, lineage export, tick summaries, and life/death ledgers are not disposable extras. They are part of how the system explains itself after execution.

## Maintainer checklist

Before committing, ask all of the following explicitly:

1. Did I intentionally change simulation semantics, or did I only intend to change visibility or instrumentation?
2. Did any config field change meaning, liveness, or supported values?
3. Did I change any checkpoint-visible surface: observation schema, brain topology, UID ownership meaning, PPO payloads, catastrophe state, or schema versions?
4. If checkpoint-visible surfaces changed, does save, load, restore, and validation still make sense together?
5. Did I preserve canonical UID ownership and avoid slot-based leakage?
6. Did I add telemetry or viewer features that alter hot-path runtime cost or artifact schema?
7. Did I accidentally place a semantic change in a viewer, telemetry, or compatibility layer?
8. Do determinism, resume consistency, catastrophe repro, and save-load-save probes still pass for the affected area?
9. Do run metadata, config comments, and emitted artifacts still describe reality?
10. If I touched a guarded surface, did I either preserve the guard or implement the missing runtime branch completely?
11. If I touched catastrophe logic, did I preserve reversible override semantics and checkpoint restore behavior?
12. If I touched reproduction, observation, or PPO logic, did I re-check lineage, buffer ownership, and active UID invariants?

## End-of-file recap

Tensor Crypt is operated through one authoritative config surface, but not every field in that surface is equally live or equally dangerous. The practical working split is:

- some knobs change the simulated world
- some knobs change what the learner sees
- some knobs change what the operator sees
- some knobs change what gets recorded
- some knobs decide whether saved state is trustworthy
- some knobs decide what the repository proves about itself

The repository is careful about ownership, checkpoint structure, and explicit validation. The safest way to maintain it is to preserve that discipline: route the change to the subsystem that owns it, then re-check every adjacent contract that can observe it.

## Read next

For subsystem ownership and tick ordering, return to `02_the_runtime_atlas.md`.

For observation contracts, bloodline families, learning surfaces, and lineage semantics, return to `03_the_learning_and_lineage_notebook.md`.

For world-facing mechanics and operator interpretation of what happens inside a run, return to `01_the_field_guide_to_tensor_crypt.md`.
