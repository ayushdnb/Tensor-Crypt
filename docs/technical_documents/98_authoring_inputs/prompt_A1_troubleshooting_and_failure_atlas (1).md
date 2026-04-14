# prompt_A1_troubleshooting_and_failure_atlas.md

## 1. Title and purpose

# Troubleshooting and Failure Atlas

## Purpose

This document is the repository’s fault-oriented manual for Tensor Crypt. It is written for operators, debuggers, and maintainers who need a disciplined way to diagnose failures without guessing, overgeneralizing, or silently crossing truth boundaries.

The goal is not to restate the full operator manual, checkpoint chapter, or viewer chapter. The goal is to answer a narrower question:

**When a run does not launch, does not resume, produces confusing artifacts, or violates an expected contract, what should be inspected first, what failure classes are confirmed by repository evidence, and what should be escalated rather than hand-waved away?**

This atlas is symptom-first. It prioritizes failure isolation, inspection order, and evidence-backed diagnosis.

---

## 2. Intended audience

This document is for:

- operators running the repository through the canonical root entry surfaces
- debuggers isolating a fault to startup, runtime assembly, checkpointing, telemetry, or viewer layers
- maintainers changing configuration, schema, checkpoint, validation, or observation surfaces
- advanced users running reproducibility, soak, or headless benchmark workflows

This document is **not** a generic Python, CUDA, PyTorch, SDL, or operating-system troubleshooting guide. It uses those topics only where the repository itself exposes or validates them.

---

## 3. Why this document exists in addition to the core bundle

The core bundle is canonical for architecture, operator flows, telemetry, checkpointing, validation, viewer behavior, and truth boundaries. Those documents explain what the repository does.

This add-on exists because fault handling is a different reading mode.

A maintainer under pressure does not need a full conceptual tour. They need:

- a launch-stage boundary map
- a symptom-to-cause-to-inspection workflow
- a way to separate confirmed repository failure gates from outside-environment speculation
- a way to tell whether a knob is active, guarded, unread, or concept-only
- a first-response escalation order that preserves evidence instead of destroying it

This atlas therefore extends, but does not replace, the following core documents:

- **[10]** operator manual
- **[11]** configuration reference
- **[20]** startup / runtime assembly chapter
- **[50]** telemetry and artifacts
- **[51]** checkpointing
- **[52]** validation
- **[53]** viewer diagnostics
- **[60]** truth boundaries
- **[63]** frozen terminology / truth-boundary appendix

Where the core bundle already owns the canonical explanation, this document points back to it instead of duplicating it.

---

## 4. Evidence basis and limits

### 4.1 Evidence basis

This atlas is grounded in repository evidence from the uploaded code dump and the add-on blueprint. The inspected evidence surfaces include:

- canonical launch and runtime assembly surfaces
  - `main.py`
  - `run.py`
  - `tensor_crypt.app.launch`
  - `tensor_crypt.app.runtime`
- runtime validation gates
  - `validate_runtime_config()`
  - `validate_ppo_reward_config()`
- checkpoint capture, publish, load, manifest, and latest-pointer helpers
  - `tensor_crypt.checkpointing.atomic_checkpoint`
  - `tensor_crypt.checkpointing.runtime_checkpoint`
- determinism, resume, catastrophe reproducibility, and save-load-save validation helpers
  - `tensor_crypt.validation.final_validation`
- headless benchmark and soak harnesses
  - benchmark runner
  - soak audit runner
- observation, brain, PPO, and registry invariants
  - canonical and legacy observation adaptation / validation
  - optimizer and buffer validation
  - UID, slot, family, and grid consistency assertions
- telemetry and artifact surfaces
  - `DataLogger`
  - run directory creation
- viewer input and inspection surfaces
  - viewer input handling
  - catastrophe and reproduction-overlay UI controls
  - selection, zoom, scroll, and panel behavior
- configuration comments that explicitly classify surfaces as active runtime knobs, guarded compatibility surfaces, or currently unread / effectively dead surfaces

### 4.2 Limits

This atlas does **not** claim repository behavior that was not evidenced in the uploaded materials.

It does **not** assert:

- operating-system-specific graphics-driver failure causes beyond the repository’s own SDL and viewer separation surfaces
- installation guidance for CUDA, PyTorch, SDL, HDF5, PyArrow, or Pygame beyond what the repository directly checks or uses
- compatibility guarantees for historical checkpoints beyond the current schema and validation rules
- stack traces not present in code or tests
- test coverage beyond the validation helpers and test surfaces visible in the dump

### 4.3 Reading rule

When a symptom could be caused either by repository logic or by external environment state, this document labels only the repository-backed side as confirmed. Everything else remains outside the asserted runtime truth.

> **First inspection rule**  
> Before changing code, first classify the symptom into one of these phases:  
> **launch**, **runtime assembly**, **post-tick runtime**, **checkpoint publish/load**, **resume/validation**, **telemetry interpretation**, or **viewer interaction confusion**.

---

## 5. Main sections

### 5.1 First-response workflow

Use this order before making code changes.

#### Phase A — Preserve evidence

1. Preserve the full run directory.
2. Preserve `config.json` and `run_metadata.json` from that run directory.
3. Preserve the checkpoint bundle, manifest, and latest-pointer JSON together if checkpointing is involved.
4. Record the exact entry surface used:
   - `python run.py`
   - `python main.py`
   - benchmark harness
   - soak harness
   - a direct module import path
5. Record whether the failure happened in viewer mode or headless mode.

#### Phase B — Place the failure on the launch boundary

The canonical launch path emits useful milestones:

1. determinism setup and runtime config validation
2. run directory creation
3. printed startup diagnostics:
   - device
   - grid size
   - initial agent count
   - run directory
4. runtime assembly
5. procedural map generation
6. initial population spawn
7. viewer launch

This gives a practical boundary map:

| Last confirmed milestone | Likely layer to inspect first |
|---|---|
| nothing printed | import path, config validation, device validation, early environment mismatch |
| run directory not created | launch/bootstrap path |
| device/grid/run directory printed, then stop | runtime assembly, map generation, initial spawn, subsystem construction |
| map generation printed but no live run | spawn, engine construction, viewer launch |
| ticks begin, then stop | engine runtime, checkpoint schedule, telemetry, validation, viewer interaction |

#### Phase C — Separate viewer faults from engine faults

The repository includes headless benchmark and soak runners that set dummy SDL video and audio drivers. Use them to answer a narrow question:

**Does the fault require the interactive viewer, or does it already exist in headless runtime state?**

- If the headless harnesses fail, inspect runtime / checkpoint / validation surfaces first.
- If headless harnesses succeed but the interactive run fails, inspect viewer launch, input, or display-adjacent boundaries.

#### Phase D — Do not trust every visible knob equally

The canonical runtime configuration explicitly marks many surfaces as:

1. **Implemented runtime behavior**
2. **Guarded compatibility surface**
3. **Public but currently unread / effectively dead surface**
4. **Adjacent theory / conceptual background**

A common failure in debugging practice is to keep turning a knob that the repository itself labels as unread or guarded. This atlas treats that as a diagnosis error.

> **First inspection rule**  
> If a config change “does nothing,” check whether that knob is actually active before treating the outcome as a runtime bug. Use **[11]** and **[60]** together.

---

### 5.2 Startup failures, launch failures, and environment mismatches

### 5.2.1 Canonical start surfaces

The repository-root public surfaces are compatibility-friendly:

- `run.py` is the canonical root launch entry surface
- `main.py` is a root compatibility entry surface
- `config.py` is a compatibility wrapper over the canonical runtime config
- legacy `engine.*` and `viewer.*` imports are thin re-export surfaces, not alternate implementations

The runtime commentaries make an important point: startup logic lives in `tensor_crypt.app.launch`, and root entrypoints route there so imports do not depend on the current working directory.

### 5.2.2 Confirmed repository-backed startup gates

The following launch-time failures are explicitly enforced by repository code:

- explicit CUDA request while CUDA is unavailable
- unsupported enumerated config choices
- invalid positive/zero ordering constraints on many runtime counters and durations
- checkpoint dependency contradictions
- missing `torch.func` support when experimental family-vmap inference is enabled
- cooldown overlay enabled with no active target role

These are repository-side validation failures, not generic environment speculation.

### 5.2.3 Symptom patterns

| Symptom | Confirmed repository-backed interpretation | First inspection path |
|---|---|---|
| launch fails before viewer opens | most likely early config validation or device validation | `validate_runtime_config()` and launch prints |
| launch fails only in interactive mode | viewer/display boundary may be involved | compare `run.py` with headless benchmark or soak harness |
| launch succeeds headless but not interactively | engine may be healthy; viewer path becomes primary suspect | input/viewer/layout/panel surfaces in **[53]** |
| no run directory exists | failure likely occurred before or during bootstrap path | launch surface and `create_run_directory()` |

### 5.2.4 Recommended inspection sequence

1. confirm the exact entry surface used
2. confirm whether startup diagnostics were printed
3. confirm whether the run directory was created
4. compare interactive launch with benchmark or soak harness
5. inspect config validation before inspecting subsystem internals

### 5.2.5 Environment mismatch notes

The headless benchmark and soak runners explicitly set:

- `SDL_VIDEODRIVER=dummy`
- `SDL_AUDIODRIVER=dummy`

This gives a clean repository-backed isolation tactic:

- if dummy-SDL headless runs succeed, an interactive display boundary is plausible
- if dummy-SDL headless runs fail too, do not blame the viewer first

This is as far as repository truth goes. This document does not assert operating-system-specific display-driver causes beyond that isolation boundary.

---

### 5.3 Device and CUDA selection problems

### 5.3.1 What the runtime actually validates

The runtime explicitly rejects `SIM.DEVICE` values that begin with `cuda` when `torch.cuda.is_available()` is false.

The benchmark harness also exposes a simpler device selector:

- `auto`
- `cpu`
- `cuda`

and resolves `auto` to CUDA when available, otherwise CPU.

### 5.3.2 Confirmed failure classes

| Failure class | Confirmed by repository | Operational meaning |
|---|---|---|
| explicit CUDA request with unavailable CUDA | yes | launch should fail during runtime validation |
| AMP state capture / CUDA-specific memory fields on CPU | guarded by runtime conditions | not every CUDA-related surface is always active |
| experimental vmap inference without `torch.func` support | yes | opt-in fast path is rejected before runtime assembly |

### 5.3.3 First inspection checklist

> **First inspection rule**  
> For device faults, inspect these in order:  
> `cfg.SIM.DEVICE` → `torch.cuda.is_available()` expectation → launch mode used → whether the failure occurs in benchmark harness too.

### 5.3.4 Do not overclaim

This repository documents and validates **device selection**, **AMP enablement**, and **some CUDA-specific reproducibility setup**. It does **not** by itself prove that every CUDA error is a repository bug. Treat machine-specific driver/toolkit failures as outside the implemented runtime truth unless reproduced through repository validation gates.

---

### 5.4 Configuration validation failures and misleading knob combinations

This is one of the most important fault surfaces in the repository. The runtime deliberately rejects unsupported or misleading combinations early.

### 5.4.1 Choice-gated surfaces

The runtime enforces supported values for multiple surfaces, including:

- `SIM.DTYPE`
- grid overlap mode
- spawn mode
- metabolism form
- initial family assignment
- respawn mode
- anchor parent selector
- extinction policy
- birth HP mode
- multiple reproduction overlay policy surfaces
- PPO ownership mode
- physics tie breaker
- lineage export format
- catastrophe mode and static ordering policy

This means the presence of a public field does **not** imply general support for all possible values.

### 5.4.2 Numeric and ordering constraints

The runtime also rejects invalid counts and ranges, including but not limited to:

- non-positive tick cadences
- non-positive PPO batch / epoch / minibatch counts
- negative checkpoint cadence
- negative overlay radii and durations
- non-positive parquet batch size
- non-positive catastrophe counts or durations
- max values smaller than their corresponding min values

### 5.4.3 Misleading combination traps explicitly enforced by code

The repository contains several dependency checks that should be treated as first-class troubleshooting surfaces:

| Combination | Confirmed runtime rule |
|---|---|
| `SAVE_CHECKPOINT_MANIFEST=True` with `MANIFEST_ENABLED=False` | rejected |
| `SAVE_CHECKPOINT_MANIFEST=True` with `ATOMIC_WRITE_ENABLED=False` | rejected because manifest publication currently exists only on the atomic publish path |
| `STRICT_MANIFEST_VALIDATION=True` without manifest publication in the active runtime | rejected |
| `WRITE_LATEST_POINTER=True` without manifest publication in the active runtime | rejected |
| cooldown overlay enabled with no `APPLY_TO_*` role enabled | rejected |
| experimental family-vmap inference without `torch.func` support | rejected |

### 5.4.4 Why operators get confused here

The runtime config file is intentionally documentation-heavy and honest about some surfaces being:

- active
- guarded
- unread / effectively dead
- schema-sensitive

A common diagnosis failure is to read a public field name as “fully implemented.” That is not a safe assumption in this repository.

> **First inspection rule**  
> If a knob is not taking effect, first ask:  
> **Is this an active runtime knob, a guarded compatibility surface, or an unread surface?**

### 5.4.5 Escalation criterion

Escalate to maintainers when:

- a value documented as supported is rejected unexpectedly
- a validation dependency contradicts the checkpointing chapter **[51]** or config reference **[11]**
- a knob classified as active is clearly unread in practice

Do **not** escalate merely because a currently unread compatibility knob has no visible runtime effect.

---

### 5.5 Checkpoint publish, load, manifest, and latest-pointer failures

Checkpoint handling is intentionally strict. This repository does not treat checkpoint files as loose blobs. It treats them as a file set with schema, ownership, topology, and manifest relationships.

### 5.5.1 File-set model

On the atomic publish path, the runtime can publish:

- bundle file
- manifest JSON
- latest-pointer JSON

The publish order is conservative:

1. write temp bundle
2. build temp manifest
3. atomically replace bundle
4. atomically replace manifest
5. optionally write latest pointer

This means troubleshooting should reason about the **file set**, not just the `.pt` file.

### 5.5.2 Confirmed publish/load failure classes

#### File presence and path failures

- bundle file missing
- manifest missing when validation expects it
- latest pointer missing

#### Pointer/manifest disagreement failures

- latest pointer tick differs from manifest tick
- latest pointer checksum differs from manifest checksum
- latest pointer byte count differs from manifest byte count
- manifest bundle filename does not match observed bundle path when strict directory validation is enabled

#### Integrity failures

- bundle size mismatch against manifest
- checksum mismatch against manifest

#### Bundle schema failures

- required top-level checkpoint keys missing
- checkpoint schema mismatch
- catastrophe or telemetry schema version mismatch
- missing registry state keys
- invalid tensor ranks or shape mismatches under strict schema validation

#### Ownership and topology failures

- inconsistent lineage ledgers
- duplicate active UIDs in slot bindings
- active lifecycle records disagree with slot bindings
- brain-state and brain-metadata UID sets do not match active UIDs
- unknown UIDs referenced by PPO state surfaces
- invalid family binding
- missing parent-role keys
- topology signature mismatch for family-owned brains
- optimizer metadata mismatches
- optimizer tensor-state shape mismatches
- serialized PPO buffer schema mismatch or ragged payload

### 5.5.3 First inspection checklist

> **First inspection rule**  
> For any checkpoint fault, inspect in this exact order:  
> **path → file-set completeness → manifest/pointer agreement → schema version → UID ownership → brain topology → optimizer/buffer continuity**.

### 5.5.4 Practical diagnosis table

| Symptom | First thing to inspect | Why |
|---|---|---|
| “latest” path does not resolve | latest-pointer JSON presence and contents | directory or pointer workflows depend on that file |
| bundle exists but load fails immediately | manifest presence and strict-manifest setting | load path may validate manifest before restore |
| pointer exists but resolves to wrong bundle | pointer path, relative-vs-absolute bundle path, pointer tick and checksum | pointer resolution re-validates against manifest |
| restore fails after bundle load | checkpoint schema, lineage ledgers, active UID surfaces, topology metadata | restore is ownership-safe and intentionally conservative |
| resume works only after disabling strict checks | manifest/schema/ownership contradiction likely exists | strict flags are surfacing a real incompatibility, not random noise |

### 5.5.5 Latest-pointer caveat

`WRITE_LATEST_POINTER` is **not** a free-standing feature. It requires manifest publication in the active runtime. This is explicitly enforced.

Therefore, a missing or disabled latest-pointer workflow may be intentional if manifest publication is also disabled.

### 5.5.6 Retention caveat

The engine prunes older periodic runtime checkpoints according to `KEEP_LAST`. The prune routine removes old bundles and their manifest files. When diagnosing “missing old checkpoints,” inspect retention first before assuming corruption.

### 5.5.7 Cross-link ownership

Canonical checkpoint semantics belong in **[51]**. This section is fault-oriented and should not replace that chapter.

---

### 5.6 Resume and determinism failures

### 5.6.1 What the repository actually does for determinism

The runtime seeds:

- Python random
- NumPy
- Torch CPU
- Torch CUDA all-devices when CUDA is available

It also sets deterministic CuDNN behavior and disables CuDNN benchmarking when CUDA is available.

Checkpoint capture can also persist RNG state. Restore can reinstate that RNG state.

### 5.6.2 Validation surfaces available in-repo

The repository includes explicit helpers for:

- determinism probe
- resume-consistency probe
- catastrophe reproducibility probe
- save-load-save surface signature check
- headless soak audit with periodic checkpoint validation

These are not generic testing slogans. They are concrete repository tools and should be used before theorizing about nondeterminism.

### 5.6.3 Confirmed resume-sensitive surfaces

A faithful resume depends on more than the registry tensor payload. The checkpoint bundle can capture:

- registry data
- slot-to-UID bindings
- lifecycle ledgers
- brain state by UID
- brain topology metadata by UID
- PPO buffers by UID
- PPO training state by UID
- optimizer state by UID
- scaler state
- RNG state
- catastrophe state
- respawn overlay runtime state

Therefore, resume faults should be treated as **multi-surface continuity faults**, not merely “the `.pt` file loaded.”

### 5.6.4 Common confirmed fault classes

| Fault class | Why it matters |
|---|---|
| RNG state not captured or not restored in a workflow that expects deterministic continuation | resumed future can diverge even if state tensors load |
| optimizer continuity missing or invalid | training continuation can drift even if inference state loads |
| bootstrap PPO tails not preserved | partially accumulated rollouts can resume incorrectly |
| catastrophe state not preserved where expected | scheduler-driven future can diverge |
| respawn overlay runtime state not preserved | reproduction constraints can diverge after restore |
| topology signature or family mismatch | restore is blocked because the live brain no longer matches stored assumptions |

### 5.6.5 First inspection checklist

> **First inspection rule**  
> For resume mismatch, inspect in this order:  
> `CAPTURE_RNG_STATE` → optimizer/state capture flags → bootstrap capture flag → catastrophe persistence setting → validation report from determinism/resume probes.

### 5.6.6 Important honesty note

A resumed run differing from an uninterrupted run is not, by itself, enough evidence to name the cause. Use the built-in probes first. The repository already exposes a better diagnostic path than manual guesswork.

### 5.6.7 Headless audit recommendation

When a determinism or resume fault is suspected:

1. reproduce it in the headless audit path
2. run the determinism and resume-consistency helpers
3. compare bundle-level continuity before changing simulation code

That escalation order is far cheaper than debugging viewer behavior or theory-level explanations first.

---

### 5.7 Shape mismatch and observation / brain contract failures

This repository treats observation and brain contracts as strict runtime surfaces.

### 5.7.1 Canonical observation path

The bloodline-aware MLP brain expects canonical observations with:

- canonical rays
- canonical self features
- canonical context features

These are validated for rank, shape, and shared batch dimension.

### 5.7.2 Legacy fallback boundary

Legacy observation adaptation exists only as a guarded compatibility path. It is not a license for arbitrary input drift.

If canonical keys are missing:

- the runtime may adapt from legacy keys **only if** legacy fallback is enabled
- otherwise the brain requires canonical observations

The legacy adapter also validates exact legacy feature widths.

### 5.7.3 Confirmed failure classes

- canonical ray tensor rank/shape mismatch
- canonical self tensor rank/width mismatch
- canonical context tensor rank/width mismatch
- canonical batch mismatch across the three canonical surfaces
- missing canonical keys when legacy fallback is not allowed
- missing required legacy keys when adaptation is attempted
- legacy tensor rank mismatches
- legacy feature-width mismatches
- unsupported family ID
- unsupported activation name
- invalid split-input widths for families that require split inputs

### 5.7.4 Practical diagnosis table

| Symptom | First thing to inspect |
|---|---|
| observation-related `ValueError` during inference | canonical shapes against current config constants |
| `KeyError` on observation keys | whether canonical keys exist, and whether legacy fallback is enabled |
| checkpoint restore reaches brain construction and then fails | family/topology metadata versus live family spec |
| optimizer restore fails after brain load | named parameter order, shapes, and optimizer metadata |

### 5.7.5 Maintainer note

These failures are usually **schema or wiring failures**, not “training instability.” When the contract breaks before a valid forward pass or restore, treat it as a dataflow / migration bug first.

Cross-link to:

- **[11]** for the active-versus-guarded status of observation-related knobs
- **[51]** for checkpoint-visible topology invariants
- **[60]** and **[63]** for truth boundaries around compatibility and migration language

---

### 5.8 PPO state, UID ownership, and optimizer continuity failures

The repository is explicit: PPO ownership is UID-based, not slot-based.

This has direct troubleshooting consequences.

### 5.8.1 Confirmed ownership invariants

- rollout buffers belong to canonical UIDs
- optimizer state belongs to canonical UIDs
- active slot lookup is only a runtime locator, not the ownership definition
- slot reuse must not silently recycle UID-owned training state

### 5.8.2 Confirmed failure classes

- PPO state exists for a UID with no active slot binding
- PPO state exists for an inactive UID
- slot brain does not belong to the UID that claims it
- serialized buffer payload is ragged or schema-mismatched
- bootstrap state is missing for a non-terminal active tail where required
- optimizer state param-group counts do not match current brain topology
- optimizer parameter IDs duplicate or mismatch
- optimizer tensor-state shapes do not match current parameter shapes
- non-finite tensors appear in stored rollout or update computations

### 5.8.3 Inspection order

> **First inspection rule**  
> For PPO continuity faults, inspect:  
> **UID binding → active slot ownership → buffer schema → bootstrap tail state → optimizer metadata → parameter shapes**.

### 5.8.4 Why this matters operationally

A checkpoint can be structurally present and still be unusable for faithful training continuation if PPO ownership or optimizer metadata is invalid. Do not equate “bundle loaded from disk” with “training continuity is preserved.”

---

### 5.9 Telemetry and artifact-interpretation failures

Telemetry problems are often not runtime failures. Many are interpretation failures.

### 5.9.1 What a run directory actually contains

Run directory creation writes:

- a timestamped run directory under the configured log root
- `config.json`
- `run_metadata.json`
- subdirectories such as `snapshots`, `brains`, and `heatmaps`

The logger additionally owns artifacts such as:

- `simulation_data.hdf5`
- `birth_ledger.parquet`
- `genealogy.parquet`
- `life_ledger.parquet`
- `death_ledger.parquet`
- `collisions.parquet`
- `ppo_events.parquet`
- `tick_summary.parquet`
- `family_summary.parquet`
- `catastrophes.parquet`
- `lineage_graph.json`

### 5.9.2 Buffered-write interpretation trap

The logger buffers parquet rows and flushes them at controlled boundaries. Therefore:

- the on-disk artifact set may lag the in-memory state during a live run
- a nonzero buffered-row count during a running benchmark is not, by itself, corruption
- final artifact expectations should be checked **after** proper close/flush boundaries

### 5.9.3 Shutdown interpretation trap

`DataLogger.close()` behaves differently depending on whether a registry is supplied.

When a registry is supplied and open-life flushing is enabled, close can:

- finalize still-open life rows
- export lineage
- flush buffered ledgers

When close is called **without** a registry, flush still happens, but open-life finalization and lineage export behavior are not equivalent.

This matters because not all harnesses close the logger in the same way.

### 5.9.4 Confirmed artifact-interpretation failure classes

| Symptom | Confirmed repository-backed interpretation |
|---|---|
| a ledger file is present but fewer rows exist than expected during a live run | buffered writes may not yet be flushed |
| lineage export is missing or incomplete after a special harness | inspect whether close happened with registry context |
| operator expects every brain snapshot on every tick | snapshots are cadence-driven, not continuous |
| “missing checkpoint” is reported from telemetry context | checkpoint retention may have pruned older bundles |

### 5.9.5 First inspection checklist

> **First inspection rule**  
> For telemetry confusion, inspect:  
> **artifact path → flush boundary → close path used → cadence knobs → retention rules**.

### 5.9.6 Cross-link ownership

Canonical artifact semantics belong to **[50]**. This section covers only failure and interpretation traps.

---

### 5.10 Viewer, input, and diagnostics confusion

Not every viewer complaint is a rendering bug. Some are input-mode or state-selection misunderstandings.

### 5.10.1 Confirmed interaction semantics

The viewer input layer supports repository-backed controls such as:

- `Escape` to exit
- `Alt+Enter` to toggle fullscreen
- `Space` to pause/unpause
- `.` to advance one tick while paused
- `+` / `=` and `-` for either simulation speed or selected-zone rate adjustment
- `r` to toggle rays
- `b` to toggle HP bars
- `h` to toggle heal/harm zones
- `g` to toggle grid
- `f` to fit camera to world
- `i`, `y`, `u`, `o`, and function keys for catastrophe control surfaces where enabled
- `Shift+1`, `Shift+2`, `Shift+3` to toggle reproduction overlay doctrine overrides when overlay hotkeys are enabled
- `Shift+0` to clear reproduction overlay overrides
- mouse wheel over the world to zoom
- mouse wheel over the side panel to scroll
- left click to select nearest visible agent, then cell occupant, then heal/harm zone

### 5.10.2 Common confirmed confusion traps

| Symptom | Confirmed interpretation |
|---|---|
| `+` or `-` changes something unexpected | behavior depends on whether a heal/harm zone is selected |
| mouse wheel does not zoom | it scrolls the side panel when the pointer is over that panel |
| `.` does nothing | single-step is only active while paused |
| catastrophe overlay visibility changes unexpectedly | panel visibility and overlay enablement are linked |
| reproduction overlay hotkeys seem dead | viewer hotkeys for that subsystem can be disabled |

### 5.10.3 Inspector and panel honesty rule

The configuration explicitly marks some viewer surfaces as presentation-only enrichment. A missing panel detail is not, by itself, a simulation bug.

### 5.10.4 First inspection checklist

> **First inspection rule**  
> For viewer confusion, inspect:  
> **selection state → pointer location → pause state → panel visibility → relevant viewer enable flags**.

### 5.10.5 Cross-link ownership

Canonical viewer behavior belongs in **[53]**. This section is only the quick-fault layer.

---

### 5.11 Symptom-to-cause-to-inspection quick atlas

The table below is intentionally compressed for first-response use.

| Symptom | Confirmed cause class | First inspection path | Escalate when |
|---|---|---|---|
| explicit CUDA config fails immediately | device validation gate | `SIM.DEVICE`, CUDA availability, entry surface | same config succeeds in one canonical path and fails in another |
| config value is accepted in file comments but rejected at runtime | guarded compatibility surface | `validate_runtime_config()`, **[11]**, active-vs-guarded status | docs and runtime disagree |
| latest checkpoint resolution fails | missing or inconsistent latest pointer / manifest | pointer JSON, manifest, bundle, strict manifest policy | pointer was produced by canonical publish path but disagrees anyway |
| checkpoint loads from disk but restore fails | schema / UID / topology / optimizer mismatch | checkpoint validation, topology metadata, PPO ownership | restore contradicts **[51]** without intentional migration |
| resumed run diverges from uninterrupted run | continuity surface mismatch | determinism probe, resume-consistency probe, capture flags | validation suite passes but real workload still diverges reproducibly |
| observation forward fails | canonical or legacy contract mismatch | observation keys and shapes against current config constants | same contract succeeds in one brain family and fails in another unexpectedly |
| telemetry artifact seems incomplete | buffered write or close-path interpretation issue | flush boundary, close path, cadence | artifacts remain incomplete after clean close |
| viewer control feels inconsistent | stateful input semantics | selection, pause state, pointer location, enable flags | control mapping contradicts viewer code or **[53]** |

---

### 5.12 First-response checklists and escalation paths

#### Operator checklist

1. preserve run directory
2. preserve config and metadata snapshots
3. preserve checkpoint bundle, manifest, and latest pointer together
4. classify failure phase
5. reproduce headless if viewer is involved
6. inspect validation gates before editing code

#### Debugger checklist

1. determine whether the fault is pre-tick or post-tick
2. determine whether it is structural, continuity-related, or interaction-related
3. use built-in validation helpers before writing ad hoc probes
4. check whether the complained-about knob is active, guarded, or unread

#### Maintainer escalation path

**Escalation Level 1 — operator-facing fault**

- invalid config combination
- missing file-set artifact
- viewer interaction misunderstanding
- cadence or retention misunderstanding

**Escalation Level 2 — repository continuity fault**

- checkpoint schema disagreement
- manifest/pointer contradiction
- determinism/resume mismatch reproduced by bundled validation helpers
- UID ownership inconsistency

**Escalation Level 3 — migration / architecture fault**

- topology signature mismatch after code changes
- observation contract drift
- optimizer metadata incompatibility after family or brain refactor
- truth-boundary violation in docs or config comments

> **Escalation rule**  
> Do not “fix” a continuity failure by turning off strict validation unless the work is explicitly a migration and the migration boundary is documented in **[51]**, **[60]**, and **[63]**.

---

## 6. Figures/tables/diagrams to include or defer

### 6.1 Include

1. **Launch boundary decision tree**  
   A one-page flow from entry surface to viewer, keyed by the printed startup milestones.

2. **Checkpoint file-set diagram**  
   Bundle, manifest, latest pointer, publish order, and validation order.

3. **Resume continuity map**  
   Registry state, brain state, PPO state, RNG state, catastrophe state, and overlay runtime state.

4. **Symptom-to-cause quick table**  
   A compact operator-facing table derived from Section 5.11.

5. **Viewer control confusion table**  
   Input, hidden state dependency, and observed effect.

### 6.2 Defer

1. platform-specific CUDA installation matrix  
   Deferred because it would exceed repository truth.

2. OS-specific SDL/display troubleshooting appendix  
   Deferred because the repository only evidences the headless dummy-SDL isolation path.

3. speculative stack-trace atlas  
   Deferred because unsupported failure classes should not be invented as facts.

4. historical checkpoint migration cookbook  
   Deferred unless actual migration boundaries are defined and archived.

---

## 7. Cross-links to core bundle documents

Use this atlas with the core bundle, not instead of it.

- **[10] Operator Manual**  
  Use for normal run procedures, not fault triage.

- **[11] Configuration Reference**  
  Use to verify whether a knob is active, guarded, schema-sensitive, or unread.

- **[20] Startup / Runtime Assembly Chapter**  
  Use when diagnosing launch-order and subsystem-construction boundaries.

- **[50] Telemetry and Artifact Chapter**  
  Use for canonical artifact semantics and ledger meaning.

- **[51] Checkpointing Chapter**  
  Use for checkpoint ownership, schema, manifest, and resume semantics.

- **[52] Validation Chapter**  
  Use for the full intended role of determinism, resume-consistency, catastrophe reproduction, and save-load-save checks.

- **[53] Viewer Diagnostics Chapter**  
  Use for canonical viewer behavior, panel surfaces, and inspection semantics.

- **[60] Truth Boundaries**  
  Use whenever debugging language is starting to outrun repository evidence.

- **[63] Frozen Terminology / Truth-Boundary Appendix**  
  Use to keep terminology, partitions, and compatibility language stable during maintenance.

---

## 8. Truth-boundary notes

This document reuses the repository’s frozen truth partitions exactly.

### 8.1 Implemented runtime behavior

Use this label when the uploaded repository evidence shows the runtime really does the thing.

Examples used in this atlas:

- explicit config validation failures
- manifest/pointer cross-checking
- canonical observation validation
- UID-owned PPO state
- run-directory and artifact emission
- viewer hotkey and selection semantics

### 8.2 Guarded compatibility surface

Use this label when a public surface exists, but current runtime validation only accepts a limited subset or only one currently live path.

Examples used in this atlas:

- guarded config enums
- latest-pointer workflow dependency on manifest publication
- legacy observation fallback
- public compatibility wrappers around canonical module paths

### 8.3 Public but currently unread / effectively dead surface

Use this label when a public config or compatibility surface exists but the uploaded dump does not show it driving the current in-repo runtime.

This matters operationally because turning an unread knob is not evidence of a runtime bug.

### 8.4 Adjacent theory / conceptual background

Use this label for explanations that help readers think clearly but do not assert current implementation behavior.

This atlas minimizes adjacent theory because fault handling must stay close to repository truth.

### 8.5 Documentation honesty rule

When evidence is partial, the document must say so plainly. It must not upgrade a plausible explanation into an asserted runtime cause.

---

## 9. Maintainer notes where relevant

### 9.1 When adding new fault surfaces

When the repository adds a new validation gate, checkpoint field, artifact, or viewer control, update three places together:

1. the canonical chapter that owns the feature
2. the config reference if a knob changed status
3. this atlas if the change introduces a new operator-visible failure class

### 9.2 When changing checkpoint or schema surfaces

Any change to checkpoint-visible state should be treated as a documentation event, not only a code event.

At minimum, re-audit:

- checkpoint capture
- manifest contents
- latest-pointer compatibility
- save-load-save validation helpers
- resume-consistency probe behavior
- this atlas sections 5.5 and 5.6

### 9.3 When changing observation or brain contracts

Re-audit:

- canonical observation dimensions
- legacy fallback language
- topology signature expectations
- optimizer metadata continuity
- checkpoint validation messages
- this atlas sections 5.7 and 5.8

### 9.4 When changing viewer controls

Re-audit:

- hotkey table in **[53]**
- side-panel vs world scroll behavior
- selection semantics
- overlay toggle dependencies
- this atlas section 5.10

### 9.5 Final maintainer discipline

A troubleshooting atlas becomes dangerous when it drifts into folklore.

Every future edit should answer these questions before it is merged:

1. Is this failure class directly evidenced in code, tests, or validation helpers?
2. Is the cause confirmed, or only plausible?
3. Does the atlas preserve the active / guarded / unread / adjacent-theory partitions?
4. Does the atlas point back to the canonical core chapter instead of replacing it?

If any answer is no, the text should be revised before publication.
