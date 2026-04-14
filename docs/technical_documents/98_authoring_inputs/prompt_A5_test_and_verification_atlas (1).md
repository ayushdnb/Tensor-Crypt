# Test and Verification Atlas

## 1. Title and purpose

This add-on document maps the repository’s verification ecosystem. Its purpose is to show **how behavioral truth is checked**, **which verification surfaces exist**, **what each surface can and cannot establish**, and **how maintainers should choose the right verification path before modifying risky subsystems**.

This atlas is intentionally narrower than a full testing manual. It does not restate the whole validation chapter from the core bundle. It indexes the verification surfaces that are evidenced in the repository and explains their operational meaning.

---

## 2. Intended audience

This document is for:

- maintainers changing identity, checkpointing, catastrophe, reproduction, PPO, or observation contracts
- contributors who need to choose a safe verification path before and after edits
- researchers and operators who need to understand what the repository currently checks directly, what it corroborates indirectly, and what it only guards at runtime

---

## 3. Why this document exists in addition to the core bundle

The core bundle remains authoritative for canonical subsystem explanations. This add-on exists because the verification story is distributed across several different surfaces:

- pytest tests
- compact audit probes
- runtime checkpoint validators
- registry and PPO invariant guards
- a headless soak runner
- a headless benchmark harness
- configuration gating that rejects unsupported states before launch

Without a dedicated atlas, those surfaces are easy to confuse. In particular, maintainers can wrongly treat:

- a benchmark as semantic proof
- a round-trip restore as resumed-behavior proof
- runtime validation as equivalent to direct testing
- a short determinism probe as universal proof across all seeds, durations, and devices

This document closes that gap. It explains the verification ecosystem without duplicating the core canonical explanations of identity, tick order, PPO, checkpointing, and truth boundaries.

---

## 4. Evidence basis and limits

### 4.1 Evidence basis

This atlas is grounded only in repository evidence supplied for this task. The inspected evidence targets include:

- `tensor_crypt.audit.final_validation`
  - `run_determinism_probe`
  - `run_resume_consistency_probe`
  - `run_catastrophe_repro_probe`
  - `save_load_save_surface_signature`
  - `run_final_validation_suite`
  - `_runtime_signature`
- `tensor_crypt.checkpointing.runtime_checkpoint`
  - `capture_runtime_checkpoint`
  - `validate_runtime_checkpoint`
  - `restore_runtime_checkpoint`
  - `load_runtime_checkpoint`
  - `validate_checkpoint_artifacts`
- `tensor_crypt.checkpointing.atomic_checkpoint`
  - manifest validation
  - latest-pointer resolution
  - atomic bundle publication
- `tensor_crypt.app.runtime`
  - `validate_runtime_config`
  - `setup_determinism`
- `tensor_crypt.agents.state_registry.Registry`
  - `assert_identity_invariants`
  - `check_invariants`
- `tensor_crypt.learning.ppo`
  - buffer validation
  - optimizer-state validation
  - serialized-buffer validation
- the headless soak runner in the raw dump
- `scripts/benchmark_runtime.py`
- the pytest corpus surfaced in the raw dump, including direct tests of the audit probes, checkpoint validation, deterministic runtime behavior, benchmark smoke execution, identity invariants, catastrophe scheduling, PPO update math, reward gating, viewer hotkeys, and reproduction overlays

### 4.2 Evidence limits

> **Uncertainty box — repository-input limits**
>
> This document does **not** claim:
>
> - a coverage percentage
> - CI execution status
> - a full platform matrix
> - formal verification
> - mutation-test strength
> - property-based testing breadth
> - that every surfaced test is always run in every environment
>
> The raw dump demonstrates that these verification surfaces exist and what they check. It does not justify any stronger claim.

### 4.3 Frozen truth partitions used in this atlas

This document reuses the frozen truth partitions exactly:

1. **Implemented runtime behavior**
2. **Guarded compatibility surface**
3. **Public but currently unread / effectively dead surface**
4. **Adjacent theory / conceptual background**

### 4.4 Interpretation rule

Repository truth outranks elegance, teaching convenience, and outside testing theory. When this atlas labels a surface as narrow, guarded, or only corroborative, that judgment follows the implementation boundary rather than testing folklore.

---

## 5. Main sections

### 5.1 Verification ecosystem map

| Surface | Primary role | What it can establish | What it cannot establish |
|---|---|---|---|
| Pytest unit and integration tests | Encode direct assertions for concrete subsystem behaviors | That the asserted behaviors hold for the covered scenarios | Full coverage, universal proof, or long-horizon stability |
| `run_determinism_probe` | Compare short same-seed execution traces across fresh runtimes | That the chosen signature matches across the compared runs for the chosen tick budget | Universal determinism across all seeds, durations, platforms, or future edits outside the signature |
| `run_resume_consistency_probe` | Compare post-resume execution against uninterrupted execution | That checkpoint restore plus resumed stepping matches the uninterrupted branch for the compared horizon | That every checkpoint artifact is structurally valid in every mode or that all restore futures remain equal indefinitely |
| `run_catastrophe_repro_probe` | Compare catastrophe status traces across fresh runs | That catastrophe status planning is reproducible for the chosen trace horizon | Full world-state equivalence after every catastrophe effect |
| `save_load_save_surface_signature` | Compare structural checkpoint surfaces before and after restore-and-recapture | That selected bundle surfaces remain equal after a save-load-save cycle | Resumed forward execution equivalence by itself |
| `run_final_validation_suite` | Compose enabled audit probes into one report | A single gated report over the enabled checks | More than the enabled sub-probes actually measure |
| Headless soak runner | Long-form invariant and non-finite-state audit under a reproducible configuration | That a specific long-form run stayed finite and passed embedded invariant checks and checkpoint cadence checks | Performance truth, universal semantic truth, or broad coverage beyond its configured path |
| Headless benchmark harness | Reproducible performance and memory measurement | Throughput, memory, final tick, checkpoint cadence metadata, and inference-path counters for the measured run | Semantic correctness or behavioral proof |
| Runtime validators and invariant guards | Fail fast on invalid state, unsupported config, or malformed checkpoint payloads | That certain illegal states are rejected before or during runtime operations | That all legal states are semantically correct |

### 5.2 The direct audit probes

#### 5.2.1 Determinism probe

`run_determinism_probe` executes two fresh runtimes for a fixed tick budget and compares a structured runtime signature after every step.

The compared signature is stronger than a single scalar checksum. It includes, among other fields:

- engine tick
- active UID-to-slot bindings
- alive slots
- `slot_uid`
- `slot_parent_uid`
- `uid_family`
- `uid_generation_depth`
- catastrophe mode and next auto tick
- respawn overlay runtime state
- SHA digests of registry data, registry fitness, and the world grid
- per-active-UID brain-state digests
- PPO update counters
- PPO buffer sizes
- optimizer UID keys

This is an **Implemented runtime behavior** probe. It directly checks that two same-seed fresh runs produce matching short-horizon execution signatures.

Its limit is equally important: it is only as strong as the captured signature and chosen tick budget. It does not prove universal determinism across all horizons or all state surfaces not included in the signature.

#### 5.2.2 Resume-consistency probe

`run_resume_consistency_probe` advances a baseline runtime for `pre_ticks`, captures a checkpoint, saves and reloads it, restores RNG state when present, then compares the uninterrupted branch against a resumed branch after `post_ticks`.

This is the strongest direct repository-evidenced probe for the statement:

> **“Checkpoint restore preserves the future behavior needed for resumed execution, not only the stored bundle surface.”**

That makes it stronger than a plain round-trip equality check.

Its limit is that the proof window is finite and chosen by the caller. It also depends on the correctness of the runtime signature it compares.

#### 5.2.3 Catastrophe reproducibility probe

`run_catastrophe_repro_probe` compares catastrophe status traces across fresh runs by recording `build_status(...)` before each step.

This is a targeted probe for catastrophe scheduling and catastrophe status reproducibility. It is **not** a full world-state equality probe. It proves that the catastrophe status trace matches for the chosen horizon. It does not, by itself, prove that every downstream field mutation, reward consequence, or PPO consequence stays equal outside the compared status surface.

#### 5.2.4 Save-load-save surface signature

`save_load_save_surface_signature` captures runtime A, saves it, loads it, restores it into runtime B, recaptures, and compares selected checkpoint-visible surfaces such as:

- tick
- active UID count
- `slot_uid`
- `slot_parent_uid`
- `uid_family`
- `uid_generation_depth`
- PPO buffer key sets
- PPO training-state key sets
- registry data digest
- fitness digest
- grid digest

This probe is best understood as a **checkpoint structural idempotence** check. It is stronger than “the file loads” but weaker than “resumed future execution is identical.”

#### 5.2.5 Final validation suite

`run_final_validation_suite` is a composition layer. It does not create new proof power. It runs the enabled probes based on `cfg.VALIDATION` flags and reports:

- per-surface results
- skip reasons for disabled checks
- `all_passed`

This means the suite’s truth is bounded by the probes it actually runs. Disabling a sub-check narrows the suite accordingly.

### 5.3 Verification-adjacent operational harnesses

#### 5.3.1 Headless soak runner

The headless soak runner is a long-form audit surface, not a benchmark and not a theorem prover.

What it actively checks during execution:

- `registry.check_invariants(runtime.grid)`
- finiteness of registry state
- finiteness of grid state
- PPO buffers are keyed only by known lifecycle UIDs
- PPO optimizers are keyed only by known lifecycle UIDs
- all live brain parameters remain finite

At checkpoint-validation cadence it also:

- captures a runtime checkpoint
- calls `validate_runtime_checkpoint`
- saves the checkpoint
- reloads it
- checks that engine tick is unchanged
- checks that `slot_uid` bindings are unchanged
- checks that grid tensor shape is unchanged

This makes the soak runner an **Implemented runtime behavior** audit surface for “longer run stayed inside guarded invariants under this configuration.”

Its limits:

- it is configuration-specific
- it is horizon-limited
- it is not a coverage report
- it does not prove throughput
- the evidenced repository input did not surface a dedicated pytest smoke test for this runner

That last point matters. The runner is real and meaningful, but the current evidence set shows it mainly as an executable audit harness rather than as a separately smoke-tested script surface.

#### 5.3.2 Headless benchmark harness

`scripts/benchmark_runtime.py` is a reproducible measurement harness. It configures a runtime, warms it up, runs for a fixed tick budget, and emits structured JSON including:

- device
- tick budget
- elapsed time
- ticks per second
- RSS memory before and after
- CUDA peak memory when applicable
- final tick
- final alive count
- runtime checkpoint metadata
- buffered parquet row count
- run directory
- optional cProfile summary
- inference-path counters, including loop and vmap bucket counts

This harness is **verification-adjacent**, not a semantic validator. It can prove that the benchmark script executed, that it produced structured output, and that a given run had certain measured throughput or memory characteristics. It cannot prove that the simulation semantics are correct.

The pytest corpus does directly evidence a benchmark smoke test that executes this script and validates the output shape and several invariant fields. That is valuable. It still does not turn the benchmark into a behavioral proof surface.

### 5.4 Runtime invariant and validation surfaces

#### 5.4.1 Registry identity and occupancy invariants

`Registry.assert_identity_invariants` and `Registry.check_invariants` are central fail-fast guards.

They enforce or check, among other things:

- UID lifecycle ledger continuity
- agreement between lifecycle records and slot bindings
- one active UID per live slot
- no live slot without a bound UID
- no dead slot still owning an active UID
- family ledger agreement
- live-slot brain/family agreement
- no NaN in registry data
- non-negative HP
- HP not exceeding HP max
- grid occupancy consistency under no-stacking mode

These surfaces are crucial, but they are not substitutes for targeted tests. They reject illegal or inconsistent runtime states. They do not independently prove that the whole tick pipeline is semantically correct.

#### 5.4.2 Runtime configuration validation

`validate_runtime_config` rejects unsupported enums, invalid numeric ranges, and incompatible feature combinations before runtime assembly.

Examples include:

- unsupported config choices
- invalid PPO cadence values
- invalid checkpoint publication dependencies
- invalid catastrophe timing ranges
- missing `torch.func` support when experimental family vmap inference is enabled
- cooldown overlay enablement without any active parent-role application

This is a **Guarded compatibility surface** checker. It prevents misleading configuration states from being treated as live supported modes.

It is not a behavior proof. It says: “this configuration is currently allowed.” It does not say: “every allowed configuration is already exhaustively tested.”

#### 5.4.3 Checkpoint validation

`validate_runtime_checkpoint` is one of the repository’s strongest structural verification surfaces. It validates:

- required top-level bundle keys
- checkpoint schema version
- selected subsystem schema versions
- required registry surfaces
- tensor ranks and shape agreement under strict schema validation
- consistency of lifecycle, family, parent-role, trait-latent, and generation ledgers
- duplicate active UID rejection
- active lifecycle-record agreement with slot bindings under strict UID validation
- active UID equality with brain-state and brain-metadata surfaces
- unknown UID rejection in PPO state under strict PPO validation
- family validity
- required parent-role fields
- brain topology-signature agreement with freshly instantiated canonical family brains
- serialized PPO buffer payload validity
- catastrophe schema consistency when strict validation is enabled
- manifest and bundle agreement when a manifest is present

This is a structural truth surface for checkpoint validity. It is stronger than file existence checks and stronger than plain deserialization.

Its limit is also clear: it validates the checkpoint substrate. It does not, by itself, prove resumed future execution equality. That is why resume-consistency exists separately.

#### 5.4.4 PPO-owned validation surfaces

The PPO implementation also contains internal validation surfaces that matter for verification:

- serialized-buffer schema validation
- ragged-buffer rejection
- non-finite observation or transition rejection
- bootstrap-state consistency checks
- optimizer-topology and tensor-shape validation on restore
- UID ownership checks tying optimizers and buffers to active lifecycle UIDs

These surfaces matter because many silent training corruptions do not first appear as catastrophic runtime crashes. They appear as ownership drift, buffer drift, or shape drift. The repository guards against several of those failures directly.

#### 5.4.5 Brain observation-shape validation

The canonical observation extractor and brain forward path validate canonical tensor shapes, batch agreement, and legacy-adaptation expectations.

This is a narrow but important validation layer. It protects the canonical observation contract from silent shape drift. It does not prove that the observation semantics are scientifically correct. It proves that the shaped surface reaching the brain matches the enforced contract.

### 5.5 What the evidenced test corpus directly covers

The raw dump contains a large pytest corpus with many subsystem-specific `test_*` functions. Based on repository evidence, the test corpus directly covers at least the following verification-relevant areas.

#### 5.5.1 Direct tests of the audit probes

There are explicit tests that:

- run the determinism probe and require `match is True`
- run the resume-consistency probe and require `match is True`
- run the catastrophe reproducibility probe and require `match is True`
- run the save-load-save signature probe and require equality on key checkpoint-visible surfaces
- run the composed final validation suite and require `all_passed is True`
- verify that the suite respects per-check enable flags and correctly reports skipped checks

This is strong direct evidence that the compact audit helpers are not merely exposed; they are exercised by the test corpus.

#### 5.5.2 Checkpoint artifact and corruption handling

The test corpus directly evidences checks for:

- manifest and latest-pointer publication
- latest-pointer resolution
- checksum-corruption detection
- strict manifest validation behavior
- checkpoint round-trip behavior for multiple subsystems
- checkpoint validation rejection paths such as duplicate active UIDs and topology mismatches

#### 5.5.3 Identity, ownership, and lifecycle contracts

The test corpus directly evidences checks for:

- UID monotonicity across slot reuse
- dead UID finalization
- PPO ownership being UID-keyed rather than slot-keyed
- new child UIDs not inheriting dead optimizer state
- shadow-column agreement with canonical UID surfaces
- training counters remaining UID-owned
- inactive UID buffer truncation accounting
- checkpoint restore preserving training-state and optimizer schema surfaces

#### 5.5.4 Catastrophe scheduling and restoration

The test corpus directly evidences checks for:

- scheduler arming and replanning behavior
- deterministic auto-dynamic scheduling under fixed seed
- interval and ordering rules
- manual trigger and clear behavior
- overlap and max-concurrency behavior
- catastrophe-specific field effects
- checkpoint round-trip restoration of catastrophe state
- viewer hotkey routing into catastrophe control

#### 5.5.5 Reproduction, overlays, and parent-role logic

The test corpus directly evidences checks for:

- distinct parent-role assignment
- no parentless normal births
- child UID freshness
- brain-parent inheritance
- trait-parent mutation path
- overlay-specific crowding, cooldown, and local-parent logic
- reproduction overlay runtime-state checkpoint preservation
- extinction behavior under fail-run policy

#### 5.5.6 Engine, physics, observations, rewards, and PPO math

The test corpus directly evidences checks for:

- end-to-end engine stepping with artifact output and invariant preservation
- seeded runtime determinism
- observation-bundle context logic
- empty-batch observation shapes
- perception and wall semantics
- collision, ram, contest, and death processing
- PPO return and advantage boundary behavior
- gradient and minibatch error handling
- reward-gating validation and live reward-path behavior

#### 5.5.7 Viewer and operator-path smoke checks

The test corpus directly evidences checks for:

- viewer draw smoke paths
- resize handling
- side-panel controls
- fullscreen toggling
- catastrophe control hotkeys
- reproduction overlay hotkeys
- inspector-facing family and trait surfaces

#### 5.5.8 Benchmark smoke coverage and soak coverage boundary

The benchmark harness is directly exercised by a smoke test.

The inspected evidence does **not** surface an equivalent dedicated pytest smoke test for the headless soak runner. The soak runner still matters; the correct interpretation is that its repository-evidenced status is **executable audit harness**, not **directly smoke-tested script surface**.

### 5.6 Directly tested, indirectly corroborated, and runtime-validated truths

| Truth statement | Directly tested | Indirectly corroborated | Runtime-validated | Limit |
|---|---|---|---|---|
| Same-seed short-horizon runtime signatures match | Yes | Yes, by seeded runtime tests | No | Finite horizon and finite signature |
| Resume after restore matches uninterrupted future for the compared window | Yes | Yes, by checkpoint round-trip tests | Partly, because restore validation guards prerequisites | Finite window only |
| Checkpoint artifact set is structurally coherent | Yes | Yes, by save-load-save and resume probes | Yes, through checkpoint validators and manifest validators | Structural validity is not future-execution proof |
| Catastrophe scheduler status is reproducible | Yes | Yes, by catastrophe scheduler tests | No | Status-trace equality is narrower than full-world equality |
| Long-form run remains finite and invariant-clean under one soak configuration | Not directly surfaced as a pytest script test in the inspected input | Yes, through the soak runner logic itself | Partly, because invariant checks and checkpoint validators are embedded in the runner | Configuration-specific and horizon-limited |
| Benchmark output is reproducible enough to measure performance surfaces | Yes, via smoke test | No | No | Measurement is not semantic proof |
| Unsupported config combinations are rejected | Yes in several targeted tests | No | Yes | Rejection of invalid config is not proof of broad correctness for all valid config |
| Observation, optimizer, and buffer shape/ownership corruption are rejected | Yes in targeted tests | No | Yes | Guardrails catch known malformed states, not every scientific or semantic error |

### 5.7 How maintainers should choose the right verification surface

The smallest sufficient verification path is the right starting point. Escalate only as the risk surface expands.

| Change type | Minimum verification path | Recommended escalation | Why |
|---|---|---|---|
| UID lifecycle, slot reuse, lineage ownership, family binding | targeted pytest coverage for UID and ownership behavior | determinism probe, resume-consistency probe, checkpoint validation checks | These changes can silently corrupt identity continuity even when the sim still runs |
| Checkpoint schema, manifest logic, restore order, RNG capture, PPO restore | checkpoint artifact tests, save-load-save signature, resume-consistency probe | soak run with checkpoint cadence enabled | Structural restore validity and future resume equality are both at risk |
| Catastrophe scheduler, catastrophe persistence, catastrophe runtime modifiers | catastrophe reproducibility probe and catastrophe-focused tests | determinism probe and checkpoint round-trip catastrophe tests | Scheduler truth and persisted catastrophe state must stay aligned |
| PPO buffer layout, optimizer ownership, bootstrap logic, reward surface | PPO math tests and restore-validation tests | determinism probe and short soak run | Training can degrade silently through ownership or bootstrap mistakes |
| Tick order, physics, death finalization, reward emission | engine and physics tests | determinism probe and short soak run | These edits affect the shortest semantic path from world step to learning signal |
| Performance-only path such as experimental same-family vmap inference | equivalence tests plus benchmark smoke | determinism probe and benchmark comparison under identical settings | Performance work must not be accepted on throughput numbers alone |
| Viewer-only control or HUD work | viewer smoke tests and hotkey routing tests | catastrophe/reproduction overlay interaction tests when relevant | UI work can still break operator control paths |

### 5.8 Common interpretation traps

1. **Benchmark success is not semantic proof.**  
   A benchmark can show that the runtime runs fast, not that it runs correctly.

2. **Save-load-save equality is not resumed-future equality.**  
   Structural round-trip checks and future-behavior checks are different surfaces.

3. **Runtime validation is not test coverage.**  
   A validator only rejects known bad states. It does not automatically assert all good states.

4. **A short determinism probe is not universal determinism.**  
   It is still valuable, but only within its compared horizon and signature surface.

5. **The soak runner is not a substitute for targeted subsystem tests.**  
   It is a long-form audit, not a precise semantic locator.

---

## 6. Figures/tables/diagrams to include or defer

### 6.1 Included in this pass as Markdown tables

This document includes the following directly in Markdown form:

- verification ecosystem surface-purpose matrix
- directly tested versus indirectly corroborated versus runtime-validated matrix
- change-type to verification-choice decision table

### 6.2 Deferred visual diagram assets

The following visual assets are useful but are deferred in this pass because no separate image or slide artifact was requested:

- **Verification ecosystem map**  
  A one-page flow diagram from config validation -> runtime assembly -> step loop -> probe surfaces -> checkpoint surfaces -> soak/benchmark paths.

- **Evidence-strength ladder**  
  A compact diagram showing: runtime guard < targeted test < structural round-trip probe < resumed-future probe < long-form soak corroboration.

- **Change-risk routing diagram**  
  A decision flow for selecting the smallest sufficient verification path.

Deferred means “not produced here,” not “unnecessary.”

---

## 7. Cross-links to core bundle documents

Use this atlas together with the following core documents.

- **22 — identity contracts**  
  Read first when evaluating UID safety, slot reuse, lifecycle truth, and PPO ownership.

- **31 — tick and death order**  
  Read before judging determinism failures, death-finalization regressions, or reward-path regressions.

- **42 — PPO**  
  Read before changing buffer ownership, bootstrap semantics, optimizer restore, or training-state continuity.

- **51 — checkpointing**  
  Read before changing bundle structure, manifest rules, latest-pointer publication, restore order, or RNG persistence.

- **52 — validation and soak methods**  
  This is the closest core companion to the present atlas. The core document should remain the canonical operational explanation. This atlas exists to index the ecosystem and clarify proof limits.

- **63 — truth contract**  
  Read whenever a verification surface is at risk of being overstated. This atlas follows that document’s truth-boundary discipline.

---

## 8. Truth-boundary notes

1. **No formal verification claim is justified by the inspected repository evidence.**

2. **No coverage percentage claim is justified by the inspected repository evidence.**

3. **Benchmarks are verification-adjacent, not proof surfaces.**

4. **Soak runs are long-form corroboration under chosen settings, not universal correctness proofs.**

5. **Some public validation knobs are currently unread / effectively dead surfaces.**  
   The raw repository comments explicitly mark examples such as:
   - `VALIDATION.VALIDATION_STRICTNESS`
   - `VALIDATION.SAVE_LOAD_SAVE_COMPARE_BUFFERS`
   - `VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES`

   These public surfaces should not be described as active verification controls unless implementation evidence changes.

6. **Guarded compatibility surfaces must not be documented as fully live option spaces.**  
   Runtime config validation explicitly restricts several enums and feature combinations.

7. **Tests prove encoded assertions, not all unstated semantics.**  
   That is a strength boundary, not a weakness. The correct response is to add the missing assertion when a new semantic promise becomes important.

---

## 9. Maintainer notes where relevant

### 9.1 Before changing risky subsystems

Before editing a risky subsystem, decide which truth is actually at risk:

- structural checkpoint truth
- resumed-future truth
- short-horizon deterministic trace truth
- long-form stability truth
- performance truth
- operator control-path truth

Then choose the smallest verification surface that directly measures that truth.

### 9.2 After changing checkpoint-visible or UID-visible substrates

If a change touches any of the following, treat it as a checkpoint-and-identity-sensitive edit:

- UID ledgers
- slot bindings
- family bindings
- parent-role state
- PPO buffer ownership
- optimizer ownership
- catastrophe persisted state
- respawn overlay runtime state
- schema versions
- manifest and latest-pointer rules

For those edits, the minimum safe path is:

1. targeted tests
2. checkpoint validation path
3. save-load-save signature
4. resume-consistency probe

### 9.3 Keep signatures intentional

When a new state surface must be resume-stable or determinism-stable, decide explicitly whether it belongs in:

- `_runtime_signature`
- checkpoint capture
- checkpoint validation
- restore logic
- a dedicated targeted test

Do not let new state surfaces drift into the runtime without making that decision.

### 9.4 Treat validator failures as contract failures

A failed invariant, schema validator, ownership validator, or topology validator should be treated as a contract regression until disproved. Do not downgrade those failures to cosmetic warnings unless the underlying contract is intentionally changed and the surrounding documentation, migration story, and tests are updated together.

### 9.5 Add tests near the changed truth, not far from it

When adding or changing behavior:

- add a targeted test at the nearest subsystem boundary
- add or update a probe only when the behavior is cross-cutting
- use soak only when long-form drift is the concern
- use benchmark only when throughput or memory is the concern

This keeps the verification ecosystem sharp instead of noisy.

### 9.6 The governing interpretation rule

When in doubt, use the strongest honest wording that the repository evidence supports, and no stronger.
