# Migration and Compatibility Ledger

## 1. Title and purpose

**Document purpose.** This add-on document is the repository’s migration and compatibility ledger. It records the compatibility surfaces that are publicly exposed, the boundaries that are actively validated, the schema-visible invariants that cannot drift silently, and the safe-change protocol maintainers should follow when modifying identity, observation, PPO, telemetry, checkpoint, or wrapper layers.

This document is intentionally conservative.

It does **not** promise backward compatibility beyond what the repository actually implements and validates.
It does **not** treat public import availability as proof of full semantic compatibility.
It does **not** invent a semantic-versioning policy that the repository does not declare.

---

## 2. Intended audience

This document is for:

- maintainers changing runtime substrate, schema, or checkpoint logic
- contributors touching wrappers, bridges, validation gates, or migration-era surfaces
- operators resuming from prior artifacts and wanting to know what is actually protected
- readers of the core bundle who need a single place for compatibility-risk decisions

Beginners may use this document to understand **which surfaces are safe to treat as aliases** and **which surfaces are continuity-critical**, but the primary audience is maintainers.

---

## 3. Why this document exists in addition to the core bundle

The core bundle should remain the canonical place for architecture, identity, observation, brain, PPO, checkpointing, and truth-contract explanation.

This add-on exists because migration and compatibility questions cut **across** those chapters:

- repository-root wrappers versus canonical package modules
- legacy import re-export surfaces versus live implementation ownership
- schema stamps versus actually validated continuity rules
- checkpoint bundle content versus manifest publication versus latest-pointer semantics
- migration-era viewer/logging surfaces versus canonical runtime substrate
- safe change protocol across identity, observations, topology, optimizer state, telemetry, and published artifacts

This document therefore extends the bundle without replacing any canonical chapter.

---

## 4. Evidence basis and limits

### 4.1 Evidence basis

This ledger is grounded in repository evidence from the uploaded inputs, especially these surfaces:

- repository-root wrappers and canonical launch/config bridges
- legacy `engine.*` and `viewer.*` compatibility packages
- `tensor_crypt.app.runtime.validate_runtime_config`
- `tensor_crypt.agents.brain`
- `tensor_crypt.agents.state_registry`
- `tensor_crypt.learning.ppo`
- `tensor_crypt.checkpointing.atomic_checkpoint`
- `tensor_crypt.checkpointing.runtime_checkpoint`
- validation harness helpers for determinism, resume, catastrophe reproducibility, and save-load-save comparison
- tests covering manifest publication, latest-pointer resolution, strict manifest failure, duplicate active UID rejection, topology-signature mismatch rejection, and periodic checkpoint retention behavior
- configuration comments that explicitly classify surfaces as active, guarded compatibility, or currently unread/effectively dead

### 4.2 Evidence limits

This document is **not** a release history.
It is **not** a claim that older external artifacts from unknown repository states will resume successfully.
It is **not** proof that every public wrapper has permanent support guarantees.

Where the repository exposes a surface but the runtime validates only a restricted subset, this document classifies that surface as **guarded compatibility**, not full support.

Where a config field is publicly present but the config commentary says no direct runtime read was found, this document classifies that field as **public but currently unread / effectively dead**, not active behavior.

### 4.3 Frozen truth partitions

This document reuses the bundle’s truth partitions exactly:

1. **Implemented runtime behavior**
2. **Guarded compatibility surface**
3. **Public but currently unread / effectively dead surface**
4. **Adjacent theory / conceptual background**

Those labels are used literally below.

---

## 5. Main sections

### 5.1 Compatibility surface map: root wrappers versus canonical modules

#### 5.1.1 Canonical implementation root

**Implemented runtime behavior**

The canonical implementation lives under `tensor_crypt`.
The package-level description explicitly states that repository-root `config.py`, `run.py`, and `main.py` remain public entry surfaces, while legacy `engine.*` and `viewer.*` imports remain available through thin compatibility modules.

That means migration work should treat `tensor_crypt` as the implementation authority and treat wrapper layers as public access surfaces, not as independent behavior definitions.

#### 5.1.2 Repository-root public surfaces

| Public surface | Observed role | Compatibility class | Migration interpretation |
|---|---|---|---|
| `config.py` | Public compatibility wrapper for canonical config | Guarded compatibility surface | Public import path remains exposed, but behavior comes from canonical runtime config |
| `run.py` | Canonical root-level start surface for repository users | Implemented runtime behavior | Root launch path intended for users; startup logic lives in `tensor_crypt.app.launch` |
| `main.py` | Repository-root compatibility entrypoint to `tensor_crypt.app.launch.main` | Guarded compatibility surface | Public start alias exists, but should not be treated as a separate launch subsystem |
| `tensor_crypt.config_bridge` | Bridge module re-exporting canonical config classes and `cfg` | Implemented runtime behavior | Internal package code consumes config through this bridge; changes here are migration-sensitive |

#### 5.1.3 Legacy compatibility packages

**Implemented runtime behavior**

Two legacy compatibility package families are explicitly present:

- `engine.*` thin re-exports into canonical `tensor_crypt` modules
- `viewer.*` thin re-exports into canonical `tensor_crypt.viewer` modules

Observed examples include re-exports for brain, evolution, grid, logger, procedural map helpers, perception, physics, PPO, registry, respawn controller, engine, camera, colors, input, layout, viewer, panels, and text cache.

**Migration rule:** these packages are evidence of **import-path continuity**, not evidence that maintainers may freely diverge wrapper semantics from canonical modules.

#### 5.1.4 Safe interpretation rule for wrappers

A wrapper is safe to describe as a compatibility layer only when all it does is re-export or forward into a canonical module.

A wrapper is **not** evidence of:

- separate implementation ownership
- separate schema ownership
- broader accepted config values
- stronger backward-compatibility guarantees than the canonical module validates

If a wrapper is removed, renamed, or materially changed, that requires an explicit migration note even when the canonical implementation is unchanged.

---

### 5.2 Legacy re-export inventory and its limits

#### 5.2.1 What is actually preserved

**Implemented runtime behavior**

The repository preserves these compatibility categories:

- repository-root launch access
- repository-root config access
- legacy `engine.*` imports
- legacy `viewer.*` imports
- legacy observation fallback inside the brain path when enabled
- migration-era logging/viewer exposure of legacy slot fields alongside canonical UID fields

#### 5.2.2 What is **not** proven by that preservation

**Truth-boundary note**

The repository evidence does **not** prove:

- indefinite support duration for every wrapper
- semantic equivalence across arbitrary past versions
- compatibility with unknown older checkpoints beyond the currently validated schema and invariant checks
- compatibility with removed or renamed fields outside the explicit fallback bridges already present

#### 5.2.3 Public import continuity versus runtime continuity

A contributor can preserve import continuity while still breaking runtime continuity.

Examples:

- keeping `engine.ppo.PPO` importable but changing UID ownership semantics
- keeping `viewer.main.Viewer` importable but removing migration-era identity display surfaces
- keeping `config.py` importable while changing checkpoint strictness dependencies or schema stamps

This document therefore separates **import compatibility** from **continuity of serialized and runtime-owned state**.

---

### 5.3 Schema-visible invariants

This section lists the repository facts that are visible in checkpoint, telemetry, validation, or other migration-sensitive surfaces.

#### 5.3.1 Identity substrate invariants

**Implemented runtime behavior**

The registry defines identity, lineage, and PPO ownership semantics in terms of **monotonic canonical UIDs**, not slot identity.

Observed invariants include:

- slot reuse must not recycle canonical ownership state
- a UID cannot be rebound once its lifecycle record is historical
- `slot_uid` and `slot_parent_uid` are canonical checkpoint-visible tensors
- `active_uid_to_slot`, `uid_lifecycle`, `uid_family`, `uid_parent_roles`, `uid_trait_latent`, and `uid_generation_depth` form a mutually consistent ledger
- dead slots cannot continue to own active UIDs
- alive slots must have instantiated brains whose `family_id` matches the family ledger
- optional legacy float shadow columns may mirror UID and parent UID for visibility, but they are a bridge, not the canonical substrate

**Migration consequence:** changes that alter UID allocation, slot binding, death finalization, family ledger meaning, or parent-role semantics are not cosmetic. They are identity migrations.

#### 5.3.2 Observation contract invariants

**Implemented runtime behavior**

The brain’s canonical observation contract is:

- `canonical_rays`
- `canonical_self`
- `canonical_context`

These tensors are shape-validated.
Batch dimensions must agree.
Per-surface feature counts must match configured canonical sizes.

**Guarded compatibility surface**

Legacy observation adaptation exists only through an explicit fallback path. When canonical keys are absent, the brain may adapt legacy keys:

- `rays`
- `state`
- `genome`
- `position`
- `context`

That bridge is gated by `BRAIN.ALLOW_LEGACY_OBS_FALLBACK` and still performs strict legacy-shape checks before mapping into canonical form.

**Migration consequence:** changing canonical feature counts, feature ordering, legacy-to-canonical mapping, or fallback enablement is observation-schema work and must not be treated as a harmless refactor.

#### 5.3.3 Brain topology invariants

**Implemented runtime behavior**

The brain module states that:

- every brain instance belongs to exactly one bloodline family
- within a family, parameter topology is fully shape-identical
- forward returns `(logits, value)` on the canonical observation contract

Checkpoint capture stores `family_id` and a `topology_signature` for each active UID.
Checkpoint validation reconstructs the expected topology from `create_brain(family_id)` and rejects mismatches.

**Migration consequence:** any family topology drift, family rename, family-order semantic change, or parameter-structure change is checkpoint-sensitive and may invalidate optimizer state, brain state, and save/load compatibility.

#### 5.3.4 PPO ownership invariants

**Implemented runtime behavior**

PPO state is UID-owned.
The repository explicitly anchors optimizer state, rollout buffers, bootstrap state, counters, and update summaries to canonical UIDs.
The runtime validates only `PPO.OWNERSHIP_MODE = "uid_strict"`.

Checkpoint validation and restore rely on this ownership model.
Optimizer restore validates current brain topology against saved optimizer metadata and, when enabled, against optimizer tensor shapes.

**Migration consequence:** changing PPO ownership away from canonical UID semantics is a major migration, not a tuning change.

#### 5.3.5 Checkpoint-visible schema stamps

**Implemented runtime behavior**

The repository writes explicit schema stamps for at least these surfaces:

- `IDENTITY_SCHEMA_VERSION`
- `OBS_SCHEMA_VERSION`
- `PPO_STATE_SCHEMA_VERSION`
- `CHECKPOINT_SCHEMA_VERSION`
- `REPRODUCTION_SCHEMA_VERSION`
- `CATASTROPHE_SCHEMA_VERSION`
- `TELEMETRY_SCHEMA_VERSION`
- `LOGGING_SCHEMA_VERSION`

Observed configured values in the uploaded repository are:

| Schema stamp | Observed value |
|---|---:|
| Identity | 1 |
| Observation | 2 |
| PPO state | 1 |
| Checkpoint | 6 |
| Reproduction | 2 |
| Catastrophe | 1 |
| Telemetry | 4 |
| Logging | 5 |

**Migration rule:** do not bump these numbers casually. A bump should mean that a reader can no longer safely assume prior serialized meaning is unchanged.

---

### 5.4 Checkpoint, manifest, and latest-pointer compatibility boundaries

#### 5.4.1 Runtime checkpoint bundle boundary

**Implemented runtime behavior**

Checkpoint capture includes at least these top-level surfaces:

- `checkpoint_schema_version`
- `schema_versions`
- `config_snapshot`
- `engine_state`
- `registry_state`
- `grid_state`
- `brain_state_by_uid`
- `brain_metadata_by_uid`
- `ppo_state`
- optional `rng_state`
- `metadata`

The registry portion includes canonical UID surfaces such as `slot_uid`, `slot_parent_uid`, `uid_lifecycle`, `uid_family`, `uid_parent_roles`, `uid_trait_latent`, and `uid_generation_depth`.

This is not a minimal checkpoint. It is a continuity-oriented substrate checkpoint.

#### 5.4.2 Strict checkpoint validation boundary

**Implemented runtime behavior**

Checkpoint validation rejects, among other things:

- missing required top-level keys
- checkpoint schema mismatch
- catastrophe schema mismatch
- telemetry schema mismatch
- missing required registry surfaces
- wrong tensor ranks or incompatible registry dimensions when strict schema validation is enabled
- duplicate active UIDs in slot bindings
- active lifecycle records that disagree with slot bindings when strict UID validation is enabled
- brain-state and brain-metadata UID sets that do not match active UIDs
- PPO checkpoint surfaces that reference unknown UIDs when strict PPO validation is enabled
- invalid bloodline family names
- missing parent-role keys
- brain topology-signature mismatch
- malformed serialized PPO buffers when strict PPO validation is enabled
- manifest/bundle disagreements when manifest validation is active

#### 5.4.3 Restore ordering boundary

**Implemented runtime behavior**

Restore is deliberately ordered.
The code restores in this sequence:

1. validate bundle
2. rebuild registry tensors and canonical UID ledgers
3. reconstruct active UID-to-slot bindings
4. recreate per-UID brains by family and load state dicts
5. restore grid state and refresh static wall cache if available
6. restore engine tick and respawn-controller runtime state
7. restore PPO buffers and training state
8. validate and restore optimizer state per active UID
9. restore scaler state if present
10. restore RNG state if enabled and present
11. restore catastrophe state if persisted, otherwise reset catastrophe runtime
12. resynchronize legacy identity shadow columns
13. reassert registry and grid invariants

**Migration rule:** changes that assume a different restore order require deliberate checkpoint-migration work. They are not harmless internal refactors.

#### 5.4.4 Manifest boundary

**Implemented runtime behavior**

The atomic checkpoint path publishes a manifest that records at least:

- checkpoint schema version
- schema-version dictionary
- tick
- timestamp
- active UID count
- artifact filenames
- artifact sizes
- optional SHA-256 checksum
- config fingerprint
- presence flags for catastrophe state, RNG state, optimizer state, and buffer state

Manifest validation checks:

- bundle file existence
- manifest file existence
- expected bundle filename when strict directory validation is enabled
- bundle size match
- checksum match when checksum validation is active

#### 5.4.5 Latest-pointer boundary

**Implemented runtime behavior**

When enabled, the atomic publish path writes a latest-pointer JSON file containing at least:

- checkpoint path
- manifest path
- tick
- checkpoint schema version
- active UID count
- bundle size
- bundle checksum
- config fingerprint

Resolution of the latest pointer validates it against the manifest before returning the bundle path. The pointer’s tick, checksum, and size may all be compared against manifest data.

**Important interpretation rule:** the latest pointer is **not** an authoritative source by itself. It is a convenience pointer whose consistency is checked against the manifest.

#### 5.4.6 Dependency boundary for manifest publication and latest pointers

**Implemented runtime behavior**

The runtime computes `manifests_published` as:

- `ATOMIC_WRITE_ENABLED`
- `MANIFEST_ENABLED`
- `SAVE_CHECKPOINT_MANIFEST`

all being true.

Observed hard dependency rules:

- `SAVE_CHECKPOINT_MANIFEST` requires `MANIFEST_ENABLED`
- `SAVE_CHECKPOINT_MANIFEST` currently also requires `ATOMIC_WRITE_ENABLED`, because manifest publication exists only on the atomic save path
- `STRICT_MANIFEST_VALIDATION` requires manifest publication in the current runtime
- `WRITE_LATEST_POINTER` requires manifest publication in the current runtime

These are not style notes. They are enforced validation dependencies.

---

### 5.5 Active versus guarded versus unread/dead config surfaces relevant to migration

This section is limited to migration-sensitive examples that are evidenced in the uploaded repository.

#### 5.5.1 Active runtime knobs relevant to compatibility and migration

| Surface | Classification | Why it matters |
|---|---|---|
| `IDENTITY.ASSERT_BINDINGS` | Implemented runtime behavior | Protects slot/UID ledger consistency during runtime and restore |
| `IDENTITY.ASSERT_HISTORICAL_UIDS` | Implemented runtime behavior | Prevents invalid parent references from silently entering the lifecycle ledger |
| `IDENTITY.MIRROR_UIDS_TO_LEGACY_FLOAT_COLUMNS` | Implemented runtime behavior | Compatibility bridge between canonical UID substrate and legacy inspection/logging surfaces |
| `BRAIN.ALLOW_LEGACY_OBS_FALLBACK` | Implemented runtime behavior | Controls whether legacy observation payloads may still drive the canonical brain path |
| `CHECKPOINT.STRICT_SCHEMA_VALIDATION` | Implemented runtime behavior | Enforces shape and container checks on checkpoint restore |
| `CHECKPOINT.STRICT_UID_VALIDATION` | Implemented runtime behavior | Enforces UID-ownership continuity |
| `CHECKPOINT.STRICT_PPO_STATE_VALIDATION` | Implemented runtime behavior | Enforces PPO checkpoint-surface sanity |
| `CHECKPOINT.VALIDATE_OPTIMIZER_TENSOR_SHAPES` | Implemented runtime behavior | Detects optimizer/topology drift |
| `CHECKPOINT.VALIDATE_BUFFER_SCHEMA` | Implemented runtime behavior | Detects serialized PPO buffer drift |
| `CHECKPOINT.ATOMIC_WRITE_ENABLED` | Implemented runtime behavior | Gates atomic publish path and manifest publication path |
| `CHECKPOINT.MANIFEST_ENABLED` | Implemented runtime behavior | Part of the file-set publication contract |
| `CHECKPOINT.STRICT_MANIFEST_VALIDATION` | Implemented runtime behavior | Makes manifest integrity part of load semantics |
| `CHECKPOINT.WRITE_LATEST_POINTER` | Implemented runtime behavior | Enables latest-pointer publication and directory-based resume resolution |
| `MIGRATION.REQUIRE_CANONICAL_UID_PATHS` | Implemented runtime behavior | Hardening knob for migration-era canonical identity discipline |
| `MIGRATION.LOG_LEGACY_SLOT_FIELDS`, `MIGRATION.LOG_UID_FIELDS`, viewer migration flags | Implemented runtime behavior | Affect migration-era observability, not canonical ownership itself |

#### 5.5.2 Guarded compatibility surfaces

| Surface | Observed accepted set | Why it is guarded rather than fully open |
|---|---|---|
| `SIM.DTYPE` | `float32` only | Other values are rejected during runtime validation |
| `AGENTS.SPAWN_MODE` | `uniform` only | Public surface exists, but validation currently accepts only one mode |
| `RESPAWN.MODE` | `binary_parented` only | Public mode selector exists, but runtime semantics remain binary parented |
| `BRAIN.INITIAL_FAMILY_ASSIGNMENT` | `round_robin`, `weighted_random` | Only these root-family assignment modes are validated |
| `RESPAWN.ANCHOR_PARENT_SELECTOR` | `brain_parent`, `trait_parent`, `random_parent`, `fitter_of_two` | Explicit validated subset |
| `RESPAWN.EXTINCTION_POLICY` | `fail_run`, `seed_bank_bootstrap`, `admin_spawn_defaults` | Explicit validated subset |
| `RESPAWN.BIRTH_HP_MODE` | `full`, `fraction` | Explicit validated subset |
| `PPO.OWNERSHIP_MODE` | `uid_strict` only | Public selector exists, but runtime enforces canonical UID ownership |
| several catastrophe and overlay policy selectors | explicit validated subsets | Public enumerated surfaces with hard validation |

**Migration rule:** do not describe these as free extension points unless the validation layer is widened too.

#### 5.5.3 Public but currently unread / effectively dead examples

The config commentary itself marks some surfaces as publicly present but not directly read in the uploaded code dump.
Examples relevant to compatibility interpretation include:

- `RESPAWN.BRAIN_PARENT_SELECTOR`
- `RESPAWN.TRAIT_PARENT_SELECTOR`
- `PPO.TRACK_TRAINING_STATE`
- `IDENTITY.ASSERT_NO_SLOT_OWNERSHIP_LEAK`
- `VALIDATION.VALIDATION_STRICTNESS`

**Important rule:** these fields are not proof of active runtime behavior in the current repository state. They are part of the public config surface, but this document treats them as unread/effectively dead where the config commentary says no direct runtime read was found.

---

### 5.6 Checkpoint/resume compatibility boundaries in plain language

#### 5.6.1 What is actually protected

The repository actively protects:

- canonical UID continuity
- per-UID brain family and topology continuity
- active slot binding continuity
- parent-role and lineage-ledger continuity
- per-UID optimizer/buffer ownership continuity
- manifest/bundle integrity when strict manifest validation is enabled
- latest-pointer-to-manifest consistency when latest-pointer resolution is used
- deterministic and resume-consistency probes through explicit validation helpers

#### 5.6.2 What is **not** automatically guaranteed

The repository does **not** automatically guarantee:

- resume across arbitrary older schema versions
- resume after observation-layout drift without migration work
- resume after family topology changes without migration work
- resume after changing UID semantics, slot-binding logic, or parent-role meaning
- resume after changing telemetry schema for downstream consumers that expect older columns or payload shapes
- compatibility merely because wrapper imports still exist

#### 5.6.3 Latest-pointer semantics

A directory resume path is not a loose convenience search. It resolves through the explicit latest-pointer file name and then validates that pointer against the manifest before trusting the referenced bundle.

That means the safe mental model is:

**bundle** = state payload  
**manifest** = published integrity and metadata record  
**latest pointer** = convenience locator that must agree with the manifest

---

### 5.7 Safe migration protocol

This protocol should be followed before merging changes that touch compatibility-sensitive surfaces.

#### Step 1 — Classify the change

Choose the smallest honest class:

- wrapper/import-path change
- canonical runtime behavior change
- schema-visible serialization change
- checkpoint publish/load change
- observation contract change
- brain topology/family change
- identity substrate change
- telemetry/logging schema change

If more than one class applies, treat it as a multi-surface migration.

#### Step 2 — Identify the authoritative layer

Ask:

- Is the canonical authority `tensor_crypt` runtime code?
- Is this only a wrapper or bridge?
- Is this a migration-era observability surface rather than canonical substrate?

Do not let wrappers become shadow authorities.

#### Step 3 — Decide whether continuity is required

Continuity-sensitive surfaces include:

- active checkpoints
- manifest consumers
- latest-pointer directory resume
- per-UID optimizer state
- per-UID PPO buffers
- telemetry files or row schemas used by downstream tooling
- migration-era viewer/logging surfaces that operators rely on during transition

#### Step 4 — Add or preserve a bridge when necessary

Examples:

- retain wrapper re-export during deprecation window
- retain legacy observation adapter until consumers are migrated
- retain legacy slot/UID visibility fields while canonical UID inspection is being adopted

A bridge should be narrow, explicit, and documented.
It should not silently broaden meaning.

#### Step 5 — Bump version stamps only when meaning changes

Version bumps are justified when serialized meaning or expected interpretation changes.
Examples:

- changing canonical observation feature layout
- changing checkpoint structure or required fields
- changing reproduction-ledger meaning
- changing telemetry row or payload schema in a backward-incompatible way

Do **not** bump schema stamps for comment-only edits or purely internal refactors that preserve serialized meaning.

#### Step 6 — Update validation before claiming support

When a migration changes continuity-sensitive surfaces, update the relevant validation path first:

- runtime config validation
- checkpoint validation
- manifest validation
- latest-pointer resolution checks
- optimizer-state validation
- buffer-schema validation
- deterministic/resume/save-load-save probes

#### Step 7 — Prove the migration with focused tests

At minimum, add or preserve tests for the affected boundary.
Useful patterns already exist in the repository:

- manifest and latest-pointer emission
- missing-manifest failure under strict validation
- corruption detection via checksum failure
- duplicate active UID rejection
- topology-signature mismatch rejection
- save-load-save signature equality
- resume-consistency probe
- deterministic trace comparison
- legacy observation fallback behavior when intentionally supported

#### Step 8 — Write explicit migration notes

Every risky change should state:

- what changed
- which truth partition it belongs to
- whether old artifacts remain supported, partially supported, or unsupported
- whether wrapper/import continuity is preserved
- whether schema stamps changed
- which validation/tests prove the new behavior

If support is partial, say so plainly.

---

### 5.8 Change classes that require explicit migration notes or version gates

| Change class | Migration note required | Schema/version gate likely required | Why |
|---|---|---|---|
| Remove or rename repository-root wrapper | Yes | Usually no | Public entry/import surface changed |
| Remove or rename `engine.*` / `viewer.*` re-export | Yes | Usually no | Import-path compatibility changed |
| Change canonical observation feature counts/order | Yes | Yes | Brain input contract and checkpoint-visible interpretation changed |
| Change legacy-to-canonical observation mapping | Yes | Often yes | Compatibility bridge meaning changed |
| Disable or remove legacy observation fallback | Yes | Not always, but often | Guarded compatibility surface narrowed |
| Change family names, family order meaning, or family topology | Yes | Yes | Brain metadata, checkpoint state, optimizer-state compatibility affected |
| Change UID lifecycle semantics or slot-binding rules | Yes | Yes | Identity substrate changed |
| Change PPO ownership away from canonical UID semantics | Yes | Yes | Rollout, optimizer, checkpoint ownership model changed |
| Change checkpoint required top-level keys or registry required keys | Yes | Yes | Bundle structure changed |
| Change manifest fields or latest-pointer semantics | Yes | Often yes | Publish/load contract changed |
| Change checksum, strict validation, or manifest dependency rules | Yes | Sometimes | Load/publish acceptance boundary changed |
| Change telemetry columns or payload schema used by operators/tools | Yes | Yes when backward-incompatible | Downstream artifact interpretation changed |
| Change migration-era logging/viewer identity fields | Yes | Usually no | Operator compatibility/inspection surface changed |

---

### 5.9 Migration-risk matrix

| Surface | Break risk | Why it is dangerous | Minimum safe response |
|---|---|---|---|
| UID substrate and slot binding | Very high | Corrupts identity, lineage, PPO ownership, and resume | Add explicit migration note, update validation, run resume/save-load-save tests |
| Canonical observation contract | Very high | Breaks brain input meaning and checkpoint/runtime continuity | Gate with schema/migration work, preserve adapter if needed |
| Brain family topology | Very high | Invalidates state dict and optimizer shape assumptions | Version/migration note plus topology validation/tests |
| PPO serialized state | High | Invalidates per-UID rollout and optimizer continuity | Update buffer/optimizer validation and resume tests |
| Checkpoint bundle structure | Very high | Breaks load path directly | Schema gate and explicit migration handling |
| Manifest/latest-pointer semantics | High | Breaks operational resume and integrity checks | Preserve dependencies and add manifest/pointer tests |
| Wrapper/import paths | Medium | Breaks users and internal imports even if runtime semantics are stable | Keep bridge or publish deprecation note |
| Migration-era viewer/logging fields | Medium | Hurts operator inspection during transition | Document narrowing and, if needed, retain temporary bridge |
| Public but unread config fields | Low by runtime effect, medium by documentation risk | They can mislead readers about support | Mark honestly as unread/effectively dead |

---

### 5.10 Common interpretation traps

1. **Trap: “Public config field means implemented mode.”**  
   Not true here. Several fields are explicitly marked as unread/effectively dead, and several public selectors are validated to a small accepted subset.

2. **Trap: “Wrapper means stable semantic support.”**  
   Not true. Thin re-exports preserve import access, not full behavioral compatibility.

3. **Trap: “Latest pointer alone proves the newest valid checkpoint.”**  
   Not true. The pointer is checked against the manifest.

4. **Trap: “Schema stamps alone provide compatibility.”**  
   Not true. The repository also uses concrete invariant validation for UIDs, topology signatures, tensor ranks/shapes, manifest metadata, and PPO payload structure.

5. **Trap: “Turning strict validation off is a migration strategy.”**  
   Not a safe default. It may be a debugging step, but it is not evidence that the migrated surface is correct.

---

## 6. Figures/tables/diagrams to include or defer

### Included in this Markdown document

- compatibility surface map table
- active/guarded/unread config classification table
- change-class gate table
- migration-risk matrix

### Strongly recommended but deferred as diagrams

1. **Compatibility surface map diagram**  
   Show `config.py`, `main.py`, `run.py`, `engine.*`, and `viewer.*` flowing into canonical `tensor_crypt` modules.

2. **Checkpoint publication diagram**  
   Show bundle temp file → manifest temp file → atomic replace → validated manifest → latest-pointer write.

3. **Restore ordering diagram**  
   Show registry → brains → grid → engine runtime state → PPO → scaler → RNG → catastrophe → invariant recheck.

4. **Strictness dependency diagram**  
   Show `ATOMIC_WRITE_ENABLED` + `MANIFEST_ENABLED` + `SAVE_CHECKPOINT_MANIFEST` as prerequisites for manifest publication, and then for strict manifest validation and latest-pointer writing.

5. **Safe-change checklist card**  
   Single-page maintainer checklist for pull requests that touch migration-sensitive code.

These are deferred because the uploaded task requested a single Markdown artifact, and no repository-native figure assets were provided.

---

## 7. Cross-links to core bundle documents

Use this add-on together with these core documents by exact ID:

- **02** — architecture/foundations context for canonical package ownership and startup structure
- **11** — config reference for the authoritative knob atlas and per-field status markings
- **20** — operational/runtime context where run metadata and artifact paths matter
- **22** — identity contracts for canonical UID ownership and slot/UID truth boundaries
- **40** — observation schema for the canonical observation contract and legacy bridge context
- **41** — brain architecture for family topology, canonical inputs, and topology-signature meaning
- **42** — PPO ownership for per-UID optimizer/buffer semantics
- **50** — validation/reproducibility context
- **51** — checkpointing for the canonical save/load chapter
- **52** — telemetry/logging context for schema-visible operational artifacts
- **63** — truth contract governing implemented behavior, guarded compatibility, unread/dead surfaces, and adjacent theory

This add-on should cite those documents rather than duplicate their full explanatory content.

---

## 8. Truth-boundary notes

### 8.1 What this document states positively

It states only what the uploaded repository evidence supports:

- wrappers and re-exports that exist
- validation gates that exist
- schema stamps that exist
- invariants that are explicitly checked
- manifest/latest-pointer dependencies that are explicitly enforced
- tests and validation helpers that corroborate those boundaries

### 8.2 What this document refuses to claim

It refuses to claim:

- unspecified backward compatibility with arbitrary older artifacts
- support for config values that validation currently rejects
- semantic equivalence merely because a wrapper import exists
- external semantic-versioning rules not declared by the repository
- migration success without corresponding validation and test evidence

### 8.3 How to speak precisely about compatibility in this repository

Preferred language:

- “public wrapper”
- “thin re-export”
- “guarded compatibility surface”
- “public but currently unread / effectively dead surface”
- “checkpoint-visible invariant”
- “manifest publication dependency”
- “latest-pointer convenience surface validated against manifest”

Avoid language such as:

- “fully backward compatible” unless a specific boundary is proven
- “supports many modes” when validation accepts only a narrow set
- “legacy compatible” without specifying whether that means imports, observations, checkpoints, or operator-facing logs/UI

---

## 9. Maintainer notes where relevant

### 9.1 Do not let wrappers become shadow authorities

If canonical behavior moves, update wrappers to continue forwarding cleanly, or document their removal. Do not duplicate logic into wrapper modules unless you are intentionally creating a new supported authority surface.

### 9.2 Keep migration bridges narrow

The best bridge is explicit and temporary.
Examples:

- a thin re-export
- a shape-checked legacy observation adapter
- legacy slot/UID visibility fields during a transition window

Avoid broad silent coercions that make it impossible to tell whether a consumer is still on the old surface.

### 9.3 Treat schema stamps and strict validation as governance tools

Schema numbers, topology signatures, optimizer metadata, manifest checks, and latest-pointer verification are the repository’s anti-drift tools. They should be updated deliberately and together.

### 9.4 Preserve proof when changing risky surfaces

When you touch compatibility-sensitive code, preserve or extend the existing proof surfaces:

- manifest emission tests
- missing/corrupt manifest failure tests
- latest-pointer resolution tests
- duplicate UID rejection tests
- topology-signature mismatch tests
- save-load-save equality tests
- resume-consistency probes
- deterministic probes

### 9.5 Safe default stance

When uncertain, prefer this order:

1. preserve canonical semantics
2. preserve published artifact integrity
3. preserve explicit bridges where the repository already has them
4. document any narrowing of compatibility plainly
5. remove a bridge only with an explicit migration note

---

## Acceptance checklist

- [x] Scope stayed inside this document’s purpose
- [x] No unsupported backward-compatibility promises were made
- [x] Wrapper surfaces were separated from canonical implementation ownership
- [x] Schema-visible invariants were listed conservatively
- [x] Checkpoint, manifest, and latest-pointer dependencies were described as enforced boundaries
- [x] Active versus guarded versus unread/dead config surfaces were distinguished where evidenced
- [x] Safe migration protocol was included
- [x] Change classes requiring migration notes or version gates were identified
- [x] Cross-links to core bundle documents were added by exact ID
- [x] Figures were either represented as tables or explicitly deferred
