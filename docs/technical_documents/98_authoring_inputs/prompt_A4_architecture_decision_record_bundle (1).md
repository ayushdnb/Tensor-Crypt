# Architecture Decision Record Bundle

## 1. Title and purpose

**Document type:** Add-on architecture decision record bundle  
**Repository scope:** Tensor Crypt / `tensor_crypt` implementation surface  
**Purpose:** Record the major architectural decisions that materially constrain safe maintenance, migration, compatibility, checkpointing, and extension work.

This document does **not** replace the core bundle. It exists to preserve the decision-level logic behind the most consequential implementation choices that the repository makes visible in code, validation, configuration comments, and compatibility surfaces.

The goal is disciplined maintenance. A future contributor should be able to answer all of the following without reverse-engineering the full repository from scratch:

- Which surfaces are canonical, and which are only compatibility bridges.
- Which invariants are so central that changing them silently would be unsafe.
- Which public-looking knobs are truly implemented, which are guarded, and which are present but effectively unread.
- When a future change is large enough that it should be recorded as a new ADR instead of being smuggled in as a routine patch.

---

## 2. Intended audience

This document is written for:

- maintainers who may refactor or extend the repository
- contributors who need to understand why some edits are more dangerous than they look
- technical reviewers auditing correctness, resume safety, and compatibility claims
- advanced readers who already know the subsystem chapters and now need the decision layer

It is also intentionally readable to a careful beginner, but it is not a beginner-first walkthrough. The core bundle remains the right starting point for first-pass learning.

---

## 3. Why this document exists in addition to the core bundle

The core bundle explains the repository in canonical chapter form. That is necessary, but not sufficient.

A maintainer also needs a compact record of the repository’s **decision boundaries**. Those boundaries are different from ordinary architecture prose:

- they explain why some public surfaces are canonical and others are constrained
- they identify which invariants are checkpoint-visible or schema-visible
- they show where the code deliberately rejects broad flexibility in favor of strictness
- they make future edits governable instead of merely possible

Without this document, a contributor could read the architecture chapters correctly and still make an unsafe change because the *decision weight* of a surface was not obvious.

This add-on therefore serves a governance role. It records the durable decisions that future patches must respect unless they are explicitly replacing them.

---

## 4. Evidence basis and limits

### 4.1 Evidence basis

This ADR bundle is grounded in the uploaded repository evidence, especially the implementation surfaces corresponding to:

- `tensor_crypt.agents.state_registry`
- `tensor_crypt.agents.brain`
- `tensor_crypt.app.runtime`
- `tensor_crypt.learning.ppo`
- `tensor_crypt.checkpointing.atomic_checkpoint`
- `tensor_crypt.checkpointing.runtime_checkpoint`
- `tensor_crypt.validation.final_validation`
- `tensor_crypt.runtime_config` and `tensor_crypt.config_bridge`
- repository-root compatibility entrypoints and legacy `engine.*` / `viewer.*` re-export modules

The uploaded documentation blueprint for this add-on is also treated as a binding drafting constraint.

### 4.2 Evidence limits

This document does **not** claim access to:

- author interviews
- issue tracker intent
- commit-message history as authoritative design rationale
- undocumented historical debates

For that reason, the phrase **decision** here means **the decision embodied by the present repository**. It does **not** mean a reconstructed psychological history of the authors.

### 4.3 How alternatives are handled here

Where an ADR includes a “not-chosen” alternative, that means one of the following only:

- the current implementation clearly selects a different architecture
- runtime validation explicitly rejects the broader alternative
- the repository exposes the alternative only as a guarded or compatibility surface

This document does **not** invent fictional past proposals.

### 4.4 Frozen truth partitions

The following truth partitions are reused exactly and remain binding throughout this document:

1. **Implemented runtime behavior**  
2. **Guarded compatibility surface**  
3. **Public but currently unread / effectively dead surface**  
4. **Adjacent theory / conceptual background**

Whenever a section below discusses a surface, it states the relevant truth partition explicitly.

---

## 5. Main sections

## 5.1 ADR summary table

| ADR ID | Decision | Primary truth partition | Why it matters |
|---|---|---|---|
| ADR-01 | Identity is canonical by monotonic UID, not by slot | Implemented runtime behavior | Slot reuse must not corrupt ownership, lineage, or resume correctness |
| ADR-02 | PPO state ownership is UID-strict, even though live execution still reaches brains through slots | Implemented runtime behavior | Learning continuity survives slot churn only if optimizer/buffer ownership is UID-based |
| ADR-03 | Bloodline family is the canonical brain architecture surface | Implemented runtime behavior | Family topology is checkpoint-visible and must not drift silently |
| ADR-04 | Canonical observations are primary; legacy observation paths survive only through a guarded adapter | Implemented runtime behavior + Guarded compatibility surface | Observation migration is constrained, not free-form |
| ADR-05 | Runtime assembly is validation-first and preserves a stable build order | Implemented runtime behavior | Reordering build steps can change semantics, not just style |
| ADR-06 | Checkpoint publication uses atomic bundle/manifest publication when full manifest mode is enabled | Implemented runtime behavior | Operator-visible checkpoint integrity depends on publish order and file-set validation |
| ADR-07 | Restore is conservative and validation-led; final validation harnesses exist to test determinism and resume integrity | Implemented runtime behavior | Safe resume requires ordered reconstruction, not ad hoc loading |
| ADR-08 | Public compatibility surfaces remain thin wrappers around the canonical `tensor_crypt` implementation | Guarded compatibility surface | Legacy imports stay available without duplicating real logic |
| ADR-09 | Public configuration is intentionally partitioned into live, guarded, and effectively dead surfaces | Implemented runtime behavior + Guarded compatibility surface + Public but currently unread / effectively dead surface | Maintainers must not treat every public knob as equally live |

---

## 5.2 ADR-01 — Canonical identity is owned by monotonic UID, not by slot

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
The runtime stores dense state in slot-indexed tensors for speed. That creates a structural risk: slot reuse is convenient for execution, but dangerous for long-lived identity.

The repository resolves that risk by separating **dense runtime storage** from **canonical identity ownership**.

### Chosen architecture
Canonical identity belongs to a monotonic UID lifecycle ledger, while slots are reusable runtime containers.

The registry implements this by maintaining at least the following identity surfaces:

- `slot_uid`
- `slot_parent_uid`
- `active_uid_to_slot`
- `uid_lifecycle`
- `uid_family`
- `uid_parent_roles`
- `uid_trait_latent`
- `uid_generation_depth`

Allocation, binding, death finalization, and invariant checks all operate against that UID ledger rather than pretending slot index is identity.

### Evidence-grounded consequences

- A historical UID cannot be rebound as a fresh live agent.
- A dead slot cannot keep owning an active UID.
- An alive slot without a UID is invalid.
- UID and family ledgers are required to stay complete and synchronized.
- Shadow float columns exist only as mirrored compatibility surfaces, not as the canonical ownership source.

### Not-chosen architecture evidenced by the repository

- **Slot-owned canonical identity** was not chosen.
- **UID recycling after death** was not chosen.
- **Best-effort identity without hard invariant checks** was not chosen.

These are not hypothetical readings. They are contradicted by explicit ledger structures, rebinding guards, and invariant assertions.

### Invariants that future maintainers must preserve

- UIDs must remain monotonic and non-reused.
- Slot reuse must never imply UID reuse.
- Lifecycle ledgers must remain complete for all allocated UIDs.
- Live slot bindings, lifecycle records, and family bindings must agree.
- Any compatibility mirror must remain subordinate to the canonical UID surfaces.

### Consequences for safe edits

A patch that “only changes slot handling” can still be identity-breaking if it touches binding, death finalization, restore, or shadow-column synchronization.

### A new ADR is required if

- slots become the canonical ownership surface
- UID monotonicity or non-reuse is relaxed
- lineage, PPO ownership, or checkpoint restore cease to be keyed by UID
- the repository removes or weakens identity invariant enforcement in a material way

### Maintainer note
Read core document **22** before editing this surface.

---

## 5.3 ADR-02 — PPO ownership is UID-strict, not slot-strict

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
The engine acts on currently occupied slots, but learning continuity cannot safely follow slot index because slots can be reused after death and respawn.

### Chosen architecture
The repository anchors rollout buffers, optimizer state, training counters, and update summaries to **canonical UIDs**.

Slot lookup still exists, but only as a way to locate the currently live brain for a UID. The repository explicitly describes PPO ownership as UID-based and validates UID/brain consistency during update and restore flows.

### Evidence-grounded consequences

- PPO buffers are keyed by UID.
- Optimizers are keyed by UID.
- Training state is keyed by UID.
- A UID must map to an active slot before update can proceed.
- A buffer with missing bootstrap state is treated as structurally unsafe when strict bootstrap requirements are enabled.
- Inactive or invalid UID-owned learning state is cleared rather than silently reassigned to a reused slot.

### Not-chosen architecture evidenced by the repository

- **Per-slot PPO ownership** was not chosen.
- **Implicit optimizer reassignment during slot reuse** was not chosen.
- **Loose bootstrap semantics for active non-terminal buffers** were not chosen as the strict default.

### Invariants that future maintainers must preserve

- Learning state must remain keyed to canonical UID identity.
- Update-time validation must confirm that a UID still owns the live brain it is about to optimize.
- Buffer serialization and restore must remain UID-safe.
- Clearing dead or inactive learning state must not accidentally erase unrelated active UID state.

### Consequences for safe edits

Any refactor that simplifies PPO code by indexing state purely by slot should be treated as architecture-changing, not as cleanup.

### A new ADR is required if

- PPO ownership stops being UID-strict
- update or restore becomes slot-owned in a canonical sense
- bootstrap requirements for active buffers are weakened substantially
- family-aware update ordering is replaced by a different canonical scheduling doctrine with broader consequences

### Maintainer note
Read core document **42** before changing training ownership or buffer serialization.

---

## 5.4 ADR-03 — Bloodline family is the canonical brain architecture surface

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
The repository does not treat every agent brain as an unconstrained interchangeable MLP instance. Instead, it defines an ordered set of valid bloodline families and a per-family topology specification.

This means “brain surface” is not only a matter of tensor dimensions. It is also a matter of family membership and family topology.

### Chosen architecture
Each live UID belongs to exactly one bloodline family, and the brain instantiated for its active slot must match that family.

Within a family, topology is shape-identical. Across families, topology may differ through configuration fields such as:

- hidden widths
- activation
- normalization placement
- residual usage
- gating usage
- split-input usage
- split ray/scalar widths
- dropout

The brain module also exposes topology signatures and family descriptions, and checkpoint validation uses family-specific topology expectations.

### Evidence-grounded consequences

- A live slot without a brain is invalid.
- A live slot whose brain family disagrees with the UID family ledger is invalid.
- Brain topology signatures are part of checkpoint-visible validation.
- Family assignment affects initialization, inheritance, and family-aware update ordering.

### Not-chosen architecture evidenced by the repository

- **One undifferentiated canonical brain topology for all agents regardless of family** was not chosen.
- **Free topology drift within a family** was not chosen.
- **Family labels as cosmetic viewer metadata only** was not chosen.

### Invariants that future maintainers must preserve

- Every live UID must resolve to a valid family.
- Family order must remain authoritative for valid family identity.
- The instantiated brain at a live slot must agree with the UID family.
- Any change to family topology has checkpoint and restore consequences.

### Consequences for safe edits

A change to family specs is not a cosmetic hyperparameter tweak if checkpoints, topology signatures, restore validation, or compatibility promises already depend on it.

### A new ADR is required if

- bloodline family stops being the canonical brain surface
- topology stops being family-specific
- topology signatures cease to be a restore/checkpoint invariant
- family identity becomes decoupled from instantiated brain structure

### Maintainer note
Read core document **41** before changing family specs, family validation, or topology signatures.

---

## 5.5 ADR-04 — Canonical observations are primary; legacy observations survive only through a guarded bridge

### Status
Accepted by current repository implementation.

### Truth partition
- **Implemented runtime behavior** for canonical observation handling  
- **Guarded compatibility surface** for legacy fallback handling

### Context
The repository supports a canonical observation contract with separate canonical ray, self, and context tensors. At the same time, it still carries a legacy observation adapter.

This is a migration-sensitive surface because the brain expects canonical observations, yet compatibility with older surfaces still exists.

### Chosen architecture
The canonical observation contract is primary.

The bloodline brain extracts canonical observations directly when the canonical keys are present. If they are absent, the code can adapt legacy observation keys into canonical form **only** when legacy fallback is allowed by configuration.

The configuration surface also makes the intended direction visible:

- canonical feature counts are schema-critical
- legacy widths are documented as bridge dimensions
- canonical observations are returned by default
- legacy fallback remains explicitly optional

### Evidence-grounded consequences

- Canonical observation tensors are shape-validated.
- Legacy tensors are also shape-validated before adaptation.
- The legacy adapter performs a specific mapping into canonical ray/self/context positions.
- Disabling legacy fallback changes behavior from adaptation to hard failure.

### Not-chosen architecture evidenced by the repository

- **Legacy observation schema as the canonical long-term brain surface** was not chosen.
- **Silent acceptance of arbitrary mismatched legacy shapes** was not chosen.
- **Observation migration without explicit adapter logic** was not chosen.

### Invariants that future maintainers must preserve

- Canonical observation widths are schema-visible and brain-visible.
- Adapter dimensions must stay aligned with any surviving legacy producers.
- The adapter must remain explicit and auditable if it continues to exist.
- Brain forward paths must not silently reinterpret malformed observations.

### Consequences for safe edits

Changing feature counts, key names, or adapter mapping is not a local perception edit. It can break inference, checkpoint compatibility assumptions, tests, and documentation truth.

### A new ADR is required if

- legacy fallback is removed entirely
- a second canonical observation schema is introduced
- the adapter stops being a thin migration bridge and becomes a first-class alternate observation family
- canonical feature decomposition is redesigned in a way that changes checkpoint-visible or brain-visible contracts

### Maintainer note
Read core document **40** and core document **41** before editing this surface.

---

## 5.6 ADR-05 — Runtime assembly is validation-first and preserves launch-order semantics

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
The repository makes startup order explicit. It does not treat bootstrapping as an arbitrary convenience layer.

The runtime assembly module states that the order of map generation, initial spawn, engine construction, and viewer construction is a critical invariant unless simulation semantics are intentionally being changed.

### Chosen architecture
Runtime assembly does the following in a deliberate order:

1. validate runtime configuration
2. create run-directory-backed logging surface
3. construct grid, registry, physics, perception, PPO, and evolution objects
4. generate procedural map
5. refresh static wall cache
6. spawn initial population
7. construct engine
8. construct viewer

Determinism setup also calls configuration validation before seeding global RNGs.

### Evidence-grounded consequences

- Unsupported or misleading config combinations are rejected before runtime assembly.
- Runtime assembly order is not presented as freely reorderable.
- Certain config choices are accepted only from narrow supported sets.
- Manifest strictness, latest-pointer writing, and several runtime modes are explicitly gated by validation rules.

### Not-chosen architecture evidenced by the repository

- **Build now, validate later** was not chosen.
- **Broad acceptance of unsupported enum values** was not chosen.
- **Treating launch order as a cosmetic implementation detail** was not chosen.

### Invariants that future maintainers must preserve

- Validation must happen before unsupported config can influence runtime assembly.
- Build order must remain stable unless a semantics-changing design decision is being made.
- Startup logic must remain separate from simulation-rule ownership.

### Consequences for safe edits

A contributor who moves spawn timing, engine construction timing, or validation timing is potentially changing the simulation substrate, not merely reorganizing code.

### A new ADR is required if

- runtime assembly order is intentionally changed in a semantics-relevant way
- validation is relaxed from reject-early to accept-and-coerce behavior
- a new startup architecture replaces the current explicit assembly graph

### Maintainer note
Read core document **20** before changing launch sequence or runtime validation policy.

---

## 5.7 ADR-06 — Checkpoint publication uses atomic file-set publication when manifest mode is enabled

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
Checkpointing here is not merely `torch.save()` plus a filename. The repository contains an operator-facing checkpoint file set with a bundle, manifest, and optional latest-pointer surface.

That creates publication-order and integrity questions.

### Chosen architecture
When full manifest publication is enabled, the repository publishes checkpoints through an atomic file-set path:

- write temporary bundle on the target filesystem
- build and write temporary manifest
- atomically replace final bundle
- atomically replace final manifest
- validate the published file set
- optionally atomically replace the latest-pointer file

Strict configuration rules tie manifest validation and latest-pointer writing to this full atomic publication path.

### Evidence-grounded consequences

- Manifest publication is not treated as independent of atomic write semantics.
- Latest-pointer publication is not permitted as a free-floating convenience feature.
- Resume-through-pointer revalidates manifest-linked bundle details such as tick, size, and checksum consistency.
- File-set validation is defined in terms of what operators and resume paths will actually observe on disk.

### Not-chosen architecture evidenced by the repository

- **Manifest publication without atomic write support** was not chosen.
- **Latest pointer without manifest-backed publication** was not chosen.
- **Trust pointer contents without revalidating published artifacts** was not chosen.

### Invariants that future maintainers must preserve

- Bundle and manifest publication order must remain safe for observers.
- Pointer resolution must remain manifest-aware.
- Manifest metadata must remain aligned with the published bundle.
- Any integrity shortcut must be treated as an architecture change, not a convenience patch.

### Consequences for safe edits

A patch that rewrites checkpoint I/O for speed can easily break the operator-visible safety contract if it weakens atomicity, pointer validation, or manifest coupling.

### A new ADR is required if

- manifest publication is decoupled from atomic write guarantees
- latest-pointer semantics are redefined
- checkpoint publication ceases to be file-set based
- checksum, fingerprint, or strict directory validation policy is materially redesigned

### Maintainer note
Read core document **51** before changing checkpoint publication or manifest semantics.

---

## 5.8 ADR-07 — Restore is conservative and validation-led; the repository keeps explicit validation harnesses

### Status
Accepted by current repository implementation.

### Truth partition
**Implemented runtime behavior**

### Context
A resume system is only trustworthy if restore order is deliberate and testable. The repository makes both visible.

### Chosen architecture
Restore follows a conservative order:

1. validate checkpoint bundle, and manifest when applicable
2. rebuild registry tensors and UID ledgers
3. reconstruct active UID-to-slot bindings
4. recreate brains by family and load brain states
5. restore grid and engine-adjacent runtime state
6. restore PPO buffers, training state, and optimizer state against live UID owners
7. restore scaler state when present
8. restore RNG state when configured
9. restore catastrophe state when configured, otherwise reset it
10. resynchronize shadow identity columns and rerun invariants

In parallel, the repository contains explicit validation harnesses for:

- determinism comparison
- resume consistency
- catastrophe reproducibility
- save-load-save surface comparison
- long-form soak validation of runtime and checkpoint surfaces

### Evidence-grounded consequences

- Restore does not assume slots are enough; it rebuilds UID ownership first.
- Brain family metadata and topology signatures participate in checkpoint validation.
- PPO optimizer state is validated against the live brain it is about to serve.
- Validation harnesses check exact or near-exact structure across save/load and replay-style flows.

### Not-chosen architecture evidenced by the repository

- **Ad hoc partial restore without ordered reconstruction** was not chosen.
- **Brain restore before registry identity reconstruction** was not chosen.
- **Blind optimizer attachment without topology checks** was not chosen.
- **Resume trust without dedicated validation probes** was not chosen.

### Invariants that future maintainers must preserve

- Validation must remain able to reject malformed or mismatched restore surfaces.
- Restore order must continue to respect UID ownership and family topology.
- Save/load validation must remain tied to canonical ownership surfaces.
- Any reproducibility claim must stay bounded by what the harnesses actually test.

### Consequences for safe edits

A patch that “simplifies checkpoint restore” can silently destroy the very guarantee the repository is trying to preserve.

### A new ADR is required if

- restore order is materially changed
- validation harnesses are removed or downgraded in authority
- determinism/resume verification criteria are fundamentally redefined
- catastrophe or RNG restore semantics are materially changed

### Maintainer note
Read core documents **51** and **52** before changing restore order or final validation behavior.

---

## 5.9 ADR-08 — Public compatibility surfaces remain thin wrappers around the canonical implementation

### Status
Accepted by current repository implementation.

### Truth partition
**Guarded compatibility surface**

### Context
The repository still exposes public root-level and legacy import surfaces, but the canonical implementation lives under `tensor_crypt`.

The key architecture question is whether those public surfaces are independent implementations or merely compatibility veneers.

### Chosen architecture
They are thin compatibility veneers.

Examples visible in the repository include:

- repository-root `run.py` routing to `tensor_crypt.app.launch.main`
- repository-root compatibility entrypoint routing to the same canonical launch surface
- `config.py` re-exporting canonical runtime config
- legacy `engine.*` modules re-exporting canonical modules
- legacy `viewer.*` modules described as thin compatibility modules

### Evidence-grounded consequences

- The canonical implementation surface is concentrated under `tensor_crypt`.
- Compatibility imports remain publicly available without duplicating real subsystem logic.
- Current working directory concerns are handled by the root launch surface without relocating canonical ownership of startup logic.

### Not-chosen architecture evidenced by the repository

- **Duplicated real logic in legacy wrappers** was not chosen.
- **Separate launch semantics for root entrypoints and canonical package entrypoints** were not chosen.
- **A forked legacy engine tree with independent behavior** was not chosen.

### Invariants that future maintainers must preserve

- Compatibility wrappers should remain thin unless an explicit migration project replaces them.
- Canonical logic should not quietly drift into the wrappers.
- Public compatibility should not be described as full independent implementation.

### Consequences for safe edits

A wrapper that starts acquiring real logic becomes a second canonical surface. That is not a small refactor.

### A new ADR is required if

- compatibility wrappers gain independent behavior
- root entrypoints stop being thin launch veneers
- legacy import trees become first-class maintained implementations

### Maintainer note
Read core document **63** before changing compatibility claims or contributor-facing import guidance.

---

## 5.10 ADR-09 — Public configuration is intentionally partitioned into live, guarded, and effectively dead surfaces

### Status
Accepted by current repository implementation and documentation style.

### Truth partition
- **Implemented runtime behavior**  
- **Guarded compatibility surface**  
- **Public but currently unread / effectively dead surface**

### Context
The runtime configuration file is not written as a flat bag of equally real knobs. It explicitly labels many fields by audit status.

That is itself an architectural choice, because it tells maintainers how to talk about public surfaces honestly.

### Chosen architecture
The repository adopts an explicit truth-boundary approach for configuration and compatibility surfaces:

- some knobs are active runtime controls
- some public values are intentionally guarded to narrow supported states
- some public fields remain present but are described as effectively unread in the uploaded repository dump
- schema-critical fields are marked as high-risk

Runtime validation reinforces this by rejecting unsupported values for many public-looking enum-like surfaces.

### Evidence-grounded consequences

- Public visibility does not imply broad support.
- Unsupported values are often rejected rather than silently tolerated.
- Configuration comments act as a maintenance honesty layer, not mere ornamentation.
- The repository already distinguishes compatibility presence from implementation truth.

### Not-chosen architecture evidenced by the repository

- **All public config fields are fully live and equally supported** was not chosen.
- **Silent coercion of unsupported values into nearest working behavior** was not chosen as the dominant pattern.
- **Documentation that hides dead or guarded surfaces behind marketing-style simplification** was not chosen.

### Invariants that future maintainers must preserve

- Public documentation must keep distinguishing live, guarded, and effectively dead surfaces.
- New public knobs should not be advertised as active unless the runtime actually reads and supports them.
- Validation policy should remain aligned with documentation truth.

### Consequences for safe edits

A maintainer who “cleans up comments” by erasing these distinctions can accidentally destroy the repository’s truth boundary and overstate support.

### A new ADR is required if

- the repository abandons the explicit truth-partition model
- config validation becomes permissive in a way that changes what “guarded” means
- a broad surfacing policy change promotes many previously dead or guarded fields into supported runtime behavior

### Maintainer note
Read core documents **60** and **63** before changing truth-boundary language.

---

## 5.11 Decision-to-consequence matrix

| Decision | Immediate safety benefit | Main risk if ignored | Typical misleading edit |
|---|---|---|---|
| UID over slot identity | prevents slot reuse from corrupting canonical ownership | lineage, PPO, and resume corruption | “Simplify identity by using slot index everywhere” |
| UID-strict PPO ownership | preserves learning continuity across slot churn | optimizer/buffer reassignment bugs | “Store PPO state next to slot tensors only” |
| Family as brain surface | keeps topology and restore semantics explicit | checkpoint/restore mismatch | “Treat family as UI-only metadata” |
| Canonical observations first | stabilizes brain input contract | hidden schema drift | “Add features without updating canonical contract and bridge rules” |
| Validation-first runtime assembly | rejects unsafe config before construction | misleading or non-reproducible startup states | “Move validation later for convenience” |
| Atomic checkpoint publication | preserves operator-visible artifact integrity | partial publish / stale pointer hazards | “Write pointer whenever bundle exists” |
| Conservative ordered restore | protects resume correctness | half-valid restore states | “Load brains first, fix identity later” |
| Thin wrappers only | avoids multiple canonical code paths | logic drift across legacy entry surfaces | “Add small custom behavior to wrapper modules” |
| Explicit truth partitions | keeps docs and support claims honest | overclaiming support | “Public field exists, therefore it is supported” |

---

## 5.12 Invariant-risk table

| Invariant | Why it is dangerous to violate | Surfaces affected |
|---|---|---|
| historical UIDs are never rebound | breaks canonical identity continuity | registry, PPO, checkpoint restore, lineage telemetry |
| live slot brain family matches UID family | breaks brain ownership and restore meaning | registry, brain creation, checkpoint validation |
| canonical observation widths stay aligned | breaks inference and migration surfaces | perception, brain, config, tests |
| optimizer state matches live brain topology | breaks resume correctness and update safety | PPO, checkpoint restore |
| manifest and pointer describe the published bundle | breaks operator-visible checkpoint trust | checkpoint publish and resume paths |
| restore order rebuilds UID ownership before dependent state | breaks resume semantics | registry, brains, PPO, catastrophe state |
| wrappers remain thin | prevents duplicate canonical logic | root entrypoints, legacy imports, contributor expectations |
| truth partitions stay explicit | prevents false support claims | docs, config comments, contributor guidance |

---

## 5.13 When a future change requires a new ADR

Use the checklist below before merging a major patch.

A new ADR is warranted when **any** of the following is true:

- the change redefines canonical ownership of identity, brains, learning state, or checkpoints
- the change alters a schema-visible tensor contract
- the change broadens or narrows a guarded compatibility surface in a durable way
- the change converts a thin wrapper into a behavior-owning module
- the change weakens validation from reject-early to tolerate-and-continue
- the change redefines what a checkpoint or latest pointer means to an operator
- the change makes previously effectively dead public surfaces materially live
- the change changes resume semantics, determinism expectations, or validation authority

A new ADR is usually **not** required when all of the following are true:

- the change preserves the same canonical owner for the surface
- no checkpoint-visible schema changes are introduced
- validation logic remains equivalent in authority
- compatibility wrappers remain thin
- the patch is explanatory, local, and non-semantic

### Maintainer checklist

- [ ] Did the patch change canonical ownership of any major surface?
- [ ] Did it change a schema-visible dimension, key set, or topology signature?
- [ ] Did it alter checkpoint publication, restore order, or validation guarantees?
- [ ] Did it convert a compatibility surface into a primary implementation surface?
- [ ] Did it change support truth for a public config or import surface?
- [ ] If yes to any of the above, was a new ADR added or an existing ADR explicitly superseded?

---

## 6. Figures/tables/diagrams to include or defer

### Included in this document

- ADR summary table
- decision-to-consequence matrix
- invariant-risk table
- when-to-write-a-new-ADR checklist

### Deferred from this document

The following diagrams are useful, but they are deferred here to avoid duplicating canonical chapter material:

1. **UID lifecycle and slot-binding diagram**  
   Defer to the state/identity material anchored by core document **22**.

2. **Canonical-versus-legacy observation bridge diagram**  
   Defer to the perception/brain material anchored by core documents **40** and **41**.

3. **Checkpoint publish-order diagram**  
   Defer to the checkpointing material anchored by core document **51**.

4. **Restore-order dependency diagram**  
   Defer to the checkpointing and validation material anchored by core documents **51** and **52**.

This ADR bundle is text-first because its job is governance and decision capture, not full subsystem pedagogy.

---

## 7. Cross-links to core bundle documents

Use this add-on together with the following core documents:

- **20** — architecture foundation
- **22** — state and identity
- **40** — observation/perception prerequisite document in the core sequence
- **41** — brain architecture
- **42** — PPO ownership
- **51** — checkpointing
- **52** — validation / resume / audit-adjacent core material
- **60** — truth appendix
- **63** — contributor truth contract

### Cross-link usage guidance

- If the question is **what exists**, prefer the core chapter first.
- If the question is **why a change is dangerous**, prefer this ADR bundle.
- If the question is **what claims are safe to make**, read this document together with **60** and **63**.

---

## 8. Truth-boundary notes

### 8.1 This document records repository decisions, not mythic project history

Nothing in this document should be read as fictional origin story. It records present implementation decisions evidenced by code and validation behavior.

### 8.2 Compatibility presence is not the same as full support

The repository contains legacy wrappers, adapter paths, and documented public fields that survive for compatibility reasons. Their existence must not be overstated.

### 8.3 Guarded surfaces must not be flattened into full freedom

A surface is not “fully configurable” merely because it is public. Many values are only accepted from a narrow supported set, and some public surfaces are explicitly documented as unread in the uploaded repository evidence.

### 8.4 Validation evidence bounds the strength of claims

The repository includes meaningful validation probes, but those probes do not justify broader claims than they actually test. Determinism, resume, catastrophe reproducibility, and save-load-save equality should be discussed only within the boundaries of those harnesses.

### 8.5 Adjacent theory must not be upgraded into implementation truth

Conceptual background may explain why a surface is sensible, but it must not be presented as implemented behavior unless the repository actually does it.

---

## 9. Maintainer notes where relevant

### 9.1 Safe editing posture

Before editing any surface discussed here, first decide which of the following you are changing:

- implementation mechanics only
- compatibility behavior
- schema or checkpoint contract
- truth-boundary language
- canonical ownership

That classification should happen **before** code changes, not after.

### 9.2 Minimum review expectation for high-risk changes

At minimum, a high-risk patch should be reviewed against:

- identity invariants
- family/topology compatibility
- canonical observation compatibility
- checkpoint publication and restore ordering
- validation harness coverage impact
- contributor-facing truth claims

### 9.3 Safe language for future documentation updates

Use language such as:

- “implemented runtime behavior” when the repository clearly does it
- “guarded compatibility surface” when the public surface exists but support is deliberately narrow
- “public but currently unread / effectively dead surface” when the repository evidence says so
- “adjacent theory” when the material is explanatory but not implemented

Do **not** rewrite those distinctions into vague prose such as “the system supports many modes” unless the repository actually does.

### 9.4 Supersession rule

If a future patch intentionally overturns one of these decisions, do not silently edit this file to make the old decision disappear. Instead:

1. add a new ADR or explicitly supersede the old one  
2. name the changed invariant plainly  
3. describe migration and compatibility consequences honestly  
4. update relevant core chapters so the canonical explanation and the decision record stay aligned

---

## Appendix A — Compact maintainer reading order

For a maintainer touching the surfaces covered here, the shortest serious reading path is:

1. **20**  
2. **22**  
3. **41** and **42**  
4. **51** and **52**  
5. **60** and **63**  
6. this ADR bundle

That sequence keeps implementation truth, decision logic, and contributor honesty aligned.
