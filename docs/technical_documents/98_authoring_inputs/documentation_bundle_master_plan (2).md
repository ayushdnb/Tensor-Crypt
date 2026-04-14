# Documentation bundle master plan

## 1. Title page

**Project:** Tensor Crypt  
**Artifact type:** Master planning document for a future book-grade multi-document documentation and paper bundle  
**Planning scope:** Repository-grounded documentation architecture, curriculum sequencing, truth controls, visual strategy, bundle dependency design, and generation-wave execution plan  
**Evidence basis used for this plan:** Uploaded raw Python repository dump (`evolution.txt`), including canonical runtime package surfaces, compatibility wrappers, configuration definitions, validation harnesses, viewer/input code, catastrophe systems, checkpointing code, telemetry code, and an extensive Python test corpus inferred from the dump

---

## 2. Executive purpose of the documentation bundle

The future bundle must do five jobs at once without collapsing them into one undifferentiated wall of prose.

1. It must provide a **simple operator-facing manual** for readers who want to run the project, inspect it, change safe knobs, observe behavior, and interpret outputs without first becoming experts in reinforcement learning or systems architecture.
2. It must provide a **code-truth technical explanation** of what the repository actually implements: launch path, package structure, tick order, state substrate, perception contract, bloodline-brain architecture, PPO ownership model, respawn and mutation mechanics, catastrophe overlays, telemetry surfaces, checkpoint semantics, and validation harnesses.
3. It must provide a **pedagogical ladder** from weak background toward strong technical competence, introducing mathematics, tensors, optimization, RL vocabulary, simulation vocabulary, and software architecture vocabulary at the points where they become necessary.
4. It must provide a **research and reproducibility record** describing determinism, schema versions, checkpoint validation, manifest publication, resume consistency, soak auditing, and artifact interpretation in a way that can support serious experimentation.
5. It must provide a **maintainer-grade reference layer** that future contributors can use to extend the system without corrupting canonical ownership, observation contracts, checkpoint compatibility, or logging semantics.

The bundle must therefore be deliberately partitioned. A single monolithic paper would fail both operator usability and technical truthfulness. The correct product is a layered documentation suite with reading tracks, fixed terminology, explicit cross-links, stable reference conventions, and a strong distinction between implemented behavior and adjacent theory.

---

## 3. Repository-grounded project characterization

### 3.1 What the codebase is

The repository is not merely a toy RL demo. It is a local simulation platform centered on a canonical `tensor_crypt` implementation package, with repository-root launch wrappers (`config.py`, `main.py`, `run.py`) and legacy compatibility re-export surfaces for older `engine.*` imports. The codebase assembles a deterministic run from a central runtime configuration surface, builds a tensor-backed world grid, seeds an initial population, executes a per-tick simulation loop, trains bloodline-aware MLP policies with per-UID PPO ownership, records run artifacts into HDF5/Parquet/JSON outputs, supports runtime checkpoints with manifests and latest-pointer publication, exposes catastrophe-driven reversible runtime overlays, and provides a Pygame viewer with inspection and control surfaces.

### 3.2 Canonical package surfaces confirmed from the code dump

The dump provides direct evidence for the following canonical module families:

- `tensor_crypt.app` — launch and runtime assembly
- `tensor_crypt.agents` — brain and state registry
- `tensor_crypt.world` — grid, map generation, perception, observation schema, physics
- `tensor_crypt.population` — evolution helpers, reproduction, respawn controller
- `tensor_crypt.learning` — PPO implementation
- `tensor_crypt.simulation` — engine loop
- `tensor_crypt.telemetry` — data logger, lineage export, run-path utilities
- `tensor_crypt.checkpointing` — atomic checkpoint publication and runtime checkpoint capture/restore
- `tensor_crypt.audit` — determinism, resume, catastrophe, and save-load-save validation harnesses
- `tensor_crypt.viewer` — camera, input, layout, panels, main viewer, text cache, colors
- `tensor_crypt.runtime_config` and `tensor_crypt.config_bridge` — authoritative configuration and shared config instance bridge

### 3.3 Launch and runtime assembly truth

The observed runtime assembly order is materially important and must be documented as a protected invariant.

1. Determinism validation and seeding are applied.
2. A run directory is created and populated with `config.json` and `run_metadata.json`.
3. `DataLogger`, `Grid`, `Registry`, `Physics`, `Perception`, `PPO`, and `Evolution` are instantiated.
4. Procedural walls and heal/harm zones are generated.
5. Static wall cache is refreshed.
6. Initial population spawn occurs.
7. `Engine` is constructed.
8. `Viewer` is constructed and run.

This is not incidental boot glue; it is the concrete semantic startup path that the architecture documentation must preserve and explain.

### 3.4 State substrate truth

The simulation uses a dense slot-backed tensor substrate for runtime performance, but canonical identity is not slot-owned. The actual ownership substrate is **monotonic UID-based**, with explicit lifecycle records, slot-to-UID bindings, UID-to-family mappings, UID parent-role ledgers, trait-latent ledgers, and generation-depth ledgers. Slot reuse is allowed only after UID finalization; UIDs do not recycle.

This distinction is one of the most important conceptual and documentation boundaries in the repository.

### 3.5 Observation and policy truth

Perception produces a canonical observation contract built from:

- canonical ray features
- canonical self features
- canonical context features

A legacy observation bridge still exists. The brain module can adapt legacy surfaces into canonical tensors when legacy fallback is allowed. The canonical brain surface is a bloodline-aware MLP family system, not a generic transformer stack. Families differ by hidden widths, activation, normalization placement, residual use, gating, split-input structure, and dropout.

### 3.6 Learning truth

PPO training ownership is UID-anchored. Buffers, optimizers, training counters, last-update summaries, bootstrap state, and truncated-rollout accounting belong to canonical UIDs rather than transient slot indices. The code explicitly protects against slot reuse corrupting training ownership.

### 3.7 Population dynamics truth

Respawn is not a vague “spawn new agent” helper. It is a structured binary-parent reproduction path with:

- distinct brain parent, trait parent, and anchor parent roles
- floor-recovery logic
- extinction policy handling
- overlay doctrines for crowding, cooldown, and local-parent selection
- latent trait inheritance and mutation
- optional family shift mutation
- placement constraints and fallback behavior

### 3.8 World-mechanics truth

The world is a tensor-backed grid with walls, heal/harm zones, agent occupancy, and mass channels. Physics resolves movement, wall collisions, rams, contested moves, environment effects, and death marking. Catastrophes temporarily repaint or override world and runtime modifiers without mutating permanent trait state.

### 3.9 Telemetry and reproducibility truth

The logger writes a substantial artifact suite, including HDF5 snapshots and multiple Parquet ledgers, plus lineage JSON export and saved brain-state checkpoints within run directories. Runtime checkpoints may be published atomically with manifests and latest-pointer files. Validation harnesses check determinism, resume consistency, catastrophe reproducibility, and save-load-save surface equivalence.

### 3.10 Important evidence limitations

The current evidence source is a concatenated Python dump, not a full repository tree with all filenames, non-Python assets, packaging metadata, shell scripts, notebooks, CI files, or prose docs. Therefore:

- Python module paths are well evidenced through imports and module docstrings.
- Runtime behavior implemented in Python is strongly evidenced.
- Exact non-Python file layout is **not** fully evidenced.
- Exact test file paths are **not** fully evidenced, although the dump clearly contains a broad Python test suite.
- Installation, dependency, packaging, and CI documentation should be marked provisional until the full filesystem is inspected.

The future documentation bundle must openly state these boundaries wherever necessary.

---

## 4. Documentation philosophy and truth constraints

### 4.1 Governing principle

Repository truth outranks pedagogy, theory, convenience, and aesthetic symmetry.

The documentation bundle is allowed to teach related theory, but only after it has first established what the code actually does.

### 4.2 Required truth partitions

Every document in the future bundle must classify material into one of four explicit categories:

1. **Implemented runtime behavior**  
   Concrete behavior directly supported by the repository.
2. **Guarded compatibility surface**  
   Public surface exists, but runtime validation currently restricts accepted values or usage.
3. **Public but currently unread / effectively dead surface**  
   Present in public configuration or compatibility code but not evidenced as actively read by the current runtime.
4. **Adjacent theory / conceptual background**  
   Included for understanding, but not to be described as implemented behavior.

### 4.3 Mandatory language contract for future writers

Future documentation writers must:

- distinguish slot from UID everywhere
- distinguish canonical observation surfaces from legacy bridge surfaces everywhere
- distinguish root wrappers from canonical implementation modules everywhere
- distinguish runtime-modifier overlays from persistent trait state everywhere
- distinguish validation harnesses from production runtime behavior everywhere
- mark speculative or non-evidenced claims explicitly
- cite exact module/class/function/field names whenever making architectural claims
- avoid saying a config knob is “supported” merely because it exists in the dataclass
- avoid saying a concept is “in the system” when it appears only as background theory

### 4.4 Documentation architecture principle

The bundle should combine a research-lab rigor standard with a documentation structure that separates tutorial, task, explanation, and reference roles. A useful high-level frame is the four-way distinction between tutorials, how-to guidance, explanation, and reference described by Diátaxis, while terminology and wording discipline should follow current technical writing guidance such as Google’s documentation style guidance and Microsoft’s writing guidance. citeturn196259search0turn196259search3turn196259search18turn196259search1turn196259search2

The bundle, however, must not mechanically imitate any external framework. It must be adapted to the actual repository and its unusually strong need for implementation-truth auditing.

---

## 5. Audience model and reading tracks

### 5.1 Audience classes

#### A. Beginner / non-expert operator
Needs a runnable manual, visual explanations, safety warnings, knob recipes, artifact reading help, and zero assumption of RL fluency.

#### B. Intermediate programmer / engineer
Needs package structure, data flow, state contracts, core subsystems, and enough mathematics to understand tensorized simulation and MLP inference.

#### C. Advanced ML / RL learner
Needs observation contract details, reward design, rollout ownership, GAE/PPO derivation, update loops, optimizer state continuity, and training-surface caveats.

#### D. Systems / architecture reader
Needs boot order, package boundaries, identity substrate, compatibility layers, schema controls, artifact surfaces, and invariants that govern safe extension.

#### E. Reproducibility / validation / research-method reader
Needs determinism setup, checkpoint publication, manifest semantics, save-load-save probes, audit harnesses, and interpretation of logged artifacts.

#### F. Future contributor / maintainer
Needs extension rules, code-truth citation rules, safe edit zones, high-risk schema surfaces, terminology contract, and documentation consistency rules.

### 5.2 Reading tracks

| Track | Intended reader | Recommended order |
|---|---|---|
| Track O | Operator / beginner | 00 → 01 → 11 → 14 → 15 → 16 |
| Track E | Engineer | 00 → 02 → 03 → 04 → 05 → 06 → 08 → 09 → 11 |
| Track RL | Advanced ML/RL learner | 00 → 15 → 08 → 09 → 10 → 11 → 12 |
| Track S | Systems / architecture reader | 00 → 02 → 03 → 06 → 07 → 11 → 12 → 13 → 17 |
| Track R | Reproducibility / research-method reader | 00 → 02 → 11 → 12 → 13 → 16 → 17 |
| Track M | Future maintainer | entire bundle, with 17 read before writing any new prose |

### 5.3 Reading-order policy

- Documents `00` through `03` define baseline vocabulary and should be treated as the stable entry layer.
- Documents `04` through `10` are core implementation explanations.
- Documents `11` through `14` are operations, artifacts, UI, and reproducibility layers.
- Documents `15` through `17` are background/reference/governance layers.
- The operator manual is not sufficient for contributors.
- The background appendix is not a substitute for code-truth documents.

---

## 6. Proposed bundle hierarchy and folder tree

Because the current evidence source does not expose an existing documentation directory with certainty, the following hierarchy is a **proposed future structure**, not a claim about the current repository tree.

```text
documentations/
├── 00_meta/
│   ├── 00_documentation_bundle_index.md
│   ├── 01_reading_tracks_and_dependency_map.md
│   ├── 02_notation_glossary_and_shape_legend.md
│   └── 03_figure_artifact_and_source_reference_index.md
├── 01_operator_manual/
│   ├── 10_operator_runbook_and_game_manual.md
│   ├── 11_config_reference_active_guarded_dead.md
│   └── 12_experiment_recipes_and_safe_knob_sets.md
├── 02_system_foundations/
│   ├── 20_project_identity_runtime_boot_and_package_map.md
│   ├── 21_python_pytorch_tensors_and_simulation_foundations.md
│   └── 22_state_identity_lineage_and_ownership_contracts.md
├── 03_simulation_substrate/
│   ├── 30_world_grid_map_hzones_and_catastrophe_substrate.md
│   ├── 31_tick_order_physics_conflict_resolution_and_death.md
│   └── 32_respawn_reproduction_mutation_and_bloodline_dynamics.md
├── 04_learning_and_observation/
│   ├── 40_observation_schema_rays_and_feature_construction.md
│   ├── 41_bloodline_brain_architecture_and_inference_paths.md
│   └── 42_uid_owned_ppo_rollouts_rewards_and_updates.md
├── 05_operations_reproducibility/
│   ├── 50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md
│   ├── 51_checkpointing_atomic_publish_resume_and_schema_safety.md
│   ├── 52_validation_determinism_resume_consistency_and_soak_methods.md
│   └── 53_viewer_controls_inspection_and_diagnostics.md
└── 06_appendices/
    ├── 60_implemented_behavior_vs_adjacent_theory.md
    ├── 61_background_math_python_pytorch_and_rl_appendix.md
    ├── 62_equations_shapes_and_contract_reference.md
    └── 63_contributor_documentation_truth_contract.md
```

### 6.1 Why this hierarchy is correct

- The operator layer is isolated from deep theory.
- The system-foundations layer establishes architecture before deep mechanics.
- The simulation-substrate layer explains world evolution before PPO details.
- The learning-and-observation layer binds perception, brains, and PPO together.
- The operations/reproducibility layer isolates telemetry, checkpoints, validation, and viewer behavior.
- The appendices hold background theory, reference tables, and governance rules so that implementation chapters remain honest and readable.

---

## 7. Proposed document list with filenames

### 7.1 Primary bundle document list

| ID | Filename | Role |
|---|---|---|
| 00 | `documentations/00_meta/00_documentation_bundle_index.md` | root index and navigation spine |
| 01 | `documentations/00_meta/01_reading_tracks_and_dependency_map.md` | track guide and dependency map |
| 02 | `documentations/00_meta/02_notation_glossary_and_shape_legend.md` | shared notation, tensor shape conventions, glossary spine |
| 03 | `documentations/00_meta/03_figure_artifact_and_source_reference_index.md` | figure inventory, artifact map, code-reference rules |
| 10 | `documentations/01_operator_manual/10_operator_runbook_and_game_manual.md` | simple operator-facing manual |
| 11 | `documentations/01_operator_manual/11_config_reference_active_guarded_dead.md` | authoritative config atlas |
| 12 | `documentations/01_operator_manual/12_experiment_recipes_and_safe_knob_sets.md` | practical knob recipes and scenario templates |
| 20 | `documentations/02_system_foundations/20_project_identity_runtime_boot_and_package_map.md` | project characterization, package map, boot path |
| 21 | `documentations/02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md` | first-principles foundations |
| 22 | `documentations/02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md` | slot vs UID substrate and lineage invariants |
| 30 | `documentations/03_simulation_substrate/30_world_grid_map_hzones_and_catastrophe_substrate.md` | world substrate and catastrophe field layer |
| 31 | `documentations/03_simulation_substrate/31_tick_order_physics_conflict_resolution_and_death.md` | tick order, conflict resolution, death processing |
| 32 | `documentations/03_simulation_substrate/32_respawn_reproduction_mutation_and_bloodline_dynamics.md` | reproduction, mutation, overlays, families |
| 40 | `documentations/04_learning_and_observation/40_observation_schema_rays_and_feature_construction.md` | perception, raycasting, self/context features |
| 41 | `documentations/04_learning_and_observation/41_bloodline_brain_architecture_and_inference_paths.md` | bloodline MLP families and inference paths |
| 42 | `documentations/04_learning_and_observation/42_uid_owned_ppo_rollouts_rewards_and_updates.md` | PPO ownership, reward, rollout, update logic |
| 50 | `documentations/05_operations_reproducibility/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md` | logged outputs and artifact interpretation |
| 51 | `documentations/05_operations_reproducibility/51_checkpointing_atomic_publish_resume_and_schema_safety.md` | checkpoint capture, manifest, resume semantics |
| 52 | `documentations/05_operations_reproducibility/52_validation_determinism_resume_consistency_and_soak_methods.md` | validation harnesses and scientific method |
| 53 | `documentations/05_operations_reproducibility/53_viewer_controls_inspection_and_diagnostics.md` | viewer UI, hotkeys, inspector and diagnostics |
| 60 | `documentations/06_appendices/60_implemented_behavior_vs_adjacent_theory.md` | explicit truth boundary appendix |
| 61 | `documentations/06_appendices/61_background_math_python_pytorch_and_rl_appendix.md` | optional but detailed background scaffolding |
| 62 | `documentations/06_appendices/62_equations_shapes_and_contract_reference.md` | reference tables for equations, shapes, schemas |
| 63 | `documentations/06_appendices/63_contributor_documentation_truth_contract.md` | documentation governance and anti-hallucination rules |

### 7.2 Linear vs reference-only classification

- **Read linearly:** 00, 10, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52
- **Use as guided reference:** 01, 02, 03, 11, 12, 53, 60, 61, 62, 63

---

## 8. Detailed per-document planning sheets

### 8.1 Document 00 — bundle index

**Filename:** `documentations/00_meta/00_documentation_bundle_index.md`  
**Type:** foundational navigation document  
**Audience:** all  
**Read mode:** first and repeatedly  

**Scope**
- identify the bundle purpose
- define implemented-vs-theoretical labeling system
- summarize reading tracks
- link every document
- define stable citation style for modules, classes, methods, and config fields

**Non-scope**
- deep explanation of mechanics
- operator recipes
- mathematical derivations

**Key questions answered**
- what exists in this bundle
- where should a reader begin
- how should claims be interpreted
- which docs are required vs optional

**Required prior knowledge**
- none

**From-first-principles content needed**
- none beyond brief orientation

**Implementation surfaces it must cite or explain**
- canonical package name `tensor_crypt`
- root wrappers `config.py`, `main.py`, `run.py`
- the existence of compatibility re-export surfaces

**Figures / tables / diagrams**
- bundle map diagram
- reading track matrix
- document role legend

**Cross-links**
- all documents

---

### 8.2 Document 01 — reading tracks and dependency map

**Filename:** `documentations/00_meta/01_reading_tracks_and_dependency_map.md`

**Scope**
- dependency graph across documents
- multiple reading orders by audience
- “minimum viable path” vs “deep mastery path”

**Non-scope**
- code explanation

**Key questions answered**
- what depends on what
- how to avoid reading out of order

**Required prior knowledge**
- none

**Figures / tables / diagrams**
- document dependency graph
- audience-to-doc track table

**Cross-links**
- 00, all primary docs

---

### 8.3 Document 02 — notation, glossary, and shape legend

**Filename:** `documentations/00_meta/02_notation_glossary_and_shape_legend.md`

**Scope**
- stable terminology
- tensor-shape notation
- symbol table
- abbreviations
- UI label normalization

**Non-scope**
- extended theory chapters

**Key questions answered**
- what exactly is meant by UID, slot, family, anchor parent, canonical rays, etc.
- how tensor shapes are written
- which terms are forbidden or discouraged because they blur important distinctions

**Required prior knowledge**
- none

**From-first-principles content needed**
- vector/tensor notation basics
- schema/version vocabulary basics

**Figures / tables / diagrams**
- terminology table
- tensor-shape legend
- “do not conflate” table

**Cross-links**
- all technical docs, especially 22, 40, 41, 42, 51, 52

---

### 8.4 Document 03 — figure, artifact, and source reference index

**Filename:** `documentations/00_meta/03_figure_artifact_and_source_reference_index.md`

**Scope**
- figure numbering convention
- artifact naming convention
- code-citation convention
- run-directory artifact catalog index

**Non-scope**
- explanation of every artifact

**Key questions answered**
- how figures are referenced
- how code claims are sourced
- how artifacts and ledgers are named in the prose

**Required prior knowledge**
- none

**Figures / tables / diagrams**
- none beyond index tables

**Cross-links**
- 50, 51, 52

---

### 8.5 Document 10 — operator runbook and game manual

**Filename:** `documentations/01_operator_manual/10_operator_runbook_and_game_manual.md`

**Type:** operator-facing, beginner-friendly, deliberately simpler prose

**Scope**
- what the project is in practical terms
- how to launch it
- what appears in the viewer
- what the world contains
- what the agents do at a high level
- what artifacts appear after a run
- how to pause, inspect, speed up, and interpret a session
- safe first experiments
- hotkeys and interaction surfaces

**Non-scope**
- PPO derivations
- checkpoint internal schemas
- detailed contest resolution proofs
- exhaustive config enumeration

**Key questions answered**
- how to run the project safely
- what each major screen element means
- what a run directory contains
- which knobs matter first
- what catastrophes and reproduction overlays mean operationally

**Required prior knowledge**
- none

**Concepts introduced from first principles**
- simulation tick
- world grid
- heal/harm zone
- agent, family, birth, death
- snapshot, checkpoint, run directory
- deterministic seed

**Implementation surfaces it must cite / explain**
- `tensor_crypt.app.launch.main`
- `tensor_crypt.app.runtime.build_runtime`
- `tensor_crypt.viewer.main.Viewer`
- viewer input hotkeys and selection logic
- run directory creation and major artifacts

**Outputs / figures / diagrams**
- one-page “how a run starts” diagram
- run-directory tree diagram
- viewer annotated screenshot mock plan
- controls table
- “first five experiments” recipe cards
- “what to inspect after 500 ticks” checklist

**Cross-links**
- 11 for full config atlas
- 12 for recipes
- 53 for detailed UI/controls
- 50 for artifact interpretation

---

### 8.6 Document 11 — config reference: active, guarded, dead

**Filename:** `documentations/01_operator_manual/11_config_reference_active_guarded_dead.md`

**Type:** authoritative reference document

**Scope**
- enumerate config sections and fields
- group knobs by operational domain
- mark each knob as active, guarded, unread/dead, or schema-critical
- specify valid values, current defaults, and runtime effects
- warn about dangerous or migration-sensitive changes

**Non-scope**
- full architecture narrative
- long beginner teaching

**Key questions answered**
- which knobs actually matter
- which knobs are compatibility-only
- which knobs are dangerous
- which knobs are safe for operators

**Required prior knowledge**
- basic operator manual

**Concepts introduced from first principles**
- schema version
- compatibility surface
- validation-restricted option
- unread/dead knob meaning

**Implementation surfaces it must cite / explain**
- `tensor_crypt.runtime_config`
- `tensor_crypt.app.runtime.validate_runtime_config`
- config bridge usage

**Outputs / figures / diagrams**
- section-by-section config tables
- “danger surface” table
- “safe starter knobs” table
- “do not touch without migration” callouts

**Cross-links**
- 10, 12, 20, 22, 30, 32, 40, 42, 50, 51, 52

---

### 8.7 Document 12 — experiment recipes and safe knob sets

**Filename:** `documentations/01_operator_manual/12_experiment_recipes_and_safe_knob_sets.md`

**Scope**
- reproducible experiment templates
- beginner-safe knob bundles
- benchmarking presets
- audit/validation presets
- catastrophe demo presets
- reproduction overlay demo presets

**Non-scope**
- proving underlying mechanics

**Key questions answered**
- how to change behavior without breaking the run
- how to choose small, medium, large, debug, benchmark, and audit configurations

**Required prior knowledge**
- 10 and 11

**Figures / tables / diagrams**
- recipe cards
- “goal → knob bundle” table
- run artifact expectation table

**Cross-links**
- 10, 11, 50, 51, 52, 53

---

### 8.8 Document 20 — project identity, runtime boot, and package map

**Filename:** `documentations/02_system_foundations/20_project_identity_runtime_boot_and_package_map.md`

**Scope**
- what the project is
- canonical package topology
- compatibility layers
- boot path and assembly order
- subsystem responsibilities

**Non-scope**
- deep per-subsystem equations

**Key questions answered**
- where the real implementation lives
- what the root wrappers do
- what each package owns
- what assembly order is fixed

**Required prior knowledge**
- none beyond 00

**Concepts introduced from first principles**
- package vs wrapper
- runtime assembly graph
- subsystem ownership

**Implementation surfaces it must cite / explain**
- `tensor_crypt.app.launch`
- `tensor_crypt.app.runtime`
- legacy `engine.*` re-export surfaces
- repository-root entrypoints

**Outputs / figures / diagrams**
- package map
- launch sequence diagram
- responsibility matrix
- compatibility-layer map

**Cross-links**
- 21, 22, 30, 31, 40, 41, 42, 50, 51

---

### 8.9 Document 21 — Python, PyTorch, tensors, and simulation foundations

**Filename:** `documentations/02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md`

**Scope**
- minimal but serious background needed for the rest of the bundle
- tensors, indexing, normalization, shapes, batches
- dataclasses and configuration objects
- simulation loops and discrete ticks
- basic probabilistic action sampling vocabulary

**Non-scope**
- repository-specific walkthrough beyond examples

**Key questions answered**
- what a tensor is
- what batched simulation means
- what logits, value heads, and categorical sampling mean
- why shape discipline matters

**Required prior knowledge**
- high-school algebra only

**Concepts introduced from first principles**
- vectors and matrices
- normalization to `[0,1]` and signed normalization
- discrete-time systems
- batch dimension
- PyTorch module/state dict basics

**Implementation surfaces it must cite / explain**
- examples drawn from observation shapes and brain forward path

**Outputs / figures / diagrams**
- tensor-shape ladder
- batch vs single-item diagrams
- simulation tick cartoon
- notation sidebars

**Cross-links**
- 02, 40, 41, 42, 61, 62

---

### 8.10 Document 22 — state, identity, lineage, and ownership contracts

**Filename:** `documentations/02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md`

**Scope**
- slot-backed dense state tensor
- canonical UID substrate
- lifecycle records
- UID-to-slot binding contract
- family, parent-role, trait-latent, and generation-depth ledgers
- shadow-column compatibility surfaces

**Non-scope**
- full PPO update derivations

**Key questions answered**
- why slot is not identity
- how births allocate new identity
- how deaths finalize identity
- what invariants protect against ownership corruption

**Required prior knowledge**
- 20 recommended

**Concepts introduced from first principles**
- identifier vs storage location
- lifecycle record
- monotonic identity
- invariant checking

**Implementation surfaces it must cite / explain**
- `tensor_crypt.agents.state_registry.Registry`
- `AgentLifecycleRecord`
- identity shadow columns
- invariant methods

**Outputs / figures / diagrams**
- slot/UID relationship diagram
- birth/death/finalization timeline
- parent-role graph example
- “wrong mental model vs correct mental model” box

**Cross-links**
- 31, 32, 42, 50, 51, 52, 60

---

### 8.11 Document 30 — world grid, map, H-zones, and catastrophe substrate

**Filename:** `documentations/03_simulation_substrate/30_world_grid_map_hzones_and_catastrophe_substrate.md`

**Scope**
- grid channel contract
- border walls and map generation
- heal/harm zone semantics
- overlap modes
- catastrophe field repainting and reversible overrides
- runtime modifiers applied to physics, perception, and reproduction

**Non-scope**
- full viewer controls
- PPO internals

**Key questions answered**
- what lives in each grid channel
- how walls and zones are generated
- how catastrophes modify the world
- what is persistent vs temporary

**Required prior knowledge**
- 20 and 21 recommended

**Concepts introduced from first principles**
- tensor-backed field channel
- overlap policy
- reversible overlay
- scheduler-driven world modification

**Implementation surfaces it must cite / explain**
- `tensor_crypt.world.spatial_grid.Grid`
- `tensor_crypt.world.procedural_map`
- `tensor_crypt.catastrophes` / catastrophe manager surfaces

**Outputs / figures / diagrams**
- grid-channel diagram
- wall-generation flow diagram
- H-zone overlap comparison box
- catastrophe overlay timeline
- catastrophe type table

**Cross-links**
- 31, 40, 50, 53, 60

---

### 8.12 Document 31 — tick order, physics, conflict resolution, and death

**Filename:** `documentations/03_simulation_substrate/31_tick_order_physics_conflict_resolution_and_death.md`

**Scope**
- engine tick order
- action sparsification
- movement intents
- wall collision handling
- ram handling
- contested move resolution
- environment effects
- reward timing relative to death
- death context recording and finalization sequence

**Non-scope**
- detailed PPO derivation

**Key questions answered**
- what happens first in a tick
- how conflicts are resolved
- when rewards are computed
- when death becomes final
- where logging happens relative to retirement

**Required prior knowledge**
- 20, 21, 22, 30

**Concepts introduced from first principles**
- discrete-time state transition
- resolution ordering
- pending vs resolved death context

**Implementation surfaces it must cite / explain**
- `tensor_crypt.simulation.engine.Engine.step`
- `tensor_crypt.world.physics.Physics`
- `tensor_crypt.population.evolution.Evolution.process_deaths`

**Outputs / figures / diagrams**
- master tick timeline
- conflict resolution state machine
- death-processing sequence diagram
- reward/death ordering callout

**Cross-links**
- 22, 32, 40, 42, 50, 51, 52

---

### 8.13 Document 32 — respawn, reproduction, mutation, and bloodline dynamics

**Filename:** `documentations/03_simulation_substrate/32_respawn_reproduction_mutation_and_bloodline_dynamics.md`

**Scope**
- binary reproduction substrate
- parent-role selection logic
- floor recovery
- extinction policies
- local placement and fallback
- crowding/cooldown/local-parent overlay doctrines
- latent trait budget representation
- mutation and rare mutation paths
- family shift mutation

**Non-scope**
- viewer-only controls except as references

**Key questions answered**
- how children are created
- how parent roles differ
- how traits are inherited and mutated
- how family identity is inherited or shifted
- how overlays constrain births

**Required prior knowledge**
- 22 and 31

**Concepts introduced from first principles**
- role-separated inheritance
- latent-to-trait decoding
- floor-recovery semantics
- overlay doctrine vs base mechanism

**Implementation surfaces it must cite / explain**
- `tensor_crypt.population.reproduction`
- `tensor_crypt.population.respawn_controller.RespawnController`
- `tensor_crypt.population.evolution.Evolution.apply_policy_noise`

**Outputs / figures / diagrams**
- parent-role diagram
- birth pipeline diagram
- overlay interaction matrix
- latent budget → trait value diagram
- family inheritance vs family shift box

**Cross-links**
- 22, 41, 50, 51, 52, 53, 60, 61

---

### 8.14 Document 40 — observation schema, rays, and feature construction

**Filename:** `documentations/04_learning_and_observation/40_observation_schema_rays_and_feature_construction.md`

**Scope**
- canonical ray contract
- canonical self features
- canonical context features
- legacy adapter surface
- effective vision modifiers
- observation bundle assembly
- shape contracts and failure modes

**Non-scope**
- deep MLP family architecture internals

**Key questions answered**
- what each observation feature means
- how raycasting works conceptually
- how self and context scalars are built
- how legacy fields are derived from canonical fields

**Required prior knowledge**
- 21 and 30

**Concepts introduced from first principles**
- raycasting
- normalized distance
- feature channel semantics
- compatibility adapter

**Implementation surfaces it must cite / explain**
- `tensor_crypt.world.observation_schema`
- `tensor_crypt.world.perception.Perception.cast_rays_batched`
- `Perception.build_observations`

**Outputs / figures / diagrams**
- per-ray feature diagram
- self/context feature tables
- canonical-to-legacy mapping table
- example tensor shapes
- perception pipeline flowchart

**Cross-links**
- 02, 21, 41, 42, 60, 62

---

### 8.15 Document 41 — bloodline brain architecture and inference paths

**Filename:** `documentations/04_learning_and_observation/41_bloodline_brain_architecture_and_inference_paths.md`

**Scope**
- bloodline family system
- family specs and topology invariants
- actor/critic heads
- split-input vs non-split families
- residual and gated blocks
- family-aware inference batching
- vmap eligibility constraints
- compatibility with canonical vs legacy observations

**Non-scope**
- PPO math beyond what is needed to situate actor/critic outputs

**Key questions answered**
- what makes one family different from another
- what invariants must hold within a family
- how canonical observations are consumed
- when experimental family vmap inference may be used

**Required prior knowledge**
- 21 and 40

**Concepts introduced from first principles**
- MLP block types
- shared topology invariants
- actor/critic split
- batched module execution concept

**Implementation surfaces it must cite / explain**
- `tensor_crypt.agents.brain.Brain`
- family spec definitions in config
- engine inference path selection

**Outputs / figures / diagrams**
- family comparison matrix
- brain block diagrams for each family class
- canonical input flattening diagram
- loop vs family-vmap path comparison box

**Cross-links**
- 22, 40, 42, 51, 52, 60, 62

---

### 8.16 Document 42 — UID-owned PPO rollouts, rewards, and updates

**Filename:** `documentations/04_learning_and_observation/42_uid_owned_ppo_rollouts_rewards_and_updates.md`

**Scope**
- reward surface actually implemented
- optional reward gating
- per-UID rollout buffers
- bootstrap staging
- terminal finalization
- advantage normalization and GAE path
- optimizer ownership and metadata validation
- family-aware update ordering
- checkpointed optimizer/buffer/training state

**Non-scope**
- broad RL history or unrelated algorithms

**Key questions answered**
- how reward is computed here
- how rollouts are stored and closed
- how updates are triggered
- why ownership is UID-based
- what can break continuity

**Required prior knowledge**
- 21, 22, 31, 40, 41

**Concepts introduced from first principles**
- PPO objective terms
- clipping
- entropy bonus
- KL threshold
- bootstrap value
- trajectory truncation

**Implementation surfaces it must cite / explain**
- `tensor_crypt.learning.ppo.PPO`
- reward validation in engine/runtime
- engine update boundary and buffer staging paths

**Outputs / figures / diagrams**
- rollout buffer lifecycle diagram
- GAE equation blocks
- reward-gate behavior table
- per-UID ownership diagram
- update trigger timeline

**Cross-links**
- 22, 31, 40, 41, 50, 51, 52, 60, 61, 62

---

### 8.17 Document 50 — telemetry ledgers, HDF5, Parquet, and run artifacts

**Filename:** `documentations/05_operations_reproducibility/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md`

**Scope**
- run directory creation
- run metadata
- HDF5 snapshots and heatmaps
- Parquet ledgers
- lineage JSON export
- saved brain artifacts
- schema fields and cadence behavior

**Non-scope**
- checkpoint serialization internals

**Key questions answered**
- what artifacts a run produces
- what each file means
- how to interpret births, deaths, life rows, collisions, PPO rows, tick summaries, family summaries, catastrophes, lineage graph
- when artifacts are emitted

**Required prior knowledge**
- 10 helpful, 20 and 22 recommended

**Concepts introduced from first principles**
- ledger
- artifact cadence
- HDF5 group
- Parquet writer
- lineage graph export

**Implementation surfaces it must cite / explain**
- `tensor_crypt.telemetry.data_logger.DataLogger`
- `tensor_crypt.telemetry.run_paths`
- `tensor_crypt.telemetry.lineage_export`

**Outputs / figures / diagrams**
- run-directory artifact tree
- artifact-emission timeline
- ledger-purpose matrix
- data-consumer guidance boxes

**Cross-links**
- 10, 11, 12, 22, 31, 32, 42, 51, 52, 53

---

### 8.18 Document 51 — checkpointing, atomic publish, resume, and schema safety

**Filename:** `documentations/05_operations_reproducibility/51_checkpointing_atomic_publish_resume_and_schema_safety.md`

**Scope**
- runtime checkpoint capture surface
- bundle contents
- atomic publish path
- manifest and latest pointer semantics
- load/restore ordering
- validation rules
- schema/version gates
- strictness flags

**Non-scope**
- general serialization theory unrelated to repository behavior

**Key questions answered**
- what exactly is saved
- how safe publication works
- what strict validation checks
- how restore rebuilds runtime state
- which surfaces are checkpoint-visible invariants

**Required prior knowledge**
- 22, 41, 42, 50

**Concepts introduced from first principles**
- atomic replace
- manifest publication
- restore-order dependency
- schema validation

**Implementation surfaces it must cite / explain**
- `tensor_crypt.checkpointing.atomic_checkpoint`
- `tensor_crypt.checkpointing.runtime_checkpoint`
- `validate_runtime_checkpoint`
- `restore_runtime_checkpoint`

**Outputs / figures / diagrams**
- checkpoint file-set diagram
- capture/validate/save/load/restore sequence diagram
- bundle contents table
- strictness-gate matrix
- restore-order callout box

**Cross-links**
- 42, 50, 52, 60, 62, 63

---

### 8.19 Document 52 — validation, determinism, resume consistency, and soak methods

**Filename:** `documentations/05_operations_reproducibility/52_validation_determinism_resume_consistency_and_soak_methods.md`

**Scope**
- determinism seeding model
- final validation suite
- determinism probe
- resume consistency probe
- catastrophe reproducibility probe
- save-load-save signature probe
- benchmark and soak runners
- relation between test suite and truth verification

**Non-scope**
- generic testing philosophy unrelated to the repository

**Key questions answered**
- how to verify the runtime
- what “deterministic” means in this codebase
- what resume consistency actually checks
- what invariants the soak runner protects

**Required prior knowledge**
- 20, 22, 31, 42, 51

**Concepts introduced from first principles**
- reproducibility vs determinism
- probe vs test vs benchmark vs soak
- runtime signature

**Implementation surfaces it must cite / explain**
- `tensor_crypt.audit.final_validation`
- headless benchmark runner
- headless soak runner
- relevant test suites as executable witnesses

**Outputs / figures / diagrams**
- validation harness map
- determinism probe comparison timeline
- resume-consistency fork diagram
- benchmark vs soak purpose table

**Cross-links**
- 12, 50, 51, 63

---

### 8.20 Document 53 — viewer controls, inspection, and diagnostics

**Filename:** `documentations/05_operations_reproducibility/53_viewer_controls_inspection_and_diagnostics.md`

**Scope**
- world renderer mental model
- camera and layout
- overlay toggles
- catastrophe and reproduction overlay hotkeys
- selection behavior
- side-panel information surfaces
- interpretation of on-screen diagnostics

**Non-scope**
- low-level rendering theory unless necessary

**Key questions answered**
- what every viewer control does
- how to inspect agents and zones
- how catastrophe and reproduction overlay controls map to runtime state
- how UI state relates to simulation state

**Required prior knowledge**
- 10 and 30 helpful

**Concepts introduced from first principles**
- camera fit and zoom
- selection precedence
- overlay visibility vs simulation state

**Implementation surfaces it must cite / explain**
- `tensor_crypt.viewer.main.Viewer`
- `tensor_crypt.viewer.input.InputHandler`
- viewer panels/layout/camera modules

**Outputs / figures / diagrams**
- hotkey table
- UI region map
- interaction-state diagram
- side-panel field glossary

**Cross-links**
- 10, 11, 12, 30, 32, 50

---

### 8.21 Document 60 — implemented behavior vs adjacent theory

**Filename:** `documentations/06_appendices/60_implemented_behavior_vs_adjacent_theory.md`

**Scope**
- repository-truth boundary document
- implemented vs conceptual tables by domain
- “do not over-claim” guardrails

**Non-scope**
- teaching theory in detail

**Key questions answered**
- which related ideas are not actually implemented here
- which config surfaces are documented but not active
- where conceptual appendices begin and implementation chapters end

**Required prior knowledge**
- none, but most useful after some technical reading

**Outputs / figures / diagrams**
- domain-by-domain implemented/adjacent matrix
- “common hallucination traps” table

**Cross-links**
- all docs, especially 11, 30, 40, 42, 51, 63

---

### 8.22 Document 61 — background math, Python, PyTorch, and RL appendix

**Filename:** `documentations/06_appendices/61_background_math_python_pytorch_and_rl_appendix.md`

**Scope**
- high-school algebra refresher
- normalization and ratios
- tensors and broadcasting
- PyTorch `Module`, parameters, optimizers, `state_dict`
- probability basics for categorical action sampling
- PPO and GAE intuition

**Non-scope**
- code-truth documentation of the repository itself

**Key questions answered**
- what background a weaker reader needs to climb the rest of the bundle

**Required prior knowledge**
- high-school arithmetic only

**Outputs / figures / diagrams**
- beginner math sidebars
- tiny tensor examples
- glossary-backed formula walk-throughs

**Cross-links**
- 21, 40, 41, 42, 62

---

### 8.23 Document 62 — equations, shapes, and contract reference

**Filename:** `documentations/06_appendices/62_equations_shapes_and_contract_reference.md`

**Scope**
- compact reference tables for equations, symbols, shape contracts, schema counts, and critical invariants

**Non-scope**
- narrative pedagogy

**Key questions answered**
- what the exact shapes and equations are without long prose

**Required prior knowledge**
- prior technical reading recommended

**Outputs / figures / diagrams**
- equation table
- tensor-shape tables
- schema-version reference table
- invariant checklist

**Cross-links**
- 02, 40, 41, 42, 51

---

### 8.24 Document 63 — contributor documentation truth contract

**Filename:** `documentations/06_appendices/63_contributor_documentation_truth_contract.md`

**Scope**
- mandatory rules for future doc writing
- code-citation policy
- terminology lock file
- consistency rules across waves
- anti-hallucination safeguards

**Non-scope**
- describing project behavior itself

**Key questions answered**
- how future writers avoid drift
- how new docs remain consistent
- how unsupported claims are marked

**Required prior knowledge**
- none

**Outputs / figures / diagrams**
- red-flag checklist
- claim classification rubric
- update workflow

**Cross-links**
- all docs; should be consulted before every generation wave

---

## 9. Beginner/manual document planning sheet

The beginner/operator manual must be treated as a distinct product with its own design rules.

### 9.1 Tone and level

- lower jargon density
- shorter paragraphs
- more screenshots/diagrams/tables
- more recipes and warning boxes
- fewer equations
- all technical terms defined on first contact
- no unexplained references to PPO, GAE, schema validation, UID ownership, or catastrophe scheduler internals

### 9.2 Mandatory sections

1. What Tensor Crypt is
2. How a run begins
3. The world at a glance
4. What agents are and why they differ by family
5. What happens each tick in simple language
6. What the viewer shows
7. Hotkeys and interaction
8. Important knobs for first runs
9. What files appear after a run
10. How to inspect whether a run “looks healthy”
11. Safe first experiments
12. Troubleshooting common operator confusion

### 9.3 Mandatory practical content

- a first-run quickstart path
- a “small safe local run” recipe
- a “debug slow and inspect” recipe
- a “collect artifacts for later study” recipe
- a “turn catastrophes on/off intentionally” recipe
- a “use reproduction overlays safely” recipe
- a “read the run directory” guide

### 9.4 Mandatory visual content

- annotated viewer layout
- hotkey cheat sheet
- run directory tree
- config section map
- one-tick cartoon timeline
- one-birth cartoon timeline
- one-death cartoon timeline

### 9.5 Mandatory honesty requirements

- must not say “agents learn to cooperate” unless the run evidence supports that specific claim
- must not claim every config field is active
- must not imply catastrophes permanently change stored traits when the code applies runtime modifiers reversibly
- must explain that some config surfaces are compatibility or restricted surfaces

---

## 10. Advanced technical paper/document planning sheets

This bundle is not a single whitepaper, but several documents can later be adapted into whitepaper-style artifacts.

### 10.1 Whitepaper-ready core

The following future documents are closest to publication-grade technical papers:

- 20 — project identity, runtime boot, and package map
- 22 — state identity, lineage, and ownership contracts
- 31 — tick order, physics, and death processing
- 40 — observation schema and perception
- 41 — bloodline brain architecture and inference paths
- 42 — UID-owned PPO rollouts, rewards, and updates
- 51 — checkpointing and schema safety
- 52 — validation and reproducibility methods

### 10.2 Internal technical-doc core

The following are especially important for internal project continuity:

- 11 — config reference
- 12 — experiment recipes
- 50 — telemetry and artifacts
- 53 — viewer and diagnostics
- 63 — contributor documentation truth contract

### 10.3 Educational-note core

The following are best suited for educational adaptation:

- 10 — operator manual
- 21 — Python/PyTorch/tensors/simulation foundations
- 61 — background appendix
- 62 — equations and shapes reference

---

## 11. Coverage matrix across topics vs documents

Legend: **P** primary home, **S** secondary treatment, **R** reference-only mention.

| Topic | 10 | 11 | 20 | 21 | 22 | 30 | 31 | 32 | 40 | 41 | 42 | 50 | 51 | 52 | 53 | 60 | 61 | 62 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Project identity and goals | S | R | P | R | R | R | R | R | R | R | R | R | R | R | R | R | R | R |
| Runtime assembly | S | R | P | S | S | R | S | R | R | R | R | S | S | S | R | R | R | R |
| Package/module architecture | R | R | P | R | S | R | R | R | R | R | R | R | R | R | S | R | R | R |
| Config system | S | P | S | R | R | S | S | S | S | S | S | S | S | S | S | S | R | R |
| Grid/world substrate | S | R | R | R | R | P | S | R | S | R | R | S | R | R | S | R | R | S |
| Field zones / heal-harm mechanics | S | R | R | R | R | P | S | R | S | R | R | S | R | R | S | R | R | S |
| Catastrophe systems | S | S | R | R | R | P | S | S | S | R | R | S | S | S | S | R | R | R |
| Tick order | S | R | S | S | S | R | P | R | R | R | S | S | S | S | R | R | R | S |
| Physics / conflicts / deaths | S | R | R | R | R | S | P | R | R | R | S | S | R | R | R | R | R | S |
| Registry / UID / lineage | R | S | S | R | P | R | S | S | R | R | S | S | S | S | R | P | R | S |
| Respawn / reproduction / mutation | S | S | R | R | S | R | S | P | R | S | S | S | S | S | S | P | S | S |
| Observation pipeline | S | S | R | S | R | S | R | R | P | S | S | R | R | R | R | P | S | P |
| Legacy vs canonical observations | R | S | R | R | R | R | R | R | P | S | S | R | R | R | R | P | R | P |
| Bloodline brain architecture | S | S | R | S | S | R | R | S | S | P | S | R | S | R | S | P | S | P |
| PPO and reward surface | R | S | R | S | S | R | S | R | S | S | P | S | S | S | R | P | S | P |
| Telemetry outputs | S | S | R | R | S | R | S | S | R | R | S | P | S | S | S | R | R | S |
| HDF5 / Parquet / JSON artifacts | S | S | R | R | R | R | R | R | R | R | R | P | S | S | R | R | R | S |
| Checkpoint manifests / restore | R | S | R | R | S | R | R | R | R | S | S | S | P | S | R | P | R | S |
| Validation / determinism / soak | R | S | R | R | S | R | S | R | R | R | S | S | S | P | R | P | R | R |
| Viewer / diagnostics / controls | P | S | R | R | R | S | R | R | R | R | R | S | R | R | P | R | R | R |
| Background math / Python / RL | R | R | R | P | R | R | R | R | S | S | S | R | R | R | R | R | P | P |

---

## 12. Concepts-to-prerequisites ladder

### 12.1 What “from first principles” means in this bundle

“From first principles” does **not** mean pretending the reader knows nothing and therefore flattening the project into triviality. It means each difficult concept is introduced only after its prerequisites have been explicitly supplied.

### 12.2 Pedagogical ladder

#### Level 0 — High-school mathematical floor
- arithmetic with ratios
- basic algebra
- coordinate grid intuition
- averages and proportions

#### Level 1 — Programming floor
- variable, function, class, dataclass
- list vs dict vs tensor
- index and shape intuition
- module and package basics

#### Level 2 — Simulation floor
- discrete ticks
- state transitions
- entity attributes changing over time
- deterministic seeding

#### Level 3 — Tensor floor
- vector, matrix, batch dimension
- normalization
- broadcasting intuition
- shape-checking discipline

#### Level 4 — Learning floor
- policy logits
- value estimate
- reward
- return
- bootstrap
- optimization step

#### Level 5 — Repository-specific advanced floor
- canonical observation contract
- slot/UID ownership split
- per-UID PPO buffers and optimizers
- schema versions and compatibility
- atomic checkpoint manifests
- catastrophe-driven reversible runtime modifiers

### 12.3 Where topics enter the bundle

| Topic | First serious entry point |
|---|---|
| High-school math refresher | 21, 61 |
| Python/PyTorch basics | 21, 61 |
| Simulation / game-loop basics | 10, 21, 31 |
| Vectors / tensors / shapes | 21, 40, 62 |
| Probability and categorical action sampling | 21, 42, 61 |
| PPO / optimization | 42, 61, 62 |
| Software architecture basics | 20 |
| Reproducibility / checkpoints / manifests | 50, 51, 52 |
| Validation methodology | 52 |

### 12.4 Pedagogical rules

- Every advanced chapter must begin with a prerequisite reminder block.
- Every advanced chapter must contain at least one beginner-intuition sidebar.
- Every reference chapter must link back to the teaching chapters rather than trying to re-teach everything.

---

## 13. Visual aids and diagram plan

### 13.1 Mandatory visual strategy

The future bundle should not rely on prose alone. Complex subsystems in this repository are stateful, multi-stage, and shape-sensitive. Diagrams are therefore not decorative; they are explanatory infrastructure.

### 13.2 Visual categories and placement

#### A. Conceptual diagrams
**Where:** 10, 20, 22, 30, 32, 42  
**Why:** to establish mental models before implementation detail

Examples:
- “what a run is” diagram
- slot vs UID conceptual diagram
- parent-role conceptual diagram
- checkpoint file-set conceptual diagram

#### B. Execution-flow diagrams
**Where:** 20, 31, 42, 51, 52  
**Why:** to show ordering, dependencies, and failure-sensitive boundaries

Examples:
- runtime assembly sequence
- engine tick order timeline
- checkpoint capture → validate → publish → restore sequence
- validation probe control flow

#### C. State-transition diagrams
**Where:** 22, 31, 32  
**Why:** lifecycle, death, respawn, and overlay logic are easier to understand as transition systems

Examples:
- UID lifecycle states
- death marking vs finalization
- reproduction overlay decision tree

#### D. Tensor-shape diagrams
**Where:** 02, 21, 40, 41, 42, 62  
**Why:** observation and PPO reasoning requires strict shape literacy

Examples:
- canonical rays `[B, R, F]`
- canonical self `[B, S]`
- canonical context `[B, C]`
- action logits `[B, A]`
- value head `[B, 1]`

#### E. Pipeline diagrams
**Where:** 40, 41, 42, 50, 51  
**Why:** to connect multi-stage flows across subsystems

Examples:
- perception pipeline
- brain input pipeline
- PPO buffer/update pipeline
- artifact emission pipeline
- checkpoint restore-order pipeline

#### F. Config maps
**Where:** 10, 11, 12  
**Why:** operator cognition depends on grouping knobs by purpose rather than by dataclass class name alone

Examples:
- config domain map
- safe vs dangerous knobs
- knob recipes by outcome

#### G. Lineage / genealogy diagrams
**Where:** 22, 32, 50  
**Why:** parent-role separation is a unique conceptual burden in the repository

Examples:
- brain-parent edge vs trait-parent edge vs anchor-parent edge
- lineage-depth examples
- life/death lineage export sample

#### H. Artifact/output tree diagrams
**Where:** 10, 50, 51  
**Why:** run directories and checkpoint directories must be inspectable by humans

Examples:
- run directory tree
- checkpoint directory tree
- manifest/latest-pointer relations

#### I. Run-lifecycle timelines
**Where:** 10, 20, 31, 50, 52  
**Why:** operators and researchers both need temporal orientation

Examples:
- from launch to first viewer frame
- from tick start to PPO update boundary
- from run start to close/flush

#### J. Equations and notation blocks
**Where:** 21, 40, 42, 62  
**Why:** mathematics should be readable, bounded, and linked back to implementation

Examples:
- normalization equations
- reward form
- GAE recurrence
- PPO clipped objective

#### K. Glossary tables and callout boxes
**Where:** all major docs  
**Why:** the bundle serves mixed-skill readers

Required callout types:
- beginner intuition
- implementation truth
- guarded compatibility
- unread/dead surface
- danger / migration risk
- related theory, not implemented here

---

## 14. Notation / glossary / terminology plan

### 14.1 Terminology lock list

The future bundle must standardize the following terms and forbid sloppy substitutes.

| Correct term | Must not be casually replaced by |
|---|---|
| slot | agent identity |
| UID | slot id, index, array position |
| canonical observation | generic observation vector |
| legacy bridge | old observation system still fully equivalent |
| brain parent | parent |
| trait parent | parent |
| anchor parent | parent |
| bloodline family | team, class, species |
| run directory | checkpoint |
| checkpoint bundle | run state dump folder in general |
| catastrophe runtime modifier | permanent trait mutation |
| active runtime knob | any public field |
| guarded compatibility surface | fully implemented mode |
| unread/effectively dead surface | invalid setting |

### 14.2 Notation conventions

- `B` = batch size
- `R` = number of rays
- `F` = per-ray feature width
- `S` = canonical self width
- `C` = canonical context width
- `A` = action dimension
- `UID` always rendered as uppercase when referring to canonical identity
- config fields rendered as code identifiers, for example `cfg.RESPAWN.ANCHOR_PARENT_SELECTOR`
- module paths rendered as code identifiers, for example `tensor_crypt.learning.ppo.PPO`

### 14.3 Glossary design policy

The glossary should be split into three tiers:

1. **Core simulation terms** — agent, slot, tick, wall, H-zone, catastrophe, family
2. **Core learning terms** — logits, value, reward, rollout, bootstrap, KL, entropy
3. **Core systems/reproducibility terms** — manifest, latest pointer, schema version, atomic publish, determinism probe

Glossary entries should include:
- plain-language definition
- precise repository-specific meaning
- first document where the term is introduced
- related terms that must not be confused with it

---

## 15. Artifact and appendix plan

### 15.1 Artifact classes that future docs should explicitly show

- `config.json`
- `run_metadata.json`
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
- `brains/brains_tick_<tick>.pt`
- checkpoint bundle files
- checkpoint manifests
- latest pointer files

### 15.2 Appendix classes

- background math appendix
- implementation-vs-theory appendix
- equations/shapes reference appendix
- contributor truth contract appendix

### 15.3 What should remain out of the mainline narrative

- broad RL history
- generic PyTorch tutorials not needed for this repository
- speculative future roadmap unless clearly labeled as non-implemented
- alternative algorithms not present in the code
- unsupported configuration fantasies

---

## 16. Prompt-wave / generation-wave plan

### 16.1 One-shot generation assessment

A single prompt should **not** be used to generate the full final bundle. The bundle is too large, too interdependent, and too truth-sensitive. One-shot generation would create terminology drift, duplicated explanation, inconsistent field naming, and likely confusion between implemented and adjacent theory.

### 16.2 Recommended wave count

Recommended production plan: **8 waves** plus a final consistency pass.

### 16.3 Wave structure

#### Wave 0 — conventions freeze
Generate first:
- 00 index
- 01 reading tracks and dependency map
- 02 notation/glossary/shape legend
- 63 contributor documentation truth contract

**Goal:** freeze naming, terminology, citation style, figure conventions, and honesty policy before writing content-heavy chapters.

#### Wave 1 — operator layer
Generate:
- 10 operator manual
- 11 config reference
- 12 experiment recipes
- 53 viewer controls

**Goal:** give immediate usable material while terminology is still fresh and controlled.

#### Wave 2 — architecture foundation
Generate:
- 20 runtime boot and package map
- 21 foundations
- 22 state/identity/lineage/ownership contracts

**Goal:** establish conceptual spine for all later chapters.

#### Wave 3 — world and mechanics
Generate:
- 30 world/grid/H-zones/catastrophe substrate
- 31 tick order/physics/death

**Goal:** define how the simulation world evolves before discussing learning.

#### Wave 4 — reproduction and population dynamics
Generate:
- 32 respawn/reproduction/mutation/bloodlines

**Goal:** isolate a conceptually dense subsystem that depends strongly on 22 and 31.

#### Wave 5 — observation and brain path
Generate:
- 40 observation schema and perception
- 41 bloodline brain architecture and inference paths

**Goal:** connect world state to brain inputs.

#### Wave 6 — PPO and scientific operations
Generate:
- 42 UID-owned PPO
- 50 telemetry and artifacts
- 51 checkpointing and schema safety
- 52 validation/determinism/soak methods

**Goal:** complete the learning and reproducibility stack.

#### Wave 7 — appendices and truth-boundary reinforcement
Generate:
- 60 implemented vs adjacent theory
- 61 background appendix
- 62 equations/shapes reference
- 03 figure/artifact/source reference index

**Goal:** add support material without disturbing core chapters.

#### Wave 8 — cross-document repair pass
Tasks:
- terminology consistency audit
- cross-link repair
- duplicate explanation pruning
- figure numbering normalization
- reference and glossary consistency pass
- explicit implemented-vs-theory check across the bundle

### 16.4 Stable naming and reference conventions that must be frozen in Wave 0

- document IDs and filenames
- section numbering depth
- code citation format
- config field rendering format
- tensor-shape notation
- capitalized terminology such as UID, H-zone, PPO, GAE
- family names exactly as configured
- catastrophe IDs exactly as configured

### 16.5 Anti-drift generation rule

Later waves must not rename concepts already frozen in Wave 0 and Wave 2 unless a deliberate consistency migration is performed across the whole bundle.

---

## 17. Consistency and quality-control plan

### 17.1 Consistency controls

1. One terminology lock file: document 02
2. One contributor truth contract: document 63
3. One figure and artifact index: document 03
4. One config atlas: document 11
5. One implemented-vs-theory appendix: document 60

### 17.2 Mandatory review passes for every generated document

Each future document must pass the following checks before acceptance:

- **Truth pass:** every implementation claim traceable to code
- **Boundary pass:** adjacent theory clearly labeled
- **Terminology pass:** no slot/UID conflation
- **Shape pass:** tensor dimensions consistent with doc 62
- **Cross-link pass:** references to prerequisite docs are present
- **Operator pass:** jargon introduced or linked before use
- **Figure pass:** all promised figures either included or explicitly deferred

### 17.3 Required use of test suite as corroboration surface

The raw dump evidences an extensive Python test corpus covering checkpointing, reward gating, bloodline families, reproduction overlays, catastrophe behavior, viewer interactions, determinism, and telemetry. Future documentation should use tests as corroborating witnesses for delicate behavioral claims. Where possible, technical chapters should cite both implementation surfaces and relevant tests.

### 17.4 Required use of config comments as corroboration surface

The runtime configuration file already contains an audit legend and per-field classification comments. Future docs, especially the config atlas, should reuse this structure rather than inventing a parallel taxonomy.

---

## 18. Risks, failure modes, and anti-hallucination safeguards

### 18.1 Likely failure modes

#### Failure mode A — Monolith collapse
A single huge document causes beginners to drown and advanced readers to lose navigation.

**Safeguard:** strict multi-document partitioning.

#### Failure mode B — Theory overwrites implementation
PPO, catastrophe systems, or reproduction theory is described in broader terms than the code actually supports.

**Safeguard:** implemented-vs-adjacent labels in every chapter and dedicated appendix 60.

#### Failure mode C — Slot/UID confusion
This would poison multiple chapters at once.

**Safeguard:** doc 22 written early, terminology lock in doc 02, contributor truth contract in doc 63.

#### Failure mode D — Config dishonesty
Fields are described as though active because they exist in the config dataclass.

**Safeguard:** config atlas based on runtime-config audit comments and runtime validation code.

#### Failure mode E — Checkpoint/resume oversimplification
The restore-order contract and manifest rules are easy to flatten incorrectly.

**Safeguard:** dedicated doc 51 and validation doc 52.

#### Failure mode F — Missing evidence hidden instead of declared
The current input is a concatenated Python dump and does not prove non-Python surfaces.

**Safeguard:** explicit evidence-limit boxes in architecture, operations, and maintainer docs.

#### Failure mode G — Inconsistent wave output
Later docs rename terms or change scope.

**Safeguard:** wave 0 freeze, wave 8 repair pass, doc 63 governance rules.

### 18.2 Anti-hallucination safeguards

- every technical subsection begins with “implemented surface” identifiers
- every adjacent theory subsection begins with a label block stating it is explanatory background
- every config subsection declares active/guarded/dead/schema-critical status
- every sensitive claim about resume, determinism, or ownership cites both code and validation/test evidence where available
- unsupported future ideas go only in clearly marked “not implemented here” boxes

---

## 19. Final recommended roadmap for actually writing the bundle

### 19.1 Immediate next action

Write Wave 0 first. Do not write any large explanatory chapter before freezing terminology, citation style, document roles, and bundle hierarchy.

### 19.2 Recommended authoring order

1. 00 index
2. 01 dependency map
3. 02 notation/glossary/shape legend
4. 63 contributor truth contract
5. 10 operator manual
6. 11 config atlas
7. 20 runtime boot/package map
8. 21 foundations
9. 22 identity/ownership contracts
10. 30 world substrate
11. 31 tick order/physics/death
12. 32 reproduction/mutation/bloodlines
13. 40 observations
14. 41 brains/inference
15. 42 PPO
16. 50 telemetry/artifacts
17. 51 checkpointing
18. 52 validation/reproducibility
19. 53 viewer controls
20. 12 recipes
21. 60 truth-boundary appendix
22. 61 background appendix
23. 62 equations/shapes reference
24. 03 figure/artifact/source index
25. final full-bundle consistency pass

### 19.3 Acceptance criteria for the finished bundle

The bundle is ready only when all of the following are true:

- a beginner can follow the operator manual and run the project safely
- an engineer can locate canonical modules and understand boot order
- an ML reader can trace observations to brains to PPO updates without ambiguity
- a maintainer can distinguish slot, UID, family, parent roles, and checkpoint ownership correctly
- a researcher can interpret ledgers, manifests, and validation probes correctly
- no document silently upgrades adjacent theory into implemented behavior
- all major diagrams promised in this plan exist
- all cross-links resolve and terminology is consistent across the suite

### 19.4 Final planning recommendation

The future bundle should be treated as a **documentation system**, not a stack of independent markdown files. The index, glossary, config atlas, truth contract, and implemented-vs-theory appendix are structural control documents. They should be written early and maintained continuously. The rest of the bundle should then be authored as rigorously linked chapters whose order mirrors the real architecture of the codebase rather than an arbitrary pedagogical fantasy.

This is the only reliable path to a book-grade, research-grade, production-grade documentation suite for the repository described by the provided code dump.
