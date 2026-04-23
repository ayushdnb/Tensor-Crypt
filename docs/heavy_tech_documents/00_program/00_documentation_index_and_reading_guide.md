# Tensor Crypt Documentation Index and Reading Guide

> Scope: Provide the authoritative front door into the documentation corpus, explain the layer model, and route different reader types through the corpus efficiently.

## Who this document is for
All readers: beginners, operators, maintainers, technical auditors, and public readers who need a truthful map of the repository.

## What this document covers
- the purpose of each documentation layer
- role-based reading routes
- task-oriented entry points
- where to find deep technical versus lookup-first material
- how the corpus distinguishes repository fact from background teaching

## What this document does not cover
- full subsystem detail
- complete configuration reference tables
- benchmark interpretation guidance in depth

## Prerequisite reading
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Why this corpus exists

Tensor Crypt is not organized as a single-script toy environment. The repository is structured around a canonical `tensor_crypt` package, repository-root entry surfaces, compatibility wrappers, a dataclass-governed runtime configuration surface, slot-backed execution state, UID-owned identity and PPO state, checkpoint manifests and latest pointers, an operator-facing viewer, and explicit validation harnesses. This corpus is therefore organized as a technical book, an operator manual, and a reference system rather than as a single README-style narrative.

The corpus is designed around one central discipline: **teach broadly, but state repository truth conservatively**. Foundations chapters may explain mathematics or reinforcement learning from first principles. System, mechanics, learning, and operations chapters document only implemented behavior, explicit validation surfaces, and narrow consequences that follow from them.

## 2. Layer model

| Layer | Directory | Primary purpose | Typical reader |
| --- | --- | --- | --- |
| Program / governance | `docs/heavy_tech_documents/00_program` | Navigation, evidence rules, glossary, corpus conventions | Everyone |
| Foundations | `docs/heavy_tech_documents/01_foundations` | Build mathematical, ML, RL, and simulation prerequisites | Beginners and refreshers |
| System | `docs/heavy_tech_documents/02_system` | Explain repository identity, package boundaries, launch sequence, config taxonomy | Technical readers and maintainers |
| Mechanics | `docs/heavy_tech_documents/03_mechanics` | Explain the simulated world, agents, observations, actions, respawn, catastrophes | Technical readers |
| Learning | `docs/heavy_tech_documents/04_learning` | Explain PPO ownership, buffers, reward logic, inference paths, checkpoint-visible learning state | Technical readers and maintainers |
| Operations | `docs/heavy_tech_documents/05_operations` | Explain how to run, inspect, validate, benchmark, extend, and troubleshoot safely | Operators and maintainers |
| Game manual | `docs/heavy_tech_documents/06_game_manual` | Explain the simulation as a coherent rule-driven world | Public readers and operators |
| Reference | `docs/heavy_tech_documents/07_reference` | Fast lookup: config, schemas, compatibility, modules, FAQs, manifests | Everyone |

## 3. Start here if…

### Start here if the goal is to understand the repository quickly
1. `docs/heavy_tech_documents/02_system/00_repository_identity_entry_surfaces_and_public_contract.md`
2. `docs/heavy_tech_documents/02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md`
3. `docs/heavy_tech_documents/02_system/02_runtime_assembly_launch_sequence_and_session_graph.md`
4. `docs/heavy_tech_documents/03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md`
5. `docs/heavy_tech_documents/04_learning/00_learning_system_overview_and_data_ownership.md`
6. `docs/heavy_tech_documents/05_operations/00_operator_quickstart_and_common_run_modes.md`

### Start here if the goal is beginner study
1. `docs/heavy_tech_documents/01_foundations/00_foundations_learning_roadmap.md`
2. `docs/heavy_tech_documents/01_foundations/01_high_school_math_for_scaling_rates_and_normalization.md`
3. `docs/heavy_tech_documents/01_foundations/04_linear_algebra_tensors_shapes_and_batching.md`
4. `docs/heavy_tech_documents/01_foundations/06_reinforcement_learning_mdp_policy_value_and_advantage.md`
5. `docs/heavy_tech_documents/01_foundations/07_ppo_from_first_principles_to_uid_owned_rollouts.md`
6. bridge into `docs/heavy_tech_documents/03_mechanics/04_observation_schema_perception_and_ray_semantics.md`

### Start here if the goal is operating the system
1. `docs/heavy_tech_documents/05_operations/00_operator_quickstart_and_common_run_modes.md`
2. `docs/heavy_tech_documents/05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md`
3. `docs/heavy_tech_documents/05_operations/02_run_directory_artifacts_and_file_outputs.md`
4. `docs/heavy_tech_documents/05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md`
5. `docs/heavy_tech_documents/05_operations/05_validation_determinism_resume_consistency_and_soak.md`
6. `docs/heavy_tech_documents/05_operations/08_troubleshooting_and_failure_atlas.md`

### Start here if the goal is auditing compatibility or schema risk
1. `docs/heavy_tech_documents/02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md`
2. `docs/heavy_tech_documents/02_system/03_runtime_config_taxonomy_and_knob_safety.md`
3. `docs/heavy_tech_documents/04_learning/04_checkpoint_visible_learning_state_and_restore_order.md`
4. `docs/heavy_tech_documents/07_reference/01_schema_versions_and_compatibility_surfaces.md`
5. `docs/heavy_tech_documents/07_reference/97_repository_truth_gaps_and_explicit_unknowns.md`

### Start here if the goal is world behavior rather than implementation
1. `docs/heavy_tech_documents/06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md`
2. `docs/heavy_tech_documents/03_mechanics/00_grid_substrate_and_spatial_field_model.md`
3. `docs/heavy_tech_documents/03_mechanics/06_physics_collisions_damage_healing_and_death.md`
4. `docs/heavy_tech_documents/03_mechanics/08_catastrophe_system_scheduler_and_world_overlays.md`

## 4. Reading strategies

> **Recommended pattern**
> Do not read the corpus strictly top to bottom unless the goal is full study. The layer model exists so that operators, maintainers, and learners can enter at different depths.

### Guided study route
Follow the foundations ladder, then the system layer, then mechanics, then learning, then operations.

### Task route
Use the operations layer first, then jump outward through the “If debugging this, inspect…” links.

### Code-to-doc route
Use `docs/heavy_tech_documents/07_reference/02_module_reference_index.md`, then jump into the owning narrative document.

## 5. Narrative versus reference

Narrative documents explain mechanism, invariants, and consequences. Reference documents compress lookup detail. The corpus intentionally does **not** merge those into one format because a long-form explanation is difficult to scan and a pure reference is a poor teaching medium.

## 6. Visual structure policy

Mermaid diagrams, tables, and schematic panels are used only where they reduce ambiguity or compress structure. Diagrams have owner documents and captions. They are not decorative.

## 7. Front-door companion files

- `docs/heavy_tech_documents/00_program/01_documentation_evidence_policy_and_corpus_conventions.md` — how truth is classified and how future maintainers should update the corpus.
- `docs/heavy_tech_documents/00_program/02_glossary_notation_and_schema_legend.md` — canonical term definitions, notation rules, and schema legend.

## Read next
- [Evidence policy and corpus conventions](01_documentation_evidence_policy_and_corpus_conventions.md)
- [Glossary, notation, and schema legend](02_glossary_notation_and_schema_legend.md)
- [Foundations learning roadmap](../01_foundations/00_foundations_learning_roadmap.md)

## Related reference
- [Corpus manifest](../07_reference/99_corpus_manifest.md)
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## If debugging this, inspect…
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)
- [Module reference index](../07_reference/02_module_reference_index.md)

## Terms introduced here
- `layer model`
- `narrative document`
- `reference document`
- `task route`
- `code-to-doc route`
