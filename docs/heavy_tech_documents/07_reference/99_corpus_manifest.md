# Corpus Manifest

> Scope: List the generated documentation and asset files included in the packaged corpus, grouped by layer for audit and navigation purposes.

## Who this document is for
Auditors, maintainers, and readers who want a concise inventory of the generated corpus.

## What this document covers
- generated files by layer
- asset inventory
- count summary

## What this document does not cover
- prose explanation of each file’s contents

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)

## 1. Count summary

| Item | Count |
| --- | ---: |
| narrative/reference markdown files | 48 |
| asset markdown files | 17 |
| total markdown files | 65 |

## 2. Inventory by layer

### `00_program`

- `docs/00_program/00_documentation_index_and_reading_guide.md`
- `docs/00_program/01_documentation_evidence_policy_and_corpus_conventions.md`
- `docs/00_program/02_glossary_notation_and_schema_legend.md`

### `01_foundations`

- `docs/01_foundations/00_foundations_learning_roadmap.md`
- `docs/01_foundations/01_high_school_math_for_scaling_rates_and_normalization.md`
- `docs/01_foundations/02_probability_statistics_and_expected_value_for_rl.md`
- `docs/01_foundations/03_calculus_gradients_and_backprop_for_this_project.md`
- `docs/01_foundations/04_linear_algebra_tensors_shapes_and_batching.md`
- `docs/01_foundations/05_neural_networks_mlp_design_and_function_approximation.md`
- `docs/01_foundations/06_reinforcement_learning_mdp_policy_value_and_advantage.md`
- `docs/01_foundations/07_ppo_from_first_principles_to_uid_owned_rollouts.md`
- `docs/01_foundations/08_grid_world_simulation_engines_and_state_ownership_primer.md`

### `02_system`

- `docs/02_system/00_repository_identity_entry_surfaces_and_public_contract.md`
- `docs/02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md`
- `docs/02_system/02_runtime_assembly_launch_sequence_and_session_graph.md`
- `docs/02_system/03_runtime_config_taxonomy_and_knob_safety.md`

### `03_mechanics`

- `docs/03_mechanics/00_grid_substrate_and_spatial_field_model.md`
- `docs/03_mechanics/01_procedural_map_generation_walls_and_zones.md`
- `docs/03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md`
- `docs/03_mechanics/03_traits_bloodlines_families_and_brain_instantiation.md`
- `docs/03_mechanics/04_observation_schema_perception_and_ray_semantics.md`
- `docs/03_mechanics/05_action_surface_intents_and_move_resolution.md`
- `docs/03_mechanics/06_physics_collisions_damage_healing_and_death.md`
- `docs/03_mechanics/07_reproduction_respawn_mutation_and_lineage.md`
- `docs/03_mechanics/08_catastrophe_system_scheduler_and_world_overlays.md`

### `04_learning`

- `docs/04_learning/00_learning_system_overview_and_data_ownership.md`
- `docs/04_learning/01_ppo_buffers_bootstrap_and_update_cadence.md`
- `docs/04_learning/02_reward_design_gate_logic_and_value_targets.md`
- `docs/04_learning/03_inference_execution_paths_loop_vs_family_vmap.md`
- `docs/04_learning/04_checkpoint_visible_learning_state_and_restore_order.md`

### `05_operations`

- `docs/05_operations/00_operator_quickstart_and_common_run_modes.md`
- `docs/05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md`
- `docs/05_operations/02_run_directory_artifacts_and_file_outputs.md`
- `docs/05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md`
- `docs/05_operations/04_telemetry_ledgers_snapshots_exports_and_lineage_graph.md`
- `docs/05_operations/05_validation_determinism_resume_consistency_and_soak.md`
- `docs/05_operations/06_benchmarking_and_performance_probe_manual.md`
- `docs/05_operations/07_extension_safety_testing_and_change_protocol.md`
- `docs/05_operations/08_troubleshooting_and_failure_atlas.md`

### `06_game_manual`

- `docs/06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md`

### `07_reference`

- `docs/07_reference/00_config_reference_index.md`
- `docs/07_reference/01_schema_versions_and_compatibility_surfaces.md`
- `docs/07_reference/02_module_reference_index.md`
- `docs/07_reference/03_faq_and_reader_paths.md`
- `docs/07_reference/97_repository_truth_gaps_and_explicit_unknowns.md`
- `docs/07_reference/98_crosslink_integrity_and_publication_checklist.md`
- `docs/07_reference/99_corpus_manifest.md`

### `assets`

- `docs/assets/diagrams/learning/buffer_bootstrap_update_flow.md`
- `docs/assets/diagrams/learning/checkpoint_visible_learning_state_map.md`
- `docs/assets/diagrams/learning/inference_path_comparison.md`
- `docs/assets/diagrams/learning/uid_owned_ppo_lifecycle_panel.md`
- `docs/assets/diagrams/mechanics/action_resolution_flow.md`
- `docs/assets/diagrams/mechanics/catastrophe_scheduler_state_map.md`
- `docs/assets/diagrams/mechanics/collision_and_death_context_flow.md`
- `docs/assets/diagrams/mechanics/observation_tensor_atlas.md`
- `docs/assets/diagrams/mechanics/respawn_and_lineage_flow.md`
- `docs/assets/diagrams/operations/checkpoint_publish_and_restore_sequence.md`
- `docs/assets/diagrams/operations/run_directory_artifact_tree.md`
- `docs/assets/diagrams/operations/troubleshooting_decision_tree.md`
- `docs/assets/diagrams/operations/validation_harness_matrix.md`
- `docs/assets/diagrams/operations/viewer_hud_and_panels_atlas.md`
- `docs/assets/diagrams/system/config_taxonomy_map.md`
- `docs/assets/diagrams/system/package_and_compatibility_surface_map.md`
- `docs/assets/diagrams/system/runtime_assembly_sequence.md`
## Read next
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## If debugging this, inspect…
- [Module reference index](02_module_reference_index.md)

## Terms introduced here
- `manifest`
- `inventory`
- `layer inventory`
