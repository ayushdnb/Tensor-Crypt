# D01 - Reading Tracks and Dependency Map

## Purpose

This document defines practical reading orders for the audited Tensor Crypt technical-document bundle. It exists to keep readers from entering detail-heavy chapters without the identity, notation, or runtime-construction context those chapters depend on.

## Scope Boundary

This document is a navigation aid. It does not restate subsystem behavior in full, and it does not override the local scope statements in the destination chapters.

## Evidence Basis

The dependency map below is based on the current repository structure and on the implementation relationships verified during the documentation audit. In particular, the hard prerequisites reflect actual coupling among:

- runtime assembly in `tensor_crypt.app.launch` and `tensor_crypt.app.runtime`
- identity and lineage ownership in `tensor_crypt.agents.state_registry`
- world, physics, observation, reproduction, and PPO code under `tensor_crypt.world`, `tensor_crypt.population`, `tensor_crypt.learning`, and `tensor_crypt.simulation`
- artifact, checkpoint, validation, and viewer code under `tensor_crypt.telemetry`, `tensor_crypt.checkpointing`, `tensor_crypt.audit`, and `tensor_crypt.viewer`

## Hard and Soft Dependencies

| Dependency class | Meaning |
|---|---|
| Hard prerequisite | The later document relies on terms, invariants, or ordering introduced earlier. |
| Soft prerequisite | The later document can be read alone, but the earlier document materially lowers ambiguity. |

## Bundle Dependency Map

### Constitutional layer

- [D00](./00_documentation_bundle_index.md) before all other documents
- [D02](./02_notation_glossary_and_shape_legend.md) before all subsystem-heavy documents
- [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md) before making documentation changes

### Runtime and identity spine

- [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md) before [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md), [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md), [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md), and [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md) before [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md), [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md), [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md), and [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)

### Simulation and learning path

- [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md) before [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md) and [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md) before [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md), [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md), [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md), and [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md) before [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md), [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md), and [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)
- [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md) before [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)

### Reproducibility and operations path

- [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md) before [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md) and [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md) before [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
- [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md) before [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md), [D12](../01_operations_and_config/12_experiment_recipes_and_safe_knob_sets.md), and [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)

## Recommended Reading Tracks

### Operator track

1. [D00](./00_documentation_bundle_index.md)
2. [D02](./02_notation_glossary_and_shape_legend.md)
3. [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md)
4. [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md)
5. [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)
6. [D12](../01_operations_and_config/12_experiment_recipes_and_safe_knob_sets.md)
7. [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)

This path is optimized for launching runs, using the viewer, and understanding the artifact surface without first reading the full implementation lineage.

### Architecture track

1. [D00](./00_documentation_bundle_index.md)
2. [D02](./02_notation_glossary_and_shape_legend.md)
3. [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
4. [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
5. [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
6. [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
7. [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)

This path establishes package ownership, construction order, state layout, and world-step semantics before branching into reproduction or learning details.

### Learning track

1. [D00](./00_documentation_bundle_index.md)
2. [D02](./02_notation_glossary_and_shape_legend.md)
3. [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
4. [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
5. [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
6. [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
7. [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
8. [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md)
9. [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
10. [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)

This path keeps the observation contract ahead of brain topology and keeps brain topology ahead of PPO ownership and update semantics.

### Reproducibility track

1. [D00](./00_documentation_bundle_index.md)
2. [D02](./02_notation_glossary_and_shape_legend.md)
3. [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
4. [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
5. [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
6. [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
7. [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
8. [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
9. [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md)

This path is appropriate for checkpoint reviewers, validation auditors, and maintainers concerned with run reproducibility.

### Contributor track

1. [D00](./00_documentation_bundle_index.md)
2. [D02](./02_notation_glossary_and_shape_legend.md)
3. [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md)
4. [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
5. [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
6. [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
7. [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
8. [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
9. [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
10. [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
11. [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)

This path is intentionally stricter because documentation errors often come from crossing identity, runtime, or artifact boundaries without first reading the corresponding implementation contracts.

## High-Risk Skips

| Skip | Risk introduced |
|---|---|
| Reading D42 before D22 | PPO ownership, checkpoint state, and transition buffers are easy to misread as slot-owned instead of UID-owned. |
| Reading D41 before D40 | Brain topology becomes detached from the canonical observation keys and shapes it consumes. |
| Reading D51 before D22 | Checkpoint schema descriptions lose the identity substrate needed to interpret restored tensors and ledgers. |
| Reading D11 without D10 | Config fields are easy to overread as guarantees rather than operator-facing knobs with status labels. |
| Reading D52 before D50 and D51 | Validation probes are easy to confuse with ordinary runtime behavior or ordinary save paths. |

## Suggested Full Linear Order

Readers who want the highest-stability single pass can use:

1. [D00](./00_documentation_bundle_index.md)
2. [D01](./01_reading_tracks_and_dependency_map.md)
3. [D02](./02_notation_glossary_and_shape_legend.md)
4. [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md)
5. [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md)
6. [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
7. [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
8. [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)
9. [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
10. [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
11. [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md)
12. [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
13. [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md)
14. [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
15. [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
16. [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
17. [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
18. [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)
19. [D12](../01_operations_and_config/12_experiment_recipes_and_safe_knob_sets.md)
20. [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md)
21. [D61](../06_boundaries_and_appendices/61_background_math_python_pytorch_and_rl_appendix.md)
22. [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)
23. [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md)
24. [D03](./03_figure_artifact_and_source_reference_index.md)
