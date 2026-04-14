# D00 - Documentation Bundle Index

## Purpose

This document is the entry point for the audited Tensor Crypt technical-document bundle in `docs/technical_documents/`. It defines the publication hierarchy, explains how the bundle is partitioned, and identifies which materials are normative for implementation-truth claims.

This document does not explain subsystem behavior in depth. Runtime mechanics, learning logic, artifact semantics, and validation methods are owned by the later chapters cited below.

## Scope Boundary

This index governs the technical-document bundle only. It does not replace the repository README, test suite, or source tree. When a prose claim conflicts with the implementation, the implementation remains authoritative until the documentation is repaired.

## Evidence Basis

The bundle has been audited against the current repository tree, with special reliance on:

- canonical runtime and package surfaces under `tensor_crypt/`
- compatibility wrappers at repository root and under `engine/` and `viewer/`
- tests under `tests/`
- utility scripts under `scripts/`
- the current `pyproject.toml`, `requirements.txt`, and generated-artifact helpers

Planning prompts and authoring notes preserved in `98_authoring_inputs/` are not implementation authority.

## Bundle Structure

The published bundle is organized as follows.

```text
docs/technical_documents/
  00_meta/
    00_documentation_bundle_index.md
    01_reading_tracks_and_dependency_map.md
    02_notation_glossary_and_shape_legend.md
    03_figure_artifact_and_source_reference_index.md
    99_documentation_audit_change_log.md
  01_operations_and_config/
    10_operator_runbook_and_game_manual.md
    11_config_reference_active_guarded_dead.md
    12_experiment_recipes_and_safe_knob_sets.md
  02_system_foundations/
    20_project_identity_runtime_boot_and_package_map.md
    21_python_pytorch_tensors_and_simulation_foundations.md
    22_state_identity_lineage_and_ownership_contracts.md
  03_world_and_simulation/
    30_world_grid_map_hzones_and_catastrophe_substrate.md
    31_tick_order_physics_conflict_resolution_and_death.md
    32_respawn_reproduction_mutation_and_bloodline_dynamics.md
  04_perception_brains_and_learning/
    40_observation_schema_rays_and_feature_construction.md
    41_bloodline_brain_architecture_and_inference_paths.md
    42_uid_owned_ppo_rollouts_rewards_and_updates.md
  05_artifacts_validation_and_viewer/
    50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md
    51_checkpointing_atomic_publish_resume_and_schema_safety.md
    52_validation_determinism_resume_consistency_and_soak_methods.md
    53_viewer_controls_inspection_and_diagnostics.md
  06_boundaries_and_appendices/
    60_implemented_behavior_vs_adjacent_theory.md
    61_background_math_python_pytorch_and_rl_appendix.md
    62_equations_shapes_and_contract_reference.md
    63_contributor_documentation_truth_contract.md
  98_authoring_inputs/
    documentation_bundle_master_plan*.md
    prompt_A*.md
  99_assets/
    diagrams/
    figure_sources/
    tables/
```

## Document Families

### 00 Meta

These documents define navigation, terminology, reference conventions, and the documentation-truth posture for the bundle.

- [D00](./00_documentation_bundle_index.md): bundle entry point and structure
- [D01](./01_reading_tracks_and_dependency_map.md): recommended reading paths and dependency order
- [D02](./02_notation_glossary_and_shape_legend.md): shared vocabulary, status labels, and shape notation
- [D03](./03_figure_artifact_and_source_reference_index.md): figure, table, artifact, and code-reference conventions
- [D99](./99_documentation_audit_change_log.md): audit log for the present repair pass

### 01 Operations and config

These documents explain how to launch, inspect, and configure the system without overstating support for dormant or compatibility-only surfaces.

- [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md)
- [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md)
- [D12](../01_operations_and_config/12_experiment_recipes_and_safe_knob_sets.md)

### 02 System foundations

These documents describe package ownership, runtime construction, tensor-oriented implementation context, and the slot-versus-UID identity contract.

- [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md)
- [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
- [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md)

### 03 World and simulation

These documents cover the world substrate, step order, physics and death handling, and reproduction or respawn semantics.

- [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
- [D31](../03_world_and_simulation/31_tick_order_physics_conflict_resolution_and_death.md)
- [D32](../03_world_and_simulation/32_respawn_reproduction_mutation_and_bloodline_dynamics.md)

### 04 Perception, brains, and learning

These documents define the observation contract, brain-family architecture, and UID-owned PPO implementation.

- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- [D41](../04_perception_brains_and_learning/41_bloodline_brain_architecture_and_inference_paths.md)
- [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)

### 05 Artifacts, validation, and viewer

These documents describe generated run artifacts, checkpoint publication and restore behavior, validation harnesses, and the interactive viewer.

- [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md)
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
- [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md)
- [D53](../05_artifacts_validation_and_viewer/53_viewer_controls_inspection_and_diagnostics.md)

### 06 Boundaries and appendices

These documents separate implemented behavior from background theory, provide mathematical and shape references, and govern future documentation changes.

- [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md)
- [D61](../06_boundaries_and_appendices/61_background_math_python_pytorch_and_rl_appendix.md)
- [D62](../06_boundaries_and_appendices/62_equations_shapes_and_contract_reference.md)
- [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md)

## Status Labels Used Throughout the Bundle

The following labels are used consistently across the bundle.

| Label | Meaning |
|---|---|
| Active | The repository shows a current read path or behavior path for the surface. |
| Guarded | The surface is real, but the accepted values, call paths, or semantics are narrower than the public name alone suggests. |
| Compatibility surface | The surface exists to preserve imports, entry points, or file names, but the canonical implementation lives elsewhere. |
| Currently unread | The configuration or compatibility surface is present, but the current repository does not show a live runtime read path. |
| Adjacent background | Material included for orientation or theory only; it does not claim current implementation. |

These labels are defined more fully in [D02](./02_notation_glossary_and_shape_legend.md) and applied in [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md) and [D60](../06_boundaries_and_appendices/60_implemented_behavior_vs_adjacent_theory.md).

## Publication Notes

- `98_authoring_inputs/` is retained for traceability, but it is not part of the public technical narrative.
- `99_assets/` currently provides reserved locations for diagrams, source tables, and figure inputs. The audit did not find a checked-in figure corpus that should be treated as present publication evidence.
- The bundle intentionally prefers direct code and test references over speculative architecture prose.

## Recommended Starting Points

- Operators: start with [D10](../01_operations_and_config/10_operator_runbook_and_game_manual.md), then [D11](../01_operations_and_config/11_config_reference_active_guarded_dead.md).
- Maintainers: start with [D20](../02_system_foundations/20_project_identity_runtime_boot_and_package_map.md), [D22](../02_system_foundations/22_state_identity_lineage_and_ownership_contracts.md), and [D63](../06_boundaries_and_appendices/63_contributor_documentation_truth_contract.md).
- Reproducibility reviewers: start with [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md), [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md), and [D52](../05_artifacts_validation_and_viewer/52_validation_determinism_resume_consistency_and_soak_methods.md).
