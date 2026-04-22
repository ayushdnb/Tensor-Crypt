# Module Reference Index

> Scope: Provide a lookup-first map from module names to responsibilities so readers can move quickly from code to the correct narrative chapter.

## Who this document is for
Maintainers, auditors, and readers tracing code references into documentation.

## What this document covers
- main canonical modules
- wrapper surfaces
- responsibility map
- where to read next for each module family

## What this document does not cover
- full deep dives already covered elsewhere

## Prerequisite reading
- [Package layout and wrappers](../02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md)

## 1. Canonical module map

| Module or region | Responsibility | Canonical? | Narrative anchor |
| --- | --- | --- | --- |
| `tensor_crypt.app.launch` | root launch orchestration | yes | `docs/02_system/02_runtime_assembly_launch_sequence_and_session_graph.md` |
| `tensor_crypt.app.runtime` | runtime validation and subsystem assembly | yes | `docs/02_system/02_runtime_assembly_launch_sequence_and_session_graph.md` |
| `tensor_crypt.agents.brain` | bloodline-aware policy/value architecture | yes | `docs/03_mechanics/03_traits_bloodlines_families_and_brain_instantiation.md` |
| `tensor_crypt.agents.state_registry` | slot storage and UID lifecycle substrate | yes | `docs/03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md` |
| `tensor_crypt.population.reproduction` | birth mutation and placement helpers | yes | `docs/03_mechanics/07_reproduction_respawn_mutation_and_lineage.md` |
| `tensor_crypt.population.respawn_controller` | runtime reproduction controller and overlay doctrines | yes | `docs/03_mechanics/07_reproduction_respawn_mutation_and_lineage.md` |
| `tensor_crypt.simulation.engine` | tick integration and reward path | yes | `docs/03_mechanics/05_action_surface_intents_and_move_resolution.md` + `docs/04_learning/00_learning_system_overview_and_data_ownership.md` |
| `tensor_crypt.world.spatial_grid` | world substrate tensor | yes | `docs/03_mechanics/00_grid_substrate_and_spatial_field_model.md` |
| `tensor_crypt.world.procedural_map` | wall and H-zone generation | yes | `docs/03_mechanics/01_procedural_map_generation_walls_and_zones.md` |
| `tensor_crypt.world.observation_schema` | canonical/legacy observation bundle helpers | yes | `docs/03_mechanics/04_observation_schema_perception_and_ray_semantics.md` |
| `tensor_crypt.world.perception` | batched ray casting and observation construction | yes | `docs/03_mechanics/04_observation_schema_perception_and_ray_semantics.md` |
| `tensor_crypt.world.physics` | movement, collisions, environment effects, death staging | yes | `docs/03_mechanics/06_physics_collisions_damage_healing_and_death.md` |
| `tensor_crypt.learning.ppo` | UID-owned PPO state and updates | yes | `docs/04_learning/01_ppo_buffers_bootstrap_and_update_cadence.md` |
| `tensor_crypt.checkpointing.atomic_checkpoint` | manifest and latest-pointer publication | yes | `docs/05_operations/03_checkpointing_manifests_restore_and_latest_pointer.md` |
| `tensor_crypt.checkpointing.runtime_checkpoint` | capture, validate, save, load, restore | yes | `docs/04_learning/04_checkpoint_visible_learning_state_and_restore_order.md` |
| `tensor_crypt.telemetry.data_logger` | run ledgers and exports | yes | `docs/05_operations/04_telemetry_ledgers_snapshots_exports_and_lineage_graph.md` |
| `tensor_crypt.telemetry.run_paths` | run-directory and metadata creation | yes | `docs/05_operations/02_run_directory_artifacts_and_file_outputs.md` |
| `tensor_crypt.audit.final_validation` | validation probe suite | yes | `docs/05_operations/05_validation_determinism_resume_consistency_and_soak.md` |
| `tensor_crypt.viewer.*` | UI loop, panels, colors, controls | yes | `docs/05_operations/01_viewer_ui_controls_hud_and_inspector_manual.md` |
| `run.py` | root-level start surface | public root surface | `docs/02_system/00_repository_identity_entry_surfaces_and_public_contract.md` |
| `main.py` | compatibility entrypoint | wrapper / root surface | `docs/02_system/00_repository_identity_entry_surfaces_and_public_contract.md` |
| `config.py` | config wrapper | wrapper / root surface | `docs/02_system/00_repository_identity_entry_surfaces_and_public_contract.md` |

## 2. Test and script surfaces

The repository also contains scripts and tests for:
- benchmark runtime probing
- long-form soak checks
- checkpoint publication behavior
- catastrophe scheduler correctness
- observation and brain invariants
- viewer dependency and UI behaviors

These should be read as validation and operations support surfaces rather than as canonical simulation owners.
## Read next
- [FAQ and reader paths](03_faq_and_reader_paths.md)

## Related reference
- [Documentation index and reading guide](../00_program/00_documentation_index_and_reading_guide.md)
- [Validation harnesses](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## If debugging this, inspect…
- [Package and compatibility surface map](../assets/diagrams/system/package_and_compatibility_surface_map.md)

## Terms introduced here
- `module reference`
- `responsibility map`
- `root surface`
- `canonical region`
