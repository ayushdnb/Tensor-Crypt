# Package Layout, Canonical Modules, and Compatibility Wrappers

> Scope: Map the repository’s implementation modules, clarify which surfaces own behavior, and isolate legacy or compatibility layers so they are not misread as peers of canonical code.

## Who this document is for
Maintainers, architecture readers, and anyone tracing imports from code into the docs.

## What this document covers
- canonical package layout by responsibility
- root-level compatibility surfaces
- legacy re-export surfaces
- which modules own which responsibilities

## What this document does not cover
- detailed algorithmic behavior inside every module
- full configuration reference tables

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Repository identity and public contract](00_repository_identity_entry_surfaces_and_public_contract.md)

## 1. Canonical layout overview

The code dump exposes a stable responsibility pattern:

| Area | Canonical package region | Primary responsibility |
| --- | --- | --- |
| App | `tensor_crypt.app.*` | launch-time assembly and runtime construction |
| Agents | `tensor_crypt.agents.*` | brains, registry, state ownership |
| Population | `tensor_crypt.population.*` | evolution, reproduction, respawn control |
| Simulation | `tensor_crypt.simulation.*` | engine tick loop and reward path |
| World | `tensor_crypt.world.*` | grid, map generation, perception, physics, observation schema |
| Learning | `tensor_crypt.learning.*` | PPO buffers, optimizers, update logic |
| Checkpointing | `tensor_crypt.checkpointing.*` | atomic publish, manifest, latest pointer, capture/restore |
| Telemetry | `tensor_crypt.telemetry.*` | run paths, loggers, lineage export |
| Viewer | `tensor_crypt.viewer.*` | UI loop, rendering, controls, panels |
| Validation | `tensor_crypt.validation.*` | determinism, resume, catastrophe, save-load-save probes |

## 2. Wrapper and bridge surfaces

The dump also shows several non-canonical surfaces:

| Surface | Status | What it does |
| --- | --- | --- |
| `config.py` | public compatibility wrapper | re-exports canonical runtime config |
| `main.py` | repository-root compatibility entrypoint | calls the canonical app launch |
| `run.py` | canonical root-level start surface | root launch script for users |
| `engine.*` modules | legacy compatibility wrappers | thin re-exports of canonical implementation modules |

> **Common confusion**
> A thin re-export is part of the import surface, not proof of duplicate ownership.

## 3. Module ownership map

- `tensor_crypt.app.launch` owns startup orchestration, not simulation rules.
- `tensor_crypt.app.runtime` owns subsystem assembly, launch-order validation, and runtime configuration checks.
- `tensor_crypt.agents.brain` owns the bloodline-aware policy/value architecture and canonical/legacy observation adaptation at the brain interface.
- `tensor_crypt.agents.state_registry` owns slot storage and UID lifecycle mapping.
- `tensor_crypt.simulation.engine` owns per-tick integration across catastrophe scheduling, observation, action sampling, physics, rewards, deaths, respawn, PPO update, snapshots, checkpoints, and tick logging.
- `tensor_crypt.learning.ppo` owns UID-keyed rollout buffers, training counters, optimizers, and PPO update math.
- `tensor_crypt.checkpointing.runtime_checkpoint` owns capture, validation, restore, and load/save entry points.
- `tensor_crypt.checkpointing.atomic_checkpoint` owns manifest and latest-pointer publication.
- `tensor_crypt.validation.final_validation` owns determinism and resume probes.

## 4. Why the separation matters

This separation prevents misleading documentation:
- launch code should not be confused with engine code
- telemetry should not be confused with checkpoint ownership
- compatibility imports should not be mistaken for canonical modules
- slot-backed registry state should not be confused with UID-owned learning state

## 5. Asset map

See the dedicated system diagram:
- [Package and compatibility surface map](../assets/diagrams/system/package_and_compatibility_surface_map.md)
## Read next
- [Runtime assembly, launch sequence, and session graph](02_runtime_assembly_launch_sequence_and_session_graph.md)
- [Module reference index](../07_reference/02_module_reference_index.md)

## Related reference
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## If debugging this, inspect…
- [Runtime config taxonomy and knob safety](03_runtime_config_taxonomy_and_knob_safety.md)
- [Operator quickstart and common run modes](../05_operations/00_operator_quickstart_and_common_run_modes.md)

## Terms introduced here
- `canonical module`
- `wrapper surface`
- `compatibility import`
- `ownership map`
