# D20 - Project Identity, Runtime Boot, and Package Map

## Purpose

This document identifies the canonical implementation surface of Tensor Crypt, the runtime boot path, and the module families that own the major responsibilities of the repository.

## Scope Boundary

This chapter establishes package ownership and boot order. It does not restate detailed world mechanics, observation semantics, PPO math, checkpoint schemas, or viewer diagnostics. Those topics belong to later chapters.

## Evidence Basis

This chapter is grounded in:

- `run.py`
- `main.py`
- `config.py`
- `tensor_crypt.app.launch`
- `tensor_crypt.app.runtime`
- `tensor_crypt.runtime_config`
- `tensor_crypt.config_bridge`
- compatibility packages `engine/` and `viewer/`
- `pyproject.toml`
- `requirements.txt`

## What the Repository Is

Tensor Crypt is a Python simulation repository whose canonical implementation lives under the `tensor_crypt` package. The runtime assembles a tensor-backed world, launches a Pygame-based viewer, persists telemetry artifacts, and optionally publishes runtime checkpoints.

The root of the repository exposes public entry and compatibility surfaces, but it is not where the main implementation logic lives.

## Canonical and Compatibility Surfaces

### Canonical implementation owners

The current implementation owners are under `tensor_crypt.*`, including:

- `tensor_crypt.app`
- `tensor_crypt.agents`
- `tensor_crypt.world`
- `tensor_crypt.population`
- `tensor_crypt.learning`
- `tensor_crypt.simulation`
- `tensor_crypt.telemetry`
- `tensor_crypt.checkpointing`
- `tensor_crypt.audit`
- `tensor_crypt.viewer`

### Compatibility and entry surfaces

The repository also exposes:

- `run.py`
- `main.py`
- `config.py`
- `engine/*`
- `viewer/*`

These are compatibility or convenience surfaces. They preserve root-level launches and legacy imports, but they do not supersede canonical ownership under `tensor_crypt.*`.

## Runtime Boot Path

The current launch path is:

1. root entry file `run.py` or `main.py`
2. `tensor_crypt.app.launch.main`
3. `tensor_crypt.app.runtime.setup_determinism`
4. `tensor_crypt.telemetry.run_paths.create_run_directory`
5. `tensor_crypt.app.runtime.build_runtime`
6. `SimulationRuntime.viewer.run()`

## Runtime Assembly Order

`tensor_crypt.app.runtime.build_runtime` currently assembles the session in the following order:

1. `DataLogger`
2. `Grid`
3. `Registry`
4. `Physics`
5. `Perception`
6. `PPO`
7. `Evolution`
8. procedural walls
9. procedural H-zones
10. wall-cache refresh
11. initial population spawn
12. `Engine`
13. `Viewer`

This order is semantically relevant because world generation precedes spawn, and spawn precedes engine and viewer construction.

## `SimulationRuntime` as the Assembled Session Graph

The build path returns a `SimulationRuntime` dataclass that carries the assembled runtime graph, including:

- run directory
- data logger
- grid
- registry
- physics
- perception
- PPO module
- evolution module
- engine
- viewer

This makes the session boundary explicit rather than scattering ownership across unrelated globals.

## Responsibility Map

| Surface | Primary responsibility |
|---|---|
| `tensor_crypt.runtime_config` | authoritative config dataclass tree |
| `tensor_crypt.config_bridge` | shared access to the active config object |
| `tensor_crypt.app.launch` | top-level launch orchestration |
| `tensor_crypt.app.runtime` | runtime validation, determinism setup, subsystem assembly |
| `tensor_crypt.agents.state_registry` | dense slot substrate plus UID lifecycle and lineage ledgers |
| `tensor_crypt.agents.brain` | family-aware policy and value networks |
| `tensor_crypt.world.*` | grid, map generation, perception, and physics |
| `tensor_crypt.population.*` | reproduction, respawn, and evolution coordination |
| `tensor_crypt.learning.ppo` | UID-owned rollout buffers, optimizers, and update logic |
| `tensor_crypt.simulation.engine` | tick loop, inference staging, reward storage, and update scheduling |
| `tensor_crypt.telemetry.*` | run paths, logging, lineage export, and artifact writing |
| `tensor_crypt.checkpointing.*` | checkpoint capture, validation, publish, and restore |
| `tensor_crypt.audit.final_validation` | determinism and resume-oriented validation harnesses |
| `tensor_crypt.viewer.*` | presentation, layout, panels, input, and camera behavior |

## Public Runtime Environment Facts

The repository metadata currently states:

- Python requirement: `>=3.10`
- runtime viewer dependency: `pygame-ce>=2.5.6,<2.6`

The code imports the `pygame` namespace. The packaging metadata and README clarify that this is provided by `pygame-ce`, not by a separate checked-in viewer backend.

## Guarded Runtime Surfaces

Several config fields are public but deliberately constrained by runtime validation in `tensor_crypt.app.runtime`. Examples include:

- `cfg.SIM.DTYPE`
- `cfg.AGENTS.SPAWN_MODE`
- `cfg.TRAITS.METAB_FORM`
- `cfg.RESPAWN.MODE`
- `cfg.RESPAWN.ANCHOR_PARENT_SELECTOR`
- `cfg.PPO.OWNERSHIP_MODE`
- `cfg.PHYS.TIE_BREAKER`
- `cfg.TELEMETRY.LINEAGE_EXPORT_FORMAT`
- `cfg.CATASTROPHE.DEFAULT_MODE`

The presence of a public field does not imply that all plausible values are implemented.

## Immediate Architectural Consequences

Readers should carry the following forward into later chapters:

- `tensor_crypt` is the canonical code owner
- root files and compatibility packages are not alternative semantic centers
- runtime construction is centralized rather than emergent
- run-directory creation is part of boot semantics, not an optional afterthought
- validation narrows several public config surfaces to supported subsets

## Cross References

- Tensor and simulation foundations: [D21](./21_python_pytorch_tensors_and_simulation_foundations.md)
- UID, slot, lineage, and ownership contract: [D22](./22_state_identity_lineage_and_ownership_contracts.md)
- World substrate: [D30](../03_world_and_simulation/30_world_grid_map_hzones_and_catastrophe_substrate.md)
- Artifacts and checkpoints: [D50](../05_artifacts_validation_and_viewer/50_telemetry_ledgers_hdf5_parquet_and_run_artifacts.md), [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
