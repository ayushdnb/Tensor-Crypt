# Repository Identity, Entry Surfaces, and Public Contract

> Scope: Explain what the repository is, what the public launch surfaces are, and which code paths are canonical versus compatibility-facing.

## Who this document is for
Technical readers, maintainers, and operators who need a correct top-level model before diving into mechanics.

## What this document covers
- the repository’s architectural center of gravity
- repository-root entry surfaces and what they do
- the difference between launch convenience and canonical ownership
- the public contract expressed by the checked-in entry surfaces and package boundaries

## What this document does not cover
- full runtime assembly order
- deep observation or PPO mechanics

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Foundations primer](../01_foundations/08_grid_world_simulation_engines_and_state_ownership_primer.md)

## 1. What the repository is

Tensor Crypt is a simulation-and-learning codebase built around:
- a canonical `tensor_crypt` implementation package
- repository-root entry surfaces such as `run.py`, `main.py`, and `config.py`
- legacy compatibility wrappers for older import paths
- a tensor-backed world grid and slot-backed live runtime storage
- UID-owned identity, lineage, and PPO training state
- checkpoint manifests and latest-pointer publication
- an interactive pygame-ce viewer
- validation, benchmark, and soak harnesses

This is therefore best understood as a repository with **application**, **simulation**, **learning**, **telemetry**, **checkpointing**, **viewer**, and **validation** layers rather than as a bare training script.

## 2. Public launch and import surfaces

The public repository surface exposes:
- `run.py` as the canonical root-level start surface for repository users
- `main.py` as a repository-root compatibility entrypoint that calls `tensor_crypt.app.launch.main`
- `config.py` as a public compatibility wrapper over the canonical runtime configuration
- legacy `engine.*` compatibility modules that thinly re-export canonical `tensor_crypt` surfaces

> **Invariant**
> Repository-root convenience does not make a module the implementation owner. Ownership remains in the canonical package unless the code says otherwise.

## 3. What the public contract means here

A practical public contract in this repository includes:
- users can launch from the root-level entry surface
- legacy imports continue to resolve through thin compatibility wrappers
- the canonical implementation remains under `tensor_crypt`
- subsystem logic should be read in the canonical package, not in wrappers

## 4. Architecture center of gravity

The package docstring states plainly that the canonical implementation lives under `tensor_crypt`. That sentence matters because it prevents several documentation errors:
- treating root-level scripts as subsystem owners
- narrating compatibility wrappers as if they were the real modules
- flattening the architecture into a vague “repo files” story

## 5. Consequences for readers

### For operators
The root-level entry surface is the operational front door.

### For maintainers
The canonical package is the implementation truth.

### For auditors
Compatibility wrappers must be documented as wrappers, not as alternate canonical implementations.

### For documentation
Every deep-dive chapter should point into `tensor_crypt` modules when describing ownership.
## Read next
- [Package layout, canonical modules, and compatibility wrappers](01_package_layout_canonical_modules_and_compatibility_wrappers.md)
- [Runtime assembly, launch sequence, and session graph](02_runtime_assembly_launch_sequence_and_session_graph.md)

## Related reference
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)
- [Module reference index](../07_reference/02_module_reference_index.md)

## If debugging this, inspect…
- [Operator quickstart and common run modes](../05_operations/00_operator_quickstart_and_common_run_modes.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Terms introduced here
- `canonical package`
- `public entry surface`
- `compatibility wrapper`
- `public contract`
