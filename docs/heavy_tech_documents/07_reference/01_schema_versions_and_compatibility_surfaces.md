# Schema Versions and Compatibility Surfaces

> Scope: Summarize the explicit schema-version surfaces, wrapper layers, compatibility-sensitive contracts, and the main 'do not silently change' boundaries enforced by the current repository.

## Who this document is for
Maintainers and auditors concerned with migration safety, checkpoint compatibility, and public import continuity.

## What this document covers
- schema versions
- compatibility wrappers
- guarded enum surfaces
- checkpoint and observation compatibility boundaries
- do-not-silently-change surfaces

## What this document does not cover
- full subsystem explanation

## Prerequisite reading
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)
- [Package layout and wrappers](../02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md)

## 1. Explicit schema-version surfaces

The repository exposes schema-version fields for:
- identity
- observation
- PPO state
- checkpoint
- reproduction
- catastrophe
- telemetry
- logging

These versions are carried in runtime metadata and checkpoint bundles.

## 2. Compatibility surface ledger

| Surface | Compatibility role |
| --- | --- |
| `config.py` | public compatibility wrapper over canonical runtime config |
| `main.py` | root-level compatibility entrypoint |
| `engine.*` legacy re-exports | old import paths preserved over canonical modules |
| legacy observation adapter | allows old observation keys to be adapted into the canonical contract |
| manifest and latest pointer | compatibility-sensitive checkpoint file-set surface |

## 3. Guarded enum and mode surfaces

The runtime validation layer visibly constrains several string selectors. Examples include:
- respawn mode
- anchor parent selector
- extinction policy
- catastrophe modes
- static catastrophe ordering policy
- PPO ownership mode
- overlap and cooldown policies

Treat these as **validated public surfaces**, not as informal comments.

## 4. Do-not-silently-change list

The following boundaries should not change silently:
- canonical observation feature counts and ordering
- family order and family topology signatures
- action dimension and value dimension
- UID/slot restore ordering expectations
- checkpoint schema fields and manifest logic
- latest-pointer semantics
- lineage parent-role meaning
## Read next
- [Module reference index](02_module_reference_index.md)
- [Repository truth gaps and explicit unknowns](97_repository_truth_gaps_and_explicit_unknowns.md)

## Related reference
- [Checkpoint-visible learning state and restore order](../04_learning/04_checkpoint_visible_learning_state_and_restore_order.md)
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)

## If debugging this, inspect…
- [Module reference index](02_module_reference_index.md)

## Terms introduced here
- `schema version`
- `compatibility surface`
- `do-not-silently-change`
