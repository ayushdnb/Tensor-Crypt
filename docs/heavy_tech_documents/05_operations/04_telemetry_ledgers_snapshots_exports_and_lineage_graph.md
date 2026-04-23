# Telemetry Ledgers, Snapshots, Exports, and Lineage Graph

> Scope: Explain the durable logging surfaces produced by the data logger and how they relate to the canonical UID and parent-role substrate.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- ledger families
- lineage export
- birth/death/life interpretation
- PPO event logging
- catastrophe logging
- schema meaning

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Run directory artifacts](02_run_directory_artifacts_and_file_outputs.md)
- [Reproduction, mutation, and lineage](../03_mechanics/07_reproduction_respawn_mutation_and_lineage.md)

## 1. Telemetry philosophy

The logger’s own docstring states that telemetry should record durable ledgers without changing simulation semantics. This matters because the docs can treat telemetry as observability infrastructure rather than as behavior owner.

## 2. Identity-safe telemetry

The logger’s deepest surfaces are UID-oriented:
- birth rows identify child UID and parent-role UIDs
- life ledgers track UID-bounded life records
- lineage export derives graph nodes and edges from canonical UID and parent-role state

## 3. Ledger families

| Ledger | Main purpose |
| --- | --- |
| birth | record new child identity and birth context |
| genealogy | backward-compatible alias surface |
| life | record open and finalized life summaries |
| death | record death context, family, position, and schema version |
| collisions | record collision events |
| PPO events | record update stats |
| tick summary | periodic operational summary |
| family summary | family-level aggregation |
| catastrophes | catastrophe event log |

## 4. Snapshots and brains

Separate snapshot-style outputs record:
- agent snapshots
- heatmaps
- saved brains

These are periodic artifacts rather than event-by-event ledgers.

## 5. Lineage graph

The lineage JSON export serializes:
- nodes with UID, family, birth/death ticks, lineage depth, and parent-role links
- edges typed by role such as brain parent, trait parent, and anchor parent

That makes the lineage graph a faithful canonical-identity export rather than a slot graph.


## Read next
- [Validation, determinism, resume consistency, and soak](05_validation_determinism_resume_consistency_and_soak.md)
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)

## Related reference
- [Lineage and ledger notes in the game manual](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md#lineage-and-bloodline-reading-notes)

## If debugging this, inspect…
- [Module reference index](../07_reference/02_module_reference_index.md)

## Terms introduced here
- `deep ledger`
- `life ledger`
- `birth ledger`
- `lineage graph`
- `UID-safe telemetry`
