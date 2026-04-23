# Grid-World Simulation Engines and State Ownership Primer

> Scope: Explain the simulation-engine concepts needed to understand Tensor Crypt’s dense world grid, deterministic tick loop, slot storage, reversible runtime modifiers, and audit-heavy state ownership model.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- tick loops
- world state and agent state
- update ordering
- determinism
- reversible modifiers
- identity versus storage location

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. A simulation engine is an ordered state transformer

A tick-based engine updates world state in a fixed order. That order matters because the result of one subsystem can change the legal inputs to the next subsystem.

## 2. State is not one thing

A serious simulation contains several kinds of state:
- world substrate state
- agent storage state
- identity and lineage state
- learning state
- checkpoint metadata
- UI state
- runtime modifiers that must revert cleanly

Tensor Crypt exposes all of those categories.

## 3. Determinism depends on order

If two runs use the same seed but call subsystems in a different order, they may diverge. This is why runtime assembly, catastrophe scheduling, physics, death processing, respawn, PPO updates, snapshotting, and checkpoint publication are all documented with ordering in mind.

## 4. Reversible runtime modifiers

Some runtime effects, such as catastrophe-driven vision scaling or metabolism scaling, should affect current behavior without mutating the canonical inherited trait substrate. That distinction makes restore logic and telemetry interpretation more reliable.

## 5. Identity versus location

A slot is a storage location. A UID is a persistent conceptual identity. Good simulation systems keep those separate when slot reuse would otherwise blur ownership.


## Why this matters for Tensor Crypt
This primer explains why Tensor Crypt treats runtime assembly order, UID ownership, checkpoint restore order, and reversible catastrophe modifiers as architectural issues rather than as implementation trivia.

## Read next
- [Repository identity, entry surfaces, and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)
- [Runtime assembly, launch sequence, and session graph](../02_system/02_runtime_assembly_launch_sequence_and_session_graph.md)
- [Agent registry, UID ownership, and lifecycle](../03_mechanics/02_agent_registry_uid_ownership_and_lifecycle.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
