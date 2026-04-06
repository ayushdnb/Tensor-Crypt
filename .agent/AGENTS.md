# Agent Operating Notes

## Mission
Perform a full repository hierarchy restructuring for Tensor Crypt while preserving runtime semantics, training semantics, checkpoint semantics, telemetry semantics, viewer controls, and public entrypoint behavior.

## Hard Invariants
- `config.py` remains the canonical user knob surface.
- `run.py` and `main.py` remain public launch entrypoints.
- No intentional algorithm/logic/math changes across simulation, PPO, physics, perception, reproduction, catastrophes, checkpointing, telemetry, or viewer behavior.
- Preserve legacy import surfaces (`engine.*`, `viewer.*`) and `tensor_crypt.*` imports.
- Preserve checkpoint and telemetry schemas/field contracts unless explicit compatibility bridges are added and validated.

## Execution Discipline
- Plan-first migration: docs and mapping before moves.
- Move code structurally, not behaviorally.
- Keep compatibility wrappers explicit and minimal.
- Validate each migration gate before continuing.
- Record all validation commands and outcomes in `docs/restructure_validation_report.md`.

## Current Branch
- `refactor/repository-restructure-semantic-preservation`
