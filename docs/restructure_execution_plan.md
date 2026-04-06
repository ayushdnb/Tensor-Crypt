# Tensor Crypt Repository Restructure Execution Plan

## Execution Status
- Plan authored and executed.
- Structural migration complete.
- Validation gates passed (see `docs/restructure_validation_report.md`).

## Scope
- Perform a full repository hierarchy restructuring for maintainability, packaging correctness, and clearer public/internal boundaries.
- Consolidate implementation into one coherent package tree.
- Preserve behavioral parity across runtime, simulation, learning, checkpointing, telemetry, viewer, and public entrypoints.
- Preserve or explicitly bridge compatibility for existing import/launch surfaces.

## Non-Goals
- No algorithm redesign.
- No PPO hyperparameter changes.
- No change to observation/action semantics.
- No change to catastrophe, reproduction, checkpoint, telemetry, or viewer semantics.
- No opportunistic feature additions.

## Behavior Invariants (Must Not Change)
- Root `config.py` remains canonical knob surface.
- Root `run.py` and `main.py` remain public launch entrypoints.
- Simulation tick order remains unchanged.
- Registry UID lifecycle semantics remain unchanged.
- PPO buffer ownership and update semantics remain unchanged.
- Checkpoint schema/version behavior and file-set behavior remain unchanged.
- Telemetry schema fields and artifact names remain unchanged.
- Viewer control and hotkey behavior remains unchanged.
- Compatibility imports (`engine.*`, `viewer.*`, `tensor_crypt.*`) remain valid.

## Current Architecture Summary (Discovery)
- Public root surfaces:
  - `config.py`
  - `run.py`
  - `main.py`
- Compatibility roots:
  - `engine/*` re-exporting `tensor_crypt.*`
  - `viewer/*` re-exporting `tensor_crypt.*`
- Real implementation package:
  - `tensor_crypt/app` runtime bootstrap/wiring
  - `tensor_crypt/simulation` engine loop/catastrophes
  - `tensor_crypt/world` grid/mapgen/perception/physics
  - `tensor_crypt/agents` brain + state registry
  - `tensor_crypt/population` reproduction/evolution/respawn
  - `tensor_crypt/learning` PPO
  - `tensor_crypt/checkpointing` runtime + atomic checkpoint helpers
  - `tensor_crypt/telemetry` run paths + data logging
  - `tensor_crypt/viewer` pygame viewer internals
  - `tensor_crypt/audit` validation harness
- Tests assert import compatibility and semantic contracts heavily under `tests/`.

## Proposed Target Architecture

```text
Tensor_Crypt/
  config.py
  run.py
  main.py
  pyproject.toml
  requirements.txt
  README.md
  engine/                  # legacy compatibility wrappers
  viewer/                  # legacy compatibility wrappers
  tensor_crypt/            # root namespace shim for src package
    __init__.py
  src/
    tensor_crypt/          # single authoritative implementation tree
      app/
      agents/
      audit/
      checkpointing/
      learning/
      population/
      simulation/
      telemetry/
      viewer/
      world/
  scripts/
  tests/
  docs/
  .agent/
```

## Migration Phases

### Phase A: Discovery and Planning (current)
- Complete inventory and coupling audit.
- Produce required planning and mapping docs.

### Phase B: Compatibility and Packaging Design
- Define import preservation strategy for `tensor_crypt.*` after `src/` migration.
- Define how root entrypoints keep behavior unchanged.
- Define packaging metadata (`pyproject.toml`) and test path handling.

### Phase C: Structural Moves
- Move implementation tree from `tensor_crypt/` to `src/tensor_crypt/`.
- Introduce root `tensor_crypt/__init__.py` namespace shim that extends package path to `src/tensor_crypt`.
- Keep `engine/*` and `viewer/*` wrappers stable.

### Phase D: Wiring Adjustments
- Update file-path-sensitive modules (notably config bridge path resolution) to remain root-config compatible after move.
- Keep all internal import semantics unchanged.

### Phase E: Documentation Alignment
- Update README to describe new hierarchy and run/development workflow.
- Add `docs/architecture_overview.md` with final tree and module boundaries.
- Preserve compatibility policy documentation.

### Phase F: Verification and Final Audit
- Run import checks, entrypoint smoke checks, simulation smoke checks, checkpoint checks, and full tests.
- Update validation report with command outputs and pass/fail status.
- Final semantic-preservation audit summary.

## Risks
- Import resolution drift after moving package into `src/`.
- Config bridge fallback path drift if it assumes old directory depth.
- Hidden file-path assumptions in scripts/tests.
- Checkpoint/telemetry behavior drift due to accidental import or module initialization changes.
- Compatibility break for external users importing directly from repository root.

## Validation Gates
- Gate 1: `tensor_crypt.*` imports resolve from repo root.
- Gate 2: legacy `engine.*` / `viewer.*` imports resolve.
- Gate 3: `python run.py` and `python main.py` launch path imports resolve.
- Gate 4: deterministic short runtime smoke executes without invariant/assertion failures.
- Gate 5: checkpoint save/load/manifest validations pass.
- Gate 6: telemetry outputs still emit required artifact files and schema fields.
- Gate 7: full `pytest` suite passes.

## Rollback and Compatibility Strategy
- Structural moves done in small commits with immediate validation.
- Preserve root entrypoints and wrapper modules.
- Preserve `tensor_crypt.*` import surface via root namespace shim.
- No deletion of compatibility wrappers unless proven unnecessary and covered by passing compatibility tests.
- If a gate fails, rollback that phase before proceeding.

## Acceptance Criteria
- Cleaner root and single authoritative implementation tree under `src/tensor_crypt`.
- Required docs completed and accurate.
- Mapping audit demonstrates no dropped responsibilities.
- Validation report shows passing compatibility/runtime/test gates.
- No intentional semantic drift.
