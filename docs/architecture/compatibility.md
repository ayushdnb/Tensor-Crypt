# Compatibility Notes

## Preserved Import Surfaces

- `tensor_crypt.*` is the canonical implementation surface from both source checkouts and installed builds.
- Legacy `engine.*` imports remain valid via thin root-level re-export modules.
- Legacy `viewer.*` imports remain valid via thin root-level re-export modules.
- Root `config.py` remains importable and shares the exact same `cfg` object as `tensor_crypt.runtime_config`.

## Preserved Launch Surfaces

- `python run.py`
- `python main.py`
- `tensor-crypt`

## Canonical Truth Locations

- Implementation code: `tensor_crypt/`
- Config singleton and dataclasses: `tensor_crypt/runtime_config.py`
- Public config wrapper: `config.py`
- Legacy import shims: `engine/` and `viewer/`

## Scope of Compatibility

- Compatibility is maintained for source-tree execution, package builds, and the public entry surfaces listed above.
- Wrapper modules remain intentionally thin and contain no business logic.
- No compatibility layer owns simulation, PPO, checkpointing, telemetry, or viewer semantics.
