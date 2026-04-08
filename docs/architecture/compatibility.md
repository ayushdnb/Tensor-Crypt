# Compatibility Notes

## Preserved Import Surfaces

- `tensor_crypt.*` remains valid from repository root via `tensor_crypt/__init__.py`.
- Legacy `engine.*` imports remain valid from a source checkout via the root package stub plus `src/engine`.
- Legacy `viewer.*` imports remain valid from a source checkout via the root package stub plus `src/viewer`.
- Package builds install `engine` and `viewer` directly from `src/engine` and `src/viewer`.

## Preserved Launch Surfaces

- `python run.py`
- `python main.py`
- `tensor-crypt`

## Preserved Config Surface

- Root `config.py` remains the canonical configuration source.
- Internal modules continue to consume config through `tensor_crypt.config_bridge`.
- Package builds include the root `config` module so the bridge works outside a source checkout.

## Scope of Compatibility

- Compatibility is guaranteed for source-tree execution, package builds, and repository test contracts.
- Wrapper modules remain intentionally thin and contain no business logic.
