# Compatibility Notes

## Preserved Import Surfaces
- `tensor_crypt.*` remains valid from repository root via `tensor_crypt/__init__.py` namespace shim.
- Legacy `engine.*` imports remain valid via root compatibility wrappers.
- Legacy `viewer.*` imports remain valid via root compatibility wrappers.

## Preserved Launch Surfaces
- `python run.py`
- `python main.py`

## Preserved Config Surface
- Root `config.py` remains the canonical configuration source.
- Internal modules continue to consume config through `tensor_crypt.config_bridge`.

## Scope of Compatibility
- Compatibility is guaranteed for source-tree execution and repository test contracts.
- Wrapper modules remain intentionally thin and contain no business logic.
