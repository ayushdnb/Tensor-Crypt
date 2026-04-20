# Documentation

## Release Identity

Tensor Crypt Single-Brain Vmap is the self-centric single-family release line. It supports one active branch family at startup, experimental self-centric observations, UID-owned PPO, binary-parented reproduction, catastrophe controls, telemetry ledgers, runtime checkpoints, and the guarded `torch.func` vmap path.

## Operator Defaults

- Entry points: `python run.py`, `python main.py`, `tensor-crypt`
- Startup preset: `apply_experimental_single_family_launch_defaults()`
- Active family: `BRAIN.EXPERIMENTAL_BRANCH_FAMILY`, defaulting to `House Nocthar`
- Observation contract: `experimental_selfcentric_v1`
- Vmap: enabled by default and rejected if `torch.func` is unavailable

## Validation Summary

- Targeted release suite: 58 passed.
- Full pytest: 193 passed.
- Benchmark smoke: passed; final tick 18, final alive 4, vmap slots 64.
- Benchmark record: `artifacts/release/single_brain_vmap_benchmark_smoke.json`.

## Known Limits

- This is a `0.9.0` research-preview line, not the stable five-brain line.
- The config uses `EXPERIMENTAL_*` field names for compatibility with checkpoint-visible contracts.
- Vmap enablement is not proof that every tick uses vmap; bucket size and topology eligibility still decide execution path.

## Release Notes

### 0.9.0 Research Preview

- Establishes the self-centric single-family line on top of the hardened runtime base.
- Makes public entrypoints and headless scripts apply the branch preset.
- Refreshes branch-facing README and architecture documentation.
- Validated with full pytest and a headless benchmark smoke that exercised vmap.
