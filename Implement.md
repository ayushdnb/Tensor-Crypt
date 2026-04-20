# Implementation Log

## Mode Diagnosis

Contamination found on the hardened base:

- `tests/test_imports_and_compat.py` expected root launch to apply a single-family experimental preset.
- `tensor_crypt/runtime_config.py` exposed `apply_experimental_single_family_launch_defaults()` even though this branch should launch the five-family runtime.
- Public README and architecture docs described the runtime as general Tensor Crypt rather than a branch-specific five-brain release.
- Public docs contained phase-report residue in `docs/technical_documents/99_assets/telemetry_spawn_schema_hardening_phase_report.md`.

## Decisions

| Decision | Reason | Impact |
|---|---|---|
| Remove the single-family launch preset helper | The five-brain branch must not expose a startup path that mutates config into another product line | Launch tests now assert five-family defaults |
| Keep optional family-vmap disabled by default | It is already guarded, benchmark-only, and not a startup mode | Benchmark tests remain useful without changing branch identity |
| Keep root and legacy wrappers | They are documented public compatibility surfaces | No implementation logic added there |

## Validation Status

- Targeted release suite: 42 passed.
- Benchmark smoke: passed; final tick 18, final alive 4, `experimental_family_vmap_inference=false`.
- Full pytest: 180 passed.
