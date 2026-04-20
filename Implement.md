# Implementation Log

## Mode Diagnosis

Contamination found on the hardened base:

- The hardened base had partial launch-preset tests and config helper residue, but not the complete self-centric observation implementation.
- Benchmark, soak, and resume audit scripts used the generic runtime path and did not apply the single-family preset.
- Public README and architecture docs described the generic five-family runtime instead of this branch.
- Public docs contained phase-report and authoring-input residue.

## Decisions

| Decision | Reason | Impact |
|---|---|---|
| Cherry-pick the existing self-centric implementation onto the hardened base | Reuses tested branch-specific code instead of recreating it | Preserves observation, checkpoint, registry, and vmap tests |
| Keep experimental config field names | They are already checkpoint-visible and tested | Branch is versioned as a research preview |
| Apply the preset in benchmark and audit scripts | Branch tooling should exercise the branch identity by default | Benchmark default now reports vmap enabled |
| Keep compatibility family roster | Telemetry, viewer, checkpoint, and test surfaces still use family metadata | Normal startup still activates one branch family |

## Validation Status

- Targeted release suite: 58 passed.
- Benchmark smoke: passed; final tick 18, final alive 4, `vmap_slots=64`.
- Full pytest: 193 passed.
