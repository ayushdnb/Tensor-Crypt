# Validation Strategy

## Standard
This campaign is breakage-oriented. Claims are accepted only when backed by executable evidence from this branch.

## Execution Order
1. Reconnaissance and static audit.
2. Baseline targeted validation runs.
3. Test and harness expansion around uncovered high-risk surfaces.
4. Fault injection and accelerated late-stage probes.
5. Soak and memory instrumentation.
6. Convergence/stability analysis from produced metrics.
7. Validated fixes only.

## Evidence Rules
- Prefer tests over prose when behavior can be made executable.
- Distinguish pass, fail, and partial coverage explicitly.
- Record any limitation that prevents proof.
- Avoid relying on generated or staged directories as implementation truth.

## Initial Campaign Shape
- Fast targeted pytest slices by subsystem.
- Runtime script invocations for benchmark and soak harnesses.
- Repeated checkpoint/resume probes.
- State-factory or monkeypatched fault injection where direct waiting is weak.
- Memory-growth tracking using process RSS and Python allocation snapshots when practical.

## Executed Campaign Shape
- Full repo pytest baseline and post-fix verification.
- Checkpoint-publication fault injection via monkeypatch.
- Canonical CPU benchmark probe.
- Experimental family-vmap CPU benchmark probe.
- Medium soak with periodic checkpoint validation.
- Sampled RSS window probe over 512 ticks.
- Repeated resume-chain audit over five save/load/restore cycles.
