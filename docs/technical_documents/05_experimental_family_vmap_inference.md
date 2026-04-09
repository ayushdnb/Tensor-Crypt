# Experimental Same-Family Inference Fast Path

## Purpose
This mode reduces Python dispatch overhead during action selection by batching
alive agents that share the same bloodline family and topology through
`torch.func.stack_module_state + functional_call + vmap`.

## Safety posture
- experimental
- opt-in
- inference-only
- same-family only
- threshold-gated
- no checkpoint schema change
- no optimizer ownership change
- canonical per-brain loop remains available at all times

## Config knobs
- `cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE`
- `cfg.SIM.EXPERIMENTAL_FAMILY_VMAP_MIN_BUCKET`

## When to use
Prefer this mode only when:
- the run is headless
- many alive agents share one family
- you have benchmarked the mode on your actual hardware

## When not to use
Prefer the default path when:
- family buckets are small
- the run is viewer-bound
- you are debugging inference correctness
- your local PyTorch build shows regressions with `torch.func`

## Measurement guidance
Use `scripts/benchmark_runtime.py` with the same seed and workload twice:
1. default path
2. experimental path enabled

Compare:
- ticks/sec
- inference path stats
- CPU/GPU memory
- determinism / parity / soak results

## Rollback trigger
Disable immediately if any of the following occur:
- parity mismatch
- determinism mismatch
- resume inconsistency
- lower throughput on target workloads
- unexplained crashes inside `torch.func` execution

