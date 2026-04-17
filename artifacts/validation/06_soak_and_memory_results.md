# Soak And Memory Results

## Medium Soak
- Command: `python scripts\run_soak_audit.py --ticks 768 --seed 42 --width 20 --height 20 --agents 12 --walls 4 --hzones 2 --log-dir artifacts\validation\logs\soak_seed42 --checkpoint-validate-every 64`
- Result: `SOAK_OK ticks=768 alive_final=12 min_alive=12 max_alive=12`
- Checkpoint validation cadence: every `64` ticks, yielding `12` periodic checkpoint validations across the run.
- Observed failures: none. The harness would have raised on registry/grid non-finite values, unknown PPO UID ownership, or non-finite brain parameters.

## RSS Window Probe
- Artifact: `artifacts/validation/metrics/memory_window_probe_seed77.json`
- Window: `512` ticks sampled every `64` ticks.
- RSS at first sample (`tick 64`): `752.66 MB`
- RSS peak: `827.80 MB` at `tick 384`
- RSS at final sample (`tick 512`): `817.25 MB`
- Net delta first-to-last sample: `+64.58 MB`
- Max buffered parquet rows observed: `395`

## Interpretation
- The sampled window shows allocator/backlog growth that later partially recedes; it is not evidence of catastrophic runaway growth, but it is also not evidence that leaks are absent.
- The soak sample maintained a stable alive population under the tested ceiling/floor settings, but that population stability is configuration-driven and should not be misread as a convergence proof.
