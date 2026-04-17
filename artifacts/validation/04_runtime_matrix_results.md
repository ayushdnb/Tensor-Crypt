# Runtime Matrix Results

| Scenario | Executed surface | Result | Key evidence |
| --- | --- | --- | --- |
| Full test suite | `python -m pytest -q` | Pass | `180 passed in 54.27s` |
| Checkpoint + benchmark targeted slice | `python -m pytest tests\test_checkpoint_atomicity.py tests\test_benchmark_and_scheduler.py -q` | Pass | `9 passed in 9.29s` |
| Canonical CPU benchmark | `scripts/benchmark_runtime.py` | Pass | `256` measured ticks, `7.10` ticks/sec, RSS delta `-19.11 MB` |
| Experimental family-vmap CPU benchmark | `scripts/benchmark_runtime.py --experimental-family-vmap-inference --experimental-family-vmap-min-bucket 2` | Pass | `2011` vmap slots, `37` loop slots, `8.16` ticks/sec |
| Repeated resume-chain audit | `scripts/run_resume_chain_audit.py` | Pass | `5` cycles x `8` ticks, all cycle signatures matched |
| Medium soak with checkpoint validation | `scripts/run_soak_audit.py` | Pass | `768` ticks, checkpoint validation every `64` ticks, no invariant/non-finite failures |

## Notes
- The executed mode evidence is CPU-only.
- Canonical and experimental family-vmap inference paths were both exercised.
- The root launch preset path was validated in unit tests rather than an end-to-end viewer subprocess, to avoid interactive viewer blocking.
