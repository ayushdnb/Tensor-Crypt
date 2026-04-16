# Benchmarking and Performance Probe Manual

> Scope: Explain the headless benchmark script, what it measures, what its counters mean, and how not to over-interpret the results.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- benchmark script inputs
- elapsed time and ticks/sec
- memory counters
- inference-path stats
- profiling surface
- interpretation limits

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Operator quickstart](00_operator_quickstart_and_common_run_modes.md)
- [Inference execution paths](../04_learning/03_inference_execution_paths_loop_vs_family_vmap.md)

## 1. Benchmark purpose

The benchmark harness is a reproducible headless probe. It is useful for comparing controlled runtime conditions, not for proving broad performance superiority.

## 2. Inputs visible in the script

The benchmark script accepts knobs for:
- ticks and warmup ticks
- seed
- device
- grid width and height
- agent count
- wall and H-zone counts
- ray count
- PPO cadence parameters
- checkpoint cadence
- log directory and JSON output path
- optional cProfile top-N output
- experimental family-vmap inference gate and minimum bucket size

## 3. Outputs

The script writes or prints values including:
- elapsed seconds
- ticks per second
- RSS before and after
- RSS delta
- CUDA peak memory, when relevant
- final tick
- final alive count
- runtime checkpoint metadata
- buffered parquet row count
- run directory
- inference-path stats

## 4. Interpretation discipline

A benchmark result should be read as:
- a statement about one configuration
- under one device choice
- under one map and population size
- with one logging and checkpoint configuration

It should **not** be read as a universal claim about the repository.


## Read next
- [Extension safety, testing, and change protocol](07_extension_safety_testing_and_change_protocol.md)
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)

## Related reference
- [Inference path comparison](../assets/diagrams/learning/inference_path_comparison.md)

## If debugging this, inspect…
- [Validation harnesses](05_validation_determinism_resume_consistency_and_soak.md)

## Terms introduced here
- `benchmark harness`
- `ticks per second`
- `RSS`
- `profiling surface`
- `interpretation limit`
