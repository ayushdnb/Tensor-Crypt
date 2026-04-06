# Performance Audit And Results

## Baseline Benchmark Commands
Baseline and post-change throughput were collected with the same headless harness parameters before and after the patch.

Equivalent workload parameters:
- ticks: 96
- world: 24 x 24
- agents: 16
- walls: 4
- harm zones: 2
- rays: 8
- PPO update_every: 16
- PPO batch_size: 8
- PPO mini_batches: 2
- PPO epochs: 1

The baseline runs were taken before `scripts/benchmark_runtime.py` existed, using an inline harness with the exact workload above. The repository now contains a reusable harness for the same surface:

```powershell
python scripts/benchmark_runtime.py --device cpu --ticks 96 --warmup-ticks 0 --width 24 --height 24 --agents 16 --walls 4 --hzones 2 --num-rays 8 --update-every 16 --batch-size 8 --mini-batches 2 --epochs 1 --log-dir .pytest_tmp/benchmark_script_cpu --output .pytest_tmp/benchmark_script_cpu.json
python scripts/benchmark_runtime.py --device cuda --ticks 96 --warmup-ticks 0 --width 24 --height 24 --agents 16 --walls 4 --hzones 2 --num-rays 8 --update-every 16 --batch-size 8 --mini-batches 2 --epochs 1 --log-dir .pytest_tmp/benchmark_script_cuda --output .pytest_tmp/benchmark_script_cuda.json
```

## Baseline Results

| Device | Elapsed Seconds | Ticks/Sec | RSS Delta MiB | CUDA Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| CPU | 9.3235 | 10.2965 | 171.60 | n/a |
| CUDA | 19.7938 | 4.8500 | 806.75 | 122.91 |

## Hotspot Findings Before Changes
- `Engine.step` orchestration was dominated by `_sample_actions`, `_batched_brain_forward`, and `PPO.update`.
- Repeated `brain.eval()` churn showed up in steady-state profiling even though the brains were already inference-only outside optimizer updates.
- `DataLogger.log_tick_summary` and parquet writes were visible in the hot path.
- Physics and engine loops were paying repeated `.item()` extraction and Python churn costs.
- Broad compilation was not a safe candidate because orchestration is dynamic and graph-break-heavy.

## Implemented Optimization Program
- Switched inference-only runtime paths to `torch.inference_mode()`.
- Restored brains to eval mode after training so inference does not flip module mode every tick.
- Replaced repeated scalar extraction with bulk `.tolist()` conversion in engine and physics hot loops.
- Vectorized environment-effect application and death-range clamping.
- Buffered parquet writes behind bounded row queues.
- Added summary export cadence control so operators can explicitly reduce summary cost.

## Rejected Optimization Candidates
- `torch.compile` on the full runtime path: rejected because dynamic orchestration, `.item()` usage, logging, and Python control flow would create graph-break-heavy behavior with limited upside and elevated risk.
- `torch.func` family-stacked execution: rejected because UID-owned modules and optimizers are currently first-class state, and stacking parameters would materially complicate optimizer, checkpoint, and ownership guarantees.

## Post-Change Benchmark Commands
Representative post-change reproducible harness surfaces:

```powershell
python scripts/benchmark_runtime.py --device cpu --ticks 96 --warmup-ticks 0 --width 24 --height 24 --agents 16 --walls 4 --hzones 2 --num-rays 8 --update-every 16 --batch-size 8 --mini-batches 2 --epochs 1 --profile-top 20 --log-dir .pytest_tmp/benchmark_script_cpu --output .pytest_tmp/benchmark_script_cpu.json
```

The measured delta table below uses the same pre-script inline benchmark harness used for the baseline so the comparison remains apples-to-apples.

## Post-Change Results

| Device | Elapsed Seconds | Ticks/Sec | RSS Delta MiB | CUDA Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| CPU | 9.2863 | 10.3378 | 168.93 | n/a |
| CUDA | 17.1816 | 5.5874 | 798.41 | 122.91 |

## Measured Deltas

| Device | Ticks/Sec Delta | Relative Delta | RSS Delta |
| --- | ---: | ---: | ---: |
| CPU | +0.0413 | +0.4% | -2.67 MiB |
| CUDA | +0.7374 | +15.2% | -8.34 MiB |

## Profiling Interpretation
- The CUDA path benefited most because redundant mode toggles, hot-path telemetry writes, and extra host syncs were reduced.
- CPU throughput changed only slightly because the dominant remaining costs are still per-agent forward passes, perception raycasting, and PPO update work.
- Steady-state cProfile after the patch no longer shows `DataLogger.log_tick_summary` parquet writes or repeated `Module.eval()` among the top cumulative costs.
- `_AgentBuffer` finite validation still costs measurable time, but that cost is intentional because it protects ownership and numeric integrity on the PPO path.

## Regressions Or Tradeoffs
- Telemetry rows are now buffered, so filesystem visibility shifts from immediate-per-event persistence to bounded batch persistence. Semantic content is unchanged.
- Checkpoint validation is stricter and can now fail runs that previously limped forward with inconsistent or missing manifest state.

## Benchmark Outcome
The patch produced a material end-to-end gain on the GPU path and a small but real gain on the CPU path while hardening checkpointing and telemetry surfaces. The remaining dominant costs are architecture-level model and perception work, not the audit-driven safety and observability additions.
