# Performance and Scaling Report

## 1. Title and purpose

This document is the repository’s standalone performance and scaling report for Tensor Crypt.

Its purpose is to describe, in implementation-grounded terms, how runtime cost should be measured, how the existing benchmark and soak runners should be interpreted, which performance surfaces are real in the current repository, and where optimization work remains safe versus semantically dangerous.

This document is an add-on to the core bundle. It does not replace the canonical explanations of PPO, telemetry, checkpointing, validation, runtime assembly, or observation/model contracts. It exists to make performance work operationally safe, reproducible, and honest.

---

## 2. Intended audience

This document is written for three overlapping readers:

1. Operators who need to run controlled measurements without corrupting scientific comparisons.
2. Maintainers who need to understand which code paths dominate runtime cost and which changes are likely to change semantics.
3. Researchers who want to compare configurations, devices, telemetry settings, and experimental inference paths without overclaiming speed or portability.

The writing stays beginner-readable where needed, but it assumes that the reader is willing to treat repository truth as the governing source.

---

## 3. Why this document exists in addition to the core bundle

The core bundle explains what the system is, how it is assembled, how PPO works, how telemetry and checkpointing behave, and how validation is structured.

That is necessary, but it is not sufficient for safe performance work.

Performance work creates a different class of failure:

- measuring the wrong thing
- comparing runs that changed semantics instead of only cost
- treating environment-specific numbers as architecture truths
- turning experimental fast paths into assumed defaults
- disabling telemetry or checkpoints without understanding what evidence was lost
- using a soak/invariant run as if it were a throughput benchmark

This add-on exists to close that gap.

It provides a disciplined frame for reading the repository’s headless benchmark harness, headless soak runner, runtime assembly order, checkpoint cadence, telemetry buffering, AMP/device controls, and experimental same-family vmap inference path.

---

## 4. Evidence basis and limits

### 4.1 Evidence basis

This document is grounded in repository-visible evidence from the uploaded codebase dump and the supplied blueprint.

The most relevant implementation surfaces are:

- the headless benchmark harness
- the headless soak runner
- runtime assembly and determinism setup
- engine tick ordering
- telemetry buffering and summary export logic
- runtime checkpoint scheduling and retention
- PPO AMP/scaler usage
- experimental family-vmap inference eligibility and path accounting
- validation and test surfaces that prove these paths exist

### 4.2 Evidence limits

This repository contains performance instrumentation, but it does **not** ship a body of authoritative benchmark results.

Accordingly, this document does **not** claim:

- absolute throughput on any machine
- universal speedups
- a guaranteed gain from CUDA over CPU for every configuration
- a guaranteed gain from the experimental family-vmap path
- a guaranteed cost reduction from any proposed optimization unless the repository already measures that exact effect

Where the repository exposes counters, timings, memory samples, or path statistics, this document explains how to interpret them.

Where the repository only exposes a tuning surface, this document treats it as a cost surface or hypothesis surface, not as proof of benefit.

### 4.3 What counts as measured versus inferred here

**Measured by repository instrumentation**
- elapsed benchmark window time
- ticks per second for the measured window
- process RSS before and after the measured window
- CUDA peak allocated memory during the measured window when the run is on CUDA
- buffered parquet row count at result-capture time
- whether runtime checkpoints fired during the benchmark window
- which inference path categories were used during the benchmark window
- cProfile cumulative listings when profiling is enabled

**Validated rather than benchmarked**
- non-finite-state failures during soak
- checkpoint save/load surface stability during soak
- deterministic or resume-consistency surfaces in the validation harness
- same-output parity checks around the experimental family-vmap path

**Not measured by the repository as shipped**
- end-to-end scaling curves across hardware tiers
- ablation tables of telemetry overhead
- ablation tables of checkpoint overhead
- comparative charts across family distributions
- long-run memory fragmentation studies
- detailed GPU utilization traces
- per-subsystem wall-clock splits beyond optional cProfile listings

---

## 5. Main sections

### 5.1 Measurement methodology and evidence hierarchy

Performance work in this repository should follow a strict measurement hierarchy.

#### Level A: use the repository’s benchmark harness for throughput-oriented comparisons

The benchmark harness exists to run a reproducible headless measurement window. It uses dummy SDL drivers, builds a normal runtime, performs optional warmup ticks, measures the selected tick window, and emits a JSON result payload.

This is the correct surface for statements such as:

- “configuration X produced higher measured ticks per second than configuration Y on this machine”
- “telemetry cadence change A reduced buffered rows and improved measured throughput under this benchmark configuration”
- “the experimental family-vmap path was or was not actually exercised in this run”

It is **not** the right surface for long-form safety auditing.

#### Level B: use the soak runner for invariant durability, not speed claims

The soak runner is built for long-form correctness pressure, not for throughput comparison. It forces CPU execution, disables AMP, repeatedly checks finite-state invariants, validates PPO ownership surfaces, and optionally runs checkpoint save/load validation at a configured cadence.

This is the correct surface for statements such as:

- “this configuration stayed finite for N ticks”
- “checkpoint save/load did not alter tick, slot-UID bindings, or grid shape during periodic validation”
- “this tuning experiment did or did not survive long-form invariant pressure”

It is **not** the right surface for a throughput leaderboard.

#### Level C: use the validation harness and tests to delimit truth boundaries

The validation harness and tests matter for performance interpretation because they prove specific contracts:

- the benchmark harness exists and emits the expected fields
- the experimental family-vmap path can be parity-tested against the loop path
- topology and ownership constraints are deliberate
- checkpoint and resume surfaces have explicit validation logic

These surfaces do not replace direct measurement, but they strongly constrain what may be claimed.

#### Practical rule

Use the benchmark harness to say **how fast a measured run was**.

Use the soak runner to say **whether a long-form configuration stayed valid**.

Use tests and validation logic to say **which performance paths are real, guarded, or experimental**.

---

### 5.2 Headless benchmark runner: purpose, inputs, outputs, and interpretation

#### Purpose

The headless benchmark runner is the repository’s direct throughput-and-memory measurement harness.

It exists to produce a controlled benchmark window without the interactive viewer and without accidental dependence on real display or audio drivers.

#### What it configures

The benchmark harness exposes these runtime-facing inputs:

- measured ticks and warmup ticks
- seed
- device selection: `auto`, `cpu`, or `cuda`
- grid width and height
- initial agent count
- wall count and heal-zone count
- ray count
- PPO cadence and optimizer-shape controls:
  - update frequency
  - batch size
  - mini-batch count
  - epoch count
- telemetry controls:
  - summary cadence
  - parquet batch rows
- checkpoint controls:
  - periodic checkpoint interval
  - keep-last retention
- output path
- optional cProfile depth
- optional experimental family-vmap enable flag
- optional family-vmap minimum bucket threshold

#### What it deliberately suppresses

The benchmark harness sets `LOG.LOG_TICK_EVERY` and `LOG.SNAPSHOT_EVERY` beyond the warmup-plus-measurement window.

That means the benchmark is intentionally trying to avoid progress-print and snapshot overhead during the measured interval.

This matters. A benchmark result from this harness is **not** the same thing as a full-fidelity operator session with normal snapshots and prints enabled.

#### Measurement window structure

The benchmark window has four phases:

```text
runtime build
-> warmup ticks (not included in elapsed window)
-> memory baseline capture
-> timed measured ticks
-> JSON result capture
-> logger close / flush
```

Important interpretation detail: the result JSON is captured **before** final logger close. The close path then flushes remaining parquet buffers and closes file handles.

#### Reported outputs

The benchmark result payload contains:

| Field | Meaning | Interpretation note |
|---|---|---|
| `device` | Effective runtime device | `auto` resolves to `cuda` only when CUDA is available, otherwise `cpu` |
| `ticks` | Number of measured ticks | This excludes warmup ticks |
| `warmup_ticks` | Warmup count | Used before time and memory sampling |
| `elapsed_sec` | Measured wall time for the benchmark window | On CUDA, the runner synchronizes before reading elapsed time |
| `ticks_per_sec` | `ticks / elapsed_sec` | Environment-specific measured throughput |
| `rss_before_mb` | Process RSS before measured ticks | CPU-side process memory, not a full device memory picture |
| `rss_after_mb` | Process RSS after measured ticks | Compare with care; allocator behavior matters |
| `rss_delta_mb` | RSS change across the measured window | Useful as a run-local signal, not a universal memory law |
| `cuda_peak_mb` | Peak CUDA allocated memory during the measured window | Zero on CPU runs |
| `final_tick` | Engine tick after warmup plus measured ticks | This usually equals `warmup_ticks + ticks` if nothing exited early |
| `final_alive` | Alive population at capture time | Helps distinguish cost changes caused by population collapse versus real efficiency changes |
| `last_runtime_checkpoint_tick` | Last scheduled runtime checkpoint tick hit in-window | Zero or unchanged if checkpoint cadence was disabled or not reached |
| `last_runtime_checkpoint_path` | Path of the last scheduled runtime checkpoint | Useful when benchmarking checkpoint cost surfaces |
| `buffered_parquet_rows` | In-memory parquet rows at capture time | Captured before final close/flush |
| `run_dir` | Run directory used for artifacts | Useful for later inspection |
| `profile_top_cumulative` | Optional cProfile cumulative listing | Present only when profiling depth is requested |
| `experimental_family_vmap_inference` | Whether the experimental fast path was enabled | Enablement is not proof of actual use |
| `experimental_family_vmap_min_bucket` | Vmap eligibility threshold | Only relevant when the experimental flag is on |
| `inference_path_stats` | Aggregate path counts across measured ticks | The most important field for interpreting whether fast-path inference was actually exercised |

#### How to read `inference_path_stats`

The benchmark accumulates four counters across measured ticks:

| Counter | Meaning |
|---|---|
| `loop_slots` | Number of slot-forwards executed through the canonical per-brain loop path |
| `vmap_slots` | Number of slot-forwards executed through the experimental same-family vmap path |
| `family_loop_buckets` | Number of family buckets processed through the loop path |
| `family_vmap_buckets` | Number of family buckets processed through the vmap path |

This lets the operator distinguish two very different situations:

1. the experimental flag was enabled but no bucket ever became eligible, and
2. the experimental flag was enabled and some measured work actually moved to the vmap path.

That distinction is essential. An “enabled” fast path that never fired is not evidence of a fast-path speed result.

#### Benchmark interpretation traps

**Trap 1: treating `final_tick` as measured-tick count.**  
It includes warmup progress. A run with `warmup_ticks = 16` and `ticks = 128` will end at a later engine tick than 128.

**Trap 2: treating `buffered_parquet_rows` as data loss.**  
The benchmark records this value before final logger close. The logger close path then flushes buffered ledgers. A non-zero value at capture time means “still buffered,” not necessarily “not persisted.”

**Trap 3: treating RSS delta as the full memory story.**  
RSS is process memory, not a complete device-memory or allocator-fragmentation account.

**Trap 4: treating `cuda_peak_mb` as a CPU/CUDA-neutral metric.**  
It only exists on CUDA runs and is collected after resetting peak stats before the measured window.

**Trap 5: treating `ticks_per_sec` as architecture-only truth.**  
It depends on hardware, device path, allocator state, benchmark parameters, family distribution, telemetry settings, checkpoint cadence, and whether PPO updates fired.

---

### 5.3 Soak runner versus benchmark runner

The repository intentionally separates these tools because they answer different questions.

| Surface | Primary purpose | Device policy | AMP policy | Main success criterion |
|---|---|---|---|---|
| Benchmark runner | Measure timed throughput and memory signals | Configurable | Enabled automatically on CUDA in the harness | Stable result payload with meaningful throughput/memory counters |
| Soak runner | Pressure-test invariants over a long-form headless run | CPU only | Disabled | No invariant or checkpoint-validation failure |

#### What the soak runner validates

During the soak run, the repository checks:

- registry invariants against the grid
- finiteness of registry state
- finiteness of grid state
- whether PPO buffer keys still refer to known lifecycle UIDs
- whether PPO optimizer keys still refer to known lifecycle UIDs
- finiteness of all live brain parameters
- periodic checkpoint capture, validation, save, and reload surfaces

It also verifies that checkpoint save/load does not alter at least these surfaces during periodic validation:

- engine tick
- `slot_uid` bindings
- grid tensor shape

#### Why soak results are not benchmark results

The soak runner:

- forces CPU execution
- forces AMP off
- uses a different objective
- spends work on invariant checking
- optionally spends work on periodic checkpoint surface validation
- reports a correctness-style success line rather than a throughput payload

A configuration that passes soak is not automatically fast.  
A configuration that is fast under benchmark is not automatically safe under soak.

A serious optimization workflow should use both.

---

### 5.4 CPU, GPU, AMP, and device-path considerations

#### Runtime validation boundaries

The runtime validation surface currently enforces several important device and numeric constraints:

- `SIM.DTYPE` is validated against a runtime-supported set that currently accepts only `float32`.
- requesting a CUDA device when CUDA is unavailable is rejected during runtime validation.
- the experimental family-vmap path is rejected if `torch.func` support is unavailable and the flag is forced on.

These are not stylistic comments. They are explicit runtime boundaries.

#### Determinism setup

The determinism path seeds:

- Python random
- NumPy
- Torch CPU
- Torch CUDA seed-all when CUDA is available

On CUDA-capable systems, the runtime also sets:

- float32 matmul precision to high
- cuDNN deterministic mode to true
- cuDNN benchmark mode to false

This matters for performance interpretation because a “reproducibility-first” configuration can differ from a “maximum auto-tuned throughput” configuration.

This repository chooses the reproducibility-first side of that trade.

#### AMP behavior

AMP is not a universal always-on runtime fact.

In the benchmark harness:
- AMP is enabled when the effective device is CUDA.
- AMP is disabled on CPU.

In the soak runner:
- AMP is forced off.

Inside PPO:
- a CUDA GradScaler is created only when `LOG.AMP` is true and CUDA is available.
- the mixed-precision training branch is used only under that condition.

That means AMP affects the PPO update path, not merely a cosmetic device flag.

#### Practical comparison rule for CPU versus CUDA

When comparing CPU and CUDA runs:

1. keep seed, grid, agents, ray count, map density, PPO cadence, telemetry cadence, checkpoint cadence, and experimental flags fixed
2. record whether PPO updates actually occurred during the measured window
3. treat `ticks_per_sec`, RSS, and CUDA peak memory as machine-specific measurements
4. do not generalize a machine-local result into a repository-wide guarantee

---

### 5.5 Throughput, cadence, buffering, checkpoint, and telemetry cost surfaces

The repository exposes several real cost surfaces. They do not all have the same semantic risk.

#### 5.5.1 Simulation-size and world-density surfaces

These surfaces change how much work exists per tick:

- grid width and height
- initial agent count
- wall count and wall lengths
- heal-zone count and scale
- ray count

These are valid benchmark knobs, but they change the problem size. They do not isolate implementation efficiency by themselves.

#### 5.5.2 PPO cadence and optimizer-work surfaces

These surfaces change how much learning work is performed:

- update interval
- rollout batch threshold
- mini-batch count
- epoch count
- KL early-stop threshold
- gradient clipping branch execution
- AMP on/off for CUDA training updates

These strongly affect runtime cost. They also affect learning behavior, optimizer pressure, and often scientific meaning. They are therefore measurement knobs **and** semantics-sensitive knobs.

#### 5.5.3 Telemetry cost surfaces

The logger is intentionally buffered so that hot-path telemetry does not collapse into per-event file I/O.

The most visible cost-control knobs are:

- summary export cadence
- family summary cadence
- summary skip-on-non-emit behavior
- parquet batch row threshold
- the broader telemetry ledger enable flags

The logger buffers row groups in memory and flushes a buffer once its row count reaches `PARQUET_BATCH_ROWS`. It also flushes all parquet buffers on controlled boundaries such as close.

This leads to two important conclusions:

1. telemetry cost is partly a **frequency** problem and partly a **flush granularity** problem  
2. in-memory buffered rows at an intermediate point are part of the normal design, not automatically a fault

The tick summary path itself also includes useful operator-facing counters, including:

- live population
- births and deaths this tick
- catastrophe status
- PPO buffer UID count
- PPO buffer transition count
- optimizer UID count
- buffered parquet row count

That means some telemetry is also introspective about its own cost surfaces.

#### 5.5.4 Snapshot cost surfaces

Snapshot work is separate from parquet ledgers. The engine’s snapshot path can emit:

- agent snapshots
- heatmap snapshots
- brain snapshots

The benchmark harness suppresses snapshot cadence during its timed window. That is appropriate for measuring the simulation/training loop more cleanly.

An operator should not compare a benchmark result with snapshots suppressed against an interactive or audit run that keeps normal snapshot cadence, then treat the difference as a pure device or model effect.

#### 5.5.5 Checkpoint cost surfaces

The checkpoint path is deliberately durable and therefore has real cost.

Relevant surfaces include:

- periodic save interval
- keep-last retention
- atomic publish enablement
- manifest enablement
- checksum enablement
- latest-pointer writing
- strict validation behaviors

The engine publishes runtime checkpoints only after the post-tick state has settled. The checkpoint call occurs after:

- physics
- death handling
- births
- tick summary logging
- tick increment
- PPO update path
- snapshot path

This design is good for resume integrity, but it means scheduled checkpoints are not a zero-cost background feature. They are a durable post-tick publish surface.

Retention pruning is also active when `KEEP_LAST > 0`, which adds directory enumeration and old-artifact cleanup work.

#### 5.5.6 Progress-print cost surface

The engine also has a tick progress print cadence. This is usually a smaller cost than snapshots or checkpoint publication, but it is still a measurable surface in high-frequency short benchmarks. The benchmark harness pushes this beyond the measured window.

---

### 5.6 Experimental family-vmap or experimental inference-path considerations

The repository contains an experimental same-family inference fast path.

This is not the canonical default path.

#### What it is

During batched action sampling, the engine groups alive slots by family. For each family bucket, it checks whether that bucket is eligible for a same-family vmap forward.

If a bucket is eligible, the engine stacks module state across the bucket and runs a vmap-based forward over the bucket entries.

If a bucket is not eligible, it falls back to the loop path.

#### Eligibility conditions

A family bucket must satisfy all of the following conditions to be vmap-eligible:

- the experimental flag is enabled
- `torch.func` support is available
- bucket size is at least the configured minimum threshold
- the exemplar brain exists and is not in training mode
- every brain in the bucket exists
- every brain in the bucket is the same type
- every brain in the bucket has the same topology signature

These checks matter because they make the fast path a real guarded surface, not a blanket replacement for all per-agent inference.

#### What may be claimed safely

The following claims are safe when supported by a measured run:

- the experimental path was enabled
- some or no slots were actually processed through the vmap path
- some or no family buckets were actually processed through the vmap path
- parity tests exist that compare vmap and loop outputs in tested conditions
- tests also exist showing threshold behavior and mixed-singleton fallback behavior

#### What may not be claimed safely without new measurement

The following claims require a real benchmark campaign and are **not** implied by enablement alone:

- that the vmap path is always faster
- that the vmap path benefits every family distribution
- that the vmap path benefits small-bucket workloads
- that the vmap path improves training quality
- that the vmap path should replace the canonical ownership-preserving loop in all cases

#### Operational reading rule

Always read these three items together:

- `experimental_family_vmap_inference`
- `experimental_family_vmap_min_bucket`
- `inference_path_stats`

Without the third field, the first two fields are not enough to interpret the run.

---

### 5.7 What can be optimized safely versus what risks changing semantics

The table below is the practical safety core of this document.

| Change category | Examples | Truth partition | Performance relevance | Semantic risk |
|---|---|---|---|---|
| Measurement-only harness use | Changing output path, profiling depth, run directory, warmup length for methodology purposes | Implemented runtime behavior | High | Low, if the measured window definition is reported honestly |
| Telemetry emission cadence | Summary cadence, family-summary cadence, parquet batch rows, skip-non-emit work | Implemented runtime behavior | High | Low to moderate for runtime semantics; high for observability and artifact density |
| Snapshot cadence | `LOG.SNAPSHOT_EVERY` | Implemented runtime behavior | Moderate to high | Low for simulation state, moderate for auditability |
| Checkpoint cadence and retention | `SAVE_EVERY_TICKS`, `KEEP_LAST` | Implemented runtime behavior | Moderate to high | Low for immediate simulation semantics, high for durability/resume evidence |
| Device selection | CPU versus CUDA | Implemented runtime behavior | High | Low for declared runtime semantics, but environment-specific |
| AMP enablement on CUDA | `LOG.AMP` in CUDA runs | Implemented runtime behavior | Moderate to high | Moderate for numeric behavior; must be compared carefully |
| Experimental family-vmap enablement | experimental flag and min bucket | Implemented runtime behavior plus guarded experimental path | Potentially high | Moderate; guarded and opt-in |
| Problem-size changes | grid size, agent count, wall count, heal-zone count, ray count | Implemented runtime behavior | Very high | High for benchmark comparability because the workload changed |
| PPO work changes | update interval, batch size, mini-batches, epochs, reward config | Implemented runtime behavior | Very high | High; this changes learning cadence and often scientific meaning |
| Ownership/topology changes | brain family structure, UID ownership assumptions, checkpoint schema surfaces | High-risk schema or semantics surfaces | Potentially high | Very high |
| Unsupported compatibility values | unsupported dtype, unsupported spawn mode, unsupported mode values | Guarded compatibility surface | Not valid benchmark surfaces unless runtime support is extended | Very high or invalid |
| Theory-only optimization ideas not present in repo | custom kernel work, alternate batching architectures, allocator tricks not implemented here | Adjacent theory / conceptual background | Unknown | Unknown until implemented and validated |

#### Safe-first optimization policy

A conservative performance pass should optimize in this order:

1. measurement clarity
2. telemetry overhead
3. snapshot overhead
4. checkpoint cadence for non-durability tests
5. experimental inference path measurement
6. only then deeper algorithmic or topology changes

That ordering minimizes the chance of confusing “less work” with “better implementation.”

---

### 5.8 How to design reproducible performance experiments for this repository

#### 5.8.1 Baseline recipe

Use this sequence for a clean baseline:

1. choose one device path
2. freeze seed
3. freeze grid, agent count, wall count, heal-zone count, ray count, PPO cadence, and experimental flags
4. decide whether snapshots, checkpoints, and telemetry are part of the workload you want to measure
5. run the benchmark harness
6. archive the JSON result and run directory
7. repeat enough times to understand run-to-run variation on the same machine

#### 5.8.2 Telemetry sweep recipe

When studying telemetry cost, hold these fixed:

- device
- seed
- world size and density
- PPO cadence
- checkpoint cadence
- experimental inference flags

Then vary only:

- summary cadence
- family summary cadence if relevant
- parquet batch rows
- deep-ledger enable surfaces if your experiment includes them

Interpret the result using:

- `ticks_per_sec`
- RSS fields
- `buffered_parquet_rows`
- run-directory artifact density

#### 5.8.3 Checkpoint sweep recipe

When studying checkpoint overhead, hold all non-checkpoint factors fixed and vary only:

- `SAVE_EVERY_TICKS`
- `KEEP_LAST`
- manifest/checksum/pointer surfaces only if you are intentionally studying publication durability cost

Interpret using:

- `ticks_per_sec`
- `last_runtime_checkpoint_tick`
- `last_runtime_checkpoint_path`
- run-directory checkpoint contents

Remember that the checkpoint path is post-tick and durability-oriented. It is not designed as a minimal-cost ephemeral cache.

#### 5.8.4 Experimental family-vmap recipe

To study the experimental same-family inference path:

1. hold device and workload fixed
2. run once with the experimental flag off
3. run once with the experimental flag on
4. choose a minimum bucket threshold that is realistic for the family distribution under test
5. inspect `inference_path_stats`

Do not treat the experiment as informative unless the vmap counters are actually non-zero in the run you are comparing.

#### 5.8.5 Aggressive tuning safety rule

Any aggressive performance tuning that changes:
- PPO cadence
- reward surfaces
- topology
- ownership assumptions
- checkpoint schema surfaces
- inference ownership behavior

should be followed by:
- soak validation
- resume or checkpoint validation if relevant
- parity testing where available

#### Reproducible performance run recipe box

```text
freeze: seed, device, grid, agents, walls, hzones, num_rays,
        PPO cadence, checkpoint cadence, telemetry cadence,
        experimental flags

measure with benchmark runner:
    - warmup
    - timed ticks
    - JSON result
    - run_dir capture

validate with soak runner when tuning is aggressive:
    - finite-state checks
    - UID/PPO ownership checks
    - brain-parameter finiteness
    - periodic checkpoint save/load validation
```

---

### 5.9 Which reported numbers are environment-specific versus architecture-specific

This repository makes it possible to collect both environment-specific and structure-specific signals. They must not be mixed carelessly.

| Signal type | Examples | How to treat it |
|---|---|---|
| Environment-specific measured result | `ticks_per_sec`, RSS fields, `cuda_peak_mb`, cProfile cumulative time | Valid only for the tested machine, software stack, and configuration |
| Run-shape-specific measured result | `final_alive`, `buffered_parquet_rows`, `last_runtime_checkpoint_tick` | Interpretable for that run; useful for explaining cost differences |
| Path-usage signal | `loop_slots`, `vmap_slots`, family-bucket counters | Explains which implementation path was exercised during that run |
| Architecture or contract boundary | dtype restriction, CUDA availability validation, vmap eligibility rules, checkpoint ordering | Repository-level implementation truth; not a machine-local opinion |
| Scientific-workload change | grid size, agent count, ray count, PPO cadence changes | Not a pure performance comparison; the workload changed |

A practical summary is:

- **throughput and memory numbers are environment-specific**
- **path eligibility rules and validation boundaries are architecture-specific**
- **path usage counters are run-specific evidence about what the architecture actually did during one measurement**

---

## 6. Figures/tables/diagrams to include or defer

### Included in this document

1. **Measurement pipeline diagram**  
   Included in Section 5.2 as a compact text pipeline.

2. **Benchmark output interpretation table**  
   Included in Section 5.2.

3. **Benchmark versus soak distinction table**  
   Included in Section 5.3.

4. **Safe optimization versus semantics-risk table**  
   Included in Section 5.7.

5. **Environment-specific versus architecture-specific table**  
   Included in Section 5.9.

6. **Reproducible performance run recipe box**  
   Included in Section 5.8.

### Explicitly deferred

The following artifacts are deliberately deferred because the repository inputs do not provide authoritative measured datasets:

- comparative throughput charts
- scaling curves across population sizes
- CPU-versus-CUDA bar charts
- checkpoint-overhead trend charts
- telemetry-overhead trend charts
- vmap-versus-loop speedup charts

Those figures should be generated only from an actual benchmark campaign and then attached as run-specific evidence, not written into static documentation as if they were repository facts.

---

## 7. Cross-links to core bundle documents

This add-on should be read together with the core bundle, especially these exact IDs:

- **20**
- **21**
- **40**
- **41**
- **42** — PPO
- **50** — telemetry artifacts
- **51** — checkpointing
- **52** — validation and soak methods

Cross-linking discipline for maintainers:

- use **42** when explaining why PPO cadence changes alter both cost and learning behavior
- use **50** when expanding telemetry artifact semantics beyond the cost surface described here
- use **51** when expanding checkpoint durability, manifest, and resume semantics
- use **52** when expanding deterministic, resume, soak, and final-validation methodology

This document should not duplicate those canonical chapters. It should only connect them into a performance-and-scaling operating frame.

---

## 8. Truth-boundary notes

This document reuses the repository’s frozen truth partitions exactly.

### 8.1 Implemented runtime behavior

The following are implemented and can be documented as live repository behavior:

- headless benchmark runner
- headless soak runner
- runtime config validation
- deterministic setup
- buffered telemetry parquet writes
- scheduled runtime checkpoints with retention pruning
- post-tick checkpoint publication
- CUDA AMP support in PPO updates when enabled
- experimental same-family vmap inference eligibility and counters

### 8.2 Guarded compatibility surface

The following are real public surfaces but are not open-ended:

- dtype selection
- certain mode strings validated against explicit supported sets
- the experimental family-vmap path, which is opt-in and eligibility-gated

These must not be described as broad free-form implementation spaces.

### 8.3 Public but currently unread / effectively dead surface

If a surface is public in config but not read in the active runtime, it should not be upgraded in documentation into a live performance feature merely because it exists in configuration comments.

This rule is especially important for performance writing because speculative knobs are often mistaken for active optimization levers.

### 8.4 Adjacent theory / conceptual background

General ML systems advice, kernel-fusion ideas, allocator strategies, or alternate batching architectures belong here unless the repository actually implements and validates them.

They may inform future work, but they are not current repository behavior.

---

## 9. Maintainer notes where relevant

### 9.1 Preserve the benchmark-versus-soak separation

Do not collapse the benchmark harness and soak runner into one “does everything” tool.

They serve different purposes, and merging them would make both interpretation and maintenance less clear.

### 9.2 Treat new performance counters as documentation surfaces

If maintainers add new counters, timers, or artifact fields, they should update this document and the core telemetry/checkpoint references together.

A performance counter that is undocumented is easy to misuse.

### 9.3 Do not claim speedups from path existence

The existence of:
- a fast path
- a guard
- a test
- a benchmark field
- an optimization idea

is not itself a speed result.

Measured claims should stay in benchmark evidence, not in structural prose.

### 9.4 Keep post-tick durability ordering explicit

The engine currently saves runtime checkpoints only after the post-tick state settles. That ordering is part of the resume-safety story and should not be “optimized away” casually for lower apparent checkpoint cost.

Any change there should be treated as both a performance change and a correctness change.

### 9.5 When extending the benchmark harness, prefer additive instrumentation

Good extensions include:

- extra measured fields
- clearer run metadata
- explicit hardware metadata capture
- optional artifact manifests for benchmark campaigns
- more direct telemetry and checkpoint cost counters

Riskier extensions include:

- changing benchmark semantics silently
- mixing correctness pressure into the timed window
- hiding whether a fast path was actually used
- changing the default workload shape without making it explicit

### 9.6 Minimum responsible workflow for a serious performance patch

A serious performance patch should normally produce all of the following:

1. benchmark results on a declared machine and declared configuration
2. evidence that the intended path actually fired when that matters
3. soak or validation evidence when semantics-sensitive code changed
4. updated documentation of any new measured fields or guarded surfaces

That is the minimum standard for turning an optimization idea into a repository-backed claim.
