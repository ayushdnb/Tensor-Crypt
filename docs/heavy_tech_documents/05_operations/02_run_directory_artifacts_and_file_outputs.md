# Run Directory Artifacts and File Outputs

> Scope: Explain the per-run directory created at launch, the major durable artifacts the logger and checkpointing path emit, and how to interpret them conservatively.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- run directory creation
- subdirectories
- config snapshot
- run metadata
- ledger files
- snapshots and brains
- checkpoint directory

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Operator quickstart](00_operator_quickstart_and_common_run_modes.md)
- [Runtime assembly](../02_system/02_runtime_assembly_launch_sequence_and_session_graph.md)

## 1. Run directory creation

The run-path helper creates a timestamped `run_YYYYMMDD_HHMMSS` directory beneath the configured log root, adding a numeric suffix if the timestamp already exists.

## 2. Baseline structure visible in the dump

The helper ensures at least these subdirectories exist:
- `snapshots`
- `brains`
- `heatmaps`

It also writes:
- `config.json`
- `run_metadata.json`

## 3. Major logger outputs

The data logger can emit:
- `simulation_data.hdf5`
- `birth_ledger.parquet`
- `genealogy.parquet`
- `life_ledger.parquet`
- `death_ledger.parquet`
- `collisions.parquet`
- `ppo_events.parquet`
- `tick_summary.parquet`
- `family_summary.parquet`
- `catastrophes.parquet`
- `lineage_graph.json`

## 4. What these artifacts mean

| Artifact | Meaning |
| --- | --- |
| `config.json` | snapshot of runtime configuration |
| `run_metadata.json` | compact description of runtime identity, checkpoint, observation, catastrophe, telemetry, and viewer surfaces |
| HDF5 file | snapshot-style structured storage |
| parquet ledgers | append-oriented operational records |
| lineage JSON | graph-form lineage export derived from canonical UID surfaces |
| `checkpoints/` directory | runtime checkpoint bundles and their side files |

## 5. Buffered parquet writes

The logger buffers parquet rows and flushes them in batches according to telemetry settings. A missing parquet write during a hot run does not necessarily mean the event did not occur; it may still be buffered until flush or close.


## Read next
- [Checkpointing, manifests, restore, and latest pointer](03_checkpointing_manifests_restore_and_latest_pointer.md)
- [Telemetry ledgers, snapshots, exports, and lineage graph](04_telemetry_ledgers_snapshots_exports_and_lineage_graph.md)

## Related reference
- [Run directory artifact tree](../assets/diagrams/operations/run_directory_artifact_tree.md)

## If debugging this, inspect…
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)

## Terms introduced here
- `run directory`
- `parquet ledger`
- `HDF5 snapshot`
- `run metadata`
- `buffered flush`
