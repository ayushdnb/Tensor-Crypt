# Run Directory Artifact Tree

> Owning document: [Run directory artifacts and file outputs](../../../05_operations/02_run_directory_artifacts_and_file_outputs.md)

## What this asset shows
- the major folders and files created around a run

## What this asset intentionally omits
- every optional artifact variant

```mermaid
flowchart TD
    A[run_<timestamp>] --> B[config.json]
    A --> C[run_metadata.json]
    A --> D[snapshots/]
    A --> E[brains/]
    A --> F[heatmaps/]
    A --> G[simulation_data.hdf5]
    A --> H[parquet ledgers]
    A --> I[lineage_graph.json]
    A --> J[checkpoints/]
    J --> K[bundle.pt]
    J --> L[manifest.json]
    J --> M[latest.json]

```
