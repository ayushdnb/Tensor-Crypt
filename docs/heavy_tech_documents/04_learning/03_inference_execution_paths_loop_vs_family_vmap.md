# Inference Execution Paths: Loop versus Family-Vmap

> Scope: Explain the engine’s default per-slot inference path and the optional experimental family-vmap path without overstating the latter as a semantic replacement.

## Who this document is for
Technical readers, maintainers, and auditors studying the learning path, data ownership, and checkpoint-sensitive training surfaces.

## What this document covers
- default loop inference
- family bucketing
- vmap eligibility
- topology checks
- what stays unchanged semantically

## What this document does not cover
- generic RL exposition beyond what is needed for the repository
- viewer operation details

## Prerequisite reading
- [Learning overview](00_learning_system_overview_and_data_ownership.md)
- [Traits, bloodlines, families, and brain instantiation](../03_mechanics/03_traits_bloodlines_families_and_brain_instantiation.md)

## 1. Default path

The engine’s baseline inference route iterates through family buckets and, when needed, falls back to per-slot loop execution. This is the stable, non-experimental path.

## 2. Experimental path

An explicitly experimental switch can enable family-vmap inference. Eligibility requires:
- the experimental gate enabled
- `torch.func` support available
- bucket size above the configured minimum
- same module type across the bucket
- identical topology signatures
- non-training mode for the exemplar and peers

## 3. What vmap does not imply

It does **not** imply:
- shared parameters
- shared optimizers
- shared PPO ownership
- shared identity
- guaranteed speedup under all conditions

It is a batched execution convenience for a subset of inference situations.

## 4. Why topology signature checks exist

A family name alone is not sufficient if topology drift could occur. The engine therefore checks parameter-shape signature compatibility before using the batched path.

## 5. Logging and benchmark consequence

The benchmark script records inference-path stats such as:
- loop slots
- vmap slots
- family loop buckets
- family vmap buckets

Those counters are interpretation aids, not proof that the experimental path is always beneficial.


## Read next
- [Checkpoint-visible learning state and restore order](04_checkpoint_visible_learning_state_and_restore_order.md)
- [Benchmarking and performance probe manual](../05_operations/06_benchmarking_and_performance_probe_manual.md)

## Related reference
- [Inference path comparison](../assets/diagrams/learning/inference_path_comparison.md)

## If debugging this, inspect…
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)

## Terms introduced here
- `family bucket`
- `vmap eligibility`
- `topology signature`
- `experimental inference path`
