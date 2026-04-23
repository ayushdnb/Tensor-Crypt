# Troubleshooting and Failure Atlas

> Scope: Provide symptom-oriented operational guidance for launch failures, viewer issues, checkpoint issues, validation failures, and suspicious runtime behavior.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- symptom-to-first-check mapping
- checkpoint troubleshooting
- viewer troubleshooting
- validation interpretation
- config mistakes

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Operator quickstart](00_operator_quickstart_and_common_run_modes.md)
- [Checkpointing and manifests](03_checkpointing_manifests_restore_and_latest_pointer.md)

## 1. Launch fails before viewer appears

First checks:
1. inspect device selection versus CUDA availability
2. inspect runtime validation errors for unsupported enum values
3. confirm dependency surfaces are using `pygame-ce` rather than legacy `pygame`
4. inspect the configured log root and permissions

## 2. Viewer opens but behaves unexpectedly

First checks:
1. verify whether the simulation is paused
2. check whether `+` / `-` is editing a selected H-zone rather than changing speed
3. confirm catastrophe panel and overlay toggles are in the intended state
4. fit the world with `F` if the camera is disoriented after resize or zoom

## 3. Resume from checkpoint fails

First checks:
1. validate the bundle, manifest, and latest pointer as a file set
2. confirm schema versions and config fingerprint policy expectations
3. inspect whether family topology or observation-shape changes have occurred since the checkpoint was created
4. inspect optimizer metadata mismatch reports rather than bypassing them casually

## 4. Determinism or resume probe fails

Treat this as a correctness issue, not as harmless noise. Inspect:
- RNG capture and restore policy
- restore ordering
- checkpoint bundle completeness
- catastrophe scheduler state
- any recent change to UID, observation, or family-topology surfaces

## 5. Performance probe looks worse than expected

Interpretation checks:
- confirm logging cadence and checkpoint cadence
- confirm benchmark map and population size
- confirm device
- confirm whether the experimental family-vmap path was actually eligible
- inspect inference-path stats rather than assuming vmap was used


## Read next
- [Documentation index and reading guide](../00_program/00_documentation_index_and_reading_guide.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Related reference
- [Troubleshooting decision tree](../assets/diagrams/operations/troubleshooting_decision_tree.md)

## If debugging this, inspect…
- [Validation harnesses](05_validation_determinism_resume_consistency_and_soak.md)

## Terms introduced here
- `symptom atlas`
- `first check`
- `resume failure`
- `interpretation error`
