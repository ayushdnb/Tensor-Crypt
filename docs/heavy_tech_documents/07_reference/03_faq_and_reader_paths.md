# FAQ and Reader Paths

> Scope: Answer common orientation questions and provide compact path recommendations for different readers and tasks.

## Who this document is for
All readers who want a short answer before diving into the full narrative documents.

## What this document covers
- common questions
- reader routes
- where to jump next

## What this document does not cover
- full subsystem detail

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)

## FAQ

### Is `run.py` the architecture owner?
No. It is the canonical root-level start surface for users. The implementation center of gravity still lives under `tensor_crypt`.

### Are slot ids the same thing as agent identity?
No. Slot ids are runtime storage locations. Canonical identity is monotonic UID.

### Does family-vmap mean shared-brain training?
No. The experimental vmap path is an inference batching path, not parameter sharing.

### Does the existence of a config field prove the runtime uses it?
No. The config comments and validation path explicitly distinguish active, guarded, and currently unread surfaces.

### Is checkpointing just a `.pt` file?
No. The active checkpoint story includes bundle validation, manifests, latest pointers, optimizer metadata checks, and restore ordering.

### Where should an operator start?
Start in `docs/heavy_tech_documents/05_operations/00_operator_quickstart_and_common_run_modes.md`.

### Where should a beginner start?
Start in `docs/heavy_tech_documents/01_foundations/00_foundations_learning_roadmap.md`.

### Where should a maintainer start before making changes?
Read:
1. `docs/heavy_tech_documents/02_system/01_package_layout_canonical_modules_and_compatibility_wrappers.md`
2. `docs/heavy_tech_documents/02_system/03_runtime_config_taxonomy_and_knob_safety.md`
3. `docs/heavy_tech_documents/05_operations/07_extension_safety_testing_and_change_protocol.md`
## Read next
- [Documentation index and reading guide](../00_program/00_documentation_index_and_reading_guide.md)
- [Operator quickstart and common run modes](../05_operations/00_operator_quickstart_and_common_run_modes.md)
- [Foundations learning roadmap](../01_foundations/00_foundations_learning_roadmap.md)

## Related reference
- [Corpus manifest](../07_reference/99_corpus_manifest.md)
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)

## If debugging this, inspect…
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Terms introduced here
- `reader path`
- `orientation question`
