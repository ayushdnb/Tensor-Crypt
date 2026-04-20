# Release Notes

## Single-Brain Vmap 0.9.0 Research Preview

This release line presents Tensor Crypt as a self-centric single-family research runtime with guarded vmap-capable inference.

### Changed

- Public startup applies the self-centric single-family preset.
- Benchmark, soak, and resume-audit scripts apply the branch preset by default.
- README and architecture docs identify this branch as the single-brain-vmap line.
- Internal phase-report and authoring-input residue was removed from public technical assets.

### Compatibility

- `run.py`, `main.py`, `config.py`, `engine.*`, and `viewer.*` remain compatibility surfaces.
- Checkpoint, telemetry, UID, PPO, and viewer artifact names remain unchanged.
- Experimental branch metadata is checkpoint-visible and validated on restore.

### Validation

- Full pytest: 193 passed.
- Headless benchmark smoke: passed with `vmap_slots=64`.
