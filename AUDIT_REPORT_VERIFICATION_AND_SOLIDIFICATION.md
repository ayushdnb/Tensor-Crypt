# Audit Report: Verification And Solidification

## 1. Executive Summary
Tensor Crypt already had a stronger-than-average regression base, but the audit found several materially important gaps between "tests pass" and "runtime is trustworthy for long learning runs". The most serious proven defects were: a movement-resolution bug that could create two live agents with the same position, a reward computation path that could assign positive reward to negative HP because squaring happened before clamping, a partially wired observation config path that broke when canonical observations were disabled, missing explicit brain shape guards, insufficient checkpoint topology validation, incomplete poison-zone death attribution, a PPO KL-sign issue that weakened the target-KL safeguard, and an audit harness that compared too little state and initially produced false confidence.

The repository is materially stronger after this pass. The test suite grew from 88 to 98 passing tests. The audit harness now compares stronger state surfaces, the soak helper now validates runtime/checkpoint/PPO integrity during endurance runs, and the core simulation/learning code now fails faster and more explicitly when shape or non-finite state contracts are violated.

## 2. Repository Understanding And Intended Architecture
The repository truth matches the documented top-level architecture with one important qualifier: some config knobs were declared more broadly than their runtime wiring actually supported.

Proven architecture facts from code inspection:
- `config.py` is the canonical knob surface. Internal modules consume it through `src/tensor_crypt/config_bridge.py`.
- `run.py` and `main.py` are thin launch entrypoints. Runtime assembly happens in `src/tensor_crypt/app/runtime.py`.
- `src/tensor_crypt/*` is the canonical implementation tree.
- `engine/*` and `viewer/*` are compatibility re-export wrappers, not independent logic trees.
- Runtime assembly order is stable: create logger, grid, registry, physics, perception, PPO, evolution, map, initial population, engine, viewer.
- The simulation is slot-backed for dense runtime tensors, but identity, lineage, and PPO ownership are UID-owned.
- Observation flow is canonical-first (`canonical_rays`, `canonical_self`, `canonical_context`) with a legacy adapter surface kept alive for compatibility.
- PPO ownership is keyed by canonical UID, not slot index.
- Checkpoint restore order is conservative and intentional: registry tensors and ledgers, then brains, then PPO state, then RNG, then catastrophe state.
- Catastrophes are reversible per-tick overlays and runtime modifiers, not permanent mutations of the baseline world definition.

## 3. Critical Invariants Identified
The following invariants were treated as critical and either directly tested, hardened, or both:
- One active UID per live slot.
- No live slot without a valid canonical UID binding.
- No duplicate active UID bindings.
- No duplicate live positions when `NO_STACKING` is enabled.
- Grid occupancy must agree with registry slot positions.
- Each active UID owns exactly one live brain and the brain family must match the UID family ledger.
- PPO buffers, optimizer state, and training counters are UID-owned and must not leak across slot reuse.
- Rollout buffers must be schema-consistent, non-ragged, and finite.
- Active non-terminal buffers require explicit bootstrap closure before PPO update.
- Canonical observation tensors must have the documented ranks, widths, and shared batch dimension.
- Checkpoint brain topology metadata must match the configured family topology.
- Save/load validation must preserve registry, grid, and family state surfaces.
- Catastrophe overlays and runtime modifiers must be reversible and must not leak into baseline world state.
- Telemetry death rows must preserve meaningful causal context when available.

## 4. Test Strategy Implemented
The verification stack now consists of:
- Unit tests for observation contracts, PPO math helpers, finite-buffer validation, checkpoint topology validation, and deterministic telemetry fields.
- Integration/regression tests for engine reward handling, contested movement resolution, legacy-only observation flow, and poison-zone death attribution.
- Existing subsystem tests retained and extended rather than replaced.
- Stronger audit-harness tests validating determinism, resume consistency, catastrophe reproducibility, and save-load-save surfaces.
- Endurance validation via `scripts/run_soak_audit.py`, now augmented with periodic checkpoint validation and PPO/brain finite-state checks.

## 5. New Tests Added
New tests added in this audit pass:
- `tests/test_physics.py::test_contested_move_into_occupied_cell_does_not_create_duplicate_positions`
- `tests/test_perception.py::test_legacy_only_observation_surface_still_supports_brain_forward`
- `tests/test_bloodline_brains.py::test_brain_rejects_canonical_shape_mismatch_with_clear_error`
- `tests/test_bloodline_brains.py::test_brain_rejects_canonical_batch_size_mismatch`
- `tests/test_runtime_checkpoint_substrate.py::test_checkpoint_validation_rejects_brain_topology_signature_mismatch`
- `tests/test_engine_and_viewer_integration.py::test_engine_reward_clamps_negative_health_before_squaring`
- `tests/test_ppo.py::test_buffer_validation_rejects_non_finite_transition_values`
- `tests/test_ppo.py::test_ppo_approx_kl_uses_old_minus_new_log_prob_sign`
- `tests/test_prompt7_audit.py::test_prompt7_validation_suite_respects_config_flags`
- `tests/test_verification_telemetry_integrity.py::test_poison_zone_death_records_zone_id`

## 6. Existing Tests Improved
Existing test surfaces were expanded rather than replaced:
- `tests/test_prompt7_audit.py` now validates stronger save/load equivalence and config-flag wiring in the audit harness.
- `tests/test_engine_and_viewer_integration.py` now checks reward clamping on lethal HP.
- `tests/test_runtime_checkpoint_substrate.py` now verifies checkpoint topology metadata integrity.
- `tests/test_bloodline_brains.py`, `tests/test_perception.py`, `tests/test_physics.py`, and `tests/test_ppo.py` now cover previously unguarded contract boundaries.

## 7. Bugs Or Defects Found
Proven defects found during the audit:
- `Physics.step()` could produce duplicate live positions when a contest winner moved into an occupied cell whose occupant later failed to vacate.
- `Engine.step()` computed squared HP-ratio reward before clamping HP into `[0, hp_max]`, allowing dead or over-healed states to produce wrong reward.
- `Perception.build_observations()` unconditionally mutated `obs["canonical_self"]`, breaking the documented `RETURN_CANONICAL_OBSERVATIONS=False` surface.
- Canonical observation shape assumptions were implicit and could fail deep inside matrix multiplications with poor error messages.
- Checkpoint validation ignored saved brain topology signatures even though they were persisted.
- Poison-zone death telemetry omitted `zone_id` even though the grid could resolve it.
- PPO buffer validation did not reject non-finite rollout contents.
- PPO KL telemetry/thresholding used `new_log_prob - old_log_prob`, which weakens the target-KL safeguard because the sign is wrong for the intended old-vs-new comparison.
- The audit harness compared too little state and initially produced false confidence.
- The original concurrent determinism/resume probes were also invalid because multiple runtimes in one process share process-global RNG surfaces.
- Checkpoint save/load under the system `python` 3.14 interpreter perturbed the global torch RNG and triggered a temp-directory/permission anomaly during soak checkpoint validation. Under the installed Python 3.10 test environment, the repository behaved correctly.

## 8. Defects Fixed
Fixed in this pass:
- Reworked movement resolution in `src/tensor_crypt/world/physics.py` so blocked-vacate cases no longer create duplicate live positions.
- Clamped HP-derived reward input in `src/tensor_crypt/simulation/engine.py` before squaring.
- Routed effective vision through `src/tensor_crypt/world/observation_schema.py` so both canonical and legacy observation surfaces reflect fog modifiers without mutating stored traits.
- Added explicit canonical and legacy observation shape validation in `src/tensor_crypt/agents/brain.py`.
- Added checkpoint topology-signature validation in `src/tensor_crypt/checkpointing/runtime_checkpoint.py`.
- Added poison-zone `zone_id` attribution in `src/tensor_crypt/world/physics.py`.
- Hardened PPO buffer validation against non-finite rollout payloads in `src/tensor_crypt/learning/ppo.py`.
- Corrected PPO approximate-KL sign in `src/tensor_crypt/learning/ppo.py`.
- Strengthened `src/tensor_crypt/audit/final_validation.py` to compare registry/grid/brain digests and to respect validation feature flags.
- Serialized the determinism/catastrophe/resume probes and restored captured RNG before live-continuation comparisons so the harness measures runtime equivalence rather than shared-RNG contamination or checkpoint-I/O side effects.
- Strengthened `scripts/run_soak_audit.py` with periodic checkpoint validation plus finite PPO/brain checks.

## 9. Defects Not Fixed And Why
Not fixed in this pass:
- `cfg.CHECKPOINT.STRICT_SCHEMA_VALIDATION` appears declared but unused. I did not invent semantics for it without a stronger project-level decision.
- `cfg.VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES` appears declared but unused. I did not tighten telemetry writer behavior without a clearer migration policy for missing/added columns.
- Logger continuation is not checkpoint-restored. Simulation/runtime state can now be restored to a logically equivalent engine/registry/PPO/catastrophe state, but open parquet writers and logger-internal bookkeeping are not resumed mid-run.
- Tensor Crypt still uses process-global RNG surfaces rather than per-runtime RNG objects for policy sampling. The audit harness now compensates, but runtime isolation between multiple concurrent runtimes in one process remains unimplemented.
- `ActiveCatastrophe.remaining_ticks` is still a misleading name because it returns event duration, not live remaining time. It is currently unused, so I did not change that surface without checking for downstream expectations.

## 10. Shape-Contract Audit
Proven after this pass:
- Canonical observation tensors are now explicitly validated for rank, feature width, and shared batch dimension.
- Legacy observation tensors are now explicitly validated for rank and shared batch dimension before adaptation.
- Canonical-to-legacy and legacy-to-canonical surfaces are now both exercised by tests.
- Family-specific brain topology remains stable within family and allowed to differ across families.
- Checkpoint metadata now validates saved topology signatures against current family topology.

Remaining shape risks:
- The project still relies on config-driven widths for many surfaces, so incompatible manual config edits can still be destructive if they invalidate persisted checkpoints or old telemetry. Those paths now fail earlier, but they are not dynamically migrated.

## 11. PPO / RL / Learning Audit
Proven findings:
- PPO ownership remains UID-owned, not slot-owned.
- Slot reuse does not inherit dead UID optimizer state.
- Rollout buffers are now validated for both structure and finiteness.
- The target-KL statistic had the wrong sign for old-vs-new comparison; this is now corrected.
- Reward computation could previously become mathematically wrong when HP became negative or temporarily exceeded `hp_max`; this is now clamped.

Inference-based caution, not a proven defect:
- The squared HP-ratio reward is coherent but still a strong shaping choice. It compresses already-low-health differences and heavily rewards remaining near full health. That may be intentional; I did not change it.

## 12. Checkpoint And Restore Audit
Proven after code and test changes:
- Checkpoint bundles contain registry tensors, UID ledgers, grid state, active brain state, PPO buffer/training state, catastrophe state, and RNG state.
- Topology metadata is now validated, not just stored.
- Stronger save-load-save validation now compares registry, fitness, and grid digests in addition to prior identity surfaces.
- Resume consistency probes are now valid for the repository’s actual global-RNG model.

Remaining limitation:
- Checkpoints restore simulation/learning state, not logger continuation state.

## 13. Catastrophe And Runtime Modifier Audit
Proven facts:
- Catastrophe field overlays remain reversible by rebuild-per-tick design.
- World modifier reset/apply order is stable.
- Fog now correctly affects both exported observation surfaces.
- Poison-zone deaths now carry `zone_id` into telemetry when resolvable.

Residual limitation:
- Attribution for overlapping catastrophes remains first-active-event oriented in physics death context. Overlap is disabled by default, but if overlap is enabled this attribution remains simplified.

## 14. Long-Run Robustness Audit
Proven by validation:
- Full `pytest` passes with 98 tests.
- 128-tick headless soak with periodic checkpoint validation passes under the Python 3.10 environment used by the test stack.
- The duplicate-position bug previously exposed by the soak path is fixed.
- PPO buffer validation, finite brain parameter checks, and checkpoint validation now execute during soak.

Still unproven:
- Very long multi-thousand-tick or multi-hour training stability.
- GPU-specific long-run behavior.
- Performance/throughput regressions under larger populations and wider grids.

## 15. Telemetry And Observability Audit
Improved in this pass:
- Death telemetry now records poison-zone `zone_id` when available.
- Death ledger coverage around null vs non-null killer UID remains preserved by pre-existing worktree changes and tests.
- Validation artifacts now make it easier to audit save/load equivalence and runtime drift.

Residual limitations:
- Telemetry writer strictness config remains undeclared in runtime behavior.
- Logger state is not checkpoint-restored.

## 16. Code Hardening Changes
Primary hardening changes:
- Explicit observation shape guards.
- Explicit non-finite PPO buffer validation.
- Reward clamping before squaring.
- Checkpoint topology validation.
- Stronger audit harness digests.
- Stronger soak validation with checkpoint surface checks.

## 17. Remaining Risks / Unknowns
Remaining risks after this audit:
- Concurrent multi-runtime execution in one process is still not an isolated simulation model because of shared global RNG.
- Logger continuation across checkpoint resume remains unproven and currently unsupported.
- Some config knobs remain declared but semantically stale.
- Catastrophe overlap attribution remains simplified if overlap is manually enabled.
- Very long-run numerical behavior on GPU remains unproven.

## 18. Enhancement Recommendations
Near-term high-value improvements:
- Introduce per-runtime RNG objects for policy sampling and any other stochastic runtime paths, then route all determinism/resume tests through those isolated streams.
- Decide and implement semantics for `STRICT_SCHEMA_VALIDATION` and `STRICT_TELEMETRY_SCHEMA_WRITES` rather than leaving them declarative only.
- Capture/logger-resume state if checkpoint resume is expected to preserve uninterrupted telemetry ledgers.
- Add a longer soak tier with 1k+ ticks and periodic checkpoint save-load cycles under both CPU and CUDA when available.

Medium-term architecture improvements:
- Separate simulation state hashing and audit probes into a reusable validation utility module instead of embedding those surfaces only in `final_validation.py`.
- Add explicit runtime state schemas for catastrophe manager and logger state if those surfaces are intended to be durable.
- Consider replacing process-global RNG use with explicit per-domain RNG ownership.

Better observability / diagnostics:
- Add a compact per-tick invariant ledger containing hash digests of registry/grid/brain state for long-run drift diagnosis.
- Emit checkpoint validation summaries into run artifacts during soak or resume drills.
- Add explicit PPO anomaly telemetry for non-finite gradients, clipped-value saturation rate, and target-KL early-stop frequency.

Research / evaluation recommendations:
- Add benchmark scenarios with fixed seeds for catastrophe-heavy learning, extinction-recovery pressure, and repeated slot-reuse stress.
- Add family-wise policy divergence and reward-distribution diagnostics to detect unintended cross-family collapse.

Performance ideas that preserve semantics:
- Family-batch forward grouping in `Engine._batched_brain_forward()` could reduce Python overhead while preserving behavior if implemented carefully.
- State hashing in audit probes should remain opt-in for production-scale training runs if probe cost becomes meaningful.

## 19. Exact Commands Run
Commands executed during the audit included:
- `git status --short --branch`
- `rg --files`
- `pytest -q`
- `pytest -q tests\test_physics.py tests\test_perception.py tests\test_bloodline_brains.py tests\test_runtime_checkpoint_substrate.py tests\test_engine_and_viewer_integration.py tests\test_verification_telemetry_integrity.py`
- `pytest -q tests\test_prompt7_audit.py`
- `pytest -q tests\test_ppo.py`
- `python scripts\run_soak_audit.py --ticks 128`
- `python scripts\run_soak_audit.py --ticks 128 --checkpoint-validate-every 32`
- `C:\Users\ayush\AppData\Local\Programs\Python\Python310\python.exe scripts\run_soak_audit.py --ticks 128 --checkpoint-validate-every 32`

Notes on command outcomes:
- The `pytest` command used the installed Python 3.10 environment and passed.
- The system `python` command resolved to Python 3.14 in this environment; soak checkpoint validation under that interpreter hit a temp-directory/permission anomaly during torch serialization. The soak validation succeeded under the Python 3.10 environment used by the test stack.

## 20. Validation Results
Validated successfully:
- Full repository test suite: `98 passed`.
- Prompt 7 audit harness tests: passed with stronger state comparisons and flag wiring.
- 128-tick headless soak with periodic checkpoint validation: passed under Python 3.10.

## 21. Branch Name And Commit Hash
Branch name: `audit/research-grade-verification-solidification`
Commit hash: `b137ea3`


