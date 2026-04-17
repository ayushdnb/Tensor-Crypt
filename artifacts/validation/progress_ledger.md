# Validation Progress Ledger

## Cycle 1 - 2026-04-17

### Completed in this cycle
- Created branch `validation/full-hardening-audit-20260417-a`.
- Established `artifacts/validation` evidence tree.
- Verified canonical live-repo evidence sources and read the supplied validation blueprint.
- Identified current canonical entrypoints and config/runtime validators under `tensor_crypt/`.
- Confirmed the worktree was already dirty before this audit and recorded that constraint.
- Checked for the additional `evolution.txt` evidence named in the prompt; no separate attached `evolution.txt` artifact was present in the working tree.

### Total completed so far
- Phase 0 reconnaissance started.
- Phase 1 scaffold started.

### Still remaining
- Deep static audit across runtime assembly, ownership invariants, checkpointing, telemetry, viewer, and compatibility wrappers.
- Baseline validation runs and failure collection.
- Test-suite expansion for uncovered or weakly-covered high-risk surfaces.
- Fault injection, soak/memory instrumentation, convergence analysis, and validated fixes.
- Final evidence package and reproduction guide.

### Confidence level
- Low. Reconnaissance only; no runtime verdict yet.

### Immediate next action
- Read the canonical runtime, checkpoint, registry, PPO, telemetry, and viewer surfaces in detail, then run a targeted baseline validation subset to find real breakage.

### Blocking uncertainties
- The repository contains many pre-existing local modifications in canonical modules and docs that are not from this audit.
- No standalone `evolution.txt` evidence file was available despite being referenced in the task prompt.

## Cycle 2 - 2026-04-17

### Completed in this cycle
- Fixed checkpoint latest-pointer resolution for relative bundle paths and added a regression test that exercises checkpoint-directory loading from relative run directories.
- Repaired the repo test harness for Windows/Python 3.14 by removing the forced pytest `--basetemp` and replacing built-in `tmp_path` usage with a repo-owned workspace fixture.
- Disabled pytest cacheprovider in repo config because its atomic cache temp writes were hitting the same permission class without adding validation value.
- Added checkpoint fault-injection coverage for pointer-checksum corruption and temp-file cleanup after bundle-write failure.
- Added explicit launch-entrypoint validation that the root `run.py`/`main.py` surface applies the experimental single-family preset before building the runtime.
- Added repeated resume-chain validation in `tensor_crypt.audit.final_validation.run_resume_chain_probe()` plus a reusable `scripts/run_resume_chain_audit.py` harness and smoke coverage.
- Ran the full test suite after the changes: `180 passed in 54.27s`.
- Produced benchmark, soak, resume-chain, and sampled RSS evidence under `artifacts/validation/metrics` and `artifacts/validation/logs`.

### Total completed so far
- Phase 0 reconnaissance completed.
- Phase 1 scaffold completed.
- Phase 2 static and structural audit materially advanced.
- Phase 3 test-suite expansion completed for the chosen high-risk gaps.
- Phase 4 runtime-mode evidence collected for canonical CPU and experimental family-vmap CPU probes.
- Phase 5 fault-injection campaign completed for checkpoint publication surfaces.
- Phase 6 medium soak and repeated resume-chain evidence collected.
- Phase 7 convergence/stability assessment drafted from actual runs.

### Still remaining
- Final git commit/push bookkeeping.
- Final concise report.

### Confidence level
- Medium for repository submission readiness under the exercised CPU/headless surfaces.
- Low for unexercised CUDA or multi-million-tick claims.

### Immediate next action
- Finalize artifact summaries, commit the audit branch, and report the evidence-disciplined verdict.

### Blocking uncertainties
- 10,000,000-tick operation is still not directly proven.
- Memory-leak absence is still not proven; the sampled RSS window rose over 512 ticks before partially dropping.
- No CUDA-path validation was executed in this audit run.
