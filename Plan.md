# Plan

## Milestones

- [x] Map canonical code, entrypoints, config, tests, docs, packaging, and git state.
- [x] Isolate `release/single-brain-vmap` worktree from clean commit `049fdfe`.
- [x] Apply self-centric single-family implementation commits onto the hardened base.
- [x] Align README, scripts, architecture docs, AGENTS, and working artifacts to this branch.
- [x] Run targeted validation.
- [x] Review final diff for semantic drift, stale five-brain assumptions, release residue, and config/test mismatch.
- [x] Prepare release commit, tag target, and publication command set.

## Acceptance Checks

- `python run.py` and `python main.py` apply the self-centric single-family preset.
- Benchmark, soak, and resume-audit scripts exercise this branch's preset by default.
- Checkpoint validation records observation mode and branch-family metadata.
- Tests cover launch, dependency governance, experimental observation shapes, vmap parity, registry identity, and checkpoint mismatch rejection.

## Validation Commands

```powershell
python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py
python -m pytest tests\test_experimental_family_vmap_inference.py tests\test_bloodline_brains.py tests\test_perception.py
python -m pytest tests\test_registry_respawn.py tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py
python scripts\benchmark_runtime.py --device cpu --ticks 16 --warmup-ticks 2 --experimental-family-vmap-min-bucket 2 --output artifacts\release\single_brain_vmap_benchmark_smoke.json
```

## Validation Results

- `python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py tests\test_experimental_family_vmap_inference.py tests\test_bloodline_brains.py tests\test_perception.py tests\test_registry_respawn.py tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py tests\test_benchmark_and_scheduler.py` - 58 passed.
- `python scripts\benchmark_runtime.py --device cpu --ticks 16 --warmup-ticks 2 --width 10 --height 10 --agents 4 --walls 0 --hzones 0 --update-every 8 --batch-size 4 --mini-batches 1 --epochs 1 --experimental-family-vmap-min-bucket 2 --output artifacts\release\single_brain_vmap_benchmark_smoke.json` - passed; final tick 18, final alive 4, vmap slots 64.
- `python -m pytest` - 193 passed.

## Review Results

- `git diff --check` reported no whitespace errors; only expected CRLF conversion warnings from Git on Windows.
- Stale-name scan found no conflict markers, removed authoring-input paths, old patch-report filenames, or old renamed test filenames.
- Remaining uses of `temporary` are legitimate catastrophe/checkpoint semantics.
- Code-bearing diff is limited to branch preset startup/tooling, package metadata, benchmark expectations, and branch-facing documentation.
