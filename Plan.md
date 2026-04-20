# Plan

## Milestones

- [x] Map canonical code, entrypoints, config, tests, docs, packaging, and git state.
- [x] Isolate `release/five-brain` worktree from clean commit `049fdfe`.
- [x] Remove single-family launch contamination from this branch.
- [x] Add branch-specific README, architecture, AGENTS, and working artifacts.
- [x] Run targeted validation.
- [x] Review final diff for semantic drift, stale single-family language, release residue, and config/test mismatch.
- [x] Prepare release commit, tag target, and publication command set.

## Acceptance Checks

- `python run.py` and `python main.py` preserve five-family defaults.
- Runtime config exposes canonical five-family defaults with vmap disabled.
- Public wrappers remain thin.
- Tests cover launch, dependency governance, brain families, perception, checkpoint, PPO, and vmap guard behavior.

## Validation Commands

```powershell
python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py
python -m pytest tests\test_bloodline_brains.py tests\test_perception.py tests\test_registry_respawn.py
python -m pytest tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py tests\test_experimental_family_vmap_inference.py
python scripts\benchmark_runtime.py --device cpu --ticks 16 --warmup-ticks 2 --output artifacts\release\five_brain_benchmark_smoke.json
```

## Validation Results

- `python -m pytest tests\test_imports_and_compat.py tests\test_dependency_governance.py tests\test_bloodline_brains.py tests\test_perception.py tests\test_registry_respawn.py tests\test_runtime_checkpoint_substrate.py tests\test_uid_ppo_hardening.py tests\test_experimental_family_vmap_inference.py` - 42 passed.
- `python scripts\benchmark_runtime.py --device cpu --ticks 16 --warmup-ticks 2 --width 10 --height 10 --agents 4 --walls 0 --hzones 0 --update-every 8 --batch-size 4 --mini-batches 1 --epochs 1 --output artifacts\release\five_brain_benchmark_smoke.json` - passed; final tick 18, final alive 4, vmap disabled.
- `python -m pytest` - 180 passed.

## Review Results

- `git diff --check` reported no whitespace errors; only expected CRLF conversion warnings from Git on Windows.
- Stale-name scan found no conflict markers, removed authoring-input paths, old patch-report filenames, or old renamed test filenames.
- Remaining uses of `temporary` are legitimate catastrophe/checkpoint semantics.
- Code-bearing diff is limited to five-brain launch/config expectations, package metadata, and branch-facing documentation.
