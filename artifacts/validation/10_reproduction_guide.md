# Reproduction Guide

## Core Verification

```powershell
python -m pytest -q
```

## Targeted Validation Slices

```powershell
python -m pytest tests\test_imports_and_compat.py tests\test_checkpoint_atomicity.py tests\test_validation_harness.py -q
python -m pytest tests\test_benchmark_and_scheduler.py tests\test_validation_harness.py -q
```

## Runtime Evidence Commands Executed

```powershell
python scripts\benchmark_runtime.py --device cpu --ticks 256 --warmup-ticks 32 --width 20 --height 20 --agents 12 --walls 4 --hzones 2 --update-every 16 --batch-size 8 --mini-batches 2 --epochs 1 --log-dir artifacts\validation\logs\benchmark_cpu_256 --output artifacts\validation\metrics\benchmark_cpu_256.json
python scripts\benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --width 20 --height 20 --agents 16 --walls 4 --hzones 2 --update-every 16 --batch-size 8 --mini-batches 2 --epochs 1 --experimental-family-vmap-inference --experimental-family-vmap-min-bucket 2 --log-dir artifacts\validation\logs\benchmark_vmap_128 --output artifacts\validation\metrics\benchmark_vmap_128.json
python scripts\run_resume_chain_audit.py --cycles 5 --ticks-per-cycle 8 --width 16 --height 16 --agents 8 --walls 0 --hzones 0 --log-dir artifacts\validation\logs\resume_chain --output artifacts\validation\metrics\resume_chain_audit.json
python scripts\run_soak_audit.py --ticks 768 --seed 42 --width 20 --height 20 --agents 12 --walls 4 --hzones 2 --log-dir artifacts\validation\logs\soak_seed42 --checkpoint-validate-every 64
```

## Key Artifacts
- `artifacts/validation/metrics/benchmark_cpu_256.json`
- `artifacts/validation/metrics/benchmark_vmap_128.json`
- `artifacts/validation/metrics/resume_chain_audit.json`
- `artifacts/validation/metrics/memory_window_probe_seed77.json`
- `artifacts/validation/logs/soak_seed42/run_20260417_213234`
