# Commenting Audit Report

## Documentation Objective
Strengthen the operational readability of the repository without adding tutorial noise. The documentation pass focused on invariants, sequencing boundaries, ownership expectations, checkpoint publish safety, telemetry batching semantics, and the configuration surface.

## Files Materially Documented
- `config.py`
- `src/tensor_crypt/app/runtime.py`
- `src/tensor_crypt/simulation/engine.py`
- `src/tensor_crypt/checkpointing/atomic_checkpoint.py`
- `src/tensor_crypt/checkpointing/runtime_checkpoint.py`
- `src/tensor_crypt/telemetry/data_logger.py`
- `src/tensor_crypt/world/physics.py`

## Config.py Documentation Improvements
- Rewrote the module docstring so `config.py` explicitly states its role as the canonical operational control surface.
- Documented the active-versus-guarded knob policy so unsupported compatibility knobs are not mistaken for live features.
- Added targeted comments on checkpoint cadence, telemetry cadence, viewer overlay control, tie-break policy, and guarded semantic surfaces such as `SIM.DTYPE`, `RESPAWN.MODE`, `PPO.REWARD_FORM`, and `PPO.OWNERSHIP_MODE`.
- Preserved defaults while making the operational intent of each newly wired knob explicit.

## Non-Comment Code Changes Required During The Doc Pass
- Added runtime config validation so the documented guarded knobs actually fail loudly when misused.
- Added docstrings to checkpoint publishing, runtime checkpoint validation, telemetry buffering, tick summary emission, and death-boundary handling functions.
- Added a targeted inline comment in `Engine._batched_brain_forward` documenting why the code deliberately avoids unsafe parameter stacking across UID-owned modules.

## Stale Or Misleading Commentary Removed Or Neutralized
- The checkpoint publish path is now documented as same-directory temp-file promotion rather than implying generic atomicity without filesystem constraints.
- Engine ordering comments were tightened around catastrophe timing and post-tick checkpoint timing.
- Telemetry batching is now documented as a flush-timing optimization rather than a semantic change to ledger contents.

## Checks Run
- `python -m compileall src scripts tests config.py run.py main.py`
- `pytest -q --basetemp .pytest_tmp_clean4 tests/test_prompt7_checkpoint_atomicity.py tests/test_benchmark_and_scheduler.py tests/test_physics.py`

## Result
The repository now has denser operational guidance in the code paths where sequencing, ownership, and publish semantics matter most, with `config.py` serving as the primary operator reference as required.
