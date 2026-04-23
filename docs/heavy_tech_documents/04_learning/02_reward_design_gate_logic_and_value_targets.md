# Reward Design, Gate Logic, and Value Targets

> Scope: Explain the repository’s PPO reward surface, gate modes, and the relationship between rewards, values, returns, and advantage targets.

## Who this document is for
Technical readers, maintainers, and auditors studying the learning path, data ownership, and checkpoint-sensitive training surfaces.

## What this document covers
- configured reward form
- gate modes
- below-gate replacement value
- reward versus value distinction
- why validation rejects invalid reward surfaces

## What this document does not cover
- generic RL exposition beyond what is needed for the repository
- viewer operation details

## Prerequisite reading
- [Reinforcement learning background](../01_foundations/06_reinforcement_learning_mdp_policy_value_and_advantage.md)
- [PPO buffers and update cadence](01_ppo_buffers_bootstrap_and_update_cadence.md)

## 1. Reward surface defined by the current implementation

The engine exposes a configurable PPO reward computation path. The current supported reward form is `sq_health_ratio`, which squares the clamped HP ratio.

## 2. Reward gating

The repository also supports reward gating with at least these modes:
- `off`
- `hp_ratio_min`
- `hp_abs_min`

If the gate condition fails, the system substitutes the configured below-gate value rather than silently leaving the reward unchanged.

## 3. Why validation matters

The runtime validates reward-surface configuration before the run starts. This prevents semantically inconsistent combinations such as illegal thresholds or unsupported forms from entering the active path.

## 4. Reward is not value

Reward is what the environment emits at the current step. Value is the critic’s estimate of expected future return. Return and advantage are derived targets used for training, not primary environment emissions.

## 5. Health-clamp consequence

The checked-in tests explicitly guard against negative-health squaring mistakes. That means the reward surface is not a casual algebraic helper; it is treated as correctness-sensitive.


## Read next
- [Inference execution paths: loop versus family-vmap](03_inference_execution_paths_loop_vs_family_vmap.md)
- [Validation harnesses](../05_operations/05_validation_determinism_resume_consistency_and_soak.md)

## Related reference
- [Benchmarking and performance probe manual](../05_operations/06_benchmarking_and_performance_probe_manual.md)

## If debugging this, inspect…
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)

## Terms introduced here
- `reward form`
- `reward gate`
- `below-gate value`
- `value target`
- `advantage`
