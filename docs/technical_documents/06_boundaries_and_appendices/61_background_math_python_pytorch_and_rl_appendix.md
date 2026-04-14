# D61 - Background Math, Python, PyTorch, and RL Appendix

## Purpose

This appendix supplies optional background for readers who want brief conceptual refreshers while reading the implementation-oriented chapters.

## Scope Boundary

This appendix is background only. It should not be cited as the source of implementation truth for Tensor Crypt.

## Evidence Basis

This appendix draws on general background needed to read the repository and on the implementation-facing chapters that anchor Tensor Crypt-specific terminology:

- [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)

## Topics Covered

### Python modules and dataclasses

Tensor Crypt uses Python modules for subsystem organization and dataclasses for the main config tree and several structured runtime records.

### Tensors and shapes

Tensors are ordered numeric arrays. In this repository they represent world state, batched observations, logits, values, and rollout data. Shape order is semantically meaningful.

### Categorical action sampling

The current live policy path samples from categorical distributions constructed from logits. In background terms, logits are unnormalized scores that become a discrete sampling distribution.

### Value estimation

The value head predicts a scalar-like estimate used by PPO-style training logic. It is distinct from the action distribution.

### Advantage-style training intuition

PPO-style training generally relies on returns, value baselines, and policy-ratio constraints. Tensor Crypt implements a specific PPO substrate, but this appendix does not claim every PPO variant or every RL training refinement.

### Determinism background

In practical simulation work, determinism usually depends on explicit seeding, stable execution order, and careful handling of serialized state. Tensor Crypt includes explicit probe surfaces for these concerns, but the general principle remains background here.

## Reading Guidance

Readers who need exact Tensor Crypt contracts should return to:

- [D21](../02_system_foundations/21_python_pytorch_tensors_and_simulation_foundations.md)
- [D40](../04_perception_brains_and_learning/40_observation_schema_rays_and_feature_construction.md)
- [D42](../04_perception_brains_and_learning/42_uid_owned_ppo_rollouts_rewards_and_updates.md)
- [D51](../05_artifacts_validation_and_viewer/51_checkpointing_atomic_publish_resume_and_schema_safety.md)
