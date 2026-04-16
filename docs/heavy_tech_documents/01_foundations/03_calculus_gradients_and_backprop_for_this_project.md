# Calculus, Gradients, and Backprop for This Project

> Scope: Explain the derivative ideas behind policy optimization, value fitting, clipping, and gradient-based neural parameter updates in Tensor Crypt.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- derivatives as sensitivity
- chain rule
- loss surfaces
- policy and value losses
- gradient clipping
- why AMP and optimizer state matter

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Derivatives as sensitivity

A derivative answers a small-change question: if a parameter changes a little, how does the output or loss change? Neural training uses many such sensitivities simultaneously.

## 2. Chain rule

The policy network is a composition of layers. The chain rule lets a gradient flow from:
- loss
- to logits and values
- to hidden activations
- to weights and biases in earlier layers

Backpropagation is simply efficient repeated use of the chain rule.

## 3. Two losses, one trunk

A policy/value architecture often shares a trunk and splits into:
- an actor head producing logits
- a critic head producing values

The repository’s bloodline brain follows this general pattern. The actor supports action sampling and log-prob evaluation. The critic supports return and advantage learning.

## 4. Why clipping exists

PPO does not allow unconstrained policy jumps. It clips the policy-ratio objective so that one update step does not move too far from the data-collecting policy. This is a stability device, not a guarantee of perfect behavior.

## 5. Gradient clipping

Very large gradients can destabilize optimization. Gradient norm clipping limits the overall update magnitude. The repository logs gradient norm and treats non-finite gradient situations as errors rather than as tolerable noise.

## 6. Optimizer state matters

An optimizer such as Adam does not only store parameters. It also stores moving statistics about past gradients. Resuming training without that state changes the optimization trajectory. This is why optimizer state continuity is documented as checkpoint-sensitive.


## Why this matters for Tensor Crypt
This background makes the PPO chapter legible: policy loss, value loss, entropy, gradient norm, optimizer continuity, and AMP scaler state all depend on gradient-based optimization rather than on custom repository magic.

## Read next
- [Linear algebra, tensors, shapes, and batching](04_linear_algebra_tensors_shapes_and_batching.md)
- [PPO from first principles to UID-owned rollouts](07_ppo_from_first_principles_to_uid_owned_rollouts.md)
- [PPO buffers, bootstrap, and update cadence](../04_learning/01_ppo_buffers_bootstrap_and_update_cadence.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
