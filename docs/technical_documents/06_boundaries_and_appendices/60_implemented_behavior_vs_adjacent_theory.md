# D60 - Implemented Behavior vs Adjacent Theory

## Purpose

This document separates repository-verified behavior from explanatory background. It exists to prevent the bundle from overstating what Tensor Crypt currently implements.

## Scope Boundary

This chapter does not add new subsystem behavior. It summarizes the distinction between code-grounded truth and nearby conceptual language used elsewhere in the bundle.

## Evidence Basis

The implemented side of each boundary below is grounded in the current repository code and tests. The adjacent-theory side is included only when it helps interpret the implementation.

## Boundary Table

| Topic | Repository-verified implementation | Adjacent background that must remain labeled |
|---|---|---|
| Identity | UID-ledger substrate with reusable slots | generic discussion of identity systems or entity-component design |
| Reproduction | binary-parented birth path with explicit parent roles | broader evolutionary theory or alternative reproduction schemes |
| Observation | canonical rays, self, and context tensors with fixed feature order | general sensor-design theory or arbitrary feature-engineering alternatives |
| Brains | five configured family architectures plus an optional same-family vmap optimization path | broader neural-architecture design space |
| PPO | UID-owned buffers and optimizers with the current reward and bootstrap logic | broader RL algorithm comparisons or alternate policy-gradient schemes |
| Catastrophes | named catastrophe roster with implemented field and runtime modifiers | generic disaster-model literature or unimplemented catastrophe types |
| Checkpointing | full-state runtime checkpoint capture, validation, and atomic publish path | generic persistence design patterns not present in the code |
| Validation | determinism, resume, catastrophe, save-load-save, benchmark, and soak harnesses | universal guarantees of determinism, compatibility, or performance |

## Frequent Overstatement Risks

The current bundle should explicitly avoid the following overstatements:

- treating compatibility wrappers as the canonical implementation
- treating dormant config fields as active runtime switches
- treating validation probes as universal correctness guarantees
- treating background RL or PyTorch explanations as evidence that the repository implements every nearby option
- treating the existence of checkpoint code as proof of cross-version compatibility

## Safe Writing Pattern

When evidence is strong, state the behavior directly.

When evidence is limited, use wording such as:

- "the current implementation defines"
- "the active runtime validates"
- "the public field remains present, but the current runtime does not read it"
- "background only"

## Practical Use

This boundary chapter should be consulted whenever a later chapter is tempted to:

- broaden a supported-value set beyond what runtime validation accepts
- describe a planning prompt as a shipped feature
- infer public guarantees from a test harness alone
- import theory into the implementation narrative without a code path

## Cross References

- bundle terminology and status labels: [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- contributor truth rules: [D63](./63_contributor_documentation_truth_contract.md)
