# Remaining Risks And Unproven Claims

## Not Proven
- 10,000,000-tick operation is not established.
- Memory-leak absence is not established.
- Convergence behavior is not established.
- CUDA-path correctness and memory behavior are not established.

## Partial Support
- Checkpoint/replay continuity has stronger support than before: full test suite pass plus a repeated `5`-cycle resume-chain audit with matching signatures.
- Medium-horizon runtime integrity has support: `768` soak ticks with checkpoint validation every `64` ticks completed without invariant or non-finite failures.

## Residual Risks
- RSS grew by about `64.58 MB` across the sampled `64`-to-`512` tick memory window, with a peak above the final value. This does not prove a leak, but it prevents a leak-free claim.
- The executed runtime matrix is intentionally selective, not exhaustive. High-risk CPU/headless surfaces were exercised; CUDA and very long-duration regimes were not.
