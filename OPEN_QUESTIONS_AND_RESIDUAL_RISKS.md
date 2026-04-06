# Open Questions And Residual Risks

## Open Questions
- Should checkpoint resume be expected to continue telemetry writers and open-life ledgers seamlessly, or is checkpoint restore intentionally scoped to simulation/learning state only?
- What exact semantics should `cfg.CHECKPOINT.STRICT_SCHEMA_VALIDATION` enforce? It is declared but still not wired to distinct behavior.
- What exact semantics should `cfg.VALIDATION.STRICT_TELEMETRY_SCHEMA_WRITES` enforce? The current telemetry writer still aligns schemas permissively.
- If catastrophe overlap is enabled intentionally, should death attribution support multiple catastrophe IDs instead of the current first-active-event simplification?
- Is process-global RNG ownership an accepted design constraint, or should the runtime move to per-runtime RNG streams for true in-process isolation?

## Residual Risks
- Very long-run CPU/GPU stability beyond the 128-tick soak remains unproven.
- Logger continuation after checkpoint restore remains unsupported.
- Some config knobs remain declarative rather than behaviorally authoritative.
- The audit harness is now materially stronger, but it is still a targeted probe suite rather than a substitute for full-scale benchmark/eval runs.
- The repository still depends on interpreter/environment consistency. In this workspace, the system Python 3.14 interpreter exhibited checkpoint-serialization/temp-directory issues that were not present in the Python 3.10 test environment.
