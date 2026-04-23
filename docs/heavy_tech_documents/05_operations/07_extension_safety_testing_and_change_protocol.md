# Extension Safety, Testing, and Change Protocol

> Scope: Describe which kinds of code changes are high risk, which adjacent documents and validations must be re-audited, and how to avoid silent compatibility damage.

## Who this document is for
Operators, maintainers, and technical readers who need actionable procedures and conservative operational guidance.

## What this document covers
- high-risk change classes
- required re-audits
- validation focus by subsystem
- documentation sync obligations

## What this document does not cover
- deep subsystem theory unless operationally necessary
- speculative performance claims

## Prerequisite reading
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)
- [Validation harnesses](05_validation_determinism_resume_consistency_and_soak.md)

## 1. Change classes

| Change class | Risk level | Why |
| --- | --- | --- |
| observation feature order or count | severe | checkpoint-visible and brain-facing |
| family topology or family order | severe | affects checkpoint, inference bucketing, and UI identity |
| UID/slot mapping logic | severe | identity and PPO ownership risk |
| checkpoint schema or manifest logic | severe | resume and publication risk |
| reward gating or PPO buffer logic | high | training and validation integrity risk |
| catastrophe scheduler or overlay rules | high | cross-cutting runtime behavior risk |
| viewer keybindings or inspector fields | medium | operator-document drift risk |
| telemetry schema additions | medium | downstream analysis and reference drift |

## 2. Mandatory re-audit map

- observation changes → observation docs, brain docs, PPO docs, checkpoint docs, glossary, validation probes
- family changes → bloodline docs, inference docs, checkpoint docs, viewer docs, validation probes
- registry changes → lifecycle docs, learning ownership docs, checkpoint docs, telemetry docs, validation probes
- respawn changes → reproduction docs, game manual, telemetry docs, validation docs, validation probes
- checkpoint changes → checkpoint docs, schema ledger, troubleshooting, validation docs, validation probes

## 3. Documentation sync rule

If code changes a compatibility-sensitive surface, update the relevant docs **in the same change set** whenever possible. The repository is explicit enough that documentation lag can create false beliefs about ownership and restore safety.


## Read next
- [Troubleshooting and failure atlas](08_troubleshooting_and_failure_atlas.md)
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Related reference
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)

## If debugging this, inspect…
- [Module reference index](../07_reference/02_module_reference_index.md)

## Terms introduced here
- `high-risk change`
- `re-audit`
- `doc-sync obligation`
