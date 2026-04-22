# Documentation Evidence Policy and Corpus Conventions

> Scope: Define the truth model, update discipline, terminology governance, and structural conventions that control the entire documentation corpus.

## Who this document is for
Maintainers, auditors, and any reader who wants to understand how the corpus distinguishes hard repository fact from background explanation.

## What this document covers
- the evidence taxonomy used throughout the corpus
- how canonical versus compatibility surfaces are labeled
- how unknowns are handled
- how diagrams, tables, and cross-links should be maintained
- what kinds of code changes require documentation updates

## What this document does not cover
- subsystem mechanics in detail
- complete user instructions for operating the viewer

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Source-of-truth order used by this corpus

This corpus is maintained against the following precedence order:

1. checked-in repository files visible in the workspace
2. repository tests, utility scripts, and packaging metadata
3. official external sources, only where narrowly needed
4. general background knowledge, only for foundations content

The present corpus build is grounded in a directly inspectable repository tree. Implementation claims are therefore written against checked-in source, tests, scripts, and repository metadata rather than against detached secondary material.

## 2. Evidence taxonomy

### 2.1 Repository-evidenced fact
A statement grounded directly in checked-in source files, configuration comments, control flow, docstrings, tests, or helper surfaces.

### 2.2 Repository-grounded interpretation
A higher-level explanation derived from several repository facts without adding new mechanics.

### 2.3 Background explanatory material
General mathematics, machine learning, reinforcement learning, systems, or simulation knowledge that helps a reader understand the repository.

### 2.4 Recommended practice
Conservative operational or maintainer guidance derived from the repository’s safety posture.

### 2.5 Unknown or unverified
Anything not established by the checked-in repository or explicitly separated background context.

> **Rule**
> Background explanation must never be written as repository fact. Unknowns must remain unknown.

## 3. Canonical versus compatibility labeling

The corpus uses the following labels consistently.

| Label | Meaning | Documentation consequence |
| --- | --- | --- |
| Canonical implementation surface | The owning implementation under `tensor_crypt` | Treated as the architecture center of gravity |
| Public entry surface | A repository-root launch or public import path | Documented as user-facing, but not as subsystem owner unless the code proves ownership |
| Compatibility wrapper | Thin re-export or legacy import bridge | Documented explicitly as non-canonical |
| Guarded compatibility surface | A config field or selector that exists but is runtime-constrained | Listed, but not narrated as an active alternate behavior |
| Currently unread | A documented field present in config structure but not read on the current runtime or validation path | Included for honesty; not narrated as active behavior |

## 4. Unknown handling policy

> **Preferred failure mode**
> Omit unsupported claims rather than fill gaps with plausible prose.

If the repository does not establish something, the document should say one of the following:
- the current runtime does not establish this
- the public field remains present, but no live read path is documented here
- this boundary is inferred conservatively from the current implementation
- this is background explanation rather than repository fact

## 5. Diagram drift policy

Diagrams are intentionally stored as documentation-native assets under `docs/heavy_tech_documents/assets/`. Every diagram must declare:
- owning document
- abstraction level
- intentional omissions

Diagram edits are required whenever:
- module ownership changes
- checkpoint ordering changes
- observation layout changes
- family topology rules change
- operator-visible controls change materially

## 6. Terminology governance

The glossary is authoritative for high-load terms such as `UID`, `slot`, `canonical observation contract`, `bloodline family`, `bootstrap`, `latest pointer`, `compatibility wrapper`, and `currently unread`. First heavy use of a term in a document should match glossary wording closely.

## 7. Document structure conventions

Every substantial document in the corpus should have:
- a title
- a one-sentence scope declaration
- explicit reader targeting
- coverage and non-coverage blocks
- prerequisite reading where relevant
- definitions before dense use of specialized terms
- mechanism and invariant sections
- a closing navigation block

## 8. Markdown conventions

The corpus favors portable GitHub-flavored Markdown:
- standard headings
- tables
- fenced code blocks
- mermaid diagrams only where they materially help
- `details` blocks only for dense lookup sections
- quote blocks as definition, invariant, and consequence panels

## 9. Update obligations after code changes

The following changes should trigger documentation review immediately.

| Code change type | Documentation surfaces to re-audit |
| --- | --- |
| Entry-surface change | system identity docs, quickstart, module index |
| Observation layout or feature order change | observation docs, brain docs, PPO docs, checkpoint docs, glossary |
| Brain family or topology change | traits/bloodlines docs, inference docs, checkpoint docs, compatibility ledger |
| Respawn or lineage rule change | registry docs, reproduction docs, game manual, telemetry docs |
| Checkpoint schema or manifest change | checkpoint docs, schema ledger, troubleshooting, validation docs |
| Viewer control change | viewer manual, quickstart, troubleshooting, game manual |
| New catastrophe mode or scheduler rule | catastrophe docs, game manual, validation docs |
| Config taxonomic change | config taxonomy doc, config reference index, compatibility ledger |

## 10. Failure modes the corpus must resist

- flattening canonical and compatibility surfaces into one story
- treating every config field as active
- confusing slot lookup with UID ownership
- treating checkpointing as a simple save/load convenience
- overclaiming the experimental family-vmap path as a guaranteed performance win
- treating validation harnesses as decorative rather than credibility infrastructure

## Read next
- [Glossary, notation, and schema legend](02_glossary_notation_and_schema_legend.md)
- [Runtime config taxonomy and knob safety](../02_system/03_runtime_config_taxonomy_and_knob_safety.md)
- [Schema versions and compatibility surfaces](../07_reference/01_schema_versions_and_compatibility_surfaces.md)

## Related reference
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## If debugging this, inspect…
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)
- [Module reference index](../07_reference/02_module_reference_index.md)

## Terms introduced here
- `repository-evidenced fact`
- `repository-grounded interpretation`
- `background explanatory material`
- `guarded compatibility surface`
- `diagram drift`
