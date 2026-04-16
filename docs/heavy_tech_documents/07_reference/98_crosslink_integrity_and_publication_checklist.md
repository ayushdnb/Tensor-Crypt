# Cross-Link Integrity and Publication Checklist

> Scope: Record the corpus-level publication checks run before packaging, including link integrity, navigation completeness, audience coverage, and known cautions.

## Who this document is for
Auditors and maintainers performing a final publication pass.

## What this document covers
- link sanity summary
- coverage checklist
- publication cautions
- final packaging checks

## What this document does not cover
- per-file prose explanations

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)

## 1. Link integrity pass

A local scan over markdown links in the generated corpus was used to verify internal relative links. Any broken links detected during generation were repaired before final packaging.

## 2. Publication checklist

| Check | Status | Note |
| --- | --- | --- |
| Front door exists | pass | `docs/00_program/00_documentation_index_and_reading_guide.md` |
| Evidence policy exists | pass | governance layer present |
| Glossary exists | pass | terminology and notation stabilized |
| Foundations layer exists | pass | study ladder present |
| System layer exists | pass | canonical versus compatibility boundaries documented |
| Mechanics layer exists | pass | world, registry, observation, respawn, catastrophe chapters present |
| Learning layer exists | pass | PPO ownership, inference path, and checkpoint-visible state documented |
| Operations layer exists | pass | quickstart, viewer, outputs, checkpoints, validation, benchmarking, troubleshooting present |
| Game manual exists | pass | readable rulebook present |
| Reference layer exists | pass | config, schema, module, FAQ, audit artifacts present |
| Diagram assets exist | pass | system, mechanics, learning, operations assets included |
| Audit ledger exists | pass | generation ledger included |
| Explicit truth-gap note exists | pass | uncertainty recorded transparently |
| Unsupported empirical claims avoided | pass | corpus remains conservative |
| ZIP packaging support artifacts present | pass | corpus manifest and build manifest included |

## 3. Reader-route integrity

The corpus supports:
- beginner path
- technical architecture path
- operator path
- maintainer/auditor path
- direct reference lookup path

## 4. Known cautions

- This corpus was generated from the attached specification and code dump rather than from a directly inspectable live repository tree.
- The documentation therefore remains conservative about live-file drift, README breadth, and non-Python assets not visible in the workspace.
## Read next
- [Corpus manifest](../07_reference/99_corpus_manifest.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Related reference
- [Generation ledger and audit trail](../00_program/99_generation_ledger_and_audit_trail.md)

## If debugging this, inspect…
- [Module reference index](02_module_reference_index.md)

## Terms introduced here
- `publication checklist`
- `integrity pass`
- `reader-route integrity`
