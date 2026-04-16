# Generation Ledger and Audit Trail

> Scope: Record the source-of-truth precedence used for this corpus build, the completed workstreams, the major documentation decisions, and the main audit findings.

## Who this document is for
Auditors and maintainers who need a terse provenance record rather than a narrative explanation.

## What this document covers
- source precedence
- completed workstreams
- major decisions
- conflicts reconciled
- remaining unknowns
- final audit outcomes

## What this document does not cover
- full prose explanation of every document

## Prerequisite reading
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)

## 1. Source-of-truth precedence actually used

1. live repository files in the workspace, if available
2. attached master specification
3. attached repository code dump
4. official external sources only if narrowly needed
5. background knowledge only for foundations content

### Effective result for this build
No directly inspectable live repository tree was exposed in the workspace. The corpus therefore used:
- the attached specification as the documentation-program contract
- the attached `evolution.txt` dump as the repository evidence substrate

## 2. Completed workstreams

- governance layer
- foundations layer
- system layer
- mechanics layer
- learning layer
- operations layer
- game manual layer
- reference layer
- diagram asset layer
- integrity and publication artifacts
- zip packaging support artifacts

## 3. Major documentation decisions

- treated `tensor_crypt` as canonical and root surfaces as public or compatibility-facing
- preserved the slot-versus-UID distinction throughout
- preserved canonical observation versus legacy fallback distinction
- documented family topology as checkpoint-visible and compatibility-sensitive
- documented checkpointing as a validated substrate rather than a plain `torch.save` convenience
- documented the experimental family-vmap path conservatively

## 4. Conflicts or tensions reconciled

- The specification asked for live-repo precedence, but only the specification and dump were directly inspectable; this was recorded as an explicit truth gap rather than hidden.
- The config surface is large, so the corpus separates narrative taxonomy from lookup-first reference tables.
- The game manual was made more readable than the technical layers without inventing lore or unsupported emergent claims.

## 5. Remaining unknowns

See:
- `docs/07_reference/97_repository_truth_gaps_and_explicit_unknowns.md`

## 6. Final audit outcomes

- canonical-versus-compatibility boundaries: preserved
- slot-versus-UID distinction: preserved
- canonical observation versus legacy fallback: preserved
- family-topology checkpoint sensitivity: preserved
- checkpoint manifest/latest-pointer story: preserved
- validation harnesses documented as credibility infrastructure: preserved
- unsupported empirical claims: intentionally omitted
- final packaging artifacts present: yes

## 7. Build timestamp
- 2026-04-16 08:07 UTC
## Read next
- [Corpus manifest](../07_reference/99_corpus_manifest.md)
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Repository truth gaps and explicit unknowns](../07_reference/97_repository_truth_gaps_and_explicit_unknowns.md)

## Related reference
- [Documentation index and reading guide](00_documentation_index_and_reading_guide.md)

## If debugging this, inspect…
- [Module reference index](../07_reference/02_module_reference_index.md)

## Terms introduced here
- `source precedence`
- `audit trail`
- `truth gap`
