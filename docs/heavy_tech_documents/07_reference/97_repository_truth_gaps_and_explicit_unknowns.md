# Repository Truth Gaps and Explicit Unknowns

> Scope: Record the main limits and unresolved uncertainties that remain even after direct repository inspection during corpus revision.

## Who this document is for
Auditors, maintainers, and readers who want a transparent statement of what this corpus could and could not verify directly.

## What this document covers
- repository inspection boundary
- unknowns
- conservative omissions
- where further empirical or migration verification would still help

## What this document does not cover
- excuses for weak documentation; the rest of the corpus remains strong where evidence exists

## Prerequisite reading
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)

## 1. Repository inspection boundary

The corpus is grounded in the checked-in repository tree, public validation helpers, and documentation. Full-source validation before publication also inspected the test and script surfaces that are not tracked in this public branch.

## 2. Consequences

The corpus therefore:
- documents checked-in behavior rather than inferred future intent
- treats currently unread config fields conservatively
- does not promote reserved asset directories into reviewed figure claims
- does not promote validation coverage into research outcomes
- does not extrapolate historical benchmark or migration results beyond checked-in artifacts

## 3. Specific unknowns

- the intended future use of currently unread fields that remain in `runtime_config.py`
- cross-version checkpoint compatibility beyond the explicitly versioned and validated current path
- empirical learning quality, benchmark leadership, or scientific outcomes not recorded by checked-in artifacts
- unpublished or not-yet-checked-in figures, datasets, or operator notebooks
- historical repository states not represented in the current tree

## 4. Conservative handling choices

To remain trustworthy, the corpus:
- prefers canonical package explanations over speculative root-level narratives
- marks compatibility and currently unread surfaces carefully
- treats benchmark and validation coverage as infrastructure, not as proof of scientific outcomes
- avoids claiming performance leadership, learning results, or undocumented future migrations
## Read next
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Corpus manifest](../07_reference/99_corpus_manifest.md)

## If debugging this, inspect…
- [Module reference index](02_module_reference_index.md)

## Terms introduced here
- `truth gap`
- `inspection boundary`
- `conservative omission`
