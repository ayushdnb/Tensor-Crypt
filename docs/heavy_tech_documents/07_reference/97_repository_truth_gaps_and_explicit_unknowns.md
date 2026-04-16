# Repository Truth Gaps and Explicit Unknowns

> Scope: Record the main limits and unresolved uncertainties that remain because only the attached specification and code dump were available directly in the workspace during corpus generation.

## Who this document is for
Auditors, maintainers, and readers who want a transparent statement of what this corpus could and could not verify directly.

## What this document covers
- workspace limits
- unknowns
- conservative omissions
- where live-repo verification would still help

## What this document does not cover
- excuses for weak documentation; the rest of the corpus remains strong where evidence exists

## Prerequisite reading
- [Evidence policy](../00_program/01_documentation_evidence_policy_and_corpus_conventions.md)

## 1. Workspace limitation

The workspace exposed the attached master specification and the attached `evolution.txt` source dump directly. It did **not** expose a browsable live repository tree under `tensor_crypt/` during corpus generation.

## 2. Consequences

The corpus therefore:
- treats the code dump as the main repository evidence substrate
- treats comment-labeled “currently unread / effectively dead” config fields conservatively
- does not claim to have verified every on-disk non-Python artifact that may exist in a live repository
- avoids making claims about README structure, pyproject contents, or packaging metadata beyond what the dump itself or its tests reveal

## 3. Specific unknowns

- the exact final repository directory layout outside the dumped Python files
- the full content of README and dependency files, except where tests referenced expected strings
- any empirical run results or benchmark histories not included in the dump
- any non-Python assets that may exist in a live repository but were not visible here
- whether live files have drifted from the uploaded dump after the dump was created

## 4. Conservative handling choices

To remain trustworthy, the corpus:
- prefers canonical package explanations over speculative root-level narratives
- marks compatibility and dead config surfaces carefully
- treats benchmark and validation coverage as infrastructure, not as proof of scientific outcomes
- avoids claiming performance or learning results
## Read next
- [Cross-link integrity and publication checklist](../07_reference/98_crosslink_integrity_and_publication_checklist.md)
- [Corpus manifest](../07_reference/99_corpus_manifest.md)

## Related reference
- [Generation ledger and audit trail](../00_program/99_generation_ledger_and_audit_trail.md)

## If debugging this, inspect…
- [Module reference index](02_module_reference_index.md)

## Terms introduced here
- `truth gap`
- `workspace limitation`
- `conservative omission`
