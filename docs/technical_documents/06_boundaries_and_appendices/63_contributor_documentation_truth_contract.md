# D63 - Contributor Documentation Truth Contract

## Purpose

This document governs future edits to the Tensor Crypt technical-document bundle. Its purpose is to keep documentation aligned with repository truth as the code evolves.

## Scope Boundary

This document governs documentation work. It does not define runtime behavior by itself and should not be cited as evidence that a subsystem is implemented.

## Evidence Basis

This contract is derived from the repository's current source, tests, and from the terminology and boundary chapters that stabilize the bundle:

- [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- [D60](./60_implemented_behavior_vs_adjacent_theory.md)

## Required Standard

Future contributors should treat the repository as the source of truth for implementation claims. Documentation must not elevate a configuration field, wrapper import, test helper, or background concept into an implementation guarantee unless the code and tests support that conclusion.

## Required Claim Discipline

Before writing a strong claim, contributors should determine whether it is:

- directly implemented
- evidenced by tests or runtime helpers
- strongly constrained inference from adjacent code
- background only

The stronger the prose, the stronger the required evidence.

## Required Editing Rules

Contributors should:

- verify implementation claims against the current repository
- use the status labels defined in [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- keep compatibility wrappers distinct from canonical implementation owners
- mark currently unread config surfaces explicitly rather than treating them as active
- keep background theory clearly separated from implementation chapters
- repair cross-document contradictions instead of adding new ones

Contributors should not:

- invent performance claims
- invent safety guarantees
- invent compatibility guarantees
- describe guarded fields as open-ended feature switches
- describe validation probes as universal proof

## Required Cross-Checks for Major Edits

For substantial documentation changes, contributors should re-check at least:

- launch and package ownership
- identity and slot-versus-UID language
- observation feature counts and ordering
- brain-family roster and topology assumptions
- checkpoint artifact names and manifest dependencies
- validation-script names and purposes
- viewer-control descriptions

## Required Placement Rules

- published technical chapters remain in the numbered public folders

## Required Honesty About Uncertainty

If the repository evidence is incomplete, contributors should say so directly. Documentation quality is improved by explicit uncertainty and reduced by confident speculation.

## Audit Habit

The preferred maintenance habit is:

1. identify the code owner of the subject
2. identify tests or scripts that exercise it
3. confirm artifact names and config field names exactly
4. update cross references after structural edits
5. run targeted searches for stale terminology and stale paths

## Cross References

- bundle structure: [D00](../00_meta/00_documentation_bundle_index.md)
- reading order: [D01](../00_meta/01_reading_tracks_and_dependency_map.md)
- status labels and terminology: [D02](../00_meta/02_notation_glossary_and_shape_legend.md)
- implementation-versus-background boundary: [D60](./60_implemented_behavior_vs_adjacent_theory.md)
