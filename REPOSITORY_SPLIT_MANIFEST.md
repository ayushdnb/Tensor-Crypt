# Repository Split Manifest

## Line

- Branch: `release/five-brain`
- Tag target: `five-brain/v1.0.0`
- Package version: `1.0.0`
- Identity: stable five-family bloodline simulation runtime

## Completed Locally

- Isolated worktree created from hardened base commit `049fdfe`.
- Public startup preserves five-family defaults.
- Branch README, architecture docs, release notes, validation artifacts, and durable working artifacts are present.
- Public authoring-input and phase-report residue was removed.
- Full pytest and benchmark smoke validation passed.

## Available Remote Publication

Push this branch and tag to the existing remote:

```powershell
git push -u origin release/five-brain
git push origin five-brain/v1.0.0
```

## Optional Repository Extraction

If a separate public repository is created for this line, publish this branch as that repository's main branch:

```powershell
git remote add five-brain-origin <five-brain-repository-url>
git push five-brain-origin release/five-brain:main
git push five-brain-origin five-brain/v1.0.0
```
