# Repository Split Manifest

## Line

- Branch: `release/single-brain-vmap`
- Tag target: `single-brain-vmap/v0.9.0`
- Package version: `0.9.0`
- Identity: self-centric single-family runtime with guarded vmap-capable inference

## Completed Locally

- Isolated worktree created from hardened base commit `049fdfe`.
- Self-centric implementation commits were applied on top of the hardened base.
- Public startup and headless validation scripts apply the branch preset.
- Branch README, architecture docs, release notes, validation artifacts, and durable working artifacts are present.
- Public authoring-input and phase-report residue was removed.
- Full pytest and benchmark smoke validation passed.

## Available Remote Publication

Push this branch and tag to the existing remote:

```powershell
git push -u origin release/single-brain-vmap
git push origin single-brain-vmap/v0.9.0
```

## Optional Repository Extraction

If a separate public repository is created for this line, publish this branch as that repository's main branch:

```powershell
git remote add single-brain-vmap-origin <single-brain-vmap-repository-url>
git push single-brain-vmap-origin release/single-brain-vmap:main
git push single-brain-vmap-origin single-brain-vmap/v0.9.0
```
