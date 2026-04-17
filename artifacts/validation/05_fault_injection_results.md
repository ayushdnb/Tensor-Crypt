# Fault Injection Results

| Surface | Injection | Expected behavior | Result |
| --- | --- | --- | --- |
| Checkpoint manifest absence | Delete `.pt.manifest.json` after save | Strict load rejects missing manifest | Pass |
| Checkpoint bundle corruption | Append bytes after save | Load rejects checksum mismatch | Pass |
| Latest-pointer checksum corruption | Rewrite `bundle_sha256` in `latest_checkpoint.json` | Pointer resolution rejects checksum mismatch | Pass |
| Atomic bundle write failure | Monkeypatch `torch.save` to raise during atomic publish | Save fails loudly and temp files are cleaned up | Pass |
| Relative latest-pointer paths | Save checkpoints under a relative run directory and load from checkpoint dir | Resolver returns the published bundle instead of synthesizing an invalid doubled path | Fixed and covered |

## Key Fix
- Relative checkpoint paths in latest pointers were being resolved incorrectly. The resolver now prefers an already-existing relative path before rebasing it onto the pointer directory, and new pointers are written relative to the pointer directory.
