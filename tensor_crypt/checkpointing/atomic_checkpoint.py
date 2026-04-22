"""Atomic checkpoint file-set helpers for Tensor Crypt.

The publish path writes bundle and manifest temp files into the target
directory and then promotes them with ``os.replace``. Keeping the temp files on
the final filesystem preserves atomic rename semantics and avoids cross-device
publish races.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from ..config_bridge import cfg


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(config_snapshot: dict) -> str:
    payload = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def manifest_path_for(bundle_path: str | Path) -> Path:
    bundle_path = Path(bundle_path)
    suffix = bundle_path.suffix or cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX
    return bundle_path.with_suffix(f"{suffix}{cfg.CHECKPOINT.MANIFEST_FILENAME_SUFFIX}")


def latest_pointer_path_for(bundle_path: str | Path) -> Path:
    bundle_path = Path(bundle_path)
    return bundle_path.parent / cfg.CHECKPOINT.LATEST_POINTER_FILENAME


def load_latest_checkpoint_pointer(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / cfg.CHECKPOINT.LATEST_POINTER_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Latest checkpoint pointer does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        pointer = json.load(handle)
    return pointer


def resolve_latest_checkpoint_bundle(path: str | Path) -> Path:
    pointer_path = Path(path)
    if pointer_path.is_dir():
        pointer_path = pointer_path / cfg.CHECKPOINT.LATEST_POINTER_FILENAME
    pointer = load_latest_checkpoint_pointer(pointer_path)

    raw_bundle_path = Path(pointer["checkpoint_path"])
    if raw_bundle_path.is_absolute():
        bundle_path = raw_bundle_path
    elif raw_bundle_path.exists():
        # Older or relative-root pointers may already be resolvable from the
        # current working directory. Prefer that exact path over rebasing it
        # onto the pointer directory, which can duplicate parent segments.
        bundle_path = raw_bundle_path.resolve()
    else:
        bundle_path = (pointer_path.parent / raw_bundle_path).resolve()

    manifest = validate_checkpoint_file_set(bundle_path)
    if int(pointer.get("tick", -1)) != int(manifest["tick"]):
        raise ValueError("Latest checkpoint pointer tick does not match manifest")

    pointer_checksum = pointer.get("bundle_sha256")
    manifest_checksum = manifest.get("checksums", {}).get("bundle_sha256")
    if pointer_checksum and manifest_checksum and pointer_checksum != manifest_checksum:
        raise ValueError("Latest checkpoint pointer checksum does not match manifest")

    pointer_bytes = pointer.get("bundle_bytes")
    manifest_bytes = manifest.get("artifact_sizes", {}).get("bundle_bytes")
    if pointer_bytes is not None and manifest_bytes is not None and int(pointer_bytes) != int(manifest_bytes):
        raise ValueError("Latest checkpoint pointer size does not match manifest")

    return bundle_path


def build_checkpoint_manifest(bundle: dict, bundle_path: str | Path) -> dict:
    bundle_path = Path(bundle_path)
    ppo_state = bundle.get("ppo_state", {})
    checksum = _sha256_file(bundle_path) if cfg.CHECKPOINT.CHECKSUM_ENABLED else None
    config_snapshot = bundle.get("config_snapshot", {})
    resume_contract = bundle.get("metadata", {}).get("resume_contract", {})
    exact_capability = resume_contract.get("exact_resume_capability", {})
    return {
        "checkpoint_schema_version": int(bundle["checkpoint_schema_version"]),
        "schema_versions": dict(bundle.get("schema_versions", {})),
        "tick": int(bundle["engine_state"]["tick"]),
        "timestamp_unix": time.time(),
        "active_uid_count": int(len(bundle["brain_state_by_uid"])),
        "artifact_filenames": {
            "bundle": bundle_path.name,
        },
        "artifact_sizes": {
            "bundle_bytes": int(bundle_path.stat().st_size),
        },
        "checksums": {
            "bundle_sha256": checksum,
        },
        "config_fingerprint": _config_fingerprint(config_snapshot),
        "catastrophe_state_present": bundle["engine_state"].get("catastrophe_state") is not None,
        "rng_state_present": bundle.get("rng_state") is not None,
        "optimizer_state_present": bool(ppo_state.get("optimizer_state_by_uid")),
        "buffer_state_present": bool(ppo_state.get("buffer_state_by_uid")),
        "resume_taxonomy_version": resume_contract.get("resume_taxonomy_version"),
        "exact_resume_capable": bool(exact_capability.get("exact_resume_capable", False)),
        "contract_hashes": dict(resume_contract.get("contract_hashes", {})),
    }


def validate_checkpoint_file_set(bundle_path: str | Path) -> dict:
    """Validate the published file set exactly as operators and resume paths will observe it."""
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Checkpoint bundle file does not exist: {bundle_path}")

    manifest_path = manifest_path_for(bundle_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest is missing: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    artifact_filenames = manifest.get("artifact_filenames", {})
    expected_bundle_name = artifact_filenames.get("bundle")
    if cfg.CHECKPOINT.STRICT_DIRECTORY_STRUCTURE_VALIDATION and expected_bundle_name != bundle_path.name:
        raise ValueError(
            f"Checkpoint manifest references bundle '{expected_bundle_name}', expected '{bundle_path.name}'"
        )

    bundle_bytes = manifest.get("artifact_sizes", {}).get("bundle_bytes")
    if bundle_bytes is not None and int(bundle_bytes) != int(bundle_path.stat().st_size):
        raise ValueError("Checkpoint bundle size mismatch")

    checksum = manifest.get("checksums", {}).get("bundle_sha256")
    if checksum and cfg.CHECKPOINT.CHECKSUM_ENABLED:
        actual = _sha256_file(bundle_path)
        if actual != checksum:
            raise ValueError("Checkpoint checksum mismatch")

    return manifest


def atomic_save_checkpoint_files(bundle_path: str | Path, bundle: dict) -> dict:
    """Atomically publish a checkpoint bundle, manifest, and latest pointer in publish order."""
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(bundle_path)

    bundle_fd, tmp_bundle_name = tempfile.mkstemp(
        dir=str(bundle_path.parent),
        prefix=cfg.CHECKPOINT.TEMPFILE_PREFIX,
        suffix=f"_{bundle_path.name}.tmp",
    )
    manifest_fd, tmp_manifest_name = tempfile.mkstemp(
        dir=str(bundle_path.parent),
        prefix=cfg.CHECKPOINT.TEMPFILE_PREFIX,
        suffix=f"_{manifest_path.name}.tmp",
    )
    os.close(bundle_fd)
    os.close(manifest_fd)
    tmp_bundle_path = Path(tmp_bundle_name)
    tmp_manifest_path = Path(tmp_manifest_name)

    try:
        torch.save(bundle, tmp_bundle_path)
        # The manifest hashes the temporary payload bytes but must name the final
        # published artifact so strict directory validation remains stable after rename.
        manifest = build_checkpoint_manifest(bundle, tmp_bundle_path)
        manifest["artifact_filenames"]["bundle"] = bundle_path.name
        _write_json(tmp_manifest_path, manifest)

        os.replace(tmp_bundle_path, bundle_path)
        os.replace(tmp_manifest_path, manifest_path)
    finally:
        tmp_bundle_path.unlink(missing_ok=True)
        tmp_manifest_path.unlink(missing_ok=True)

    validated_manifest = validate_checkpoint_file_set(bundle_path)

    if cfg.CHECKPOINT.WRITE_LATEST_POINTER:
        pointer_path = latest_pointer_path_for(bundle_path)
        temp_pointer = pointer_path.with_name(f"{pointer_path.name}.tmp")
        relative_bundle_path = os.path.relpath(bundle_path, start=pointer_path.parent)
        relative_manifest_path = os.path.relpath(manifest_path, start=pointer_path.parent)
        _write_json(
            temp_pointer,
            {
                "checkpoint_path": str(relative_bundle_path),
                "manifest_path": str(relative_manifest_path),
                "tick": int(bundle["engine_state"]["tick"]),
                "checkpoint_schema_version": int(bundle["checkpoint_schema_version"]),
                "active_uid_count": int(validated_manifest["active_uid_count"]),
                "bundle_bytes": int(validated_manifest.get("artifact_sizes", {}).get("bundle_bytes", bundle_path.stat().st_size)),
                "bundle_sha256": validated_manifest.get("checksums", {}).get("bundle_sha256"),
                "config_fingerprint": validated_manifest.get("config_fingerprint"),
                "resume_taxonomy_version": validated_manifest.get("resume_taxonomy_version"),
                "exact_resume_capable": bool(validated_manifest.get("exact_resume_capable", False)),
            },
        )
        os.replace(temp_pointer, pointer_path)

    return validated_manifest


def load_checkpoint_bundle(bundle_path: str | Path) -> tuple[dict, dict]:
    bundle_path = Path(bundle_path)
    manifest = validate_checkpoint_file_set(bundle_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return bundle, manifest
