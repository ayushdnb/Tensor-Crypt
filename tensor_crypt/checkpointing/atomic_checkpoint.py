"""Atomic checkpoint file-set helpers for Tensor Crypt."""

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


def build_checkpoint_manifest(bundle: dict, bundle_path: str | Path) -> dict:
    bundle_path = Path(bundle_path)
    ppo_state = bundle.get("ppo_state", {})
    checksum = _sha256_file(bundle_path) if cfg.CHECKPOINT.CHECKSUM_ENABLED else None
    config_snapshot = bundle.get("config_snapshot", {})
    return {
        "checkpoint_schema_version": int(bundle["checkpoint_schema_version"]),
        "schema_versions": dict(bundle.get("schema_versions", {})),
        "tick": int(bundle["engine_state"]["tick"]),
        "timestamp_unix": time.time(),
        "active_uid_count": int(len(bundle["brain_state_by_uid"])),
        "artifact_filenames": {
            "bundle": bundle_path.name,
        },
        "checksums": {
            "bundle_sha256": checksum,
        },
        "config_fingerprint": _config_fingerprint(config_snapshot),
        "catastrophe_state_present": bundle["engine_state"].get("catastrophe_state") is not None,
        "rng_state_present": bundle.get("rng_state") is not None,
        "optimizer_state_present": bool(ppo_state.get("optimizer_state_by_uid")),
        "buffer_state_present": bool(ppo_state.get("buffer_state_by_uid")),
    }


def validate_checkpoint_file_set(bundle_path: str | Path) -> dict:
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

    checksum = manifest.get("checksums", {}).get("bundle_sha256")
    if checksum and cfg.CHECKPOINT.CHECKSUM_ENABLED:
        actual = _sha256_file(bundle_path)
        if actual != checksum:
            raise ValueError("Checkpoint checksum mismatch")

    return manifest


def atomic_save_checkpoint_files(bundle_path: str | Path, bundle: dict) -> dict:
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(bundle_path)

    with tempfile.TemporaryDirectory(
        dir=str(bundle_path.parent),
        prefix=cfg.CHECKPOINT.TEMPFILE_PREFIX,
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        tmp_bundle_path = tmp_dir / bundle_path.name
        tmp_manifest_path = tmp_dir / manifest_path.name

        torch.save(bundle, tmp_bundle_path)
        manifest = build_checkpoint_manifest(bundle, tmp_bundle_path)
        _write_json(tmp_manifest_path, manifest)

        os.replace(tmp_bundle_path, bundle_path)
        os.replace(tmp_manifest_path, manifest_path)

    if cfg.CHECKPOINT.WRITE_LATEST_POINTER:
        pointer_path = latest_pointer_path_for(bundle_path)
        temp_pointer = pointer_path.with_name(f"{pointer_path.name}.tmp")
        _write_json(
            temp_pointer,
            {
                "checkpoint_path": str(bundle_path),
                "manifest_path": str(manifest_path),
                "tick": int(bundle["engine_state"]["tick"]),
                "checkpoint_schema_version": int(bundle["checkpoint_schema_version"]),
            },
        )
        os.replace(temp_pointer, pointer_path)

    return validate_checkpoint_file_set(bundle_path)


def load_checkpoint_bundle(bundle_path: str | Path) -> tuple[dict, dict]:
    bundle_path = Path(bundle_path)
    manifest = validate_checkpoint_file_set(bundle_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return bundle, manifest
