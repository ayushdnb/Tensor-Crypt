import json

import pytest

from tensor_crypt.checkpointing import atomic_checkpoint
from tensor_crypt.checkpointing import resolve_latest_checkpoint_bundle
from tensor_crypt.checkpointing.runtime_checkpoint import (
    capture_runtime_checkpoint,
    load_runtime_checkpoint,
    save_runtime_checkpoint,
    validate_checkpoint_artifacts,
)


def test_atomic_checkpoint_emits_manifest_and_latest_pointer(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=21, agents=6, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "checkpoint_bundle.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)
    manifest = validate_checkpoint_artifacts(checkpoint_path)
    latest_pointer_path = checkpoint_path.parent / "latest_checkpoint.json"

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".pt.manifest.json").exists()
    assert latest_pointer_path.exists()
    assert int(manifest["tick"]) == int(runtime.engine.tick)

    loaded = load_runtime_checkpoint(checkpoint_path)
    assert int(loaded["engine_state"]["tick"]) == int(bundle["engine_state"]["tick"])

def test_checkpoint_checksum_corruption_is_detected(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=22, agents=6, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "corruptible_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)

    with checkpoint_path.open("ab") as handle:
        handle.write(b"checkpoint-corruption")

    with pytest.raises(ValueError):
        load_runtime_checkpoint(checkpoint_path)

def test_latest_pointer_references_published_bundle(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=23, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "latest_pointer_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)

    pointer_path = checkpoint_path.parent / "latest_checkpoint.json"
    with pointer_path.open("r", encoding="utf-8") as handle:
        pointer = json.load(handle)

    assert pointer["checkpoint_path"].endswith("latest_pointer_checkpoint.pt")
    assert int(pointer["tick"]) == int(runtime.engine.tick)


def test_resolve_latest_pointer_returns_published_bundle(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=24, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "resolved_latest_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)

    resolved = resolve_latest_checkpoint_bundle(tmp_path)
    assert resolved == checkpoint_path


def test_resolve_latest_pointer_handles_relative_run_directory_paths(runtime_builder):
    runtime = runtime_builder(seed=240, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_dir = runtime.data_logger.run_dir
    checkpoint_path = checkpoint_dir / "relative_pointer_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)

    resolved = resolve_latest_checkpoint_bundle(checkpoint_dir)
    assert resolved == checkpoint_path.resolve()

    loaded = load_runtime_checkpoint(checkpoint_dir)
    assert int(loaded["engine_state"]["tick"]) == int(bundle["engine_state"]["tick"])


def test_resolve_latest_pointer_rejects_pointer_checksum_mismatch(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=241, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "checksum_pointer_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)

    pointer_path = checkpoint_path.parent / "latest_checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["bundle_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        resolve_latest_checkpoint_bundle(tmp_path)


def test_atomic_checkpoint_save_cleans_up_temp_files_on_bundle_write_failure(runtime_builder, tmp_path, monkeypatch):
    runtime = runtime_builder(seed=242, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "write_failure_checkpoint.pt"

    def boom(*args, **kwargs):
        raise RuntimeError("forced torch.save failure")

    monkeypatch.setattr(atomic_checkpoint.torch, "save", boom)

    with pytest.raises(RuntimeError, match="forced torch.save failure"):
        save_runtime_checkpoint(checkpoint_path, bundle)

    assert checkpoint_path.exists() is False
    assert checkpoint_path.with_suffix(".pt.manifest.json").exists() is False
    leftover = list(tmp_path.glob(f"{atomic_checkpoint.cfg.CHECKPOINT.TEMPFILE_PREFIX}*"))
    assert leftover == []


def test_strict_manifest_validation_rejects_missing_manifest(runtime_builder, tmp_path):
    runtime = runtime_builder(seed=25, agents=4, walls=0, hzones=0)
    bundle = capture_runtime_checkpoint(runtime)
    checkpoint_path = tmp_path / "manifest_required_checkpoint.pt"

    save_runtime_checkpoint(checkpoint_path, bundle)
    checkpoint_path.with_suffix(".pt.manifest.json").unlink()

    with pytest.raises(FileNotFoundError):
        load_runtime_checkpoint(checkpoint_path)
