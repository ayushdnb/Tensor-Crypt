import pytest
import torch

from tensor_crypt.checkpointing.runtime_checkpoint import capture_runtime_checkpoint, validate_runtime_checkpoint
from tensor_crypt.config_bridge import cfg


def test_checkpoint_bundle_contains_identity_and_rng_substrate(runtime_builder):
    runtime = runtime_builder(seed=201, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)

    bundle = capture_runtime_checkpoint(runtime)

    assert {
        "checkpoint_schema_version",
        "schema_versions",
        "config_snapshot",
        "engine_state",
        "registry_state",
        "grid_state",
        "brain_state_by_uid",
        "ppo_state",
        "rng_state",
        "metadata",
    }.issubset(bundle.keys())
    assert bundle["checkpoint_schema_version"] == cfg.SCHEMA.CHECKPOINT_SCHEMA_VERSION
    assert bundle["schema_versions"]["IDENTITY_SCHEMA_VERSION"] == cfg.SCHEMA.IDENTITY_SCHEMA_VERSION
    assert bundle["registry_state"]["slot_uid"].dtype == torch.int64
    assert bundle["registry_state"]["slot_parent_uid"].dtype == torch.int64
    assert bundle["rng_state"] is not None
    assert {"python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all"}.issubset(bundle["rng_state"].keys())


def test_checkpoint_validation_rejects_duplicate_active_uid(runtime_builder):
    runtime = runtime_builder(seed=202, width=10, height=10, agents=4, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)

    bundle = capture_runtime_checkpoint(runtime)
    alive_slots = runtime.registry.get_alive_indices().tolist()
    assert len(alive_slots) >= 2

    slot_uid = bundle["registry_state"]["slot_uid"].clone()
    duplicate_uid = int(slot_uid[alive_slots[0]].item())
    slot_uid[alive_slots[1]] = duplicate_uid
    bundle["registry_state"]["slot_uid"] = slot_uid

    with pytest.raises(ValueError, match="Duplicate active UID"):
        validate_runtime_checkpoint(bundle, cfg)

