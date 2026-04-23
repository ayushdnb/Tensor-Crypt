"""Checkpoint resume compatibility policy.

This module deliberately separates launch intent from substrate restore
validation. Runtime checkpoint validation answers whether a bundle is safe to
restore; this policy answers whether the requested launch mode may truthfully
claim exact resume, drifted resume, or fork semantics.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..agents.brain import create_brain, validate_bloodline_family
from ..config_bridge import cfg


RESUME_TAXONOMY_VERSION = 1
COMPATIBILITY_REPORT_VERSION = 1

FRESH_RUN = "fresh_run"
RESUME_EXACT = "resume_exact"
RESUME_WITH_DRIFT = "resume_with_drift"
FORK_FROM_CHECKPOINT = "fork_from_checkpoint"

CANONICAL_LAUNCH_MODES = frozenset(
    {
        FRESH_RUN,
        RESUME_EXACT,
        RESUME_WITH_DRIFT,
        FORK_FROM_CHECKPOINT,
    }
)
CHECKPOINT_BACKED_MODES = frozenset(
    {
        RESUME_EXACT,
        RESUME_WITH_DRIFT,
        FORK_FROM_CHECKPOINT,
    }
)

LEGACY_METADATA_POLICY_INFER_CONSERVATIVE = "infer_conservative"
LEGACY_METADATA_POLICIES = frozenset({LEGACY_METADATA_POLICY_INFER_CONSERVATIVE, "reject"})

DEFAULT_COMPATIBILITY_REPORT_FILENAME = "resume_compatibility_report.json"

BLOCKED = "blocked"
ALLOWED_DRIFT = "allowed_drift"
FORK_ONLY = "fork_only"
IGNORED_ON_RESUME = "ignored_on_resume"
EXACT_COMPLETENESS = "exact_completeness"

HARD_FIXED_CONFIG_PATHS = (
    "SIM.DTYPE",
    "GRID.W",
    "GRID.H",
    "AGENTS.N",
    "PERCEPT.OBS_MODE",
    "PERCEPT.NUM_RAYS",
    "PERCEPT.CANONICAL_RAY_FEATURES",
    "PERCEPT.CANONICAL_SELF_FEATURES",
    "PERCEPT.CANONICAL_CONTEXT_FEATURES",
    "PERCEPT.EXPERIMENTAL_RAY_FEATURES",
    "PERCEPT.EXPERIMENTAL_SELF_FEATURES",
    "PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES",
    "BRAIN.ACTION_DIM",
    "BRAIN.VALUE_DIM",
    "BRAIN.FAMILY_ORDER",
    "BRAIN.EXPERIMENTAL_BRANCH_PRESET",
    "BRAIN.EXPERIMENTAL_BRANCH_FAMILY",
    "PPO.OWNERSHIP_MODE",
)

DRIFT_CONFIG_PATHS = (
    "VIEW.SHOW_BLOODLINE_LEGEND",
    "VIEW.SHOW_CATASTROPHE_PANEL",
    "VIEW.SHOW_CATASTROPHE_OVERLAY",
    "VIEW.SHOW_CATASTROPHE_STATUS_IN_HUD",
    "LOG.LOG_TICK_EVERY",
    "LOG.SNAPSHOT_EVERY",
    "TELEMETRY.FAMILY_SUMMARY_EVERY_TICKS",
    "TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS",
    "TELEMETRY.SUMMARY_SKIP_NON_EMIT_WORK",
    "TELEMETRY.PARQUET_BATCH_ROWS",
    "CHECKPOINT.SAVE_EVERY_TICKS",
    "CHECKPOINT.KEEP_LAST",
    "CHECKPOINT.DIRECTORY_NAME",
    "CHECKPOINT.SAVE_CHECKPOINT_MANIFEST",
    "CHECKPOINT.WRITE_LATEST_POINTER",
)

FORK_ONLY_CONFIG_PATHS = (
    "SIM.DEVICE",
    "SIM.EXPERIMENTAL_FAMILY_VMAP_INFERENCE",
    "LOG.AMP",
    "PPO.GAMMA",
    "PPO.LAMBDA",
    "PPO.CLIP_EPS",
    "PPO.ENTROPY_COEF",
    "PPO.VALUE_COEF",
    "PPO.LR",
    "PPO.BATCH_SZ",
    "PPO.MINI_BATCHES",
    "PPO.EPOCHS",
    "PPO.TARGET_KL",
    "PPO.GRAD_NORM_CLIP",
    "PPO.REWARD_FORM",
    "PPO.REWARD_GATE_MODE",
    "PPO.REWARD_GATE_THRESHOLD",
    "PPO.REWARD_BELOW_GATE_VALUE",
    "PPO.UPDATE_EVERY_N_TICKS",
    "RESPAWN.RESPAWN_PERIOD",
    "RESPAWN.MAX_SPAWNS_PER_CYCLE",
    "RESPAWN.POPULATION_FLOOR",
    "RESPAWN.POPULATION_CEILING",
    "RESPAWN.MODE",
    "RESPAWN.BRAIN_PARENT_SELECTOR",
    "RESPAWN.TRAIT_PARENT_SELECTOR",
    "RESPAWN.ANCHOR_PARENT_SELECTOR",
    "RESPAWN.EXTINCTION_POLICY",
    "RESPAWN.BIRTH_HP_MODE",
    "EVOL.SELECTION",
    "EVOL.FITNESS_DECAY",
    "EVOL.POLICY_NOISE_SD",
    "EVOL.FITNESS_TEMP",
    "EVOL.TRAIT_LOGIT_MUTATION_SIGMA",
    "EVOL.TRAIT_BUDGET_MUTATION_SIGMA",
    "EVOL.RARE_MUT_PROB",
    "EVOL.ENABLE_FAMILY_SHIFT_MUTATION",
    "CATASTROPHE.DEFAULT_MODE",
    "CATASTROPHE.DEFAULT_SCHEDULER_ARMED",
    "CATASTROPHE.ALLOW_OVERLAP",
    "CATASTROPHE.MAX_CONCURRENT",
    "CATASTROPHE.AUTO_DYNAMIC_GAP_MIN_TICKS",
    "CATASTROPHE.AUTO_DYNAMIC_GAP_MAX_TICKS",
    "CATASTROPHE.AUTO_STATIC_INTERVAL_TICKS",
    "CATASTROPHE.AUTO_STATIC_ORDERING_POLICY",
    "CATASTROPHE.PERSIST_STATE_IN_CHECKPOINTS",
)

IGNORED_CONFIG_PATH_PREFIXES = (
    "MAPGEN.",
)
IGNORED_CONFIG_PATHS = (
    "AGENTS.SPAWN_MODE",
    "BRAIN.INITIAL_FAMILY_ASSIGNMENT",
    "BRAIN.INITIAL_FAMILY_WEIGHTS",
)


def normalize_launch_mode(mode: object) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in CANONICAL_LAUNCH_MODES:
        supported = ", ".join(sorted(CANONICAL_LAUNCH_MODES))
        raise ValueError(f"Unsupported checkpoint launch mode {mode!r}; expected one of {{{supported}}}")
    return normalized


def normalize_legacy_metadata_policy(policy: object) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in LEGACY_METADATA_POLICIES:
        supported = ", ".join(sorted(LEGACY_METADATA_POLICIES))
        raise ValueError(f"Unsupported legacy checkpoint metadata policy {policy!r}; expected one of {{{supported}}}")
    return normalized


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(item) for item in shape]


def _get_path(source: Any, dotted_path: str, default: Any = None) -> Any:
    current = source
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _selected_config_values(source: dict | object, paths: tuple[str, ...]) -> dict[str, Any]:
    values = {}
    for path in paths:
        value = _get_path(source, path, None)
        if value is not None:
            values[path] = value
    return values


def _all_config_surface_values(source: dict | object) -> dict[str, Any]:
    paths = HARD_FIXED_CONFIG_PATHS + DRIFT_CONFIG_PATHS + FORK_ONLY_CONFIG_PATHS + IGNORED_CONFIG_PATHS
    return _selected_config_values(source, paths)


def _family_contracts_from_current_config(cfg_obj) -> dict[str, dict]:
    contracts: dict[str, dict] = {}
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    try:
        for raw_family in list(cfg_obj.BRAIN.FAMILY_ORDER):
            family_id = validate_bloodline_family(str(raw_family))
            brain = create_brain(family_id)
            description = brain.describe_family()
            contracts[family_id] = {
                "family_id": family_id,
                "observation_contract": str(description.get("observation_contract", "canonical_v2")),
                "topology_signature": [
                    {"name": str(name), "shape": [int(dim) for dim in shape]}
                    for name, shape in brain.get_topology_signature()
                ],
                "topology_signature_hash": _stable_hash(brain.get_topology_signature()),
                "action_dim": int(cfg_obj.BRAIN.ACTION_DIM),
                "value_dim": int(cfg_obj.BRAIN.VALUE_DIM),
            }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    return contracts


def _family_contracts_from_active_metadata(bundle: dict) -> dict[str, dict]:
    contracts: dict[str, dict] = {}
    saved_brain = bundle.get("config_snapshot", {}).get("BRAIN", {})
    action_dim = int(saved_brain.get("ACTION_DIM", cfg.BRAIN.ACTION_DIM))
    value_dim = int(saved_brain.get("VALUE_DIM", cfg.BRAIN.VALUE_DIM))
    for payload in bundle.get("brain_metadata_by_uid", {}).values():
        family_id = validate_bloodline_family(str(payload.get("family_id")))
        if family_id in contracts:
            continue
        topology = [
            {"name": str(item[0]), "shape": [int(dim) for dim in item[1]]}
            for item in payload.get("topology_signature", [])
        ]
        contracts[family_id] = {
            "family_id": family_id,
            "observation_contract": str(payload.get("observation_contract", saved_brain.get("OBSERVATION_CONTRACT", "legacy_inferred"))),
            "topology_signature": topology,
            "topology_signature_hash": _stable_hash(topology),
            "action_dim": action_dim,
            "value_dim": value_dim,
        }
    return contracts


def _observation_summary_from_config(config_source: dict | object) -> dict:
    return {
        "obs_mode": str(_get_path(config_source, "PERCEPT.OBS_MODE", "canonical_v2")),
        "obs_schema_version": int(_get_path(config_source, "SCHEMA.OBS_SCHEMA_VERSION", cfg.SCHEMA.OBS_SCHEMA_VERSION)),
        "num_rays": int(_get_path(config_source, "PERCEPT.NUM_RAYS", cfg.PERCEPT.NUM_RAYS)),
        "canonical_ray_features": int(_get_path(config_source, "PERCEPT.CANONICAL_RAY_FEATURES", cfg.PERCEPT.CANONICAL_RAY_FEATURES)),
        "canonical_self_features": int(_get_path(config_source, "PERCEPT.CANONICAL_SELF_FEATURES", cfg.PERCEPT.CANONICAL_SELF_FEATURES)),
        "canonical_context_features": int(_get_path(config_source, "PERCEPT.CANONICAL_CONTEXT_FEATURES", cfg.PERCEPT.CANONICAL_CONTEXT_FEATURES)),
        "experimental_ray_features": int(_get_path(config_source, "PERCEPT.EXPERIMENTAL_RAY_FEATURES", cfg.PERCEPT.EXPERIMENTAL_RAY_FEATURES)),
        "experimental_self_features": int(_get_path(config_source, "PERCEPT.EXPERIMENTAL_SELF_FEATURES", cfg.PERCEPT.EXPERIMENTAL_SELF_FEATURES)),
        "experimental_context_features": int(_get_path(config_source, "PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES", cfg.PERCEPT.EXPERIMENTAL_CONTEXT_FEATURES)),
    }


def _substrate_summary_from_bundle(bundle: dict) -> dict:
    registry_state = bundle.get("registry_state", {})
    grid_state = bundle.get("grid_state", {})
    return {
        "registry_data_shape": _shape_list(registry_state.get("data")),
        "slot_uid_shape": _shape_list(registry_state.get("slot_uid")),
        "slot_parent_uid_shape": _shape_list(registry_state.get("slot_parent_uid")),
        "fitness_shape": _shape_list(registry_state.get("fitness")),
        "grid_shape": _shape_list(grid_state.get("grid")),
    }


def _substrate_summary_from_config(config_source: dict | object) -> dict:
    max_agents = int(_get_path(config_source, "AGENTS.N", cfg.AGENTS.N))
    width = int(_get_path(config_source, "GRID.W", cfg.GRID.W))
    height = int(_get_path(config_source, "GRID.H", cfg.GRID.H))
    return {
        "registry_data_shape": [15, max_agents],
        "slot_uid_shape": [max_agents],
        "slot_parent_uid_shape": [max_agents],
        "fitness_shape": [max_agents],
        "grid_shape": [4, height, width],
    }


def _capture_presence(bundle: dict) -> dict:
    ppo_state = bundle.get("ppo_state", {})
    engine_state = bundle.get("engine_state", {})
    buffers = ppo_state.get("buffer_state_by_uid", {}) or {}
    nonempty_buffers = 0
    for payload in buffers.values():
        if len(payload.get("actions", [])) > 0:
            nonempty_buffers += 1
    return {
        "rng_state_present": bundle.get("rng_state") is not None,
        "optimizer_state_uid_count": int(len(ppo_state.get("optimizer_state_by_uid", {}) or {})),
        "optimizer_metadata_uid_count": int(len(ppo_state.get("optimizer_metadata_by_uid", {}) or {})),
        "training_state_uid_count": int(len(ppo_state.get("training_state_by_uid", {}) or {})),
        "buffer_state_uid_count": int(len(buffers)),
        "nonempty_buffer_uid_count": int(nonempty_buffers),
        "scaler_state_present": ppo_state.get("scaler_state") is not None,
        "catastrophe_state_present": engine_state.get("catastrophe_state") is not None,
        "respawn_overlay_runtime_state_present": engine_state.get("respawn_overlay_runtime_state") is not None,
    }


def _exact_resume_deficits(bundle: dict, *, legacy_contract_missing: bool) -> list[str]:
    deficits: list[str] = []
    config_snapshot = bundle.get("config_snapshot", {})
    ppo_state = bundle.get("ppo_state", {})
    presence = _capture_presence(bundle)

    if legacy_contract_missing:
        deficits.append("resume_contract_metadata_missing")
    if bool(_get_path(config_snapshot, "CHECKPOINT.CAPTURE_RNG_STATE", True)) and not presence["rng_state_present"]:
        deficits.append("rng_state_missing")
    if bool(_get_path(config_snapshot, "CHECKPOINT.CAPTURE_OPTIMIZER_STATE", True)) is False:
        deficits.append("optimizer_state_capture_disabled")
    if bool(_get_path(config_snapshot, "CHECKPOINT.CAPTURE_PPO_TRAINING_STATE", True)) is False:
        deficits.append("ppo_training_state_capture_disabled")
    if bool(_get_path(config_snapshot, "CHECKPOINT.CAPTURE_BOOTSTRAP_STATE", True)) is False and presence["nonempty_buffer_uid_count"] > 0:
        deficits.append("bootstrap_state_capture_disabled_with_active_buffers")
    if bool(_get_path(config_snapshot, "LOG.AMP", False)) and bool(_get_path(config_snapshot, "CHECKPOINT.CAPTURE_SCALER_STATE", True)):
        if ppo_state.get("scaler_state") is None:
            deficits.append("amp_scaler_state_missing")
    if bool(_get_path(config_snapshot, "CATASTROPHE.PERSIST_STATE_IN_CHECKPOINTS", True)) and not presence["catastrophe_state_present"]:
        deficits.append("catastrophe_state_missing")
    if not presence["respawn_overlay_runtime_state_present"]:
        deficits.append("respawn_overlay_runtime_state_missing")
    return deficits


def build_checkpoint_contract_snapshot(bundle: dict, cfg_obj=cfg) -> dict:
    """Build compact resume contract metadata for a newly captured bundle."""
    config_snapshot = bundle.get("config_snapshot", asdict(cfg_obj))
    substrate = _substrate_summary_from_bundle(bundle)
    observation = _observation_summary_from_config(config_snapshot)
    family_contracts = _family_contracts_from_current_config(cfg_obj)
    capture_presence = _capture_presence(bundle)
    deficits = _exact_resume_deficits(bundle, legacy_contract_missing=False)
    contract_hashes = {
        "substrate_shape_contract": _stable_hash(substrate),
        "observation_contract": _stable_hash(observation),
        "family_contract_set": _stable_hash(family_contracts),
    }
    return {
        "resume_taxonomy_version": RESUME_TAXONOMY_VERSION,
        "compatibility_report_version": COMPATIBILITY_REPORT_VERSION,
        "hard_surface_summary": {
            "checkpoint_schema_version": int(bundle.get("checkpoint_schema_version", cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION)),
            "schema_versions": dict(bundle.get("schema_versions", {})),
        },
        "substrate_shape_summary": substrate,
        "observation_contract_summary": observation,
        "family_contracts": family_contracts,
        "family_contracts_complete": True,
        "capture_presence": capture_presence,
        "config_surface_values": _all_config_surface_values(config_snapshot),
        "contract_hashes": contract_hashes,
        "exact_resume_capability": {
            "exact_resume_capable": len(deficits) == 0,
            "deficits": deficits,
        },
    }


def infer_legacy_checkpoint_contract(bundle: dict, cfg_obj=cfg) -> dict:
    """Infer a conservative contract summary for checkpoints without contract metadata."""
    config_snapshot = bundle.get("config_snapshot", {})
    substrate = _substrate_summary_from_bundle(bundle)
    observation = _observation_summary_from_config(config_snapshot)
    family_contracts = _family_contracts_from_active_metadata(bundle)
    capture_presence = _capture_presence(bundle)
    deficits = _exact_resume_deficits(bundle, legacy_contract_missing=True)
    contract_hashes = {
        "substrate_shape_contract": _stable_hash(substrate),
        "observation_contract": _stable_hash(observation),
        "family_contract_set": _stable_hash(family_contracts),
    }
    return {
        "resume_taxonomy_version": 0,
        "compatibility_report_version": COMPATIBILITY_REPORT_VERSION,
        "legacy_inferred": True,
        "hard_surface_summary": {
            "checkpoint_schema_version": int(bundle.get("checkpoint_schema_version", cfg_obj.SCHEMA.CHECKPOINT_SCHEMA_VERSION)),
            "schema_versions": dict(bundle.get("schema_versions", {})),
        },
        "substrate_shape_summary": substrate,
        "observation_contract_summary": observation,
        "family_contracts": family_contracts,
        "family_contracts_complete": False,
        "capture_presence": capture_presence,
        "config_surface_values": _all_config_surface_values(config_snapshot),
        "contract_hashes": contract_hashes,
        "exact_resume_capability": {
            "exact_resume_capable": False,
            "deficits": deficits,
        },
    }


def get_checkpoint_contract(bundle: dict, cfg_obj=cfg) -> tuple[dict, bool]:
    metadata = bundle.get("metadata", {})
    resume_contract = metadata.get("resume_contract")
    if isinstance(resume_contract, dict):
        return resume_contract, False
    policy = normalize_legacy_metadata_policy(getattr(cfg_obj.CHECKPOINT, "LEGACY_METADATA_POLICY", LEGACY_METADATA_POLICY_INFER_CONSERVATIVE))
    if policy == "reject":
        raise ValueError("Checkpoint lacks resume contract metadata and LEGACY_METADATA_POLICY='reject'")
    return infer_legacy_checkpoint_contract(bundle, cfg_obj), True


def build_current_contract_summary(cfg_obj=cfg) -> dict:
    config_snapshot = asdict(cfg_obj)
    substrate = _substrate_summary_from_config(config_snapshot)
    observation = _observation_summary_from_config(config_snapshot)
    family_contracts = _family_contracts_from_current_config(cfg_obj)
    return {
        "substrate_shape_summary": substrate,
        "observation_contract_summary": observation,
        "family_contracts": family_contracts,
        "contract_hashes": {
            "substrate_shape_contract": _stable_hash(substrate),
            "observation_contract": _stable_hash(observation),
            "family_contract_set": _stable_hash(family_contracts),
        },
        "config_surface_values": _all_config_surface_values(config_snapshot),
    }


def _delta_entry(surface: str, category: str, saved: Any, current: Any, message: str) -> dict:
    return {
        "surface": surface,
        "category": category,
        "saved": saved,
        "current": current,
        "message": message,
    }


def classify_surface_deltas(bundle: dict, cfg_obj=cfg) -> tuple[list[dict], bool, dict, dict]:
    checkpoint_contract, legacy_inferred = get_checkpoint_contract(bundle, cfg_obj)
    current_contract = build_current_contract_summary(cfg_obj)
    deltas: list[dict] = []

    for surface_name, message in (
        ("substrate_shape_contract", "Checkpoint substrate shape must match the restore runtime"),
        ("observation_contract", "Observation contract is checkpoint-visible and cannot drift on resume"),
        ("family_contract_set", "Family topology and observation contracts are checkpoint-visible"),
    ):
        if surface_name == "family_contract_set" and not bool(checkpoint_contract.get("family_contracts_complete", False)):
            continue
        saved_hash = checkpoint_contract.get("contract_hashes", {}).get(surface_name)
        current_hash = current_contract.get("contract_hashes", {}).get(surface_name)
        if saved_hash is not None and current_hash is not None and saved_hash != current_hash:
            deltas.append(_delta_entry(surface_name, BLOCKED, saved_hash, current_hash, message))

    saved_values = checkpoint_contract.get("config_surface_values", {})
    current_values = current_contract.get("config_surface_values", {})

    for path in HARD_FIXED_CONFIG_PATHS:
        saved = saved_values.get(path)
        current = current_values.get(path)
        if saved is not None and current is not None and saved != current:
            deltas.append(
                _delta_entry(path, BLOCKED, saved, current, "Hard-fixed restore surface differs")
            )

    for path in DRIFT_CONFIG_PATHS:
        saved = saved_values.get(path)
        current = current_values.get(path)
        if saved is not None and current is not None and saved != current:
            deltas.append(
                _delta_entry(path, ALLOWED_DRIFT, saved, current, "Operator or future-artifact cadence surface differs")
            )

    for path in FORK_ONLY_CONFIG_PATHS:
        saved = saved_values.get(path)
        current = current_values.get(path)
        if saved is not None and current is not None and saved != current:
            deltas.append(
                _delta_entry(path, FORK_ONLY, saved, current, "Future simulation or numeric behavior surface differs")
            )

    for path in IGNORED_CONFIG_PATHS:
        saved = saved_values.get(path)
        current = current_values.get(path)
        if saved is not None and current is not None and saved != current:
            deltas.append(
                _delta_entry(path, IGNORED_ON_RESUME, saved, current, "Fresh-run-only surface ignored by resume runtime")
            )

    saved_snapshot = bundle.get("config_snapshot", {})
    current_snapshot = asdict(cfg_obj)
    for prefix in IGNORED_CONFIG_PATH_PREFIXES:
        section = prefix.rstrip(".")
        if saved_snapshot.get(section) != current_snapshot.get(section):
            deltas.append(
                _delta_entry(prefix + "*", IGNORED_ON_RESUME, saved_snapshot.get(section), current_snapshot.get(section), "Fresh-run map generation surface ignored by resume runtime")
            )

    return deltas, legacy_inferred, checkpoint_contract, current_contract


def build_resume_compatibility_report(
    *,
    requested_mode: str,
    bundle: dict,
    cfg_obj=cfg,
    source_checkpoint_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> dict:
    requested_mode = normalize_launch_mode(requested_mode)
    if requested_mode == FRESH_RUN:
        raise ValueError("Fresh runs do not have checkpoint compatibility reports")

    deltas, legacy_inferred, checkpoint_contract, current_contract = classify_surface_deltas(bundle, cfg_obj)
    exact_deficits = list(checkpoint_contract.get("exact_resume_capability", {}).get("deficits", []))
    hard_failures = [delta for delta in deltas if delta["category"] == BLOCKED]
    drift_surfaces = [delta for delta in deltas if delta["category"] == ALLOWED_DRIFT]
    fork_required_surfaces = [delta for delta in deltas if delta["category"] == FORK_ONLY]
    ignored_surfaces = [delta for delta in deltas if delta["category"] == IGNORED_ON_RESUME]

    allowed = False
    resolved_mode = None
    failure_class = None
    if hard_failures:
        failure_class = "blocked_restore_safety_mismatch"
    elif requested_mode == RESUME_EXACT:
        if exact_deficits:
            failure_class = "exact_resume_completeness_deficit"
        elif fork_required_surfaces:
            failure_class = "fork_semantics_required"
        elif drift_surfaces:
            failure_class = "drift_acknowledgment_required"
        else:
            allowed = True
            resolved_mode = RESUME_EXACT
    elif requested_mode == RESUME_WITH_DRIFT:
        if fork_required_surfaces:
            failure_class = "fork_semantics_required"
        else:
            allowed = True
            resolved_mode = RESUME_WITH_DRIFT
    elif requested_mode == FORK_FROM_CHECKPOINT:
        allowed = True
        resolved_mode = FORK_FROM_CHECKPOINT

    source_checkpoint = None if source_checkpoint_path is None else str(source_checkpoint_path)
    source_manifest = None if source_manifest_path is None else str(source_manifest_path)
    source_identity = {
        "checkpoint_path": source_checkpoint,
        "manifest_path": source_manifest,
        "tick": int(bundle.get("engine_state", {}).get("tick", -1)),
        "checkpoint_schema_version": int(bundle.get("checkpoint_schema_version", -1)),
        "active_uid_count": int(len(bundle.get("brain_state_by_uid", {}))),
        "config_fingerprint": bundle.get("metadata", {}).get("config_fingerprint"),
    }

    return {
        "compatibility_report_version": COMPATIBILITY_REPORT_VERSION,
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "allowed": bool(allowed),
        "failure_class": failure_class,
        "legacy_contract_inference_used": bool(legacy_inferred),
        "source_checkpoint_identity": source_identity,
        "surface_deltas": deltas,
        "hard_failures": hard_failures,
        "drift_surfaces": drift_surfaces,
        "fork_required_surfaces": fork_required_surfaces,
        "ignored_on_resume_surfaces": ignored_surfaces,
        "exact_resume_completeness_deficits": exact_deficits,
        "checkpoint_contract_hashes": checkpoint_contract.get("contract_hashes", {}),
        "current_contract_hashes": current_contract.get("contract_hashes", {}),
        "ancestry": {
            "session_kind": resolved_mode,
            "source_checkpoint_path": source_checkpoint,
            "source_checkpoint_tick": source_identity["tick"],
            "fork_reason": getattr(cfg_obj.CHECKPOINT, "FORK_REASON", "") if requested_mode == FORK_FROM_CHECKPOINT else "",
        },
    }


def resolve_resume_request(
    *,
    requested_mode: str,
    bundle: dict,
    cfg_obj=cfg,
    source_checkpoint_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> dict:
    """Return a compatibility report and enforce mode-specific semantics."""
    return build_resume_compatibility_report(
        requested_mode=requested_mode,
        bundle=bundle,
        cfg_obj=cfg_obj,
        source_checkpoint_path=source_checkpoint_path,
        source_manifest_path=source_manifest_path,
    )


def write_resume_compatibility_report(path: str | Path, report: dict) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=_json_default)
    return report_path


def session_metadata_from_report(report: dict) -> dict:
    source_identity = report.get("source_checkpoint_identity", {})
    resolved_mode = report.get("resolved_mode") or report.get("requested_mode")
    return {
        "launch_mode_requested": report.get("requested_mode"),
        "launch_mode_resolved": resolved_mode,
        "session_kind": resolved_mode,
        "source_checkpoint_path": source_identity.get("checkpoint_path"),
        "source_manifest_path": source_identity.get("manifest_path"),
        "source_checkpoint_tick": source_identity.get("tick"),
        "source_checkpoint_schema_version": source_identity.get("checkpoint_schema_version"),
        "legacy_contract_inference_used": bool(report.get("legacy_contract_inference_used", False)),
        "compatibility_report_path": DEFAULT_COMPATIBILITY_REPORT_FILENAME,
        "fork_reason": report.get("ancestry", {}).get("fork_reason", ""),
        "ancestor_session_kind": "checkpointed_session",
        "compatibility_failure_class": report.get("failure_class"),
    }


__all__ = [
    "ALLOWED_DRIFT",
    "BLOCKED",
    "CANONICAL_LAUNCH_MODES",
    "CHECKPOINT_BACKED_MODES",
    "COMPATIBILITY_REPORT_VERSION",
    "DEFAULT_COMPATIBILITY_REPORT_FILENAME",
    "FORK_FROM_CHECKPOINT",
    "FORK_ONLY",
    "FRESH_RUN",
    "IGNORED_ON_RESUME",
    "LEGACY_METADATA_POLICIES",
    "RESUME_EXACT",
    "RESUME_TAXONOMY_VERSION",
    "RESUME_WITH_DRIFT",
    "build_checkpoint_contract_snapshot",
    "build_current_contract_summary",
    "build_resume_compatibility_report",
    "classify_surface_deltas",
    "get_checkpoint_contract",
    "infer_legacy_checkpoint_contract",
    "normalize_launch_mode",
    "normalize_legacy_metadata_policy",
    "resolve_resume_request",
    "session_metadata_from_report",
    "write_resume_compatibility_report",
]
