"""
Bridge module for the root configuration surface.

The project requirement is that `config.py` at repository root remains the
primary knob surface. Internal package modules should not depend on cwd-based
imports though, so this module centralizes the one controlled dependency on the
root config module.

Normal case:
- `config.py` is already importable from repository root, so we reuse that
  module directly.

Fallback case:
- if the package is imported from a context where `config` is not already on
  `sys.path`, we resolve the repository-local `config.py` by file path relative
  to this package. That keeps the project machine-independent without copying
  configuration state into a second source of truth.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


def _load_root_config_module() -> ModuleType:
    existing = sys.modules.get("config")
    if existing is not None:
        return existing

    try:
        import config as root_config

        return root_config
    except ModuleNotFoundError:
        config_path = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "config.py"
            if candidate.is_file():
                config_path = candidate
                break
        if config_path is None:
            raise RuntimeError("Unable to locate root config.py for config bridge loading")
        spec = spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load root config module from {config_path}")

        module = module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
        return module


_root_config = _load_root_config_module()

Config = _root_config.Config
SimConfig = _root_config.SimConfig
GridConfig = _root_config.GridConfig
MapgenConfig = _root_config.MapgenConfig
AgentsConfig = _root_config.AgentsConfig
RespawnConfig = _root_config.RespawnConfig
TraitInit = _root_config.TraitInit
TraitClamp = _root_config.TraitClamp
TraitsConfig = _root_config.TraitsConfig
PhysicsConfig = _root_config.PhysicsConfig
PerceptionConfig = _root_config.PerceptionConfig
BrainConfig = _root_config.BrainConfig
PPOConfig = _root_config.PPOConfig
EvolutionConfig = _root_config.EvolutionConfig
ViewerConfig = _root_config.ViewerConfig
LogConfig = _root_config.LogConfig
IdentityConfig = _root_config.IdentityConfig
SchemaConfig = _root_config.SchemaConfig
CheckpointConfig = _root_config.CheckpointConfig
MigrationConfig = _root_config.MigrationConfig
cfg = _root_config.cfg
