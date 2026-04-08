"""Bridge module for the canonical runtime configuration."""

from . import runtime_config as _runtime_config


Config = _runtime_config.Config
SimConfig = _runtime_config.SimConfig
GridConfig = _runtime_config.GridConfig
MapgenConfig = _runtime_config.MapgenConfig
AgentsConfig = _runtime_config.AgentsConfig
RespawnConfig = _runtime_config.RespawnConfig
TraitInit = _runtime_config.TraitInit
TraitClamp = _runtime_config.TraitClamp
TraitsConfig = _runtime_config.TraitsConfig
PhysicsConfig = _runtime_config.PhysicsConfig
PerceptionConfig = _runtime_config.PerceptionConfig
BrainConfig = _runtime_config.BrainConfig
PPOConfig = _runtime_config.PPOConfig
EvolutionConfig = _runtime_config.EvolutionConfig
ViewerConfig = _runtime_config.ViewerConfig
LogConfig = _runtime_config.LogConfig
IdentityConfig = _runtime_config.IdentityConfig
SchemaConfig = _runtime_config.SchemaConfig
CheckpointConfig = _runtime_config.CheckpointConfig
TelemetryConfig = _runtime_config.TelemetryConfig
ValidationConfig = _runtime_config.ValidationConfig
MigrationConfig = _runtime_config.MigrationConfig
CatastropheConfig = _runtime_config.CatastropheConfig
cfg = _runtime_config.cfg
