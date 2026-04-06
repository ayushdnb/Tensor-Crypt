# Tensor Crypt Architecture

## Public Surface

- `config.py` is the authoritative knob surface.
- `run.py` and `main.py` are thin launch wrappers.
- Internal implementation lives under `tensor_crypt/`.
- Legacy `engine/` and `viewer/` paths remain as compatibility facades.

## Package Boundaries

- `tensor_crypt.app`: bootstrap and runtime assembly only.
- `tensor_crypt.simulation`: cross-subsystem tick orchestration.
- `tensor_crypt.world`: grid, procedural map generation, perception, and physics.
- `tensor_crypt.agents`: brain architecture and slot-based agent registry.
- `tensor_crypt.learning`: PPO buffering and optimization.
- `tensor_crypt.population`: evolutionary mutation helpers and respawn timing.
- `tensor_crypt.telemetry`: run paths and persistent logging artifacts.
- `tensor_crypt.viewer`: pygame viewer, layout, input, and rendering.

## Behavior-Sensitive Modules

The following modules are the main semantic danger zones for future edits:

- `tensor_crypt/simulation/engine.py`
- `tensor_crypt/world/physics.py`
- `tensor_crypt/world/perception.py`
- `tensor_crypt/learning/ppo.py`
- `tensor_crypt/population/respawn_controller.py`
- `tensor_crypt/agents/state_registry.py`

These files carry sequencing, tensor-shape, slot-ownership, or persistence
contracts that affect simulation and training behavior.

## Config Bridge

The package uses `tensor_crypt/config_bridge.py` to access the root `config.py`
without depending on the working directory. That bridge is the only deliberate
link from the package internals back to the root configuration surface.
