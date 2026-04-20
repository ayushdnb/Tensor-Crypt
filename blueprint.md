# Blueprint

## Architecture

This branch is the five-brain release line. Its canonical behavior is a multi-family bloodline simulation with five configured MLP topology signatures:

- House Nocthar
- House Vespera
- House Umbrael
- House Mourndveil
- House Somnyr

The canonical implementation root is `tensor_crypt/`. Root files and `engine/` or `viewer/` packages are compatibility surfaces.

## Split Doctrine

The five-brain line keeps canonical observations, family-specific brain topology, UID-owned PPO, binary-parented respawn, telemetry ledgers, checkpoint validation, catastrophe controls, and the pygame-ce viewer.

It removes the single-family launch preset from the public startup contract. The optional same-family vmap inference path remains disabled by default and is treated as a benchmark accelerator, not as branch identity.

## Invariants

- Slots are storage; UIDs are identity.
- Family assignment is part of UID lineage and checkpoint-visible brain topology.
- Brain topology signatures must not drift silently.
- Checkpoints must reject schema, UID, topology, optimizer, or manifest mismatches.
- Runtime validation should fail unsupported config values loudly.

## Branch Policy

Branch: `release/five-brain`

Version target: `1.0.0` if validation and review pass.

Tag target: `five-brain/v1.0.0` only after successful validation.
