# Mode Matrix

## Claimed Runtime-Supported Choice Surfaces Observed

| Surface | Observed supported values |
| --- | --- |
| `GRID.HZ_OVERLAP_MODE` | `max_abs`, `sum_clamped`, `last_wins` |
| `AGENTS.SPAWN_MODE` | `uniform` |
| `TRAITS.METAB_FORM` | `affine_combo` |
| `PERCEPT.OBS_MODE` | `canonical_v2`, `experimental_selfcentric_v1` |
| `BRAIN.INITIAL_FAMILY_ASSIGNMENT` | `round_robin`, `weighted_random` |
| `RESPAWN.MODE` | `binary_parented` |
| `RESPAWN.ANCHOR_PARENT_SELECTOR` | `brain_parent`, `trait_parent`, `random_parent`, `fitter_of_two` |
| `RESPAWN.EXTINCTION_POLICY` | `fail_run`, `seed_bank_bootstrap`, `admin_spawn_defaults` |
| `RESPAWN.BIRTH_HP_MODE` | `full`, `fraction` |
| `PPO.OWNERSHIP_MODE` | `uid_strict` |
| `PHYS.TIE_BREAKER` | `strength_then_lowest_id`, `random_seeded` |
| `TELEMETRY.LINEAGE_EXPORT_FORMAT` | `json` |
| `CATASTROPHE.DEFAULT_MODE` | `off`, `manual_only`, `auto_dynamic`, `auto_static` |

## Initial Priority Intersections
- `PERCEPT.OBS_MODE` x `BRAIN.EXPERIMENTAL_BRANCH_PRESET` x experimental family vmap.
- `CATASTROPHE.DEFAULT_MODE` x checkpoint round-trip x scheduler armed/pause state.
- `RESPAWN` overlay policies x below-floor/extinction behaviors.
- `CHECKPOINT` strictness toggles x manifest/latest-pointer presence.
- canonical entrypoints vs compatibility wrappers under the same config singleton.

## Executed Cases

| Case | Coverage | Outcome |
| --- | --- | --- |
| Root launch preset | `run.py` / `main.py` compatibility surface applies experimental single-family launch defaults before runtime construction | Pass |
| Canonical CPU runtime | Full test suite + benchmark + soak | Pass |
| Experimental family-vmap CPU path | Full test suite + dedicated benchmark with `--experimental-family-vmap-inference` | Pass |
| Checkpoint strictness | Manifest absence, checksum corruption, pointer checksum corruption, atomic temp-file cleanup, repeated resume chain | Pass |
| Catastrophe scheduler and restore | Existing catastrophe and scheduler tests in full suite + soak harness with periodic checkpoint validation | Pass |

## Not Executed
- CUDA device modes
- Extremely long horizon (`>= 10^7` ticks)
