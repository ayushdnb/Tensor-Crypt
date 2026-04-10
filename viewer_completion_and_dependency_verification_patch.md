# Viewer Completion And Dependency Verification Patch

## 1. Title and scope

This artifact documents a full repository-grounded audit of Tensor Crypt's viewer completion status and its dependency / packaging / documentation / environment surfaces, with a specific focus on the previously unconfirmed `pygame-ce` governance question.

Scope covered in this audit:
- Real repository inspection, not the Python-only dump
- Viewer layout / overflow / fit / resize behavior verification
- Viewer bloodline color palette and low-HP modulation verification
- Catastrophe scheduler controls, pause/arm/clear semantics, and checkpoint-safety verification
- Dependency manifests, setup surfaces, docs, and CI/environment surface inspection
- Runtime environment verification against the installed `pygame` import provider
- Minimal-disturbance patching only where the repo was still incomplete

## 2. Executive verdict

Verdict: the planned viewer feature work is already present in the current working tree, with meaningful code and tests behind it.

The only concrete repository gap found by this audit was dependency governance and documentation alignment around the viewer backend:
- the repo code and live environment are using `pygame-ce`
- but the checked-in manifests still declared `pygame`
- and the README still described one catastrophe control (`U`) using older semantics

This audit fixed those gaps by:
- replacing floating/ambiguous `pygame` manifest entries with explicit `pygame-ce>=2.5.6,<2.6`
- documenting that `pygame-ce` is the intended backend while the code continues to `import pygame`
- aligning the README hotkey wording with the current scheduler arm/disarm behavior
- adding a fast manifest-governance test to keep this from drifting back

## 3. Checklist of the original planned items

- Viewer layout cleanup: confirmed complete
- Viewer color work: confirmed complete
- Catastrophe controls: confirmed complete
- Dependency manifests explicitly governing the intended backend: incomplete and fixed
- Packaging/install docs aligned with the intended backend: incomplete and fixed
- CI/workflow dependency surface: confirmed complete relative to repository reality because no in-repo CI/workflow files were present to fix
- Runtime compatibility verification for the current viewer API against `pygame-ce`: confirmed complete
- Full downloadable patch artifact: completed by this file

## 4. Full repository-grounded audit summary

Repository surfaces actually present:
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- viewer code under `tensor_crypt/viewer/`
- catastrophe runtime code under `tensor_crypt/simulation/catastrophes.py`
- compatibility import shims under top-level `viewer/` and `engine/`
- viewer-focused tests under `tests/test_viewer_patch_package.py`, `tests/test_viewer_layout_cleanup.py`, `tests/test_viewer_color_semantics.py`, and `tests/test_catastrophe_scheduler_controls.py`

Repository surfaces specifically checked and found absent:
- `AGENTS.md`
- `setup.cfg`
- `setup.py`
- `requirements-dev.txt`
- `poetry.lock`
- `uv.lock`
- `Pipfile`
- `Pipfile.lock`
- `environment.yml` / `environment.yaml`
- `.github/workflows/*`
- `.vscode/*`
- `.devcontainer/*`

Viewer layout audit findings:
- `tensor_crypt/viewer/layout.py` already contains a dedicated `LayoutManager` with explicit world, side-panel, HUD, and content rectangle calculations.
- `tensor_crypt/viewer/main.py` already refreshes viewer geometry coherently on resize and fullscreen transitions without forcing an unnecessary refit on every resize event.
- `tensor_crypt/viewer/panels.py` already contains wrapped text, scroll offset clamping, content-height calculation, clipped rendering, and scrollbar drawing for overflowed inspector content.
- `tests/test_viewer_layout_cleanup.py` and `tests/test_viewer_patch_package.py` already cover screen bounds, resize semantics, wrapping, scroll behavior, and draw smoke.

Viewer color audit findings:
- `tensor_crypt/runtime_config.py` already defines the intended family palette direction: blue / green / yellow / red / violet.
- `tensor_crypt/viewer/colors.py` already separates canonical base family color from HP-based modulation via `get_bloodline_base_color()` and `get_bloodline_agent_color()`.
- `cfg.VIEW.BLOODLINE_LOW_HP_COLOR_MODULATION_ENABLED` already exists and is live.
- `tests/test_viewer_color_semantics.py` already verifies the exact palette, contrast floor, modulation-off behavior, modulation curve behavior, and routing of renderer/legend drawing through the canonical color helpers.

Catastrophe-control audit findings:
- `tensor_crypt/viewer/input.py` already exposes clear-active (`C`), mode-cycle (`Y`), scheduler arm/disarm (`U`), panel toggle (`I`), and scheduler pause/resume (`O`) hotkeys.
- `tensor_crypt/simulation/catastrophes.py` already distinguishes mode, scheduler armed state, scheduler paused state, active state, and next-auto-tick semantics.
- checkpoint restore logic already persists and restores catastrophe scheduler state.
- `tests/test_catastrophe_scheduler_controls.py` already covers arm/disarm behavior, stale next-tick replanning, clear-active restoration, checkpoint round-trip semantics, and viewer hotkey routing.

## 5. What was previously unconfirmed, and how it was verified

Previously unconfirmed item:
- whether the repository actually governed `pygame-ce` correctly outside the Python-only source dump

How this audit verified it:
- inspected `pyproject.toml`
- inspected `requirements.txt`
- inspected `README.md`
- searched the repository for `pygame-ce` and `pygame` references
- verified that no other packaging/lock/environment files or CI workflow files were present
- queried the live environment and confirmed:
  - `import pygame` succeeds
  - the imported module comes from the `pygame-ce` distribution
  - installed version is `pygame-ce 2.5.6`
  - no separate `pygame` distribution is installed
- forced setuptools to regenerate distribution metadata far enough to emit `tensor_crypt.egg-info/requires.txt` and `PKG-INFO`, which both showed `Requires-Dist: pygame-ce<2.6,>=2.5.6` after the patch

Bottom line:
- before this patch, the repo's manifests were ambiguous/wrong for the intended backend
- after this patch, the manifest and docs surfaces now match the actual runtime/backend strategy

## 6. Exact dependency / packaging / docs / CI findings

Dependency manifest findings before the patch:
- `pyproject.toml` declared `pygame`
- `requirements.txt` declared `pygame`
- there was no explicit `pygame-ce` mention in manifests
- there was no version bound governing the intended backend line

Dependency/runtime findings from the live environment:
- the active runtime import path is `import pygame`
- that module is currently provided by `pygame-ce 2.5.6`
- the `pygame` distribution itself is not installed in this environment
- that means the manifests were not matching the actual installed provider

Packaging surface findings:
- standard setuptools packaging is present through `pyproject.toml`
- no `setup.cfg` / `setup.py` duplication surface exists
- no lockfile-based package manager surface exists in-repo
- no conda/devcontainer/editor dependency surface exists in-repo

Documentation findings before the patch:
- README install instructions did not explain that the repo depends on `pygame-ce`
- README hotkey text still described `U` as toggling catastrophe auto mode, while the current runtime semantics are scheduler arm/disarm with mode retained separately
- no other install docs were found that needed correction

CI/workflow findings:
- there are no in-repo GitHub Actions workflows or other CI definitions to update
- therefore there was no CI file to patch
- to compensate for the lack of CI surface, this audit added a lightweight repository test that asserts the manifests and README stay aligned on `pygame-ce`

Historical/report doc findings:
- some historical audit/report files mention "pygame" generically
- these were not changed because they are historical/reporting artifacts, not active installation instructions
- the code still intentionally uses the `pygame` import namespace even when the installed backend is `pygame-ce`

AGENTS findings:
- no `AGENTS.md` was present
- no new `AGENTS.md` was added because the repository did not show a narrow recurring agent-handling gap that justified adding one during this focused audit

## 7. Exact decision on pygame-ce version strategy and why

Chosen strategy:
- `pygame-ce>=2.5.6,<2.6`

Why this strategy was chosen:
- the live environment in this audit is already running successfully on `pygame-ce 2.5.6`
- the repository's viewer code uses mainstream pygame 2 APIs such as `VIDEORESIZE`, `WINDOWRESIZED`, `MOUSEWHEEL`, `pygame.display.get_surface()`, `pygame.display.get_desktop_sizes()`, `pygame.RESIZABLE`, and `pygame.FULLSCREEN`
- the pygame-ce docs confirm the event/display APIs used by the viewer are supported on the current pygame-ce line
- PyPI shows that `pygame-ce` has newer 2.5.x releases, including `2.5.7` as of March 2, 2026, but this environment was not upgraded during the audit
- for a high-stakes repo, requiring the exact backend family and constraining to the verified minor line is safer than switching the repo to an unverified newer minor or leaving it floating entirely

Why this was not pinned to latest exact upstream release:
- exact-pin-to-latest would have forced a dependency version not actually installed and runtime-validated in this environment
- the chosen bound keeps the repo on the validated 2.5 line while still permitting safe patch uptake inside that line

Why this was not left as `pygame`:
- the runtime/backend intent is clearly `pygame-ce`
- `pygame` and `pygame-ce` are different distributions even though both expose the `pygame` import namespace
- leaving the manifests on `pygame` creates avoidable ambiguity and can install the wrong provider

## 8. Patch ledger listing every changed file

Files changed by this audit pass:
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `tests/test_dependency_governance.py`

Files intentionally not changed by this audit pass:
- all viewer/runtime feature files already containing the planned work
- all historical audit/report documents
- no CI files, because none exist in-repo

## 9. Full updated file content for every changed file

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tensor-crypt"
version = "0.1.0"
description = "Tensor-backed multi-agent simulation runtime with lineage-aware PPO"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "h5py",
  "numpy",
  "pandas",
  "psutil",
  "pyarrow",
  "pygame-ce>=2.5.6,<2.6",
  "torch",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
tensor-crypt = "tensor_crypt.app.launch:main"

[tool.setuptools]
py-modules = ["config", "main", "run"]

[tool.setuptools.packages.find]
where = ["."]
include = ["tensor_crypt*", "engine*", "viewer*"]
namespaces = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --basetemp=.pytest_tmp_run"
cache_dir = ".pytest_cache_run"

```

### `requirements.txt`

```text
# Runtime dependencies for Tensor Crypt.
# If you need a CUDA-specific PyTorch build, install the appropriate wheel first.
h5py
numpy
pandas
psutil
pyarrow
pygame-ce>=2.5.6,<2.6
torch

```

### `README.md`

```md
# Tensor Crypt

Tensor Crypt is a tensor-backed multi-agent simulation runtime with an interactive pygame-ce viewer. Agents live on a 2D grid with walls and heal zones, perceive the world through batched ray casting, act through per-agent policy/value networks, learn with PPO, reproduce through a binary parent model, and emit structured logs, checkpoints, and validation data.

The project name refers to the repository's dense tensor substrate and its emphasis on durable runtime records such as identity ledgers, lineage, telemetry, and checkpoints. It is a simulation and learning project, not a cryptography library.

## Main capabilities

- Interactive viewer with pan, zoom, selection, overlays, and catastrophe controls
- Procedural map generation with walls and heal zones
- Slot-backed agent storage with canonical UID ownership and lineage tracking
- Bloodline-aware MLP policy/value networks with multiple family topologies
- Batched perception with canonical observations and a legacy observation adapter
- PPO training keyed by UID rather than by slot
- Structured reproduction overlay doctrines: The Ashen Press, The Widow Interval, and The Bloodhold Radius
- Structured telemetry in HDF5, Parquet, JSON, and PyTorch checkpoint files
- Atomic checkpoint publishing with manifests, checksums, and a latest-pointer file
- A pytest suite covering determinism, checkpoint restore, catastrophe scheduling, and runtime invariants

## How the system works

At startup the launcher seeds all random sources, creates a run directory, builds the runtime graph, generates a procedural map, spawns the initial population, and starts the viewer.

Each tick follows the same broad order:

1. Update catastrophe scheduling and apply temporary world modifiers.
2. Build observations for all alive agents.
3. Run each agent's brain to sample actions and value estimates.
4. Resolve movement, collisions, contests, and environment effects.
5. Compute PPO rewards and store transitions.
6. Finalize deaths, evolve the population, handle respawn, and write telemetry.
7. Optionally publish a runtime checkpoint.

The runtime keeps dense tensors for speed, but identity is defined by monotonic UIDs. That distinction matters for lineage, checkpoints, and PPO ownership: slot reuse does not recycle agent identity or optimizer state.

## Repository structure

```text
.
├── config.py                  # Public config compatibility wrapper
├── run.py                     # Primary launch entrypoint
├── main.py                    # Alternate launch entrypoint
├── tensor_crypt/              # Canonical implementation package
│   ├── runtime_config.py      # Canonical config dataclasses and singleton cfg
│   ├── agents/                # Brains and slot-backed registry
│   ├── app/                   # Launch and runtime assembly
│   ├── audit/                 # Determinism and checkpoint probes
│   ├── checkpointing/         # Capture, restore, atomic publish, validation
│   ├── learning/              # PPO
│   ├── population/            # Evolution, reproduction, respawn
│   ├── simulation/            # Engine and catastrophe manager
│   ├── telemetry/             # Run paths, logger, lineage export
│   ├── viewer/                # Pygame viewer
│   └── world/                 # Grid, map generation, perception, physics
├── engine/                    # Legacy compatibility imports
├── viewer/                    # Legacy compatibility imports
├── scripts/
│   ├── benchmark_runtime.py   # Headless benchmark harness
│   ├── run_soak_audit.py      # Headless soak audit
│   └── dump_py_to_text.py     # Source dump helper
├── docs/
│   ├── architecture/          # Structure and compatibility notes
│   ├── reports/               # Audit, validation, and patch reports
│   └── technical_documents/   # Deep technical reference material
└── tests/                     # Pytest suite
```

`tensor_crypt/` is the only implementation root. The repository root keeps a small public surface (`config.py`, `run.py`, `main.py`) plus compatibility-only `engine/` and `viewer/` packages for legacy imports.

## Installation

The repository ships with standard packaging metadata. An editable install keeps imports, scripts, and tests aligned with the checked-out tree.

The viewer backend is intentionally pinned to the `pygame-ce` 2.5.x line. The code imports it as `pygame`, because `pygame-ce` provides the `pygame` module namespace. The checked-in manifests currently require `pygame-ce>=2.5.6,<2.6`, which matches the version line validated in this repository audit while still allowing patch-level updates within 2.5.x.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Quick start

Run the simulation from the repository root:

```bash
python run.py
```

`main.py` is equivalent:

```bash
python main.py
```

Startup prints the selected device, the grid size, the configured population size, and the run directory. The launcher also writes `config.json` and `run_metadata.json` into that run directory before entering the viewer.

## Viewer controls

The viewer binds a small set of direct controls in `viewer.input.InputHandler`:

- `Esc`: quit
- `Space`: pause / resume
- `.`: advance one tick while paused
- `+` / `-`: increase or decrease simulation speed
- `WASD` or arrow keys: pan
- Mouse wheel: zoom at cursor
- Left click: select an agent or heal zone
- `R`: toggle rays
- `B`: toggle HP bars
- `H`: toggle heal-zone overlay
- `G`: toggle grid
- `Shift+1`: toggle The Ashen Press runtime override
- `Shift+2`: toggle The Widow Interval runtime override
- `Shift+3`: toggle The Bloodhold Radius runtime override
- `Shift+0`: clear reproduction doctrine runtime overrides
- `F1`-`F12`: trigger catastrophes manually
- `C`: clear active catastrophes
- `Y`: cycle catastrophe mode
- `U`: arm or disarm the catastrophe scheduler
- `I`: toggle catastrophe panel
- `O`: pause or resume the catastrophe scheduler

## Configuration

The public configuration entry surface is `config.py`, which re-exports the canonical dataclasses and singleton `cfg` from `tensor_crypt/runtime_config.py`.

The file is organized by concern:

- `SIM`: seed, device, run length, and top-level runtime posture
- `GRID` and `MAPGEN`: world size, heal/harm-field composition, and procedural substrate
- `AGENTS`, `TRAITS`, `RESPAWN`, `EVOL`: population size, latent trait budgets/clamps, binary-parent respawn, structured overlay doctrines, and mutation
- `PERCEPT`: ray casting and observation layout
- `BRAIN`: action/value dimensions, bloodline families, topology, and observation-compatibility policy
- `PPO`: reward form, reward gating, rollout/update cadence, and UID ownership enforcement
- `VIEW`: window size, supported startup overlays, and catastrophe UI
- `LOG`, `TELEMETRY`, `CHECKPOINT`, `VALIDATION`: logging cadence, export cadence, checkpoint policy, and audit switches
- `IDENTITY`, `SCHEMA`, `MIGRATION`, `CATASTROPHE`: UID invariant strictness, schema versions, legacy visibility flags, and catastrophe scheduling
- `SIM.EXPERIMENTAL_FAMILY_VMAP_*`: opt-in same-family inference batching for benchmarking on headless workloads without changing the default per-brain ownership-preserving loop

Treat the checked-in values as one concrete scenario, not as universal recommendations. Many settings trade off visibility, logging volume, checkpoint frequency, and runtime cost.

The reproduction surface now includes a structured `RESPAWN.OVERLAYS` subtree. `CROWDING` configures The Ashen Press (crowding-gated reproduction overlay), `COOLDOWN` configures The Widow Interval (parent refractory reproduction overlay), `LOCAL_PARENT` configures The Bloodhold Radius (local lineage parent-selection overlay), and `VIEWER` controls HUD and hotkey exposure for the runtime override surface. When all three doctrines are disabled, the controller falls back to legacy binary-parent selection and placement behavior.

The surface is intentionally narrower than older audit prose may imply. The dead and documentary-only knobs that were not wired have been removed. Two notable special cases remain: `TRAITS.INIT` is a legacy/template container retained for compatibility and documentation even though the live birth path uses latent decoding, and `TELEMETRY.ENABLE_DEEP_LEDGERS` only gates initial root-seed deep-ledger seeding rather than the broader telemetry stack.

The runtime also rejects unsupported or misleading combinations during startup instead of accepting them silently. For example, the current code path requires:

- `SIM.DTYPE == "float32"`
- `AGENTS.SPAWN_MODE == "uniform"`
- `TRAITS.METAB_FORM == "affine_combo"`
- `RESPAWN.MODE == "binary_parented"`
- `PPO.OWNERSHIP_MODE == "uid_strict"`
- `TELEMETRY.LINEAGE_EXPORT_FORMAT == "json"`
- manifest strictness and latest-pointer features to run only on the manifest-publishing atomic path (`ATOMIC_WRITE_ENABLED`, `MANIFEST_ENABLED`, and `SAVE_CHECKPOINT_MANIFEST`)

## Outputs, logs, and checkpoints

Each run creates a timestamped directory under `cfg.LOG.DIR`:

```text
logs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── run_metadata.json
    ├── simulation_data.hdf5
    ├── birth_ledger.parquet
    ├── genealogy.parquet
    ├── life_ledger.parquet
    ├── death_ledger.parquet
    ├── collisions.parquet
    ├── ppo_events.parquet
    ├── tick_summary.parquet
    ├── family_summary.parquet
    ├── catastrophes.parquet
    ├── lineage_graph.json
    ├── brains/
    │   └── brains_tick_<tick>.pt
    └── checkpoints/          # Created when periodic runtime checkpointing is enabled
```

`simulation_data.hdf5` stores agent snapshots, heatmaps, and identity datasets. The run directory is also created with `snapshots/`, `brains/`, and `heatmaps/` subdirectories; in the current logger, snapshots and heatmaps are written into the HDF5 file, while brain state files are written into `brains/`.

Runtime checkpoints are controlled by `cfg.CHECKPOINT`. When periodic checkpointing is enabled, the engine publishes bundle files under the run directory's checkpoint folder using the configured filename prefix. In the current runtime, manifest files and `latest_checkpoint.json` are published only when `ATOMIC_WRITE_ENABLED`, `MANIFEST_ENABLED`, and `SAVE_CHECKPOINT_MANIFEST` are all true. On that path each checkpoint can include:

- a `.pt` bundle
- a manifest file with checksums and metadata
- `latest_checkpoint.json` pointing to the most recent published checkpoint

The checkpoint code validates schema versions, UID bindings, PPO state, and manifest metadata during load. Checkpoint bundles also persist reproduction doctrine runtime state: viewer-toggled doctrine overrides plus The Widow Interval cooldown ledgers are restored on resume instead of snapping back to config defaults.

## Benchmarking

The repository includes a headless benchmark harness:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --output benchmark.json
```

The benchmark configures a small runtime, executes a fixed number of ticks, and writes a JSON summary with elapsed time, ticks per second, memory use, final tick, final alive count, and the run directory.

The harness also exposes the experimental inference fast path so you can compare the canonical loop against the family-local `torch.func` path under identical seeds and workloads:

```bash
python scripts/benchmark_runtime.py --device cpu --ticks 128 --warmup-ticks 16 --experimental-family-vmap-inference --experimental-family-vmap-min-bucket 8 --output benchmark_experimental.json
```

When enabled, the output JSON includes `experimental_family_vmap_inference`, `experimental_family_vmap_min_bucket`, and `inference_path_stats` so benchmark runs can distinguish loop-routed versus vmap-routed slots and buckets.

## Testing and validation

Run the test suite with:

```bash
python -m pytest
```

The repository includes a substantial pytest suite. Based on the test names and helper modules, coverage includes:

- deterministic seeding and repeatable runtime traces
- public and compatibility imports
- observation-shape checks and legacy observation bridging
- bloodline family instantiation and topology checks
- UID ownership, slot reuse, and PPO buffer ownership
- reward gating behavior
- checkpoint round-trip and restore validation
- atomic checkpoint publish and manifest validation
- catastrophe scheduling, replay, and viewer state
- reproduction doctrine behavior, runtime hotkeys, and overlay checkpoint round-trips
- lineage export and telemetry schema checks
- benchmark smoke coverage

There is also a programmatic validation package under `tensor_crypt.audit` with helpers for determinism probes, resume-consistency probes, save-load-save checks, catastrophe replay checks, and a combined final validation suite.

## License

MIT

```

### `tests/test_dependency_governance.py`

```python
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
README = ROOT / "README.md"
PYGAME_CE_SPEC = "pygame-ce>=2.5.6,<2.6"


def _requirements_entries(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _pyproject_dependencies() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"dependencies\s*=\s*\[(?P<body>.*?)\]", text, re.DOTALL)
    assert match is not None, "pyproject.toml is missing a project.dependencies block"
    return re.findall(r'"([^"]+)"', match.group("body"))


def test_runtime_dependency_surfaces_require_pygame_ce():
    pyproject_dependencies = _pyproject_dependencies()
    requirements_dependencies = _requirements_entries(REQUIREMENTS)

    assert PYGAME_CE_SPEC in pyproject_dependencies
    assert PYGAME_CE_SPEC in requirements_dependencies
    assert "pygame" not in pyproject_dependencies
    assert "pygame" not in requirements_dependencies


def test_readme_install_instructions_explain_pygame_ce_namespace():
    readme = README.read_text(encoding="utf-8")

    assert "pygame-ce>=2.5.6,<2.6" in readme
    assert "`pygame-ce`" in readme
    assert "`pygame`" in readme

```

## 10. Any new tests in full

New test added in this audit:
- `tests/test_dependency_governance.py`

Full content:

```python
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
README = ROOT / "README.md"
PYGAME_CE_SPEC = "pygame-ce>=2.5.6,<2.6"


def _requirements_entries(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _pyproject_dependencies() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"dependencies\s*=\s*\[(?P<body>.*?)\]", text, re.DOTALL)
    assert match is not None, "pyproject.toml is missing a project.dependencies block"
    return re.findall(r'"([^"]+)"', match.group("body"))


def test_runtime_dependency_surfaces_require_pygame_ce():
    pyproject_dependencies = _pyproject_dependencies()
    requirements_dependencies = _requirements_entries(REQUIREMENTS)

    assert PYGAME_CE_SPEC in pyproject_dependencies
    assert PYGAME_CE_SPEC in requirements_dependencies
    assert "pygame" not in pyproject_dependencies
    assert "pygame" not in requirements_dependencies


def test_readme_install_instructions_explain_pygame_ce_namespace():
    readme = README.read_text(encoding="utf-8")

    assert "pygame-ce>=2.5.6,<2.6" in readme
    assert "`pygame-ce`" in readme
    assert "`pygame`" in readme

```

## 11. Validation commands run and outcomes

Repository inspection and environment verification:
- `git status --short`
  - confirmed the repository already had unrelated in-progress viewer/runtime changes before this audit pass
- `rg -n "pygame-ce|pygame" ...`
  - confirmed manifests still said `pygame` while the runtime/tests used the `pygame` import namespace throughout
- `python -` with `import importlib.metadata as md; import pygame`
  - confirmed live environment is `pygame-ce 2.5.6`
  - confirmed imported module path is the `pygame` namespace provided by `pygame-ce`
  - confirmed no `pygame` distribution is installed separately

Primary pytest validation that passed:
- `python -m pytest -q tests/test_viewer_color_semantics.py tests/test_catastrophe_scheduler_controls.py tests/test_dependency_governance.py tests/test_imports_and_compat.py --basetemp=.pytest_tmp_release_audit_b -o cache_dir=.pytest_tmp_release_audit_b/cache`
  - outcome: `20 passed, 1 warning in 0.84s`
  - warning was a pytest cache-path access warning inside this sandbox, not a repo test failure

Direct viewer/runtime probes that passed:
- manual runtime-build probe using workspace-local run directories and dummy SDL drivers
  - outcome: runtime built successfully, viewer initialized successfully
- manual viewer layout probe covering resize geometry, inspector overflow, scrolling, and draw smoke
  - outcome: `viewer_layout_probe_ok`
- manual engine/viewer integration probe covering engine stepping, invariant check, viewer drawing, and resize-event handling
  - outcome: `engine_viewer_probe_ok`

Validation attempts blocked by environment filesystem behavior:
- `python -m pytest tests/test_viewer_layout_cleanup.py::test_layout_rects_stay_inside_window_across_sizes ...`
  - outcome: errored/hung under this sandbox's temp/cache filesystem behavior before producing a stable traceback
  - workaround used: equivalent workspace-local direct probe passed
- `python -m pip wheel --no-deps --no-build-isolation . -w audit_tmp\pkg_wheels_release_audit`
  - outcome: failed with `PermissionError` in pip build-tracker temp handling
- direct setuptools backend build via `setuptools.build_meta.build_wheel()` / `build_sdist()`
  - outcome: progressed through build metadata generation and egg-info emission, then failed with `PermissionError` while writing temporary archive output in this environment
  - metadata confirmation still succeeded through generated `tensor_crypt.egg-info/requires.txt` and `PKG-INFO`

Metadata verification that passed after the build attempts:
- inspected `tensor_crypt.egg-info/requires.txt`
  - confirmed `pygame-ce<2.6,>=2.5.6`
- inspected `tensor_crypt.egg-info/PKG-INFO`
  - confirmed `Requires-Python: >=3.10`
  - confirmed `Requires-Dist: pygame-ce<2.6,>=2.5.6`

## 12. Residual risks / caveats

- Full wheel/sdist archive creation could not be completed in this environment because temporary build-output writes were denied by the filesystem sandbox. This appears environmental rather than repository-caused, because metadata generation proceeded far enough to emit correct dependency metadata.
- One viewer-layout pytest slice also misbehaved under sandbox temp/cache handling. Equivalent direct workspace-local runtime probes succeeded, so this does not currently point to a viewer regression.
- The repository has no in-repo CI/workflow definition. Dependency alignment is now protected by manifests plus the new repository test, but there is still no automated external workflow file in the repo to enforce it on every push.
- This audit did not modify the already-present viewer feature implementation files, because they were already aligned with the planned feature set in the working tree.
- The working tree contains unrelated pre-existing modifications and untracked files outside this audit's four-file patch set; those were preserved.

## 13. Final yes/no answer

Yes.

Everything that was planned is now applied as intended in the current working tree, with one qualification: the viewer feature implementation was already present before this audit, and this audit's concrete fix was to bring the dependency / packaging / install-documentation surface into alignment with that implementation by explicitly governing `pygame-ce`.

## External sources used

- `pygame-ce` project page / release history: https://pypi.org/project/pygame-ce/
- `pygame-ce` display documentation (`get_desktop_sizes`, fullscreen/window sizing guidance): https://pyga.me/docs/ref/display.html
- `pygame-ce` event documentation (`MOUSEWHEEL`, `WINDOWRESIZED`, `VIDEORESIZE`): https://pyga.me/docs/ref/event.html
- `pygame` project page for comparison of the separate distribution surface: https://pypi.org/project/pygame/
