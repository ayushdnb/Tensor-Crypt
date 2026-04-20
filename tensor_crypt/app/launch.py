"""Single-brain-vmap launch entrypoint.

This module keeps user-facing startup logic separate from the simulation
subsystems. It owns only launch-time concerns:
- determinism setup
- run-directory setup
- diagnostic prints
- viewer start

It intentionally does not own simulation rules.
"""

from __future__ import annotations

from .runtime import build_runtime, setup_determinism
from ..config_bridge import cfg
from ..runtime_config import apply_experimental_single_family_launch_defaults
from ..telemetry.run_paths import create_run_directory


def main() -> None:
    apply_experimental_single_family_launch_defaults()
    setup_determinism()
    run_dir = create_run_directory()

    print(f"Device: {cfg.SIM.DEVICE}")
    print(
        "Startup mode: self-centric single-family vmap preset "
        f"({cfg.BRAIN.EXPERIMENTAL_BRANCH_FAMILY})"
    )
    print(
        f"Starting simulation with {cfg.AGENTS.N} agents on "
        f"{cfg.GRID.W}x{cfg.GRID.H} grid"
    )
    print(f"Logging all data to: {run_dir}")

    runtime = build_runtime(run_dir)
    runtime.viewer.run()


__all__ = ["main"]
