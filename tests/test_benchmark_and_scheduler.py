import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tensor_crypt.checkpointing import resolve_latest_checkpoint_bundle
from tensor_crypt.checkpointing.runtime_checkpoint import load_runtime_checkpoint
from tensor_crypt.config_bridge import cfg


def test_periodic_runtime_checkpoint_scheduler_retains_latest_files(runtime_builder):
    cfg.CHECKPOINT.SAVE_EVERY_TICKS = 2
    cfg.CHECKPOINT.KEEP_LAST = 2
    cfg.CHECKPOINT.MANIFEST_ENABLED = True
    cfg.CHECKPOINT.SAVE_CHECKPOINT_MANIFEST = True
    cfg.CHECKPOINT.WRITE_LATEST_POINTER = True

    runtime = runtime_builder(seed=401, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(6):
        runtime.engine.step()

    checkpoint_dir = Path(runtime.data_logger.run_dir) / cfg.CHECKPOINT.DIRECTORY_NAME
    bundle_paths = sorted(checkpoint_dir.glob(f"{cfg.CHECKPOINT.FILENAME_PREFIX}*{cfg.CHECKPOINT.BUNDLE_FILENAME_SUFFIX}"))

    assert len(bundle_paths) == 2
    assert bundle_paths[-1].name.endswith("00000006.pt")

    resolved = resolve_latest_checkpoint_bundle(checkpoint_dir)
    assert resolved == bundle_paths[-1]

    loaded = load_runtime_checkpoint(checkpoint_dir)
    assert int(loaded["engine_state"]["tick"]) == 6


def test_tick_summary_cadence_and_buffer_flush(runtime_builder):
    cfg.TELEMETRY.SUMMARY_EXPORT_CADENCE_TICKS = 2
    cfg.TELEMETRY.PARQUET_BATCH_ROWS = 16

    runtime = runtime_builder(seed=402, agents=6, walls=0, hzones=0, update_every=99, batch_size=99, mini_batches=1)
    for _ in range(4):
        runtime.engine.step()
    runtime.data_logger.close(runtime.registry)

    df = pd.read_parquet(runtime.data_logger.tick_summary_path)
    assert list(df["tick"]) == [0, 2]


def test_benchmark_runtime_script_smoke(tmp_path):
    output_path = tmp_path / "benchmark.json"
    log_dir = tmp_path / "benchmark_logs"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_runtime.py",
            "--device",
            "cpu",
            "--ticks",
            "4",
            "--warmup-ticks",
            "1",
            "--width",
            "10",
            "--height",
            "10",
            "--agents",
            "4",
            "--walls",
            "0",
            "--hzones",
            "0",
            "--update-every",
            "8",
            "--batch-size",
            "4",
            "--mini-batches",
            "1",
            "--epochs",
            "1",
            "--log-dir",
            str(log_dir),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["device"] == "cpu"
    assert result["ticks"] == 4
    assert result["final_tick"] == 5
    assert result["ticks_per_sec"] > 0.0
    assert Path(result["run_dir"]).exists()
