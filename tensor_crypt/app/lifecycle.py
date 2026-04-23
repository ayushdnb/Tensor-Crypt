"""Runtime lifecycle finalization for Tensor Crypt."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config_bridge import cfg
from ..simulation.engine import SAVE_REASON_SHUTDOWN
from ..telemetry.run_paths import update_session_metadata


@dataclass
class RuntimeFinalizationResult:
    """Best-effort lifecycle finalization summary."""

    already_finalized: bool = False
    close_reason: str = "normal_exit"
    tick: int | None = None
    alive_agents: int | None = None
    run_dir: str | None = None
    checkpoint_enabled: bool = False
    checkpoint_attempted: bool = False
    checkpoint_path: str | None = None
    checkpoint_error: str | None = None
    logger_closed: bool = False
    logger_error: str | None = None
    metadata_error: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


def _runtime_alive_count(runtime) -> int | None:
    getter = getattr(runtime.registry, "get_num_alive", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def format_runtime_finalization_summary(result: RuntimeFinalizationResult) -> str:
    """Format the operator-facing shutdown details printed by the viewer path."""
    lines = [
        "Runtime shutdown details:",
        f"  reason: {result.close_reason}",
        f"  tick: {'unknown' if result.tick is None else result.tick}",
        f"  alive_agents: {'unknown' if result.alive_agents is None else result.alive_agents}",
        f"  run_dir: {result.run_dir or 'unknown'}",
    ]
    if result.checkpoint_path:
        lines.append(f"  checkpoint: OK | {result.checkpoint_path}")
    elif result.checkpoint_error:
        lines.append(f"  checkpoint: FAILED | {result.checkpoint_error}")
    elif result.checkpoint_attempted:
        lines.append("  checkpoint: unavailable")
    elif result.checkpoint_enabled:
        lines.append("  checkpoint: not attempted")
    else:
        lines.append("  checkpoint: disabled")

    if result.logger_closed:
        lines.append("  telemetry_close: OK")
    elif result.logger_error:
        lines.append(f"  telemetry_close: FAILED | {result.logger_error}")
    else:
        lines.append("  telemetry_close: not closed")

    if result.metadata_error:
        lines.append(f"  session_metadata: FAILED | {result.metadata_error}")
    if result.errors:
        lines.append(f"  errors: {len(result.errors)} | {'; '.join(result.errors)}")
    if result.already_finalized:
        lines.append("  already_finalized: true")
    return "\n".join(lines)


def finalize_runtime(
    runtime,
    *,
    close_reason: str = "normal_exit",
    print_summary: bool = False,
) -> RuntimeFinalizationResult:
    """Flush telemetry, optionally publish a shutdown checkpoint, and close the logger once."""
    previous = getattr(runtime, "_lifecycle_finalization_result", None)
    if getattr(runtime, "_lifecycle_finalized", False):
        if previous is not None:
            result = RuntimeFinalizationResult(
                already_finalized=True,
                close_reason=previous.close_reason,
                tick=previous.tick,
                alive_agents=previous.alive_agents,
                run_dir=previous.run_dir,
                checkpoint_enabled=previous.checkpoint_enabled,
                checkpoint_attempted=previous.checkpoint_attempted,
                checkpoint_path=previous.checkpoint_path,
                checkpoint_error=previous.checkpoint_error,
                logger_closed=previous.logger_closed,
                logger_error=previous.logger_error,
                metadata_error=previous.metadata_error,
                errors=list(previous.errors),
            )
        else:
            result = RuntimeFinalizationResult(already_finalized=True, close_reason=str(close_reason))
        if print_summary:
            print(format_runtime_finalization_summary(result))
        return result
    if getattr(runtime, "_lifecycle_finalizing", False):
        result = RuntimeFinalizationResult(already_finalized=True, close_reason=str(close_reason))
        if print_summary:
            print(format_runtime_finalization_summary(result))
        return result

    runtime._lifecycle_finalizing = True
    result = RuntimeFinalizationResult(
        close_reason=str(close_reason),
        tick=int(getattr(runtime.engine, "tick", -1)),
        alive_agents=_runtime_alive_count(runtime),
        run_dir=str(getattr(runtime, "run_dir", getattr(runtime.data_logger, "run_dir", ""))),
        checkpoint_enabled=bool(cfg.CHECKPOINT.ENABLE_SHUTDOWN_CHECKPOINT and cfg.CHECKPOINT.ENABLE_SUBSTRATE_CHECKPOINTS),
    )
    raise_after_close: Exception | None = None
    try:
        try:
            if not getattr(runtime.data_logger, "_closed", False):
                runtime.data_logger.flush_parquet_buffers()
                h5_file = getattr(runtime.data_logger, "h5_file", None)
                if h5_file is not None:
                    h5_file.flush()
        except Exception as exc:
            result.errors.append(f"telemetry_flush:{exc}")

        if cfg.CHECKPOINT.ENABLE_SHUTDOWN_CHECKPOINT:
            result.checkpoint_attempted = True
            try:
                checkpoint_path = runtime.engine.publish_runtime_checkpoint(SAVE_REASON_SHUTDOWN, force=True)
                result.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
            except Exception as exc:
                result.checkpoint_error = str(exc)
                result.errors.append(f"shutdown_checkpoint:{exc}")
                if not cfg.CHECKPOINT.SHUTDOWN_CHECKPOINT_BEST_EFFORT:
                    raise_after_close = exc

        try:
            runtime.data_logger.close(
                runtime.registry,
                finalize_open_lives=bool(cfg.TELEMETRY.FINALIZE_OPEN_LIVES_ON_SESSION_CLOSE),
                close_reason=close_reason,
                close_tick=int(runtime.engine.tick),
            )
            result.logger_closed = True
        except Exception as exc:
            result.logger_error = str(exc)
            result.errors.append(f"logger_close:{exc}")

        try:
            update_session_metadata(
                runtime.session_plan,
                finalization_close_reason=str(close_reason),
                finalization_checkpoint_path=result.checkpoint_path,
                finalization_checkpoint_error=result.checkpoint_error,
                finalization_logger_closed=bool(result.logger_closed),
                finalization_logger_error=result.logger_error,
                finalization_errors=list(result.errors),
            )
        except Exception as exc:
            result.metadata_error = str(exc)
            result.errors.append(f"session_metadata:{exc}")

        if raise_after_close is not None:
            raise RuntimeError(f"Shutdown checkpoint failed: {raise_after_close}") from raise_after_close
        return result
    finally:
        runtime._lifecycle_finalized = True
        runtime._lifecycle_finalizing = False
        runtime._lifecycle_finalization_result = result
        if print_summary:
            print(format_runtime_finalization_summary(result))


__all__ = ["RuntimeFinalizationResult", "finalize_runtime", "format_runtime_finalization_summary"]
