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
    checkpoint_path: str | None = None
    checkpoint_error: str | None = None
    logger_closed: bool = False
    logger_error: str | None = None
    metadata_error: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


def finalize_runtime(runtime, *, close_reason: str = "normal_exit") -> RuntimeFinalizationResult:
    """Flush telemetry, optionally publish a shutdown checkpoint, and close the logger once."""
    previous = getattr(runtime, "_lifecycle_finalization_result", None)
    if getattr(runtime, "_lifecycle_finalized", False):
        if previous is not None:
            return RuntimeFinalizationResult(
                already_finalized=True,
                checkpoint_path=previous.checkpoint_path,
                checkpoint_error=previous.checkpoint_error,
                logger_closed=previous.logger_closed,
                logger_error=previous.logger_error,
                metadata_error=previous.metadata_error,
                errors=list(previous.errors),
            )
        return RuntimeFinalizationResult(already_finalized=True)
    if getattr(runtime, "_lifecycle_finalizing", False):
        return RuntimeFinalizationResult(already_finalized=True)

    runtime._lifecycle_finalizing = True
    result = RuntimeFinalizationResult()
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


__all__ = ["RuntimeFinalizationResult", "finalize_runtime"]
