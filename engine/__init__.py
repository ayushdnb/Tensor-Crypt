"""
Legacy compatibility package.

Only the package stub remains at repository root. The thin re-export modules
now live under `src/engine`, while this package extends
its import path so legacy imports such as `engine.physics` still resolve.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_compat_dir = Path(__file__).resolve().parents[1] / "src" / "engine"
if _compat_dir.is_dir():
    compat_dir_str = str(_compat_dir)
    if compat_dir_str not in __path__:
        __path__.append(compat_dir_str)
