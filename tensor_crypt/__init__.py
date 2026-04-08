"""
Repository-root namespace shim for the Tensor Crypt implementation package.

The canonical implementation now lives in `src/tensor_crypt`. This shim keeps
`import tensor_crypt.*` working when running directly from repository root
without requiring users/tests to alter their import paths.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_src_pkg_dir = Path(__file__).resolve().parents[1] / "src" / "tensor_crypt"
if _src_pkg_dir.is_dir():
    src_pkg_str = str(_src_pkg_dir)
    if src_pkg_str not in __path__:
        __path__.append(src_pkg_str)
