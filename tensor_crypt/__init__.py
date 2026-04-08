"""
Repository-root namespace bridge for the Tensor Crypt implementation package.

The canonical implementation lives in `src/tensor_crypt`. This module keeps
`import tensor_crypt.*` working from a source checkout without requiring users
or tests to modify their import paths.
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
