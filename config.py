"""Public compatibility wrapper for the canonical Tensor Crypt config."""

from tensor_crypt.runtime_config import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
