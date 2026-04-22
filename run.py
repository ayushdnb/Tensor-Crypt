"""Repository-root launch entrypoint for Tensor Crypt.

`run.py` is the canonical root-level start surface for repository users. All
startup logic lives in `tensor_crypt.app.launch` so imports do not depend on
the current working directory.
"""

from tensor_crypt.app.launch import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
