"""
Root launch entrypoint.

`run.py` remains the obvious public start surface for repository users, but all
real startup logic now lives in the package so imports do not depend on the
current working directory.
"""

from tensor_crypt.app.launch import main


if __name__ == "__main__":
    main()
