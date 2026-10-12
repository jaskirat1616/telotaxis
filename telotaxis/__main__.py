import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

from telotaxis.cli import main

if __name__ == "__main__":
    main()
