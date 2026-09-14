"""Put `src` on the path so tests run without an install step.

A `pip install -e .` also works, but this repo is meant to be runnable from a
clone with nothing but the standard scientific stack present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
