"""Repository paths, resolved once.

Every module previously derived its own directories from `__file__`, which
breaks the moment a file moves between folders. Resolving the root in one
place means a restructure cannot silently repoint the cache or the results.
"""

import os
from pathlib import Path

# paths.py -> warningshot -> src -> repository root
ROOT = Path(__file__).resolve().parents[2]

CACHE = ROOT / "cache"      # raw API responses, refreshable, not a deliverable
DATA = ROOT / "data"        # prepared datasets committed for reuse
RESULTS = ROOT / "results"  # committed numbers that verify.py checks
OUT = ROOT / "out"          # generated figures and tables
DOCS = ROOT / "docs"

# External corpora live outside this repo and are read-only. Overridable so a
# reproducer can point at their own clone instead of this machine's layout.
REPOS = Path(os.environ.get("WARNINGSHOT_REPOS", ROOT.parent / "repos"))
EGRESS_BENCH = REPOS / "agent-egress-bench"
ESCAPE_BENCH = REPOS / "sandbox_escape_bench"


def ensure(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
