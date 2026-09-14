"""Feasibility gate: can pyhip (Rizoiu et al. reference implementation) fit the
two-shock Hugging_Face attention series?

x = exogenous stimulus series (the two disclosure/report shocks)
y = endogenous response  (daily Wikipedia pageviews above baseline)

If this fits, review finding M4 -- "the two-impulse limitation is a solved
problem you didn't apply" -- becomes an implementable result rather than a
future-work note.
"""
import datetime as dt
import sys

import numpy as np

sys.path.insert(0, r"D:\ins respinse\warningshot\src")
sys.path.insert(0, r"D:\ins respinse\repos\hip-popularity")

from warningshot.core import metrics as m          # noqa: E402
from warningshot.data import sources as src        # noqa: E402

FMT = "%Y%m%d"


def daily(page, start, end):
    s = src.pageviews(page, start, end)
    days = m.date_range(start, end)
    return days, np.array([s.get(d, 0) for d in days], dtype=float)


# Full window spanning BOTH impulses: 15 Jul -> 15 Sep 2026
START, END = "20260715", "20260903"
days, views = daily("Hugging_Face", START, END)
base = float(np.median(views[:5]))          # pre-event level
y = np.maximum(views - base, 0.0)

# Exogenous shocks: the events that injected attention, as unit impulses.
# 21 Jul = OpenAI attribution; 26 Aug = OpenAI report + METR/Redwood report.
SHOCKS = {"20260721": 1.0, "20260826": 1.0}
x = np.array([SHOCKS.get(d, 0.0) for d in days], dtype=float)

print(f"series: {len(days)} days  {days[0]}..{days[-1]}")
print(f"baseline={base:.0f}  peak={y.max():.0f} on {days[int(np.argmax(y))]}")
print(f"exogenous impulses at: {[d for d in days if SHOCKS.get(d)]}")
print(f"y nonzero days: {int((y > 0).sum())}")

try:
    from pyhip import HIP
except Exception as exc:
    print(f"IMPORT FAIL: {exc}")
    sys.exit(1)
print("pyhip imported OK (python 3 compatible)")

# HIP wants train/test split; our series is short so use most of it for train.
n = len(y)
num_train = n - 10
num_test = 10
print(f"num_train={num_train} num_test={num_test}")

try:
    hip = HIP()
    hip.initial(list(x), list(y), num_train, num_test, num_initialization=6)
    hip.fit_with_bfgs()
    print("\n=== FIT SUCCEEDED ===")
    hip.print_parameters()
    p = hip.get_parameters()
    print(f"\nparameters: {p}")
    print(f"endo (branching factor): {hip.get_endo():.4f}")
except Exception as exc:
    import traceback
    print(f"\n=== FIT FAILED: {type(exc).__name__}: {exc} ===")
    traceback.print_exc(limit=3)
