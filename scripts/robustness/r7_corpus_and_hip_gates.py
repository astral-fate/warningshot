"""Two feasibility gates for the 'radical alternatives' search.

GATE A -- multi-event corpus (genai_incidents -> Wikipedia):
  Does a landmark, day-dated AI incident actually move the Wikipedia page of
  the organisation involved? If not, the n=2 limitation cannot be fixed this
  way, and the corpus is unusable for attention-decay work regardless of size.

GATE B -- HIP on a clean two-impulse window (excluding the Nvidia acquisition):
  Is the supercritical endo=8.47 a property of the model on short series, or
  an artifact of the contaminating third spike?
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\ins respinse\warningshot\src")
sys.path.insert(0, r"D:\ins respinse\repos\hip-popularity")

from warningshot.core import metrics as m          # noqa: E402
from warningshot.data import sources as src        # noqa: E402

print("=" * 74)
print("GATE A -- do landmark incidents move the involved org's Wikipedia page?")
print("=" * 74)

# Hand-picked from genai_incidents landmark+day-dated 2026 rows, choosing only
# incidents whose actor/victim plausibly HAS a Wikipedia article. This is the
# best case for the corpus; if it fails here it fails everywhere.
CASES = [
    ("2026-01-07", "Waymo",      "Waymo robotaxi entered Phoenix light-rail tracks"),
    ("2026-01-07", "Grok_(chatbot)", "Grok generated false images of ICE agent"),
    ("2026-01-05", "Perplexity_AI", "Perplexity misstated CLL research"),
    ("2026-01-02", "Tencent",    "Tencent Yuanbao chatbot insulted user"),
]

def ratio(page, date):
    y, mo, d = date.split("-")
    iso = f"{y}{mo}{d}"
    a = m.date_range(f"{y}{mo}01", iso)
    start = (m.date_range("20251101", iso))[0]
    s = src.pageviews(page, "20251201", f"{y}{mo}28")
    if not s:
        return None
    pre = [v for k, v in s.items() if k < iso]
    if len(pre) < 20:
        return None
    base = float(np.median(pre[-30:]))
    win = [s.get(dd, 0) for dd in m.date_range(iso, m.window_from(iso, 4)[-1])]
    return base, max(win), (max(win) / base if base else None)

for date, page, desc in CASES:
    r = ratio(page, date)
    if r is None:
        print(f"  {page:16s} {date}  NO DATA")
        continue
    base, pk, rt = r
    verdict = "MOVED" if rt and rt >= 2.0 else ("weak" if rt and rt >= 1.3 else "FLAT")
    print(f"  {page:16s} {date}  base={base:8.0f} peak={pk:8.0f} ratio={rt:5.2f}x  {verdict}")
    print(f"      {desc[:66]}")

print()
print("=" * 74)
print("GATE B -- HIP on the clean two-impulse window (ends 2 Sep, pre-Nvidia)")
print("=" * 74)

START, END = "20260715", "20260902"
days = m.date_range(START, END)
s = src.pageviews("Hugging_Face", START, END)
views = np.array([s.get(d, 0) for d in days], dtype=float)
base = float(np.median(views[:5]))
y = np.maximum(views - base, 0.0)
SHOCKS = {"20260721": 1.0, "20260826": 1.0}
x = np.array([SHOCKS.get(d, 0.0) for d in days], dtype=float)
print(f"series {len(days)} days, baseline={base:.0f}, peak={y.max():.0f} on "
      f"{days[int(np.argmax(y))]}")

from pyhip import HIP  # noqa: E402
n = len(y)
hip = HIP()
hip.initial(list(x), list(y), n - 8, 8, num_initialization=6)
hip.fit_with_bfgs()
hip.print_parameters()
endo = hip.get_endo()
print(f"\nendo (branching factor) = {endo:.3f}")
print("  interpretation:", "SUPERCRITICAL (>1) -- degenerate, not physical"
      if endo > 1 else "subcritical (<1) -- physically sensible")
