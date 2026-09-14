"""W-4 and W-7.

W-4: the §4.7 inversion (28,170 / 19,645 = 1.43x) is computed in RAW excess
     views across pages with very different baselines, which §3.2 says is the
     one thing raw excess cannot do. Recompute in baseline-days.

W-7: "lookups vs media: not shown to differ" is inferred from two overlapping
     confidence intervals. Overlap is not a test. Bootstrap the RATIO of the
     two half-lives directly and see whether its interval contains 1.
"""
import datetime as _dt
import sys

import numpy as np

sys.path.insert(0, r"D:\ins respinse\warningshot\src")

from warningshot.core import metrics as m          # noqa: E402
from warningshot.core import regression as rg      # noqa: E402
from warningshot.data import modalities as mo      # noqa: E402
from warningshot.data import sources as src        # noqa: E402

print("=" * 76)
print("W-4  Recompute the inversion in baseline-days, per the paper's own rule")
print("=" * 76)


def excess_and_base(page, t0, window_days, baseline_len=30, gap=7):
    end = _dt.datetime.strptime(t0, "%Y%m%d") - _dt.timedelta(days=gap)
    start = end - _dt.timedelta(days=baseline_len - 1)
    a = (start - _dt.timedelta(days=20)).strftime("%Y%m%d")
    b = (_dt.datetime.strptime(t0, "%Y%m%d")
         + _dt.timedelta(days=window_days + 5)).strftime("%Y%m%d")
    s = src.pageviews(page, a, b)
    if not s:
        return None
    base = m.baseline(s, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    exc = m.excess(s, base, m.window_from(t0, window_days))
    return base, exc


JUL_T0, JUN_T0, W = "20260721", "20260613", 3      # 48-hour window = 3 days

rows = {}
for label, page, t0 in [("victim  Hugging_Face", "Hugging_Face", JUL_T0),
                        ("dev     OpenAI", "OpenAI", JUL_T0),
                        ("dev     Anthropic (Jun)", "Anthropic", JUN_T0)]:
    r = excess_and_base(page, t0, W)
    if r:
        base, exc = r
        rows[label] = (base, exc, exc / base)
        print(f"  {label:26s} base={base:8.0f}/day  excess={exc:8.0f}  "
              f"baseline-days={exc / base:7.2f}")

v = rows["victim  Hugging_Face"]
a = rows["dev     Anthropic (Jun)"]
o = rows["dev     OpenAI"]

print()
print(f"  RAW EXCESS      victim/Anthropic = {v[1] / a[1]:.2f}x   "
      f"(the paper's 1.43x)")
print(f"  BASELINE-DAYS   victim/Anthropic = {v[2] / a[2]:.2f}x")
print(f"  RAW EXCESS      Anthropic/OpenAI = {a[1] / o[1]:.2f}x   "
      f"(the published ~7x)")
print(f"  BASELINE-DAYS   Anthropic/OpenAI = {a[2] / o[2]:.2f}x")
print()
if (v[1] / a[1] > 1) == (v[2] / a[2] > 1):
    print("  VERDICT: the inversion SURVIVES the change of units (direction")
    print("  unchanged), though the magnitude moves. W-4 is answered, not fatal.")
else:
    print("  VERDICT: the inversion REVERSES in baseline-days. W-4 is FATAL to")
    print("  the paper's headline replication finding as computed.")

print()
print("=" * 76)
print("W-7  Bootstrap the RATIO of half-lives instead of comparing intervals")
print("=" * 76)


def series(kind):
    if kind == "lookups":
        ch = mo.lookups("Hugging_Face", JUL_T0, horizon=25)
    else:
        ch = mo.media('"Hugging Face" OpenAI', JUL_T0, horizon=25)
    t, y = ch["t"], ch["y"]
    k = y > 0
    return t[k], y[k]


tl, yl = series("lookups")
tm, ym = series("media")
fl, fm = rg.fit_exponential(tl, yl), rg.fit_exponential(tm, ym)
print(f"  lookups t_half = {fl['half_life']:.3f} d")
print(f"  media   t_half = {fm['half_life']:.3f} d")
print(f"  point ratio media/lookups = {fm['half_life'] / fl['half_life']:.3f}")


def block_resample(t, y, fit, rng, block=4):
    pred = np.log(fit["amplitude"]) - t / fit["tau"]
    resid = np.log(y) - pred
    N = resid.size
    nb = int(np.ceil(N / block))
    starts = rng.integers(0, N - block + 1, size=nb)
    r = np.concatenate([resid[s:s + block] for s in starts])[:N]
    return np.exp(pred + r)


rng = np.random.default_rng(4)
ratios = []
for _ in range(2000):
    a1 = rg.fit_exponential(tl, block_resample(tl, yl, fl, rng))
    b1 = rg.fit_exponential(tm, block_resample(tm, ym, fm, rng))
    if a1 and b1 and a1["half_life"] > 0:
        ratios.append(b1["half_life"] / a1["half_life"])
ratios = np.array(ratios)
lo, hi = np.percentile(ratios, [2.5, 97.5])
print(f"  bootstrap ratio 95% CI = ({lo:.3f}, {hi:.3f})  n={ratios.size}")
print(f"  contains 1.0? {'YES' if lo <= 1.0 <= hi else 'NO'}")
print()
if lo <= 1.0 <= hi:
    print("  VERDICT: the ratio interval contains 1 -> 'not shown to differ' is")
    print("  correctly supported, now by the RIGHT test. W-7 is a method fix,")
    print("  not a result change.")
else:
    print("  VERDICT: the ratio interval EXCLUDES 1 -> the two channels DO")
    print("  differ, and the paper's 'not shown to differ' was an artifact of")
    print("  the overlap heuristic. W-7 changes a stated conclusion.")
