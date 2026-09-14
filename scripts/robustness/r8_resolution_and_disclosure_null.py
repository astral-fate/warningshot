"""Test reviewer weaknesses W-5 and W-3 empirically.

W-5: the 12-21x spread may be an artifact of measurement resolution -- the
     engagement channel is fitted hourly, the others daily. A daily series
     cannot resolve a sub-day half-life. Control: bin HN comments to DAILY
     and refit. If the daily-binned half-life lands near 5 days, the spread
     is substantially an instrument effect.

W-3: "the 16 July disclosure moved public attention by nothing measurable"
     rests on a single day at 1.22x baseline, with no distribution of
     ordinary daily ratios behind it. 1.22x is uninterpretable until we know
     how often the page crosses it anyway.
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\ins respinse\warningshot\src")

from warningshot.core import metrics as m          # noqa: E402
from warningshot.core import regression as rg      # noqa: E402
from warningshot.data import modalities as mo      # noqa: E402
from warningshot.data import sources as src        # noqa: E402

print("=" * 76)
print("W-5  Is the cross-channel spread an artifact of hourly vs daily resolution?")
print("=" * 76)

ch = mo.engagement("48997548")
hourly = np.array(ch["hourly"], dtype=float)           # 48 hourly buckets
th = np.arange(1.0, hourly.size + 1.0)
f_h = rg.fit_exponential(th, hourly)
print(f"  HOURLY  bins n={hourly.size:3d}  t_half = {f_h['half_life']:6.2f} h "
      f"= {f_h['half_life'] / 24:5.3f} d   R2={f_h['r2']:.3f}")

# Same comments, binned to days, over the full thread span.
times = np.asarray(sorted(src.hn_comment_times("48997548")), dtype=float)
rel_h = (times - times[0]) / 3600.0
span_d = int(np.ceil(rel_h.max() / 24.0))
daily = np.array([int(((rel_h >= 24 * k) & (rel_h < 24 * (k + 1))).sum())
                  for k in range(span_d)], dtype=float)
td = np.arange(1.0, daily.size + 1.0)
f_d = rg.fit_exponential(td, daily)
print(f"  DAILY   bins n={daily.size:3d}  t_half = {f_d['half_life']:6.3f} d"
      f"                 R2={f_d['r2']:.3f}")
print(f"  daily counts: {[int(x) for x in daily]}")

print()
print(f"  lookups half-life (paper) = 5.450 d")
print(f"  ratio lookups / engagement-HOURLY = {5.450 / (f_h['half_life'] / 24):6.1f}x")
print(f"  ratio lookups / engagement-DAILY  = {5.450 / f_d['half_life']:6.1f}x")
print()
if f_d["half_life"] > 3.0:
    print("  VERDICT: daily-binned HN half-life is DAYS -> the spread collapses.")
    print("           W-5 would be CONFIRMED: the gap is largely instrument.")
else:
    print("  VERDICT: daily-binned HN half-life stays SUB-DAY-ish -> the spread")
    print("           is not merely a binning artifact. W-5 partially answered.")

print()
print("=" * 76)
print("W-3  Is 1.22x on 16 Jul actually unremarkable for this page?")
print("=" * 76)

s = src.pageviews("Hugging_Face", "20260101", "20260715")
days = sorted(s)
vals = np.array([s[d] for d in days], dtype=float)

# Same construction as the paper: each day's views over a trailing 30-day median.
ratios, rdays = [], []
for i in range(30, len(days)):
    base = float(np.median(vals[i - 30:i]))
    if base > 0:
        ratios.append(vals[i] / base)
        rdays.append(days[i])
ratios = np.array(ratios)

obs = 1.22
pct = 100.0 * float((ratios < obs).mean())
print(f"  window: {rdays[0]}..{rdays[-1]}  n={ratios.size} days")
print(f"  daily ratio distribution: min={ratios.min():.2f} "
      f"p50={np.percentile(ratios, 50):.2f} p75={np.percentile(ratios, 75):.2f} "
      f"p90={np.percentile(ratios, 90):.2f} max={ratios.max():.2f}")
print(f"  observed 16 Jul ratio = {obs:.2f}")
print(f"  percentile of {obs:.2f} in ordinary daily variation = {pct:.1f}")
print(f"  days at or above {obs:.2f}x: {int((ratios >= obs).sum())} of {ratios.size} "
      f"({100 * (ratios >= obs).mean():.0f}%)")
print()
if pct < 90:
    print("  VERDICT: 1.22x is NOT unusual for this page. The claim 'moved")
    print("           nothing measurable' is supported -- but only because 1.22x")
    print("           sits inside ordinary noise, which is what must be reported.")
else:
    print("  VERDICT: 1.22x is actually high for this page; the 'nothing")
    print("           measurable' claim would be WRONG.")
