"""W-2: the headline engagement half-life has no null.

The reviewer's objection: a single Hacker News thread decays in hours BY
CONSTRUCTION -- age-based ranking demotion plus a finite commenter pool. If
7.05 h sits at the median of comparable threads, the "two clocks" result is a
fact about HN thread lifecycles, not about this warning shot.

Test: sample comparable high-scoring HN threads, fit the same exponential to
their hourly comment-arrival counts, and report where 7.05 h falls.

Comparability rule, fixed BEFORE looking at any half-life: front-page-scale
threads (>= 500 points) with >= 200 comments, drawn from the same period, and
excluding the incident's own threads.
"""
import datetime as dt
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\ins respinse\warningshot\src")

from warningshot.core import regression as rg      # noqa: E402
from warningshot.data import sources as src        # noqa: E402

TARGET_STORY = "48997548"
TARGET_HALF_LIFE = 7.05          # hours, from the paper
MIN_POINTS, MIN_COMMENTS = 500, 200


def ep(s):
    return int(dt.datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=dt.timezone.utc).timestamp())


def hourly_counts(story_id, n_hours=48):
    times = src.hn_comment_times(story_id)
    if not times or len(times) < MIN_COMMENTS:
        return None
    t0 = min(times)
    h = np.zeros(n_hours)
    for ts in times:
        k = (ts - t0) // 3600
        if 0 <= k < n_hours:
            h[k] += 1
    return h


# Build a comparison pool from broad queries across the same months, so the
# selection is not about content.
pool = {}
for q in ["OpenAI", "Google", "AI", "security", "Show HN", "Rust", "Apple",
          "model", "software", "Linux"]:
    try:
        rows = src.hn_stories(q, ep("2026-06-01"), ep("2026-09-01"))
    except Exception as exc:
        print(f"  query {q!r} failed: {exc}")
        continue
    for r in rows:
        if (r["points"] or 0) >= MIN_POINTS and (r["comments"] or 0) >= MIN_COMMENTS:
            pool[r["story_id"]] = r
    time.sleep(0.15)

pool.pop(TARGET_STORY, None)
print(f"comparison pool: {len(pool)} threads "
      f"(>= {MIN_POINTS} pts, >= {MIN_COMMENTS} comments, Jun-Aug 2026)")

rows = []
for sid, meta in sorted(pool.items(), key=lambda kv: -(kv[1]["points"] or 0))[:30]:
    try:
        h = hourly_counts(sid)
    except Exception:
        continue
    if h is None or h.sum() < MIN_COMMENTS:
        continue
    t = np.arange(1.0, h.size + 1.0)
    f = rg.fit_exponential(t, h)
    if not f:
        continue
    rows.append({"id": sid, "pts": meta["points"], "n": int(h.sum()),
                 "half_life": f["half_life"], "r2": f["r2"],
                 "title": (meta["title"] or "")[:52]})

if not rows:
    print("no comparable threads fitted -- cannot build the null")
    sys.exit(1)

hl = np.array([r["half_life"] for r in rows])
print(f"\nfitted {len(rows)} comparison threads")
print(f"{'points':>7}{'n':>7}{'t_half(h)':>11}{'R2':>7}  title")
for r in sorted(rows, key=lambda r: r["half_life"]):
    print(f"{r['pts']:>7}{r['n']:>7}{r['half_life']:>11.2f}{r['r2']:>7.3f}  {r['title']}")

pct = 100.0 * float((hl < TARGET_HALF_LIFE).mean())
print(f"\nNULL DISTRIBUTION of HN thread half-lives (hours)")
print(f"  min={hl.min():.2f}  p25={np.percentile(hl, 25):.2f}  "
      f"median={np.median(hl):.2f}  p75={np.percentile(hl, 75):.2f}  max={hl.max():.2f}")
print(f"  incident thread = {TARGET_HALF_LIFE:.2f} h")
print(f"  percentile against the null = {pct:.0f}")
print()
if 20 <= pct <= 80:
    print("  VERDICT: the incident thread decays like an ORDINARY front-page")
    print("  thread. W-2 is CONFIRMED -- the engagement half-life is a fact")
    print("  about HN thread lifecycles, not about this warning shot, and the")
    print("  'two clocks' framing cannot rest on it.")
else:
    print("  VERDICT: the incident thread is atypical against comparable")
    print("  threads; the engagement number carries event-specific signal.")
