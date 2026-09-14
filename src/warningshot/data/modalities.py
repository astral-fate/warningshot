"""Three independent attention channels, reduced to a common (t, y) form.

The point of using three is that each measures a different act:

  lookups     Wikipedia pageviews      someone went to find out what happened
  engagement  Hacker News comments     someone argued about it
  media       GDELT coverage volume    an outlet published about it

They have different owners, different populations and different failure modes,
so agreement between them is real corroboration rather than one signal counted
three times. Each is returned post-peak as days-since-peak against excess above
baseline, which is the form regression.py consumes.

Resolution differs and matters: HN comment timestamps are exact to the second,
Wikipedia and GDELT are daily. The HN channel is therefore the only one that
can resolve decay inside the first day.
"""

import datetime as _dt

import numpy as np

from warningshot.core import metrics as m
from warningshot.data import events as ev
from warningshot.data import sources as src

_FMT = "%Y%m%d"


def _post_peak(series, base, peak_date, horizon):
    """(t, y) arrays of days-since-peak and excess above baseline."""
    t, y = [], []
    for i in range(0, horizon + 1):
        d = (_dt.datetime.strptime(peak_date, _FMT)
             + _dt.timedelta(days=i)).strftime(_FMT)
        if d in series:
            t.append(float(i) + 1.0)  # 1-based: power-law needs t > 0
            y.append(float(series[d] - base))
    return np.array(t), np.array(y)


def lookups(page, t0, baseline_len=30, horizon=45, search_days=14,
            exclude_contaminated=False, **kw):
    """Wikipedia pageviews channel.

    `exclude_contaminated=True` drops the dates declared in
    events.CONTAMINATED_DATES for this page (review response C2). The dates
    and their reasons are declared in events.py, never chosen here.
    """
    a = (_dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=baseline_len + 40)).strftime(_FMT)
    b = (_dt.datetime.strptime(t0, _FMT) + _dt.timedelta(days=horizon + 5)).strftime(_FMT)
    today = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime(_FMT)
    series = src.pageviews(page, a, min(b, today), **kw)
    if not series:
        return None
    bs, be = _baseline_window(t0, baseline_len)
    base = m.baseline(series, bs, be)
    peak_date, peak_val = m.peak(series, t0, m.window_from(t0, search_days)[-1])
    t, y = _post_peak(series, base, peak_date, horizon)

    dropped = []
    if exclude_contaminated:
        bad = ev.contaminated_for(page)
        if bad:
            keep = []
            for i in range(t.size):
                d = (_dt.datetime.strptime(peak_date, _FMT)
                     + _dt.timedelta(days=int(t[i]) - 1)).strftime(_FMT)
                if d in bad:
                    dropped.append(d)
                else:
                    keep.append(i)
            idx = np.array(keep, dtype=int)
            t, y = t[idx], y[idx]

    return {"channel": "lookups", "unit": "excess daily pageviews", "source": page,
            "baseline": base, "peak_date": peak_date, "peak_value": peak_val,
            "t": t, "y": y, "excluded_dates": dropped}


def engagement(story_id, horizon_days=14, **kw):
    """Hacker News comment-arrival channel.

    Baseline is zero by construction: a story's comment thread has no ambient
    rate to subtract. Peak is the first bucket, since HN threads front-load.
    """
    times = src.hn_comment_times(story_id, **kw)
    if not times or len(times) < 30:
        return None
    t0 = times[0]
    per_day = {}
    for ts in times:
        per_day[(ts - t0) // 86400] = per_day.get((ts - t0) // 86400, 0) + 1
    per_hour = {}
    for ts in times:
        per_hour[(ts - t0) // 3600] = per_hour.get((ts - t0) // 3600, 0) + 1

    days = sorted(d for d in per_day if d <= horizon_days)
    t = np.array([float(d) + 1.0 for d in days])
    y = np.array([float(per_day[d]) for d in days])
    return {"channel": "engagement", "unit": "comments per day", "source": f"HN {story_id}",
            "baseline": 0.0, "peak_date": "day0", "peak_value": max(y) if y.size else 0,
            "t": t, "y": y, "n_comments": len(times),
            "hourly": [per_hour.get(h, 0) for h in range(0, 48)]}


def media(query, t0, horizon=45, baseline_days=6, search_days=14, **kw):
    """GDELT coverage-volume channel.

    Baseline is the median of the days before t0 inside the fetch window. It is
    short because GDELT only serves a rolling ~3-month window, which is a real
    constraint on this channel and not a design choice.
    """
    a = (_dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=baseline_days)).strftime(_FMT)
    b = (_dt.datetime.strptime(t0, _FMT) + _dt.timedelta(days=horizon + 5)).strftime(_FMT)
    today = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime(_FMT)
    series = src.gdelt_timeline(query, a, min(b, today), **kw)
    if not series:
        return None
    pre = [v for k, v in series.items() if k < t0]
    base = float(np.median(pre)) if pre else 0.0
    peak_date, peak_val = m.peak(series, t0, m.window_from(t0, search_days)[-1])
    t, y = _post_peak(series, base, peak_date, horizon)
    return {"channel": "media", "unit": "excess coverage volume (% of monitored)",
            "source": query, "baseline": base, "peak_date": peak_date,
            "peak_value": peak_val, "t": t, "y": y}


def _baseline_window(t0, length_days, gap_days=ev.BASELINE_GAP_DAYS):
    end = _dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=gap_days)
    start = end - _dt.timedelta(days=length_days - 1)
    return start.strftime(_FMT), end.strftime(_FMT)


def all_channels(event, hn_query=None, gdelt_query=None, **kw):
    """Every available channel for one event. Missing channels are omitted
    rather than faked, and the caller is told which are absent."""
    out = []
    page = event.entity_pages.get("victim") or event.entity_pages.get("developer")
    ch = lookups(page, event.t0, **kw)
    if ch:
        out.append(ch)

    if hn_query:
        stories = src.hn_stories(hn_query, 0, 4102444800)
        top = next((s for s in stories if s["points"] > 500), None)
        if top:
            ch = engagement(top["story_id"])
            if ch:
                ch["story_title"] = top["title"]
                ch["story_points"] = top["points"]
                out.append(ch)

    if gdelt_query:
        try:
            ch = media(gdelt_query, event.t0)
            if ch:
                out.append(ch)
        except Exception as exc:  # noqa: BLE001 - a missing channel is reportable
            print(f"    NOTE: media channel unavailable ({exc})")
    return out
