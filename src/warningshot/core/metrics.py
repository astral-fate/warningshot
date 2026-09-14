"""Attention metrics over daily pageview series.

Pure functions only — no network, no I/O. A "series" is a dict mapping
YYYYMMDD strings to daily view counts, which is what sources.py returns.

Definitions used throughout, fixed here so the paper and the code cannot
drift apart:

  baseline      median daily views over a stated quiet window
  excess        sum over a window of max(0, views - baseline)
  baseline_days excess expressed as a multiple of one ordinary day, which
                is the only form comparable across pages of different size
  half_life     days after the peak until excess falls below half its peak

The negative-day floor in `excess` is deliberate: without it a quiet day
adjacent to a spike silently cancels part of the spike.
"""

import datetime as _dt
import statistics

_FMT = "%Y%m%d"


# --- dates -----------------------------------------------------------------

def date_range(start, end):
    """Inclusive list of YYYYMMDD strings from start to end."""
    a = _dt.datetime.strptime(start, _FMT)
    b = _dt.datetime.strptime(end, _FMT)
    if b < a:
        raise ValueError(f"end {end} precedes start {start}")
    out = []
    while a <= b:
        out.append(a.strftime(_FMT))
        a += _dt.timedelta(days=1)
    return out


def window_from(start, n_days):
    """n_days-long window beginning at start (inclusive)."""
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    a = _dt.datetime.strptime(start, _FMT)
    return [(a + _dt.timedelta(days=i)).strftime(_FMT) for i in range(n_days)]


def days_between(a, b):
    return (_dt.datetime.strptime(b, _FMT) - _dt.datetime.strptime(a, _FMT)).days


# --- core metrics ----------------------------------------------------------

def baseline(series, start, end):
    """Median daily views over [start, end]. Days absent from the series are
    ignored rather than counted as zero — a gap in the API is not a quiet day."""
    vals = [v for k, v in series.items() if start <= k <= end]
    if not vals:
        raise ValueError(f"no observations in baseline window {start}..{end}")
    return statistics.median(vals)


def excess(series, base, days):
    """Total views above baseline across `days`. Missing days count as zero
    views (a real absence of traffic), which floors to zero contribution."""
    return sum(max(0, series.get(d, 0) - base) for d in days)


def baseline_days(exc, base):
    """Excess expressed in units of one ordinary day's traffic."""
    if base <= 0:
        raise ValueError(f"baseline must be positive, got {base}")
    return exc / base


def peak(series, start, end):
    """(date, value) of the maximum in [start, end]; ties go to the earliest."""
    window = {k: v for k, v in series.items() if start <= k <= end}
    if not window:
        raise ValueError(f"no observations in peak window {start}..{end}")
    best = max(window.values())
    return min(k for k, v in window.items() if v == best), best


def half_life(series, base, peak_date, max_days=60):
    """Days after `peak_date` until excess first falls to <= half its peak.

    Returns None if it has not halved within max_days — reported as such
    rather than silently truncated, since "did not decay" is a real outcome.
    """
    peak_excess = series.get(peak_date, 0) - base
    if peak_excess <= 0:
        return None
    half = peak_excess / 2
    for i in range(1, max_days + 1):
        d = (_dt.datetime.strptime(peak_date, _FMT) + _dt.timedelta(days=i)).strftime(_FMT)
        if d not in series:
            continue
        if series[d] - base <= half:
            return i
    return None


# --- placebo / null distribution -------------------------------------------

def placebo_windows(start, end, n_days, step, exclude_from, exclude_to):
    """Sliding windows of n_days across [start, end], skipping any window that
    overlaps [exclude_from, exclude_to].

    The exclusion is what makes the result a null: a placebo window that
    touched the event would contain the signal it is meant to be a null for.
    """
    out = []
    for s in date_range(start, end)[::step]:
        w = window_from(s, n_days)
        if w[-1] > end:
            break
        overlaps = w[-1] >= exclude_from and w[0] <= exclude_to
        if not overlaps:
            out.append(w)
    return out


def percentile_rank(observed, null):
    """Percentage of null values strictly below `observed`."""
    if not null:
        raise ValueError("null distribution is empty")
    return 100.0 * sum(1 for v in null if v < observed) / len(null)
