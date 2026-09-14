"""Nulls and controls demanded by the third review.

Each function here answers one reviewer objection that the manuscript had
asserted rather than tested. They are in the package, not in a scratch script,
because their outputs are now headline numbers.

  hn_thread_null            W-2. Is the engagement half-life special, or is it
                            what every front-page thread does?
  daily_ratio_distribution  W-3. Is 1.22x on the disclosure day unusual for
                            this page, or an ordinary busy day?
  inversion_units           W-4. The counting inversion in the baseline-days
                            units the methods section actually mandates.
  resolution_control        W-5. How much of the cross-channel spread is
                            hourly-vs-daily binning rather than attention?
  halflife_ratio_ci         W-7. A bootstrap interval on the RATIO of two
                            half-lives, since overlapping separate intervals
                            do not constitute a test.
"""

import datetime as _dt

import numpy as np

from warningshot.core import metrics as m
from warningshot.core import regression as rg
from warningshot.data import events as ev
from warningshot.data import modalities as mo
from warningshot.data import sources as src

_FMT = "%Y%m%d"

# Comparability rule for the W-2 null, fixed before any half-life was looked at.
NULL_MIN_POINTS = 500
NULL_MIN_COMMENTS = 200
NULL_QUERIES = ("OpenAI", "Google", "AI", "security", "Show HN", "Rust",
                "Apple", "model", "software", "Linux")
NULL_SPAN = ("2026-06-01", "2026-09-01")
NULL_MAX_THREADS = 30
INCIDENT_STORY = "48997548"


def _epoch(s):
    return int(_dt.datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=_dt.timezone.utc).timestamp())


def _hourly(story_id, n_hours=48, min_comments=NULL_MIN_COMMENTS):
    times = src.hn_comment_times(story_id)
    if not times or len(times) < min_comments:
        return None
    t0 = min(times)
    h = np.zeros(n_hours)
    for ts in times:
        k = (ts - t0) // 3600
        if 0 <= k < n_hours:
            h[k] += 1
    return h


def hn_thread_null(target=INCIDENT_STORY, max_threads=NULL_MAX_THREADS):
    """W-2 — the null the manuscript's headline lacked.

    Samples front-page-scale threads from the same period across ten broad
    topic queries, so selection is by prominence rather than content, fits the
    identical exponential to each, and reports where the incident thread falls.
    """
    pool = {}
    for q in NULL_QUERIES:
        try:
            rows = src.hn_stories(q, _epoch(NULL_SPAN[0]), _epoch(NULL_SPAN[1]))
        except Exception:
            continue
        for r in rows:
            if ((r["points"] or 0) >= NULL_MIN_POINTS
                    and (r["comments"] or 0) >= NULL_MIN_COMMENTS):
                pool[r["story_id"]] = r
    pool.pop(target, None)

    fitted = []
    for sid, meta in sorted(pool.items(),
                            key=lambda kv: -(kv[1]["points"] or 0))[:max_threads]:
        try:
            h = _hourly(sid)
        except Exception:
            continue
        if h is None:
            continue
        f = rg.fit_exponential(np.arange(1.0, h.size + 1.0), h)
        if f:
            fitted.append({"story_id": sid, "points": meta["points"],
                           "n_comments": int(h.sum()),
                           "half_life_h": f["half_life"], "r2": f["r2"],
                           "title": meta["title"]})
    if not fitted:
        return None

    th = _hourly(target)
    tf = rg.fit_exponential(np.arange(1.0, th.size + 1.0), th) if th is not None else None
    obs = tf["half_life"] if tf else None
    hl = np.array([r["half_life_h"] for r in fitted])
    return {
        "pool_size": len(pool),
        "n_fitted": len(fitted),
        "observed_half_life_h": obs,
        "percentile": (100.0 * float((hl < obs).mean())) if obs else None,
        "null_min": float(hl.min()),
        "null_p25": float(np.percentile(hl, 25)),
        "null_median": float(np.median(hl)),
        "null_p75": float(np.percentile(hl, 75)),
        "null_max": float(hl.max()),
        "threads": sorted(fitted, key=lambda r: r["half_life_h"]),
    }


def daily_ratio_distribution(page, observed_date, observed_ratio,
                             start="20260101", trailing=30):
    """W-3 — the distribution a single ratio has to be read against."""
    end = observed_date
    s = src.pageviews(page, start, end)
    if not s:
        return None
    days = sorted(s)
    vals = np.array([s[d] for d in days], dtype=float)
    ratios, rdays = [], []
    for i in range(trailing, len(days)):
        base = float(np.median(vals[i - trailing:i]))
        if base > 0:
            ratios.append(vals[i] / base)
            rdays.append(days[i])
    ratios = np.array(ratios)
    if ratios.size < 30:
        return None
    return {
        "page": page,
        "window": [rdays[0], rdays[-1]],
        "n_days": int(ratios.size),
        "observed_ratio": observed_ratio,
        "percentile": 100.0 * float((ratios < observed_ratio).mean()),
        "days_at_or_above": int((ratios >= observed_ratio).sum()),
        "min": float(ratios.min()),
        "median": float(np.median(ratios)),
        "p75": float(np.percentile(ratios, 75)),
        "p90": float(np.percentile(ratios, 90)),
        "max": float(ratios.max()),
    }


def _excess_and_base(page, t0, window_days, baseline_len=30,
                     gap=ev.BASELINE_GAP_DAYS):
    end = _dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=gap)
    start = end - _dt.timedelta(days=baseline_len - 1)
    a = (start - _dt.timedelta(days=20)).strftime(_FMT)
    b = (_dt.datetime.strptime(t0, _FMT)
         + _dt.timedelta(days=window_days + 5)).strftime(_FMT)
    s = src.pageviews(page, a, b)
    if not s:
        return None
    base = m.baseline(s, start.strftime(_FMT), end.strftime(_FMT))
    exc = m.excess(s, base, m.window_from(t0, window_days))
    return {"page": page, "baseline": base, "excess": exc,
            "baseline_days": exc / base if base else None}


def inversion_units(window_days=3):
    """W-4 — the counting inversion in raw views AND in baseline-days.

    The methods section states baseline-days is the only form comparable
    across pages of very different size, then the inversion was reported in
    raw views. Both are computed here so the units are visible.
    """
    v = _excess_and_base("Hugging_Face", ev.JULY_INCIDENT.t0, window_days)
    o = _excess_and_base("OpenAI", ev.JULY_INCIDENT.t0, window_days)
    a = _excess_and_base("Anthropic", ev.JUNE_BAN.t0, window_days)
    if not (v and o and a):
        return None
    return {
        "victim": v, "developer_july": o, "developer_june": a,
        "victim_vs_june_dev_raw": v["excess"] / a["excess"],
        "victim_vs_june_dev_baseline_days": v["baseline_days"] / a["baseline_days"],
        "june_dev_vs_july_dev_raw": a["excess"] / o["excess"],
        "june_dev_vs_july_dev_baseline_days": a["baseline_days"] / o["baseline_days"],
    }


def resolution_control(story_id=INCIDENT_STORY, lookups_half_life_days=5.450):
    """W-5 — refit the same comments at daily resolution.

    A daily series cannot resolve a sub-day half-life, so part of any
    cross-channel spread is imposed by binning.
    """
    ch = mo.engagement(story_id)
    if not ch:
        return None
    h = np.array(ch["hourly"], dtype=float)
    f_h = rg.fit_exponential(np.arange(1.0, h.size + 1.0), h)

    times = np.asarray(sorted(src.hn_comment_times(story_id)), dtype=float)
    rel_h = (times - times[0]) / 3600.0
    span_d = int(np.ceil(rel_h.max() / 24.0))
    daily = np.array([int(((rel_h >= 24 * k) & (rel_h < 24 * (k + 1))).sum())
                      for k in range(span_d)], dtype=float)
    f_d = rg.fit_exponential(np.arange(1.0, daily.size + 1.0), daily)
    if not (f_h and f_d):
        return None
    return {
        "hourly_half_life_h": f_h["half_life"],
        "hourly_half_life_d": f_h["half_life"] / 24.0,
        "hourly_r2": f_h["r2"], "hourly_bins": int(h.size),
        "daily_half_life_d": f_d["half_life"], "daily_r2": f_d["r2"],
        "daily_bins": int(daily.size),
        "daily_counts": [int(x) for x in daily],
        "ratio_hourly": lookups_half_life_days / (f_h["half_life"] / 24.0),
        "ratio_daily": lookups_half_life_days / f_d["half_life"],
    }


def halflife_ratio_ci(n_boot=2000, block=4, seed=4, horizon=25):
    """W-7 — bootstrap the ratio of two half-lives directly.

    Two separately estimated intervals overlapping is not a test that the
    parameters are indistinguishable; an interval on their ratio is.
    """
    def series(kind):
        ch = (mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=horizon)
              if kind == "lookups"
              else mo.media('"Hugging Face" OpenAI', ev.JULY_INCIDENT.t0,
                            horizon=horizon))
        if not ch:
            return None
        t, y = ch["t"], ch["y"]
        k = y > 0
        return t[k], y[k]

    a, b = series("lookups"), series("media")
    if not (a and b):
        return None
    (tl, yl), (tm, ym) = a, b
    fl, fm = rg.fit_exponential(tl, yl), rg.fit_exponential(tm, ym)
    if not (fl and fm):
        return None

    rng = np.random.default_rng(seed)

    def resample(t, y, fit):
        pred = np.log(fit["amplitude"]) - t / fit["tau"]
        resid = np.log(y) - pred
        N = resid.size
        nb = int(np.ceil(N / block))
        starts = rng.integers(0, N - block + 1, size=nb)
        r = np.concatenate([resid[s:s + block] for s in starts])[:N]
        return np.exp(pred + r)

    vals = []
    for _ in range(n_boot):
        x = rg.fit_exponential(tl, resample(tl, yl, fl))
        z = rg.fit_exponential(tm, resample(tm, ym, fm))
        if x and z and x["half_life"] > 0:
            vals.append(z["half_life"] / x["half_life"])
    vals = np.asarray(vals)
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return {
        "lookups_half_life_d": fl["half_life"],
        "media_half_life_d": fm["half_life"],
        "point_ratio": fm["half_life"] / fl["half_life"],
        "ci_lo": lo, "ci_hi": hi, "n_resamples": int(vals.size),
        "contains_one": bool(lo <= 1.0 <= hi),
        "seed": seed, "block": block,
    }


# ===================================================================
# Fourth-review additions.
#   concept_null_mde      W-1. A null result needs a power statement.
#   june_candidate_pages  W-2. The absorbing-page rule, applied to June too.
#   inversion_ci          W-3. An interval on the headline ratio.
#   control_vs_null       W-7. The generic control held to the same null.
# ===================================================================

def concept_null_mde(null_values, observed):
    """W-1 — minimum detectable effect for the concept null.

    A percentile alone does not say whether the design *could* have found an
    effect. This reports the multiple of the null median that would have been
    needed to clear the 90th and 95th percentiles.
    """
    v = np.asarray(sorted(null_values), dtype=float)
    med = float(np.median(v))
    p90, p95 = float(np.percentile(v, 90)), float(np.percentile(v, 95))
    return {
        "null_median": med, "null_p90": p90, "null_p95": p95,
        "observed": float(observed),
        "observed_over_median": float(observed) / med if med else None,
        "mde_at_p90": p90 / med if med else None,
        "mde_at_p95": p95 / med if med else None,
        "n_windows": int(v.size),
    }


JUNE_CANDIDATES = ("Anthropic", "Claude_(language_model)",
                   "United_States_Department_of_Commerce", "Export_control",
                   "Artificial_intelligence", "OpenAI")


def june_candidate_pages(window_days=3):
    """W-2 — run the absorbing-page rule on the June event as well.

    The headline comparison pairs the July *victim* against the June
    *developer*. The paper's own rule says count the page that absorbed the
    attention; that rule was demonstrated for July and assumed for June.
    """
    out = []
    for page in JUNE_CANDIDATES:
        try:
            r = _excess_and_base(page, ev.JUNE_BAN.t0, window_days)
        except Exception:
            r = None
        if r:
            out.append(r)
    if not out:
        return None
    out.sort(key=lambda r: -(r["baseline_days"] or 0))
    return {"candidates": out,
            "absorbing_page": out[0]["page"],
            "absorbing_baseline_days": out[0]["baseline_days"]}


def inversion_ci(window_days=3, n_boot=2000, seed=11, baseline_len=30):
    """W-3 — bootstrap interval on the baseline-days inversion ratio.

    The dominant uncertainty in a baseline-days figure is the baseline itself,
    a median over 30 daily values. Resampling those days with replacement and
    recomputing both the baseline and the excess propagates that uncertainty
    into the ratio. The observed spike days are treated as fixed.
    """
    rng = np.random.default_rng(seed)

    def parts(page, t0):
        end = _dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=ev.BASELINE_GAP_DAYS)
        start = end - _dt.timedelta(days=baseline_len - 1)
        s = src.pageviews(page, (start - _dt.timedelta(days=20)).strftime(_FMT),
                          (_dt.datetime.strptime(t0, _FMT)
                           + _dt.timedelta(days=window_days + 5)).strftime(_FMT))
        if not s:
            return None
        bdays = m.date_range(start.strftime(_FMT), end.strftime(_FMT))
        bvals = np.array([s[d] for d in bdays if d in s], dtype=float)
        wvals = np.array([s.get(d, 0) for d in m.window_from(t0, window_days)],
                         dtype=float)
        return bvals, wvals

    a = parts("Hugging_Face", ev.JULY_INCIDENT.t0)
    b = parts("Anthropic", ev.JUNE_BAN.t0)
    if not (a and b):
        return None

    def bdays_of(bvals, wvals):
        base = float(np.median(bvals))
        if base <= 0:
            return None
        return float(np.sum(np.maximum(0.0, wvals - base))) / base

    point = bdays_of(*a) / bdays_of(*b)
    vals = []
    for _ in range(n_boot):
        ra = bdays_of(rng.choice(a[0], a[0].size, replace=True), a[1])
        rb = bdays_of(rng.choice(b[0], b[0].size, replace=True), b[1])
        if ra and rb and rb > 0:
            vals.append(ra / rb)
    vals = np.asarray(vals)
    return {
        "point_ratio": point,
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
        "n_resamples": int(vals.size),
        "excludes_one": bool(np.percentile(vals, 2.5) > 1.0),
        "seed": seed,
    }


def control_vs_null(page=None, n_days=30, step=5):
    """W-7 — put the generic control through the same season-matched null.

    The concept pages were dismissed against a null; the control was reported
    as a bare ratio. Holding both to one standard is the point.
    """
    page = page or ev.GENERIC_CONTROL
    null = []
    for span_start, span_end in ev.SEASON_MATCHED_SPANS:
        fetch = (_dt.datetime.strptime(span_start, _FMT)
                 - _dt.timedelta(days=60)).strftime(_FMT)
        s = src.pageviews(page, fetch, span_end)
        if not s:
            continue
        for w in m.placebo_windows(start=span_start, end=span_end, n_days=n_days,
                                   step=step, exclude_from="99999999",
                                   exclude_to="99999999"):
            bs, be = _excess_base_window(w[0])
            try:
                base = m.baseline(s, bs, be)
            except ValueError:
                continue
            null.append(m.excess(s, base, w))
    if len(null) < 5:
        return None
    obs = _excess_and_base(page, ev.JULY_INCIDENT.t0, n_days)
    if not obs:
        return None
    v = np.asarray(null, dtype=float)
    return {"page": page, "n_windows": int(v.size),
            "null_median": float(np.median(v)), "null_max": float(v.max()),
            "observed_excess": obs["excess"],
            "percentile": 100.0 * float((v < obs["excess"]).mean())}


def _excess_base_window(t0, length_days=30, gap=ev.BASELINE_GAP_DAYS):
    end = _dt.datetime.strptime(t0, _FMT) - _dt.timedelta(days=gap)
    start = end - _dt.timedelta(days=length_days - 1)
    return start.strftime(_FMT), end.strftime(_FMT)
