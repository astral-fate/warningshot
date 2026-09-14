"""The four analyses behind the survey's findings.

  two_clock          entity vs concept attention across 48h / 7d / 30d
  sensitivity_sweep  every excess figure against four baseline windows
  placebo_null       OPEN ITEM 1 - null distribution for 30-day concept drift
  control_event      OPEN ITEM 2 - the same two-clock analysis on the June ban

The last two are what the survey listed as required before any causal reading
of the 30-day concept rise. Until they run, the two-clock result is suggestive
only, and this module is what settles it either way.
"""

import datetime as _dt

from warningshot.core import metrics as m
from warningshot.data import events as ev
from warningshot.data import sources as src


def _baseline_window(t0, length_days, gap_days=ev.BASELINE_GAP_DAYS):
    """The `length_days` window ending `gap_days` before t0.

    The gap keeps the pre-event run-up out of the baseline; without it, early
    coverage would inflate the baseline and shrink the measured excess.
    """
    end = _dt.datetime.strptime(t0, "%Y%m%d") - _dt.timedelta(days=gap_days)
    start = end - _dt.timedelta(days=length_days - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _fetch_span(t0, pad_before=140, pad_after=60):
    a = _dt.datetime.strptime(t0, "%Y%m%d") - _dt.timedelta(days=pad_before)
    b = _dt.datetime.strptime(t0, "%Y%m%d") + _dt.timedelta(days=pad_after)
    today = _dt.datetime.now()
    if b > today:
        b = today - _dt.timedelta(days=1)
    return a.strftime("%Y%m%d"), b.strftime("%Y%m%d")


def page_profile(article, t0, baseline_len=30, peak_window_days=14,
                 late_window_days=60, **kw):
    """Baseline, per-horizon excess, peak and half-life for one page.

    peak_window_days is deliberately short. A long window silently attributes
    a later event's peak to this one: at 45 days the June ban's window reaches
    into the 21 Jul incident and reports late-July concept-page peaks as if the
    June ban had caused them. 14 days keeps the peak attributable.
    """
    s, e = _fetch_span(t0)
    series = src.pageviews(article, s, e, **kw)
    if not series:
        return None
    bs, be = _baseline_window(t0, baseline_len)
    base = m.baseline(series, bs, be)
    row = {
        "page": article,
        "baseline": base,
        "baseline_window": f"{bs}..{be}",
    }
    for label, n in ev.HORIZONS.items():
        exc = m.excess(series, base, m.window_from(t0, n))
        row[f"excess_{label}"] = exc
        row[f"bdays_{label}"] = round(m.baseline_days(exc, base), 2) if base else None
    pk_date, pk_val = m.peak(series, t0, m.window_from(t0, peak_window_days)[-1])
    row["peak_date"] = pk_date
    row["peak_views"] = pk_val
    row["peak_ratio"] = round(pk_val / base, 2) if base else None
    row["half_life_days"] = m.half_life(series, base, pk_date)

    # A second, later maximum is a real feature of this incident: the 26-27 Aug
    # reports outdrew the July news peak. It is reported separately and labelled,
    # because at 60 days it is NOT safely attributable to t0 by position alone —
    # attribution rests on the external record, not on this number.
    late_end = m.window_from(t0, late_window_days)[-1]
    lp_date, lp_val = m.peak(series, t0, min(late_end, max(series)))
    row["late_peak_date"] = lp_date
    row["late_peak_views"] = lp_val
    row["late_peak_ratio"] = round(lp_val / base, 2) if base else None
    row["late_exceeds_early"] = lp_val > pk_val
    return row


def two_clock(event, baseline_len=30, **kw):
    """Entity vs concept attention at each horizon, with the generic control."""
    rows = []
    for role, page in event.entity_pages.items():
        r = page_profile(page, event.t0, baseline_len, **kw)
        if r:
            r["role"] = role
            rows.append(r)
    for page in ev.CONCEPT_PAGES:
        r = page_profile(page, event.t0, baseline_len, **kw)
        if r:
            r["role"] = "concept"
            rows.append(r)
    r = page_profile(ev.GENERIC_CONTROL, event.t0, baseline_len, **kw)
    if r:
        r["role"] = "generic_control"
        rows.append(r)
    return rows


def entity_concept_ratio(rows, horizon, entity_role="victim"):
    """Ratio of entity excess to summed concept excess at one horizon.

    Returns None when the event has no page in `entity_role` — the June ban
    has no victim, which is the whole point of using it as a control.
    """
    ent = [r for r in rows if r["role"] == entity_role]
    con = [r for r in rows if r["role"] == "concept"]
    if not ent or not con:
        return None
    csum = sum(r[f"excess_{horizon}"] for r in con)
    return (ent[0][f"excess_{horizon}"] / csum) if csum else None


def sensitivity_sweep(event, pages, **kw):
    """Excess at each horizon under all four baseline windows.

    A single-window number would not survive an adversarial read; this reports
    the spread so a reviewer can see which findings are window-dependent.
    """
    out = []
    for page in pages:
        for label, length in ev.BASELINE_LENGTHS:
            r = page_profile(page, event.t0, length, **kw)
            if r:
                r["baseline_len"] = label
                out.append(r)
    return out


def placebo_null(pages=None, n_days=30, **kw):
    """OPEN ITEM 1 — the null distribution for 30-day concept-page excess.

    Slides a 30-day window across a quiet span, computing summed concept
    excess in each. Windows overlapping any known event are excluded, or the
    null would contain the signal it is meant to be a null for.

    Without this, "concept pages rose over 30 days" has no yardstick: concept
    pages drift, and some of that drift is large.
    """
    pages = pages or ev.CONCEPT_PAGES
    span_start, span_end = ev.PLACEBO_SPAN
    excl_from, excl_to = ev.PLACEBO_EXCLUSIONS[0]

    windows = m.placebo_windows(
        start=span_start, end=span_end, n_days=n_days,
        step=ev.PLACEBO_STEP_DAYS, exclude_from=excl_from, exclude_to=excl_to,
    )

    fetch_start = (_dt.datetime.strptime(span_start, "%Y%m%d")
                   - _dt.timedelta(days=60)).strftime("%Y%m%d")
    series = {p: src.pageviews(p, fetch_start, span_end, **kw) for p in pages}

    null = []
    for w in windows:
        total = 0
        ok = True
        for p in pages:
            s = series[p]
            if not s:
                ok = False
                break
            bs, be = _baseline_window(w[0], 30)
            try:
                base = m.baseline(s, bs, be)
            except ValueError:
                ok = False
                break
            total += m.excess(s, base, w)
        if ok:
            null.append({"window_start": w[0], "window_end": w[-1], "concept_excess": total})
    return null


def control_event(baseline_len=30, **kw):
    """OPEN ITEM 2 — the two-clock analysis applied to the June ban.

    Tests whether the two-clock structure is a property of warning shots or
    just of this one event. If June shows the same slow concept conversion,
    the July result is not warning-shot-specific.
    """
    return two_clock(ev.JUNE_BAN, baseline_len, **kw)


def placebo_null_season_matched(pages=None, n_days=30, step=5, **kw):
    """OPEN ITEM / review response M3 — a season-matched null.

    The original null (`placebo_null`) draws windows from 15 Jan - 31 May 2026.
    Measured against the pre-event June-July baseline, the concept pages run
    6-47% busier in that period (`Artificial_general_intelligence`: median
    1,761 vs 1,200). Busier pages produce larger absolute excess, so the null
    is inflated and the observed value ranks artificially LOW - which is the
    direction that supports this project's own null result. That is a
    self-serving bias and had to be removed rather than merely disclosed.

    This version draws windows from the SAME calendar season in prior years
    (`events.SEASON_MATCHED_SPANS`), so seasonality is matched by construction
    and no 2026 event can leak in.
    """
    pages = pages or ev.CONCEPT_PAGES
    null = []
    for span_start, span_end in ev.SEASON_MATCHED_SPANS:
        fetch_start = (_dt.datetime.strptime(span_start, "%Y%m%d")
                       - _dt.timedelta(days=60)).strftime("%Y%m%d")
        series = {p: src.pageviews(p, fetch_start, span_end, **kw) for p in pages}
        windows = m.placebo_windows(
            start=span_start, end=span_end, n_days=n_days, step=step,
            exclude_from="99999999", exclude_to="99999999",
        )
        for w in windows:
            total, ok = 0, True
            for p in pages:
                s = series[p]
                if not s:
                    ok = False
                    break
                bs, be = _baseline_window(w[0], 30)
                try:
                    base = m.baseline(s, bs, be)
                except ValueError:
                    ok = False
                    break
                total += m.excess(s, base, w)
            if ok:
                null.append({"window_start": w[0], "window_end": w[-1],
                             "concept_excess": total, "season": span_start[:4]})
    return null
