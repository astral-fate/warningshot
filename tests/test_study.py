"""Tests for study-level logic, using a stubbed source so no network is hit.

These cover the parts a reviewer would attack: whether the baseline window
actually excludes the run-up to the event, whether the placebo null really
excludes the event, and whether the entity/concept ratio is computed over the
right rows.
"""

import datetime as dt

import pytest

from warningshot.core import metrics as m
from warningshot.data import events as ev
from warningshot.data import sources as src
from warningshot.eval import study


@pytest.fixture
def flat_then_spike(monkeypatch):
    """A page at 100/day that jumps to 1,100 on t0 and decays over a week."""
    spike = {0: 1000, 1: 600, 2: 300, 3: 150, 4: 80, 5: 40, 6: 20}

    def fake_pageviews(article, start, end, **kw):
        out = {}
        for d in m.date_range(start, end):
            v = 100
            off = m.days_between("20260721", d)
            if off in spike:
                v += spike[off]
            out[d] = v
        return out

    monkeypatch.setattr(src, "pageviews", fake_pageviews)
    return fake_pageviews


def test_baseline_window_ends_before_the_event(flat_then_spike):
    bs, be = study._baseline_window("20260721", 30)
    assert be < "20260721", "baseline must not include the event day"
    assert m.days_between(be, "20260721") == ev.BASELINE_GAP_DAYS
    assert len(m.date_range(bs, be)) == 30


def test_baseline_gap_excludes_pre_event_runup(flat_then_spike):
    """With a 7-day gap, coverage in the days just before t0 cannot inflate
    the baseline and thereby shrink the measured excess."""
    bs, be = study._baseline_window("20260721", 30)
    assert be == "20260714"
    assert bs == "20260615"


def test_page_profile_recovers_known_excess_and_peak(flat_then_spike):
    r = study.page_profile("Fake_Page", "20260721", 30)
    assert r["baseline"] == 100
    assert r["peak_date"] == "20260721"
    assert r["peak_views"] == 1100
    assert r["peak_ratio"] == 11.0
    # 48h horizon is 3 days: 1000 + 600 + 300
    assert r["excess_48h"] == 1900
    # 7d: + 150 + 80 + 40 + 20
    assert r["excess_7d"] == 2190
    assert r["bdays_48h"] == 19.0


def test_half_life_of_synthetic_spike(flat_then_spike):
    r = study.page_profile("Fake_Page", "20260721", 30)
    # peak excess 1000 -> half is 500; day+1 is 600 (above), day+2 is 300 (below)
    assert r["half_life_days"] == 2


def test_entity_concept_ratio_uses_summed_concepts():
    rows = [
        {"role": "victim", "excess_48h": 1000},
        {"role": "concept", "excess_48h": 30},
        {"role": "concept", "excess_48h": 20},
        {"role": "generic_control", "excess_48h": 900},
    ]
    # 1000 / (30 + 20) — the control must be excluded from the denominator
    assert study.entity_concept_ratio(rows, "48h") == pytest.approx(20.0)


def test_entity_concept_ratio_none_when_role_absent():
    """The June ban has no victim page; the ratio must decline to answer
    rather than silently substituting the developer."""
    rows = [
        {"role": "developer", "excess_48h": 1000},
        {"role": "concept", "excess_48h": 50},
    ]
    assert study.entity_concept_ratio(rows, "48h", entity_role="victim") is None


def test_placebo_windows_never_overlap_the_event_span():
    excl_from, excl_to = ev.PLACEBO_EXCLUSIONS[0]
    windows = m.placebo_windows(
        start=ev.PLACEBO_SPAN[0], end=ev.PLACEBO_SPAN[1], n_days=30,
        step=ev.PLACEBO_STEP_DAYS, exclude_from=excl_from, exclude_to=excl_to,
    )
    assert len(windows) >= 10, "too few placebo windows to form a null"
    for w in windows:
        assert w[-1] < excl_from or w[0] > excl_to


def test_placebo_null_is_computed_over_quiet_windows(monkeypatch):
    """A perfectly flat world must produce an all-zero null, so any non-zero
    observed value would rank at the top."""
    def flat(article, start, end, **kw):
        return {d: 200 for d in m.date_range(start, end)}

    monkeypatch.setattr(src, "pageviews", flat)
    null = study.placebo_null(pages=["A", "B"])
    assert len(null) >= 10
    assert all(row["concept_excess"] == 0 for row in null)
    assert m.percentile_rank(1, [r["concept_excess"] for r in null]) == 100.0
