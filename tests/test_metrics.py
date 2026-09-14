"""Tests for the metric core.

Every number that reaches the paper is produced by metrics.py. A hand-computed
ratio in the first survey pass was wrong (reported 40:1, actual 20:1), which is
the reason these are pure functions with tests rather than inline arithmetic.

No network. All fixtures are synthetic and hand-checkable.
"""

import pytest

from warningshot.core import metrics as m


# --- date_range ------------------------------------------------------------

def test_date_range_inclusive_both_ends():
    assert m.date_range("20260721", "20260723") == ["20260721", "20260722", "20260723"]


def test_date_range_single_day():
    assert m.date_range("20260721", "20260721") == ["20260721"]


def test_date_range_crosses_month_boundary():
    assert m.date_range("20260730", "20260802") == [
        "20260730", "20260731", "20260801", "20260802",
    ]


def test_date_range_rejects_reversed():
    with pytest.raises(ValueError):
        m.date_range("20260723", "20260721")


def test_window_from_returns_n_days_starting_at_start():
    assert m.window_from("20260721", 3) == ["20260721", "20260722", "20260723"]


# --- baseline --------------------------------------------------------------

def test_baseline_median_of_window_only():
    # Values outside the window must not influence the baseline.
    series = {"20260601": 10, "20260602": 20, "20260603": 30, "20260701": 9999}
    assert m.baseline(series, "20260601", "20260603") == 20


def test_baseline_median_even_count_averages_middle_two():
    series = {"20260601": 10, "20260602": 20, "20260603": 30, "20260604": 40}
    assert m.baseline(series, "20260601", "20260604") == 25


def test_baseline_raises_on_empty_window():
    with pytest.raises(ValueError):
        m.baseline({"20260601": 10}, "20260701", "20260703")


# --- excess ----------------------------------------------------------------

def test_excess_sums_positive_deviations():
    series = {"20260721": 150, "20260722": 200, "20260723": 120}
    # (150-100) + (200-100) + (120-100)
    assert m.excess(series, 100, m.date_range("20260721", "20260723")) == 170


def test_excess_floors_negative_days_at_zero():
    """A below-baseline day contributes 0, never a negative offset.

    This matters: without the floor, a quiet day before a spike would
    silently cancel part of the spike.
    """
    series = {"20260721": 50, "20260722": 200}
    assert m.excess(series, 100, m.date_range("20260721", "20260722")) == 100


def test_excess_treats_missing_days_as_zero_views():
    series = {"20260721": 150}
    # 20260722 absent -> 0 views -> max(0, 0-100) = 0
    assert m.excess(series, 100, m.date_range("20260721", "20260722")) == 50


def test_excess_of_flat_series_at_baseline_is_zero():
    series = {d: 100 for d in m.date_range("20260721", "20260730")}
    assert m.excess(series, 100, m.date_range("20260721", "20260730")) == 0


# --- baseline_days ---------------------------------------------------------

def test_baseline_days_expresses_excess_in_ordinary_days():
    # 1368 excess views on a page that normally gets 188/day
    assert m.baseline_days(1368, 188) == pytest.approx(7.277, abs=1e-3)


def test_baseline_days_raises_on_zero_baseline():
    with pytest.raises(ValueError):
        m.baseline_days(100, 0)


# --- peak ------------------------------------------------------------------

def test_peak_returns_date_and_value_of_maximum():
    series = {"20260721": 10, "20260722": 900, "20260723": 40}
    assert m.peak(series, "20260721", "20260723") == ("20260722", 900)


def test_peak_restricted_to_window():
    series = {"20260721": 10, "20260722": 900, "20260723": 40}
    assert m.peak(series, "20260721", "20260721") == ("20260721", 10)


def test_peak_ties_resolve_to_earliest_date():
    series = {"20260721": 500, "20260722": 500}
    assert m.peak(series, "20260721", "20260722") == ("20260721", 500)


# --- half_life -------------------------------------------------------------

def test_half_life_counts_days_until_excess_halves():
    # baseline 100, peak 900 -> excess 800 -> half is 500 views
    series = {
        "20260722": 900,   # peak
        "20260723": 700,   # excess 600, still above half
        "20260724": 400,   # excess 300, below half -> 2 days after peak
    }
    assert m.half_life(series, 100, "20260722") == 2


def test_half_life_of_one_day_drop():
    series = {"20260722": 900, "20260723": 200}
    assert m.half_life(series, 100, "20260722") == 1


def test_half_life_none_when_never_decays_in_range():
    series = {"20260722": 900, "20260723": 880, "20260724": 870}
    assert m.half_life(series, 100, "20260722") is None


# --- placebo ---------------------------------------------------------------

def test_placebo_percentile_of_observed_against_null():
    null = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # 95 exceeds 9 of 10 (100 is not below it) -> 90th percentile
    assert m.percentile_rank(95, null) == pytest.approx(90.0)
    # 105 exceeds all 10
    assert m.percentile_rank(105, null) == pytest.approx(100.0)
    # 55 exceeds 5 of 10
    assert m.percentile_rank(55, null) == pytest.approx(50.0)
    # 5 exceeds none
    assert m.percentile_rank(5, null) == pytest.approx(0.0)


def test_percentile_rank_raises_on_empty_null():
    with pytest.raises(ValueError):
        m.percentile_rank(5, [])


def test_placebo_windows_excludes_windows_overlapping_the_event():
    """Placebo windows must not touch the real event, or the null is
    contaminated by the very signal it is meant to be a null for."""
    windows = m.placebo_windows(
        start="20260201", end="20260630", n_days=30, step=15,
        exclude_from="20260615", exclude_to="20260715",
    )
    for w in windows:
        assert not (w[-1] >= "20260615" and w[0] <= "20260715"), (
            f"window {w[0]}..{w[-1]} overlaps the excluded event span"
        )
    assert len(windows) > 0


def test_placebo_windows_all_have_requested_length():
    windows = m.placebo_windows(
        start="20260201", end="20260531", n_days=30, step=10,
        exclude_from="99999999", exclude_to="99999999",
    )
    assert all(len(w) == 30 for w in windows)
