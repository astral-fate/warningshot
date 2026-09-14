"""Tests for the decay-regression core.

The proposal these implement (research-agent/AI Incident Communication Strategy
Research.txt) called for "exponential regression models ... to calculate the
exact half-life". Published work on the same data type rejects that model:
Igarashi et al. (arXiv 2209.07033) fit a two-phase exponential-then-power-law
curve with a switching point ~10-11 days after peak. Both are implemented so
the comparison is empirical rather than asserted.

Synthetic fixtures with known parameters, so a failure means the estimator is
wrong rather than the data being awkward. No network.
"""

import math

import numpy as np
import pytest

from warningshot.core import regression as rg


# --- exponential -----------------------------------------------------------

def test_exponential_recovers_known_timescale():
    t = np.arange(1, 21, dtype=float)
    tau_true = 4.0
    y = 1000.0 * np.exp(-t / tau_true)
    fit = rg.fit_exponential(t, y)
    assert fit["tau"] == pytest.approx(tau_true, rel=1e-6)
    assert fit["r2"] > 0.999


def test_exponential_half_life_is_tau_times_ln2():
    t = np.arange(1, 21, dtype=float)
    y = 500.0 * np.exp(-t / 6.0)
    fit = rg.fit_exponential(t, y)
    assert fit["half_life"] == pytest.approx(6.0 * math.log(2), rel=1e-6)


def test_exponential_survives_moderate_noise():
    rng = np.random.default_rng(0)
    t = np.arange(1, 31, dtype=float)
    y = 2000.0 * np.exp(-t / 5.0) * rng.lognormal(0, 0.15, size=t.size)
    fit = rg.fit_exponential(t, y)
    assert fit["tau"] == pytest.approx(5.0, rel=0.15)


# --- power law -------------------------------------------------------------

def test_power_law_recovers_known_exponent():
    t = np.arange(1, 41, dtype=float)
    y = 900.0 * t ** (-1.35)
    fit = rg.fit_power_law(t, y)
    assert fit["alpha"] == pytest.approx(1.35, rel=1e-6)
    assert fit["r2"] > 0.999


# --- model selection -------------------------------------------------------

def test_aic_prefers_exponential_on_exponential_data():
    t = np.arange(1, 31, dtype=float)
    y = 1000.0 * np.exp(-t / 5.0)
    assert rg.fit_exponential(t, y)["aic"] < rg.fit_power_law(t, y)["aic"]


def test_aic_prefers_power_law_on_power_law_data():
    t = np.arange(1, 31, dtype=float)
    y = 1000.0 * t ** (-1.5)
    assert rg.fit_power_law(t, y)["aic"] < rg.fit_exponential(t, y)["aic"]


# --- two-phase -------------------------------------------------------------

def test_two_phase_recovers_synthetic_switching_point():
    """Exponential for 10 days, power-law after, joined continuously."""
    s_true = 10
    t = np.arange(1, 41, dtype=float)
    tau, alpha, a0 = 3.0, 1.2, 5000.0
    y = np.empty_like(t)
    early = t <= s_true
    y[early] = a0 * np.exp(-t[early] / tau)
    join = a0 * math.exp(-s_true / tau)
    y[~early] = join * (t[~early] / s_true) ** (-alpha)

    fit = rg.fit_two_phase(t, y, min_seg=4)
    assert fit["switch_day"] == pytest.approx(s_true, abs=1.0)
    assert fit["tau"] == pytest.approx(tau, rel=0.2)
    assert fit["alpha"] == pytest.approx(alpha, rel=0.25)


def test_two_phase_beats_single_exponential_on_two_phase_data():
    s_true = 9
    t = np.arange(1, 41, dtype=float)
    y = np.empty_like(t)
    early = t <= s_true
    y[early] = 4000.0 * np.exp(-t[early] / 2.5)
    join = 4000.0 * math.exp(-s_true / 2.5)
    y[~early] = join * (t[~early] / s_true) ** (-1.1)
    assert rg.fit_two_phase(t, y, min_seg=4)["aic"] < rg.fit_exponential(t, y)["aic"]


def test_two_phase_refuses_when_series_too_short():
    t = np.arange(1, 6, dtype=float)
    y = 100.0 * np.exp(-t / 2.0)
    assert rg.fit_two_phase(t, y, min_seg=4) is None


# --- bootstrap -------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    t = np.arange(1, 26, dtype=float)
    y = 1500.0 * np.exp(-t / 5.0) * rng.lognormal(0, 0.2, size=t.size)
    ci = rg.bootstrap_ci(t, y, rg.fit_exponential, "half_life", n=300, seed=1)
    point = rg.fit_exponential(t, y)["half_life"]
    assert ci[0] < point < ci[1]
    assert ci[0] > 0


def test_bootstrap_ci_narrows_with_cleaner_data():
    t = np.arange(1, 26, dtype=float)
    clean = 1500.0 * np.exp(-t / 5.0)
    rng = np.random.default_rng(4)
    noisy = clean * rng.lognormal(0, 0.35, size=t.size)
    w_clean = np.diff(rg.bootstrap_ci(t, clean, rg.fit_exponential, "half_life", n=300, seed=1))
    w_noisy = np.diff(rg.bootstrap_ci(t, noisy, rg.fit_exponential, "half_life", n=300, seed=1))
    assert w_clean[0] < w_noisy[0]


# --- input handling --------------------------------------------------------

def test_nonpositive_values_are_dropped_not_crashed():
    """Excess series hit zero once attention returns to baseline. Log-space
    fitting cannot use those points; they must be dropped, not NaN-poisoned."""
    t = np.arange(1, 11, dtype=float)
    y = np.array([900., 600., 400., 250., 150., 0., 80., -5., 40., 20.])
    fit = rg.fit_exponential(t, y)
    assert fit is not None
    assert fit["n"] == 8
    assert math.isfinite(fit["tau"])


def test_fit_returns_none_when_too_few_usable_points():
    t = np.array([1., 2., 3.])
    y = np.array([0., 0., 5.])
    assert rg.fit_exponential(t, y) is None


# ===================================================================
# Review response: C1 (estimator dependence), M1 (block bootstrap),
# M2 (selection-corrected AIC).
# ===================================================================

def test_nls_recovers_known_timescale_on_raw_scale():
    t = np.arange(1, 21, dtype=float)
    y = 1000.0 * np.exp(-t / 4.0)
    fit = rg.fit_exponential_nls(t, y)
    assert fit["tau"] == pytest.approx(4.0, rel=1e-4)
    assert fit["half_life"] == pytest.approx(4.0 * math.log(2), rel=1e-4)
    assert fit["model"] == "exponential_nls"


def test_nls_and_log_ols_agree_on_clean_data():
    """With no noise the two estimators must coincide. Divergence on real data
    is therefore a statement about the noise, not about the implementations."""
    t = np.arange(1, 26, dtype=float)
    y = 500.0 * np.exp(-t / 6.0)
    a = rg.fit_exponential(t, y)["half_life"]
    b = rg.fit_exponential_nls(t, y)["half_life"]
    assert b == pytest.approx(a, rel=1e-3)


def test_nls_and_log_ols_diverge_when_tail_is_noisy():
    """Log-space OLS up-weights small tail values, so a raised tail pulls its
    half-life up while NLS (which weights by absolute residual) resists.

    This is the mechanism behind review finding C1; the test pins the
    direction so a future refactor cannot silently remove it.
    """
    t = np.arange(1, 26, dtype=float)
    y = 5000.0 * np.exp(-t / 3.0)
    y[-8:] += 40.0                      # a raised, flat tail
    ols = rg.fit_exponential(t, y)["half_life"]
    nls = rg.fit_exponential_nls(t, y)["half_life"]
    assert ols > nls, f"expected log-OLS to overestimate: {ols} vs {nls}"


def test_nls_returns_none_on_too_few_points():
    assert rg.fit_exponential_nls(np.array([1.0, 2.0]), np.array([5.0, 3.0])) is None


def test_nls_refuses_non_decaying_series():
    t = np.arange(1, 11, dtype=float)
    assert rg.fit_exponential_nls(t, 10.0 * t) is None


# --- block bootstrap ------------------------------------------------------

def test_block_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(11)
    t = np.arange(1, 31, dtype=float)
    y = 2000.0 * np.exp(-t / 5.0) * rng.lognormal(0, 0.2, size=t.size)
    ci = rg.block_bootstrap_ci(t, y, rg.fit_exponential, "half_life",
                               block=4, n=400, seed=2)
    point = rg.fit_exponential(t, y)["half_life"]
    assert ci[0] < point < ci[1]


def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_residuals():
    """The whole point of M1: with serially correlated residuals the i.i.d.
    bootstrap understates uncertainty. Build a series with an AR(1) residual
    and check the block interval is not narrower."""
    rng = np.random.default_rng(5)
    t = np.arange(1, 41, dtype=float)
    e = np.zeros(t.size)
    for i in range(1, t.size):
        e[i] = 0.7 * e[i - 1] + rng.normal(0, 0.18)
    y = 3000.0 * np.exp(-t / 6.0) * np.exp(e)
    iid = rg.bootstrap_ci(t, y, rg.fit_exponential, "half_life", n=600, seed=3)
    blk = rg.block_bootstrap_ci(t, y, rg.fit_exponential, "half_life",
                                block=5, n=600, seed=3)
    assert (blk[1] - blk[0]) >= (iid[1] - iid[0]) * 0.98


def test_block_bootstrap_rejects_bad_block_length():
    t = np.arange(1, 11, dtype=float)
    y = 100.0 * np.exp(-t / 3.0)
    with pytest.raises(ValueError):
        rg.block_bootstrap_ci(t, y, rg.fit_exponential, "half_life", block=0)


# --- selection-corrected AIC ---------------------------------------------

def test_selection_penalty_is_two_log_candidates():
    assert rg.selection_penalty(17) == pytest.approx(2 * math.log(17))
    assert rg.selection_penalty(1) == 0.0
    assert rg.selection_penalty(0) == 0.0


def test_two_phase_advantage_can_fail_selection_correction():
    """A two-phase fit whose AIC gain is smaller than the cost of having
    searched for its breakpoint is not evidence for two phases."""
    gap, candidates = 5.58, 17          # the lookups channel, per review M2
    assert gap < rg.selection_penalty(candidates)


def test_breakpoint_candidate_count_matches_grid():
    t = np.arange(1, 27, dtype=float)
    y = 1000.0 * np.exp(-t / 4.0)
    assert rg.n_breakpoint_candidates(t.size, min_seg=5) == 17
