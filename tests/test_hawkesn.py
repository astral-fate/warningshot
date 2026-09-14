"""Tests for the HawkesN port.

This module exists because the first port was wrong twice, in ways that would
have produced confident nonsense:

  1. A literal transcription of the reference `lambda()` included the event
     itself in lambda(t_i). Its self-term is theta*exp(0) = theta, so
     sum log lambda ~ n log theta and the NLL fell without bound as
     theta -> inf. Fitting drove theta and N to their box constraints and
     recovered population sizes 400-500x too large.
  2. The closed-form integral had the population count off by one. Invisible
     at large N (ratio to quadrature 1.001), a 6.5% error at N = 20.

Both are pinned below, along with the discriminating behaviour the model is
actually used for: telling an exhausted cascade from a non-exhausted one.
"""

import numpy as np
import pytest
from scipy import integrate

from warningshot.core import hawkesn as hn


# --- intensity ------------------------------------------------------------

def test_first_event_has_zero_left_intensity():
    """No background rate and no earlier event, so lambda(t_0^-) = 0. If this
    is nonzero the self-term bug is back."""
    t = np.array([0.0, 1.0, 2.0, 4.0])
    lam = hn._intensity_at_events(t, 1.0, 0.5, 100)
    assert lam[0] == 0.0


def test_intensity_uses_strictly_earlier_events():
    """lambda(t_1^-) must equal the single earlier event's kernel value."""
    t = np.array([0.0, 2.0])
    K, th, N = 1.3, 0.7, 50
    lam = hn._intensity_at_events(t, K, th, N)
    expected = K * (1.0 - 1 / N) * th * np.exp(-th * 2.0)
    assert lam[1] == pytest.approx(expected, rel=1e-12)


def test_nll_is_not_unbounded_in_theta():
    """The pathology that broke the first port: NLL must not fall forever as
    theta grows."""
    t = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    vals = [hn.neg_log_likelihood((1.0, th, 1000), t)
            for th in (0.1, 1.0, 10.0, 100.0)]
    assert vals[-1] > vals[0], f"NLL still decreasing in theta: {vals}"


# --- integral -------------------------------------------------------------

@pytest.mark.parametrize("K,theta,N", [
    (1.0, 0.5, 1000), (2.0, 1.0, 50), (0.8, 0.2, 20),
    (3.0, 2.0, 15), (1.5, 0.05, 200),
])
def test_closed_form_integral_matches_quadrature(K, theta, N):
    """Catches the off-by-one in the population count, which only shows at
    small N."""
    t = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0])

    def lam(u):
        prior = t[t < u]
        if prior.size == 0:
            return 0.0
        n_u = min(prior.size, N)
        return K * (1.0 - n_u / N) * theta * float(np.sum(np.exp(-theta * (u - prior))))

    closed = hn._integral(t, K, theta, N)
    num, _ = integrate.quad(lam, 0, t[-1], limit=800)
    assert closed == pytest.approx(num, rel=1e-3)


def test_integral_is_zero_for_single_event():
    assert hn._integral(np.array([0.0]), 1.0, 1.0, 10) == 0.0


# --- guards ---------------------------------------------------------------

def test_nll_rejects_population_below_event_count():
    t = np.arange(10.0)
    assert hn.neg_log_likelihood((1.0, 0.5, 5), t) == np.inf


@pytest.mark.parametrize("bad", [(0.0, 0.5, 100), (1.0, 0.0, 100), (-1.0, 0.5, 100)])
def test_nll_rejects_nonpositive_parameters(bad):
    assert hn.neg_log_likelihood(bad, np.arange(20.0)) == np.inf


def test_fit_declines_on_too_few_events():
    assert hn.fit(np.arange(5.0)) is None


# --- the behaviour the model is used for ---------------------------------

def _sim_multiseed(K, theta, N, n_seed, t_max, seed=0):
    """Subcritical cascade with many immigrants: many events WITHOUT
    exhausting the population. The discriminating case."""
    rng = np.random.default_rng(seed)
    events = list(np.zeros(n_seed))
    t = 0.0
    while t < t_max and len(events) < N:
        arr = np.asarray(events)
        n_t = min(arr.size, N)
        lb = K * (1 - n_t / N) * theta * float(np.sum(np.exp(-theta * (t - arr)))) + 1e-12
        t += rng.exponential(1.0 / lb)
        if t >= t_max:
            break
        arr = np.asarray(events)
        n_t = min(arr.size, N)
        lam = K * (1 - n_t / N) * theta * float(np.sum(np.exp(-theta * (t - arr))))
        if rng.uniform() <= lam / lb:
            events.append(t)
    return np.asarray(sorted(events))


def test_detects_an_exhausted_cascade():
    """A supercritical cascade in a small population runs the pool down;
    the fit should report exhaustion near 1."""
    ev = hn.simulate(3.0, 0.5, 400, t_max=300, seed=3)
    if ev.size < 60:
        pytest.skip("simulation produced too few events")
    f = hn.fit(ev, n_starts=8, seed=3)
    assert f is not None
    assert f["exhaustion"] > 0.8, f"expected exhaustion, got {f['exhaustion']:.2f}"


def test_detects_a_non_exhausted_cascade():
    """A subcritical cascade in a large population does NOT run the pool down;
    the fit must not claim exhaustion."""
    ev = _sim_multiseed(0.6, 0.5, 5000, n_seed=200, t_max=300, seed=3)
    if ev.size < 60:
        pytest.skip("simulation produced too few events")
    f = hn.fit(ev, n_starts=8, seed=3)
    assert f is not None
    assert f["exhaustion"] < 0.6, f"falsely claimed exhaustion: {f['exhaustion']:.2f}"


def test_exhaustion_estimate_is_biased_toward_exhaustion():
    """Documented bias: on a known non-exhausted cascade (true n/N ~ 0.09) the
    estimate comes out higher. A 'not exhausted' verdict is therefore the
    trustworthy direction, and this test records why."""
    ev = _sim_multiseed(0.6, 0.5, 5000, n_seed=200, t_max=300, seed=3)
    if ev.size < 60:
        pytest.skip("simulation produced too few events")
    f = hn.fit(ev, n_starts=8, seed=3)
    true_exhaustion = ev.size / 5000
    assert f["exhaustion"] >= true_exhaustion
