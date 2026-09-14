"""Decay regression: exponential, power-law, and the two-phase model.

Why three models rather than the one the proposal asked for. The in-project
proposal specified "exponential regression models ... to calculate the exact
half-life". For Wikipedia attention series that model is known to be wrong:

  Igarashi, N., Okada, Y., Sayama, H., & Sano, Y. (2022). A two-phase model of
  collective memory decay with a dynamical switching point. arXiv:2209.07033.

They fit an initial exponential phase followed by a power-law phase, and report
a switching point roughly 10-11 days after peak across earthquakes, deaths of
notable persons, aviation accidents, mass murders and terrorist attacks, with
model parameters "similar across all the event categories".

So a single exponential understates the tail and overstates how fast an event
is forgotten. All three are fitted here and compared by AIC, which makes the
choice of model an empirical result rather than an assumption.

Fitting is done in log space by least squares, which is what the linearised
forms permit and what keeps the estimators transparent:

  exponential   log y = log A - t / tau
  power law     log y = log B - alpha * log t

Uncertainty is by residual bootstrap in log space, reported as a 95% interval.
"""

import math

import numpy as np

MIN_POINTS = 5


def _clean(t, y):
    """Keep strictly positive points only.

    An excess series reaches zero once attention returns to baseline, and
    log-space fitting cannot use those points. Dropping them is the honest
    move; silently passing them to log() would poison the fit with NaNs.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (y > 0) & np.isfinite(y) & (t > 0)
    return t[keep], y[keep]


def _r2(logy, pred):
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _aic(logy, pred, k):
    """AIC under Gaussian errors in log space. k counts fitted parameters."""
    n = logy.size
    ss_res = float(np.sum((logy - pred) ** 2))
    if ss_res <= 0:
        return -math.inf
    return n * math.log(ss_res / n) + 2 * k


def fit_exponential(t, y):
    """y = A exp(-t / tau). Returns None if too few usable points."""
    t, y = _clean(t, y)
    if t.size < MIN_POINTS:
        return None
    logy = np.log(y)
    slope, intercept = np.polyfit(t, logy, 1)
    if slope >= 0:
        return None  # not a decay; refuse rather than report a negative tau
    tau = -1.0 / slope
    pred = intercept + slope * t
    return {
        "model": "exponential",
        "tau": tau,
        "half_life": tau * math.log(2.0),
        "amplitude": math.exp(intercept),
        "r2": _r2(logy, pred),
        "aic": _aic(logy, pred, 2),
        "n": int(t.size),
    }


def fit_power_law(t, y):
    """y = B t^(-alpha)."""
    t, y = _clean(t, y)
    if t.size < MIN_POINTS:
        return None
    logy, logt = np.log(y), np.log(t)
    slope, intercept = np.polyfit(logt, logy, 1)
    if slope >= 0:
        return None
    pred = intercept + slope * logt
    return {
        "model": "power_law",
        "alpha": -slope,
        "amplitude": math.exp(intercept),
        "r2": _r2(logy, pred),
        "aic": _aic(logy, pred, 2),
        "n": int(t.size),
    }


def fit_two_phase(t, y, min_seg=5):
    """Exponential up to a switching day, power-law after it.

    The switching day is chosen by grid search over candidate days, minimising
    the pooled log-space residual sum. Both segments must retain at least
    `min_seg` points, or the "fit" is just interpolating a handful of noise.

    Returns None when the series is too short to support two segments — which
    is a real outcome for a short-lived event, not an error to paper over.
    """
    t, y = _clean(t, y)
    if t.size < 2 * min_seg:
        return None
    logy, logt = np.log(y), np.log(t)

    best = None
    for i in range(min_seg, t.size - min_seg + 1):
        te, tl = t[:i], t[i:]
        ye, yl = logy[:i], logy[i:]

        se, ie = np.polyfit(te, ye, 1)
        pe = ie + se * te
        sl, il = np.polyfit(np.log(tl), yl, 1)
        pl = il + sl * np.log(tl)

        if se >= 0 or sl >= 0:
            continue
        ss = float(np.sum((ye - pe) ** 2) + np.sum((yl - pl) ** 2))
        if best is None or ss < best["ss"]:
            best = {
                "ss": ss, "switch_idx": i, "switch_day": float(t[i - 1]),
                "tau": -1.0 / se, "alpha": -sl,
                "pred": np.concatenate([pe, pl]),
            }

    if best is None:
        return None
    return {
        "model": "two_phase",
        "switch_day": best["switch_day"],
        "tau": best["tau"],
        "half_life": best["tau"] * math.log(2.0),
        "alpha": best["alpha"],
        "r2": _r2(logy, best["pred"]),
        # 5 parameters: two slopes, two intercepts, one switching point.
        "aic": _aic(logy, best["pred"], 5),
        "n": int(t.size),
    }


def best_model(t, y, min_seg=5):
    """Fit all three and return them ranked by AIC, lowest first."""
    fits = [f for f in (fit_exponential(t, y), fit_power_law(t, y),
                        fit_two_phase(t, y, min_seg)) if f]
    return sorted(fits, key=lambda f: f["aic"])


def bootstrap_ci(t, y, fitter, key, n=1000, seed=0, level=95):
    """Residual bootstrap in log space for one fitted quantity.

    Residuals are resampled with replacement and added back to the fitted log
    curve, so the interval reflects scatter about the fit rather than assuming
    a parametric error model.
    """
    t, y = _clean(t, y)
    base = fitter(t, y)
    if base is None:
        return None
    logy = np.log(y)

    if base["model"] == "power_law":
        pred = math.log(base["amplitude"]) - base["alpha"] * np.log(t)
    elif base["model"] == "exponential":
        pred = math.log(base["amplitude"]) - t / base["tau"]
    else:
        return None  # two-phase residuals are not exchangeable across segments

    resid = logy - pred
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        boot = np.exp(pred + rng.choice(resid, size=resid.size, replace=True))
        f = fitter(t, boot)
        if f and key in f and math.isfinite(f[key]):
            vals.append(f[key])
    if len(vals) < 20:
        return None
    lo = (100 - level) / 2
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo)))


# ===================================================================
# Review response: C1, M1, M2.
#
# C1. Log-space OLS is not a neutral estimator. Clauset, Shalizi & Newman
#     (2009, arXiv:0706.1062) show that least squares on log-transformed data
#     gives biased parameters and invalid standard errors, because the noise in
#     the logarithmic dependent variable is not Gaussian. On this project's own
#     data the two estimators disagree by up to 2.7x on the daily channels, so
#     both are now reported and neither is privileged.
#
# M1. Residuals here are positively autocorrelated (Durbin-Watson 1.17 and
#     1.36 on the two daily channels), so the i.i.d. residual bootstrap
#     understates uncertainty. A moving-block bootstrap preserves short-range
#     dependence.
#
# M2. The two-phase breakpoint is chosen by grid search but AIC charged it as
#     one free parameter. Searching C candidates costs roughly 2*ln(C) more.
# ===================================================================

def fit_exponential_nls(t, y, max_nfev=2000):
    """y = A exp(-t/tau) fitted on the RAW scale by nonlinear least squares.

    Minimises absolute residuals rather than log residuals, so large early
    values dominate the fit instead of small tail values. This is the
    estimator log-space OLS should be checked against, not replaced by:
    neither is unconditionally correct, and their disagreement is itself the
    reportable quantity.
    """
    t, y = _clean(t, y)
    if t.size < MIN_POINTS:
        return None
    try:
        from scipy import optimize
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return None

    def resid(p):
        return p[0] * np.exp(-t / p[1]) - y

    guess = [float(np.max(y)), max(1.0, float(t[-1]) / 3.0)]
    try:
        sol = optimize.least_squares(
            resid, x0=guess, bounds=([0.0, 1e-3], [np.inf, 1e5]), max_nfev=max_nfev
        )
    except Exception:  # pragma: no cover - solver failure is a real outcome
        return None
    if not sol.success:
        return None

    A, tau = float(sol.x[0]), float(sol.x[1])
    # A fitted decay must actually decay over the observed span; a tau far
    # larger than the window is a flat line, not a half-life.
    if tau <= 0 or tau > 50 * float(t[-1]):
        return None
    pred = A * np.exp(-t / tau)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "model": "exponential_nls",
        "tau": tau,
        "half_life": tau * math.log(2.0),
        "amplitude": A,
        # r2 is on the raw scale here, so it is NOT comparable to the
        # log-space r2 of fit_exponential. Labelled to prevent that mistake.
        "r2_raw": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(t.size),
    }


def block_bootstrap_ci(t, y, fitter, key, block=4, n=1000, seed=0, level=95):
    """Moving-block bootstrap on log-space residuals.

    Resamples contiguous blocks of length `block` rather than single
    residuals, so short-range serial dependence survives resampling. Use this
    wherever residual autocorrelation is non-negligible; `bootstrap_ci` is
    retained only for the i.i.d. comparison the paper now reports.
    """
    if block is None or block < 1:
        raise ValueError(f"block length must be >= 1, got {block}")
    t, y = _clean(t, y)
    base = fitter(t, y)
    if base is None:
        return None
    logy = np.log(y)

    if base["model"] == "exponential":
        pred = math.log(base["amplitude"]) - t / base["tau"]
    elif base["model"] == "power_law":
        pred = math.log(base["amplitude"]) - base["alpha"] * np.log(t)
    else:
        return None

    resid = logy - pred
    N = resid.size
    if N < block:
        raise ValueError(f"series of {N} points cannot support block {block}")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(N / block))
    vals = []
    for _ in range(n):
        starts = rng.integers(0, N - block + 1, size=n_blocks)
        r = np.concatenate([resid[s:s + block] for s in starts])[:N]
        f = fitter(t, np.exp(pred + r))
        if f and key in f and math.isfinite(f[key]):
            vals.append(f[key])
    if len(vals) < 20:
        return None
    lo = (100 - level) / 2
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo)))


def n_breakpoint_candidates(n_points, min_seg=5):
    """How many breakpoints fit_two_phase actually searches over."""
    return max(0, n_points - 2 * min_seg + 1)


def selection_penalty(n_candidates):
    """Extra AIC penalty for having selected a breakpoint by search.

    Choosing the best of C candidates is worth roughly 2*ln(C) in AIC terms.
    Charging nothing for the search makes a segmented fit look better than it
    is; this is review finding M2.
    """
    if n_candidates is None or n_candidates < 2:
        return 0.0
    return 2.0 * math.log(n_candidates)


def two_phase_survives_selection(t, y, min_seg=5):
    """Does two-phase still beat the exponential after paying for the search?

    Returns a dict with the gap, the penalty and the verdict, or None when
    either fit is unavailable. The verdict is what the paper may claim.
    """
    tp = fit_two_phase(t, y, min_seg=min_seg)
    ex = fit_exponential(t, y)
    if not tp or not ex:
        return None
    tc, _ = _clean(t, y)
    cands = n_breakpoint_candidates(tc.size, min_seg=min_seg)
    pen = selection_penalty(cands)
    gap = ex["aic"] - tp["aic"]
    return {
        "aic_gap": gap,
        "candidates": cands,
        "selection_penalty": pen,
        "survives": bool(gap > pen),
    }
