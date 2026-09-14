"""HawkesN: Hawkes process with a finite population, ported to Python.

Ported from the reference R implementation in
`repos/sir-hawkes/scripts/functions-SIR-HawkesN.R` (Rizoiu, Mishra, Kong,
Carman & Xie, *SIR-Hawkes: Linking Epidemic Models and Hawkes Processes to
Model Diffusions in Finite Populations*, WWW 2018, arXiv:1711.01679).
That repository is CC BY-NC 4.0; this is a re-implementation of the published
model for non-commercial research use, not a copy of their code.

WHY THIS MODEL. A review objection to the parent study held that a Hacker News
thread's comment rate decays partly because the pool of people who will ever
comment on it is exhausted, not because attention decayed -- so its half-life
is not commensurable with a Wikipedia pageview half-life. HawkesN does not
argue with that objection, it *measures* it: the fitted population size `N`
separates exhaustion from kernel decay `theta`.

  N close to the observed event count  -> exhaustion dominates; the observed
                                          decay is largely mechanical
  N much larger than the count         -> the pool was never near exhausted;
                                          the decay is genuine attention loss

MODEL (exponential kernel, finite population). For events at times
t_1 <= ... <= t_n, the conditional intensity is

    lambda(t) = K * (1 - N_t/N) * theta * sum_{t_i <= t} exp(-theta (t - t_i))

with N_t = min(#{t_i <= t}, N). Free parameters: K (virality), theta (decay
rate), N (effective population). The `c` parameter in the R signature belongs
to the power-law kernel and is unused for the exponential kernel.

Negative log-likelihood over [0, T], T = t_n:

    NLL = integral_0^T lambda(u) du  -  sum_i log lambda(t_i)

The integral uses the closed form of the reference `integrateLambda`.
"""

import math

import numpy as np


def _intensity_at_events(t, K, theta, N):
    """lambda(t_i^-): the LEFT LIMIT of the intensity at each event time.

    Strictly earlier events only. This matters and was got wrong once: a
    literal transcription of the reference `lambda()` includes the event
    itself, whose self-term is theta*exp(0) = theta, so
    sum_i log lambda(t_i) ~ n*log(theta) and the NLL falls without bound as
    theta -> infinity. Fitting then drives theta and N to their box
    constraints and recovers population sizes 400-500x too large. A Hawkes
    likelihood requires the left limit.

    The exponential sum uses the recursion
    S_i = exp(-theta*dt) * (S_{i-1} + 1), with S_1 = 0, turning an O(n^2)
    evaluation into O(n) -- what makes a 1,000-event thread tractable.
    """
    n = t.size
    S = np.zeros(n)
    for i in range(1, n):
        S[i] = math.exp(-theta * (t[i] - t[i - 1])) * (S[i - 1] + 1.0)
    # events strictly before t_i, capped at the population size
    counts = np.minimum(np.arange(n), N)
    return K * (1.0 - counts / N) * theta * S


def _integral(t, K, theta, N):
    """Closed-form integral of lambda over [0, T], T = t_{n-1}.

    Derived interval-wise rather than transcribed, because a transcription got
    the population count off by one -- invisible at large N (ratio to
    quadrature 1.001) but a 6.5% error at N=20.

    On the gap (t_k, t_{k+1}) exactly k+1 events have occurred, so the
    population factor is (1 - (k+1)/N) and applies to the sum over all
    sources i <= k:

        int = K * sum_{k=0}^{n-2} (1 - (k+1)/N)
                  * sum_{i<=k} [ e^{-theta(t_k - t_i)} - e^{-theta(t_{k+1} - t_i)} ]

    The inner sum is accumulated by the same recursion used for the
    intensity, keeping the whole thing O(n).
    """
    n = t.size
    if n < 2:
        return 0.0
    total = 0.0
    # A_k = sum_{i<=k} exp(-theta (t_k - t_i)), recursively
    A = 1.0  # k = 0: only i = 0, exponent 0
    for k in range(n - 1):
        if k > 0:
            A = 1.0 + math.exp(-theta * (t[k] - t[k - 1])) * A
        decay = math.exp(-theta * (t[k + 1] - t[k]))
        avail = N - (k + 1)
        if avail < 0:
            avail = 0.0
        total += (avail / N) * A * (1.0 - decay)
    return K * total


def neg_log_likelihood(params, t):
    """NLL for (K, theta, N). Returns +inf on invalid or non-finite values."""
    K, theta, N = params
    n = t.size
    if K <= 0 or theta <= 0 or N < n:
        return np.inf
    try:
        lam = _intensity_at_events(t, K, theta, N)
        # lambda(t_0^-) is identically zero: there is no background rate and no
        # earlier event, so the first event contributes no likelihood term.
        # The reference does the same via its `startEvent` index.
        lam = lam[1:]
        if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
            return np.inf
        val = _integral(t, K, theta, N) - float(np.sum(np.log(lam)))
    except (FloatingPointError, OverflowError, ValueError):
        return np.inf
    return val if np.isfinite(val) else np.inf


def fit(t, n_starts=6, seed=0, verbose=False):
    """Fit (K, theta, N) by bounded L-BFGS-B from several random starts.

    `N` is searched as a multiplier over the observed event count so the
    optimiser works on a scale-free quantity; the returned `N` is absolute.
    Multiple starts are used because the likelihood is not convex and a single
    start tends to stop on a boundary.
    """
    from scipy import optimize

    t = np.asarray(t, dtype=float)
    t = t - t[0]
    n = t.size
    if n < 20:
        return None

    rng = np.random.default_rng(seed)
    span = max(float(t[-1]), 1e-9)
    best = None

    for s in range(n_starts):
        x0 = [float(rng.uniform(0.2, 3.0)),
              float(rng.uniform(0.5, 10.0)) / span,
              float(rng.uniform(1.02, 4.0))]

        def obj(p):
            return neg_log_likelihood((p[0], p[1], p[2] * n), t)

        try:
            res = optimize.minimize(
                obj, x0=x0, method="L-BFGS-B",
                bounds=[(1e-6, 1e4), (1e-9, 1e4), (1.0 + 1e-9, 500.0)])
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
        if verbose:
            print(f"    start {s + 1}: nll={res.fun:.3f} K={res.x[0]:.4f} "
                  f"theta={res.x[1]:.5f} N={res.x[2] * n:.0f}")

    if best is None:
        return None

    K, theta, mult = float(best.x[0]), float(best.x[1]), float(best.x[2])
    N = mult * n
    return {
        "K": K,
        "theta": theta,
        "N": N,
        "n_events": n,
        "N_multiplier": mult,
        # Half-life of the endogenous kernel: the decay that is NOT exhaustion.
        "kernel_half_life": math.log(2.0) / theta if theta > 0 else float("nan"),
        # Share of the fitted population consumed by the observed cascade.
        # Near 1 means exhaustion drove the observed decay.
        "exhaustion": n / N,
        "nll": float(best.fun),
        "span": float(t[-1]),
        "at_N_bound": mult <= 1.0 + 1e-6 or mult >= 499.0,
    }


def simulate(K, theta, N, t_max, seed=0, max_events=20000):
    """Simulate a HawkesN cascade by Ogata thinning.

    Exists so `fit` can be validated against data with known parameters rather
    than trusted on the strength of the port alone.
    """
    rng = np.random.default_rng(seed)
    events = [0.0]
    t = 0.0
    while t < t_max and len(events) < max_events and len(events) < N:
        arr = np.asarray(events)
        n_t = min(arr.size, N)
        lam_bar = K * (1.0 - n_t / N) * theta * float(
            np.sum(np.exp(-theta * (t - arr)))) + 1e-12
        t = t + rng.exponential(1.0 / lam_bar)
        if t >= t_max:
            break
        arr = np.asarray(events)
        n_t = min(arr.size, N)
        lam = K * (1.0 - n_t / N) * theta * float(
            np.sum(np.exp(-theta * (t - arr))))
        if rng.uniform() <= lam / lam_bar:
            events.append(t)
    return np.asarray(events)


def profile_N(t, n_grid=22, max_mult=40.0, n_starts=4, seed=0):
    """Profile likelihood over the population size N.

    Validation on simulated cascades showed K and theta trade off against each
    other (K over-estimated exactly where theta is under-estimated), so their
    point estimates are not trustworthy at these cascade lengths. N is
    recovered far better, and N is the parameter the finite-population
    question actually turns on: is the pool near exhaustion (N ~ n) or far
    from it (N >> n)?

    For each N on a log grid, K and theta are re-optimised and the profile NLL
    recorded. The interval is the standard chi-square(1)/2 drop of 1.92 in NLL
    from the minimum.

    Returns the grid, the profile, the best N, and the interval - reported as
    an interval precisely because the point estimate is not reliable alone.
    """
    from scipy import optimize

    t = np.asarray(t, dtype=float)
    t = t - t[0]
    n = t.size
    if n < 20:
        return None
    span = max(float(t[-1]), 1e-9)
    rng = np.random.default_rng(seed)

    mults = np.unique(np.round(np.geomspace(1.001, max_mult, n_grid), 4))
    prof = []
    for mlt in mults:
        Nv = mlt * n
        best = np.inf
        for _ in range(n_starts):
            x0 = [float(rng.uniform(0.2, 6.0)),
                  float(rng.uniform(0.5, 10.0)) / span]

            def obj(p):
                return neg_log_likelihood((p[0], p[1], Nv), t)

            try:
                r = optimize.minimize(obj, x0=x0, method="L-BFGS-B",
                                      bounds=[(1e-6, 1e5), (1e-9, 1e5)])
            except Exception:
                continue
            if np.isfinite(r.fun) and r.fun < best:
                best = float(r.fun)
        prof.append(best)

    prof = np.asarray(prof, dtype=float)
    ok = np.isfinite(prof)
    if not ok.any():
        return None
    mults, prof = mults[ok], prof[ok]
    j = int(np.argmin(prof))
    thresh = prof[j] + 1.92
    inside = mults[prof <= thresh]
    return {
        "n_events": n,
        "N_hat": float(mults[j] * n),
        "N_mult_hat": float(mults[j]),
        "N_lo": float(inside.min() * n),
        "N_hi": float(inside.max() * n),
        "exhaustion_hat": float(1.0 / mults[j]),
        "exhaustion_lo": float(1.0 / inside.max()),
        "exhaustion_hi": float(1.0 / inside.min()),
        "hit_grid_edge": bool(inside.max() >= mults[-1] - 1e-9),
        "nll_min": float(prof[j]),
        "grid_mult": mults.tolist(),
        "profile_nll": prof.tolist(),
    }
