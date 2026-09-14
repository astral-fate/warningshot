"""Recompute the headline numbers of the paper and README from the shipped artifacts.

Pattern taken from `repos/secret-loyalties-testbed/src/verify.py`, which this
project's repo index calls the single best idea in any of the winner repos: the
README claims its numbers can be checked, so checking them is one command that
needs no network and no API key.

  ok    the README and the committed results agree
  FAIL  they disagree, which is a defect in the README
  SKIP  the result this row checks is absent from the artifacts

SKIP exists because one data channel is genuinely fragile: GDELT rate-limits to
one request every five seconds and returned HTTP 429 through five retries on a
cold-cache run, so `media` can be missing through no fault of the analysis.
Crashing on that would make the whole check unusable to a reproducer who is
merely unlucky, and silently dropping the row would let a real regression hide.

    python scripts/verify.py [results_dir]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warningshot import paths  # noqa: E402

CHECKS = []


# Optional argv[1] overrides the results directory, so the SKIP path can be
# exercised against a fixture instead of only when GDELT happens to fail.
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.RESULTS


def load(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def check(label, claimed, computed, tol=0.0):
    """computed=None means the underlying result is absent -> SKIP."""
    if computed is None:
        CHECKS.append((label, claimed, "absent", "SKIP"))
        return
    if isinstance(claimed, (int, float)) and isinstance(computed, (int, float)):
        ok = abs(claimed - computed) <= tol
    else:
        ok = claimed == computed
    CHECKS.append((label, claimed, computed, "ok" if ok else "FAIL"))


def dig(d, *keys, rnd=None):
    """Walk nested keys, returning None if any is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, (dict, list)) or (
            isinstance(cur, dict) and k not in cur
        ):
            return None
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return None
    if cur is None:
        return None
    return round(cur, rnd) if rnd is not None else cur


mm = load("multimodal.json")
ch = {c["channel"]: c for c in mm["channels"]}
sp = mm.get("spread") or {}

# --- Key Result: half-lives and intervals ---------------------------------
check("KR engagement half-life (hours)", 7.05, dig(ch, "engagement", "half_life", rnd=2), 0.01)
check("KR engagement CI low (hours)", 6.04, dig(ch, "engagement", "half_life_ci", 0, rnd=2), 0.01)
check("KR engagement CI high (hours)", 8.49, dig(ch, "engagement", "half_life_ci", 1, rnd=2), 0.01)
check("KR engagement R2", 0.787, dig(ch, "engagement", "exponential_r2", rnd=3), 0.001)
check("KR engagement n (hourly buckets)", 48, dig(ch, "engagement", "n_points"))
check("KR engagement comments", 1157, dig(ch, "engagement", "n_comments"))

check("KR lookups half-life (days)", 5.45, dig(ch, "lookups", "half_life", rnd=2), 0.01)
check("KR lookups CI low (days)", 4.64, dig(ch, "lookups", "half_life_ci", 0, rnd=2), 0.01)
check("KR lookups CI high (days)", 6.59, dig(ch, "lookups", "half_life_ci", 1, rnd=2), 0.01)
check("KR lookups R2", 0.818, dig(ch, "lookups", "exponential_r2", rnd=3), 0.001)
check("KR lookups n (days)", 26, dig(ch, "lookups", "n_points"))

# GDELT-dependent. SKIP, not FAIL, when the channel could not be fetched.
check("KR media half-life (days)", 6.11, dig(ch, "media", "half_life", rnd=2), 0.01)
check("KR media CI low (days)", 4.82, dig(ch, "media", "half_life_ci", 0, rnd=2), 0.01)
check("KR media CI high (days)", 8.86, dig(ch, "media", "half_life_ci", 1, rnd=2), 0.01)
check("KR media R2", 0.630, dig(ch, "media", "exponential_r2", rnd=3), 0.001)

# --- Key Finding: spread, and which pairs actually separate ----------------
# The ratio depends on which channels were available: with media present the
# slowest channel is media (20.8x); without it, lookups (18.6x). Both are
# checked so a partial run still verifies the one it can produce.
ratio = dig(sp, "ratio", rnd=1)
if dig(ch, "media", "half_life") is not None:
    check("KF slowest/fastest ratio (all 3 channels)", 20.8, ratio, 0.05)
    check("KF slowest channel", "media", dig(sp, "slowest"))
else:
    check("KF slowest/fastest ratio (media absent)", 18.6, ratio, 0.05)
    check("KF slowest channel", "lookups", dig(sp, "slowest"))
check("KF fastest channel", "engagement", dig(sp, "fastest"))

pair = {(p["a"], p["b"]): p for p in sp.get("pairwise", [])}
check("KF lookups vs media intervals overlap", True,
      pair.get(("lookups", "media"), {}).get("intervals_overlap"))
check("KF lookups vs engagement separated", True,
      pair.get(("lookups", "engagement"), {}).get("separated"))
check("KF media vs engagement separated", True,
      pair.get(("media", "engagement"), {}).get("separated"))

# --- Second-impulse claim: the full window cannot be fitted ---------------
si = {s["horizon_days"]: s for s in mm.get("second_impulse", [])}
check("SI 25d window R2", 0.883, dig(si, 25, "r2", rnd=3), 0.001)
check("SI 25d switch day", 20.0, dig(si, 25, "switch_day"), 0.01)
check("SI 45d window R2", 0.632, dig(si, 45, "r2", rnd=3), 0.001)
check("SI 45d switch lands on second impulse", 36.0, dig(si, 45, "switch_day"), 0.01)

# --- Model selection: two-phase beats the proposal's single exponential ---
check("MS lookups best model", "two_phase", dig(ch, "lookups", "best_model"))
check("MS engagement best model", "two_phase", dig(ch, "engagement", "best_model"))
check("MS media best model", "two_phase", dig(ch, "media", "best_model"))


# --- Review response: C1 estimator dependence, C2 contamination, ---------
# --- M1 block intervals, M2 selection-corrected AIC, M3 season null ------
# These live in a separate artifact so a partial run can still verify the
# multimodal numbers above.
try:
    rr = load("robustness.json")
except FileNotFoundError:
    rr = None

if rr is None:
    for label in ("C1 spread log-OLS", "C1 spread NLS", "C2 OLS shift ratio",
                  "M2 channels surviving selection", "M3 season-matched percentile"):
        check(label, "see robustness.json", None)
else:
    est = rr.get("estimator_comparison") or {}
    check("C1 spread log-OLS", 20.8, dig(est, "spread_ols", rnd=1), 0.05)
    check("C1 spread NLS", 12.2, dig(est, "spread_nls", rnd=1), 0.05)
    check("C1 ordering preserved under both", True, dig(est, "ordering_preserved"))
    ec = {c["channel"]: c for c in est.get("channels", [])}
    check("C1 lookups NLS half-life (d)", 2.05,
          dig(ec, "lookups", "half_life_nls_days", rnd=2), 0.01)
    check("C1 media NLS half-life (d)", 3.55,
          dig(ec, "media", "half_life_nls_days", rnd=2), 0.01)

    con = rr.get("contamination_sensitivity") or {}
    check("C2 excluded date count", 2, len(dig(con, "excluded_dates") or []))
    check("C2 OLS shift ratio", 0.99, dig(con, "ols_shift_ratio", rnd=2), 0.01)
    check("C2 excluded-fit OLS half-life (d)", 5.41,
          dig(con, "excluded", "half_life_ols", rnd=2), 0.01)

    ivl = rr.get("interval_comparison") or {}
    ic = {c["channel"]: c for c in ivl.get("channels", [])}
    check("M1 block length", 4, dig(ivl, "block_length"))
    check("M1 lookups Durbin-Watson", 1.17,
          dig(ic, "lookups", "durbin_watson", rnd=2), 0.01)
    check("M1 media Durbin-Watson", 1.35,
          dig(ic, "media", "durbin_watson", rnd=2), 0.01)
    bp = {(p["a"], p["b"]): p for p in ivl.get("pairwise_block", [])}
    check("M1 lookups vs media still overlap under block CI", True,
          bp.get(("lookups", "media"), {}).get("intervals_overlap"))
    check("M1 lookups vs engagement still separated under block CI", False,
          bp.get(("lookups", "engagement"), {}).get("intervals_overlap"))

    sel = rr.get("selection_corrected_models") or {}
    check("M2 channels surviving selection", 2, dig(sel, "n_surviving"))
    check("M2 channels tested", 3, dig(sel, "n_tested"))

    smn = rr.get("season_matched_null") or {}
    check("M3 season-matched windows", 13, dig(smn, "n_windows"))
    check("M3 season-matched percentile", 53.8, dig(smn, "percentile", rnd=1), 0.05)
    check("M3 observed concept excess", 17699.0, dig(smn, "observed", rnd=1), 0.5)


# --- Review 3: the nulls and unit corrections that reshaped the paper ------
# These are now the paper's headline numbers. The withdrawn cross-channel
# decay claim is checked too, as the value the paper reports as retracted.
try:
    r3 = load("nulls.json")
except FileNotFoundError:
    r3 = None

if r3 is None:
    for label in ("W2 thread-null percentile", "W3 disclosure percentile",
                  "W4 inversion in baseline-days", "W5 daily-binned half-life",
                  "W7 ratio CI contains 1"):
        check(label, "see nulls.json", None)
else:
    hn = r3.get("hn_thread_null") or {}
    check("W2 thread-null n fitted", 30, dig(hn, "n_fitted"))
    check("W2 incident half-life (h)", 7.05, dig(hn, "observed_half_life_h", rnd=2), 0.01)
    check("W2 thread-null percentile", 33.0, dig(hn, "percentile", rnd=0), 0.5)
    check("W2 thread-null median (h)", 7.69, dig(hn, "null_median", rnd=2), 0.01)
    check("W2 incident slower than median", True,
          (dig(hn, "observed_half_life_h") or 0) < (dig(hn, "null_median") or 0))

    dr = r3.get("disclosure_ratio") or {}
    check("W3 disclosure-day percentile", 91.0, dig(dr, "percentile", rnd=0), 0.5)
    check("W3 days at or above 1.22x", 15, dig(dr, "days_at_or_above"))
    check("W3 n days in null", 166, dig(dr, "n_days"))

    inv = r3.get("inversion_units") or {}
    check("W4 victim baseline-days", 40.16, dig(inv, "victim", "baseline_days", rnd=2), 0.01)
    check("W4 June developer baseline-days", 1.74,
          dig(inv, "developer_june", "baseline_days", rnd=2), 0.01)
    check("W4 July developer baseline-days", 0.35,
          dig(inv, "developer_july", "baseline_days", rnd=2), 0.01)
    check("W4 inversion in baseline-days", 23.06,
          dig(inv, "victim_vs_june_dev_baseline_days", rnd=2), 0.02)
    check("W4 inversion in raw views", 1.46,
          dig(inv, "victim_vs_june_dev_raw", rnd=2), 0.01)
    check("W4 published 7x as raw", 8.41,
          dig(inv, "june_dev_vs_july_dev_raw", rnd=2), 0.01)
    check("W4 published 7x in baseline-days", 4.98,
          dig(inv, "june_dev_vs_july_dev_baseline_days", rnd=2), 0.02)

    rc = r3.get("resolution_control") or {}
    check("W5 hourly half-life (h)", 7.05, dig(rc, "hourly_half_life_h", rnd=2), 0.01)
    check("W5 daily-binned half-life (d)", 0.759, dig(rc, "daily_half_life_d", rnd=3), 0.002)
    check("W5 ratio hourly", 18.6, dig(rc, "ratio_hourly", rnd=1), 0.05)
    check("W5 ratio daily", 7.2, dig(rc, "ratio_daily", rnd=1), 0.05)

    rr = r3.get("halflife_ratio") or {}
    check("W7 point ratio", 1.12, dig(rr, "point_ratio", rnd=2), 0.01)
    check("W7 ratio CI contains 1", True, dig(rr, "contains_one"))


def main():
    width = max(len(c[0]) for c in CHECKS) + 2
    print(f"{'check'.ljust(width)}{'claimed':>14}{'computed':>14}   status")
    print("-" * (width + 45))
    tally = {"ok": 0, "FAIL": 0, "SKIP": 0}
    for label, claimed, computed, status in CHECKS:
        tally[status] += 1
        print(f"{label.ljust(width)}{str(claimed):>14}{str(computed):>14}   {status}")
    print("-" * (width + 45))
    n = len(CHECKS)
    line = f"{n} checks, {tally['ok']} agree, {tally['FAIL']} disagree"
    if tally["SKIP"]:
        line += f", {tally['SKIP']} skipped (data channel unavailable)"
    print(line)
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
