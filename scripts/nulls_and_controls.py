"""Null distributions and controls for the single-day and cross-channel claims.

Each block states a claim against a distribution rather than against nothing:
the Hacker News thread null, the disclosure-day ratio null, the counting
inversion in both unit systems, the hourly-vs-daily resolution control, and a
direct bootstrap of the half-life ratio.

Writes results/nulls.json, which scripts/verify.py checks.

    python scripts/nulls_and_controls.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warningshot import paths                      # noqa: E402
from warningshot.eval import nulls                 # noqa: E402


def main():
    print("\n[nulls] null distributions and controls")

    print("\n  W-2  Hacker News thread null")
    hn = nulls.hn_thread_null()
    if hn:
        print(f"    pool={hn['pool_size']} threads, fitted={hn['n_fitted']}")
        print(f"    null  min={hn['null_min']:.2f}  p25={hn['null_p25']:.2f}  "
              f"median={hn['null_median']:.2f}  p75={hn['null_p75']:.2f}  "
              f"max={hn['null_max']:.2f}  (hours)")
        print(f"    incident thread = {hn['observed_half_life_h']:.2f} h  "
              f"-> {hn['percentile']:.0f}th percentile")
        print("    verdict: " + ("ORDINARY - the half-life is a property of the "
                                 "platform" if 20 <= hn["percentile"] <= 80
                                 else "atypical against comparable threads"))

    print("\n  W-3  Disclosure-day ratio against ordinary daily variation")
    dr = nulls.daily_ratio_distribution("Hugging_Face", "20260715", 1.22)
    if dr:
        print(f"    window {dr['window'][0]}..{dr['window'][1]}  n={dr['n_days']} days")
        print(f"    median={dr['median']:.2f}  p90={dr['p90']:.2f}  max={dr['max']:.2f}")
        print(f"    observed 1.22x -> {dr['percentile']:.0f}th percentile; "
              f"{dr['days_at_or_above']} of {dr['n_days']} days reach it")

    print("\n  W-4  Counting inversion, both unit systems")
    inv = nulls.inversion_units()
    if inv:
        for k in ("victim", "developer_june", "developer_july"):
            r = inv[k]
            print(f"    {k:16s} {r['page']:14s} base={r['baseline']:8.0f}/d  "
                  f"excess={r['excess']:8.0f}  baseline-days={r['baseline_days']:6.2f}")
        print(f"    victim vs June developer:  raw {inv['victim_vs_june_dev_raw']:.2f}x"
              f"   baseline-days {inv['victim_vs_june_dev_baseline_days']:.2f}x")
        print(f"    published '7x' recomputed:  raw {inv['june_dev_vs_july_dev_raw']:.2f}x"
              f"   baseline-days {inv['june_dev_vs_july_dev_baseline_days']:.2f}x")

    print("\n  W-5  Resolution control")
    rc = nulls.resolution_control()
    if rc:
        print(f"    hourly  {rc['hourly_bins']:3d} bins  t_half={rc['hourly_half_life_h']:6.2f} h "
              f"= {rc['hourly_half_life_d']:.3f} d   R2={rc['hourly_r2']:.3f}")
        print(f"    daily   {rc['daily_bins']:3d} bins  t_half={rc['daily_half_life_d']:6.3f} d"
              f"                R2={rc['daily_r2']:.3f}")
        print(f"    ratio to lookups: hourly {rc['ratio_hourly']:.1f}x   "
              f"daily {rc['ratio_daily']:.1f}x")

    print("\n  W-7  Ratio bootstrap (lookups vs media)")
    rr = nulls.halflife_ratio_ci()
    if rr:
        print(f"    point ratio={rr['point_ratio']:.3f}  "
              f"95% CI=({rr['ci_lo']:.3f}, {rr['ci_hi']:.3f})  "
              f"contains 1: {rr['contains_one']}")

    paths.ensure(paths.RESULTS)
    out = {"hn_thread_null": hn, "disclosure_ratio": dr,
           "inversion_units": inv, "resolution_control": rc,
           "halflife_ratio": rr}
    with open(paths.RESULTS / "nulls.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n    wrote {(paths.RESULTS / 'nulls.json').relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
