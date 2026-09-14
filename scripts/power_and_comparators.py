"""Power of the concept null, and the comparator sweep behind the inversion.

Six blocks, all of which the paper quotes and none of which the other result
files carry:

  P1  minimum detectable effect for the season-matched concept null -- the
      multiple of ordinary drift a response would have had to reach to clear
      the null's 90th and 95th percentiles (paper section 4.2)
  P2  the absorbing-page rule applied to the June side as well, across six
      candidate pages, which is what makes the inversion's direction robust
      and its magnitude not (section 4.1)
  P3  the June anchor in UTC days, 12 vs 13 June, and what the choice does
      to the headline multiple (section 4.1)
  P4  the decomposition of each comparator's ratio into a raw-excess ratio
      times an inverse-baseline ratio, which is why the direction is robust
      and the magnitude is not (section 4.1, appendix A)
  P5  a bootstrap interval on the baseline-days inversion ratio, resampling
      the baseline window that dominates its uncertainty (section 4.1)
  P6  the generic control put through the identical season-matched null, so
      the control and the concept pages are held to one standard (section 4.2)

Writes results/power_and_comparators.json, which scripts/verify.py checks.
Runs offline against the committed cache.

    python scripts/power_and_comparators.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warningshot import paths                        # noqa: E402
from warningshot.data import events as ev            # noqa: E402
from warningshot.eval import nulls                   # noqa: E402
from warningshot.eval import study                   # noqa: E402


def concept_null_and_observed():
    """The season-matched null values and the observed concept excess.

    Both are recomputed here rather than read out of robustness.json, so this
    script does not depend on the order the result files happen to be built in.
    """
    windows = study.placebo_null_season_matched()
    if not windows:
        return None, None
    values = sorted(r["concept_excess"] for r in windows)
    rows = study.two_clock(ev.JULY_INCIDENT)
    observed = (sum(r["excess_30d"] for r in rows if r["role"] == "concept")
                if rows else None)
    return values, observed


def main():
    print("\n[power] minimum detectable effect and comparator sweep")

    print("\n  P1  concept null -- minimum detectable effect")
    values, observed = concept_null_and_observed()
    mde = nulls.concept_null_mde(values, observed) if values and observed else None
    if mde:
        print(f"    n={mde['n_windows']} windows   median={mde['null_median']:,.0f}"
              f"   p90={mde['null_p90']:,.0f}   p95={mde['null_p95']:,.0f}")
        print(f"    observed={mde['observed']:,.0f} "
              f"= {mde['observed_over_median']:.2f}x the null median")
        print(f"    a response had to reach {mde['mde_at_p90']:.2f}x drift to clear p90 "
              f"({mde['mde_at_p95']:.2f}x for p95)")
    else:
        print("    SKIP -- concept null unavailable (cold cache)")

    print("\n  P2  June-side comparator sweep (absorbing-page rule, applied symmetrically)")
    jc = nulls.june_candidate_pages()
    if jc:
        print(f"    {'page':34s}{'base/day':>10}{'excess':>10}{'base-days':>11}")
        for r in jc["candidates"]:
            print(f"    {r['page']:34s}{r['baseline']:10.0f}{r['excess']:10.0f}"
                  f"{r['baseline_days']:11.2f}")
        print(f"    absorbing June page = {jc['absorbing_page']} "
              f"({jc['absorbing_baseline_days']:.2f} baseline-days)")
    else:
        print("    SKIP -- June candidate pages unavailable (cold cache)")

    print("\n  P3  June anchor sensitivity (12 vs 13 June, UTC days)")
    ja = nulls.june_anchor_sensitivity()
    if ja:
        for a in ja["anchors"]:
            print(f"    anchor {a['anchor']}   base={a['baseline']:9.0f}/d"
                  f"   excess={a['excess']:8.0f}"
                  f"   base-days={a['baseline_days']:6.2f}"
                  f"   ratio={a['ratio_to_july_victim']:7.2f}x")
    else:
        print("    SKIP -- anchor sensitivity unavailable (cold cache)")

    print("\n  P4  ratio decomposition: baseline-days = raw ratio x baseline ratio")
    dc = nulls.comparator_decomposition()
    if dc:
        print(f"    {'comparator':36s}{'base-days':>11}{'= raw':>10}{'x base':>10}")
        for r in dc["rows"]:
            print(f"    {r['comparator']:36s}{r['baseline_days_ratio']:10.2f}x"
                  f"{r['raw_ratio']:9.2f}x{r['baseline_ratio']:9.2f}x")
        print(f"    normalised {dc['normalised_min']:.2f}x - {dc['normalised_max']:.2f}x"
              f"    raw {dc['raw_min']:.2f}x - {dc['raw_max']:.2f}x")
    else:
        print("    SKIP -- decomposition unavailable (cold cache)")

    print("\n  P5  bootstrap interval on the inversion ratio")
    ci = nulls.inversion_ci()
    if ci:
        print(f"    point={ci['point_ratio']:.2f}x   "
              f"95% CI=({ci['ci_lo']:.2f}, {ci['ci_hi']:.2f})   "
              f"n={ci['n_resamples']} resamples, seed {ci['seed']}")
        print(f"    excludes 1: {ci['excludes_one']}")
    else:
        print("    SKIP -- inversion CI unavailable (cold cache)")

    print("\n  P6  generic control against the identical season-matched null")
    cv = nulls.control_vs_null()
    if cv:
        print(f"    page={cv['page']}   n={cv['n_windows']} windows   "
              f"null median={cv['null_median']:,.0f}   max={cv['null_max']:,.0f}")
        print(f"    observed={cv['observed_excess']:,.0f}  "
              f"-> {cv['percentile']:.0f}th percentile")
    else:
        print("    SKIP -- control null unavailable (cold cache)")

    paths.ensure(paths.RESULTS)
    out = {"mde": mde, "june_candidates": jc, "june_anchor": ja,
           "comparator_decomposition": dc,
           "inversion_ci": ci, "control_vs_null": cv}
    path = paths.RESULTS / "power_and_comparators.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n    wrote {path.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
