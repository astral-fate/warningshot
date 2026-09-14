"""Fit HawkesN to the Hacker News comment thread.

The engagement channel's half-life is not directly commensurable with the
pageview channels, because a thread decays partly by exhausting the pool of
people who will ever comment on it. HawkesN separates the two: N is the
fitted population, n/N the share consumed, and theta the kernel decay that is
NOT exhaustion.

    python scripts/hawkesn_engagement.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warningshot import paths                      # noqa: E402
from warningshot.core import hawkesn as hn         # noqa: E402
from warningshot.core import regression as rg      # noqa: E402
from warningshot.data import modalities as mo      # noqa: E402
from warningshot.data import sources as src        # noqa: E402

STORY = "48997548"


def main():
    ts = src.hn_comment_times(STORY)
    t = (np.asarray(sorted(ts), dtype=float) - min(ts)) / 3600.0

    print(f"\n[hawkesn] HN thread {STORY}")
    print(f"  n={t.size} comments, span {t[-1]:.1f} h, "
          f"{100 * (t < 24).mean():.0f}% in the first 24 h")

    f = hn.fit(t, n_starts=14, seed=11)
    if f is None:
        print("  fit failed")
        return 1

    # The paper's simple exponential on hourly counts, for comparison.
    ch = mo.engagement(STORY)
    h = np.array(ch["hourly"], dtype=float)
    ht = np.arange(1.0, h.size + 1.0)
    simple = rg.fit_exponential(ht, h)

    print(f"\n  simple exponential on hourly counts : t_half = "
          f"{simple['half_life']:.2f} h   (the paper's headline)")
    print(f"  HawkesN kernel                      : t_half = "
          f"{f['kernel_half_life']:.2f} h")
    print(f"  ratio                               : "
          f"{f['kernel_half_life'] / simple['half_life']:.2f}x slower")
    print(f"\n  K (virality)      {f['K']:.3f}")
    print(f"  theta (per hour)  {f['theta']:.4f}")
    print(f"  N (population)    {f['N']:.0f}   -> N/n = {f['N'] / f['n_events']:.2f}")
    print(f"  exhaustion (n/N)  {f['exhaustion']:.2f}")
    print(f"  at N bound        {f['at_N_bound']}")

    verdict = ("pool NOT near exhaustion; decay is kernel-dominated"
               if f["exhaustion"] < 0.8 else
               "pool near exhaustion; observed decay is largely mechanical")
    print(f"\n  verdict: {verdict}")
    print("  caveat: validation shows this estimator is biased TOWARD")
    print("  exhaustion, so a 'not exhausted' verdict is the safe direction.")

    out = {"story": STORY, "hawkesn": f,
           "simple_exponential_half_life_h": simple["half_life"],
           "kernel_vs_simple_ratio": f["kernel_half_life"] / simple["half_life"],
           "verdict": verdict}
    paths.ensure(paths.RESULTS)
    with open(paths.RESULTS / "hawkesn_engagement.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n  wrote {(paths.RESULTS / 'hawkesn_engagement.json').relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
