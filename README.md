# Whose Page Did You Count?

**Counting-dependence and a null result in measuring attention to an AI incident.**

How much attention did the July 2026 OpenAI/Hugging Face intrusion get? The answer depends on a
choice nobody states: **which page you count.** This repository re-measures a published reach
comparison, finds that it inverts under a defensible change of counting rule, and puts three
further claims against null distributions that the original account did not have.

| | |
|---|---|
| **Paper** | [`paper/main.pdf`](paper/main.pdf) — 8-page body, references and appendix excluded |
| **Deck** | [`slides.html`](slides.html) — 15-slide pitch deck, open in any browser |
| **Venue** | Apart Research × CeSIA, AI Incident Response Sprint — Track 4 (Communication) |
| **Author** | Fatimah Emad Eldin · Trouvé Works · `Fatimah@trouve.works` |
| **Licence** | MIT (code, data, figures) |
| **Cost to reproduce** | $0 — every source is public and keyless, and responses are cached in-repo |

---

## How it works

<img src="docs/pipeline.svg" alt="warningshot measurement pipeline: three public sources feed a committed on-disk cache, which feeds the three channels; metrics, decay regression and HawkesN produce estimates; five null distributions gate every claim; committed results are checked by verify.py and rendered into the paper." width="100%">

---

## Headline numbers

| Result | Value | Where |
|---|---|---|
| Published comparison, recomputed on developer pages | `8.41×` raw / `4.98×` baseline-days | §4.1 |
| **Same comparison on the page that absorbed the attention** | **`1.46×` raw / `23.06×` baseline-days — direction inverts** | §4.1 |
| Range of that inversion across six June comparators | `3.89×` to `145.54×` normalised | §4.1 |
| AI-risk concept pages vs season-matched placebo null | **`53.8`th percentile — no detectable movement** | §4.2 |
| Smallest response that null could have detected | `2.22×` ordinary drift | §4.2 |
| Victim's own breach disclosure | `1.22×` baseline = **91st percentile — an ordinary day** | §4.3 |
| Attribution to a named lab | **`24.7×` baseline** — ~16× beyond the prior 166-day maximum | §4.3 |
| Largest spike in the whole series | `49.4×` on 3 Sep — **an acquisition, not the incident** | §4.4 |
| Incident's Hacker News thread vs 30 comparable threads | **`33`rd percentile — the half-life measures the platform** | §4.5 |

Every one of these is recomputed from committed artifacts by `python scripts/verify.py`
(67 checks, no network).

---

## The three findings

**1. Reach comparisons are counting-dependent, and the published one inverts.**
The published account reported that June's export-control ban drove roughly seven times more
Wikipedia traffic than the July intrusion. That holds when developer pages are compared against
each other. Counting instead the page that actually absorbed the attention in each event — the
*victim's* page in July, the *developer's* in June — reverses the direction, in raw excess views
and in baseline-normalised units alike. The direction is robust across all six June comparators
tested; the magnitude is not, spanning `3.89×` to `145.54×`, and the paper decomposes why rather
than quoting a single multiple.

**2. The risk vocabulary shows no movement we could detect.**
Four AI-risk concept pages accumulate 17,699 excess views over 30 days. Against thirteen
season-matched placebo windows drawn from the same calendar season of 2024 and 2025, that sits at
the **53.8th percentile** — the middle of ordinary drift. The generic `Machine_learning` control
sits at the 31st. The honest form of the claim is bounded: this design could not have detected a
response below about `2.2×` ordinary drift, so what it shows is that *no large conversion
occurred*, not that none did.

**3. A decay comparison that looks headline-grade does not survive its null.**
Fitting the three channels separately gives half-lives of `7.05` h (Hacker News), `5.45` d
(Wikipedia) and `6.11` d (GDELT) — a `20.8×` spread that reads as evidence of separate clocks.
Two tests remove that reading. The incident's thread sits at the **33rd percentile** of 30
comparable front-page threads, so the number measures Hacker News rather than the event; and a
direct bootstrap of the lookups/media ratio gives `1.120`, 95% CI `(0.788, 1.740)`, which contains
1. What survives is general and weaker: *every* front-page thread turns over in about eight hours,
so a conversion scorecard read at 48 hours is always reading a channel that has already closed.

---

## Quick start

```bash
git clone <this-repo> && cd warningshot
pip install -e ".[dev]"

python -m pytest                      # 92 tests, no network
python scripts/verify.py              # 67 headline numbers vs committed results
```

Both run offline against the cached API responses committed under `cache/`.

### Regenerating the analysis

```bash
python scripts/run.py all             # channel fits, figures, tables -> out/
python scripts/robustness_suite.py    # estimator/autocorrelation/AIC/null robustness
python scripts/nulls_and_controls.py  # thread null, daily-ratio null, resolution control
python scripts/hawkesn_engagement.py  # HawkesN fit: decay vs population exhaustion
python scripts/robustness/r1_autocorr_and_estimator.py   # (r1..r10, one probe each)
```

These re-fetch only on a cache miss. `cache/` is committed precisely so that a clean clone needs
no credentials and hits no rate limit — GDELT allows one request every five seconds, and a cold
run without the cache skips three tests and five verify checks.

### Building the paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Repository layout

```
src/warningshot/
  core/regression.py    exponential / power-law / two-phase fits, AIC with
                        selection penalty, residual + moving-block bootstrap
  core/metrics.py       baseline, excess, baseline-days, half-life, placebo
                        windows, percentile rank
  core/hawkesn.py       HawkesN MLE — separates kernel decay from population
                        exhaustion on the comment thread
  data/sources.py       Wikimedia Pageviews, HN Algolia, GDELT DOC 2.0; all
                        responses SHA-cached to disk on first fetch
  data/modalities.py    the three channels (lookups / engagement / media)
  data/events.py        event anchors, baseline gaps, season-matched spans
  eval/mmdecay.py       cross-channel decay comparison + impulse checks
  eval/nulls.py         every null distribution used in the paper
  eval/study.py         the end-to-end study object
  viz/plots.py          figures

scripts/                runnable entry points (see above)
scripts/robustness/     r1..r10, one adversarial probe per script
cache/                  committed raw API responses — the reproducibility guarantee
results/                committed numbers that verify.py checks
out/                    generated figures and CSV tables
data/corpus/            see "Unrelated component" below
paper/                  main.tex, references.bib, main.pdf
docs/pipeline.svg       system diagram (rendered above)
slides.html             self-contained 15-slide pitch deck
tests/                  92 tests, no network
```

### Unrelated component

`src/warningshot/detect/corpus.py`, `data/corpus/` and `scripts/prepare_corpus.py` are a
labelled attack corpus built for a **different** track and are not used by the paper. They are
retained because the test suite covers them. They read from sibling repositories
(`agent-egress-bench`, `sandbox_escape_bench`) via `$WARNINGSHOT_REPOS`; a clean checkout without
those siblings simply skips the corpus tests. Safe to delete if you want the repository to
contain only what the paper uses.

---

## How the claims are kept honest

- **Nothing is claimed without a distribution.** Five nulls are used — season-matched concept
  null, daily-ratio null, thread null, a matched June event, and a generic control. The
  comparability rule for the thread null (≥500 points, ≥200 comments, ten broad topic queries)
  was fixed before any half-life was inspected.
- **Both unit systems are always reported.** Baseline-days is the only form comparable across
  pages of very different size, but it rewards low-traffic pages, so raw excess is reported
  alongside it and the choice stays visible.
- **Estimators are not assumed.** Log-space OLS is biased with invalid standard errors, so every
  exponential fit is also computed by raw-scale nonlinear least squares. Magnitudes move by up to
  `2.7×`; the ordering does not.
- **Autocorrelation is handled.** Durbin–Watson is `1.17` and `1.35` on the daily channels, so
  intervals are moving-block bootstraps (block 4, n=800, seed 1) rather than i.i.d.
- **Ratios are bootstrapped directly.** Two separately estimated intervals overlapping is not a
  test that the parameters are indistinguishable.
- **Negative results are reported.** Finding 3 is a claim the analysis withdraws under its own
  null, and it is in the paper as a result rather than omitted.

## Known limits

`n = 1` event with one matched control; pageviews proxy lookups, not awareness; the concept null
collapses to a handful of independent samples once window overlap is accounted for; the
absorbing-page rule is retrospective and cannot be applied prospectively; the 27 August impulse
has three candidate drivers and none is separable. Full list, with the dual-use analysis, in §6
of the paper and Appendix A.

**Reproducibility gap, stated plainly:** `results/power_and_comparators.json` holds the MDE
(`2.22×`), the six-comparator table and the control percentile. The functions that produce them
live in `eval/nulls.py`, but no script currently regenerates that file end-to-end and
`verify.py` does not cover it. Those numbers are therefore committed artifacts rather than
recomputed ones. The other 67 are recomputed.

## Citation

```bibtex
@techreport{eldin2026whosepage,
  author      = {Fatimah Emad Eldin},
  title       = {Whose Page Did You Count? Counting-Dependence and a Null Result
                 in Measuring Attention to an {AI} Incident},
  institution = {Apart Research and CeSIA, AI Incident Response Sprint},
  year        = {2026},
  month       = {9}
}
```
