"""The README's headline figures must match the artifacts they claim to report.

The README is normally the one document a build does not cover, which is exactly
why it drifts: a number is quoted, the analysis is re-run, and the front page
quietly starts lying. Each case below recomputes a figure from `results/` and
asserts the README still states it, formatted the way the README writes it.

A failure here means one of two things, and both are worth stopping for:

  * the analysis moved and the README was not updated -- fix the README;
  * the README was edited and the number was mistyped -- fix the README.

Cases skip rather than fail when the underlying result is absent, so a cold
checkout without the GDELT channel does not report a false drift.
"""

import json

import pytest

from warningshot import paths

README = paths.ROOT / "README.md"


def _load(name):
    path = paths.RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dig(obj, *keys):
    cur = obj
    for k in keys:
        if cur is None:
            return None
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def artifacts():
    return {
        "nulls": _load("nulls.json"),
        "robustness": _load("robustness.json"),
        "power": _load("power_and_comparators.json"),
        "multimodal": _load("multimodal.json"),
    }


def _x2(v):
    """Format as the README writes a multiple: two decimals and a times sign."""
    return "`{:.2f}×`".format(v)


# (case id, artifact key, key path, formatter)
CASES = [
    # 1 -- the inversion
    ("published 7x in raw excess", "nulls",
     ("inversion_units", "june_dev_vs_july_dev_raw"), _x2),
    ("published 7x in baseline-days", "nulls",
     ("inversion_units", "june_dev_vs_july_dev_baseline_days"), _x2),
    ("inversion in raw excess", "nulls",
     ("inversion_units", "victim_vs_june_dev_raw"), _x2),
    ("inversion in baseline-days", "nulls",
     ("inversion_units", "victim_vs_june_dev_baseline_days"), _x2),
    ("inversion CI low", "power",
     ("inversion_ci", "ci_lo"), lambda v: "({:.2f},".format(v)),
    ("inversion CI high", "power",
     ("inversion_ci", "ci_hi"), lambda v: "{:.2f})".format(v)),
    ("comparator range, normalised minimum", "power",
     ("comparator_decomposition", "normalised_min"), _x2),
    ("comparator range, normalised maximum", "power",
     ("comparator_decomposition", "normalised_max"), _x2),
    ("comparator range, raw maximum", "power",
     ("comparator_decomposition", "raw_max"), _x2),

    # 2 -- the concept null
    ("observed concept excess", "robustness",
     ("season_matched_null", "observed"), lambda v: "`{:,.0f}`".format(v)),
    ("season-matched percentile", "robustness",
     ("season_matched_null", "percentile"), lambda v: "`{:.1f}`th".format(v)),
    ("minimum detectable effect", "power",
     ("mde", "mde_at_p90"), _x2),
    ("observed over null median", "power",
     ("mde", "observed_over_median"), _x2),
    ("generic control percentile", "power",
     ("control_vs_null", "percentile"), lambda v: "`{:.0f}`st".format(v)),

    # 3 -- the decay result that fails its null
    ("thread-null percentile", "nulls",
     ("hn_thread_null", "percentile"), lambda v: "`{:.0f}`rd".format(v)),
    ("thread-null median half-life", "nulls",
     ("hn_thread_null", "null_median"), lambda v: "{:.2f}".format(v)),
    ("incident thread half-life", "nulls",
     ("hn_thread_null", "observed_half_life_h"), lambda v: "`{:.2f}` h".format(v)),
    ("lookups half-life", "multimodal",
     ("spread", "half_lives_days", "lookups"), lambda v: "`{:.2f}` d".format(v)),
    ("media half-life", "multimodal",
     ("spread", "half_lives_days", "media"), lambda v: "`{:.2f}` d".format(v)),
    ("cross-channel spread", "multimodal",
     ("spread", "ratio"), lambda v: "`{:.1f}×`".format(v)),
    ("halflife ratio point estimate", "nulls",
     ("halflife_ratio", "point_ratio"), lambda v: "`{:.3f}`".format(v)),
]


@pytest.mark.parametrize("case_id,source,path,fmt",
                         CASES, ids=[c[0] for c in CASES])
def test_readme_figure_matches_artifact(readme, artifacts, case_id, source, path, fmt):
    value = _dig(artifacts.get(source), *path)
    if value is None:
        pytest.skip("%s absent from results/ -- channel unavailable" % source)
    expected = fmt(value)
    assert expected in readme, (
        "README no longer states the computed value for %r.\n"
        "  computed from results/%s.json%s = %r\n"
        "  expected the README to contain: %s\n"
        "Update the README, or explain why the artifact moved."
        % (case_id, source, list(path), value, expected)
    )


def test_readme_check_count_matches_verify():
    """The README quotes how many checks verify.py runs; keep the two in step."""
    text = README.read_text(encoding="utf-8")
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, str(paths.ROOT / "scripts" / "verify.py")],
        capture_output=True, text=True, cwd=str(paths.ROOT),
    ).stdout
    tail = [ln for ln in out.splitlines() if "checks," in ln]
    if not tail:
        pytest.skip("verify.py produced no summary line")
    n = int(tail[-1].split()[0])
    assert "%d checks" % n in text, (
        "README says a different number of verify.py checks than verify.py runs "
        "(verify.py reports %d)." % n
    )


def test_readme_links_resolve():
    """Every relative path the README points at must exist."""
    import re
    text = README.read_text(encoding="utf-8")
    targets = set(re.findall(r"\]\(([^)#]+)\)", text))
    targets |= set(re.findall(r'<img src="([^"]+)"', text))
    missing = []
    for t in sorted(targets):
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        if not (paths.ROOT / t).exists():
            missing.append(t)
    assert not missing, "README points at paths that do not exist: %s" % missing


# --- the diagram is an asset that states numbers, so it drifts like prose ----
# docs/pipeline.svg is embedded in both the README and the deck. It carried
# "67 checks / 92 tests" for two commits after those counts changed, which is
# the classic stale-figure failure: nothing rebuilds a hand-drawn asset, so
# nothing notices. These pin the claims it makes.

PIPELINE_SVG = paths.ROOT / "docs" / "pipeline.svg"


def test_pipeline_diagram_states_the_real_check_count():
    import subprocess
    import sys
    svg = PIPELINE_SVG.read_text(encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(paths.ROOT / "scripts" / "verify.py")],
        capture_output=True, text=True, cwd=str(paths.ROOT),
    ).stdout
    lines = [ln for ln in out.splitlines() if "checks," in ln]
    if not lines:
        pytest.skip("verify.py produced no summary line")
    n = int(lines[-1].split()[0])
    assert "%d checks" % n in svg, (
        "docs/pipeline.svg claims a different number of verify.py checks than "
        "verify.py runs (%d). The diagram is embedded in the README and the "
        "deck; update it." % n
    )


def test_pipeline_diagram_agrees_with_the_readme():
    """The two documents share one diagram, so they must agree about it."""
    svg = PIPELINE_SVG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    import re
    m = re.search(r"(\d+) tests, no network", svg)
    assert m, "the diagram no longer states a test count"
    assert "%s tests, no network" % m.group(1) in readme, (
        "docs/pipeline.svg says %s tests; the README says something else"
        % m.group(1)
    )


def test_deck_references_the_same_diagram():
    """The deck must embed the diagram, not a hand-exported copy of it."""
    deck = (paths.ROOT / "index.html").read_text(encoding="utf-8")
    assert 'src="docs/pipeline.svg"' in deck, (
        "the deck no longer points at docs/pipeline.svg; a duplicated figure "
        "will drift from the README's"
    )


# --- the deck states counts too, and it drifted twice ------------------------
# index.html claimed "92 tests" and then "118 tests" after the suite had grown.
# Prose counts are cheap to write and invisible when they rot, so derive the
# real number and compare. --collect-only does not execute anything, so this is
# safe to run from inside a pytest session.

DECK = paths.ROOT / "index.html"


def _collected_test_count():
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, cwd=str(paths.ROOT),
    ).stdout
    for line in reversed(out.splitlines()):
        m = re.match(r"(\d+) tests? collected", line.strip())
        if m:
            return int(m.group(1))
    return None


import re  # noqa: E402  (used by the helper above and the tests below)


def test_deck_states_the_real_test_count():
    n = _collected_test_count()
    if n is None:
        pytest.skip("could not determine the collected test count")
    deck = DECK.read_text(encoding="utf-8")
    assert "%d tests" % n in deck, (
        "index.html states a different test count than pytest collects (%d)" % n
    )


def test_readme_states_the_real_test_count():
    n = _collected_test_count()
    if n is None:
        pytest.skip("could not determine the collected test count")
    text = README.read_text(encoding="utf-8")
    assert "%d tests" % n in text, (
        "README states a different test count than pytest collects (%d)" % n
    )


def test_deck_and_readme_agree_on_the_check_count():
    deck = DECK.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    m = re.search(r"(\d+) checks", readme)
    assert m, "the README no longer states a check count"
    assert "%s/%s" % (m.group(1), m.group(1)) in deck or "%s checks" % m.group(1) in deck, (
        "the deck and the README disagree about how many checks verify.py runs "
        "(README says %s)" % m.group(1)
    )
