"""Tests for external-corpus ingest.

The most valuable test here is `test_matches_upstream_stats`: it parses
agent-egress-bench's own STATS.md and asserts our loader reproduces its
published counts. A first version of the loader walked `*.json` only, silently
dropped the six directory-style `mcp_drift` cases along with the corpus's only
`warn` label, and reported 250/17 instead of 256/18 — a wrong denominator under
every rate that would have been computed downstream. This test is what caught
the shape of that bug and what keeps it caught.

These tests read the sibling `repos/` clones and skip cleanly when absent, so
the suite still passes for someone who only cloned this project.
"""

import json
import re

import pytest

from warningshot import paths
from warningshot.detect import corpus as C

pytestmark = pytest.mark.skipif(
    not (paths.EGRESS_BENCH / "cases").is_dir(),
    reason="agent-egress-bench clone not present",
)


@pytest.fixture(scope="module")
def cases():
    return C.load_egress_bench()


def _upstream_stats():
    """Counts declared by the corpus itself, parsed from its STATS.md."""
    p = paths.EGRESS_BENCH / "cases" / "STATS.md"
    text = p.read_text(encoding="utf-8")
    out = {}
    for key in ("cases_total", "categories", "block", "allow", "warn"):
        m = re.search(rf"^{key}:\s*(\d+)", text, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


def test_matches_upstream_stats(cases):
    """Our count must equal the corpus's own published count."""
    up = _upstream_stats()
    rep = C.coverage_report(cases)
    assert rep["n_cases"] == up["cases_total"]
    assert rep["n_families"] == up["categories"]
    assert rep["verdicts"]["block"] == up["block"]
    assert rep["verdicts"]["allow"] == up["allow"]
    assert rep["verdicts"]["warn"] == up["warn"]


def test_every_case_has_a_valid_ground_truth_label(cases):
    assert cases
    for c in cases:
        assert c.expected_verdict in C.VERDICTS, c.case_id


def test_case_ids_are_unique(cases):
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_is_attack_tracks_the_block_label(cases):
    for c in cases:
        assert c.is_attack == (c.expected_verdict == "block"), c.case_id


def test_temporal_cases_are_the_mcp_drift_family(cases):
    temporal = [c for c in cases if c.is_temporal]
    assert len(temporal) == 6
    assert {c.family for c in temporal} == {"mcp_drift"}


def test_temporal_cases_carry_two_distinct_states(cases):
    """A before/after pair whose states are identical would be unsolvable, and
    concatenating them would hide the mutation the detector must find."""
    for c in (c for c in cases if c.is_temporal):
        assert c.before_text, c.case_id
        assert c.after_text, c.case_id
        assert c.before_text != c.after_text, c.case_id


def test_temporal_family_has_both_block_and_no_block_cases(cases):
    """The discrimination that matters: malicious mutation vs benign change.
    Without both, the family measures nothing."""
    verdicts = {c.expected_verdict for c in cases if c.is_temporal}
    assert "block" in verdicts
    assert verdicts & {"warn", "allow"}


def test_trap_kind_is_set_exactly_for_benign_cases(cases):
    for c in cases:
        if c.is_attack:
            assert c.fp_trap_kind == "", c.case_id
        else:
            assert c.fp_trap_kind in ("dedicated", "in_family_control"), c.case_id


def test_both_kinds_of_benign_case_exist_and_differ_in_count(cases):
    """If these were equal in count to the benign total, the split would be
    vacuous. The dedicated family is 22; the in-family controls outnumber it."""
    rep = C.coverage_report(cases)
    assert rep["n_fp_dedicated"] == 22
    assert rep["n_fp_in_family"] > rep["n_fp_dedicated"]
    assert rep["n_fp_dedicated"] + rep["n_fp_in_family"] == rep["n_benign"]


def test_high_fp_risk_benign_cases_exist(cases):
    """The upstream difficulty label is what makes a specificity result
    interpretable: passing easy benign cases proves little."""
    rep = C.coverage_report(cases)
    assert rep["n_benign_high_fp_risk"] >= 15


def test_payload_text_is_populated_for_text_detectors(cases):
    empty = [c.case_id for c in cases if not c.payload_text.strip()]
    assert not empty, f"cases with no flattened payload text: {empty[:5]}"


def test_confusable_families_present_for_specificity_testing(cases):
    rep = C.coverage_report(cases)
    # At least most of the named confusable families must exist, or the
    # specificity claim has nothing to be specific against.
    assert len(rep["confusable_families_present"]) >= 4


def test_coverage_report_states_what_is_not_measurable(cases):
    rep = C.coverage_report(cases)
    assert rep["not_measurable"], "must not ship a coverage report with no limits"
    assert any("transport" in s or "covert" in s for s in rep["not_measurable"])


def test_cases_serialise_to_json(cases):
    blob = json.dumps([c.to_json() for c in cases[:20]])
    assert json.loads(blob)


# --- sandbox_escape_bench: the negative result ----------------------------

@pytest.mark.skipif(not (paths.ESCAPE_BENCH / "scenarios").is_dir(),
                    reason="sandbox_escape_bench clone not present")
def test_escape_scenarios_load():
    sc = C.load_escape_scenarios()
    assert len(sc) >= 15
    assert any(s["has_exploit_code"] for s in sc)


@pytest.mark.skipif(not (paths.ESCAPE_BENCH / "scenarios").is_dir(),
                    reason="sandbox_escape_bench clone not present")
def test_escape_transcripts_are_reported_absent_not_assumed_present():
    """The index lists 162 runs whose `log` paths point outside the repo. Any
    idea depending on those transcripts must die here, not after a day's work.
    """
    st = C.escape_transcript_status()
    assert st["index_present"] is True
    assert st["n_rows"] > 100
    assert st["transcripts_present"] is False
    assert st["n_resolvable_locally"] == 0
