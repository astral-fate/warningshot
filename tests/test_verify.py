"""verify.py must survive a missing data channel.

A cold-cache run hit GDELT's rate limit (HTTP 429 through five retries) and the
first version of verify.py crashed with a KeyError, which would leave a
reproducer unable to check any of the other 24 numbers. These tests pin the
behaviour: absent channel -> SKIP, wrong value -> FAIL, and a non-zero exit
only for FAIL.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify.py"
REAL = ROOT / "results" / "multimodal.json"


def _run(results_dir):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(results_dir)],
        capture_output=True, text=True,
    )


@pytest.fixture
def full_results():
    if not REAL.exists():
        pytest.skip("results/multimodal.json not generated yet")
    return json.loads(REAL.read_text(encoding="utf-8"))


def test_full_results_all_agree(full_results, tmp_path):
    (tmp_path / "multimodal.json").write_text(json.dumps(full_results), encoding="utf-8")
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    assert "0 disagree" in r.stdout
    assert "FAIL" not in r.stdout


def test_missing_media_channel_skips_instead_of_crashing(full_results, tmp_path):
    stripped = dict(full_results)
    stripped["channels"] = [c for c in full_results["channels"] if c["channel"] != "media"]
    # spread as it is actually reported when media is absent
    stripped["spread"] = dict(full_results["spread"], slowest="lookups", ratio=18.6)
    stripped["spread"]["pairwise"] = [
        p for p in full_results["spread"]["pairwise"]
        if "media" not in (p["a"], p["b"])
    ]
    (tmp_path / "multimodal.json").write_text(json.dumps(stripped), encoding="utf-8")

    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout          # missing data is not a failure
    assert "SKIP" in r.stdout
    assert "skipped (data channel unavailable)" in r.stdout
    assert "0 disagree" in r.stdout


def test_wrong_value_is_reported_as_fail(full_results, tmp_path):
    tampered = json.loads(json.dumps(full_results))
    for c in tampered["channels"]:
        if c["channel"] == "lookups":
            c["half_life"] = 99.0
    (tmp_path / "multimodal.json").write_text(json.dumps(tampered), encoding="utf-8")

    r = _run(tmp_path)
    assert r.returncode == 1, r.stdout
    assert "FAIL" in r.stdout
    assert "0 disagree" not in r.stdout
