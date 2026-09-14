"""Normalise external attack corpora into one labelled record type.

Purpose: measure whether a detector built for one attack class stays quiet on
other attack classes and on benign traffic. That is a *specificity* question,
and it is the one the available data can actually answer.

What ships, verified by reading the repositories rather than their summaries:

  agent-egress-bench      256 case files, 18 attack families, every case
                          carrying `expected_verdict` in {block, allow, warn},
                          including 22 cases in a `false_positive` family built
                          specifically to trap over-blocking. Complete and
                          local. PRIMARY SOURCE.

  sandbox_escape_bench    19 container-breakout scenario definitions with real
                          exploit code, a 162-row index of (model x task) eval
                          runs, and aggregate behaviour-scan tables over 9
                          models. THE TRANSCRIPTS DO NOT SHIP: the `log` column
                          of included_logs.csv holds absolute paths under
                          /Users/jeromewynne/..., so per-sample data is not
                          recoverable from the clone. Usable for its scenario
                          taxonomy and aggregate priors, not for per-sample
                          detection scoring.

On the covert-channel scope limit, and a correction to it. The
agent-egress-bench README (line 247) says:

    "Covert channels. Timing, header ordering, HTTP/2 framing, and
     steganography are out of scope by design."

That exclusion is narrower than it first reads: it covers *transport-level*
channels. The `mcp_drift` family contradicts the broad reading, and its own
case files say so. From `mcp-drift-rugpull-desc-002/case.yaml`, verbatim:

    "Tool metadata is a covert channel because clients render descriptions to
     the LLM context but rarely re-display them after the initial approval.
     The attacker's window opens after the operator stops paying attention."

So the corpus does contain a metadata covert channel, in six temporal cases
that pair a baseline snapshot with a later mutation and label the mutation
`block` while labelling a benign tool *addition* `warn`. That block/warn
distinction — state changed maliciously versus state changed legitimately — is
the same discrimination a channel detector has to make after a substrate is
wiped and repopulated, which makes this the closest shipping analogue to the
Track 1 scenario. Six cases is small; first place in a prior sprint used five
hand-crafted traces.

What remains genuinely unmeasurable here: transport-level covert channels
(timing, framing, steganography), and detection over multi-step agent
episodes, since these cases are wire payloads rather than trajectories.
`coverage_report` emits both lists so a downstream reader cannot mistake a
specificity result for a sensitivity one.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from warningshot import paths

VERDICTS = ("block", "allow", "warn")

# Families that a covert-channel detector could plausibly confuse with its own
# target, because both involve hiding information inside a permitted operation.
# Firing here is the informative failure, so they are named up front rather
# than chosen after seeing results.
CONFUSABLE_FAMILIES = (
    "encoding_evasion",
    "hostname_exfiltration",
    "shell_obfuscation",
    "websocket_dlp",
    "request_body",
)


@dataclass
class Case:
    """One labelled attack or benign case, source-agnostic."""

    case_id: str
    source: str
    family: str
    title: str
    expected_verdict: str          # ground truth: block / allow / warn
    is_attack: bool                # derived: expected_verdict == "block"
    # `safe_example: true` is set on every benign case upstream, so a single
    # "is this an FP trap" flag would be exactly `not is_attack` wearing an
    # informative-sounding name. The benign set actually has two distinct
    # kinds, and they test different things:
    #   dedicated         the 22-case `false_positive` family, purpose-built
    #                     over-blocking traps
    #   in_family_control benign cases sitting inside an attack family - "looks
    #                     like the attack, is fine" - which is the harder test
    fp_trap_kind: str = ""         # dedicated | in_family_control | "" if attack
    false_positive_risk: str = ""  # upstream difficulty label: low/medium/high
    input_type: str = ""
    transport: str = ""
    severity: str = ""
    capability_tags: list = field(default_factory=list)
    why_expected: str = ""
    payload_text: str = ""         # flattened payload, for text-based detectors
    payload: Any = None            # original structure, preserved verbatim
    path: str = ""
    # Temporal cases only: a baseline snapshot and a later one. A detector that
    # sees only `payload_text` cannot solve these, which is the point of
    # keeping the two states separate rather than concatenating them.
    is_temporal: bool = False
    before_text: str = ""
    after_text: str = ""

    def to_json(self):
        return asdict(self)


def _trap_kind(verdict, family):
    """Which kind of benign case this is, or "" for an attack."""
    if verdict == "block":
        return ""
    return "dedicated" if family == "false_positive" else "in_family_control"


def _flatten(obj, out):
    """Collect every string in a nested payload, so a text detector has one
    field to scan without each caller re-implementing the walk."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif obj is not None:
        out.append(str(obj))
    return out


def _load_temporal_case(d, root):
    """A directory-style case: case.yaml plus before/after snapshots.

    These exist only in `mcp_drift`. Loading them is what closes the gap
    between this loader and the corpus's own STATS.md (256 cases, 18
    families, one `warn`); a flat *.json walk silently drops all six and the
    only warn-labelled case in the corpus, which would put a wrong
    denominator under every rate reported downstream.
    """
    meta_path = d / "case.yaml"
    if not meta_path.exists():
        return None
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or "expected_verdict" not in meta:
        return None

    files = meta.get("files") or {}
    before = after = None
    for key, target in (("before", "before"), ("after", "after")):
        name = files.get(key) or f"{target}.json"
        p = d / name
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if key == "before":
                before = payload
            else:
                after = payload

    verdict = meta["expected_verdict"]
    family = meta.get("category", "unknown")
    return Case(
        case_id=meta.get("id") or d.name,
        source="agent-egress-bench",
        family=family,
        title=(meta.get("title") or "").strip(),
        expected_verdict=verdict,
        is_attack=(verdict == "block"),
        # A `warn` case here is the explicit over-blocking guardrail: benign
        # drift that a detector must notice without blocking.
        fp_trap_kind=_trap_kind(verdict, family),
        false_positive_risk=meta.get("false_positive_risk", ""),
        input_type=meta.get("input_type", ""),
        transport=meta.get("transport", ""),
        severity=meta.get("severity", ""),
        capability_tags=list(meta.get("capability_tags") or []),
        why_expected=" ".join((meta.get("why_expected") or "").split()),
        payload_text=" ".join(_flatten(after if after is not None else before, [])),
        payload={"before": before, "after": after},
        path=str(d.relative_to(root)),
        is_temporal=True,
        before_text=" ".join(_flatten(before, [])),
        after_text=" ".join(_flatten(after, [])),
    )


def load_egress_bench(root=None):
    """Every agent-egress-bench case as a Case.

    Handles both shapes the corpus uses: flat single-payload `*.json` files,
    and directory cases carrying `case.yaml` with before/after snapshots.
    """
    root = paths.EGRESS_BENCH if root is None else root
    cases_dir = root / "cases"
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"agent-egress-bench cases not found at {cases_dir}")

    out = []
    for d in sorted(p for p in cases_dir.rglob("*") if p.is_dir()):
        c = _load_temporal_case(d, root)
        if c:
            out.append(c)

    for path in sorted(cases_dir.rglob("*.json")):
        # before/after snapshots belong to their temporal case, not standalone
        if (path.parent / "case.yaml").exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or "expected_verdict" not in raw:
            continue  # capability registry, contracts, manifests
        verdict = raw["expected_verdict"]
        family = raw.get("category", "unknown")
        out.append(Case(
            case_id=raw.get("id") or path.stem,
            source="agent-egress-bench",
            family=family,
            title=raw.get("title", ""),
            expected_verdict=verdict,
            is_attack=(verdict == "block"),
            fp_trap_kind=_trap_kind(verdict, family),
            false_positive_risk=raw.get("false_positive_risk", ""),
            input_type=raw.get("input_type", ""),
            transport=raw.get("transport", ""),
            severity=raw.get("severity", ""),
            capability_tags=list(raw.get("capability_tags") or []),
            why_expected=raw.get("why_expected", ""),
            payload_text=" ".join(_flatten(raw.get("payload"), [])),
            payload=raw.get("payload"),
            path=str(path.relative_to(root)),
        ))
    return out


def load_escape_scenarios(root=None):
    """sandbox_escape_bench scenario taxonomy.

    Returns scenario names and which support files each ships. Deliberately
    does NOT return transcripts, because none are present in the clone.
    """
    root = paths.ESCAPE_BENCH if root is None else root
    sdir = root / "scenarios"
    if not sdir.is_dir():
        return []
    out = []
    for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
        out.append({
            "scenario": d.name,
            "source": "sandbox_escape_bench",
            "files": sorted(f.name for f in d.iterdir() if f.is_file()),
            "has_exploit_code": (d / "exploit.c").exists(),
        })
    return out


def escape_transcript_status(root=None):
    """Whether sandbox_escape_bench transcripts are actually retrievable.

    Reported rather than assumed: an idea that depends on absent data should
    die at corpus-prep time, not after a day of work. This project has already
    lost one candidate to a corpus that indexed data it did not ship.
    """
    root = paths.ESCAPE_BENCH if root is None else root
    idx = (root / "transcript-analysis" / "scripts" / "misconfiguration-checks"
           / "included_logs.csv")
    if not idx.exists():
        return {"index_present": False, "transcripts_present": False, "n_rows": 0}

    import csv
    rows = list(csv.DictReader(idx.open(encoding="utf-8")))
    local = 0
    for r in rows:
        p = (r.get("log") or "").strip()
        if p and (root / p).exists():
            local += 1
    return {
        "index_present": True,
        "n_rows": len(rows),
        "transcripts_present": local > 0,
        "n_resolvable_locally": local,
        "example_log_path": (rows[0].get("log") if rows else None),
        "note": ("index rows point at absolute paths outside the repository; "
                 "per-sample transcripts are not recoverable from the clone"),
    }


def coverage_report(cases):
    """What this corpus can and cannot establish, as data rather than prose."""
    fam = {}
    for c in cases:
        f = fam.setdefault(c.family, {"n": 0, "block": 0, "allow": 0, "warn": 0})
        f["n"] += 1
        f[c.expected_verdict] = f.get(c.expected_verdict, 0) + 1

    return {
        "n_cases": len(cases),
        "n_families": len(fam),
        "verdicts": {
            v: sum(1 for c in cases if c.expected_verdict == v) for v in VERDICTS
        },
        "n_attack": sum(1 for c in cases if c.is_attack),
        "n_benign": sum(1 for c in cases if not c.is_attack),
        "n_fp_dedicated": sum(1 for c in cases if c.fp_trap_kind == "dedicated"),
        "n_fp_in_family": sum(1 for c in cases if c.fp_trap_kind == "in_family_control"),
        "n_benign_high_fp_risk": sum(
            1 for c in cases
            if not c.is_attack and c.false_positive_risk == "high"),
        "by_family": fam,
        "confusable_families_present": [f for f in CONFUSABLE_FAMILIES if f in fam],
        "measurable": [
            "specificity: how often a detector fires on non-covert attack traffic",
            "false-positive rate against 22 purpose-built over-blocking traps",
            "the harder false-positive test: benign controls inside attack "
            "families, which resemble the attack they sit beside",
            "per-family firing rates across 18 independent attack families",
        ],
        "not_measurable": [
            "sensitivity to covert channels: absent from this corpus by design "
            "(agent-egress-bench README: 'Covert channels. Timing, header "
            "ordering, HTTP/2 framing, and steganography are out of scope by "
            "design.')",
            "detection on agent transcripts: these cases are wire payloads, "
            "not multi-step agent episodes",
        ],
    }
