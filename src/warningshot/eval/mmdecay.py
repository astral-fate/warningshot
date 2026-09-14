"""Multi-modal decay study: fit the same decay models to three independent
attention channels and compare their timescales.

The headline quantity is each channel's half-life with a bootstrap interval.
The comparison that matters is whether the intervals overlap: two channels
whose intervals overlap are not shown to decay at different rates, however
different their point estimates look.

Every number produced here is written to results/*.json so verify.py can
recompute it against what the README claims.
"""

import json
import os

import numpy as np

from warningshot import paths
from warningshot.core import regression as rg
from warningshot.data import events as ev
from warningshot.data import modalities as mo

RESULTS = paths.RESULTS

# The horizon stops before 26 Aug 2026, when the OpenAI technical report and the
# METR/Redwood investigation published and re-injected attention. A collective-
# memory decay model assumes a single impulse; fitting across the second one
# makes the estimator chase the new peak instead of the decay. Fitting the full
# 45-day window puts the "switching point" at 35-36 days with R^2 of 0.63,
# which is the model detecting the second event. This is reported as a finding,
# not hidden as a nuisance.
FIRST_IMPULSE_DAYS = 25

HN_STORY_QUERY = "OpenAI and Hugging Face address security incident"
GDELT_QUERY = '"Hugging Face" OpenAI'
BOOTSTRAP_N = 800


def _fit_block(t, y, label, unit, n_boot=BOOTSTRAP_N, min_seg=5):
    """All three models plus a bootstrap interval on the exponential half-life.

    The exponential carries the interval because it is the only model whose
    residuals are exchangeable across the whole series — two-phase residuals
    are not, so bootstrap_ci declines to produce one for it.
    """
    ranked = rg.best_model(t, y, min_seg=min_seg)
    expo = rg.fit_exponential(t, y)
    ci = rg.bootstrap_ci(t, y, rg.fit_exponential, "half_life",
                         n=n_boot, seed=1) if expo else None
    return {
        "channel": label,
        "unit": unit,
        "n_points": int(t.size),
        "half_life": expo["half_life"] if expo else None,
        "half_life_ci": list(ci) if ci else None,
        "exponential_r2": expo["r2"] if expo else None,
        "best_model": ranked[0]["model"] if ranked else None,
        "models": [
            {k: v for k, v in f.items() if k != "pred"} for f in ranked
        ],
    }


def multimodal(horizon=FIRST_IMPULSE_DAYS, n_boot=BOOTSTRAP_N):
    """The three-channel comparison. Returns one block per available channel."""
    out = []

    ch = mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=horizon)
    if ch:
        b = _fit_block(ch["t"], ch["y"], "lookups", "days", n_boot)
        b.update(source="en.wikipedia/Hugging_Face", peak_date=ch["peak_date"],
                 baseline=ch["baseline"], half_life_days=b["half_life"])
        out.append(b)

    try:
        ch = mo.media(GDELT_QUERY, ev.JULY_INCIDENT.t0, horizon=horizon)
    except Exception as exc:  # noqa: BLE001
        print(f"    NOTE: media channel unavailable ({exc})")
        ch = None
    if ch:
        b = _fit_block(ch["t"], ch["y"], "media", "days", n_boot)
        b.update(source=f"GDELT {GDELT_QUERY}", peak_date=ch["peak_date"],
                 baseline=ch["baseline"], half_life_days=b["half_life"])
        out.append(b)

    stories = mo.src.hn_stories(HN_STORY_QUERY, 0, 4102444800)
    top = next((s for s in stories if s["points"] > 500), None)
    if top:
        ch = mo.engagement(top["story_id"])
        if ch:
            h = np.array(ch["hourly"], dtype=float)
            t = np.arange(1, h.size + 1, dtype=float)
            b = _fit_block(t, h, "engagement", "hours", n_boot, min_seg=6)
            b.update(source=f"HN story {top['story_id']}",
                     story_title=top["title"], story_points=top["points"],
                     n_comments=ch["n_comments"], peak_date="hour 0",
                     baseline=0.0,
                     half_life_days=b["half_life"] / 24.0 if b["half_life"] else None)
            out.append(b)
    return out


def spread(blocks):
    """Slowest / fastest half-life in common units, and which pairs separate.

    Overlapping intervals mean the channels are NOT shown to differ. Stating
    that explicitly is the difference between a comparison and a decoration.
    """
    hl = {b["channel"]: b["half_life_days"] for b in blocks if b.get("half_life_days")}
    if len(hl) < 2:
        return None
    slowest = max(hl, key=hl.get)
    fastest = min(hl, key=hl.get)

    ci_days = {}
    for b in blocks:
        if not b.get("half_life_ci"):
            continue
        div = 24.0 if b["unit"] == "hours" else 1.0
        ci_days[b["channel"]] = (b["half_life_ci"][0] / div, b["half_life_ci"][1] / div)

    pairs = []
    names = list(ci_days)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = ci_days[names[i]], ci_days[names[j]]
            overlap = a[0] <= b_[1] and b_[0] <= a[1]
            pairs.append({"a": names[i], "b": names[j],
                          "intervals_overlap": overlap,
                          "separated": not overlap})
    return {
        "half_lives_days": hl,
        "slowest": slowest, "fastest": fastest,
        "ratio": hl[slowest] / hl[fastest],
        "ci_days": {k: list(v) for k, v in ci_days.items()},
        "pairwise": pairs,
    }


def second_impulse_check(long_horizon=45):
    """Evidence for the claim that the full window cannot be fitted.

    Fits the same models across the full window, where the second attention
    impulse (the 26-27 Aug reports) sits, and records what the estimator does.
    """
    out = []
    for h in (FIRST_IMPULSE_DAYS, long_horizon):
        ch = mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=h)
        if not ch:
            continue
        ranked = rg.best_model(ch["t"], ch["y"], min_seg=5)
        best = ranked[0] if ranked else {}
        out.append({
            "horizon_days": h,
            "n_points": int(ch["t"].size),
            "best_model": best.get("model"),
            "r2": best.get("r2"),
            "switch_day": best.get("switch_day"),
            "contains_second_impulse": h >= 36,
        })
    return out


def save(payload, name):
    paths.ensure(RESULTS)
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


# ===================================================================
# Review response block. Produces every number the adversarial review
# demanded, so the paper can report robustness instead of asserting it.
#   C1 estimator dependence      -> estimator_comparison()
#   C2 contamination             -> contamination_sensitivity()
#   M1 autocorrelation           -> interval_comparison()
#   M2 breakpoint selection cost -> selection_corrected_models()
#   M3 season-matched null       -> in scripts/run.py via study module
# ===================================================================

BLOCK_LEN = 4


def _channels_for_review():
    """(label, t, y, unit_divisor) for each channel, cleaned identically."""
    out = []
    ch = mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=FIRST_IMPULSE_DAYS)
    if ch:
        out.append(("lookups", ch["t"], ch["y"], 1.0))
    try:
        ch = mo.media(GDELT_QUERY, ev.JULY_INCIDENT.t0, horizon=FIRST_IMPULSE_DAYS)
        if ch:
            out.append(("media", ch["t"], ch["y"], 1.0))
    except Exception as exc:  # noqa: BLE001
        print(f"    NOTE: media channel unavailable ({exc})")
    stories = mo.src.hn_stories(HN_STORY_QUERY, 0, 4102444800)
    top = next((s for s in stories if s["points"] > 500), None)
    if top:
        ch = mo.engagement(top["story_id"])
        if ch:
            h = np.array(ch["hourly"], dtype=float)
            out.append(("engagement", np.arange(1.0, h.size + 1.0), h, 24.0))
    return out


def estimator_comparison():
    """C1 — half-life under log-space OLS vs raw-scale NLS, and both spreads.

    Neither estimator is privileged. The reportable quantity is the range.
    """
    rows = []
    for name, t, y, div in _channels_for_review():
        ols = rg.fit_exponential(t, y)
        nls = rg.fit_exponential_nls(t, y)
        if not (ols and nls):
            continue
        rows.append({
            "channel": name,
            "half_life_ols_days": ols["half_life"] / div,
            "half_life_nls_days": nls["half_life"] / div,
            "ratio_nls_over_ols": nls["half_life"] / ols["half_life"],
            "r2_log_ols": ols["r2"],
            "r2_raw_nls": nls["r2_raw"],
            "n_points": ols["n"],
        })
    if not rows:
        return None
    o = [r["half_life_ols_days"] for r in rows]
    n = [r["half_life_nls_days"] for r in rows]
    return {
        "channels": rows,
        "spread_ols": max(o) / min(o),
        "spread_nls": max(n) / min(n),
        "ordering_preserved": (
            [r["channel"] for r in sorted(rows, key=lambda r: r["half_life_ols_days"])]
            == [r["channel"] for r in sorted(rows, key=lambda r: r["half_life_nls_days"])]
        ),
    }


def interval_comparison():
    """M1 — i.i.d. vs moving-block bootstrap, and whether conclusions move."""
    rows, ci = [], {}
    for name, t, y, div in _channels_for_review():
        f = rg.fit_exponential(t, y)
        if not f:
            continue
        iid = rg.bootstrap_ci(t, y, rg.fit_exponential, "half_life",
                              n=BOOTSTRAP_N, seed=1)
        blk = rg.block_bootstrap_ci(t, y, rg.fit_exponential, "half_life",
                                    block=BLOCK_LEN, n=BOOTSTRAP_N, seed=1)
        if not (iid and blk):
            continue
        # log-space residual diagnostics, which is what motivated the change
        tc, yc = rg._clean(t, y)
        resid = np.log(yc) - (np.log(f["amplitude"]) - tc / f["tau"])
        dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
        lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
        ci[name] = (blk[0] / div, blk[1] / div)
        rows.append({
            "channel": name,
            "half_life_days": f["half_life"] / div,
            "ci_iid_days": [iid[0] / div, iid[1] / div],
            "ci_block_days": [blk[0] / div, blk[1] / div],
            "width_ratio_block_over_iid": (blk[1] - blk[0]) / (iid[1] - iid[0]),
            "durbin_watson": dw,
            "lag1_autocorr": lag1,
        })
    pairs = []
    names = list(ci)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = ci[names[i]], ci[names[j]]
            pairs.append({"a": names[i], "b": names[j],
                          "intervals_overlap": bool(a[0] <= b[1] and b[0] <= a[1])})
    return {"channels": rows, "pairwise_block": pairs, "block_length": BLOCK_LEN}


def contamination_sensitivity():
    """C2 — the lookups fit with the declared contaminated dates removed."""
    full = mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=FIRST_IMPULSE_DAYS)
    excl = mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0,
                      horizon=FIRST_IMPULSE_DAYS, exclude_contaminated=True)
    if not (full and excl):
        return None
    out = {"excluded_dates": excl.get("excluded_dates", []),
           "reasons": ev.contaminated_for("Hugging_Face")}
    for tag, ch in (("full", full), ("excluded", excl)):
        o = rg.fit_exponential(ch["t"], ch["y"])
        n = rg.fit_exponential_nls(ch["t"], ch["y"])
        out[tag] = {
            "n_points": int(ch["t"].size),
            "half_life_ols": o["half_life"] if o else None,
            "half_life_nls": n["half_life"] if n else None,
            "r2_log_ols": o["r2"] if o else None,
        }
    if out["full"]["half_life_ols"] and out["excluded"]["half_life_ols"]:
        out["ols_shift_ratio"] = (out["excluded"]["half_life_ols"]
                                  / out["full"]["half_life_ols"])
    return out


def selection_corrected_models():
    """M2 — does the two-phase advantage survive paying for the breakpoint search?"""
    rows = []
    for name, t, y, _ in _channels_for_review():
        ms = 6 if name == "engagement" else 5
        v = rg.two_phase_survives_selection(t, y, min_seg=ms)
        if v:
            v["channel"] = name
            rows.append(v)
    return {
        "channels": rows,
        "n_surviving": sum(1 for r in rows if r["survives"]),
        "n_tested": len(rows),
    }
