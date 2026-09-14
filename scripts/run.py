"""Entrypoint. Produces every table and figure the paper reports.

    python run.py all          # everything, writes out/*.csv and out/*.png
    python run.py two-clock
    python run.py sweep
    python run.py placebo      # open item 1
    python run.py control      # open item 2
    python run.py hn
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warningshot import paths                      # noqa: E402
from warningshot.core import metrics as m          # noqa: E402
from warningshot.data import events as ev          # noqa: E402
from warningshot.data import sources as src        # noqa: E402
from warningshot.eval import mmdecay as md         # noqa: E402
from warningshot.eval import study                 # noqa: E402
from warningshot.viz import plots                  # noqa: E402

OUT = paths.OUT


def _write_csv(rows, name):
    if not rows:
        return None
    paths.ensure(OUT)
    path = OUT / name
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {os.path.relpath(path)}  ({len(rows)} rows)")
    return path


def _table(rows, cols, widths):
    print("    " + "".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("    " + "".join(str(r.get(c, ""))[:w - 1].ljust(w)
                               for c, w in zip(cols, widths)))


def cmd_two_clock():
    print("\n[two-clock] July incident — entity vs concept attention")
    rows = study.two_clock(ev.JULY_INCIDENT)
    _table(rows,
           ["page", "role", "baseline", "bdays_48h", "bdays_7d", "bdays_30d",
            "peak_date", "peak_ratio", "half_life_days"],
           [46, 17, 10, 11, 10, 11, 11, 12, 8])
    for h in ("48h", "7d", "30d"):
        r = study.entity_concept_ratio(rows, h)
        if r:
            print(f"    entity:concept at {h:>3} = {r:6.1f}:1")
    _write_csv(rows, "two_clock_july.csv")
    # The placebo band is fetched here rather than passed in, so the headline
    # chart can never be produced without the null that qualifies it.
    band = None
    try:
        vals = [r["concept_excess"] for r in study.placebo_null()]
        band = (min(vals), max(vals)) if vals else None
    except Exception as exc:  # noqa: BLE001 - chart still useful without band
        print(f"    WARNING: placebo band unavailable ({exc}); chart omits it")
    plots.two_clock_chart(
        rows,
        "Entity attention is unambiguous. The concept response is not.\n"
        "OpenAI/Hugging Face intrusion, from the 21 Jul 2026 attribution",
        "two_clock_july.png", placebo_range=band)
    print("    wrote out/two_clock_july.png")
    return rows


def cmd_sweep():
    print("\n[sweep] baseline sensitivity")
    pages = [ev.JULY_INCIDENT.entity_pages["victim"],
             ev.JULY_INCIDENT.entity_pages["developer"]]
    rows = study.sensitivity_sweep(ev.JULY_INCIDENT, pages)
    _table(rows, ["page", "baseline_len", "baseline", "excess_48h", "excess_30d"],
           [26, 14, 10, 12, 12])
    for p in pages:
        vals = [r["excess_48h"] for r in rows if r["page"] == p]
        if vals and min(vals):
            print(f"    {p:26s} 48h excess spread {min(vals):,} .. {max(vals):,}"
                  f"  ({max(vals) / min(vals):.1f}x)")
    _write_csv(rows, "sensitivity_sweep.csv")
    return rows


def cmd_placebo(observed=None):
    print("\n[placebo] OPEN ITEM 1 — null distribution for 30-day concept drift")
    null = study.placebo_null()
    vals = [r["concept_excess"] for r in null]
    if not vals:
        print("    no placebo windows produced — check span/exclusions")
        return None
    vals_sorted = sorted(vals)
    print(f"    windows: {len(vals)}   span {ev.PLACEBO_SPAN[0]}..{ev.PLACEBO_SPAN[1]}")
    print(f"    null min={min(vals):,}  median={vals_sorted[len(vals)//2]:,}  max={max(vals):,}")
    if observed is None:
        rows = study.two_clock(ev.JULY_INCIDENT)
        observed = sum(r["excess_30d"] for r in rows if r["role"] == "concept")
    pct = m.percentile_rank(observed, vals)
    print(f"    observed (21 Jul + 30d) = {observed:,}")
    print(f"    percentile vs null      = {pct:.1f}")
    print(f"    exceeds max of null?      {'YES' if observed > max(vals) else 'NO'}")
    _write_csv(null, "placebo_null.csv")
    plots.placebo_chart(vals, observed)
    print("    wrote out/placebo.png")
    return {"null": vals, "observed": observed, "percentile": pct}


def cmd_control():
    print("\n[control] OPEN ITEM 2 — same analysis on the June ban")
    rows = study.control_event()
    _table(rows,
           ["page", "role", "baseline", "bdays_48h", "bdays_7d", "bdays_30d",
            "peak_date", "peak_ratio"],
           [46, 17, 10, 11, 10, 11, 11, 12])
    r = study.entity_concept_ratio(rows, "48h", entity_role="developer")
    if r:
        print(f"    developer:concept at 48h = {r:6.1f}:1")
    r30 = study.entity_concept_ratio(rows, "30d", entity_role="developer")
    if r30:
        print(f"    developer:concept at 30d = {r30:6.1f}:1")
    _write_csv(rows, "two_clock_june.csv")
    plots.two_clock_chart(
        rows, "Control event: US directive suspending Fable 5 / Mythos 5\n"
              "from 13 Jun 2026", "two_clock_june.png")
    print("    wrote out/two_clock_june.png")
    return rows


def cmd_hn():
    print("\n[hn] Hacker News reach, both events")
    import datetime as dt

    def ep(s):
        return int(dt.datetime.strptime(s, "%Y-%m-%d")
                   .replace(tzinfo=dt.timezone.utc).timestamp())

    out = []
    for label, q, a, b in [
        ("july_incident", "Hugging Face", "2026-07-16", "2026-07-29"),
        ("june_ban", "Fable Mythos", "2026-06-08", "2026-06-22"),
    ]:
        for r in src.hn_stories(q, ep(a), ep(b))[:8]:
            r["event"] = label
            out.append(r)
    _table(out, ["event", "date", "points", "comments", "title"],
           [16, 12, 8, 10, 74])
    _write_csv(out, "hn_reach.csv")
    return out


def cmd_daily():
    print("\n[daily] raw series with event annotations")
    series = {
        "Hugging Face": src.pageviews("Hugging_Face", "20260701", "20260903"),
        "OpenAI": src.pageviews("OpenAI", "20260701", "20260903"),
    }
    plots.daily_series_chart(
        series,
        {"20260716": "HF discloses", "20260721": "OpenAI attributes",
         "20260826": "reports published"},
        "The disclosure moved nothing; attribution moved everything",
        "daily_july.png")
    print("    wrote out/daily_july.png")


COMMANDS = {
    "two-clock": cmd_two_clock, "sweep": cmd_sweep, "placebo": cmd_placebo,
    "control": cmd_control, "hn": cmd_hn, "daily": cmd_daily,
}


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    if what == "all":
        cmd_two_clock()
        cmd_sweep()
        cmd_control()
        cmd_placebo()
        cmd_hn()
        cmd_daily()
        cmd_multimodal()
        print(f"\nartifacts in {os.path.relpath(OUT)}/")
    elif what in COMMANDS:
        COMMANDS[what]()
    else:
        print(__doc__)
        return 1
    return 0




def cmd_multimodal():
    print("\n[multimodal] three-channel decay regression")
    blocks = md.multimodal()
    sp = md.spread(blocks)
    si = md.second_impulse_check()
    md.save({"channels": blocks, "spread": sp, "second_impulse": si}, "multimodal.json")

    print(f"    {'channel':13s}{'n':>5}{'half-life':>12}{'95% CI':>22}{'R2':>8}  best")
    for b in blocks:
        ci = (f"({b['half_life_ci'][0]:.2f}, {b['half_life_ci'][1]:.2f})"
              if b["half_life_ci"] else "n/a")
        print(f"    {b['channel']:13s}{b['n_points']:>5}"
              f"{b['half_life']:>9.2f} {b['unit']:<3}{ci:>22}"
              f"{b['exponential_r2']:>8.3f}  {b['best_model']}")
    if sp:
        print(f"    spread {sp['slowest']}/{sp['fastest']} = {sp['ratio']:.1f}x")
        for p in sp["pairwise"]:
            verdict = "SEPARATED" if p["separated"] else "overlapping (not shown to differ)"
            print(f"      {p['a']:11s} vs {p['b']:11s} {verdict}")
    for s in si:
        print(f"    horizon {s['horizon_days']:>2}d -> best={s['best_model']} "
              f"R2={s['r2']:.3f} switch={s['switch_day']}d"
              f"{'  <- chasing the second impulse' if s['contains_second_impulse'] else ''}")
    plots.multimodal_chart(blocks)
    print("    wrote out/multimodal.png, results/multimodal.json")
    return blocks


COMMANDS["multimodal"] = cmd_multimodal




if __name__ == "__main__":
    sys.exit(main(sys.argv))
