"""Figures. The two-clock chart is the paper's centrepiece: it has to carry
the whole argument to a non-technical reader, which is itself the track's
subject matter.
"""

import os

from warningshot import paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = paths.OUT

ENTITY = "#B34700"
CONCEPT = "#1F5673"
CONTROL = "#8A8A8A"


def _save(fig, name):
    paths.ensure(OUT)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def two_clock_chart(rows, title, name="two_clock.png", placebo_range=None):
    """Cumulative excess in raw views at each horizon, with the concept pages
    summed into one line.

    `placebo_range` is (min, max) of the 30-day placebo null for summed concept
    excess. Drawing it is not optional decoration: without it the concept line
    climbs and reads as conversion, when the null shows that climb is ordinary
    drift. A chart that omits the band tells the wrong story.
    """
    horizons = ["48h", "7d", "30d"]
    x = list(range(len(horizons)))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    entities = [r for r in rows if r["role"] in ("victim", "developer", "product")]
    concepts = [r for r in rows if r["role"] == "concept"]
    controls = [r for r in rows if r["role"] == "generic_control"]

    for r in entities:
        y = [r[f"excess_{h}"] for h in horizons]
        ax.plot(x, y, "-", color=ENTITY, linewidth=2.6, marker="o", markersize=5,
                zorder=4, label=f"{r['page'].replace('_', ' ')} ({r['role']})")

    if concepts:
        y = [sum(r[f"excess_{h}"] for r in concepts) for h in horizons]
        ax.plot(x, y, "-", color=CONCEPT, linewidth=2.6, marker="s", markersize=5,
                zorder=3, label=f"{len(concepts)} hazard concepts (summed)")

    for r in controls:
        y = [r[f"excess_{h}"] for h in horizons]
        ax.plot(x, y, "--", color=CONTROL, linewidth=1.8, marker="^", markersize=4,
                zorder=2, label=f"{r['page'].replace('_', ' ')} (generic control)")

    if placebo_range:
        lo, hi = placebo_range
        ax.fill_between([x[-1] - 0.34, x[-1] + 0.34], lo, hi, color=CONCEPT,
                        alpha=0.16, zorder=1)
        ax.plot([x[-1] - 0.34, x[-1] + 0.34], [hi, hi], color=CONCEPT,
                linewidth=1.0, alpha=0.55)
        ax.plot([x[-1] - 0.34, x[-1] + 0.34], [lo, lo], color=CONCEPT,
                linewidth=1.0, alpha=0.55)
        ax.text(x[-1] - 0.30, hi * 1.12,
                "ordinary 30-day drift\nin these same pages\n(placebo null)",
                fontsize=7.2, color=CONCEPT, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(["48 hours", "7 days", "30 days"])
    ax.set_xlim(-0.35, len(horizons) - 0.35)
    ax.set_ylabel("cumulative excess views above baseline")
    ax.set_title(title, fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    fig.tight_layout()
    return _save(fig, name)


def placebo_chart(null_values, observed, name="placebo.png"):
    """Where the observed 30-day concept excess sits in the null distribution."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(null_values, bins=min(18, max(5, len(null_values) // 2)),
            color=CONTROL, alpha=0.75, edgecolor="white",
            label=f"placebo windows (n={len(null_values)})")
    ax.axvline(observed, color=ENTITY, linewidth=2.4,
               label=f"observed after 21 Jul ({observed:,.0f})")
    ax.set_xlabel("summed 30-day concept-page excess (views)")
    ax.set_ylabel("placebo windows")
    ax.set_title("Is the 30-day concept rise larger than ordinary drift?", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, name)


def daily_series_chart(series_map, marks, title, name="daily.png"):
    """Raw daily series with event annotations — the disclosure-vs-attribution
    and two-peak findings are both legible only on the raw series."""
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for label, series in series_map.items():
        ks = sorted(series)
        ax.plot(range(len(ks)), [series[k] for k in ks], linewidth=1.6, label=label)
    ks = sorted(next(iter(series_map.values())))
    idx = {k: i for i, k in enumerate(ks)}
    for date, note in marks.items():
        if date in idx:
            ax.axvline(idx[date], color="#B34700", linewidth=0.9, linestyle="--", alpha=0.8)
            ax.text(idx[date] + 0.6, ax.get_ylim()[1] * 0.92, note,
                    fontsize=7, rotation=90, va="top", color="#B34700")
    step = max(1, len(ks) // 12)
    ax.set_xticks(list(range(0, len(ks), step)))
    ax.set_xticklabels([f"{k[4:6]}-{k[6:8]}" for k in ks[::step]], fontsize=8)
    ax.set_ylabel("daily pageviews (human)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, name)


def multimodal_chart(blocks, name="multimodal.png"):
    """Half-life per channel with bootstrap intervals, on a log axis in days.

    A log axis is not stylistic: the channels span 20x, so a linear axis
    compresses the community channel to nothing and hides the finding.
    """
    order = sorted(blocks, key=lambda b: b.get("half_life_days") or 0)
    labels = [b["channel"] for b in order]
    y = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.6, 2.9))

    for i, b in enumerate(order):
        hl = b["half_life_days"]
        div = 24.0 if b["unit"] == "hours" else 1.0
        lo, hi = (b["half_life_ci"][0] / div, b["half_life_ci"][1] / div)
        color = ENTITY if b["channel"] == "engagement" else CONCEPT
        ax.plot([lo, hi], [i, i], color=color, linewidth=3.0, solid_capstyle="round",
                alpha=0.55, zorder=2)
        ax.plot([hl], [i], "o", color=color, markersize=8, zorder=3)
        txt = (f"{b['half_life']:.2f} h" if b["unit"] == "hours"
               else f"{b['half_life']:.2f} d")
        ax.annotate(txt, (hl, i), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.5, color=color, weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("attention half-life (days, log scale)")
    ax.set_title("One event, three channels, a 20.8x spread in half-life", fontsize=11)
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout()
    return _save(fig, name)
