#!/usr/bin/env python
"""The reviewer asked for a trade-off, so the figure is a trade-off: quality against cost.

WHY SMALL MULTIPLES AND NOT ONE CHART.  There is no single "computation" axis.  Prefill
carries 61% of RetroPlanner's FLOPs and 1.4% of its model-seconds; decode carries 33% and
95%.  The arms therefore RANK OPPOSITELY depending on which cost you mean, and that
inversion is the answer's substance -- a figure that picked one axis would be a correct
chart making a false point, in whichever direction it picked.  So: one quality axis, four
cost axes, the same marker for an arm in every panel.  What the reader is meant to see is
the points REORDER between (b) and (d).

WHY NOT A DUAL AXIS.  Two measures of different scale on one pair of axes is the standard
way to make two unrelated series look related.  Small multiples with a shared y say the
same thing and cannot be misread.

PANEL (a) CARRIES THE CURVES, the others carry one point per arm.  Quality-vs-budget is
measured at nine budgets for every arm; the compute account exists once per arm, as a
difference between profiler samples taken while it ran.  Drawing a curve where only a
point was measured would be an extrapolation.

MISSING CELLS ARE DRAWN AS MISSING.  Score-only and Random have no compute account: they
ran before the profiler, and every counter it reads is a monotonic server total, so a
per-arm figure is recoverable only from samples taken DURING the run.  They appear in (a),
which needs no account, and are named as absent in the others rather than dropped quietly.

COLOUR follows the ENTITY, in this project's fixed validated order, so an arm keeps its
hue across every figure in the paper.  Every point is direct-labelled as well as legended.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
ARM = {
    "rpfix":        ("RetroPlanner",            "#2a78d6", "o"),
    "rpfix_sigoff": ("RetroPlanner, no signals", "#2a78d6", "o"),
    "score":        ("Score-only  k=4",          "#eda100", "^"),
    "random":       ("Random  k=4",              "#008300", "v"),
    "expand_all":   ("Expand-all frontier",      "#e87ba4", "s"),
}
ORDER = ["rpfix", "rpfix_sigoff", "score", "random", "expand_all"]
HOLLOW = {"rpfix_sigoff"}            # same entity, signals off -> same hue, open marker
# Label offsets are per PANEL, because which pairs collide changes with the x axis:
# on opens, Random (228) sits almost under Expand-all (408); on the cost axes the two
# RetroPlanner points are the close pair instead.
OFF_A = {"rpfix": (10, 4), "rpfix_sigoff": (-14, -20), "score": (12, 2),
         "random": (-20, -20), "expand_all": (10, 4)}
OFF_C = {"rpfix": (12, 2), "rpfix_sigoff": (12, -16), "expand_all": (12, 2),
         "score": (12, 2), "random": (12, 2)}


def point(ax, k, x, y, dx=10, dy=4):
    name, col, mk = ARM[k]
    hollow = k in HOLLOW
    ax.plot([x], [y], mk, ms=11, mfc=SURFACE if hollow else col, mec=col,
            mew=2.2 if hollow else 1.8, zorder=4)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8.2, color=INK, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default="logs/cost_profile/arm_cost.json")
    ap.add_argument("--curves", default="logs/cost_profile/cost_perf_m120.json")
    ap.add_argument("--spend", default="logs/cost_profile/arm_spend_m120.json")
    ap.add_argument("--png", default="logs/cost_profile/cost_quality_plane.png")
    ap.add_argument("--subset-note", default="matched subset: 120 targets")
    a = ap.parse_args()
    acc = json.load(open(a.accounts))
    cur = json.load(open(a.curves))
    # THE SPEND AXIS IS `mean_unique_opens`, NOT the curves file's
    # `mean_unique_calls_spent`.  The two differ by up to 10x (expand-all: 408 vs 39) --
    # one counts molecules the arm OPENED, the other the budget at which its credited
    # routes were reached.  The published table and cost_perf_fig both use opens, so this
    # figure must, or two figures of the same run disagree.
    spd = json.load(open(a.spend))

    def q(k, field="BTR"):
        """quality at the arm's OWN mean spend -- the end of its own line."""
        rec = cur[k]; s = spd[k]["mean_unique_opens"]; v = None
        for c in rec["curve"]:
            if c["budget"] <= s:
                v = c[field]
        return (v if v is not None else rec["curve"][0][field]) * 100

    def spend(k):
        return spd[k]["mean_unique_opens"]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharey=True)
    for ax in axes.ravel():
        ax.grid(True, color="#e8e7e4", lw=.7, zorder=0); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_ylim(20, 95)
    for ax in axes[:, 0]:
        ax.set_ylabel("BTR %   (beats the published route)", fontsize=9, color=INK2)

    # (a) every arm: the whole quality-vs-budget curve plus its operating point
    ax = axes[0, 0]
    for k in ORDER:
        if k not in cur:
            continue
        name, col, mk = ARM[k]
        xs = [c["budget"] for c in cur[k]["curve"]]
        ys = [100 * c["BTR"] for c in cur[k]["curve"]]
        s = spend(k)
        keep = [(x, y) for x, y in zip(xs, ys) if x <= s] or [(xs[0], ys[0])]
        ax.plot([p[0] for p in keep], [p[1] for p in keep], color=col, lw=2,
                ls=(0, (5, 2)) if k in HOLLOW else "-", alpha=.9, zorder=2)
        point(ax, k, s, q(k), *OFF_A[k])
    ax.set_xscale("log")
    ax.set_xlabel("unique molecules opened  (log)", fontsize=9, color=INK2)
    ax.set_title("(a)  search effort — the axis every table in the paper cuts on",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.annotate("better", (.04, .93), xycoords="axes fraction", fontsize=8.5, color=INK3)
    ax.annotate("", (.03, .99), (.14, .88), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=INK3, lw=1.1))

    # (b)(c)(d) only the arms with a measured account
    have = [k for k in ORDER if k in acc]
    missing = [ARM[k][0] for k in ORDER if k not in acc and k in cur]
    specs = [(axes[0, 1], "PFLOP", "arithmetic: PFLOP per target",
              "(b)  FLOPs — the LLM arms are the EXPENSIVE ones here"),
             (axes[1, 0], "Wh_active", "energy above idle: Wh per target",
              "(c)  active energy — same ordering as FLOPs"),
             (axes[1, 1], "min", "end-to-end wall-clock: minutes per target",
              "(d)  seconds — the ordering INVERTS")]
    for ax, field, xlab, title in specs:
        for k in have:
            point(ax, k, acc[k]["per_target"][field], q(k), *OFF_C[k])
        # linear on every cost axis: the measured spread is under 5x, and a log scale
        # there buys nothing while making 0.93 vs 3.46 look like a rounding difference.
        ax.margins(x=.22)
        ax.set_xlabel(xlab, fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        if missing:
            ax.text(.5, .04, "no measured compute account: " + ", ".join(missing),
                    transform=ax.transAxes, ha="center", fontsize=7.6, color=INK3)

    fig.text(.005, .005, a.subset_note + "   ·   quality read at each arm's own mean spend",
             fontsize=7.6, color=INK3)
    fig.tight_layout(rect=(0, .018, 1, 1), h_pad=2.0, w_pad=2.2)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=200)
    print(f"# wrote {a.png}")
    for k in ORDER:
        if k in cur:
            pt = acc.get(k, {}).get("per_target", {})
            print(f"# {ARM[k][0]:26s} opens {spend(k):6.1f}  BTR {q(k):5.1f}%  "
                  f"PFLOP {pt.get('PFLOP','--'):>6}  Wh_act {pt.get('Wh_active','--'):>5}  "
                  f"min {pt.get('min','--'):>5}")


if __name__ == "__main__":
    main()
