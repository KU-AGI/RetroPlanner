#!/usr/bin/env python
"""Two Pareto fronts over the same arms: quality against FLOPs, quality against runtime.

THE TWO PANELS DO NOT AGREE, AND THAT IS THE RESULT.

  FLOPs-quality   every arm is non-dominated.  Expand-all is 3.7x cheaper in arithmetic
                  and 49 BTR points worse; no arm is both cheaper and better than another.
                  A real trade-off, and the LLM is the EXPENSIVE end of it.
  runtime-quality RetroPlanner dominates outright -- lower minutes per target AND higher
                  BTR than everything else, so the front is a single point.  No trade-off
                  to make.

Prefill carries 61% of RetroPlanner's FLOPs and 1.4% of its model-seconds; decode carries
33% and 95%.  That is the mechanism behind the disagreement, and it is why a paper that
reports one of these axes is making a claim its own other axis contradicts.

THE RIGHT PANEL IS PROVISIONAL AND SAYS SO ON ITS FACE.  The two runtimes come from
different deployments: the LLM arms ran 16 shards x 8 workers and are I/O-bound on the
model server, expand-all ran 32 x 6 and is CPU-bound, which under the board eval's GIL
means it held 32 cores of a 192-core box.  Neither number is a property of the algorithm.
A controlled pass -- one process per target, one code version, an idle box, the same
targets -- is what settles it, and until it lands the dominance claim is marked.

MISSING ARMS ARE NAMED.  Score-only and Random have quality but no compute account: the
profiler reads monotonic server counters, so a per-arm figure exists only as a difference
between samples taken WHILE the arm ran, and they predate it.  They are listed as absent
rather than dropped, and the controlled pass is collecting their accounts now.

COLOUR follows the entity, in this project's fixed validated order, so an arm keeps its
hue across every figure in the paper.  Every point is direct-labelled as well as legended.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#17160f", "#55534a", "#9a978c"
GRID, DOMFILL = "#ebe9e3", "#f1efe9"
ARM = {
    "rpfix":        ("RetroPlanner",             "#2a78d6", "o", False),
    "rpfix_sigoff": ("RetroPlanner, no signals", "#2a78d6", "o", True),
    "expand_all":   ("Expand-all frontier",      "#e87ba4", "s", False),
    "score":        ("Score-only  k=4",          "#eda100", "^", False),
    "random":       ("Random  k=4",              "#008300", "v", False),
}
ORDER = ["rpfix", "rpfix_sigoff", "expand_all", "score", "random"]
PART: dict = {}          # arm -> how many controlled targets its median rests on


def pareto(pts):
    """Non-dominated under (minimise x, maximise y). pts: {key: (x, y)}."""
    out = []
    for k, (x, y) in pts.items():
        if not any(k2 != k and x2 <= x and y2 >= y and (x2 < x or y2 > y)
                   for k2, (x2, y2) in pts.items()):
            out.append(k)
    return sorted(out, key=lambda k: pts[k][0])


def panel(ax, pts, xlab, title, sub, note=None, off=None):
    """One cost-quality plane.

    THE DOMINATED REGION IS SHADED RATHER THAN THE FRONT DRAWN AS A LINE.  A staircase
    through the front is the conventional rendering and it reads as a trend line -- as if
    the arms were samples of one curve.  They are five separate systems.  Shading the
    quadrant each front point dominates says the one thing the panel is for: whether any
    arm sits inside another's shadow.
    """
    if not pts:
        ax.text(.5, .5, "not measured yet", transform=ax.transAxes, ha="center",
                va="center", fontsize=10, color=INK3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=20)
        return []
    front = pareto(pts)
    xs = [v[0] for v in pts.values()]; ys = [v[1] for v in pts.values()]
    span = (max(xs) - min(xs)) or max(xs)
    xlo, xhi = max(0, min(xs) - span * .30), max(xs) + span * .36
    ax.set_xlim(xlo, xhi)
    ylo = ax.get_ylim()[0]

    # ONE polygon, not a rectangle per front point: overlapping fills darken where they
    # meet and the darker band reads as data.  And it is only drawn when some arm is
    # actually dominated -- with every arm on the front the shading fills the panel and
    # says nothing.
    if len(front) < len(pts):
        px, py = [xhi], [ylo]
        for k in front:                       # ascending x
            x, y = pts[k]
            px += [x, x]; py += [py[-1], y]
        px.append(xhi); py.append(py[-1])
        ax.fill(px, py, facecolor=DOMFILL, edgecolor="none", zorder=0)
        ax.plot(px[1:-1], py[1:-1], color=INK3, lw=.9, alpha=.5, zorder=1)
        ax.text(xhi - span * .03, ylo + (ax.get_ylim()[1] - ylo) * .045, "dominated",
                ha="right", va="bottom", fontsize=8, color=INK3, style="italic", zorder=1)

    for k, (x, y) in pts.items():
        name, col, mk, hollow = ARM[k]
        on = k in front
        ax.plot([x], [y], mk, ms=14 if on else 11,
                mfc=SURFACE if hollow else col, mec=col if on else INK3,
                mew=2.6 if hollow else 2.0, alpha=1 if on else .6, zorder=5)
        ax.plot([x], [y], mk, ms=19 if on else 15, mfc="none", mec=SURFACE, mew=3, zorder=4)
        dx, dy = (off or {}).get(k, (15, 7))
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9.4, color=INK if on else INK2, zorder=6,
                    fontweight="medium" if on else "normal")
        unit = " PFLOP" if "PFLOP" in xlab else " min"
        ax.annotate(f"{x:g}{unit}  ·  {y:.1f}% BTR", (x, y), textcoords="offset points",
                    xytext=(dx, dy - 12), fontsize=8.2, color=INK3, zorder=6)
        if not on:
            ax.annotate("dominated", (x, y), textcoords="offset points",
                        xytext=(dx, dy - 23), fontsize=8, color=INK3,
                        style="italic", zorder=6)
    ax.set_xlabel(xlab, fontsize=9.6, color=INK2, labelpad=9)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=22, fontweight="semibold")
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.2, color=INK2, va="bottom")
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default="logs/cost_profile/arm_cost.json")
    ap.add_argument("--curves", default="logs/cost_profile/cost_perf_m120.json")
    ap.add_argument("--spend", default="logs/cost_profile/arm_spend_m120.json")
    ap.add_argument("--png", default="logs/cost_profile/pareto_cost_quality.png")
    ap.add_argument("--subset", default="USPTO-190, matched subset of 120 targets")
    ap.add_argument("--arms", default="rpfix,expand_all,score,random",
                    help="which arms to plot. THE DEFAULT IS THE QUESTION, not whatever "
                         "has a compute account: this figure asks what LLM-guided OPEN "
                         "SELECTION saves against expanding everything, so the set is the "
                         "selection strategies. rpfix_sigoff is a signals ablation -- it "
                         "answers a different question, sits on top of rpfix in x, and "
                         "belongs in its own figure.")
    ap.add_argument("--ctrl-dir", default=None,
                    help="controlled-pass root; replaces panel (b)'s x with per-target "
                         "LATENCY medians measured there")
    a = ap.parse_args()
    acc, cur, spd = (json.load(open(f)) for f in (a.accounts, a.curves, a.spend))

    def btr(k):
        """BTR at the arm's OWN mean unique opens -- the end of its own line."""
        s = spd[k]["mean_unique_opens"]; v = None
        for c in cur[k]["curve"]:
            if c["budget"] <= s:
                v = c["BTR"]
        return 100 * (v if v is not None else cur[k]["curve"][0]["BTR"])

    want = [k.strip() for k in a.arms.split(",") if k.strip()]
    bad = [k for k in want if k not in ARM]
    if bad:
        raise SystemExit(f"unknown arm(s): {bad}; known: {list(ARM)}")
    have = [k for k in ORDER if k in want and k in acc and k in cur]
    absent = [ARM[k][0] for k in ORDER if k in want and k not in acc and k in cur]
    fl = {k: (round(acc[k]["per_target"]["PFLOP"], 2), btr(k)) for k in have}

    # PANEL (b) IS LATENCY, AND THE PUBLISHED COLUMN IS NOT.  `arm_cost.json`'s
    # `per_target.min` is arm wall-hours / targets -- THROUGHPUT at whatever concurrency
    # that arm happened to run (192-way for expand-all, 128-way for the LLM arms), which
    # is why it reads 2.88 min against a per-target wall of 301.  Mixing the two on one
    # axis understates expand-all ~20x.  With --ctrl-dir every arm is re-read as the
    # MEDIAN of its own per-target wall at the controlled pass's 40-way concurrency: one
    # definition, one concurrency, all arms.
    rt, pending = {}, []
    if a.ctrl_dir:
        import glob, statistics as _st
        m = {"expand_all": "all", "score": "score", "random": "random", "rpfix": "rpfix"}
        for k in ORDER:
            if k not in cur or k not in want:
                continue
            fs = glob.glob(os.path.join(a.ctrl_dir, m.get(k, k), "out*.jsonl"))
            v = [json.loads(l)["wall_s"] / 60 for f in fs for l in open(f)
                 if l.strip() and json.loads(l).get("wall_s")]
            if v:
                rt[k] = (round(_st.median(v), 1), btr(k)); PART[k] = len(v)
            elif k in acc:
                pending.append(ARM[k][0])
        xlab2 = "minutes per target, median      ← faster"
        sub2 = "latency, one process per target, idle box, 40 running at once"
        note2 = ("Panel (b) is LATENCY, and the published min/target column is not: that one "
                 "is arm wall-clock divided by targets, i.e. throughput at whatever "
                 "concurrency the arm happened to run (192-way for expand-all, 128-way for "
                 "the LLM arms). It reads 2.88 min for expand-all against a per-target wall "
                 "of 301. One definition and one concurrency for every arm here. "
                 "Median over " + ", ".join(f"{ARM[k][0]} n={PART[k]}" for k in rt)
                 + ("; still pending: " + ", ".join(pending) + "." if pending else "."))
    else:
        rt = {k: (acc[k]["per_target"]["min"], btr(k)) for k in have}
        xlab2 = "minutes per target      ← faster"
        sub2 = "throughput as published: arm wall-clock / targets"
        note2 = ("PROVISIONAL. The arms ran at different parallelism (LLM arms 16x8 and "
                 "I/O-bound; expand-all 32x6 and GIL-bound on 32 of 192 cores), and this "
                 "column is throughput, not per-target latency.")

    # Titles are DERIVED, never asserted: this figure is redrawn each time an arm lands,
    # and a hardcoded "X dominates" outlives the data that justified it.
    def _title(tag, pts, axis):
        f = pareto(pts)
        if len(pts) < 2:
            return f"{tag}  {axis} vs quality"
        if len(f) == 1:
            return f"{tag}  {axis} vs quality — {ARM[f[0]][0]} dominates"
        if len(f) == len(pts):
            return f"{tag}  {axis} vs quality — every arm non-dominated"
        return f"{tag}  {axis} vs quality — {len(f)} of {len(pts)} on the front"

    plt.rcParams.update({
        "font.size": 9.4, "axes.edgecolor": "#d8d5cd", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.linewidth": .9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True)
    ylo, yhi = 22, 97
    for ax in axes:
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(range(30, 91, 15))
        ax.grid(True, axis="y", color=GRID, lw=.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=3, width=.9)
    axes[0].set_ylabel("BTR %   route beats the published one", fontsize=9.6, color=INK2,
                       labelpad=9)

    f1 = panel(axes[0], fl, "PFLOP per target      ← cheaper",
               _title("(a)", fl, "FLOPs"),
               "arithmetic, measured from server counters — independent of how it is deployed",
               off={"rpfix": (-30, 20), "rpfix_sigoff": (-120, -6), "expand_all": (15, 7)})
    f2 = panel(axes[1], rt, xlab2,
               _title("(b)", rt, "runtime"),
               sub2, note=note2,
               off={"rpfix": (15, 7), "rpfix_sigoff": (15, -30), "expand_all": (-150, 7)})

    # No "better" compass: both x labels already carry their own direction arrow and up
    # is obviously better on a percentage, so a third indicator is clutter that collided
    # with the axis anyway.

    import textwrap
    head = a.subset + " · quality read at each arm's own mean spend"
    if absent:
        head += " · no compute account yet: " + ", ".join(absent)
    cap = textwrap.fill(head + "   " + note2, 168)
    fig.text(.012, .020, cap, fontsize=8.2, color=INK3, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=.072, right=.988, top=.845,
                        bottom=.16 + .028 * cap.count(chr(10)), wspace=.10)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=200)
    print(f"# wrote {a.png}")
    print(f"# FLOPs front  : {[ARM[k][0] for k in f1]}")
    print(f"# runtime front: {[ARM[k][0] for k in f2]}")
    for k in have:
        _m = f"{rt[k][0]:5.2f}" if k in rt else "   --"
        print(f"#   {ARM[k][0]:26s} PFLOP {fl[k][0]:5.3f}  min {_m}  BTR {fl[k][1]:5.1f}%"
              + (f"  (controlled n={PART[k]})" if k in PART else ""))


if __name__ == "__main__":
    main()
