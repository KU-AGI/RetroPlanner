#!/usr/bin/env python
"""What a target actually costs, and why the stored wall_s could not say.

THREE PANELS, ONE ARGUMENT.

 (a) THE CONTAMINATION, AND WHY IT IS NOT A CONSTANT.  Every arm ran its targets as
     THREADS inside a shard process, and the board eval is GIL-bound: the process holds
     1.0 core whether one target or six are live (measured on 15 workers, 1.05 cores each,
     one thread at 1.01 and the rest at zero).  Under processor sharing a target's CPU time
     inverts exactly from the shard's completion times,

         cpu_j = SUM_{i<=j} (T_i - T_{i-1}) / (n - i + 1)

     and the identity SUM_j cpu_j = T_n holds to the decimal on every shard.  The recovered
     factors are 2.10x / 3.59x / 4.07x / 5.37x -- they differ BY ARM, so the raw ratio is
     wrong in both directions, not merely inflated.  That is the panel: a constant would be
     harmless, a per-arm factor is not.

 (b) THE HONEST NUMBER.  One process per target, --workers 1, one code version, an idle
     box, the same 40 targets for every arm, stratified into four call-count bands because
     the mean is driven by the tail.

 (c) WHAT THE SECONDS ARE MADE OF.  CPU against model calls, one dot per target.  On the
     finished expand-all rows CPU correlates with calls at r=0.72 and with route count at
     r=0.28, so the headline cost is chemistry the arm asked for, not bookkeeping -- the
     O(R^2) route dedup (fixed) dominated only the extreme tail.  Printing (b) without (c)
     would let a reader attribute the whole gap to an implementation detail.

COLOUR.  The arm hues are this project's existing categorical set, already validated at the
all-pairs CVD threshold (worst unsimulated dE 19.6, worst protan/deutan 13.0; the floors are
15 and 8).  No hue is added here: panel (a) separates its two measures by POSITION on one
axis, not by colour, and every series is direct-labelled as well as legended.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
ARM = {"expand_all": ("Expand-all frontier", "#e87ba4"),
       "score":      ("Score-only  k=4",     "#eda100"),
       "random":     ("Random  k=4",         "#008300")}
ORDER = ["expand_all", "score", "random"]
TRUNC: dict = {}


# ------------------------------------------------------------------ deconvolution
def cpu_from_walls(walls):
    """Per-target CPU seconds for targets that SHARED one GIL-bound process."""
    n, out, prev, acc = len(walls), [], 0.0, 0.0
    for i, T in enumerate(sorted(walls)):
        acc += (T - prev) / (n - i)
        prev = T
        out.append(acc)
    return out


def load(path, keys=("target", "wall_s", "calls", "solved", "stop")):
    rows = []
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                d = {k: r.get(k) for k in keys}
                d["n_routes"] = len(r.get("routes") or [])
                rows.append(d)
    return rows


def deconv(groups):
    out = []
    for g in groups:
        g = [r for r in g if r.get("wall_s")]
        if not g:
            continue
        g.sort(key=lambda r: r["wall_s"])
        for r, c in zip(g, cpu_from_walls([r["wall_s"] for r in g])):
            r["cpu_s"] = c
            out.append(r)
    return out


def shard_groups(path, shard_dir):
    """Group a 190-row dump by the shard list that produced it."""
    t2s = {}
    for f in sorted(glob.glob(f"{shard_dir}/t*.jsonl")):
        sh = os.path.basename(f)[:-6]
        for ln in open(f):
            if ln.strip():
                d = json.loads(ln)
                t2s[d.get("smiles") or d.get("target")] = sh
    by = {}
    for r in load(path):
        by.setdefault(t2s.get(r["target"]), []).append(r)
    return list(by.values())


# ------------------------------------------------------------------ panels
def panel_a(ax, stored):
    """CPU -> wall per arm: a dumbbell on one axis, so the gap IS the contamination.

    Arm names are TICK LABELS, not text inside the axes: at log scale the expand-all
    dumbbell reaches the right edge and in-plot names landed on top of it.
    """
    ticks, names = [], []
    for i, k in enumerate(ORDER):
        rows = stored[k]
        y = len(ORDER) - 1 - i
        cpu = st.median(r["cpu_s"] for r in rows) / 60
        wall = st.median(r["wall_s"] for r in rows) / 60
        name, col = ARM[k]
        ticks.append(y); names.append(name)
        ax.plot([cpu, wall], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([cpu], [y], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=3)
        ax.plot([wall], [y], "o", ms=9, color=SURFACE, mec=col, mew=2.2, zorder=3)
        ax.text(cpu, y + 0.22, f"{cpu:.0f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
        ax.text(wall, y + 0.22, f"{wall:.0f} min", ha="center", va="bottom",
                fontsize=8.5, color=INK)
        ax.text(math.sqrt(cpu * wall), y - 0.26, f"x{wall / cpu:.2f}", ha="center",
                va="top", fontsize=9, color=INK2, style="italic")
    ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=9, color=INK2)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_xscale("log"); ax.set_xlim(4, 900)
    ax.set_xlabel("minutes per target  (log)", fontsize=9, color=INK2)
    ax.set_title("(a)  the stored wall-clock inflates by a DIFFERENT factor per arm",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=INK3, mec=SURFACE,
                              mew=1.6, label="recovered CPU"),
                       Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=INK3,
                              mew=2, label="stored wall_s")],
              frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2,
              handletextpad=.4, borderaxespad=.2)


def panel_b(ax, ctrl):
    if not ctrl:
        ax.text(.5, .5, "controlled pass not finished", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=INK3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("(b)  one process per target, one code version, idle box",
                     fontsize=10, color=INK, loc="left", pad=10)
        return
    ticks, names = [], []
    for i, k in enumerate(ORDER):
        rows = ctrl.get(k) or []
        name, col = ARM[k]
        y = len(ORDER) - 1 - i
        ticks.append(y); names.append(name)
        if not rows:
            continue
        v = sorted(r["wall_s"] / 60 for r in rows)
        ax.scatter(v, [y] * len(v), s=16, color=col, alpha=.35,
                   edgecolors="none", zorder=2)
        med = st.median(v)
        ax.plot([med], [y], "|", ms=26, mew=2.6, color=col, zorder=4)
        # THE SCATTER IS RIGHT-TRUNCATED WHERE AN ARM WAS STOPPED EARLY, and a truncated
        # cloud shown without saying so reads as a tighter distribution than it is.  All
        # targets start together, so rows arrive in ascending wall_s: once 21 of 40 have
        # landed the 20th and 21st are known and the MEDIAN is exact, but the mean, the
        # spread and the max are not.  Hence the median tick is the only summary drawn,
        # and the cut is written on the row.
        cut = (TRUNC or {}).get(k)
        lab = f"median {med:.1f} min  (n={len(v)}"
        lab += f" of {cut['total']}, {cut['stragglers_cut']} slower ones cut)" if cut else ")"
        ax.text(med, y + 0.26, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
        if cut:
            ax.annotate("", (max(v), y), (max(v) * 1.25, y),
                        arrowprops=dict(arrowstyle="<-", color=col, lw=1.4, alpha=.6))
    ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=9, color=INK2)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_xscale("log")
    ax.set_xlabel("minutes per target, uncontended  (log)", fontsize=9, color=INK2)
    ax.set_title("(b)  measured honestly: one process per target, idle box, same 40 targets",
                 fontsize=10, color=INK, loc="left", pad=10)


def panel_c(ax, stored):
    for k in ORDER:
        name, col = ARM[k]
        rows = [r for r in stored[k] if r.get("calls")]
        ax.scatter([r["calls"] for r in rows], [r["cpu_s"] / 60 for r in rows],
                   s=13, color=col, alpha=.45, edgecolors="none", label=name, zorder=2)
    # score and random sit almost on top of each other at the median, so their direct
    # labels are pushed to opposite sides rather than both up-right.
    OFF = {"expand_all": (12, 6), "score": (12, 10), "random": (12, -18)}
    for k in ORDER:
        name, col = ARM[k]
        rows = [r for r in stored[k] if r.get("calls")]
        mx, my = st.median(r["calls"] for r in rows), st.median(r["cpu_s"] for r in rows) / 60
        ax.plot([mx], [my], "o", ms=10, mfc=col, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{name}\nmedian {mx:.0f} calls, {my:.0f} min", (mx, my),
                    textcoords="offset points", xytext=OFF[k], fontsize=8.5, color=INK,
                    zorder=6)
    ax.set_xlabel("model calls spent on the target", fontsize=9, color=INK2)
    ax.set_ylabel("recovered CPU, minutes  (log)", fontsize=9, color=INK2)
    ax.set_yscale("log")
    # A handful of targets stopped at 0-1 calls and cost milliseconds; letting them set
    # the floor spends three decades of the axis on noise and flattens everything real.
    ax.set_ylim(0.04, 400)
    ax.text(.99, .02, "targets below 0.04 min (0-1 calls) clipped", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=INK3)
    ax.set_title("(c)  the cost tracks CALLS, not the route bookkeeping that was fixed",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expand-all", default="eval_protocol/results/policy/all_190.jsonl")
    ap.add_argument("--score", default="eval_protocol/results/policy/score_k4.jsonl")
    ap.add_argument("--random", default="eval_protocol/results/policy/random_k4_190.jsonl")
    ap.add_argument("--shards", default="eval_protocol/shards190_32")
    ap.add_argument("--ctrl-dir", default=None,
                    help="controlled-pass root with <arm>/out*.jsonl")
    ap.add_argument("--png", default="logs/cost_profile/timing_honest.png")
    a = ap.parse_args()

    stored = {"expand_all": deconv(shard_groups(a.expand_all, a.shards)),
              "score":      deconv(shard_groups(a.score, a.shards)),
              "random":     deconv(shard_groups(a.random, a.shards))}
    for k, v in stored.items():
        print(f"# {k:11s} n={len(v):3d}  wall med {st.median(r['wall_s'] for r in v)/60:6.1f}m"
              f"  CPU med {st.median(r['cpu_s'] for r in v)/60:6.1f}m")

    ctrl = {}
    global TRUNC
    TRUNC = {}
    if a.ctrl_dir:
        _ms = os.path.join(a.ctrl_dir, "median_stop.jsonl")
        if os.path.exists(_ms):
            _m = {"all": "expand_all", "score": "score", "random": "random"}
            for ln in open(_ms):
                if ln.strip():
                    d = json.loads(ln)
                    TRUNC[_m.get(d["arm"], d["arm"])] = d
            print(f"# truncation record: {TRUNC}")
        for k, arm in (("expand_all", "all"), ("score", "score"), ("random", "random")):
            rows = []
            for f in glob.glob(f"{a.ctrl_dir}/{arm}/out*.jsonl"):
                rows += [r for r in load(f) if r.get("wall_s")]
            if rows:
                ctrl[k] = rows
                print(f"# ctrl {k:11s} n={len(rows):3d} med {st.median(r['wall_s'] for r in rows)/60:.1f}m")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.2))
    for ax in axes:
        ax.grid(True, which="major", color="#e8e7e4", lw=.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    panel_a(axes[0], stored)
    panel_b(axes[1], ctrl)
    panel_c(axes[2], stored)
    fig.tight_layout(h_pad=2.4)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=200)
    print(f"# wrote {a.png}")


if __name__ == "__main__":
    main()
