#!/usr/bin/env python
"""Runtime against route quality for the three LLM planner variants.

THE AXIS IS CALLS, AND THAT IS A FINDING RATHER THAN A FALLBACK.  These arms are not
CPU-bound: a board call waits on an LLM over HTTP, so the GIL is released and the
processor-sharing recovery that the policy arms needed does not apply to them and is not
used here.  What the dumps show instead is that `iter_wall_s / calls_total` lands at
2.4-3.6 s for all three -- the seconds are generation time, nearly flat per call -- so
wall-clock carries almost no information that the call count does not already carry.
Panel (c) is that check, not decoration: without it, quoting calls would be an assumption.

WHAT THE STORED SECONDS STILL CANNOT SAY.  All three ran at 8 shards x 6 workers against
one vLLM fleet, i.e. 48 concurrent clients, so every absolute second includes queueing at
the model server.  The three ran back to back under identical conditions, which makes
their RATIO readable and their absolute values not.  No per-arm GPU or token accounting
exists for them: the cost profiler samples monotonic server counters, so a per-arm figure
is only recoverable as a difference between two samples taken WHILE the arm ran, and
these three predate that.  Hence no PFLOP or kWh panel here -- an estimate would be a
model, and every other compute number in this paper is a measurement.

QUALITY IS BTR ON THE MAIN TABLE'S OWN METRIC, gated by each route's `found_at_unique`,
so a route counts at a budget only if it was actually reached within it.

COLOUR.  Slots of this project's validated categorical palette, in fixed order, and colour
follows the ENTITY: RetroPlanner (main) keeps the blue it has in the cost/perf figure, and
the three variants take the next unused slots.  Every series is direct-labelled as well as
legended, so identity never rests on hue alone.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
# validated categorical palette, fixed order; rpfix pinned to the hue it owns elsewhere
ARM = {
    "single":    ("Single",            "#eb6834", "o", "eval_protocol/results/v20_single_route_cf190/iter.jsonl"),
    "multi_seq": ("Multi-Seq",         "#1baf7a", "s", "eval_protocol/results/v20_multi_seq_cf190/iter.jsonl"),
    "multi_par": ("Multi-Par",         "#4a3aa7", "D", "eval_protocol/results/v20_multi_par_cf190/iter.jsonl"),
    "rpfix":     ("RetroPlanner (main)", "#2a78d6", "^", None),
}
ORDER = ["single", "multi_seq", "multi_par", "rpfix"]


def rows_of(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def crossover(c1, c2, field):
    """First budget at which c2 overtakes c1, linearly interpolated."""
    b1 = {c["budget"]: c[field] for c in c1}
    b2 = {c["budget"]: c[field] for c in c2}
    bs = sorted(set(b1) & set(b2))
    for lo, hi in zip(bs, bs[1:]):
        d0, d1 = b2[lo] - b1[lo], b2[hi] - b1[hi]
        if d0 < 0 <= d1 and d1 != d0:
            return lo + (hi - lo) * (-d0) / (d1 - d0)
    return None


def panel_curve(ax, curves, field, ylabel, title, logy=False):
    for k in ORDER:
        if k not in curves:
            continue
        name, col, mk, _ = ARM[k]
        rec = curves[k]
        xs = [c["budget"] for c in rec["curve"]]
        ys = [c[field] for c in rec["curve"]]
        dashed = k == "rpfix"
        ax.plot(xs, ys, color=col, lw=2.4 if not dashed else 1.6,
                ls="-" if not dashed else (0, (5, 2)),
                marker=mk, ms=5.5, mec=SURFACE, mew=1.1, label=name,
                zorder=3 if not dashed else 2, alpha=1 if not dashed else .75)
        # the arm's own operating point: a hollow ring at its mean unique spend
        s = rec.get("mean_unique_calls_spent")
        if s:
            v = None
            for c in rec["curve"]:
                if c["budget"] <= s:
                    v = c[field]
            if v is not None:
                ax.plot([s], [v], "o", ms=12, mfc="none", mec=col, mew=2.2, zorder=5)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("matched budget: unique molecules opened  (log)", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)


def panel_wall(ax, data):
    for k in ("single", "multi_seq", "multi_par"):
        name, col, mk, _ = ARM[k]
        rs = [r for r in data[k] if r.get("iter_wall_s") and r.get("calls_total")]
        ax.scatter([r["calls_total"] for r in rs], [r["iter_wall_s"] / 60 for r in rs],
                   s=12, color=col, alpha=.30, edgecolors="none", zorder=2)
        mc = st.median(r["calls_total"] for r in rs)
        mw = st.median(r["iter_wall_s"] for r in rs) / 60
        spc = st.median(r["iter_wall_s"] / r["calls_total"] for r in rs)
        ax.plot([mc], [mw], mk, ms=11, mfc=col, mec=SURFACE, mew=2, zorder=5, label=name)
        ax.annotate(f"{name}\n{mc:.0f} calls · {mw:.1f} min · {spc:.1f} s/call",
                    (mc, mw), textcoords="offset points",
                    xytext=(14, -6 if k != "multi_seq" else 12), fontsize=8.5, color=INK,
                    zorder=6)
    ax.set_xlabel("model calls per target (calls_total)", fontsize=9, color=INK2)
    ax.set_ylabel("iterative wall-clock, minutes", fontsize=9, color=INK2)
    ax.set_title("(c)  seconds are ~2.4-3.6 s per call for all three, so CALLS is the cost axis",
                 fontsize=10, color=INK, loc="left", pad=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="logs/cost_profile/planner_curves.json")
    ap.add_argument("--png", default="logs/cost_profile/planner_tradeoff.png")
    a = ap.parse_args()
    curves = json.load(open(a.curves))
    data = {k: rows_of(v[3]) for k, v in ARM.items() if v[3]}

    x = crossover(curves["single"]["curve"], curves["multi_seq"]["curve"], "BTR")
    print(f"# Single -> Multi-Seq BTR crossover at B = {x:.0f} unique opens" if x
          else "# no BTR crossover")
    for k in ORDER:
        rec = curves[k]
        s = rec["mean_unique_calls_spent"]
        at = None
        for c in rec["curve"]:
            if c["budget"] <= s:
                at = c
        print(f"# {ARM[k][0]:20s} spend {s:6.1f}  BTR {at['BTR']:.3f}  "
              f"Success {at['Success']:.3f}  Routes {at['Routes']:.2f}")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 11.0))
    for ax in axes:
        ax.grid(True, color="#e8e7e4", lw=.7, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    panel_curve(axes[0], curves, "BTR", "BTR  (beats the published route)",
                "(a)  quality crosses over: Single wins cheap, Multi-Seq wins rich")
    if x:
        axes[0].axvline(x, color=INK3, lw=1, ls=":", zorder=1)
        axes[0].annotate(f"crossover\nB ≈ {x:.0f}", (x, 0.06), textcoords="offset points",
                         xytext=(6, 0), fontsize=8.5, color=INK2, va="bottom")
    panel_curve(axes[1], curves, "Routes", "routes found per target",
                "(b)  Multi-Seq buys DIVERSITY; Multi-Par spends the same calls for half of it",
                logy=True)
    panel_wall(axes[2], data)
    fig.tight_layout(h_pad=2.2)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=200)
    print(f"# wrote {a.png}")


if __name__ == "__main__":
    main()
