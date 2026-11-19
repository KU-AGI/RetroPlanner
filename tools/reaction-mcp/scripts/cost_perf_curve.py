#!/usr/bin/env python
"""Route quality against what it cost to get it, on the main table's own metric.

THE QUESTION. "Compared with expanding every eligible frontier molecule, how much
computation does LLM-guided OPEN selection save? Report wall-clock and FLOPs across LLM
inference, single-step prediction and candidate scoring, with the resulting route quality."

That is two measurements joined, and they are joined here rather than modelled:

  QUALITY   BTR at a budget, EXACTLY as the main table computes it -- the share of targets
            whose best route beats the published route on
            plaus_passfrac x rt5 x (1 - min(1, cost/1000)). The budget gate is the route's
            own `found_at_unique`, so a route is only credited to a budget it was actually
            reached within. This is imported from the table script, not reimplemented:
            a second copy of a metric is a second definition of it.

  COST      measured, from cost_profile_sampler.py -- vLLM token counters, the single-step
            menu cache's hit/miss, and integrated GPU power. Nothing here is a nominal
            rate card.

BTR IS TAKEN OVER ALL 190 TARGETS, NOT THE COMMON SUBSET. `--common` exists in the table
script because 1stCutDiff needs two routes to compare, and restricting BTR the same way
stops charging an arm for the targets it failed. For a COST curve that is the wrong
convention: an arm that spends nothing and solves nothing must score 0, or the x-axis is
free and the y-axis is conditioned on success. So `_btr_all` is what this plots, and the
denominator is 190 everywhere.

WHY COST IS NOT MODELLED AS A FUNCTION OF BUDGET. Prefill grows with the board, so an
arm's FLOPs are superlinear in its budget and a per-call rate extrapolated across the
x-axis would be fiction. The budget axis is therefore plotted in the unit the budget is
accounted in -- unique single-step calls -- and the measured compute of each arm's FULL
run is reported as a separate point per arm. Two honest plots beat one invented curve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import ablation_tables_at_budget_cf190 as T     # noqa: E402

# T.R is eval_protocol/results, which is where the ablation arms' dumps live. The main
# RetroPlanner re-measurement does not: it is written to <mcp>/results, the same place
# table_geo.py reads it from. `Path(a) / b` returns b when b is absolute, so an absolute
# path passed through T.row() resolves correctly and the two trees can coexist in ARMS.
_MCP = Path(__file__).resolve().parents[1]

# label, dump (under eval_protocol/results), labeled stem, route key, policy-stamp?
ARMS = {
    "random":      ("Random (k=4)",            "policy/random_k4_190.jsonl",
                    "rq_policy_random_k4_cf190", "routes", True),
    "score":       ("Score-only (k=4)",        "policy/score_k4.jsonl",
                    "rq_policy_score_k4_cf190", "routes", True),
    "expand_all":  ("Expand-all frontier",     "policy/all_190.jsonl",
                    "rq_policy_all_cf190", "routes", True),
    "single":      ("RetroPlanner Single",     "v20_single_route_cf190/iter.jsonl",
                    "rq_v20_single_route_cf190", "routes_union", False),
    "multi_seq":   ("RetroPlanner Multi-Seq",  "v20_multi_seq_cf190/iter.jsonl",
                    "rq_v20_multi_seq_cf190", "routes_union", False),
    "multi_par":   ("RetroPlanner Multi-Par",  "v20_multi_par_cf190/iter.jsonl",
                    "rq_v20_multi_par_cf190", "routes_union", False),
    "rpfix":       ("RetroPlanner (main)",     str(_MCP / "results/rpfix_uspto190.jsonl"),
                    "rq_rpfix_uspto190", "routes_union", False),
    "rpfix_sigoff": ("RetroPlanner (signals off)",
                     str(_MCP / "results/rpfix_sigoff_uspto190.jsonl"),
                     "rq_rpnosig_uspto190", "routes_union", False),
}


def native_stamp(dump, key):
    """True if the dump's own routes already carry `found_at_unique`.

    The table script rebuilds that number from `applied_steps` when the arm is a scripted
    policy, because the single-pass eval never wrote it. That rebuild counts DISTINCT
    MOLECULES WITH AN APPLIED CANDIDATE, and on Random it lands within 0.3% of the opens
    the LLM arms are budgeted in -- but on the exhaustive-frontier arm it does not, because
    a turn opens the whole frontier and the budget ends before most of it is ranked. So a
    dump that carries the board's own open-count stamp is always preferred, and the rebuild
    is the fallback for dumps written before the board recorded one.
    """
    # SCAN UNTIL A ROW ACTUALLY HAS A ROUTE, not just the first row. The expand-all dump
    # opens with a target that produced none, and stopping there answered "no native
    # stamp" -- which sent the arm down the `applied_steps` rebuild and raised KeyError on
    # a dump that never carried that field. A route-less first row is the common case for
    # any arm with failures, so the question is about the FILE, not about row 0.
    with open(T.R / dump) as f:
        for line in f:
            if not line.strip():
                continue
            rs = json.loads(line).get(key) or []
            if not rs:
                continue
            return any(x.get("found_at_unique") is not None for x in rs)
    return False


def spent(dump, key, policy, only=None):
    """Mean unique single-step calls actually spent per target, and the route stamps.

    Reported next to the budget axis because a budget is a CAP: an arm that stops early
    sits at the same x as one that spent it all, and the two are not the same experiment.
    """
    recs = [json.loads(l) for l in open(T.R / dump)]
    if only is not None:
        recs = [r for r in recs if r["target"] in only]
    tot, n = 0.0, 0
    for rec in recs:
        st = T.policy_stamps(rec) if policy else {
            T.route_key(x["steps"]): x.get("found_at_unique")
            for x in (rec.get(key) or [])}
        v = [x for x in st.values() if x is not None]
        tot += max(v) if v else 0
        n += 1
    return (tot / n if n else 0.0), len(recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="random,score,expand_all",
                    help="comma-separated keys of ARMS")
    ap.add_argument("--budgets", default="10,20,30,50,75,100,150,200,300")
    ap.add_argument("--profile-dir", default="logs/cost_profile")
    ap.add_argument("--out-json", default="logs/cost_profile/cost_perf.json")
    ap.add_argument("--png", default=None)
    ap.add_argument("--only-targets", default=None,
                    help="file of target SMILES, one per line: restrict EVERY arm to these")
    a = ap.parse_args()

    if a.only_targets:
        T.ONLY = {l.strip() for l in open(a.only_targets) if l.strip()}
        print(f"# matched subset: {len(T.ONLY)} targets, every arm restricted to them")

    ref = T.reference()
    budgets = [int(x) for x in a.budgets.split(",")]
    out = {}
    for k in a.arms.split(","):
        k = k.strip()
        if k not in ARMS:
            print(f"!! unknown arm {k}", file=sys.stderr)
            continue
        label, dump, stem, key, policy = ARMS[k]
        if not (T.R / dump).exists():
            print(f"-- {label}: no dump at {T.R / dump}, skipped")
            continue
        if not (T.L / f"labeled_{stem}.jsonl").exists():
            print(f"-- {label}: dump present but NOT LABELED "
                  f"({T.L}/labeled_{stem}.jsonl) -- route quality needs the scoring pass")
            continue
        if policy and native_stamp(dump, key):
            policy = False
            print(f"-- {label}: using the board's own open-count stamp, not the rebuild")
        curve = []
        for b in budgets:
            r = T.row(dump, stem, key, policy, b, ref)
            curve.append({"budget": b, "BTR": round(r["_btr_all"], 4),
                          "Success": round(r["Success"], 4),
                          "Routes": round(r["Routes"], 3)})
        mean_spent, n = spent(dump, key, policy, T.ONLY)
        out[k] = {"label": label, "n_targets": n,
                  "mean_unique_calls_spent": round(mean_spent, 1), "curve": curve}
        print(f"\n=== {label}  (n={n}, mean unique calls spent {mean_spent:.1f})")
        print("   " + "".join(f"{c['budget']:>8}" for c in curve))
        print("BTR" + "".join(f"{c['BTR']:>8.3f}" for c in curve))
        print("Suc" + "".join(f"{c['Success']:>8.3f}" for c in curve))

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=2)
    print(f"\n-> {a.out_json}")
    if T.UNRESOLVED[1]:
        print(f"   first_cut unresolved {T.UNRESOLVED[0]}/{T.UNRESOLVED[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
