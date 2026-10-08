#!/usr/bin/env python
"""Build the SPEND file used by table_geo.from_pooled.

WHY. The pooled files for Retro-R1 and RetroAgent merge several independent runs,
and `found_at_calls` is a WITHIN-RUN stamp. Comparing on a shared budget axis means
laying the runs end-to-end, which needs the budget each (target, run) actually spent.
Without SPEND, from_pooled falls back to 'the last stamp at which that run produced a
route', which misses (a) budget spent after finding a route and (b) the entire cost
of runs that found no route at all, and so overstates success at a fixed budget.

Sources:
  RetroAgent   per_experiment[e][x].total_budget in logs/eval/<run>/summary.json
               (stops as soon as solved, so total_budget == solved_at_budget)
  Retro-R1     calls_cumulative of the last round of each run in ncurve_*.r*.jsonl
               (already cumulative across rounds, so max is that run's total spend)

Keys must be byte-identical to the strings table_geo._run_id builds:
  RetroAgent   eval_000000 .. eval_000009
  Retro-R1     ncurve_rsmiles_runt0_1 ..  /  ncurve_rsmiles_runchembl_t0_1 ..
The two arms' namespaces do not overlap, so they can share one dict. table_geo
reads only this one file under tag 'rsmiles', so USPTO and ChEMBL go in together.

Usage (from any directory; every path is resolved through config/paths.py):
  python evaluation/eval_protocol/runners/build_spend.py            # write to the default path
  python evaluation/eval_protocol/runners/build_spend.py --out DIR
"""
import argparse
import collections
import glob
import json
import importlib.util
import os
import statistics as st

# config/paths.py, loaded by file location under its own name so it cannot shadow the
# analysis pipeline's `paths` module.
_spec = importlib.util.spec_from_file_location(
    "rp_paths", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "config", "paths.py"))
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)

RA_USPTO = os.path.join(RP.BASELINE, "RetroAgent", "logs", "eval", "retroagent_rsmiles_t0_k10")
RA_CHEMBL = os.path.join(RP.BASELINE, "RetroAgent", "logs", "eval", "retroagent_rsmiles_chembl1000_t0_k10")
R1_LOGS = os.path.join(RP.BASELINE, "Retro-R1", "logs")
ROUTE_SEARCH = os.path.join(RP.DATA, "route_search")


def targets(path):
    return [json.loads(l)["target"] for l in open(path) if l.strip()]


def from_retroagent(summary_dir, target_file, sp):
    """Spread total_budget from summary.json into (target, eval_%06d)."""
    T = targets(target_file)
    s = json.load(open(os.path.join(summary_dir, "summary.json")))
    n = 0
    vals = []
    for e, exp in enumerate(s["per_experiment"]):
        for x in exp:
            i = x.get("target_index")
            if i is None or not (0 <= i < len(T)):
                continue
            v = float(x.get("total_budget") or 0)
            sp.setdefault(T[i], {})[f"eval_{e:06d}"] = v
            vals.append(v)
            n += 1
    return n, vals


def from_retro_r1(stem, runs, sp):
    """Use each run's last-round calls_cumulative as that run's total spend."""
    n = 0
    vals = []
    for r in runs:
        rid = f"{stem}{r}"
        tot = {}
        for f in sorted(glob.glob(f"{R1_LOGS}/{rid}.r*.jsonl")):
            for l in open(f):
                l = l.strip()
                if not l:
                    continue
                try:
                    d = json.loads(l)
                except Exception:
                    continue
                t = d.get("target")
                if not t:
                    continue
                v = d.get("calls_cumulative")
                if v is None:
                    v = d.get("expansions")
                if v is None:
                    continue
                tot[t] = max(tot.get(t, 0.0), float(v))
        for t, v in tot.items():
            sp.setdefault(t, {})[rid] = v
            vals.append(v)
            n += 1
    return n, vals


def q(v, name):
    if not v:
        print(f"  {name}: none")
        return
    v = sorted(v)
    print(f"  {name:34} cells {len(v):>6}  med={v[len(v)//2]:>6.0f} "
          f"p90={v[int(.9*len(v))]:>6.0f} max={v[-1]:>7.0f} mean={st.mean(v):>6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROUTE_SEARCH, "spend"),
                    help="directory to write ra_spend_uspto.json into (the value to pass as SPEND_DIR)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = {}

    n, v = from_retroagent(RA_USPTO, os.path.join(ROUTE_SEARCH, "targets_uspto190.jsonl"), sp)
    q(v, "RetroAgent USPTO-190")
    n, v = from_retroagent(RA_CHEMBL, os.path.join(ROUTE_SEARCH, "targets_chembl1000.jsonl"), sp)
    q(v, "RetroAgent ChEMBL-1000")
    n, v = from_retro_r1("ncurve_rsmiles_runt0_", range(1, 11), sp)
    q(v, "Retro-R1 USPTO-190 (run 1-10)")
    n, v = from_retro_r1("ncurve_rsmiles_runchembl_t0_", range(1, 5), sp)
    q(v, "Retro-R1 ChEMBL-1000 (run 1-4)")

    p = os.path.join(a.out, "ra_spend_uspto.json")
    json.dump(sp, open(p, "w"))
    cells = sum(len(v) for v in sp.values())
    print(f"\n-> {p}   targets {len(sp)}  cells {cells}")


if __name__ == "__main__":
    main()
