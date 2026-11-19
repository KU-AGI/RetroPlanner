#!/usr/bin/env python
"""Per-run ORDERED expansion lists for the two pooled baselines on USPTO-190.

WHY. `table_geo.from_pooled` lays the 10 independent runs end to end and charges each one
its full scalar spend (`off += cost[r]`), so a molecule an earlier run expanded is paid for
again. Both baselines' own code charges a molecule ONCE per target -- RetroAgent's
`slime_retro/eval.py:156` keeps `seen_smiles` outside the rollout loop, Retro-R1 keeps one
`mol_dict` across rounds -- and that is also RetroPlanner's `found_at_unique` rule. To put
the pooled arms on that rule the scalar cost is not enough: we need WHICH molecules each run
opened, in order, so a route's stamp can be re-expressed as "unique molecules seen so far".

SOURCES (no re-inference; both are the runs that produced the pooled files)
  Retro-R1    logs/ncurve_graph_rsmiles_runt0_<k>/<md5>.pkl -> {target, mol_dict, count}
              `mol_dict` is insertion-ordered and its length matches `count` (the reported
              budget) on 39/40 spot-checked targets, off by one where `process_chira` added
              a canonical alias key under one count.
  RetroAgent  logs/eval/retroagent_rsmiles_t0/eval_<e>/target_<i>/rollout_*/transcript.json
              -> `tool_calls_log` predict_templates calls, molecule_id resolved through
              `graph_state.nodes`. Rollouts of one experiment are concatenated in order,
              which is what `seen_smiles` accumulates over.

OUT: {arm: {run_id: {target_smiles: [smiles, ...]}}} as JSON.
"""
import glob
import json
import os
import pickle
import re
import sys
from collections import OrderedDict

BASE = "/mnt/data/self-evolving-agent-chem/sci-orchestrator/baseline"
OUT = sys.argv[1] if len(sys.argv) > 1 else "expansions.json"

# WHICH RUN DIRECTORY. This is not cosmetic: `retroagent_rsmiles_t0` and
# `retroagent_rsmiles_t0_k10` both number their experiments eval_000000..9, so an
# expansions file built from one and applied to the other's pool passes every guard in
# table_geo (the ids are all present) and silently charges the pool using another run set's
# molecules. Exactly the trap that `ra_spend_uspto.json` (built from _k10) already fell into
# against `pooled_retroagent_rsmiles_fa.jsonl` (from _t0). The source roots are recorded in
# the output under "_meta" so a reader can check the pairing.
RA_ROOT = os.environ.get("RA_ROOT", f"{BASE}/RetroAgent/logs/eval/retroagent_rsmiles_t0")
R1_GLOB = os.environ.get("R1_GLOB", f"{BASE}/Retro-R1/logs/ncurve_graph_rsmiles_runt0_*")


def retro_r1():
    arm = {}
    for d in sorted(glob.glob(R1_GLOB)):
        run = os.path.basename(d).replace("ncurve_graph_", "ncurve_")
        per = {}
        for f in glob.glob(f"{d}/*.pkl"):
            try:
                with open(f, "rb") as fh:
                    rec = pickle.load(fh)
            except Exception:
                continue
            t = rec.get("target")
            md = rec.get("mol_dict")
            if not t or not isinstance(md, (dict, OrderedDict)):
                continue
            per[t] = list(md.keys())          # insertion order == expansion order
        arm[run] = per
        print(f"  R1 {run:28} {len(per):4} targets", flush=True)
    return arm


def retroagent():
    arm = {}
    root = RA_ROOT
    for ed in sorted(glob.glob(f"{root}/eval_*")):
        run = os.path.basename(ed)
        per = {}
        for td in sorted(glob.glob(f"{ed}/target_*")):
            seq, tgt = [], None
            for rd in sorted(glob.glob(f"{td}/rollout_*")):
                p = f"{rd}/transcript.json"
                if not os.path.exists(p):
                    continue
                try:
                    tr = json.load(open(p))
                except Exception:
                    continue
                tgt = tgt or tr.get("target")
                ids = {n.get("id"): n.get("smiles")
                       for n in (tr.get("graph_state") or {}).get("nodes") or []}
                for c in (tr.get("tool_calls_log") or []):
                    if c.get("name") != "predict_templates":
                        continue
                    mid = (c.get("args") or {}).get("molecule_id")
                    s = ids.get(mid)
                    if s:
                        seq.append(s)
            if tgt and seq:
                per[tgt] = seq
        arm[run] = per
        print(f"  RA {run:28} {len(per):4} targets", flush=True)
    return arm


print("extracting Retro-R1 ...", flush=True)
r1 = retro_r1()
print("extracting RetroAgent ...", flush=True)
ra = retroagent()
json.dump({"retro_r1": r1, "retroagent": ra,
           "_meta": {"ra_root": RA_ROOT, "r1_glob": R1_GLOB}}, open(OUT, "w"))
print(f"\nwrote {OUT}")
