#!/usr/bin/env python
"""Route-level axes from the per-node caches, then the Pareto front per target.

Reads pooled_<stem>.jsonl plus the node caches traj_route_score.py filled, and writes
labeled_<stem>.jsonl: every route with its feasibility/cost/shape vector, its Pareto rank
within its target, and — with --select — the front routes marked as SFT candidates.

AGGREGATION IS A CHOICE and the wrong choice makes the axis a route-length proxy. Check any
aggregation by scoring the PUBLISHED routes and seeing where they land.

  plaus_passfrac   share of steps clearing AiZynthFinder's own 0.05 cut-off. Length-free,
                   uses no bar we chose, drops no step. THE PRIMARY plausibility axis.
  plaus_min        the weakest step. REPORTED, NOT USED for selection: the filter model puts
                   a share of published reactions under 0.05, so a min falls with length and
                   the gold routes score LAST on it. That is the aggregation talking, not the
                   chemistry.
  plaus_allpass    does every step clear 0.5. A second, stricter reading. 0.5 is our bar, so
                   it is only readable against the gold row.
  plaus_median     length-neutral but blind to one fatal step: four 0.99s and one 0.001 -> 0.99.

  rt1 / rt5        share of steps the forward model recovers at rank 1 / within top 5. A
                   per-step RATE, so it cannot become route length. Read against the ceiling
                   the published steps reach, not against 100%. Pareto ranking uses rt1, the
                   stricter criterion; evaluation uses top-5.
  ord_ep/ord_jrn   share of steps with an ORD record (exact or pair). ord_jrn is the
                   circularity control — much of ORD is patent-derived — so only the ORDERING
                   transfers between the two, never the absolute rates.
  sc_descent       share of steps whose HARDEST precursor is simpler than the product. Against
                   the max, not the mean: splitting a molecule trivially lowers a mean.

  cost_usd         sum over leaves of exp(MolPrice), i.e. actual USD/mmol. None if ANY leaf is
                   unpriced — a partial sum understates cost and silently favours routes with
                   exotic inputs. `n_unpriced` says how many were missing.
  cost_logsum      sum of the raw ln-prices. Kept for comparability with the evaluation
                   `cost` column, but it is a log-product and not a price; quote cost_usd.

Pareto. Within one target, a route is dominated if another is at least as good on every
objective and strictly better on one. Fronts are peeled repeatedly, so `pareto_rank` 1 is the
front, 2 the front of what is left, and so on. Routes missing any objective are excluded from
the ranking (pareto_rank null) rather than defaulted — a default is an invented value that
would put an unscored route on the front.

Objective sets (`--objectives`):
  feas       plaus_passfrac^  rt1^  ord_ep^                    pure step quality
  feas_cost  plaus_passfrac^  rt1^  cost_usd v                 the three synthesis objectives
  full       plaus_passfrac^  rt1^  ord_ep^  cost_usd v  n_steps v
  Or a custom spec: "plaus_passfrac:max,cost_usd:min".

Selection for the training corpus: the three objectives, the first two Pareto fronts as the
candidate pool, and up to eight routes per target with distinct starting-material sets:
  python scripts/traj_route_label.py --stem paroutes --objectives feas_cost \
      --diverse --pool-rank 2 --select 8
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import traj_route_common as C


from rdkit import Chem, DataStructs, RDLogger                    # noqa: E402
from rdkit.Chem import rdFingerprintGenerator                    # noqa: E402

RDLogger.DisableLog("rdApp.*")

CUT, STRONG = 0.05, 0.5

OBJ_SETS = {
    "feas": [("plaus_passfrac", "max"), ("rt1", "max"), ("ord_ep", "max")],
    "feas_cost": [("plaus_passfrac", "max"), ("rt1", "max"), ("cost_usd", "min")],
    "full": [("plaus_passfrac", "max"), ("rt1", "max"), ("ord_ep", "max"),
             ("cost_usd", "min"), ("n_steps", "min")],
}


def load_caches(rt_cache):
    def rd(name):
        p = C.NODE_SCORES / f"{name}.json"
        return json.load(open(p)) if p.exists() else {}
    sc = {"plausibility": rd("plausibility"), "roundtrip": rd(rt_cache),
          "ord": rd("ord"), "ord_journal": rd("ord_journal"),
          "scscore": rd("scscore"), "molprice": rd("molprice")}
    for k, v in sc.items():
        print(f"  cache {k:14} {len(v)} entries", flush=True)
    return sc


def step_tags(steps, sc):
    """Per-STEP and per-LEAF tags, kept beside the route aggregate.

    The aggregate is what selection ranks on; these are what a reasoning trace can cite. A
    step's CoT can say "the filter policy gives this disconnection 0.87 and ORD has an exact
    record of it" only if the per-step number survived aggregation, and no aggregate can be
    inverted back into one. Distinguish the two kinds of missing value: `rt: null` WITH the key
    present means the forward model did not recover the product (a result); `rt_scored: false`
    means it was never asked (not a result).
    """
    out = []
    for p, rs in steps:
        k = C.rxn_key(p, rs)
        vp = sc["scscore"].get(p)
        vr = [sc["scscore"].get(x) for x in rs]
        d = None
        if vp is not None and all(v is not None for v in vr) and vr:
            d = round(vp - max(vr), 4)
        out.append({
            "rxn_key": k, "product": p, "reactants": list(rs),
            "plaus": sc["plausibility"].get(k),
            "rt": sc["roundtrip"].get(k),
            "rt_scored": k in sc["roundtrip"],
            "ord": sc["ord"].get(k),
            "ord_jrn": sc["ord_journal"].get(k),
            "sc_product": vp, "sc_max_reactant": (max(vr) if vr and
                                                  all(v is not None for v in vr) else None),
            "sc_delta": d,
            "bimol": int(len(set(rs)) >= 2),
        })
    return out


def leaf_tags(steps, sc):
    lv = C.leaves(steps)
    return [{"smiles": m, "molprice_ln": sc["molprice"].get(m),
             "molprice_usd_mmol": (round(math.exp(sc["molprice"][m]), 4)
                                   if sc["molprice"].get(m) is not None else None),
             "scscore": sc["scscore"].get(m)} for m in lv]


def route_axes(steps, sc, target):
    """-> dict of every route-level axis, with None where the inputs are not all present."""
    keys = [C.rxn_key(p, rs) for p, rs in steps]
    pl = [sc["plausibility"].get(k) for k in keys]
    rt = [sc["roundtrip"].get(k) for k in keys]
    oa = [sc["ord"].get(k) for k in keys]
    oj = [sc["ord_journal"].get(k) for k in keys]
    lv = C.leaves(steps)
    pr = [sc["molprice"].get(m) for m in lv]

    out = {"n_steps": len(steps), "n_leaves": len(lv)}

    pl_ok = [x for x in pl if x is not None]
    out["n_steps_scored_plaus"] = len(pl_ok)
    if len(pl_ok) == len(steps) and pl_ok:
        out["plaus_passfrac"] = round(sum(1 for x in pl_ok if x >= CUT) / len(pl_ok), 4)
        out["plaus_min"] = round(min(pl_ok), 6)
        out["plaus_median"] = round(st.median(pl_ok), 6)
        out["plaus_allpass"] = int(min(pl_ok) >= STRONG)
    else:
        out.update(plaus_passfrac=None, plaus_min=None, plaus_median=None,
                   plaus_allpass=None)

    # A cached None means "the forward model did not recover it", which is a RESULT; an
    # absent key means "not scored", which is not. The two must not be conflated, so
    # membership is counted rather than non-nullness.
    n_rt = sum(1 for k in keys if k in sc["roundtrip"])
    out["n_steps_scored_rt"] = n_rt
    if n_rt == len(steps) and steps:
        out["rt1"] = round(sum(1 for x in rt if x == 1) / len(rt), 4)
        out["rt5"] = round(sum(1 for x in rt if x is not None and x <= 5) / len(rt), 4)
    else:
        out.update(rt1=None, rt5=None)

    def ep(v):
        got = [x for x in v if x is not None]
        if len(got) != len(steps) or not got:
            return None
        return round(sum(1 for x in got if x in ("exact", "pair")) / len(got), 4)
    out["ord_ep"] = ep(oa)
    out["ord_jrn_ep"] = ep(oj)
    out["ord_exact"] = (None if any(x is None for x in oa) or not oa
                        else round(sum(1 for x in oa if x == "exact") / len(oa), 4))

    out["n_unpriced"] = sum(1 for x in pr if x is None)
    if lv and out["n_unpriced"] == 0:
        out["cost_usd"] = round(float(sum(math.exp(x) for x in pr)), 4)
        out["cost_logsum"] = round(float(sum(pr)), 4)
    else:
        out.update(cost_usd=None, cost_logsum=None)

    # SCScore descent from the cached molecule values, so no ONNX session is needed here.
    good = n = 0
    for p, rs in steps:
        vp = sc["scscore"].get(p)
        vr = [sc["scscore"].get(x) for x in rs]
        if vp is None or any(v is None for v in vr):
            continue
        n += 1
        good += int(max(vr) < vp)
    out["sc_descent"] = round(good / n, 4) if n else None
    out["n_steps_scored_sc"] = n

    try:
        lls = C.longest_linear_sequence([(p, list(rs)) for p, rs in steps], target)
    except Exception:                                           # noqa: BLE001
        lls = None
    out["lls"] = lls
    out["convergence"] = round(len(steps) / lls, 4) if lls else None
    out["bimol"] = round(sum(1 for _, rs in steps if len(set(rs)) >= 2) / len(steps), 4)
    return out


def pareto_ranks(rows, objectives):
    """Successive non-dominated fronts. Rows missing an objective get rank None."""
    names = [n for n, _ in objectives]
    idx = [i for i, r in enumerate(rows)
           if all(r.get(n) is not None for n in names)]
    ranks = [None] * len(rows)

    def vec(i):
        return [(-rows[i][n] if d == "max" else rows[i][n]) for n, d in objectives]

    remaining = list(idx)
    rank = 1
    while remaining:
        V = {i: vec(i) for i in remaining}
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if j == i:
                    continue
                a, b = V[j], V[i]
                if all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b)):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        if not front:                        # all mutually equal: one flat front
            front = list(remaining)
        for i in front:
            ranks[i] = rank
        remaining = [i for i in remaining if i not in set(front)]
        rank += 1
    return ranks


def parse_objectives(spec):
    if spec in OBJ_SETS:
        return OBJ_SETS[spec]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"objective {part!r} needs :max or :min")
        n, d = part.split(":", 1)
        if d not in ("max", "min"):
            raise SystemExit(f"objective direction {d!r} must be max or min")
        out.append((n, d))
    if not out:
        raise SystemExit(f"empty objective spec; use one of {sorted(OBJ_SETS)}")
    return out



def _leaf_key(smis, _c={}):
    """Leaf set as a chemistry key, not a string key.

    Selecting on raw SMILES sets counts `Ar-Cl` / `Ar-Br` / `Ar-OH` and E/Z isomers as three
    different starting materials, and the front is full of exactly that: routes that differ
    only in a leaving group or a double-bond geometry make up much of the apparent leaf
    diversity. InChIKey's first block is connectivity only, so it
    folds the stereoisomers; the leaving-group pairs survive it and are folded by the Tanimoto
    pass in `pick_diverse`.
    """
    out = []
    for x in smis:
        k = _c.get(x)
        if k is None:
            m = Chem.MolFromSmiles(x)
            k = Chem.MolToInchiKey(m)[:14] if m else x
            _c[x] = k
        out.append(k)
    return frozenset(out)


def _fp(smi, _c={}):
    f = _c.get(smi)
    if f is None:
        m = Chem.MolFromSmiles(smi)
        f = _c[smi] = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                       .GetFingerprint(m) if m else None)
    return f


def _near_dup(a, b, cut=0.9):
    """True if every leaf of the smaller set has a >=cut Tanimoto partner in the other.

    This is what catches the leaving-group pairs InChIKey14 lets through.
    """
    A, B = sorted(a), sorted(b)
    if len(A) != len(B):
        return False
    used = set()
    for x in A:
        fx = _fp(x)
        hit = None
        for j, y in enumerate(B):
            if j in used:
                continue
            fy = _fp(y)
            if fx is not None and fy is not None and DataStructs.TanimotoSimilarity(fx, fy) >= cut:
                hit = j
                break
        if hit is None:
            return False
        used.add(hit)
    return True


def pick_diverse(rows, n, objectives, pool_rank=2):
    """-> up to `n` front routes chosen for LEAF DIVERSITY, not just for rank order.

    Taking the top `n` of the Pareto front by objective order does not give diverse starting
    materials: the first front alone tends to concentrate on routes with similar inexpensive
    starting-material sets. `cost_usd` is a sum over leaf prices, so it actively pulls the
    front toward one cheap set of starting materials; the second front broadens it.

    So: take Pareto ranks 1..pool_rank as the CANDIDATE pool (rank 1 alone is often smaller
    than `n`), order it by the objectives, then walk it greedily and keep a route only if its
    leaf set is new — new by InChYKey14 identity and not a Tanimoto near-duplicate of one
    already kept. If fewer than `n` survive, top up in objective order so the count is stable.
    """
    pool = [r for r in rows if (r.get("pareto_rank") or 99) <= pool_rank]
    pool.sort(key=lambda r: ((r.get("pareto_rank") or 99),) + tuple(
        (-r[k] if d == "max" else r[k]) for k, d in objectives) + (r["first_rank"],))
    kept, keys = [], []
    for r in pool:
        k = _leaf_key(r["leaves"])
        if k in keys or any(_near_dup(k, o) for o in keys):
            continue
        kept.append(r)
        keys.append(k)
        if len(kept) >= n:
            break
    if len(kept) < n:                      # if diversity can't fill n, top up by rank
        have = {id(r) for r in kept}
        for r in pool:
            if id(r) not in have:
                kept.append(r)
                if len(kept) >= n:
                    break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="fusionretro")
    ap.add_argument("--objectives", default="feas")
    ap.add_argument("--rt-cache", default="roundtrip_rt5",
                    choices=("roundtrip", "roundtrip_rt5"))
    ap.add_argument("--diverse", action="store_true",
                    help="select for leaf-set diversity instead of pure objective order — "
                         "see pick_diverse(). Off by default so plain objective-order "
                         "selection stays reproducible")
    ap.add_argument("--pool-rank", type=int, default=2,
                    help="with --diverse, Pareto ranks 1..N form the candidate pool")
    ap.add_argument("--select", type=int, default=0,
                    help="mark up to N front routes per target as sft_candidate and write "
                         "sft_candidates_<stem>.jsonl")
    ap.add_argument("--no-per-step", action="store_true",
                    help="omit the per-step / per-leaf tag arrays. They make the file several "
                         "times larger; they are also the only form a reasoning trace can cite")
    ap.add_argument("--pooled", default=None)
    ap.add_argument("--targets", default=None,
                    help="target jsonl for the gold reference row (default targets_<stem>.jsonl)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    objectives = parse_objectives(a.objectives)
    print(f"objectives: " + ", ".join(f"{n} {d}" for n, d in objectives))
    sc = load_caches(a.rt_cache)

    pooled = a.pooled or str(C.OUT / f"pooled_{a.stem}.jsonl")
    if not os.path.exists(pooled):
        raise SystemExit(f"{pooled} missing — run traj_route_pool.py first")
    rows = list(C.read_jsonl(pooled))
    print(f"{len(rows)} pooled routes", flush=True)

    for i, r in enumerate(rows):
        steps = C.norm_steps(r["steps"])
        r.update(route_axes(steps, sc, r["target"]))
        if not a.no_per_step:
            r["step_scores"] = step_tags(steps, sc)
            r["leaf_scores"] = leaf_tags(steps, sc)
        if (i + 1) % 20000 == 0:
            print(f"  scored {i+1}/{len(rows)}", flush=True)

    # gold, as the reference row. Not a candidate — a scale, and a scale is only a scale on the
    # SAME targets: the target file can hold far more targets than the pool, and a gold
    # row averaged over all of them is a statement about a different population.
    per_targets = {r["target"] for r in rows}
    tp = Path(a.targets or (C.OUT / f"targets_{a.stem}.jsonl"))
    gold_rows = []
    if tp.exists():
        for rec in C.read_jsonl(tp):
            if rec["target"] not in per_targets:
                continue
            for s in rec["gold_routes"]:
                steps = C.norm_steps(s)
                g = {"target": rec["target"], "ds": a.stem, "is_gold": True,
                     "steps": [[p, rs] for p, rs in steps], "sources": [["gold", "-", "-"]]}
                g.update(route_axes(steps, sc, rec["target"]))
                if not a.no_per_step:
                    g["step_scores"] = step_tags(steps, sc)
                    g["leaf_scores"] = leaf_tags(steps, sc)
                gold_rows.append(g)

    per = collections.defaultdict(list)
    for r in rows:
        per[r["target"]].append(r)
    n_ranked = 0
    for t, rs in per.items():
        for r, k in zip(rs, pareto_ranks(rs, objectives)):
            r["pareto_rank"] = k
            r["pareto_objectives"] = a.objectives
            n_ranked += k is not None
    print(f"  pareto: {n_ranked}/{len(rows)} routes had every objective scored", flush=True)

    dest = a.out or str(C.OUT / f"labeled_{a.stem}.jsonl")
    C.write_jsonl(dest, rows)
    if gold_rows:
        C.write_jsonl(str(C.OUT / f"labeled_gold_{a.stem}.jsonl"), gold_rows)

    # ---------------------------------------------------------------- summary
    def col(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return st.median(v) if v else None

    def fmt(x, pct=False):
        if x is None:
            return "—"
        return f"{100*x:.1f}%" if pct else f"{x:.3f}"

    AX = [("plaus_passfrac", True), ("plaus_allpass", True), ("plaus_min", False),
          ("rt1", True), ("rt5", True), ("ord_ep", True), ("ord_jrn_ep", True),
          ("sc_descent", True), ("cost_usd", False), ("n_steps", False), ("lls", False)]
    groups = [("gold (reference)", gold_rows),
              ("pool: all", rows),
              ("pool: pareto front", [r for r in rows if r.get("pareto_rank") == 1])]
    for m in sorted({s[0] for r in rows for s in r["sources"]}):
        groups.append((f"pool: {m}", [r for r in rows
                                      if any(s[0] == m for s in r["sources"])]))
    for al in sorted({s[1] for r in rows for s in r["sources"]}):
        groups.append((f"pool: {al}", [r for r in rows
                                       if any(s[1] == al for s in r["sources"])]))

    w = max(len(g) for g, _ in groups) + 1
    head = f"{'group':{w}}{'n':>7}" + "".join(f"{k[:10]:>12}" for k, _ in AX)
    print("\n" + head)
    print("-" * len(head))
    for name, rs in groups:
        if not rs:
            continue
        print(f"{name:{w}}{len(rs):7}"
              + "".join(f"{fmt(col(rs, k), p):>12}" for k, p in AX))
    print("\nmedian over routes. Read every row against `gold (reference)`, not against 100%: "
          "published chemistry does not clear every bar either.")
    print("plaus_min is printed and NOT used for selection — it is a route-length proxy on "
          "which the gold routes score last.")

    if a.select:
        cand = []
        for t, rs in per.items():
            if a.diverse:
                keep = pick_diverse(rs, a.select, objectives, a.pool_rank)
            else:
                front = [r for r in rs if r.get("pareto_rank") == 1]
                # Tie-break inside the front by the objectives in order, then prefer a route
                # the search actually RETURNED (low first_rank) over one it merely reached.
                front.sort(key=lambda r: tuple(
                    (-r[n] if d == "max" else r[n]) for n, d in objectives) + (r["first_rank"],))
                keep = front[:a.select]
            for r in keep:
                r["sft_candidate"] = True
                cand.append(r)
        out = str(C.OUT / f"sft_candidates_{a.stem}.jsonl")
        C.write_jsonl(out, cand)
        print(f"\nselected {len(cand)} routes on {len({r['target'] for r in cand})} targets "
              f"-> {out}")
    print(f"wrote {len(rows)} -> {dest}")


if __name__ == "__main__":
    sys.exit(main())
