#!/usr/bin/env python
"""Atom-conservation validity and route diversity per arm, every arm scored on one stock.

`solve@emols` -- every arm is re-checked against `Stock("emols")` rather than trusting the
    verdict each arm recorded, since arms may have answered `done` against a different
    catalogue. A purchasability column is only comparable if one predicate produced all of it.

`valid@k` -- a target counts when one of the arm's first k routes both conserves atoms and
    closes on the stock. Solve alone only asks whether the leaves are purchasable, which a
    generative single-step model can satisfy by dropping part of the molecule.

Diversity is compared at matched k. Distinct-disconnection counts grow with the number of
routes an arm returned, so an unmatched diversity column would mostly measure route count.
k truncates every arm to the same depth in its own preference order (the order the file
carries), and the realised pool size is printed next to every row so a row that cannot reach
k is visible.

    disconnection   identity = (product, frozenset(precursors)). Counted over distinct
                    disconnections, never over steps: reuse of one reaction is not breadth.
    reaction class  from `cache/node_scores/named_reaction.json`, restricted to tier
                    `applies+makes`. The `applies` tier only says a template's groups are
                    present and its names need not describe one transform, so it is not a
                    class label. `applies+makes` names are synonyms of one transform, so the
                    whole name set is the class id. Coverage is reported, because a class
                    count over labelled steps alone would favour an arm whose chemistry the
                    labeller happens to recognise.

Class entropy measures spread, not chemical quality, and it can move with route length, so
route length is printed beside the diversity columns.

Usage:
  python scripts/feas_arm_diversity.py --k 1
  python scripts/feas_arm_diversity.py --k 3 --matched
  python scripts/feas_arm_diversity.py --k 0            # 0 = the whole pool
  python scripts/feas_arm_diversity.py --arm 'LABEL=pooled:<stem>'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from collections import Counter, defaultdict

import traj_route_common as C


from route_validity import formula                                        # noqa: E402

NAMED = C.NODE_SCORES / "named_reaction.json"
# Side cache of the --label labels (one name per disconnection). The node_scores cache above
# was built over a different step population and stores a name set per step, so mixing the
# two would make the class id mean different things in different rows.
SIDE = C.OUT / "feas" / "named_arm_diversity.json"

# per-arm three-tier counts, filled by main() and printed after the table
TIERS: dict = {}
PERT: dict = {}
BANDS: dict = {}
BAND_EDGES = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 99))

# label -> (kind, path-or-glob). `search` rows carry a pool per target; `pooled` rows are one
# route each and are grouped by target, in file order.
ARMS: list[tuple[str, str, str]] = [
    ("MCTS",          "search", "data/route_search/runs/uspto190__rsmiles__mcts__b300__d0.jsonl"),
    ("Retro*-0",      "search", "data/route_search/runs/uspto190__rsmiles__retrostar__b300__d0.jsonl"),
    ("Retro*",        "search", "data/route_search/runs/uspto190__rsmiles__retrostar-value__b300__d0.jsonl"),
    ("MORetro*",      "search", "data/route_search/runs/uspto190__rsmiles__moretro-bo-sobol__b300__d0.shard*of12.jsonl"),
    ("RetroAgent",    "pooled", "data/route_search/pooled_retroagent_rsmiles.jsonl"),
    ("Retro-R1",      "pooled", "data/route_search/pooled_retro_r1_rsmiles_t0.jsonl"),
    ("RetroPlanner",  "board",  "results/forced_iter_retroplanner_emols_div_t06_b300_ep40.jsonl"),
]


def steps_of(route) -> list[tuple[str, list[str]]]:
    return [(p, sorted(rs)) for p, rs in route]


def load_arm(kind: str, pattern: str) -> dict[str, list]:
    """target -> [route, ...] in the arm's own order, de-duplicated by step set."""
    pool: dict[str, list] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    files = sorted(glob.glob(pattern))
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("target_in_stock"):
                continue
            t = r["target"]
            if kind == "pooled":                  # one row IS one route
                routes = [r["steps"]]
            elif kind == "board":
                # `routes` is what the run reported; `routes_union` is the pool it accumulated
                # across episodes. Both are needed and in this order: k=1 must see the reported
                # route, larger k the pool. The de-duplication below collapses the overlap.
                routes = [x["steps"] for x in ((r.get("routes") or [])
                                               + (r.get("routes_union") or []))
                          if isinstance(x, dict) and x.get("steps")]
            else:                                  # search run: routes[] are step lists
                routes = r.get("routes") or []
            for raw in routes:
                s = steps_of(raw)
                key = frozenset((p, tuple(rs)) for p, rs in s)
                if not s or key in seen[t]:
                    continue
                seen[t].add(key)
                pool[t].append(s)
    return pool


def conserves(steps) -> bool:
    """Every step's product elements supplied by its precursors. An unparseable molecule is a
    missing measurement, not evidence of conjured atoms, so its step is skipped."""
    n = 0
    for prod, kids in steps:
        fp = formula(prod)
        if fp is None:
            continue
        fc: Counter = Counter()
        ok = True
        for x in kids:
            f = formula(x)
            if f is None:
                ok = False
                break
            fc += f
        if not ok:
            continue
        n += 1
        if any(fc[e] < c for e, c in fp.items()):
            return False
    return n > 0


def leaves_of(steps):
    prods = {p for p, _ in steps}
    return sorted({x for _, rs in steps for x in rs if x not in prods})


def disc_key(prod, reacts):
    return (prod, frozenset(reacts))


def rxn_key(prod, reacts):
    return f"{prod}>>{'.'.join(sorted(reacts))}"


def entropy(counts) -> tuple[float, float]:
    """(normalised entropy, effective number of classes = exp(raw entropy)).

    The normalised form divides by log(n classes), so an arm with more classes is scored
    against a harder denominator and the column partly measures class count. `exp(H_raw)` is the effective number of equally-common classes and is
    directly comparable across arms, so both are reported."""
    n = sum(counts)
    if n <= 0 or len(counts) <= 1:
        return 0.0, float(len(counts))
    h = -sum((c / n) * math.log(c / n) for c in counts if c)
    return h / math.log(len(counts)), math.exp(h)


def label_disconnections(pools, okmap, kk, universe):
    """Tier every disconnection that the table will count, from the side cache.

    The labels were computed by a named-reaction labeller this repository does not carry, so
    a disconnection missing from the side cache cannot be labelled here and stops the run.
    """

    want: dict[str, tuple] = {}
    for label in pools:
        tgts = universe if universe is not None else sorted(pools[label])
        for t in tgts:
            rs, vs = pools[label].get(t, []), okmap[label].get(t, [])
            head = list(zip(rs, vs))[:kk] if kk else list(zip(rs, vs))
            for s, v in head:
                if not v:
                    continue
                for prod, reacts in s:
                    want[rxn_key(prod, reacts)] = (prod, list(reacts))

    side = json.load(open(SIDE)) if SIDE.exists() else {}
    todo = [k for k in want if k not in side]
    print(f"\nlabelling: {len(want):,} distinct disconnections, {len(side):,} side-cached, "
          f"{len(todo):,} to compute", flush=True)
    if todo:
        raise SystemExit(f"{len(todo):,} disconnections are not in {SIDE}; named-reaction "
                         "labels are cache-only here")
    got = {k: side[k] for k in want if k in side}
    tiers = Counter(v["tier"] for v in got.values())
    print(f"  tiers: {dict(tiers)}\n", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=1,
                    help="routes kept per target, in the arm's own order; 0 = whole pool")
    ap.add_argument("--matched", action="store_true",
                    help="restrict to the targets EVERY selected arm closed on the stock")
    ap.add_argument("--stock", default="emols")
    ap.add_argument("--arm", action="append", default=[],
                    help="LABEL=kind:stem  (kind in search|pooled|board; stem resolved under "
                         "data/route_search/pooled_<stem>.jsonl for pooled)")
    ap.add_argument("--only", default=None, help="comma-separated subset of labels")
    ap.add_argument("--by-length", action="store_true",
                    help="compare classes-per-step INSIDE step-count bands, so the "
                         "comparison is not confounded by route length")
    ap.add_argument("--label", action="store_true",
                    help="read the named-reaction tier of every disconnection from the "
                         "side cache instead of the sparse node_scores cache")
    a = ap.parse_args()

    arms = list(ARMS)
    for spec in a.arm:
        label, rhs = spec.split("=", 1)
        kind, stem = rhs.split(":", 1)
        path = (stem if "/" in stem
                else f"data/route_search/pooled_{stem}.jsonl")
        arms.append((label, kind, path))
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        arms = [x for x in arms if x[0] in keep]

    stock = C.Stock(a.stock)
    print(f"stock {json.dumps(stock.describe())}", flush=True)
    named = {}
    if not a.label:
        named = json.load(open(NAMED))
        print(f"named_reaction {len(named):,} entries "
              f"({sum(1 for v in named.values() if v.get('tier') == 'applies+makes'):,} "
              f"at applies+makes)\n", flush=True)

    # Pass 1: pools, and the gate verdict per route. Cached per route so k can be varied
    # without re-running rdkit.
    pools: dict[str, dict[str, list]] = {}
    okmap: dict[str, dict[str, list[bool]]] = {}
    for label, kind, pattern in arms:
        pool = load_arm(kind, pattern)
        pools[label] = pool
        # The two predicates are kept apart: their conjunction is what the table selects on,
        # but a route lost to atom conservation is not a purchasability failure, so the two
        # halves are printed separately and neither is read as the other.
        buy = {t: [all(stock.has(x) for x in leaves_of(s)) for s in rs]
               for t, rs in pool.items()}
        cons = {t: [conserves(s) for s in rs] for t, rs in pool.items()}
        okmap[label] = {t: [b and c for b, c in zip(buy[t], cons[t])] for t in pool}
        n_buy = sum(1 for v in buy.values() if any(v))
        n_cons = sum(1 for v in cons.values() if any(v))
        n_both = sum(1 for v in okmap[label].values() if any(v))
        print(f"  {label:14} {len(pool):3} targets with routes | "
              f"in {a.stock} {n_buy:3} | conserving {n_cons:3} | both {n_both:3}", flush=True)

    universe = None
    if a.matched:
        sets = [{t for t, v in okmap[l].items() if any(v)} for l, _, _ in arms]
        universe = set.intersection(*sets) if sets else set()
        print(f"\nmatched (closed by every arm): {len(universe)} targets")

    kk = a.k if a.k > 0 else None

    if a.label:
        named = label_disconnections(pools, okmap, kk,
                                     sorted(universe) if universe is not None else None)

    hdr = (f"\n{'arm':14} {'pool':>6} {'solve':>6} {'valid':>6} {'valid%':>7} {'len':>5} "
           f"{'steps':>6} {'uniq':>6} {'ovl%':>6} {'reuse%':>7} {'disc/tgt':>9} "
           f"{'class':>6} {'cov':>6} {'H':>6} {'effN':>6} {'top1':>6}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for label, _, _ in arms:
        pool, ok = pools[label], okmap[label]
        tgts = sorted(universe) if universe is not None else sorted(pool)
        nsolve = nvalid = 0
        realised, lens, discs, nsteps = [], [], Counter(), 0
        per_route = per_target = 0
        # per-target class diversity: classes inside ONE target's kept routes, then averaged
        t_cls, t_cls_route, t_cls_step, t_spr = [], [], [], []
        per_route_pairs = []          # (steps, classes) for ONE route, for the length bands
        classes = Counter()
        tiers: Counter = Counter()
        labelled = 0
        for t in tgts:
            rs, vs = pool.get(t, []), ok.get(t, [])
            if not rs:
                continue
            if any(vs):
                nsolve += 1
            head = rs[:kk] if kk else rs
            hv = vs[:kk] if kk else vs
            realised.append(len(head))
            keep = [s for s, v in zip(head, hv) if v]
            if not keep:
                continue
            nvalid += 1
            tset = set()
            for s in keep:
                lens.append(len(s))
                ds = [disc_key(p, r) for p, r in s]
                nsteps += len(ds)
                per_route += len(set(ds))
                tset |= set(ds)
                for d in ds:
                    discs[d] += 1
            per_target += len(tset)
            tc = set()
            tsteps = sum(len(x) for x in keep)
            for d in tset:
                e = named.get(rxn_key(d[0], list(d[1])))
                if e and e.get("tier") == "applies+makes" and e.get("names"):
                    tc.add(frozenset(e["names"]))
            t_cls.append(len(tc))
            t_cls_route.append(len(tc) / max(len(keep), 1))
            t_cls_step.append(len(tc) / max(tsteps, 1))
            t_spr.append(tsteps / max(len(keep), 1))
            for one in keep:          # per-route, so a band holds routes not targets
                oc = set()
                for prod, reacts in one:
                    e = named.get(rxn_key(prod, reacts))
                    if e and e.get("tier") == "applies+makes" and e.get("names"):
                        oc.add(frozenset(e["names"]))
                per_route_pairs.append((len(one), len(oc)))
        # classes are counted over DISTINCT disconnections, once each
        for (prod, reacts) in discs:
            e = named.get(rxn_key(prod, list(reacts)))
            tiers[(e or {}).get("tier") or "unlabelled"] += 1
            if e and e.get("tier") == "applies+makes" and e.get("names"):
                labelled += 1
                classes[frozenset(e["names"])] += 1
        den = len(universe) if universe is not None else 190
        nd = len(discs)
        H, effN = entropy(list(classes.values()))
        TIERS[label] = tiers
        PERT[label] = (t_cls, t_cls_route, t_cls_step, t_spr)
        BANDS[label] = per_route_pairs
        print(f"{label:14} {st.mean(realised) if realised else 0:6.2f} {nsolve:6} "
              f"{nvalid:6} {100 * nvalid / max(den, 1):6.1f}% "
              f"{st.mean(lens) if lens else 0:5.2f} {nsteps:6} {nd:6} "
              f"{100 * (1 - per_target / per_route) if per_route else 0:5.1f}% "
              f"{100 * (1 - nd / per_target) if per_target else 0:6.1f}% "
              f"{nd / max(nvalid, 1):9.2f} "
              f"{len(classes):6} {labelled / max(nd, 1):6.2f} "
              f"{H:6.3f} {effN:6.1f} "
              f"{max(classes.values()) / max(labelled, 1) if classes else 0:6.3f}")

    print(f"\nWITHIN-TARGET class diversity -- the arm-wide `class` column is a union over "
          f"the\nwhole test set and grows with route length; these are normalised inside one "
          f"target.")
    print(f"{'arm':14} {'cls/target':>11} {'cls/route':>10} {'cls/step':>9}  "
          f"{'steps/route':>12}")
    for label, _, _ in arms:
        v = PERT.get(label)
        if not v or not v[0]:
            continue
        print(f"{label:14} {st.mean(v[0]):11.2f} {st.mean(v[1]):10.2f} {st.mean(v[2]):9.3f}  "
              f"{st.mean(v[3]):12.2f}")

    if a.by_length:
        print("\nINSIDE A STEP-COUNT BAND. `cls/step` can track route length across arms, "
              "so the\nper-step control is not length-free. Within a band it is. A band with "
              "fewer than\n5 routes is dropped rather than shown as a point estimate.")
        hdr2 = f"{'arm':14}" + "".join(f"{f'{lo}-{hi if hi < 99 else chr(43)}':>16}"
                                       for lo, hi in BAND_EDGES)
        print("\n" + hdr2)
        print("-" * len(hdr2))
        for label, _, _ in arms:
            row = f"{label:14}"
            for lo, hi in BAND_EDGES:
                v = [(sp, c) for sp, c in BANDS.get(label, []) if lo <= sp <= hi]
                if len(v) < 5:
                    row += f"{'-':>16}"
                else:
                    cs = st.mean(c / sp for sp, c in v)
                    row += f"{f'{cs:.3f} (n={len(v)})':>16}"
            print(row)
        print("\ncell = mean classes per step among that arm's routes in the band")

    print(f"\nTIER SPLIT over distinct disconnections "
          f"(applies = right groups, WRONG product -- the hallucination-shaped tier)")
    print(f"{'arm':14} {'distinct':>9} {'makes':>7} {'applies':>8} {'none':>6} "
          f"{'unlab':>6} | {'makes%':>7} {'applies%':>9}")
    for label, _, _ in arms:
        t = TIERS.get(label)
        if not t:
            continue
        n = sum(t.values())
        print(f"{label:14} {n:9} {t['applies+makes']:7} {t['applies']:8} {t['none']:6} "
              f"{t['unlabelled']:6} | {100 * t['applies+makes'] / max(n, 1):6.1f}% "
              f"{100 * t['applies'] / max(n, 1):8.1f}%")

    print(f"\nk = {a.k if a.k else 'all'}  ·  denominator = "
          f"{'matched subset' if universe is not None else '190 targets'}")
    print("pool      routes actually available after de-duplication and --k; a row below k\n"
          "          could not reach it and its diversity is not comparable at that k")
    print("valid     one of the first k routes conserves atoms AND closes on the stock")
    print("ovl%      the k routes of ONE target overlapping: 1 - (distinct per target) /\n"
          "          (distinct per route). 0 by construction at k=1. This is the\n"
          "          'are the alternatives actually different' number")
    print("reuse%    the SAME disconnection serving different targets: 1 - (arm-wide\n"
          "          distinct) / (sum of per-target distinct). Confounded by how many\n"
          "          targets an arm has valid -- use --matched to compare it")
    print("disc/tgt  distinct disconnections per valid target -- breadth, at matched k")
    print("class     distinct reaction classes over DISTINCT disconnections, tier\n"
          "          applies+makes only; cov = share of them the labeller recognised")
    print("H         Shannon entropy over class counts, normalised by log(n classes) --\n"
          "          partly a function of class count; effN is the comparable one")
    print("effN      effective number of equally-common classes, exp(raw entropy)")
    print("top1      largest class's share of labelled disconnections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
