#!/usr/bin/env python
"""Pool every search run into one deduplicated route table, and say what dedup cost.

Input: data/route_search/runs/<stem>__<model>__<algo>__b<budget>__d<draw>.jsonl
Output: data/route_search/pooled_<stem>.jsonl — one record per DISTINCT route, carrying every
        (model, algo, draw) that produced it.

The dedup key is the decision this file exists to make explicit, because the two candidates
give very different dataset sizes and only one of them is chemistry:

  seq  the ORDERED step tuple. The default for this dataset: two routes whose steps differ
       in order are two routes, for diversity. The caveat has to travel with the number —
       a retrosynthesis route is a TREE, so there is no intrinsic step order; what
       `seq` actually distinguishes is the order `iter_routes_cost_order` happened to walk the
       graph in. Two `seq`-distinct routes with the same `set` are the SAME synthesis plan
       written down twice.
  set  frozenset of (product, sorted reactants) — `three_tables.rkey`, the key evaluation
       uses. Order-blind, so one plan is one row.

Both are always computed and both counts are always printed, so the inflation factor
(n_seq / n_set) is visible rather than assumed. `--key set` is available and is what to use
when the pooled set feeds a comparison against evaluation.

What else is recorded per route, because it cannot be recovered later:
  sources       [(model, algo, draw)] — which sampler/policy found it. This is the data for
                "does localretro find routes rsmiles cannot", and it is lost
                the moment the runs are concatenated without it.
  first_rank    the best position the route held in any run's cost-ordered enumeration. A
                route ranked first by Retro* is what that algorithm would have RETURNED; a route
                far down the enumeration is something it merely reached. A Pareto front built
                from such routes is not a statement about the algorithm.
  is_gold       the route equals one of the target's published routes (set key).
  gold_prefix   share of the target's gold steps this route contains.

Usage:
  python scripts/traj_route_pool.py --stem paroutes
  python scripts/traj_route_pool.py --stem paroutes --key set
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import traj_route_common as C


def h(key) -> str:
    """Stable content hash. Python's hash() is salted per process, so it cannot be written to
    disk as an identifier — two poolings of the same route would get different `set_hash`."""
    if isinstance(key, frozenset):
        body = "|".join(sorted(f"{p}>>{'.'.join(rs)}" for p, rs in key))
    else:
        body = "|".join(f"{p}>>{'.'.join(rs)}" for p, rs in key)
    return hashlib.blake2b(body.encode(), digest_size=12).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="fusionretro")
    ap.add_argument("--key", default="seq", choices=("seq", "set"),
                    help="dedup key. seq = order-sensitive (default), set = order-blind")
    ap.add_argument("--runs", default=None, help="glob over run jsonl (default all for --stem)")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tpath = a.targets or str(C.OUT / f"targets_{a.stem}.jsonl")
    gold, gold_steps = {}, {}
    if os.path.exists(tpath):
        for r in C.read_jsonl(tpath):
            gold[r["target"]] = {C.set_key([(p, rs) for p, rs in s]) for s in r["gold_routes"]}
            gold_steps[r["target"]] = [{(p, tuple(sorted(rs))) for p, rs in s}
                                       for s in r["gold_routes"]]
    else:
        print(f"(no target file at {tpath}; is_gold / gold_prefix will be null)")

    pattern = a.runs or str(C.RUNS / f"{a.stem}__*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no run files matched {pattern}")

    # target -> dedup key -> record
    pool: dict[str, dict] = {}
    st = collections.Counter()
    trunc = collections.Counter()
    per_source = collections.Counter()
    src_unique = collections.Counter()          # routes ONLY this source found
    for f in files:
        base = os.path.basename(f)[:-6]
        parts = base.split("__")
        model, algo, draw = parts[1], parts[2], parts[-1]
        # Runs made before `max_routes` was recorded per record still know their cap: the
        # sidecar keeps the argv. Without this the saturation check cannot fire and a capped
        # run reads as a complete one.
        side_mr = None
        sp = f.replace(".jsonl", ".json")
        if os.path.exists(sp):
            av = (json.load(open(sp)) or {}).get("argv") or []
            if "--max-routes" in av:
                try:
                    side_mr = int(av[av.index("--max-routes") + 1])
                except (IndexError, ValueError):
                    side_mr = None
            elif av:
                side_mr = 200          # the default in force when argv omitted the flag
        for rec in C.read_jsonl(f):
            st["records"] += 1
            # A capped extraction is a cap on the pool the Pareto front is chosen from, so it
            # is reported per cell rather than left to be inferred from a round n_routes.
            # Runs made before routes_truncated existed are detected by saturation instead.
            mr = rec.get("max_routes") or side_mr
            if rec.get("routes_truncated"):
                trunc[(base, rec["routes_truncated"])] += 1
            elif mr and rec.get("n_routes") == mr:
                trunc[(base, f"saturated@{mr}")] += 1
            t = rec["target"]
            for rank, steps in enumerate(rec.get("routes") or []):
                steps = C.norm_steps(steps)
                if len(steps) < a.min_steps:
                    continue
                if a.max_steps and len(steps) > a.max_steps:
                    st["over_max_steps"] += 1
                    continue
                st["route_instances"] += 1
                sk = C.set_key(steps)
                qk = C.seq_key(steps)
                k = qk if a.key == "seq" else sk
                d = pool.setdefault(t, {})
                e = d.get(k)
                if e is None:
                    e = d[k] = {"target": t, "steps": [[p, rs] for p, rs in steps],
                                "n_steps": len(steps), "leaves": C.leaves(steps),
                                "sources": [], "first_rank": rank,
                                "set_hash": h(sk), "seq_hash": h(qk)}
                    st["distinct"] += 1
                e["first_rank"] = min(e["first_rank"], rank)
                s = [model, algo, draw]
                if s not in e["sources"]:
                    e["sources"].append(s)
                per_source[(model, algo)] += 1

    # A route's uniqueness is only knowable after every file is read.
    out = []
    n_set = 0
    for t, d in pool.items():
        setkeys = collections.Counter()
        for e in d.values():
            setkeys[e["set_hash"]] += 1
        n_set += len(setkeys)
        for e in d.values():
            models = sorted({s[0] for s in e["sources"]})
            algos = sorted({s[1] for s in e["sources"]})
            if len(models) == 1:
                src_unique[models[0]] += 1
            steps = [(p, rs) for p, rs in e["steps"]]
            sk = C.set_key(steps)
            g = gold.get(t)
            gp = None
            if gold_steps.get(t):
                mine = {(p, tuple(sorted(rs))) for p, rs in steps}
                gp = max((len(mine & gs) / len(gs)) for gs in gold_steps[t])
            e.update(ds=a.stem, n_models=len(models), models=models, algos=algos,
                     n_sources=len(e["sources"]),
                     seq_dupes_of_set=setkeys[e["set_hash"]],
                     is_gold=(None if g is None else bool(sk in g)),
                     gold_prefix=(None if gp is None else round(gp, 4)))
            out.append(e)

    out.sort(key=lambda e: (e["target"], e["first_rank"], e["n_steps"]))
    dest = a.out or str(C.OUT / f"pooled_{a.stem}.jsonl")
    n = C.write_jsonl(dest, out)

    print(f"\n{len(files)} run files, {st['records']} target records, "
          f"{st['route_instances']} route instances")
    print(f"  distinct by {a.key}: {n}     distinct by set: {n_set}     "
          f"inflation seq/set: {n / max(n_set, 1):.2f}x")
    print(f"  targets with >=1 route: {len(pool)}")
    if gold:
        ng = sum(1 for e in out if e["is_gold"])
        tg = len({e["target"] for e in out if e["is_gold"]})
        print(f"  gold routes recovered: {ng} routes on {tg} targets "
              f"({100*tg/max(len(pool),1):.1f}% of targets with a route)")
    if trunc:
        print("\n  TRUNCATED extractions (routes past the cap were dropped, in cost order):")
        for (b, why), c in sorted(trunc.items(), key=lambda x: -x[1]):
            print(f"    {b:52} {why:18} {c:5} targets")
        print("    -> routes/target and every diversity count on those targets report the "
              "CAP, not the graph. Re-run with a larger --max-routes to lift it.")
    print("\n  routes per (model, algo):")
    for (m, al), c in sorted(per_source.items(), key=lambda x: -x[1]):
        print(f"    {m:10} {al:16} {c:8}")
    print("\n  routes found by ONE model only (the diversity that model contributes alone):")
    for m, c in sorted(src_unique.items(), key=lambda x: -x[1]):
        print(f"    {m:10} {c:8} ({100*c/max(n,1):.1f}% of the pool)")
    print(f"\nwrote {n} -> {dest}")
    with open(dest.replace(".jsonl", ".json"), "w") as fh:
        json.dump({"stem": a.stem, "key": a.key, "argv": sys.argv, "files": files,
                   "n_distinct": n, "n_distinct_set": n_set,
                   "n_targets": len(pool), "stats": dict(st),
                   "per_source": {f"{m}|{al}": c for (m, al), c in per_source.items()},
                   "unique_by_model": dict(src_unique),
                   "truncated": {f"{b}|{w}": c for (b, w), c in trunc.items()}}, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
