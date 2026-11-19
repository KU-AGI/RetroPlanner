#!/usr/bin/env python
"""Effective route diversity for the ablation table.

WHY THIS AND NOT `routes per target`.  A target can be handed fourteen routes that are
one disconnection wearing fourteen labels.  Measured on the sequential corpus: 7.64
routes per target share 2.69 distinct first disconnections, so two thirds of the count
is variation below the first cut.  A diversity axis that cannot tell those apart
rewards a search for re-deriving the same chemistry.

So four numbers, each answering something `n_routes` cannot:

  first disconnections   how many ways the TARGET itself was taken apart.  The first cut
                         is the decision the whole route hangs off; two routes that share
                         it are variations, whatever happens below.
  pairwise Jaccard       mean step-set overlap between two routes of the same target.
                         1.0 is the same route twice.
  effective routes       routes left after collapsing everything that overlaps at or above
                         --cluster (default 0.8), single-linkage.  This is `n_routes`
                         with the relabelling removed.
  reuse                  share of distinct reactions that appear in more than one route.
                         High reuse with low first-disconnection count is the signature
                         of a search that found one tree and walked it twice.

Routes are order-blind: a route is the SET of its (product, sorted reactants) steps, the
same identity `traj_route_pool.py --key set` uses, so a step list permuted by the search
is one route here as it is there.

    python analyze_route_diversity.py ../eval_protocol/results/ladder/*_div.jsonl
    python analyze_route_diversity.py ... --curve curve.json    # routes vs call budget
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path


def step_key(step) -> tuple:
    prod, reactants = step[0], step[1]
    return (str(prod), tuple(sorted(map(str, reactants))))


def route_steps(route) -> frozenset:
    return frozenset(step_key(s) for s in (route.get("steps") or []))


def first_cut(route, target: str):
    """The disconnection applied to the target itself.

    Routes are not stored in synthetic order, so this is a lookup by product rather
    than `steps[0]` -- which would report whichever step the recorder happened to
    write first and make the axis noise.
    """
    for s in (route.get("steps") or []):
        if str(s[0]) == target:
            return tuple(sorted(map(str, s[1])))
    return None


def jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def effective_count(sets: list[frozenset], thr: float) -> int:
    """Single-linkage cluster count at `thr`.  Routes are few per target (tens), so the
    quadratic pass is cheaper than building an index for it."""
    n = len(sets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(n), 2):
        if jaccard(sets[i], sets[j]) >= thr:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return len({find(i) for i in range(n)})


def keep_route(rt: dict, max_unique: int, max_episode: int) -> bool:
    """Is this route inside the budget / restart cut?

    Both cuts matter and they answer different questions. `max_unique` caps the SEARCH
    EFFORT (distinct single-step calls) and is the axis an equal-budget ablation needs.
    `max_episode` caps the RESTARTS, and without it the diversity protocol's 40 rollouts
    dominate the route count: a single-route arm ends each rollout at `final` and so gets
    an independent route per restart, while a multi-route board carries one tree forward.
    Measured on v20 at 40 rollouts the three arms sit within 15.6-17.6 routes of each other
    and the axis says nothing; the separation is in the first few restarts.
    """
    if max_unique is not None:
        v = rt.get("found_at_unique")
        if v is None or v > max_unique:
            return False
    if max_episode is not None:
        v = rt.get("found_at_episode")
        if v is None or v >= max_episode:
            return False
    return True


def analyse(path: Path, thr: float, solved_only: bool = False,
            max_unique: int = None, max_episode: int = None) -> dict:
    n_routes, n_first, n_eff, pair, reuse, calls_per_route = [], [], [], [], [], []
    n_per_cut: list = []
    solved = total = with_routes = 0
    curve: Counter = Counter()
    for line in path.open():
        row = json.loads(line)
        total += 1
        solved += bool(row.get("solved"))
        target = str(row.get("target"))
        routes = row.get("routes_union")
        if routes is None:                      # a solve-protocol dump has no union
            routes = row.get("routes") or []
        if max_unique is not None or max_episode is not None:
            routes = [r for r in routes if keep_route(r, max_unique, max_episode)]
        sets = [route_steps(r) for r in routes]
        sets = [s for s in sets if s]
        # EVERY per-target column shares one denominator -- all targets, zero where the
        # arm found nothing.  Averaging `effective` over route-bearing targets alone while
        # `routes` averaged over all of them reported 0.65 routes at 1.11 effective, which
        # reads as more diversity than there are routes.  --solved-only switches every
        # column to the conditional view at once, rather than mixing the two.
        if solved_only and not sets:
            continue
        n_routes.append(len(sets))
        cuts = {c for c in (first_cut(r, target) for r in routes) if c}
        n_first.append(len(cuts))
        # routes = distinct-1st-disconnection x routes-per-disconnection, and the two
        # factors answer different questions: how many OPENING MOVES the arm found,
        # and how far it elaborated each one. Measured on v20_multi_par against the
        # mr_nr reference, the whole of its route advantage (12.90 vs 9.90) sits in the
        # second factor (4.27 vs 3.07); it finds FEWER openings (3.17 vs 3.33).
        if cuts:
            n_per_cut.append(len(routes) / len(cuts))
        if sets:
            with_routes += 1
            n_eff.append(effective_count(sets, thr))
            seen: Counter = Counter()
            for st in sets:
                seen.update(st)
            reuse.append(sum(1 for v in seen.values() if v > 1) / len(seen))
            c = row.get("calls_total") or row.get("calls") or 0
            if c:
                calls_per_route.append(c / len(sets))
        else:
            n_eff.append(0)
            reuse.append(0.0)
        if len(sets) > 1:
            pair.append(statistics.mean(jaccard(a, b) for a, b in combinations(sets, 2)))
        for r in routes:
            f = r.get("found_at_calls")
            if f is not None:
                curve[int(f)] += 1
    m = lambda x: statistics.mean(x) if x else float("nan")
    # `iter.jsonl` is the filename EVERY arm writes, so path.name alone labels two rows
    # identically and a table built from it silently attributes one arm's numbers to another.
    label = (f"{path.parent.name}/{path.stem}" if path.name in ("iter.jsonl", "pass1.jsonl")
             else path.name)
    return {"file": label, "targets": total, "solved": solved,
            "with_routes": with_routes,
            "solve_rate": solved / total if total else 0.0,
            "routes": m(n_routes), "first_cuts": m(n_first),
            "per_cut": m(n_per_cut),
            "effective": m(n_eff), "pairwise_jaccard": m(pair),
            "reuse": m(reuse), "calls_per_route": m(calls_per_route),
            "_curve": dict(sorted(curve.items()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-unique", type=int, default=None,
                    help="keep only routes found within this many DISTINCT single-step "
                         "calls -- the equal-search-budget cut")
    ap.add_argument("--max-episode", type=int, default=None,
                    help="keep only routes found in the first N ROLLOUTS. Without it the "
                         "40-restart diversity protocol dominates the route count and the "
                         "arms are indistinguishable on it")
    ap.add_argument("--cluster", type=float, default=0.8,
                    help="step-set overlap at or above which two routes are one (0.8)")
    ap.add_argument("--solved-only", action="store_true",
                    help="score only targets that produced at least one route, so the "
                         "columns read as route QUALITY rather than quality x coverage")
    ap.add_argument("--curve", default=None,
                    help="write {file: {calls: routes found}} for the budget plot")
    a = ap.parse_args()
    rows = [analyse(Path(f), a.cluster, a.solved_only, a.max_unique, a.max_episode)
            for f in a.files]

    hdr = (f"{'arm':22s} {'tgt':>4s} {'w/rt':>5s} {'solve':>6s} {'routes':>7s} {'1st cut':>8s} "
           f"{'rt/disc':>8s} {'eff@%.1f' % a.cluster:>8s} {'pairJ':>6s} {'reuse':>6s} {'calls/rt':>9s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['file'][:22]:22s} {r['targets']:4d} {r['with_routes']:5d} {r['solve_rate']:6.2%} "
              f"{r['routes']:7.2f} {r['first_cuts']:8.2f} {r['per_cut']:8.2f} {r['effective']:8.2f} "
              f"{r['pairwise_jaccard']:6.3f} {r['reuse']:6.2%} {r['calls_per_route']:9.2f}")
    if a.curve:
        Path(a.curve).write_text(json.dumps({r["file"]: r["_curve"] for r in rows}))
        print(f"\ncurve -> {a.curve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
