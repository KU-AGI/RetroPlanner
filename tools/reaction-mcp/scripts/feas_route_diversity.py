#!/usr/bin/env python
"""How different are the k routes an arm returns FOR ONE TARGET? Three representations.

WHY NOT REACTION CLASS. Within one route, named-reaction classes rarely repeat, so the
statistic saturates: two routes for the same target share few classes by construction and a
class-based between-route distance cannot discriminate. Class counts are also volume-driven:
`classes/route` grows with labelling rate times steps, which is length in disguise.

WHY NOT `ovl%` ALONE. `feas_arm_diversity.ovl%` is a between-route measure and it is a real
one, but it is EXACT-MATCH on disconnection identity: two routes that use the same chemistry
on a different substrate score as maximally different, and one shared step scores as partial
overlap. It cannot say HOW different two routes are.

THE THREE REPRESENTATIONS, each answering a different question. All are pairwise WITHIN one
target, averaged over pairs then over targets, at matched k.

  leaf      Jaccard distance between the routes' LEAF SETS -- the shopping lists. "Do the
            alternatives send you to different starting materials?" The one a chemist cares
            about.
  disc      Jaccard distance between the routes' DISCONNECTION sets, identity
            `(product, frozenset(precursors))`. The graded version of `ovl%`.
  fp        Tanimoto distance between the routes' MOLECULE FINGERPRINTS: every molecule in the
            route (product, intermediate and leaf) folded into one 2048-bit Morgan-2 vector by
            bitwise OR. This is the "one big blob" representation -- tolerant to substrate
            variation, and the only one of the three that sees structural similarity rather
            than set membership.

READ `fp` AGAINST ROUTE LENGTH. A longer route sets more bits, and two fuller vectors are more
similar, so `fp` distance falls as routes lengthen for reasons that have nothing to do with
the chemistry. Length is printed beside it and the length-banded table is the control.

Usage:
  python scripts/feas_route_diversity.py --k 3
  python scripts/feas_route_diversity.py --k 3 --by-length
"""
from __future__ import annotations

import argparse
import itertools
import math
import statistics as st
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import traj_route_common as C


import feas_cost as FC                                                    # noqa: E402
from scipy.stats import rankdata                                          # noqa: E402
import numpy as np                                                        # noqa: E402

sys.path.insert(0, C.SD if isinstance(C.SD, str) else str(C.SD))
import feas_arm_diversity as D                                            # noqa: E402

RDLogger.DisableLog("rdApp.*")

_fp_cache: dict[str, object] = {}
CUT = 0.05


def axes3(routes, plaus, price, rt, plaus_agg="share"):
    """Per route: (plausibility, round-trip share, -cost). None where an axis is undefined.

    TWO PLAUSIBILITY DEFINITIONS, and the choice moves the Pareto front:
      share  fraction of steps clearing 0.05. The default here.
      min    the weakest step. It tends to track route length rather than step quality, so it
             is available but not the default.
    """
    out = []
    for s_ in routes:
        pl = plaus.score([(p, list(r)) for p, r in s_])
        if any(x is None for x in pl):
            a0 = None
        else:
            a0 = (min(pl) if plaus_agg == "min"
                  else sum(1 for x in pl if x >= CUT) / len(s_))
        got = sum(1 for p, r in s_
                  if (v := rt.cache.get(FC.rxn_key(p, r))) is not None and v <= 5)
        prods = {p for p, _ in s_}
        leaves = sorted({x for _, r in s_ for x in r if x not in prods})
        pr = [price.ln_price(x) for x in leaves]
        out.append((a0, got / len(s_),
                    None if (not pr or any(x is None for x in pr))
                    else -sum(math.exp(x) for x in pr)))
    return out


def pareto_idx(A):
    """Indices non-dominated on (plaus up, rt up, -cost up). None is worst on its axis.

    Restated on the (already negated) cost axis so all three are maximise-up. A front member is optimal under SOME weighting of the three axes,
    which is why it is the right set for "are the trade-offs we offer actually different" --
    the composite's top-k are all optimal under the SAME weighting and have a reason to look
    alike.
    """
    def v(x):
        return -1e18 if x is None else x
    out = []
    for i, ai in enumerate(A):
        if not any(all(v(aj[k]) >= v(ai[k]) for k in range(3))
                   and any(v(aj[k]) > v(ai[k]) for k in range(3))
                   for j, aj in enumerate(A) if j != i):
            out.append(i)
    return out


def feas_rank(routes, plaus, price, rt):
    """Order one target's routes by OUR three-axis composite, best first.

    The composite is the mean of the three PERCENTILE RANKS inside this
    target's pool -- share of steps clearing plausibility 0.05, share the forward model
    recovers at top-5, and summed leaf price (lower better). Routes missing an axis rank
    worst rather than being dropped, so an unscored route cannot win by default.

    Selecting by a score and then measuring diversity are in tension by construction: two
    routes that both maximise the same three axes have a reason to look alike. That is the
    point of measuring it this way -- the question is whether the routes the method actually
    REPORTS are diverse, not whether some pair in its pool is.
    """
    ax = axes3(routes, plaus, price, rt)
    n = len(routes)
    if n < 2:
        return list(range(n))
    cols = []
    for j, up in ((0, True), (1, True), (2, True)):
        v = [a[j] for a in ax]
        worst = min([x for x in v if x is not None], default=0.0) - 1.0
        v = [worst if x is None else x for x in v]
        cols.append((rankdata(v, method="average") - 1.0) / (n - 1.0))
    comp = np.mean(np.vstack(cols), axis=0)
    return sorted(range(n), key=lambda i: -comp[i])


def mol_fp(smi: str):
    if smi not in _fp_cache:
        m = Chem.MolFromSmiles(smi)
        _fp_cache[smi] = (AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                          if m is not None else None)
    return _fp_cache[smi]


def route_fp(steps, drop_root=False):
    """One 2048-bit vector for the whole route: bitwise OR over every molecule in it."""
    skip = set()
    if drop_root:
        prods = {p for p, _ in steps}
        skip = {p for p in prods if not any(p in rs for _, rs in steps)}
    acc = None
    for prod, reacts in steps:
        for smi in [prod] + list(reacts):
            if smi in skip:
                continue
            f = mol_fp(smi)
            if f is None:
                continue
            acc = f if acc is None else (acc | f)
    return acc


def leaf_set(steps):
    prods = {p for p, _ in steps}
    return frozenset(x for _, rs in steps for x in rs if x not in prods)


def disc_set(steps):
    return frozenset((p, frozenset(rs)) for p, rs in steps)


def jac_dist(a: frozenset, b: frozenset) -> float | None:
    u = len(a | b)
    return None if not u else 1.0 - len(a & b) / u


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3,
                    help="routes kept per target, in the arm's own order; 0 = whole pool")
    ap.add_argument("--stock", default="emols")
    ap.add_argument("--by-length", action="store_true",
                    help="split by mean route length, because `fp` distance falls as routes "
                         "lengthen for purely combinatorial reasons")
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--rt-cache-only", action="store_true",
                    help="skip the forward-server probe and never call it. Only valid when "
                         "the rt cache already covers every step -- verify first, because a "
                         "step it lacks is scored as a MISS, which is not the same as unknown")
    ap.add_argument("--arm", action="append", default=[],
                    help="LABEL=kind:path  (kind in search|pooled|board); appended to the "
                         "built-in arm list")
    ap.add_argument("--only", default=None, help="comma-separated subset of labels")
    ap.add_argument("--rank", choices=("emit", "feas", "pareto"), default="emit",
                    help="emit = the arm's own order (Retro* cost order / file order); "
                         "feas = OUR three-axis composite, best k; pareto = the three-axis "
                         "NON-DOMINATED front")
    ap.add_argument("--plaus-agg", choices=("share", "min"), default="share",
                    help="share = steps clearing 0.05 (default); "
                         "min = the weakest step, which tends to track route length")
    ap.add_argument("--pool-cap", type=int, default=50,
                    help="routes scored per target before feasibility ranking, in emit "
                         "order. Applied equally to every arm; a search arm can return "
                         "hundreds of routes and scoring all of them is not affordable")
    ap.add_argument("--drop-root", action="store_true",
                    help="exclude the target molecule from the route fingerprint -- it is "
                         "shared by every route for that target by construction")
    a = ap.parse_args()

    stock = C.Stock(a.stock)
    print(f"stock {stock.describe()['spec']} n_keys={len(stock):,}  rank={a.rank}  "
          f"pool_cap={a.pool_cap}  drop_root={a.drop_root}\n", flush=True)
    kk = a.k if a.k > 0 else None
    plaus = price = rt = None
    if a.rank in ("feas", "pareto"):
        plaus, price = FC.PlausScorer(), FC.PriceScorer()
        rt = FC.RtScorer(overlay=C.OUT / "feas" / "rt_overlay_routediv.json")
        if not a.rt_cache_only:
            rt.probe()
        else:
            print("[rt] cache-only: no live call will be made, probe skipped", flush=True)
        print(f"three-axis ranking on: plaus live, price cache-first, "
              f"rt {len(rt.cache):,} cached\n", flush=True)

    print(f"{'arm':14} {'targets':>8} {'routes/tgt':>11} {'steps':>6} "
          f"{'leaf':>7} {'disc':>7} {'fp':>7}")
    print("-" * 68)
    rows = {}
    arms = list(D.ARMS)
    for spec in a.arm:
        lbl, rhs = spec.split("=", 1)
        kind, path = rhs.split(":", 1)
        arms.append((lbl, kind, path))
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        arms = [x for x in arms if x[0] in keep]
    for label, kind, pattern in arms:
        pool = D.load_arm(kind, pattern)
        per, fronts = [], []                      # (n_routes, mean_steps, leaf, disc, fp)
        for t, rs in pool.items():
            cand = rs[:a.pool_cap]
            if a.rank == "pareto":
                # the gate FIRST: a front computed over inadmissible routes would offer
                # trade-offs that cannot be run
                cand = [s for s in cand
                        if D.conserves(s) and all(stock.has(x) for x in D.leaves_of(s))]
                if len(cand) > 1:
                    if not a.rt_cache_only:
                        for one in cand:
                            rt.ranks([(p, list(r)) for p, r in one])
                    A = axes3(cand, plaus, price, rt, a.plaus_agg)
                    front = pareto_idx(A)
                    order = feas_rank([cand[i] for i in front], plaus, price, rt)
                    cand = [[cand[i] for i in front][j] for j in order]
                    fronts.append(len(front))
                keep = cand[:kk] if kk else cand
            else:
                if a.rank == "feas" and len(cand) > 1:
                    if not a.rt_cache_only:
                        for one in cand:          # warm rt for every candidate step
                            rt.ranks([(p, list(r)) for p, r in one])
                    cand = [cand[i] for i in feas_rank(cand, plaus, price, rt)]
                keep = [s for s in (cand[:kk] if kk else cand)
                        if D.conserves(s) and all(stock.has(x) for x in D.leaves_of(s))]
            if len(keep) < 2:
                continue
            L = [leaf_set(s) for s in keep]
            S = [disc_set(s) for s in keep]
            F = [route_fp(s, a.drop_root) for s in keep]
            dl, dd, df = [], [], []
            for i, j in itertools.combinations(range(len(keep)), 2):
                v = jac_dist(L[i], L[j])
                if v is not None:
                    dl.append(v)
                v = jac_dist(S[i], S[j])
                if v is not None:
                    dd.append(v)
                if F[i] is not None and F[j] is not None:
                    from rdkit import DataStructs
                    df.append(1.0 - DataStructs.TanimotoSimilarity(F[i], F[j]))
            if len(dl) < a.min_pairs:
                continue
            per.append((len(keep), st.mean(len(s) for s in keep),
                        st.mean(dl) if dl else None, st.mean(dd) if dd else None,
                        st.mean(df) if df else None))
        rows[label] = per
        if not per:
            print(f"{label:14} {0:8}   -- no target has 2 gate-passing routes at this k")
            continue
        f = lambda i: st.mean(x[i] for x in per if x[i] is not None)
        if fronts:
            print(f"{'':14} {'':8} front size mean {st.mean(fronts):.2f}", flush=True)
        print(f"{label:14} {len(per):8} {st.mean(x[0] for x in per):11.2f} "
              f"{st.mean(x[1] for x in per):6.2f} {f(2):7.3f} {f(3):7.3f} {f(4):7.3f}")

    if a.by_length:
        print("\nBY MEAN ROUTE LENGTH. `fp` distance falls as routes lengthen -- a fuller "
              "bit\nvector is more similar to another full one -- so only a within-band "
              "comparison is\nfree of that. Bands with fewer than 5 targets are dropped.")
        edges = ((1, 4), (5, 7), (8, 11), (12, 99))
        hdr = f"{'arm':14}" + "".join(
            f"{f'{lo}-{hi if hi < 99 else chr(43)}':>22}" for lo, hi in edges)
        print("\n" + hdr)
        print("-" * len(hdr))
        for label, _, _ in arms:
            row = f"{label:14}"
            for lo, hi in edges:
                v = [x for x in rows.get(label, []) if lo <= x[1] <= hi]
                if len(v) < 5:
                    row += f"{'-':>22}"
                else:
                    g = lambda i: st.mean(x[i] for x in v if x[i] is not None)
                    row += f"{f'{g(2):.2f}/{g(3):.2f}/{g(4):.2f} ({len(v)})':>22}"
            print(row)
        print("\ncell = leaf / disc / fp distance  (n targets)")

    if a.rank in ("feas", "pareto"):
        rt.flush()
        print(f"\nrt: {rt.hits:,} cache hits, {rt.scored:,} scored live, "
              f"{rt.calls:,} batched requests")
    print(f"\nk = {a.k if a.k else 'all'}. Only targets with >=2 gate-passing routes "
          f"contribute -- a\npool of one has no pair, so arms that return one route per "
          f"target are absent by\nconstruction, not by failing.")
    print("1.0 = the two routes share nothing in that representation; 0.0 = identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
