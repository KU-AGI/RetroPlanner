#!/usr/bin/env python
"""Tag every step and every molecule of the pooled routes on the synthesis objectives.

This does not re-implement a single axis. It calls reaction_mcp.scoring and writes into the
SAME content-keyed caches under data/node_scores/, so a reaction scored here is literally the
cache entry evaluation reads, and the train population and the benchmark population are on
one ruler. That is also why every script keys reactions with the one `rxn_key`.

Axes, and the node kind each attaches to. A retrosynthesis route is bipartite — molecules are
OR nodes, reactions are AND nodes — and a score computed for one is meaningless on the other,
so nothing here blends them into a single number:

  molecule    scscore     Coley SCScore 1-5, synthetic complexity of THIS molecule
              molprice    MolPrice USD/mmol, predicted price of THIS molecule
  reaction    plausibility  AiZynthFinder's filter policy, P(feasible); its own cut-off is 0.05
              roundtrip     rank at which a forward model recovers the product from THIS
                            disconnection's precursors (1 = top-1, None = not recovered)
              ord / ord_journal  precedent tier in the Open Reaction Database:
                            exact | pair | product | none

Two round-trip caches exist and they are NOT interchangeable:

  roundtrip.json      a second forward model on :8089, not the round-trip objective
  roundtrip_rt5.json  ReactionT5v2, served by reactiont5_forward_server.py on :8090

The round-trip objective is the ReactionT5v2 one. Which cache a run writes is decided
by `--rt-cache`, never by which server happens to be up, because the failure is silent: point
at the wrong server and the numbers are real, comparable to nothing, and indistinguishable
from a result. The server is probed with a reaction it must reproduce at rank 1 before
anything is written, so a server that ranks from zero, or is down, is refused.

Gold is scored too (`--with-gold`, on by default). Every axis needs its published-chemistry
reference row or the numbers have no scale.

Environment: one with syntheseus, onnxruntime, joblib, sklearn, pandas and pyarrow.

Usage:
  python scripts/traj_route_score.py --stem paroutes --report
  python scripts/traj_route_score.py --stem paroutes --axis plausibility,scscore
  python scripts/traj_route_score.py --stem paroutes --axis molprice
  REACTION_FORWARD_URL=http://127.0.0.1:8090 \
    python scripts/traj_route_score.py --stem paroutes --axis roundtrip --rt-cache roundtrip_rt5
  python scripts/traj_route_score.py --stem paroutes --axis ord,ord_journal
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import threading
import json
import os
import sys
import time

import traj_route_common as C


AXES = ("scscore", "molprice", "plausibility", "roundtrip", "ord", "ord_journal")
MOLECULE_AXES = ("scscore", "molprice")


def cache_path(name):
    return C.NODE_SCORES / f"{name}.json"


def read_cache(name):
    p = cache_path(name)
    return json.load(open(p)) if p.exists() else {}


def write_cache(name, new):
    """MERGE into whatever is already there. Never truncate — another axis pass, another
    population, or an earlier enumeration put entries here that this run does not mention."""
    C.NODE_SCORES.mkdir(parents=True, exist_ok=True)
    cur = read_cache(name)
    cur.update(new)
    tmp = str(cache_path(name)) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cur, fh)
    os.replace(tmp, cache_path(name))
    return len(cur)


def universe(stem, with_gold, extra_pooled=None):
    """-> (molecules, {rxn_key: (product, [reactants])}) over the pool and the gold routes."""
    mols, rxns = set(), {}

    def add(steps):
        for p, rs in steps:
            rs = sorted(set(rs))
            rxns[C.rxn_key(p, rs)] = (p, rs)
            mols.add(p)
            mols.update(rs)

    n_r = 0
    for path in ([str(C.OUT / f"pooled_{stem}.jsonl")] + list(extra_pooled or [])):
        if not os.path.exists(path):
            continue
        for rec in C.read_jsonl(path):
            add(C.norm_steps(rec["steps"]))
            n_r += 1
    print(f"  {n_r} pooled routes", flush=True)
    n_g = 0
    if with_gold:
        tp = C.OUT / f"targets_{stem}.jsonl"
        if tp.exists():
            for rec in C.read_jsonl(tp):
                for s in rec["gold_routes"]:
                    add(C.norm_steps(s))
                    n_g += 1
            print(f"  + {n_g} gold routes", flush=True)
    print(f"  {len(mols)} distinct molecules, {len(rxns)} distinct reactions", flush=True)
    return mols, rxns


# ------------------------------------------------------------------ per-axis
def do_scscore(need):
    from reaction_mcp.scoring.scscore import scscore
    return {m: scscore(m) for m in sorted(need)}


def do_molprice(need):
    """The committed price CSVs first; MolPrice predicts only what they do not cover."""
    from reaction_mcp.scoring.price import load_price_tables, predict_price
    prices = load_price_tables()
    out = {m: prices[m] for m in need if m in prices}
    rest = sorted(m for m in need if m not in prices)
    print(f"  {len(out)} from the committed price CSVs, {len(rest)} to predict", flush=True)
    if not rest:
        return out
    for i, smi in enumerate(rest):
        # None, never 0.0: 0.0 is a real price (1 USD/mmol), so a molecule MolPrice cannot
        # featurise ([H][H], an unparseable SMILES) must land in `n_unpriced`.
        out[smi] = predict_price(smi)
        if (i + 1) % 2000 == 0:
            print(f"  priced {i+1}/{len(rest)}", flush=True)
    return out


def do_plausibility(need, rxns):
    from reaction_mcp.scoring.plausibility import score_reactions
    keys = sorted(need)
    vals = score_reactions([rxns[k] for k in keys])
    return dict(zip(keys, vals))


def do_roundtrip(need, rxns, top_k, batch, rt_workers=32):
    from reaction_mcp.scoring.roundtrip import roundtrip, server_url
    probe = roundtrip([("CC(=O)Nc1ccccc1", ["CC(=O)Cl", "Nc1ccccc1"])], top_k=5)
    if probe[0] != 1:
        raise SystemExit(
            f"forward server at {server_url()} "
            f"returned rank {probe[0]} for a reaction it must reproduce at top-1: it is down, "
            "or it is a 0-indexed forward model. Nothing written.")
    print("  forward server ok (known reaction comes back at rank 1)", flush=True)
    # Batches go out CONCURRENTLY. A sequential loop pins the whole pass to ONE replica no
    # matter how large the forward pool is. `--workers` is the number of batches in flight; the proxy fans them over its replicas.
    keys, out, t0 = sorted(need), {}, time.time()
    chunks = [keys[s:s + batch] for s in range(0, len(keys), batch)]
    lock, done = threading.Lock(), [0]

    def run(chunk):
        vals = roundtrip([rxns[k] for k in chunk], top_k=top_k, batch=batch)
        with lock:
            out.update(dict(zip(chunk, vals)))
            done[0] += len(chunk)
            d = done[0]
            rate = d / max(time.time() - t0, 1e-9)
            if d % (batch * 20) < batch:
                print(f"  {d}/{len(keys)}  {rate:.1f}/s  "
                      f"eta {(len(keys)-d)/max(rate,1e-9)/60:.0f} min", flush=True)

    with cf.ThreadPoolExecutor(max_workers=rt_workers) as ex:
        list(ex.map(run, chunks))
    return out


def do_ord(need, rxns, source):
    # The ORD precedent tiers are read from their caches (ord.json, ord_journal.json); this
    # repository does not carry the ORD index that computes new ones.
    raise SystemExit(f"{len(need)} reactions lack an ORD tier ({source}); the ord axes are "
                     "cache-only here (data/node_scores/ord*.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="fusionretro")
    ap.add_argument("--axis", default="plausibility,scscore")
    ap.add_argument("--extra-pooled", default="",
                    help="comma-separated extra pooled_*.jsonl to fold into the universe, so "
                         "one scoring pass covers several datasets")
    ap.add_argument("--no-gold", action="store_true",
                    help="omit the published routes. Their row is the only scale the other "
                         "numbers have; drop it only when it is already cached")
    ap.add_argument("--rt-cache", default="roundtrip_rt5",
                    choices=("roundtrip", "roundtrip_rt5"),
                    help="which round-trip cache this run writes. roundtrip_rt5 = "
                         "ReactionT5v2 on :8090 (the round-trip objective); roundtrip = a "
                         "second forward model on :8089")
    ap.add_argument("--top-k", type=int, default=5,
                    help="rank depth asked of the forward model; the cache stores the RANK so "
                         "any cutoff is applied later")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rt-workers", type=int, default=32,
                    help="round-trip batches in flight at once. The forward pool proxy fans "
                         "them over its replicas; 1 pins the pass to a single server")
    ap.add_argument("--report", action="store_true", help="coverage only, compute nothing")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()

    extra = [x.strip() for x in a.extra_pooled.split(",") if x.strip()]
    mols, rxns = universe(a.stem, not a.no_gold, extra)
    if not rxns:
        raise SystemExit("nothing to score — run traj_route_pool.py first")

    if a.report:
        for axis in AXES:
            name = a.rt_cache if axis == "roundtrip" else axis
            c = read_cache(name)
            u = mols if axis in MOLECULE_AXES else set(rxns)
            have = u & set(c)
            val = sum(1 for k in have if c[k] is not None)
            print(f"  {axis:14} ({name+'.json':22}) {len(have):8}/{len(u):8} cached, "
                  f"{val:8} with a value, {len(c):9} in cache overall")
        return

    for axis in [x.strip() for x in a.axis.split(",") if x.strip()]:
        if axis not in AXES:
            raise SystemExit(f"unknown axis {axis}; pick from {AXES}")
        name = a.rt_cache if axis == "roundtrip" else axis
        cached = read_cache(name)
        u = mols if axis in MOLECULE_AXES else set(rxns)
        need = set(u) if a.recompute else {k for k in u if k not in cached}
        print(f"\n{axis} -> {name}.json: {len(need)} to score "
              f"({len(u)-len(need)} already cached)", flush=True)
        if not need:
            continue
        if axis == "scscore":
            new = do_scscore(need)
        elif axis == "molprice":
            new = do_molprice(need)
        elif axis == "plausibility":
            new = do_plausibility(need, rxns)
        elif axis == "roundtrip":
            new = do_roundtrip(need, rxns, a.top_k, a.batch, a.rt_workers)
        else:
            new = do_ord(need, rxns, "all" if axis == "ord" else "journal")
        total = write_cache(name, new)
        got = sum(1 for v in new.values() if v is not None)
        print(f"  wrote {len(new)} ({got} with a value) -> {cache_path(name)}, "
              f"{total} total", flush=True)


if __name__ == "__main__":
    sys.exit(main())
