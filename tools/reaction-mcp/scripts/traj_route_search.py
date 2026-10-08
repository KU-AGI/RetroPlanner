#!/usr/bin/env python
"""Retro*-0 (and MCTS / PDVN) over the LIVE single-step models, on the train targets.

The training corpus is built by searching each PaRoutes train target once with Retro*-0 and
R-SMILES. There are no precomputed expansions for the train targets, so this drives the
models over the SSR wire, and that changes two things which are the whole design of this file:

  1. The expander CAN BE STOCHASTIC. A model that decodes with unseeded randomised-root
     SMILES returns a different menu when one molecule is asked again. `--draws N` exploits
     that: N INDEPENDENT searches per (target, algo), each with its own expansion memo, pooled
     afterwards. It is also a hazard
     — see the memo below — because a search whose model answers differently on re-query is
     not reproducible and its budget stops meaning anything.

  2. Every expansion costs a real model call, so the budget is the schedule. It is counted in
     `limit_reaction_model_calls` — single-step model calls, the resource all three algorithms
     actually spend — because an ITERATION means a different amount of work in each and
     matching on iterations compares protocols rather than policies.

`stop_on_first_solution=False` throughout: the point is a route SET per target, since the
Pareto selection downstream needs alternatives to choose between.

Two per-run memos, and they are not the same thing:

  the model memo (in LiveBackwardModel)   {(smiles, k): menu} for the life of ONE search. It
      freezes the menu inside a run, so re-expanding a molecule in a second branch sees what
      the first branch saw. Without it a stochastic expander makes the graph depend on
      traversal order, two runs of the same seed disagree, and `limit_reaction_model_calls`
      caps something that is not reproducible. Budget still counts the re-expansion (syntheseus
      counts it, use_cache=False), so the accounting matches the evaluation protocol, where a
      lookup also costs a call; `http_calls` records the smaller number of real HTTP requests.

  the draw cache (--cache, optional)      {smiles: menu} on disk, shared by every run of a
      given (model, draw). Turns the draw into a reproducible frozen oracle across algorithms,
      so retrostar / mcts / pdvn on one draw see the SAME chemistry and differ only in
      their search policy — the controlled comparison. WITHOUT it the three algorithms each
      sample their own menus and the algorithm effect is confounded with the sampling effect.
      It is on by default for that reason; `--no-cache` gives independent sampling per run.

Depth. syntheseus counts depth in GRAPH HOPS and an AND-OR graph alternates OrNode -> AndNode
-> OrNode, so one reaction is two levels: `max_expansion_depth=5` caps an AND-OR search at
THREE steps. A MolSetNode's depth counts reactions, one per level. The cap is therefore
doubled for Retro*/PDVN and left alone for MCTS, or MCTS is quietly allowed twice everyone
else's depth. Gold train routes can be deep, so --max-depth defaults to 8, not 5.

Usage (an env with syntheseus + rdkit + requests):
  # smoke: does one target come back solved, and does gold come back out
  python scripts/traj_route_search.py --ds paroutes --models rsmiles --algos retrostar \
      --budget 100 --limit 3 --draws 1 --workers 3
  # the training search: Retro*-0 + R-SMILES, 300 single-step calls per target
  python scripts/traj_route_search.py --ds paroutes --models rsmiles \
      --algos retrostar --budget 300 --draws 1 --workers 16

Retro* with its learned cost-to-go (`retrostar-value`) is not offered: this repo does not carry
that checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import glob
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import traj_route_common as C


from syntheseus.interface.bag import Bag                                       # noqa: E402
from syntheseus.interface.models import (BackwardReactionModel,                # noqa: E402
                                        SingleProductReaction)
from syntheseus.interface.molecule import Molecule                             # noqa: E402
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch  # noqa: E402
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS               # noqa: E402
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order  # noqa: E402
from syntheseus.search.graph.and_or import AndNode                            # noqa: E402
from syntheseus.search.mol_inventory import BaseMolInventory                   # noqa: E402
from syntheseus.search.node_evaluation.common import (ConstantNodeEvaluator,   # noqa: E402
                                                     ReactionModelLogProbCost)

ALGOS = ("retrostar", "mcts", "pdvn")

# --feas-lambda > 0 prices the three feasibility axes into the search's own cost function
# (feas_cost.py). lam = 0 is the untouched baseline: `FeasibilityCost` with lam 0 reduces to
# ReactionModelLogProbCost and `FeasibilityReward` to ConstantNodeEvaluator(1.0), so both
# tables come out of ONE code path and differ only in lam. The scorer is built once per
# PROCESS, not per cell, so every (algo, budget) shares one round-trip cache -- the axis with
# the only per-call GPU cost.
FEAS = {"scorer": None}


def feas_scorer(lam, rt_url=None, overlay=None, use_rt=True, agg="geo"):
    if lam <= 0:
        return None
    if FEAS["scorer"] is None:
        import feas_cost                                                   # noqa: PLC0415
        feas_cost.set_agg(agg)
        FEAS["scorer"] = feas_cost.FeasScorer(rt_url=rt_url, rt_overlay=overlay,
                                              use_rt=use_rt)
    return FEAS["scorer"]


# ------------------------------------------------------------------ the expander
# DrawCache lives in traj_route_common for the same reason ssr_call does: a caller without
# syntheseus must share the SAME menus as the syntheseus searches, or an algorithm comparison
# is confounded with the sampling.
DrawCache = C.DrawCache


# The SSR wire lives in traj_route_common: a caller in an env without syntheseus would fail on
# the syntheseus imports above just to reach one HTTP helper. Re-exported here because several scripts already import it by this
# name.
ssr_call = C.ssr_call


class LiveBackwardModel(BackwardReactionModel):
    """One search's expander. `num_calls` is the budget; `http_calls` is what it cost."""

    def __init__(self, url, top_k, timeout, draw_cache=None, **kw):
        super().__init__(**kw)
        self.url, self.top_k, self.timeout = url, top_k, timeout
        self.draw_cache = draw_cache
        self.memo: dict[str, list] = {}
        self.http_calls = self.memo_hits = self.cache_hits = self.errors = 0
        self.empty_retries = self.empty_final = 0
        self.last_error = None

    def _menu(self, smi):
        m = self.memo.get(smi)
        if m is not None:
            self.memo_hits += 1
            return m
        if self.draw_cache is not None:
            m = self.draw_cache.get(smi)
            if m is not None:
                self.cache_hits += 1
                self.memo[smi] = m
                return m
        # RETRY AN EMPTY ANSWER. A 200 carrying `[]` is not a dead end and not a dropped
        # packet -- R-SMILES's augmented decoding can come back empty or full per CALL for the
        # same molecule, with no replica reporting a failure. Asking once and memoising the
        # `[]` would keep that molecule a dead end for the rest of the run. A few tries with a
        # short backoff; the cap matters because a molecule that really has nothing must not
        # stall the search.
        m = []
        for _try in range(6):
            try:
                m = ssr_call(self.url, smi, self.top_k, self.timeout)
                self.http_calls += 1
            except Exception as e:                              # noqa: BLE001
                # Recorded, not swallowed into a dead end. The record carries `errors`, and a
                # target with errors is re-runnable with --retry-errored.
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                m = []
                break
            if m:
                break
            self.empty_retries += 1
            time.sleep(0.4 * (_try + 1))
        if not m:
            self.empty_final += 1
        self.memo[smi] = m
        if self.draw_cache is not None and m:
            self.draw_cache.put(smi, m)
        return m

    def _get_reactions(self, inputs, num_results):
        out = []
        for mol in inputs:
            menu = self._menu(mol.smiles)
            k = min(num_results, self.top_k)
            rxns = []
            for rank, (rs, sc) in enumerate(menu[:k]):
                rxns.append(SingleProductReaction(
                    product=mol, reactants=Bag([Molecule(x) for x in rs]),
                    # probability is what ReactionModelLogProbCost takes -log of, so a zero
                    # confidence would be an infinite cost and would remove the candidate
                    # rather than deprioritise it.
                    metadata={"probability": max(float(sc), 1e-6), "score": float(sc),
                              "rank": rank}))
            out.append(rxns)
        return out


class StockInventory(BaseMolInventory):
    def __init__(self, stock):
        self.stock = stock

    def is_purchasable(self, mol):
        return self.stock.has(mol.smiles)


# ------------------------------------------------------------------ algorithms
def make_alg(kind, model, inventory, budget, max_depth, time_limit,
             scorer=None, lam=0.0):
    # AND-OR depth counts graph hops (2 per reaction); MolSet depth counts reactions.
    depth = max_depth if kind == "mcts" else 2 * max_depth
    common = dict(reaction_model=model, mol_inventory=inventory,
                  limit_reaction_model_calls=budget, max_expansion_depth=depth,
                  stop_on_first_solution=False, unique_nodes=False,
                  prevent_repeat_mol_in_trees=True, time_limit_s=time_limit,
                  # MCTS can select a node it may not expand (depth cap) and re-select it
                  # forever, spending no model calls: the call budget then never binds and the
                  # run sits until the wall clock. Far above what a healthy run uses.
                  limit_iterations=40 * budget)
    # The AND-node cost is the single place feasibility enters the two A* searches:
    #   lam = 0  ->  -log p_model                      (ReactionModelLogProbCost, the baseline)
    #   lam > 0  ->  -log p_model - lam * log f_step   (feas_cost.FeasibilityCost)
    # FeasibilityCost with lam=0 is arithmetically the former, but the baseline uses the
    # syntheseus class so that a lam=0 run cannot even in principle load a scorer.
    if scorer is not None and lam > 0:
        import feas_cost                                                   # noqa: PLC0415
        # The model's per-search memo IS the frozen menu, which is what a percentile rank has
        # to be taken over; `feas-agg=pct` refuses to run without it.
        cost_fn = feas_cost.feasibility_cost(
            scorer, lam, menu_fn=lambda smi: [list(rs) for rs, _ in (model.memo.get(smi) or [])])
    else:
        cost_fn = ReactionModelLogProbCost()
    if kind == "retrostar":
        # Retro*-0: the model's own -log p as the AND-node cost, no cost-to-go at all.
        return RetroStarSearch(and_node_cost_fn=cost_fn,
                               value_function=ConstantNodeEvaluator(0.0), **common)
    if kind == "mcts":
        # MCTS has no additive edge cost to hang -log f on: the reward is evaluated at
        # TERMINAL nodes. So feasibility enters as the reward -- the geometric mean of f over
        # the root path, raised to lam. lam=0 returns 1.0 for every terminal node, which is
        # ConstantNodeEvaluator(1.0), the baseline. Note the asymmetry with the A* searches and do
        # not read a lam as "the same strength" across the two: here lam is an exponent on a
        # reward in (0, 1], there it is a weight on an additive cost in nats.
        if scorer is not None and lam > 0:
            import feas_cost                                               # noqa: PLC0415
            reward = feas_cost.feasibility_reward(
                scorer, lam,
                menu_fn=lambda smi: [list(rs) for rs, _ in (model.memo.get(smi) or [])])
        else:
            reward = ConstantNodeEvaluator(1.0)
        return MolSetMCTS(reward_function=reward,
                          value_function=ConstantNodeEvaluator(0.5),
                          bound_constant=1.0, **common)
    if kind == "pdvn":
        # PDVN_MCTS. Its two value networks -- synthesizability and cost-to-go --
        # are the paper's contribution and there is no trained pair in this repo,
        # so they are CONSTANTS here. That makes this "PUCT search with a
        # dead-end cost", not the published PDVN.
        #
        # The graph runs carry `has_solution` per molecule (a synthesizability
        # label) and route costs (a cost target); a trained pair dropped in here
        # makes it the real method without any other change.
        from syntheseus.search.algorithms.pdvn import PDVN_MCTS
        from syntheseus.search.node_evaluation.common import ReactionModelProbPolicy

        if lam > 0:
            raise SystemExit(
                "pdvn has no feasibility variant: its AND-node cost is a CONSTANT and its "
                "two value functions are untrained, so a lam here would be added to a cost "
                "nothing else in the search uses. Run pdvn at lam=0 or not at all.")
        return PDVN_MCTS(c_dead=10.0,
                         value_function_syn=ConstantNodeEvaluator(0.5),
                         value_function_cost=ConstantNodeEvaluator(1.0),
                         and_node_cost_fn=ConstantNodeEvaluator(1.0),
                         policy=ReactionModelProbPolicy(),
                         bound_constant=1.0, **common)
    raise SystemExit(f"unknown algorithm {kind}; pick from {ALGOS}")


def routes_of(graph, max_routes, max_time_s):
    """-> (routes, truncated, extract_s) in the one shape every downstream stage reads.

    `truncated` is not cosmetic. The extraction is capped twice — by `max_routes` and by
    `max_time_s` — and both caps can bind. Routes past the cap are DROPPED, and because they are dropped in
    cost order the ones kept are the best ones, which is the right choice and also the reason
    the loss is invisible: `n_routes` saturates at a round number and every downstream
    statistic — routes/target, leaf-set diversity, the size of the pool the Pareto front is
    chosen from — silently reports the CAP instead of the graph. One extra route is requested
    so hitting the cap can be distinguished from happening to end there.

    The two graph types hold the reaction in different places:
      AND-OR (Retro*)  on the AndNode.
      MolSet (MCTS)    nodes are SETS of molecules and the reaction is on the EDGE between
                       two of them, so a route's steps are the edges of the induced subgraph.
                       Reading nodes only — the obvious thing to write — yields zero steps and
                       the run silently reports "solved, 0 routes".
    """
    out, t0 = [], time.time()
    for nodes in iter_routes_cost_order(graph, max_routes=max_routes + 1,
                                        max_time_s=max_time_s):
        steps = []
        if any(isinstance(n, AndNode) for n in nodes):
            for n in nodes:
                if isinstance(n, AndNode):
                    steps.append((n.reaction.product.smiles,
                                  sorted(m.smiles for m in n.reaction.reactants)))
        else:
            for _, _, d in graph._graph.subgraph(nodes).edges(data=True):
                r = d.get("reaction")
                if r is not None:
                    steps.append((r.product.smiles, sorted(m.smiles for m in r.reactants)))
        if steps:
            # A route can arrive with one disconnection twice (two branches, same reaction).
            # Deduplicate on the set key so a route's step list is its chemistry.
            seen, ded = set(), []
            for p, rs in steps:
                k = (p, tuple(rs))
                if k not in seen:
                    seen.add(k)
                    ded.append((p, rs))
            out.append(ded)
    extract_s = time.time() - t0
    truncated = ("max_routes" if len(out) > max_routes
                 else "max_time" if extract_s >= max_time_s * 0.98 else None)
    return out[:max_routes], truncated, round(extract_s, 2)


def graph_dump(graph, target, stock, menu_of):
    """-> the search graph as data: every molecule the search OPENED and how each of its
    top-k candidates turned out.

    `routes` records only what SUCCEEDED, and it is extracted from a graph that is then thrown
    away. Everything the search tried and abandoned — the dead ends — is lost, and a dead end
    is not noise here: it is the only direct evidence that a candidate was a bad choice.
    Without it a training trace can say "take candidate 3" but never "candidate 7 leads
    nowhere", which is the half of the signal a selection policy actually needs.

    Reconstructing it afterwards from the draw cache does not work. The cache says a molecule
    was expanded SOMETIME by SOME target; it cannot say under which parent, in what order, or
    whether the expansion failed or was merely unnecessary. syntheseus already carries the
    answer on the node (`has_solution`, `is_expanded`), so this reads it instead of guessing.

    Per candidate of an expanded molecule, exactly one of three labels:

      used        an AndNode exists for it and has_solution -> it reaches purchasable leaves
      dead        an AndNode exists for it and NOT has_solution -> the search tried it and the
                  subtree never closed. A real negative, at this decision point.
      unexpanded  no AndNode -> the search never opened it. NOT a negative: unknown. Counting
                  these as negatives would teach the policy that whatever the search happened
                  to skip is bad, which is a statement about the search order, not chemistry.

    AND-OR only (Retro*/Retro*-0). MolSet graphs (MCTS) put molecules in SETS and the reaction
    on the EDGE, so a node is not a decision point and the same labels do not apply; those
    return {"kind": "molset"} rather than a wrong-shaped dump.
    """
    from syntheseus.search.graph.and_or import OrNode

    nodes = list(graph._graph.nodes)
    if not any(isinstance(n, OrNode) for n in nodes):
        return {"kind": "molset", "n_nodes": len(nodes)}

    # product smiles -> {frozenset(reactants): AndNode}
    tried = collections.defaultdict(dict)
    for n in nodes:
        if isinstance(n, AndNode):
            r = n.reaction
            tried[r.product.smiles][frozenset(m.smiles for m in r.reactants)] = n

    mols, cand_stat = [], collections.Counter()
    for n in nodes:
        if not isinstance(n, OrNode):
            continue
        smi = n.mol.smiles
        row = {"smiles": smi, "depth": n.depth, "is_expanded": bool(n.is_expanded),
               "has_solution": bool(n.has_solution), "num_visit": int(n.num_visit),
               "in_stock": stock.has(smi)}
        if n.is_expanded:
            cands = []
            for rank, (rs, conf) in enumerate(menu_of(smi) or [], 1):
                rs = list(rs) if isinstance(rs, (list, tuple)) else [rs]
                a = tried[smi].get(frozenset(rs))
                lab = ("unexpanded" if a is None
                       else "used" if a.has_solution else "dead")
                cand_stat[lab] += 1
                cands.append({"rank": rank, "reactants": sorted(rs),
                              "model_confidence": conf, "label": lab,
                              "num_visit": int(a.num_visit) if a is not None else 0})
            row["candidates"] = cands
        mols.append(row)

    exp = [m for m in mols if m["is_expanded"]]
    return {"kind": "and_or",
            "n_or_nodes": len(mols), "n_and_nodes": sum(len(v) for v in tried.values()),
            "n_expanded": len(exp),
            "n_expanded_with_solution": sum(1 for m in exp if m["has_solution"]),
            "n_dead_ends": sum(1 for m in exp if not m["has_solution"]),
            "candidate_labels": dict(cand_stat),
            "molecules": mols}


def one_target(tgt, gold_setkeys, kind, url, stock, budget, top_k, max_depth,
               time_limit, max_routes, draw_cache, timeout, dump_graph=False,
               scorer=None, lam=0.0):
    model = LiveBackwardModel(url, top_k, timeout, draw_cache)
    inv = StockInventory(stock)
    rec = {"target": tgt, "algo": kind}
    if stock.has(tgt):
        # Nothing to plan. Recorded rather than searched, because a target that is already
        # buyable would otherwise come back "solved with 0 routes" and inflate the solve rate.
        rec.update(n_routes=0, routes=[], solved=False, target_in_stock=True,
                   calls=0, http_calls=0, memo_hits=0, cache_hits=0, errors=0,
                   empty_retries=0, empty_final=0,
                   model_error=None, wall_s=0.0, error=None, gold_recovered=False,
                   routes_truncated=None, extract_s=0.0, max_routes=max_routes,
                   stop_reason="in_stock", budget_used_frac=0.0, n_steps=[])
        return rec
    t0 = time.time()
    gd = None
    try:
        alg = make_alg(kind, model, inv, budget, max_depth, time_limit,
                       scorer=scorer, lam=lam)
        graph, _ = alg.run_from_mol(Molecule(tgt))
        R, trunc, extract_s = routes_of(graph, max_routes,
                                        max_time_s=min(180.0, time_limit))
        if dump_graph:
            # The menu THIS target saw, not whatever the shared cache holds now: other
            # threads append to the draw cache while this search runs, so reading it back
            # can label a candidate the search never had. `model.memo` is per-target.
            gd = graph_dump(graph, tgt, stock,
                            lambda smi: model.memo.get(smi)
                            or (draw_cache.get(smi) if draw_cache is not None else None))
        err = None
    except Exception as e:                                      # noqa: BLE001
        R, trunc, extract_s, err = [], None, 0.0, f"{type(e).__name__}: {e}"[:200]
    calls = model.num_calls() if callable(model.num_calls) else model.num_calls
    found = {C.set_key(s) for s in R}
    wall = time.time() - t0
    rec.update(
        n_routes=len(R), routes=[[[p, rs] for p, rs in s] for s in R], solved=bool(R),
        target_in_stock=False,
        calls=int(calls), http_calls=model.http_calls, memo_hits=model.memo_hits,
        # empty_retries / empty_final make the empty-menu fix observable: a run with a high
        # empty_final on a set the draw cache does not cover is still losing expansions.
        empty_retries=model.empty_retries, empty_final=model.empty_final,
        cache_hits=model.cache_hits, errors=model.errors,
        model_error=model.last_error,
        wall_s=round(wall, 2), error=err,
        # WHICH limit ended the search. On a saturated fleet the per-target time limit can bind
        # before the call budget, and a run whose targets are time-bound has not been given
        # the budget it is labelled with. Recorded per target so the shortfall is visible.
        stop_reason=("budget" if calls >= budget
                     else "time" if wall >= time_limit * 0.98 else "exhausted"),
        budget_used_frac=round(calls / max(budget, 1), 3),
        routes_truncated=trunc, extract_s=extract_s, max_routes=max_routes,
        # Free correctness check: at a budget that exhausts the window the search should reach
        # the published route. A grid where nothing ever recovers gold is a broken runner, not
        # a finding about the algorithms.
        gold_recovered=bool(found & gold_setkeys),
        n_steps=[len(s) for s in R][:50],
    )
    if dump_graph:
        rec["graph"] = gd
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="fusionretro", help="paroutes | fusionretro")
    ap.add_argument("--targets", default=None,
                    help="target jsonl to read instead of data/route_search/targets_<ds>.jsonl")
    ap.add_argument("--tag", default=None,
                    help="override the run-name stem (default the --ds value), so two target "
                         "files of one dataset do not append into one another's jsonl")
    ap.add_argument("--models", default="rsmiles,localretro")
    ap.add_argument("--algos", default="retrostar,mcts")
    ap.add_argument("--budget", type=int, default=300, help="single-step model calls")
    ap.add_argument("--draws", type=int, default=1,
                    help="independent expansion draws per (target, algo). >1 only buys "
                         "anything for a stochastic model")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=8, help="reaction steps")
    ap.add_argument("--stock", default=None,
                    help="default train_<ds>; or emols / retroagent_bb / mode:/path.txt")
    ap.add_argument("--time-limit", type=float, default=600.0, help="seconds per target")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per SSR call")
    ap.add_argument("--dump-graph", action="store_true",
                    help="also record the search graph per target: every molecule opened, "
                         "whether it reached a solution, and each top-k candidate labelled "
                         "used / dead / unexpanded. See graph_dump(). Makes the run file "
                         "several times larger; AND-OR algorithms only")
    ap.add_argument("--max-routes", type=int, default=500,
                    help="routes extracted per target, in Retro*-cost order. When it binds, "
                         "the record says so (routes_truncated) — an uncounted cap turns "
                         "every routes/target and diversity number into a report of the cap")
    ap.add_argument("--limit", type=int, default=0, help="first N targets")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RP_WORKERS", "8")),
                    help="keep at or below the fleet proxy's --max-inflight "
                         "(launch_ssr_fleet*.sh: 4 x N replicas) or calls queue and time out")
    ap.add_argument("--cache-readonly", action="store_true",
                    help="read the draw cache but never write it back. ONLY for a REPLAY "
                         "of targets the cache already covers -- several processes "
                         "re-running a recorded search, where each holds the whole dict "
                         "and a flush would rewrite the shared file from one view. On a "
                         "FRESH search it throws away every menu the model produced, and "
                         "no board can be rendered for those targets. Use "
                         "--draw-cache to give each shard its own writable cache instead. "
                         "The miss count is reported at the end of the run.")
    ap.add_argument("--draw-cache", default=None,
                    help="path to the draw cache for this run, instead of the shared "
                         "draw_cache/<model>__d<draw>__k<top_k>.json. Give each shard of "
                         "a parallel FRESH search its own and merge them afterwards "
                         "(scripts/merge_draw_cache.py): that keeps the write-back, which "
                         "--cache-readonly does not.")
    ap.add_argument("--draw-cache-base", default=None,
                    help="comma-separated paths or globs of EXTRA caches to read and never "
                         "write, on top of --draw-cache. The menus of one (model, draw, "
                         "top_k) are scattered over the shared cache plus every sharded run "
                         "that was given its own; this reads all of them so a fresh search "
                         "pays only for the frontier none of them covered, while --draw-cache "
                         "stays a small per-shard DELTA that no other process can clobber. "
                         "Pass only caches of the SAME model and draw -- draw 1 is an "
                         "independent draw and the run tag records which draw it searched.")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not share a draw's menus across algorithms. Makes the algorithm "
                         "comparison confounded with sampling; only for measuring that")
    ap.add_argument("--feas-lambda", type=float, default=0.0,
                    help="weight on the feasibility term in the search's OWN cost function "
                         "(feas_cost.py): cost = -log p_model - lam * log f_step, with "
                         "f_step the geometric mean of the plausibility / round-trip / price "
                         "terms. 0 = the untouched baseline, and the baseline and "
                         "feasibility runs then differ ONLY in this number. -log f spans a "
                         "much smaller range than -log p inside one menu, so a small lam is "
                         "a tie-breaker and a large one is feasibility-dominated: sweep "
                         "it, do not guess it. For mcts it is an EXPONENT on a reward in "
                         "(0, 1], not a weight in nats -- not the same strength")
    ap.add_argument("--feas-agg", default="pct", choices=("pct", "geo", "arith"),
                    help="how the three axes combine. pct (DEFAULT) is the evaluation "
                         "composite: the mean of the three PERCENTILE RANKS inside the "
                         "molecule's own frozen menu, the same construction as "
                         "three_axis_tiers.composite and greedy_axis.rank_within. geo/arith "
                         "use the raw terms and keep absolute scale; ranks vs raw matters "
                         "far more than the choice of mean")
    ap.add_argument("--feas-rt-url", default=None,
                    help="forward model for the round-trip axis, default REACTION_FORWARD_URL "
                         "then :8090 (ReactionT5v2). The scorer probes a known reaction and "
                         "refuses to start if it does not come back at rank 1")
    ap.add_argument("--feas-overlay", default=None,
                    help="where round-trip values computed during this run are written "
                         "(default data/route_search/feas/rt_overlay_<stem>.json). The shared "
                         "roundtrip_rt5.json is READ ONLY -- several searches run at once and "
                         "a whole-file rewrite from one process's view would drop the others'")
    ap.add_argument("--feas-no-rt", action="store_true",
                    help="drop the round-trip axis (r_t = 1.0), leaving plausibility + price. "
                         "For wiring checks with no forward server, and for the ablation that "
                         "says how much of the effect is round-trip")
    ap.add_argument("--retry-errored", action="store_true",
                    help="redo only the targets whose record carries an error")
    a = ap.parse_args()

    tpath = Path(a.targets) if a.targets else C.OUT / f"targets_{a.ds}.jsonl"
    if not tpath.exists():
        raise SystemExit(f"{tpath} missing — pass --targets or build targets_{a.ds}.jsonl")
    stem = a.tag or a.ds
    T = list(C.read_jsonl(tpath))
    if a.limit:
        T = T[:a.limit]
    # A benchmark can ship WITHOUT published routes: chembl1000 carries only (id, target). `gold_recovered` is then
    # vacuously False for every target and the run prints gold=0; that is the honest reading,
    # not a broken runner. The docstring's "a grid where nothing recovers gold is broken" check
    # applies only to sets that HAVE gold.
    gold = {r["target"]: {C.set_key([(p, rs) for p, rs in s])
                          for s in (r.get("gold_routes") or [])}
            for r in T}
    if not any(gold.values()):
        print("# no published routes in this target set -- gold_recovered is vacuous, "
              "gold=0 is expected", flush=True)
    stock = C.Stock(a.stock or f"train_{a.ds}")
    print(f"{len(T)} targets, stock {stock.describe()}", flush=True)

    lam = float(a.feas_lambda)
    overlay = a.feas_overlay or (C.OUT / "feas" / f"rt_overlay_{stem}.json")
    scorer = feas_scorer(lam, a.feas_rt_url, overlay, use_rt=not a.feas_no_rt,
                         agg=a.feas_agg)
    if scorer is not None:
        print(f"feasibility-aware search: lam={lam} agg={a.feas_agg} "
              f"rt={'off' if a.feas_no_rt else 'on'} overlay={overlay}", flush=True)

    for model_name in a.models.split(","):
        url = C.SSR.get(model_name)
        if not url:
            raise SystemExit(f"no SSR url for {model_name}; set {model_name.upper()}_SSR")
        for draw in range(a.draws):
            dc = None
            if not a.no_cache:
                dc_path = a.draw_cache or str(
                    C.OUT / "draw_cache" / f"{model_name}__d{draw}__k{a.top_k}.json")
                base = []
                for pat in (a.draw_cache_base or "").split(","):
                    pat = pat.strip()
                    if pat:
                        base.extend(sorted(glob.glob(pat)) or [pat])
                dc = DrawCache(dc_path, readonly=a.cache_readonly, base=base)
                if base:
                    st = dc.stats()
                    print(f"draw cache: {st['base']:,} molecules read-only from "
                          f"{len(base)} file(s) + {st['delta']:,} in the writable delta "
                          f"{dc_path}", flush=True)
                if a.cache_readonly:
                    print(f"!! --cache-readonly: menus fetched during this run will NOT "
                          f"be written to {dc_path}. Correct for a replay, WRONG for a "
                          f"fresh search -- the board needs them on disk.", flush=True)
            for kind in a.algos.split(","):
                # lam is in the run name, not only the sidecar: a feasibility-aware run and
                # its baseline are the same (stem, model, algo, budget, draw) and would
                # otherwise APPEND INTO one another's jsonl, which the resume logic then reads
                # as "already complete".
                tag = (f"{stem}__{model_name}__{kind}__b{a.budget}"
                       + (f"__lam{lam:g}" if lam else "")
                       + (f"__{a.feas_agg}" if lam and a.feas_agg != "pct" else "")
                       + f"__d{draw}")
                path = C.RUNS / f"{tag}.jsonl"
                done = {}
                if path.exists():
                    for r in C.read_jsonl(path):
                        done[r["target"]] = r
                todo = [r for r in T if r["target"] not in done
                        or (a.retry_errored and (done[r["target"]].get("error")
                                                 or done[r["target"]].get("errors")))]
                if a.retry_errored and todo:
                    keep = [r for t, r in done.items()
                            if not (r.get("error") or r.get("errors"))]
                    C.write_jsonl(path, keep)
                    done = {r["target"]: r for r in keep}
                if not todo:
                    print(f"{tag:56} already complete ({len(done)})", flush=True)
                    continue
                agg = collections.Counter()
                t0 = time.time()
                t0_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = threading.Lock()
                with open(path, "a") as fh:
                    def work(r):
                        return one_target(r["target"], gold[r["target"]], kind, url, stock,
                                          a.budget, a.top_k, a.max_depth, a.time_limit,
                                          a.max_routes, dc, a.timeout, a.dump_graph,
                                          scorer=scorer, lam=lam)
                    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
                        for i, rec in enumerate(ex.map(work, todo)):
                            rec.update(ds=a.ds, model=model_name, draw=draw,
                                       budget=a.budget, top_k=a.top_k,
                                       max_depth=a.max_depth, stock=stock.spec,
                                       feas_lambda=lam, feas_agg=a.feas_agg,
                                       # `wall_s` is measured under `workers`-way concurrency,
                                       # so it is NOT a per-target cost -- it carries the queue
                                       # wait for the shared expander and forward fleet. The
                                       # defensible per-molecule number is the CELL's total wall
                                       # clock over n targets, at a stated concurrency; both are
                                       # in the sidecar. Recording `workers` per record so the
                                       # inflation factor is recoverable from the record alone.
                                       workers=a.workers)
                            with lock:
                                fh.write(json.dumps(rec) + "\n")
                                fh.flush()
                            agg["n"] += 1
                            agg["solved"] += rec["solved"]
                            agg["routes"] += rec["n_routes"]
                            agg["gold"] += rec["gold_recovered"]
                            agg["http"] += rec.get("http_calls", 0)
                            agg["eret"] += rec.get("empty_retries", 0)
                            agg["efin"] += rec.get("empty_final", 0)
                            agg["calls"] += rec.get("calls", 0)
                            agg["err"] += bool(rec.get("error"))
                            agg["merr"] += rec.get("errors", 0)
                            agg["instock"] += rec.get("target_in_stock", False)
                            agg["wall"] += rec.get("wall_s", 0.0)
                            agg[f"stop_{rec.get('stop_reason', 'na')}"] += 1
                            if rec.get("routes_truncated"):
                                agg[f"trunc_{rec['routes_truncated']}"] += 1
                            if (i + 1) % 20 == 0:
                                print(f"  {tag} {i+1}/{len(todo)} "
                                      f"solved={agg['solved']} routes={agg['routes']} "
                                      f"{time.time()-t0:.0f}s", flush=True)
                if dc is not None:
                    dc.flush()
                n = max(agg["n"], 1)
                print(f"{tag:56} n={agg['n']:4} solved={agg['solved']:4} "
                      f"({100*agg['solved']/n:4.1f}%) gold={agg['gold']:4} "
                      f"routes={agg['routes']:6} calls={agg['calls']:7} "
                      f"http={agg['http']:6} empty_retry={agg['eret']:5} "
                      f"empty_final={agg['efin']:4} err={agg['err']:3} model_err={agg['merr']:4} "
                      f"in_stock={agg['instock']:3} "
                      f"stop[budget/time/exh]={agg['stop_budget']}/{agg['stop_time']}/"
                      f"{agg['stop_exhausted']} "
                      f"trunc[routes/time]={agg['trunc_max_routes']}/{agg['trunc_max_time']} "
                      f"{time.time()-t0:7.0f}s", flush=True)
                wall_total = time.time() - t0
                n_done = max(agg["n"], 1)
                side = {"tag": tag, "argv": sys.argv, "stock": stock.describe(),
                        "url": url, "agg": dict(agg), "feas_lambda": lam,
                        "feas_agg": a.feas_agg,
                        # WALL CLOCK, recorded three ways because they answer different
                        # questions and are routinely confused:
                        #   wall_total_s      the cell's own elapsed time, at `workers`
                        #                     concurrency -- the only honest basis for a
                        #                     "min/mol" column, as wall_total_s / n
                        #   wall_per_target_s that quotient, precomputed
                        #   wall_s_mean       the MEAN of the per-target wall_s, which is
                        #                     larger by roughly the concurrency factor because
                        #                     each target spends most of its time queued behind
                        #                     the others. Never quote this as min/mol.
                        "workers": a.workers, "time_limit_s": a.time_limit,
                        "started_at": t0_iso, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "wall_total_s": round(wall_total, 1),
                        "wall_per_target_s": round(wall_total / n_done, 2),
                        "wall_s_mean": round(agg["wall"] / n_done, 2),
                        "concurrency_inflation": round(
                            (agg["wall"] / n_done) / max(wall_total / n_done, 1e-9), 2)}
                if scorer is not None:
                    # CUMULATIVE over every cell this process has run, because the scorer is
                    # deliberately shared: the round-trip cache is the expensive thing and
                    # per-cell counts would misreport it as re-scored work.
                    side["feas_cumulative"] = scorer.report()
                    scorer.flush()
                with open(str(path).replace(".jsonl", ".json"), "w") as fh:
                    json.dump(side, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
