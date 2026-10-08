#!/usr/bin/env python
"""Step-level feasibility, and the syntheseus costs that carry it into the search.

The route tables score plausibility, round-trip and price after a search has finished. This
module prices the same three axes into the search's own cost function, so a search can spend
its budget where the chemistry is defensible rather than only where the single-step model is
confident.

THE SCORE. Three terms per reaction, each in [FLOOR, 1], combined by one of three
aggregations (see set_agg): the mean of their percentile ranks within the molecule's menu
(`pct`, the default), a geometric mean (`geo`) or an arithmetic mean (`arith`):

    f_step = (p_t * r_t * c_t) ** (1/3)          # geo

and carried as an ADDITIVE cost, which is what A*/Retro* requires:

    cost(AndNode) = -log p_model - lam * log f_step

Why geometric for the scale-carrying form: -log of it is (1/3) * sum of the three -log terms,
so a route's total cost is -n * log(geomean over the route). The search then minimises the
route-level geometric mean of step feasibility; an arithmetic mean has no such identity.

    lam = 0 reproduces the baseline exactly (cost = -log p_model, i.e. Retro*-0, or MCTS with a
    constant reward). The baseline and the feasibility-aware search run through this one code
    path, so the only difference between them is lam -- not the runner, the stock or the
    candidate list.

THE THREE TERMS. A term whose value is 0 for most candidates is a pruner, not an objective, so
each is shaped to stay graded over the candidate space.

  p_t  plausibility -- AiZynthFinder's filter policy, P(feasible).
       Term = max(p, 0.05). Graded above the cut-off, flat below it. The raw probability is
       already a spread [0,1] quantity, but it can reach exactly 0.0, which under a geometric
       mean is -inf and would delete the candidate instead of penalising it. 0.05 is the
       plausibility cut-off used everywhere else, so it is where grading stops.

  r_t  round-trip -- rank at which a forward model (ReactionT5v2) recovers the product.
       Term = 1.0 / 0.8 / 0.6 / 0.4 / 0.2 for rank 1..5, FLOOR for not-recovered.
       A binary rank<=5 gate would zero most of every menu under a geometric mean and shrink
       the effective branching factor, so the search would mostly be deleting candidates
       rather than preferring reproducible steps. The graded form keeps the ordering the ranks
       carry, and the floor keeps an unreproduced step in the running at a price.
       The rank<=5 gate is still the reported route axis, computed post hoc by the scorers.
       Search shape and reporting bar are separate decisions.

  c_t  precursor price -- MolPrice, natively ln(USD/mmol).
       Term = clip((ln HI - ln P_step) / (ln HI - ln LO), 0, 1) with P_step the SUM of the
       step's precursor prices in USD/mmol, LO = 30, HI = 100,000.
       A linear `1 - price/1000` would clip a large share of precursors to exactly 0 and carry
       little more than one bit. LO/HI bracket the bulk of the price distribution and keep a
       gradient across it. The log scale is also the scale MolPrice is fit on, and prices span
       several orders of magnitude, so a linear normalisation is dominated by the top tail.

  Missing is not free. A term whose axis could not be measured -- unparseable SMILES, a
  fingerprint MolPrice cannot build, a forward model that did not answer -- takes FLOOR, the
  same value as a measured failure. Treating unknown as 1.0 would make "be unscoreable" the
  cheapest way to look feasible. MolPrice's own failure mode needs an explicit guard: it
  prints a warning and returns 0.0, which on the ln scale is a valid price of 1 USD/mmol, so an
  unguarded failure would enter as "very cheap".

MAGNITUDES. -log f_step spans [0, -log FLOOR] = [0, 3.0] at FLOOR=0.05, while the single-step
model's own -log p usually spans much more inside one menu. At lam=1 feasibility mostly
reorders candidates the model already considers close, so lam is a trade-off to sweep:
--feas-lambda 0 (baseline), 1 (tie-breaker), 4 (feasibility-dominated). `report()` prints the
realised cost split of a run.

WHAT IS SHARED WITH THE REST OF THE PIPELINE. Nothing is re-implemented: plausibility,
round-trip and price are reaction_mcp.scoring (the ONNX filter policy, the ReactionT5v2
server, MORetro's numpy MolPrice), and reactions are keyed by its `rxn_key`. So a step scored
during a search gets the same number the board and the tables give it, and the round-trip
cache this module reads is `data/node_scores/roundtrip_rt5.json`, the reported axis.

CACHE POLICY. Round-trip is the only axis with a per-call GPU cost, so it is the only one whose
new values are written back. The other two are cheap to recompute, but recomputing does not
always give the same number:

  plausibility  computed live, cache not loaded. The filter policy is an ONNX graph over
                Morgan count fingerprints, which are stable across rdkit versions.
  price         cache-first, computed only on a miss. MolPrice's feature vector includes rdkit
                descriptors whose values can change between rdkit versions, so reading the
                price cache first keeps the search's price axis identical to the one the tables
                report and confines any version gap to molecules not yet priced.

Usage:
  # self-test: three known reactions, all terms, against the caches
  python scripts/feas_cost.py --selftest
  # what the anchors do to this cache's distribution
  python scripts/feas_cost.py --anchors
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import traj_route_common as C


from reaction_mcp.scoring import plausibility as SP                       # noqa: E402
from reaction_mcp.scoring import roundtrip as SR                          # noqa: E402

# --------------------------------------------------------------------------- term constants
AGG = "pct"                     # "pct" | "geo" | "arith" -- see set_agg()
FLOOR = 0.05                    # what a measured failure and an unmeasurable axis both get
PLAUS_FLOOR = 0.05              # AiZynthFinder's own cut-off: where grading stops
RT_TERM = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
RT_TOP_K = 5
PRICE_LO_USD = 30.0             # lower anchor of the log price scale
PRICE_HI_USD = 100_000.0        # upper anchor
_LN_LO, _LN_HI = math.log(PRICE_LO_USD), math.log(PRICE_HI_USD)

MORETRO = C.ROOT / "external" / "MORetro"
MOLPRICE_WEIGHTS = MORETRO / "models" / "objectives" / "model_price.pkl"
NODE_SCORES = C.NODE_SCORES
RT_CACHE = NODE_SCORES / "roundtrip_rt5.json"
PRICE_CACHE = NODE_SCORES / "molprice.json"
DEFAULT_RT_URL = f"http://127.0.0.1:{os.environ.get('RP_PORT_FORWARD', '8090')}"


def rxn_key(product, reactants):
    """The shared cache key, so the caches are SHARED with the board and the tables."""
    return C.rxn_key(product, list(reactants))


# --------------------------------------------------------------------------- the three axes
class PlausScorer:
    """AiZynthFinder filter policy. Batched; ONNX sessions are thread-safe."""

    def __init__(self):
        self.n = 0

    def score(self, rxns):
        """[(product, [reactants])] -> [P(feasible) or None]."""
        if not rxns:
            return []
        self.n += len(rxns)
        return SP.score_reactions([(p, list(rs)) for p, rs in rxns])


class PriceScorer:
    """MolPrice (MORetro's numpy port), ln(USD/mmol) per molecule. CACHE-FIRST.

    Two things this wrapper exists for:

      the all-zero-fingerprint guard. MolPrice.predict prints a warning and returns 0.0,
      which on the ln scale is 1 USD/mmol -- an unguarded failure enters the cost as the
      cheapest possible precursor.

      the rdkit gap. The descriptor half of MolPrice's features is not stable across rdkit
      versions, and the price cache may have been built under a different one. A cached value
      is therefore preferred over a recomputed one -- not for speed, for agreeing with the
      tables.
    """

    def __init__(self, weights=None, cache=True):
        # MORetro is vendored, not installed in this env; moretro/__init__.py is empty and
        # molprice.py needs only rdkit/numpy/joblib, so the path insert is the whole import.
        if str(MORETRO) not in sys.path:
            sys.path.insert(0, str(MORETRO))
        from moretro.external.molprice import MolPrice           # noqa: PLC0415
        self.m = MolPrice(Path(weights or MOLPRICE_WEIGHTS))
        self.cached: dict[str, float | None] = {}     # from disk, authoritative, never written
        self.memo: dict[str, float | None] = {}       # computed in this process
        self.lock = threading.Lock()
        self.n = self.fail = self.hits = 0
        if cache and PRICE_CACHE.exists():
            t0 = time.time()
            self.cached = json.load(open(PRICE_CACHE))
            print(f"[feas] price cache {len(self.cached):,} entries "
                  f"({time.time() - t0:.0f}s) {PRICE_CACHE}", flush=True)

    def ln_price(self, smi):
        v = self.cached.get(smi, "miss")
        if v != "miss":
            self.hits += 1
            return v
        v = self.memo.get(smi, "miss")
        if v != "miss":
            return v
        with self.lock:
            self.n += 1
            try:
                fp = self.m.smi_to_fp(smi)
                v = None if fp is None or float(np.sum(fp)) == 0.0 else float(self.m.predict(smi))
            except Exception:                                    # noqa: BLE001
                v = None
            if v is None:
                self.fail += 1
            self.memo[smi] = v
        return v


class RtScorer:
    """Round-trip rank via the forward server, read-through the shared rt5 cache.

    The cache is READ from `cache/node_scores/roundtrip_rt5.json` (the reported axis) and never
    written back: several searches run at once and a whole-file rewrite from
    one process's view would drop what the others added. New values go to a per-run overlay,
    which is folded into the shared cache afterwards.

    Two failure modes are guarded:
      * a server that answers HTTP 200 with an error body -- `reaction_mcp.scoring.roundtrip` validates
        the envelope and RAISES rather than letting a chunk enter as top-k Nones;
      * the wrong server -- one that ranks from zero would turn every top-1 into rank 0.
        `probe()` refuses to start unless a known reaction comes back at rank 1.
    """

    def __init__(self, url=None, overlay=None, warm=True, top_k=RT_TOP_K, batch=32):
        # A COMMA-SEPARATED list is a fleet: one ReactionT5v2 replica per GPU, rotated per
        # request. There is no proxy in front because `roundtrip()` already takes
        # a url_override per call -- adding one would be a second process to keep alive and a
        # second place for a 200-with-an-error-body to hide.
        spec = url or os.getenv("REACTION_FORWARD_URL", DEFAULT_RT_URL)
        self.urls = [u.strip() for u in str(spec).split(",") if u.strip()]
        self.url = self.urls[0]
        self._next = 0
        self.top_k, self.batch = top_k, batch
        # Concurrency cap: without it the search's worker count sets the forward fleet's load,
        # and many workers sending beam batches at once can exhaust GPU memory, after which
        # every request fails. Default 2 in flight per replica: enough to keep a GPU busy, far
        # short of filling it.
        per = int(os.getenv("FEAS_RT_INFLIGHT_PER_REPLICA", "2"))
        self.sem = threading.BoundedSemaphore(max(1, per * len(self.urls)))
        self.max_inflight = max(1, per * len(self.urls))
        self.cache: dict[str, int | None] = {}
        self.lock = threading.Lock()
        self.hits = self.calls = self.scored = self.errors = 0
        self.overlay = Path(overlay) if overlay else None
        self.new: dict[str, int | None] = {}
        if warm and RT_CACHE.exists():
            t0 = time.time()
            self.cache = json.load(open(RT_CACHE))
            print(f"[feas] rt cache {len(self.cache):,} entries "
                  f"({time.time() - t0:.0f}s) {RT_CACHE}", flush=True)
        if self.overlay and self.overlay.exists():
            self.cache.update(json.load(open(self.overlay)))

    def _pick(self):
        with self.lock:
            u = self.urls[self._next % len(self.urls)]
            self._next += 1
        return u

    def probe(self):
        """A reaction EVERY replica must reproduce at rank 1, or nothing runs.

        Acetic anhydride + p-aminophenol -> paracetamol. Cheap, unambiguous, and it fails
        loudly on the two errors that are otherwise silent: a dead server and a rank-from-zero
        server. Returns the rank.
        """
        prod = "CC(=O)Nc1ccc(O)cc1"
        for u in self.urls:
            got = SR.roundtrip([(prod, ["CC(=O)OC(C)=O", "Nc1ccc(O)cc1"])],
                               top_k=self.top_k, batch=1, url_override=u)[0]
            if got != 1:
                raise SystemExit(
                    f"[feas] forward server {u} returned rank {got} for a reaction it must "
                    f"reproduce at rank 1. A rank of 0 means a server that ranks from zero "
                    f"-- every top-1 would read as a miss. Point REACTION_FORWARD_URL at "
                    f"ReactionT5v2 (:8090). Every "
                    f"replica is probed, because ONE bad replica in a fleet poisons the share "
                    f"of the cache that happens to land on it.")
        return 1

    def ranks(self, rxns):
        """[(product, [reactants])] -> [rank 1..top_k or None]. Cached by rxn_key."""
        keys = [rxn_key(p, rs) for p, rs in rxns]
        out: list[int | None] = [None] * len(rxns)
        todo, todo_idx = [], []
        with self.lock:
            for i, k in enumerate(keys):
                if k in self.cache:
                    out[i] = self.cache[k]
                    self.hits += 1
                else:
                    todo.append(rxns[i])
                    todo_idx.append(i)
        if todo:
            self.calls += 1
            with self.sem:
                got = SR.roundtrip(todo, top_k=self.top_k, batch=self.batch,
                                   url_override=self._pick())
            with self.lock:
                for i, r in zip(todo_idx, got):
                    out[i] = r
                    self.cache[keys[i]] = r
                    self.new[keys[i]] = r
                self.scored += len(todo)
        return out

    def flush(self):
        if not self.overlay or not self.new:
            return
        self.overlay.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self.overlay) + ".tmp"
        with self.lock:
            json.dump(self.new, open(tmp, "w"))
        os.replace(tmp, self.overlay)


# --------------------------------------------------------------------------- the score
def plaus_term(v):
    return FLOOR if v is None else max(float(v), PLAUS_FLOOR)


def rt_term(rank):
    return RT_TERM.get(rank, FLOOR) if rank is not None else FLOOR


def set_agg(kind):
    """Choose how the three terms combine.

      pct     mean of the three PERCENTILE RANKS inside the molecule's own menu -- the same
              composite the analysis uses. The default, so the search optimises the same
              statistic the tables report.
      geo     (p_t . r_t . c_t)^(1/3) on the raw terms. Its -log is additive along a route.
      arith   (p_t + r_t + c_t)/3 on the raw terms.

    The mean type (geo vs arith) changes few decisions; ranks vs raw values changes more.

    IS A PERCENTILE RANK A LEGITIMATE A* EDGE COST? Yes, in this implementation. If the rank
    depended on which branch the reaction was reached from, A*'s accumulated g would depend on
    traversal order. It does not: `LiveBackwardModel._menu` memoises the FULL top-k menu per molecule for the life of one
    search, so the comparison set of a reaction is a function of its PRODUCT alone, and
    `pct_menu` below ranks over that whole frozen menu rather than over whatever subset the
    algorithm asked for. The cost is therefore a fixed, cached property of the reaction.

    The real limitation is a different one, and it is not fixable: a percentile rank is
    MENU-RELATIVE and carries no absolute quality. The best of ten terrible options scores 1.0,
    exactly like the best of ten excellent ones. Within one molecule that is the point -- it is
    why no axis's units can dominate, and why `greedy_axis` (which only ever compares one
    molecule's options) and an MCTS policy prior (which only ever ranks one node's children) are
    its natural homes. A*, which compares partial paths reached through DIFFERENT molecules,
    loses the across-branch signal: `-lam log pct` says "this step was the best available here",
    not "this step is good". `geo` keeps absolute scale and loses comparability with the tables.
    Both are implemented; the grid reports pct and keeps geo as the ablation.
    """
    global AGG
    if kind not in ("pct", "geo", "arith"):
        raise SystemExit(f"unknown feasibility aggregation {kind!r}; pick pct, geo or arith")
    AGG = kind


def combine(p_t, r_t, c_t):
    """The scale-carrying aggregations. `pct` needs a whole menu and lives in pct_menu()."""
    if AGG == "arith":
        return (p_t + r_t + c_t) / 3.0
    return (p_t * r_t * c_t) ** (1.0 / 3.0)


def pct_arith(terms):
    """terms: [(p_t, r_t, c_t)] for ONE menu -> [mean of the three percentile ranks].

    The analysis construction: `rankdata(..., method="average")` per axis, mapped to [0, 1] as
    (rk - 1)/(N - 1), then averaged over axes. A single-candidate menu gets 1.0 --
    a rank is undefined for N=1 and the candidate is the only choice anyway.
    """
    from scipy.stats import rankdata                                       # noqa: PLC0415
    n = len(terms)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    T = np.asarray(terms, dtype=float)
    cols = []
    for j in range(T.shape[1]):
        r = rankdata(T[:, j], method="average")
        cols.append((r - 1.0) / (n - 1.0))
    return np.mean(np.vstack(cols), axis=0).tolist()


def price_term(ln_prices):
    """Sum the step's precursor prices in USD, normalise on the LOG scale."""
    if not ln_prices or any(x is None for x in ln_prices):
        return FLOOR
    total = sum(math.exp(x) for x in ln_prices)
    if total <= 0:
        return FLOOR
    t = (_LN_HI - math.log(total)) / (_LN_HI - _LN_LO)
    return min(1.0, max(FLOOR, t))


class FeasScorer:
    """The three axes behind one call. Thread-safe; one instance per search process."""

    def __init__(self, rt_url=None, rt_overlay=None, rt_warm=True, use_rt=True,
                 probe=True, price_cache=True):
        self.plaus = PlausScorer()
        self.price = PriceScorer(cache=price_cache)
        self.rt = RtScorer(rt_url, rt_overlay, warm=rt_warm) if use_rt else None
        self.use_rt = use_rt
        if self.rt is not None and probe:
            self.rt.probe()
        self.memo: dict[str, dict] = {}
        self._menu_f: dict[str, dict] = {}      # product -> {rxn_key: f}, for AGG == "pct"
        self.lock = threading.Lock()
        self.n_scored = 0
        self.sum_neg_log_f = 0.0
        self.n_menus = 0

    def score(self, rxns):
        """[(product, [reactants])] -> [{plaus, rt, price_usd, p_t, r_t, c_t, f}].

        Batched on purpose: the caller is one expansion's worth of candidates (<=top_k), so
        the round-trip server sees one request per expansion rather than one per candidate.
        """
        if not rxns:
            return []
        keys = [rxn_key(p, rs) for p, rs in rxns]
        out: list[dict | None] = [None] * len(rxns)
        todo_idx = []
        with self.lock:
            for i, k in enumerate(keys):
                hit = self.memo.get(k)
                if hit is not None:
                    out[i] = hit
                else:
                    todo_idx.append(i)
        if todo_idx:
            sub = [rxns[i] for i in todo_idx]
            pl = self.plaus.score(sub)
            rk = self.rt.ranks(sub) if self.rt is not None else [None] * len(sub)
            for j, i in enumerate(todo_idx):
                prod, reac = rxns[i]
                lnp = [self.price.ln_price(x) for x in reac]
                p_t, r_t = plaus_term(pl[j]), (rt_term(rk[j]) if self.use_rt else 1.0)
                c_t = price_term(lnp)
                f = combine(p_t, r_t, c_t)
                rec = {"plaus": pl[j], "rt": rk[j],
                       "price_usd": (None if any(x is None for x in lnp)
                                     else round(sum(math.exp(x) for x in lnp), 3)),
                       "p_t": round(p_t, 5), "r_t": round(r_t, 5), "c_t": round(c_t, 5),
                       "f": f}
                with self.lock:
                    self.memo[keys[i]] = rec
                    self.n_scored += 1
                    if f is not None:
                        self.sum_neg_log_f += -math.log(f)
                out[i] = rec
        return out

    def menu_f(self, product, menu):
        """{rxn_key: f} for a molecule's WHOLE frozen menu. The only entry point for `pct`.

        menu: an iterable of reactant lists -- `LiveBackwardModel.memo[product]`'s candidates,
        the full top-k, NOT whatever subset the algorithm asked for this time. Ranking over the
        full menu is what makes the cost a function of the reaction rather than of the call.

        Cached per product, so the ranks a reaction gets are computed once per search.
        """
        hit = self._menu_f.get(product)
        if hit is not None:
            return hit
        menu = [list(rs) for rs in menu]
        if not menu:
            return {}
        recs = self.score([(product, rs) for rs in menu])
        if AGG == "pct":
            vals = pct_arith([(r["p_t"], r["r_t"], r["c_t"]) for r in recs])
            # A rank of exactly 0.0 -- last on all three axes -- would be an INFINITE cost and
            # would delete the candidate rather than deprioritise it. Floored at half a rank step, which is one
            # notch below the smallest achievable non-zero rank and scales with the menu.
            floor = 1.0 / (2.0 * max(len(menu) - 1, 1))
            vals = [max(v, floor) for v in vals]
        else:
            vals = [r["f"] for r in recs]
        out = {}
        for rs, r, v in zip(menu, recs, vals):
            v = float(v)
            r["f"] = v                      # the record now carries the f the cost used
            out[rxn_key(product, rs)] = v
            with self.lock:
                self.sum_neg_log_f += -math.log(v)
        with self.lock:
            self._menu_f[product] = out
            self.n_menus += 1
        return out

    def f(self, rxns, menu_fn=None):
        """Per-reaction f. For `pct` a menu_fn(product) -> menu is REQUIRED, because the value
        does not exist outside a menu; without one the reaction is scored alone and its rank is
        1.0 by definition, which would silently make every step look best-in-class."""
        if AGG != "pct":
            return [r["f"] for r in self.score(rxns)]
        if menu_fn is None:
            raise RuntimeError(
                "AGG='pct' needs menu_fn: a percentile rank is defined only inside a menu, and "
                "scoring a reaction on its own would give it rank 1.0 unconditionally.")
        out = []
        for prod, reac in rxns:
            m = self.menu_f(prod, menu_fn(prod) or [list(reac)])
            out.append(m.get(rxn_key(prod, list(reac)), 1.0))
        return out

    def report(self):
        n = max(self.n_scored, 1)
        return {"scored": self.n_scored, "agg": AGG, "menus_ranked": self.n_menus,
                "mean_neg_log_f": round(self.sum_neg_log_f / n, 4),
                "plaus_calls": self.plaus.n,
                "price_cache_hits": self.price.hits,
                "price_computed": self.price.n, "price_fail": self.price.fail,
                "rt_inflight_cap": self.rt.max_inflight if self.rt else None,
                "rt_hits": self.rt.hits if self.rt else 0,
                "rt_scored": self.rt.scored if self.rt else 0,
                "rt_requests": self.rt.calls if self.rt else 0}

    def flush(self):
        if self.rt is not None:
            self.rt.flush()


# --------------------------------------------------------------------------- syntheseus hooks
def _evaluators():
    """Imported lazily: the MORetro env has no syntheseus, and it uses the terms above."""
    from syntheseus.search.graph.and_or import AndNode                     # noqa: PLC0415
    from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator  # noqa: PLC0415

    class FeasibilityCost(NoCacheNodeEvaluator):
        """cost(AndNode) = -log p_model - lam * log f_step.  lam=0 IS ReactionModelLogProbCost.

        The confidence half is recomputed here rather than delegated to
        ReactionModelLogProbCost so that the two halves share one clip and one code path; the
        clip bounds are that class's defaults (1e-10, 0.999), so lam=0 reproduces it exactly.
        """

        def __init__(self, scorer, lam=1.0, menu_fn=None,
                     clip_min=1e-10, clip_max=0.999, **kw):
            super().__init__(**kw)
            self.scorer, self.lam = scorer, float(lam)
            self.menu_fn = menu_fn
            self.clip_min, self.clip_max = clip_min, clip_max
            if self.lam and AGG == "pct" and menu_fn is None:
                raise RuntimeError(
                    "feas-agg=pct needs menu_fn(product) -> the molecule's frozen menu. The "
                    "nodes handed to an evaluator are only the subset the algorithm asked for, "
                    "so ranking over them would make the cost depend on the call rather than "
                    "on the reaction.")

        def _evaluate_nodes(self, nodes, graph=None):
            p = np.clip(np.asarray([float(n.reaction.metadata.get("probability", self.clip_min))
                                    for n in nodes]), self.clip_min, self.clip_max)
            cost = -np.log(p)
            if self.lam:
                # Grouped by product, one menu ranking per molecule (cached in the scorer), so
                # `pct` and the scale-carrying aggregations take the same path here.
                fs = self.scorer.f(
                    [(n.reaction.product.smiles,
                      sorted(m.smiles for m in n.reaction.reactants)) for n in nodes],
                    menu_fn=self.menu_fn)
                cost = cost - self.lam * np.log(np.asarray(fs))
                for n, f in zip(nodes, fs):
                    # kept on the node so a --dump-graph run can show WHY a branch was
                    # preferred, and so the run file carries the terms the cost was built from
                    k = rxn_key(n.reaction.product.smiles,
                                sorted(m.smiles for m in n.reaction.reactants))
                    rec = dict(self.scorer.memo.get(k) or {})
                    rec["f"] = float(f)
                    n.reaction.metadata["feas"] = rec
            return cost.tolist()

    class FeasibilityReward(NoCacheNodeEvaluator):
        """MCTS terminal reward = f_path ** lam, f_path the geomean of f over the root path.

        MolSetMCTS keeps the reaction on the EDGE, so the path's steps are read off the graph
        rather than the node -- the same trap `routes_of` documents. lam=0 returns 1.0 for
        every terminal node, which is ConstantNodeEvaluator(1.0), the baseline arm.
        """

        def __init__(self, scorer, lam=1.0, menu_fn=None, **kw):
            super().__init__(**kw)
            self.scorer, self.lam = scorer, float(lam)
            self.menu_fn = menu_fn

        def _path_rxns(self, node, graph):
            steps, cur = [], node
            while True:
                preds = list(graph.predecessors(cur))
                if not preds:
                    return steps
                par = preds[0]
                r = graph._graph.edges[par, cur].get("reaction")
                if r is not None:
                    steps.append((r.product.smiles,
                                  sorted(m.smiles for m in r.reactants)))
                cur = par

        def _evaluate_nodes(self, nodes, graph=None):
            if not self.lam:
                return [1.0] * len(nodes)
            out = []
            for n in nodes:
                steps = self._path_rxns(n, graph)
                if not steps:
                    out.append(1.0)
                    continue
                fs = self.scorer.f(steps, menu_fn=self.menu_fn)
                out.append(float(np.exp(np.mean(np.log(fs))) ** self.lam))
            return out

    return AndNode, FeasibilityCost, FeasibilityReward


def feasibility_cost(scorer, lam=1.0, menu_fn=None):
    return _evaluators()[1](scorer, lam, menu_fn=menu_fn)


def feasibility_reward(scorer, lam=1.0, menu_fn=None):
    return _evaluators()[2](scorer, lam, menu_fn=menu_fn)


# --------------------------------------------------------------------------- CLI
def _selftest(rt_url, use_rt=True):
    rxns = [
        # acylation: real chemistry, cheap precursors
        ("CC(=O)Nc1ccc(O)cc1", ["CC(=O)OC(C)=O", "Nc1ccc(O)cc1"]),
        # Suzuki: real, boronic acid is priced but not cheap
        ("c1ccc(-c2ccccc2)cc1", ["OB(O)c1ccccc1", "Brc1ccccc1"]),
        # nonsense: the filter policy should reject it
        ("CC(=O)Nc1ccc(O)cc1", ["CCCCCCCCCC", "c1ccccc1"]),
    ]
    s = FeasScorer(rt_url=rt_url, rt_overlay=None, rt_warm=use_rt, use_rt=use_rt)
    if not use_rt:
        print("  !! --no-rt: r_t is pinned to 1.0, so f is the plausibility/price pair only. "
              "For wiring checks while the forward server is down, never for a run.")
    for (p, r), rec in zip(rxns, s.score(rxns)):
        print(f"  {p[:34]:34} <- {'.'.join(r)[:40]:40} "
              f"plaus={rec['plaus'] if rec['plaus'] is None else round(rec['plaus'], 4):>7} "
              f"rt={str(rec['rt']):>4} price={str(rec['price_usd']):>10} "
              f"| p_t={rec['p_t']:.3f} r_t={rec['r_t']:.3f} c_t={rec['c_t']:.3f} "
              f"f={rec['f']:.4f} -log f={-math.log(rec['f']):.3f}")
    print(" ", s.report())


def _anchors():
    """What the chosen anchors and floors do to the caches' own distributions."""
    nd = NODE_SCORES
    pl = json.load(open(nd / "plausibility.json"))
    v = np.fromiter((x for x in pl.values() if x is not None), dtype=float)
    t = np.maximum(v, PLAUS_FLOOR)
    print(f"plausibility  n={len(v):,}  >=0.05 {100 * (v >= 0.05).mean():.1f}%  "
          f"term: mean {t.mean():.3f} floored {100 * (v < PLAUS_FLOOR).mean():.1f}% "
          f"-log mean {(-np.log(t)).mean():.3f}")
    rt = json.load(open(nd / "roundtrip_rt5.json"))
    rk = list(rt.values())
    tr = np.asarray([rt_term(x) for x in rk])
    got = sum(1 for x in rk if x is not None and x <= 5)
    print(f"round-trip    n={len(rk):,}  <=5 {100 * got / len(rk):.1f}%  "
          f"rank1 {100 * sum(1 for x in rk if x == 1) / len(rk):.1f}%  "
          f"term: mean {tr.mean():.3f} floored {100 * (tr == FLOOR).mean():.1f}% "
          f"-log mean {(-np.log(tr)).mean():.3f}")
    mp = json.load(open(nd / "molprice.json"))
    lp = np.fromiter((x for x in mp.values() if x is not None), dtype=float)
    usd = np.exp(lp)
    tc = np.clip((_LN_HI - lp) / (_LN_HI - _LN_LO), FLOOR, 1.0)
    lin = np.clip(1 - usd / 1000.0, 0.0, 1.0)
    print(f"price         n={len(lp):,}  median {np.median(usd):,.0f} USD/mmol  "
          f"term: mean {tc.mean():.3f} clipped-low {100 * (tc <= FLOOR).mean():.2f}% "
          f"clipped-high {100 * (tc >= 1.0).mean():.2f}% -log mean {(-np.log(tc)).mean():.3f}")
    print(f"              (1-usd/1000 for comparison: mean {lin.mean():.3f}, "
          f"exactly 0 for {100 * (lin == 0).mean():.1f}% -- the reason it is not used)")
    # The per-molecule distribution above is NOT what the search sees. c_t is computed on the
    # SUM over a step's precursors, so a two-precursor step is dearer than either input, and it
    # has to be measured through the LIVE scorer: reading molprice.json alone floors every step
    # with an unpriced precursor, which is the cache's coverage, not the axis. PriceScorer
    # computes a miss instead of failing it, so both numbers are printed and labelled.
    import random
    random.seed(0)
    keys = random.sample(list(pl.keys()), min(5_000, len(pl)))
    ps = PriceScorer()
    t0 = time.time()
    ct, ct_cacheonly, npre, unpriced = [], [], [], 0
    for k in keys:
        reac = k.split(">>", 1)[1].split(".")
        lnp_cache = [mp.get(x) for x in reac]
        if any(x is None for x in lnp_cache):
            unpriced += 1
        ct_cacheonly.append(price_term(lnp_cache))
        ct.append(price_term([ps.ln_price(x) for x in reac]))
        npre.append(len(reac))
    ct, cc = np.asarray(ct), np.asarray(ct_cacheonly)
    print(f"price (STEP sums, n={len(ct):,} real reactions, mean {np.mean(npre):.2f} "
          f"precursors, LIVE scorer): c_t mean {ct.mean():.3f} median {np.median(ct):.3f} "
          f"at FLOOR {100 * (ct <= FLOOR).mean():.1f}% at 1.0 {100 * (ct >= 1.0).mean():.1f}% "
          f"-log mean {(-np.log(ct)).mean():.3f}")
    print(f"              cache-only for comparison: at FLOOR {100 * (cc <= FLOOR).mean():.1f}% "
          f"-- a precursor is unpriced in {100 * unpriced / len(keys):.1f}% of steps, and "
          f"{ps.n:,} molecules were computed here at {ps.n / max(time.time() - t0, 1e-9):.0f}/s")
    print(f"anchors: LO {PRICE_LO_USD:,.0f} HI {PRICE_HI_USD:,.0f} USD/mmol, "
          f"FLOOR {FLOOR}, max -log f = {-math.log(FLOOR):.2f} nats")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--rt-url", default=None)
    ap.add_argument("--no-rt", action="store_true",
                    help="skip the round-trip axis entirely (r_t = 1.0). Wiring check only")
    a = ap.parse_args()
    if a.anchors:
        _anchors()
    if a.selftest:
        _selftest(a.rt_url, use_rt=not a.no_rt)
    if not (a.anchors or a.selftest):
        ap.error("pick --selftest or --anchors")


if __name__ == "__main__":
    sys.exit(main())
