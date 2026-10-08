#!/usr/bin/env python
"""One target -> one rendered episode, joined from the three offline sources.

No tools are re-run.  Everything on screen was recorded:

  #1  data/route_search/draw_cache/<model>__d0__k10.json
      {product: [[[reactants], model_confidence] x 10]}  -- THE MENU.  This is
      what the test-time environment hands over, so reproducing it here is what
      makes the training observation identical to the inference one.

  #2  data/route_search/runs/<ds>__<model>__<algo>__b*__d0.jsonl
      graph.molecules[] with in_stock / has_solution / depth and per-candidate
      used|dead|unexpanded -- the ONLY source of dead ends.  A trajectory built
      from optimal routes alone contains no `dead` label at all.

  #3  data/route_search/sft_candidates_<ds>.jsonl
      the selected routes and their per-step / per-leaf evidence.  Used for the
      target list, gold comparison, and the numbers a <think> may cite.

Prices come from the MolPrice csv dumps, keyed by canonical SMILES; a buyable
leaf with no row is rendered as buyable-and-unpriced rather than as free.

The label policy is the DP, not the recorded route: value(mol) = max over
candidates of min(axis, min over pieces value(piece)), a leaf in stock being
unconstrained and a molecule out of depth being -inf.  `rank` lists exactly the
candidates with a finite value, ordered by it; `dead` fires when that set is
empty.  The states come from wherever the driver walks -- see rollout_policy --
so the mistakes and the recovery come from different places on purpose.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from .state import (Analyze, Board, BoardError, Candidate, Dead, Done, NoMenu,
                    Open, Rank, Terminate)

SD = Path(__file__).resolve().parent
# config/paths.py, loaded by file location under its own name: the analysis pipeline has
# a module called `paths` of its own, and that one must keep the name.
_spec = importlib.util.spec_from_file_location("rp_paths", SD.parents[2] / "config" / "paths.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)
ROOT = Path(RP.MULTISTEP)                     # tools/reaction-mcp
RS = ROOT / "data" / "route_search"
PRICE_DIR = ROOT / "external" / "MolPrice"
NODE_SCORES = Path(RP.NODE_SCORES)

INF = float("inf")


# ------------------------------------------------------------------- sources
def load_menus(path: Path) -> dict[str, list[tuple[list[str], float]]]:
    raw = json.loads(Path(path).read_text())
    return {k: [(r, float(q)) for r, q in v] for k, v in raw.items()}


def load_candidate_routes(path: Path, sft_only: bool = True
                          ) -> dict[str, list[dict]]:
    """target -> its selected routes, best pareto_rank first.

    These are the diverse routes the selection already made: several ways to the
    same target differing in feasibility, cost and shape.  One route is one
    training instance, so a target with seven routes on the front contributes
    seven -- the same board, seven different sets of decisions defended by
    different numbers.
    """
    out: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if sft_only and not rec.get("sft_candidate"):
                continue
            out.setdefault(rec["target"], []).append(rec)
    for rows in out.values():
        rows.sort(key=lambda r: (r.get("pareto_rank", 99), r.get("first_rank", 99)))
    return out


def route_map(steps) -> dict[str, list[str]]:
    """product -> its reactants, for the route being followed."""
    return {p: list(rs) for p, rs in steps}


class OffRoute(Exception):
    """The route's own disconnection is not on the board's menu (or is behind the
    fold).  The instance is dropped whole rather than relabelled: teaching a
    choice the model was never shown is worse than one instance fewer.  This is
    the exclusion rule: a route whose step is outside the top-10 menu is not
    replayed.
    """


def match_candidate(menu, reactants: list[str]) -> Optional[int]:
    want = sorted(reactants)
    for c in menu:
        if sorted(c.reactants) == want:
            return c.idx
    return None


class HttpMenus:
    """Menus from a LIVE single-step server instead of the recorded draw cache.

    The validation set is built through this so that the boards it holds are the
    ones the model is served, not the ones an older dump happened to hold.
    Contract of ``predict_standalone.py`` (R-SMILES / root_aligned):

        POST {url}  {"smiles": "...", "top_n": 10}
          -> [{"precursors": [...], "confidence": 0.54}, ...]  best first

    `budget` bounds the damage: the DP recurses over whatever the menus reach, so
    an unbounded live build can issue a very large number of calls.  Once the budget
    is spent, lookups fall back to `fallback` (the recorded cache) and the count
    of fallbacks is reported -- a val set that quietly finished on the cache would
    otherwise look like a val set built live.
    """

    def __init__(self, url: str, top_k: int = 10, timeout: float = 180.0,
                 fallback: Optional[dict] = None, budget: int = 50_000):
        self.url = url
        self.top_k = top_k
        self.timeout = timeout
        self.fallback = fallback or {}
        self.budget = budget
        self.cache: dict[str, list[tuple[list[str], float]]] = {}
        self.stats = {"live": 0, "fallback": 0, "error": 0, "agree": 0,
                      "disagree": 0}

    def _fetch(self, smiles: str):
        import json as _json
        import urllib.error
        import urllib.request

        body = _json.dumps({"smiles": smiles, "top_n": self.top_k}).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                rows = _json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            self.stats["error"] += 1
            return None
        if isinstance(rows, dict):          # the proxy reports failures as a dict
            self.stats["error"] += 1
            return None
        out = []
        for r in rows[: self.top_k]:
            pre = r.get("precursors") or r.get("reactants")
            if not pre:
                continue
            out.append((list(pre), float(r.get("confidence", 0.0))))
        return out

    def get(self, smiles: str, default=None):
        if smiles in self.cache:
            return self.cache[smiles]
        rows = None
        if self.stats["live"] < self.budget:
            rows = self._fetch(smiles)
            if rows is not None:
                self.stats["live"] += 1
                cached = self.fallback.get(smiles)
                if cached is not None:
                    same = [tuple(sorted(r)) for r, _ in rows[: self.top_k]] == \
                           [tuple(sorted(r)) for r, _ in cached[: self.top_k]]
                    self.stats["agree" if same else "disagree"] += 1
        if rows is None:
            rows = self.fallback.get(smiles)
            if rows is not None:
                self.stats["fallback"] += 1
        self.cache[smiles] = rows or []
        return self.cache[smiles] or default


def load_prices(paths: Iterable[Path]) -> dict[str, float]:
    """smi_can -> ln(USD/mmol).  Later files win, which is the newer dump."""
    out: dict[str, float] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out[row["smi_can"]] = float(row["price"])
                except (KeyError, TypeError, ValueError):
                    continue
    return out


def price_files() -> list[Path]:
    return sorted(PRICE_DIR.glob("prices*.csv"))


def graph_index(run: Path) -> dict[str, int]:
    """target -> byte offset, cached next to the run so it is built once."""
    run = Path(run)
    cache = run.parent / ".index" / (run.name + ".idx.json")
    if cache.exists() and cache.stat().st_mtime >= run.stat().st_mtime:
        return json.loads(cache.read_text())
    idx: dict[str, int] = {}
    with open(run, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                idx[json.loads(line)["target"]] = off
            except Exception:
                continue
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def load_graph(run: Path, target: str, idx: dict[str, int] = None) -> dict:
    idx = idx if idx is not None else graph_index(run)
    with open(run, "rb") as f:
        f.seek(idx[target])
        return json.loads(f.readline())


# ------------------------------------------------------------- node scores
@dataclass
class Scores:
    """The per-reaction and per-molecule caches, keyed the way they are on disk.

    A reaction key is `product>>reactant.reactant`.  The reactant order is the
    one the single-step model returned, but the caches were written from several
    producers, so a miss on that order is retried on the sorted one -- and the
    two are counted so a systematic mismatch cannot hide as a coverage number.

    With these caches every candidate carries p, rt and a price, which is what
    makes p the deciding axis rather than q.
    """

    plaus: dict = field(default_factory=dict)
    rt: dict = field(default_factory=dict)
    ord: dict = field(default_factory=dict)
    ord_journal: dict = field(default_factory=dict)
    price: dict = field(default_factory=dict)
    scscore: dict = field(default_factory=dict)
    hits: dict = field(default_factory=lambda: {"given": 0, "sorted": 0, "miss": 0})

    def _get(self, table: dict, product: str, reactants: list[str]):
        k = f"{product}>>{'.'.join(reactants)}"
        if k in table:
            self.hits["given"] += 1
            return table[k]
        k = f"{product}>>{'.'.join(sorted(reactants))}"
        if k in table:
            self.hits["sorted"] += 1
            return table[k]
        # The cache writer's rxn_key DEDUPLICATES before joining -- `sorted(set(reactants))`
        # -- so a step whose precursors repeat is stored under one copy and neither form
        # above finds it. Without this it would render as `rt?`/no `p` while its value sat
        # in the cache: an absence the board asserted and the teacher could read as evidence.
        k = f"{product}>>{'.'.join(sorted(set(reactants)))}"
        if k in table:
            self.hits["deduped"] = self.hits.get("deduped", 0) + 1
            return table[k]
        self.hits["miss"] += 1
        return None

    def _lookup(self, table: dict, product: str, reactants: list[str]):
        """(value, keyed). `_get` collapses two different facts into one None.

        A key absent from the cache and a key present with a null value are not the same
        thing: the first says nothing was asked, the second that the forward model ran and
        did not recover the product. Rendered identically, a sentence reading `rt-` as "the
        forward model could not recover it" would be reading the absence of a cache entry as
        evidence. `signals` already applies the right principle to
        `p` -- "an absent filter score is not a failed one", so it omits it -- and this is
        what lets `rt` say the same thing.
        """
        # Three spellings, and the third is the one the WRITER uses: its rxn_key builds
        # `sorted(set(reactants))`. Without it a candidate that repeats a precursor -- a
        # symmetric coupling, a double deprotection -- misses a value the cache holds. Rare,
        # so this is correctness, not a lever.
        for key in (f"{product}>>{'.'.join(reactants)}",
                    f"{product}>>{'.'.join(sorted(reactants))}",
                    f"{product}>>{'.'.join(sorted(set(reactants)))}"):
            if key in table:
                return table[key], True
        return None, False

    def signals(self, product: str, reactants: list[str], q: float) -> dict:
        """The signal group for one candidate, in the order it is rendered.

        `rt` comes in THREE states, not two, and they are rendered apart: a rank
        (`rt1`), a recorded non-recovery (`rt✗`), and no cache entry at all
        (`rt?`). "The forward model could not recover this product" is
        information; "nobody asked it" is not, and collapsing them would make
        the second readable as the first. `rt_keyed`
        carries which it was so a checker can refuse a citation on `rt?`.
        A missing p is left out entirely rather than shown as zero -- an absent
        filter score is not a failed one, and that is the same principle.
        """
        out = {"q": q}
        p = self._get(self.plaus, product, reactants)
        if p is not None:
            out["p"] = float(p)
        if self.rt:
            v, keyed = self._lookup(self.rt, product, reactants)
            out["rt"] = v
            out["rt_keyed"] = keyed
        return out


def load_node_scores(path: Path = None, want=("plaus", "rt", "price")) -> Scores:
    """Load only the caches asked for -- the plausibility cache alone is large."""
    path = Path(path or NODE_SCORES)
    files = {"plaus": "plausibility.json", "rt": "roundtrip_rt5.json",
             "ord": "ord.json", "ord_journal": "ord_journal.json",
             "price": "molprice.json", "scscore": "scscore.json"}
    # `*_filled.json` holds entries scored later by the fill_* scripts with the SAME model
    # as the base cache. They are merged, not kept apart: a board that reads only the base cache still refuses
    # a candidate total whenever one piece happens to be a cache miss, which is the thing
    # the fill exists to stop.
    extra = {"plaus": "plausibility_filled.json", "rt": "roundtrip_filled.json",
             "price": "molprice_filled.json"}
    kw = {}
    for key in want:
        f = path / files[key]
        if f.exists():
            kw[key] = json.loads(f.read_text())
        g = path / extra.get(key, "")
        if key in extra and g.exists():
            kw.setdefault(key, {}).update(json.loads(g.read_text()))
    return Scores(**kw)


# --------------------------------------------------------------------- world
@dataclass
class DataWorld:
    """The recorded environment: menus, stock, prices, and the run's own graph.

    `scope` is the set of molecules the search actually expanded.  A molecule
    outside it has no menu even if the shared draw cache happens to hold one --
    the run never asked, so neither can the board.  That keeps the DP inside the
    evidence and stops a label depending on a candidate the graph never labelled.
    """

    menus: dict[str, list[tuple[list[str], float]]]
    stock: set[str]
    prices: dict[str, float]
    scope: set[str]
    top_k: int = 10
    scores: Optional[Scores] = None

    def menu(self, smiles: str) -> list[Candidate]:
        rows = self.menus.get(smiles, [])[: self.top_k]
        if self.scores is None:
            return [Candidate(i, list(r), {"q": q}) for i, (r, q) in enumerate(rows)]
        return [Candidate(i, list(r), self.scores.signals(smiles, list(r), q))
                for i, (r, q) in enumerate(rows)]

    def info(self, smiles: str) -> tuple[bool, Optional[float]]:
        buyable = smiles in self.stock
        # Price is NOT gated on buyability. MolPrice predicts a price for an arbitrary
        # molecule (metrics.load_prices: "MolPrice predictions, ln(USD/mmol)"), so the
        # number exists for a molecule nobody sells and means the same thing there. Tying
        # the two would make `$` a solve marker rather than a cost: it would appear exactly
        # when the branch closed, so the two signals would be perfectly confounded and the
        # model would have no way to read price as an axis.
        p = None
        if self.scores is not None and self.scores.price:
            v = self.scores.price.get(smiles)
            if v is not None:
                p = float(v)
        if p is None:
            p = self.prices.get(smiles)
        return buyable, p

    def has_menu(self, smiles: str) -> bool:
        return smiles in self.scope and bool(self.menus.get(smiles))



def _decided_by(board, dp, ranked: dict) -> dict:
    """{mid: {axis, leaders, values, taken}} -- what separated the head of each ranking.

    Derived from the DP's OWN ordering, not re-derived from the menu: `DP.order` buckets on
    the DP value with width tie_eps and then applies `_tie_key`, so a reconstruction that
    reads `p` off the candidate rows can name a different candidate than the one the action
    took.

    Why it exists at all: the teacher explains an action it did not choose, and the screen
    shows every number but not which one was decisive. Left to infer the reason, the teacher
    rarely names a tie-break axis (price in particular) even where it decided the ranking.
    """
    out = {}
    for mid, order in (ranked or {}).items():
        m = board.mols.get(mid)
        if not m or not m.menu or not order:
            continue
        try:
            ranking = dp.order(board, mid)
        except Exception:
            continue
        if len(ranking) < 2:
            continue
        by_idx = {c.idx: c for c in m.menu}
        eps = max(dp.tie_eps, 1e-12)
        top = round(ranking[0][1] / eps)
        tied = [i for i, v in ranking if round(v / eps) == top]
        if len(tied) == 1:
            axis = "p"
            vals = {str(i): v for i, v in ranking[:4]}
        else:
            axis, vals = "none", {}
            for k in dp.tiebreak:
                # AVAILABILITY SEPARATES BEFORE VALUE DOES, because `_tie_key` sorts on it.
                # `_tie_key` maps a missing rt to 99 -- last -- and an incomplete cost to INF
                # -- last -- so a tied set holding one candidate with the axis and three
                # without is DECIDED at that slot, and the next axis is never consulted.
                # Dropping the absent ones and asking whether the rest differ would report
                # the wrong separator -- often `q`, whose own argmax need not be the candidate
                # taken, so the brief would tell the teacher to name an axis that
                # `_c_q_argmax` then refutes.
                if k in ("rt", "price"):
                    have = {i for i in tied
                            if (dp._complete_cost(by_idx[i]) != INF if k == "price"
                                else isinstance((by_idx[i].signals or {}).get("rt"),
                                                (int, float)))
                            if by_idx.get(i) is not None}
                    if have and len(have) < len(tied):
                        axis = "closure" if k == "price" else "rt_missing"
                        vals = {str(i): (i in have) for i in tied}
                        break
                seen = {}
                for i in tied:
                    c = by_idx.get(i)
                    if c is None:
                        continue
                    if k == "price":
                        v = dp._complete_cost(c)
                        v = None if v == INF else v
                    elif k == "rt":
                        v = c.signals.get("rt")
                        v = v if isinstance(v, (int, float)) else None
                    else:
                        v = c.signals.get(k)
                    if v is not None:
                        seen[str(i)] = v
                if len({round(v, 6) for v in seen.values()}) > 1:
                    axis, vals = k, seen
                    break
        # DOES THE NAMED AXIS ACTUALLY POINT AT THE CANDIDATE TAKEN? Often the action's
        # head is not in `tied` at all -- the rank serves a route the queue still needs
        # rather than the local menu, and `taken` then comes from a set this comparison
        # never contained ("the leaders c0, c1 tie ... c7 is the one taken"). Both facts
        # are recorded so the brief can refuse to name an axis instead of inventing one.
        taken = order[0]
        earned = taken in tied
        if earned and axis in ("p", "rt", "price", "q"):
            best, bv = None, None
            for i in tied:
                c = by_idx.get(i)
                if c is None:
                    continue
                if axis == "price":
                    v = dp._complete_cost(c)
                    v = None if v == INF else -v          # cheaper is better
                elif axis == "rt":
                    v = c.signals.get("rt")
                    v = -float(v) if isinstance(v, (int, float)) else None
                else:
                    v = c.signals.get(axis)
                if v is not None and (bv is None or v > bv):
                    best, bv = i, v
            earned = best is None or best == taken
        out[mid] = {"axis": axis, "leaders": tied, "taken": taken,
                    "values": vals, "taken_in_leaders": taken in tied, "earned": earned}
    return out


def _price_note(board, ranked: dict) -> dict:
    """{mid: {taken, taken_cost, dearest, spread}} where the taken cut is ALSO the cheapest.

    A second, weaker claim than `decided_by`, and deliberately so. `ordered by: price` can
    only be written where price actually separated the leaders, which is rare --
    `DP._complete_cost` is infinite unless every piece is purchasable, so the axis is rarely
    available. But "the cut we took is the cheapest on the menu, by 10x" is true far more
    often and is a fact a draft may cite under whatever axis DID decide.

    Without it a teacher tends to mention a price only as the deciding axis, so a board full
    of candidate prices yields a corpus that argues from none of them. The note gives the
    draft the comparison already made, so it does not have to find it and cannot get it
    wrong.

    Only emitted when the taken candidate is strictly cheapest AND the menu's spread is at
    least 3x: a note on a menu whose cuts all cost the same would teach a distinction that
    is not there.
    """
    out = {}
    for mid, order in (ranked or {}).items():
        m = board.mols.get(mid)
        if not m or not m.menu or not order:
            continue
        # PIECE prices, not candidate totals.  A candidate total is the one thing a draft may
        # not name mid-episode: `_BAD_MIDEPISODE` exists "to stop a route TOTAL being named,
        # not a leaf's price", so a note carrying totals would get every draft that used it
        # refused as `leak_posthoc`.  A piece's own price, written with its unit, is scrubbed
        # before that check and passes.  So the note carries the DEAREST PIECE each candidate
        # needs -- the number that actually drives its
        # cost, and one the board prints on its own line -- for the cut taken and for the
        # dearest rival, which keeps the comparison while staying inside the form.
        worst = {}                       # candidate -> (dearest piece price, its total)
        for c in m.menu:
            prs = [pr for _, pr in (board.world.info(x) for x in c.reactants)]
            if prs and all(x is not None for x in prs):
                worst[c.idx] = (max(math.exp(x) for x in prs),
                                sum(math.exp(x) for x in prs))
        if len(worst) < 2 or order[0] not in worst:
            continue
        totals = {k: v[1] for k, v in worst.items()}
        lo, hi = min(totals.values()), max(totals.values())
        if lo <= 0 or hi / lo < 3.0:
            continue
        if totals[order[0]] > lo:
            continue
        rival = max(totals, key=lambda k: totals[k])
        out[mid] = {"taken": order[0],
                    # what the draft may quote: a piece price each, with its unit
                    "taken_piece": worst[order[0]][0],
                    "rival": rival, "rival_piece": worst[rival][0],
                    # kept for analysis; NOT for the draft to name
                    "taken_cost": totals[order[0]], "dearest": hi,
                    "spread": round(hi / lo, 1)}
    return out

def ANCESTRY() -> bool:
    """Whether the episode renderer should append `above this piece:` to a written thought.

    The flag is kept so `--analysis text` can query it; the text it would append is not
    produced by this module (`Turn` has no `ancestry` field). Off by default. Setting
    BOARD_ANCESTRY=1 fails on `Turn.ancestry`, and should, until that field exists.
    """
    return os.environ.get("BOARD_ANCESTRY") == "1"


def world_for(graph_rec: dict, menus, prices, top_k: int = 10,
              scores: Optional[Scores] = None) -> DataWorld:
    mols = graph_rec["graph"]["molecules"]
    return DataWorld(
        menus=menus,
        stock={m["smiles"] for m in mols if m["in_stock"]},
        prices=prices,
        scope={m["smiles"] for m in mols if m.get("is_expanded")},
        top_k=top_k,
        scores=scores,
    )



# ------------------------------------------------------------- graph labels
@dataclass
class GraphLabels:
    """(smiles, c-number) -> used | dead | unexpanded, from the recorded search.

    This is the only honest source of dead ends.  `used` means the candidate's
    subtree actually reached buyable material in the run; `dead` means it was
    expanded and failed; `unexpanded` means the search never asked.  The join is
    positional -- graph candidate rank r is menu index r-1 -- and verified by
    comparing the reactant lists, because a silent off-by-one here would relabel
    every decision on the board.
    """

    label: dict[tuple[str, int], str] = field(default_factory=dict)
    has_solution: dict[str, bool] = field(default_factory=dict)
    join_ok: int = 0
    join_bad: int = 0

    def of(self, smiles: str, idx: int) -> Optional[str]:
        return self.label.get((smiles, idx))


def graph_labels(rec: dict, menus) -> GraphLabels:
    gl = GraphLabels()
    for m in rec["graph"]["molecules"]:
        smi = m["smiles"]
        gl.has_solution[smi] = bool(m.get("has_solution"))
        menu = menus.get(smi) or []
        for c in m.get("candidates") or []:
            idx = int(c["rank"]) - 1
            gl.label[(smi, idx)] = c["label"]
            if idx < len(menu) and list(menu[idx][0]) == list(c["reactants"]):
                gl.join_ok += 1
            else:
                gl.join_bad += 1
    return gl


# ------------------------------------------------------------------- the DP
class DP:
    """max-min value of a molecule over the menus, and the ordering it implies."""

    def __init__(self, world: DataWorld, max_depth: int, axis: str = "p",
                 labels: Optional[GraphLabels] = None,
                 tiebreak: tuple[str, ...] = ("rt", "price", "q"),
                 tie_eps: float = 1e-3):
        self.w = world
        self.max_depth = max_depth
        self.axis = axis
        self.labels = labels
        self.tiebreak = tiebreak
        self.tie_eps = tie_eps
        """The DP is often indifferent: p saturates near the top of a menu, so
        differences below 1e-3 in a filter that returns 1.000 are noise, and the
        top candidates of a menu frequently tie.  The tie-break axes separate
        them.  That is what earns price and rt a place in the LABEL: a reasoning trace
        that cites cost is then citing something that moved the action, which is
        the difference between weighing a signal and garnishing with it."""
        self._memo: dict[tuple[str, int], float] = {}

    def _complete_cost(self, cand: Candidate) -> float:
        """Sum of ln prices, but only when EVERY piece is purchasable and priced.

        A partial sum is not a cost, and a candidate that leaves work to do has
        no comparable price at all -- those sort last rather than cheapest.
        """
        total = 0.0
        for smi in cand.reactants:
            buyable, price = self.w.info(smi)
            if not buyable or price is None:
                return INF
            total += price
        return total

    def _tie_key(self, cand: Candidate) -> tuple:
        out = []
        for k in self.tiebreak:
            if k == "rt":
                rt = cand.signals.get("rt")
                out.append(rt if isinstance(rt, (int, float)) else 99)
            elif k == "price":
                out.append(self._complete_cost(cand))
            else:
                out.append(-(cand.signals.get(k) or 0.0))
        return tuple(out)

    def usable(self, smiles: str, idx: int) -> bool:
        """Optional extra filter: only candidates the recorded search closed.

        OFF by default, and the default is the point.  `used` is a fact about
        one search under one budget, not about this board: a candidate labelled
        dead is routinely closable from the cached menus, so filtering on the
        label produces `dead` labels the board then contradicts -- an injected
        trap comes back solved and the episode teaches the opposite of what it
        was built for.  What the board enforces is a cached menu, in_stock
        pieces and the depth bound, so that is what the DP measures.  Keep the
        labels for evidence, and turn this on only to reproduce the run.
        """
        if self.labels is None:
            return True
        return self.labels.of(smiles, idx) == "used"

    def value(self, smiles: str, left: int, _stack: frozenset = frozenset()) -> float:
        if smiles in self.w.stock:
            return INF
        if left <= 0 or smiles in _stack:
            return -INF
        if not self.w.menus.get(smiles):
            return -INF      # nothing recorded: offline the board cannot open it
        key = (smiles, left)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = -INF                    # cycle guard while recursing
        stack = _stack | {smiles}
        best = -INF
        for cand in self.w.menu(smiles):
            if not self.usable(smiles, cand.idx):
                continue
            best = max(best, self.candidate_value(cand, left, stack, smiles))
        self._memo[key] = best
        return best

    def candidate_value(self, cand: Candidate, left: int,
                        _stack: frozenset = frozenset(),
                        parent: Optional[str] = None) -> float:
        q = cand.signals.get(self.axis)
        if q is None:
            return -INF
        if parent is not None and not self.usable(parent, cand.idx):
            return -INF
        worst = min((self.value(r, left - 1, _stack) for r in cand.reactants),
                    default=INF)
        return min(q, worst)

    def coverable(self, smiles: str, left: int, width: int = 3,
                  _stack: frozenset = frozenset()) -> bool:
        """Is every molecule a probing policy could reach from here recorded?

        in_stock ends the branch; running out of depth ends it too and is a
        legitimate `dead` by arithmetic.  A molecule with no recorded menu ends
        nothing -- it is the recording stopping, and any label written there is
        a guess.
        """
        if smiles in self.w.stock:
            return True
        if left <= 0:
            return True
        if smiles in _stack:
            return True
        menu = self.w.menu(smiles)
        if not menu:
            return False
        stack = _stack | {smiles}
        ranked = sorted(menu, key=lambda c: -(c.signals.get(self.axis) or 0))
        return all(self.coverable(r, left - 1, width, stack)
                   for c in ranked[:width] for r in c.reactants)

    @staticmethod
    def _unused() -> None:
        return None

    def order(self, board: Board, mid: str) -> list[tuple[int, float]]:
        """(candidate index, value) for every candidate that actually closes.

        Ordered by value descending, ties broken by the axis itself.  A candidate
        that dead-ends is not listed however well it scores -- that asymmetry is
        the whole point: below the weakest step is arithmetic, above it is only
        permission to look.
        """
        m = board.mols[mid]
        left = board.left(mid)
        out = []
        for c in m.menu or []:
            if c.idx in m.failed or c.idx in m.applied:
                continue
            v = self.candidate_value(c, left, parent=m.smiles)
            if v > -INF:
                out.append((c.idx, v))
        # Drop what the BOARD would refuse.  A candidate that brings in the
        # molecule itself or one of its ancestors has no value -- a route cannot
        # contain itself -- and proposing one is not just useless: ranked as a
        # tail it teaches the model to name a fallback the board rejects, and
        # ranked as a head it aborts the build.
        out = [(i, v) for i, v in out if _acyclic(board, mid, by_index(m)[i])]
        by_idx = {c.idx: c for c in m.menu}
        # Bucket on the axis first, then break ties in the declared order.  The
        # bucket width is tie_eps, so "0.998 vs 1.000" is one bucket and the
        # tie-break decides -- otherwise a saturated filter decides by noise.
        out.sort(key=lambda t: (-round(t[1] / max(self.tie_eps, 1e-12)),
                                self._tie_key(by_idx[t[0]])))
        return out


# ------------------------------------------------------------- label policy
@dataclass
class Policy:
    """Where each half of a training example comes from.

    The two halves are separated on purpose.  A
    trajectory that follows the optimal route never enters a dead end, so
    `dead`, re-ranking and exhaustion would never appear in a single example.
    So: the STATES are allowed to go wrong (mistakes, probes) and the ACTIONS
    are the DP's.  Turns produced by the wrong half are marked unsupervised --
    they keep the conversation coherent and carry no loss.
    """

    analyze: bool = False
    """Buy the structural/mechanistic evidence for a molecule's candidates before ranking it.

    Off by default because it changes the SHAPE of an episode: it inserts a turn between the
    menu arriving and the ranking whose whole content is "I cannot separate these yet". On,
    the ranking turn has the EVIDENCE block on screen, which is the only way a trace can cite
    a bond position or an earned reaction name without inventing it. A run with analyze off
    cannot express the mechanism half of the reasoning brief at all: there is no bond fact
    on screen to cite."""

    analyze_top: int = 3
    """How many candidates per molecule `analyze` names, before the route's own step is added.

    Not the whole menu: the menu is ten wide, the ranking names three, and an atom mapping is
    bought per candidate. Analysing everything pays for seven answers nobody reads and puts
    facts on screen that no decision turns on."""

    rank_max: Optional[int] = 3
    # How many of the DP's kept ranking to COMMIT on the turn that ranks it. 1 is the serial
    # board: the head is applied, the rest wait for a later turn, and the OPEN frontier rarely
    # holds more than one molecule -- so `open` never has a choice and nothing about exploring
    # several branches at once is ever demonstrated. Above 1 the branches are materialised
    # where the DP already judged them viable, which is the same set the serial policy gets
    # around to eventually: replaying the collected DAG breadth-first costs few extra calls,
    # takes far fewer turns, and leaves several molecules on the frontier.
    branch: int = 1
    # `branch` commits the DP's top-N at EVERY molecule, which is a uniform BFS and grows
    # geometrically in single-step calls. That is not what the collected DAG looks like --
    # the search only branched where branching was worth it. `branch_queue` reproduces THAT: commit, at each molecule, exactly the
    # candidates some route still on the queue needs there. Same reactions, same DAG, laid out
    # in space instead of in time.
    branch_queue: bool = False
    """How many candidates a ranking may name.  With labels the DP's usable set
    is often larger than a ranking should be, and naming all of it teaches
    list-everything.  None = no cap."""

    rank_floor: bool = True
    """Drop candidates that cannot beat the best route's weakest step.  This is
    the one safe direction of the pruning rule, so it belongs in the label."""

    probe_max: int = 2
    """Doomed probes the ROLLOUT may spend before `dead` is allowed to say
    'every candidate I ranked has come back failed'.  Without probes that reason
    is a lie; with unlimited probes a trap costs a dozen calls."""

    mistakes: int = 0
    """Decision points where the rollout departs from the DP.  This is what puts
    a trap in the state stream at all."""

    trap_depth: Optional[int] = None
    """How far down a trap has to be playable out of the recording.

    `_in_data` walks the WHOLE remaining depth at width 3, and offline that is a
    test the draw cache cannot pass: the search spent its budget of single-step
    calls elsewhere and a dead branch's children are not among them, so a
    candidate labelled `dead` has almost never got menus below it.  With no cap
    the trap picker returns None everywhere and a queue episode contains no dead
    end at all.

    A cap of 1 buys the SHORT trap and only the short trap: apply the candidate,
    its pieces arrive, and they are dead on sight -- out of depth, or under the
    cut-off, or exhausted after the probes.  That is the whole `dead`-then-
    re-rank shape in three turns, and it is sound because the reason is readable
    rather than assumed.  A trap that turns out to be deeper than the cap runs
    into NoMenu, the episode is marked invalid and `--strict` drops it: the cost
    of a cap that is too shallow is counted instances, never a wrong label.

    None keeps the full check, which is correct against a LIVE menu server (any
    molecule can be expanded on demand) and empty against the cache."""

    route_mistakes: bool = False
    """Let a mistake displace the ROUTE's own step, not just an off-route one.

    `mistakes` alone cannot fire while a route queue is being followed: the route
    branch of `teacher_turn` takes the route's disconnection and returns before
    the trap picker is reached, so a queue episode contains almost no dead end.
    That is the safe
    default -- a fixed single route and a mistake are incompatible, since the
    rollout leaves the route and there is nothing to follow back.

    A QUEUE is different, and that is what this switches on: the trap is applied,
    its subtree fails, the molecule comes back open with the earlier ranking on
    screen, and the queue's own step is STILL there to be ranked.  So the episode
    contains the departure, the `dead` with a reason, and the correct re-rank --
    which is the only shape in the corpus that shows backing out of a branch."""

    mistake_mode: str = "trap"
    """readable -- take the best-scoring candidate whose failure is readable on
    the NEXT board: a piece at the depth floor, or one whose whole menu is under
    the cut-off.  The only mode that works against the recording alone; see
    _readable_trap_pick.
    greedy -- take the top-scoring candidate, whatever it is.  That is what a
    score-greedy policy does, but it lands in a FAILING subtree only some of the
    time, so most injected mistakes cost a little value and teach nothing about
    backing out.
    trap -- take the highest-scoring candidate the DP says does not close.  This
    is the same failure, selected for: it is a common case on the search graphs,
    and it is the only way `dead` and re-ranking enter the data."""

    window: Optional[int] = None
    """How many candidates are ON SCREEN.  A ranking may only name candidates the
    model can see, so this has to match render's menu_show.  None = no limit,
    which is only correct when the whole menu is rendered.  The cost of a small
    window is counted: `window_loss` on the turn note records when the DP's own
    first pick was behind the fold."""

    continue_past_solve: bool = True
    """After a route closes, keep going while a candidate could still beat it.

    The board registers every route, so `done` is the agent's decision -- but the
    LABEL has to demonstrate it.  A labeller that returns Done the moment nothing
    is open produces episodes that all end with `first route closes -> answer`,
    and a model trained on them does exactly that.

    With it on, the labeller looks for an untried candidate whose DP value beats
    the best route's weakest step -- the one direction of the pruning rule that is
    arithmetic -- and ranks it.  `done` is then emitted only when no such candidate
    exists anywhere, which is what makes it provable rather than habitual."""

    max_routes: int = 0
    """Cap on registered routes per episode (0 = none).  Passed to the board."""

    max_continuations: int = 2
    """How many times the labeller may look past a closed route.

    Not unbounded, and the reason is a real gap between what the DP promises and
    what the board realises: when a candidate's piece is ALREADY CLOSED, the new
    reaction closes against the existing subtree, so the route inherits that
    subtree's weakest step -- while the DP valued the candidate as if it would
    build the best subtree available.  So a candidate valued at .999 can register a
    route whose weakest step is still .003, the floor does not rise, and the search
    continues forever on a promise it cannot cash.

    The stopping rule is therefore empirical, not arithmetic: take the best
    remaining candidate, and if the route it produces does not beat the floor,
    stop.  That is one demonstration per episode of "above the weakest step was
    permission to look, not evidence that anything is there" -- which is the
    lesson, and it is what `done` is supposed to rest on."""

    route_queue: Optional[list] = None
    """Every selected route for this target, to be realised in ONE episode.

    This is the difference between a corpus that teaches "find a route" and one
    that teaches "find the routes".  With one route per episode the model learns
    a distribution over first steps -- several sampled rollouts recover several
    distinct routes while a single rollout returns about one -- so the diversity
    exists only across rollouts, and at temperature 0 it does not exist at all.  Walking the whole set inside one
    episode is what puts it in a single rollout."""

    route: Optional[dict] = None
    """product -> reactants.  When set, the head of every ranking is the route's
    own disconnection and the DP only orders what comes after it.  This is what
    turns one target's several selected routes into several instances."""

    cutoff: Optional[float] = None
    max_open: int = 4
    labels: object = None
    """The recorded search's own labels, for choosing TRAPS only.

    Kept separate from `DP(labels=...)` on purpose.  Giving them to the DP
    restricts what it will rank, which is the label contamination that makes
    injected traps come back solved.  Used here they answer a different and
    honest question: which candidate did the recorded search actually expand and
    fail?  That candidate's subtree is in the recording, so the episode can play
    the failure out and declare `dead` for a reason that is true.
    """

    free_routes: int = 0
    """Claim routes the graph already holds, once the queue is exhausted.

    A candidate whose pieces are all purchasable closes its molecule the moment it
    is applied, so swapping one into a route already claimed gives another
    complete route for one action and no opens.  An episode that claims only its
    queue routes leaves many such routes on the table.  This is the budget for
    how many of them to take; 0 claims none.
    """

    batch_divergence: bool = False
    """Rank every rankable route divergence in one call instead of one per turn.

    Off by default, and left switchable because the reasoning is not obvious:
    see next_route_step.
    """


@dataclass
class Step:
    actions: list
    supervised: bool = True
    note: str = ""


def by_index(mol) -> dict:
    return {c.idx: c for c in (mol.menu or [])}


def _acyclic(board: Board, mid: str, cand) -> bool:
    """Would this candidate make `mid` (or an ancestor of it) its own precursor?"""
    if cand is None:
        return False
    for smi in cand.reactants:
        existing = board.mid_of(smi)
        if existing is not None and (existing == mid or board._reaches(existing, mid)):
            return False
    return True


def dead_reason(board: Board, mid: str, dp: DP, pol: Policy) -> Optional[str]:
    """Which of the three named reasons actually holds -- None if none does.

    None is the interesting answer: it means the trap is real but unreadable
    from the screen, and the only honest way to `dead` it is to have tried.
    """
    m = board.mols[mid]
    if board.left(mid) <= 0:
        return "floor"
    if pol.cutoff is not None and m.menu:
        best = max((c.signals.get(dp.axis, 0.0) for c in m.menu), default=0.0)
        if best < pol.cutoff:
            return "cutoff"
    untried = [c for c in (m.menu or [])
               if c.idx not in m.failed and c.idx not in m.applied]
    if m.failed and not untried:
        return "exhausted"
    return None


def route_floor(board: Board, dp: DP) -> float:
    """The best route's weakest step, ON THE DP'S OWN AXIS.

    It has to be the axis the DP ranks on.  This read the reaction's first
    available signal instead, which is `q` under the default display order, so the
    floor would be a q value compared against p-valued candidates.
    """
    from .render import _route_rxns

    best = -INF
    for route in board.routes:
        vals = [board.rxns[r].signals.get(dp.axis)
                for r in _route_rxns(board, route.root_rxn, route)]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            best = max(best, min(vals))
    return best


def rank_label(board: Board, mid: str, dp: DP, pol: Policy) -> tuple[Optional[list[int]], str]:
    """The DP's ranking for one molecule, or None when it has nothing usable.

    Returns (order, note).  The note is non-empty when the window cost something:
    a candidate the DP wanted is not on screen, so the label cannot name it and
    the model is being taught a different answer from the optimal one.
    """
    ranked = dp.order(board, mid)
    if not ranked:
        return None, ""
    note = ""
    if pol.window:
        visible = [(i, v) for i, v in ranked if i < pol.window]
        if visible and visible[0][0] != ranked[0][0]:
            note = (f"window_loss on {mid}: DP wanted c{ranked[0][0]}"
                    f" (dp {ranked[0][1]:.3f}), best on screen is c{visible[0][0]}"
                    f" (dp {visible[0][1]:.3f})")
        elif not visible:
            note = f"window_loss on {mid}: nothing usable on screen"
        ranked = visible
    if pol.rank_floor and board.routes:
        floor = route_floor(board, dp)
        kept = [(i, v) for i, v in ranked if v > floor]
        ranked = kept or ranked[:1]
    if not ranked:
        return None, note
    if pol.rank_max:
        ranked = ranked[: pol.rank_max]
    return [i for i, _ in ranked], note


def _branch_order(board: Board, mid: str, order: list[int], pol: Policy,
                  state: dict) -> tuple[list[int], int]:
    """(order, how many of it to commit now).

    `branch_queue` is the faithful breadth-first replay: the candidates some route still on
    the queue needs AT THIS MOLECULE come first, and all of them are committed. Where the
    queued routes agree that is one candidate and the turn is the serial board's; where they
    diverge it is two or three, and the branches exist at the same time instead of the board
    coming back for them many turns later.

    `branch` (a plain N) is the uniform BFS and is kept for the ablation -- it commits N
    everywhere, which is a different and much more expensive policy.
    """
    if getattr(pol, "branch_queue", False) and pol.route_queue:
        m = board.mols[mid]
        cache = state.setdefault("_qmaps", None)
        if cache is None:
            cache = [route_map(q["steps"]) for q in pol.route_queue if q.get("steps")]
            state["_qmaps"] = cache
        wants = {tuple(sorted(w)) for wm in cache
                 for w in [wm.get(m.smiles)] if w}
        idxs = []
        for w in wants:
            i = match_candidate(m.menu, list(w))
            if i is not None and i in order and i not in idxs:
                idxs.append(i)
        if idxs:
            idxs.sort(key=order.index)               # the DP's own preference among them
            rest = [i for i in order if i not in idxs]
            return idxs + rest, len(idxs)
        return order, 1
    return order, min(getattr(pol, "branch", 1) or 1, len(order))


def teacher_turn(board: Board, dp: DP, pol: Policy, state: dict) -> Step:
    """One turn: open the frontier, or commit on every molecule showing a menu.

    Several molecules per turn is deliberate -- AND-siblings do not constrain
    each other, and one-piece-at-a-time is the behaviour that produced the
    budget gap this format exists to fix.
    """
    claim = claim_step(board, pol, state)
    if claim is not None:
        return claim
    if not (pol.max_routes and len(board.routes) >= pol.max_routes):
        opening = route_open_step(board, pol, state)
        if opening is not None:
            return opening

    open_ids = board.open_mols()
    with_menu = [m for m in open_ids if board.mols[m].menu is not None]
    without = [m for m in open_ids if board.mols[m].menu is None]

    if pol.analyze and board.evidence is not None:
        # One analyze covering every molecule whose menu has arrived unanalysed, naming the
        # CANDIDATES it will actually choose between rather than the whole ten-wide menu:
        # every candidate analysed costs an atom mapping, and a trace can only argue about
        # what it paid for. The set is the DP's own top `analyze_top` plus the route's own
        # step when a route is being followed, so the candidate that gets applied is always
        # among the analysed ones and the reasoning is never asked to justify an unexamined
        # choice. Lands between the open and the rank, where "what do these actually do" is
        # the live question.
        need = {}
        for mid in with_menu:
            if board.mols[mid].analyzed_turn is not None:
                continue
            top = [i for i, _ in dp.order(board, mid)][: pol.analyze_top or 3]
            active = state.get("route_map") or pol.route
            if active is not None:
                want = active.get(board.mols[mid].smiles)
                head = (match_candidate(board.mols[mid].menu, want)
                        if want is not None else None)
                if head is not None and head not in top:
                    top.append(head)
            if top:
                need[mid] = sorted(set(top))
        if need:
            return Step(actions=[Analyze([], need)], supervised=True,
                        note="analyze the candidates about to be ranked")

    if with_menu:
        acts, supervised, notes = [], True, []
        for mid in with_menu:
            active = state.get("route_map") or pol.route
            if active is not None:
                want = active.get(board.mols[mid].smiles)
                head = (match_candidate(board.mols[mid].menu, want)
                        if want is not None else None)
                if (want is not None and head is not None
                        and not _acyclic(board, mid,
                                         by_index(board.mols[mid]).get(head))):
                    # The route's own step is illegal AT THIS POSITION.  It happens
                    # after a continuation: the new branch put one of the step's
                    # reactants above this molecule, so applying the step here
                    # would make that molecule its own precursor.  The route has
                    # nothing to say about a position it never occupied, so the DP
                    # labels this decision instead.
                    want = None
                    notes.append(f"{mid}: the route's step is cyclic here "
                                 f"(a continuation moved a reactant above it); "
                                 f"labelled by the DP")
                if want is not None:
                    if head is None:
                        raise OffRoute(f"{mid}: the route's step is not on the menu")
                    if pol.window and head >= pol.window:
                        raise OffRoute(f"{mid}: the route's step is c{head}, "
                                       f"behind a window of {pol.window}")
                    if pol.route_mistakes and state["mistakes"] < pol.mistakes:
                        wrong = _pick_mistake(board, mid, dp, pol)
                        if wrong is not None and wrong != head:
                            state["mistakes"] += 1
                            acts.append(Rank(mid, [wrong]))
                            supervised = False
                            notes.append(
                                f"rollout took c{wrong} ({pol.mistake_mode}) over "
                                f"the route's own c{head}; the route's step stays "
                                f"on the menu for the re-rank after it fails")
                            continue
                    rest = [i for i, _ in dp.order(board, mid) if i != head]
                    order = ([head] + rest)[: pol.rank_max or None]
                    acts.append(Rank(mid, order))
                    continue
            order, wnote = rank_label(board, mid, dp, pol)
            if wnote:
                notes.append(wnote)
            if order is not None:
                # a mistake: the rollout takes the top-q candidate instead
                wrong = _pick_mistake(board, mid, dp, pol)
                if (state["mistakes"] < pol.mistakes and wrong is not None
                        and wrong != order[0]):
                    state["mistakes"] += 1
                    acts.append(Rank(mid, [wrong]))
                    supervised = False
                    notes.append(f"rollout took c{wrong} ({pol.mistake_mode})"
                                 f" over the DP's c{order[0]}")
                    continue
                order, take = _branch_order(board, mid, order, pol, state)
                acts.append(Rank(mid, order, take=take))
                continue
            reason = dead_reason(board, mid, dp, pol)
            if reason is not None:
                acts.append(Dead(mid, reason))
                continue
            probe = _greedy_pick(board, mid, dp)
            m = board.mols[mid]
            if probe is not None and len(m.failed) < pol.probe_max:
                acts.append(Rank(mid, [probe]))
                supervised = False
                notes.append(f"probe c{probe}: unreadable trap, has to be tried")
            elif probe is None:
                # nothing left that the board could even act on
                acts.append(Dead(mid, "exhausted"))
                notes.append("dead(exhausted): no candidate the board can act on")
            else:
                acts.append(Dead(mid, "exhausted"))
                supervised = False
                notes.append(f"UNSOUND dead(exhausted): {len(m.failed)} probes spent,"
                             f" c{probe} still untried (probe_max={pol.probe_max})")
        return Step(acts, supervised, "; ".join(notes))
    if without:
        return Step([Open(mid) for mid in without[: pol.max_open]])

    # Nothing open.  Before considering `done`, see whether another selected route
    # is still unrealised: going back for it is the whole point of the queue.
    if pol.route_queue and not (pol.max_routes
                                and len(board.routes) >= pol.max_routes):
        step = next_route_step(board, pol, state)
        if step is not None:
            return step

    free = free_route_step(board, dp, pol, state)
    if free is not None:
        return free

    if pol.continue_past_solve and board.routes:
        spent = state.get("continuations", 0)
        floor_now = route_floor(board, dp)
        improved = floor_now > state.get("floor_at_last_continuation", -INF)
        if spent and not improved:
            return Step([Terminate()], note=(
                f"terminate: the continuation did not pay -- the best remaining "
                f"candidate registered a route whose weakest step is still "
                f"{floor_now:.3f}"))
        if spent < pol.max_continuations:
            cont = continuation(board, dp, pol)
            if cont is not None:
                mid, order, value, floor = cont
                state["continuations"] = spent + 1
                state["floor_at_last_continuation"] = floor
                return Step([Rank(mid, order)],
                            note=f"continuing past route {board.routes[-1].label}: "
                                 f"{mid} c{order[0]} at {value:.3f} beats the "
                                 f"weakest step {floor:.3f}")
            return Step([Terminate()], note=(
                f"terminate: nothing untried anywhere is above the weakest step "
                f"{floor_now:.3f}"))
    return Step([Terminate()])


class _Claim(NamedTuple):
    """What a route needs before it can be claimed."""

    choices: Optional[dict]     # ready to claim, if not None
    need_open: list             # molecules that only need their menu fetched
    blocked: bool               # a reaction has to be applied first


def claim_choices(board: Board, want_map: dict, pol: Policy) -> "_Claim":
    """The choices that would claim this route, if the board can make it NOW.

    Returns molecule id -> candidate number, or None when the board cannot get
    there yet.  "Yet" is the load-bearing word: a route can be claimed only where
    every molecule it disconnects is already on the board showing a menu, because
    a candidate number is something the model read off that menu.  A route that
    needs an intermediate the board has never built has to be walked first -- and
    that is the honest boundary, since applying a reaction is what brings its
    pieces into existence at all.

    What this makes free is the sibling case: the route that differs from one
    already built by a reaction or two, over molecules that are all open -- a
    large share of the routes an episode finds after its first.
    """
    choices: dict[str, int] = {}
    need_open: list[str] = []
    stack, seen = [board.root], set()
    while stack:
        mid = stack.pop()
        if mid in seen:
            continue
        seen.add(mid)
        m = board.mols[mid]
        want = want_map.get(m.smiles)
        if want is None:
            # The route ends here, so this has to be material you can buy.
            if not m.buyable:
                return _Claim(None, [], True)
            continue
        if m.status == "dead":
            return _Claim(None, [], True)
        if m.menu is None:
            # One `open` away.  A candidate number is only knowable from a menu,
            # so this cannot be folded into the same call -- but it does NOT need
            # the reaction ranked first, which is the turn worth saving.
            need_open.append(mid)
            continue
        head = match_candidate(m.menu, want)
        if head is None or head in m.failed:
            return _Claim(None, [], True)
        if pol.window and head >= pol.window:
            return _Claim(None, [], True)
        if not _acyclic(board, mid, by_index(m).get(head)):
            return _Claim(None, [], True)
        choices[mid] = head
        for smi in want:
            piece = board.mid_of(smi)
            if piece is not None:
                stack.append(piece)
                continue
            # Not on the board at all, so it cannot even be opened: the reaction
            # that would produce it has to be applied first.  That is the one case
            # where walking is unavoidable.
            if not board.world.info(smi)[0]:
                return _Claim(None, [], True)
    if need_open:
        return _Claim(None, need_open, False)
    return _Claim(choices if board.root in choices else None, [], True)


def claim_step(board: Board, pol: Policy, state: dict) -> Optional[Step]:
    """Claim every queued route the board can already make, in one call."""
    queue = pol.route_queue or []
    if not queue:
        return None
    done_idx = state.setdefault("claimed", set())
    room = (pol.max_routes - len(board.routes)) if pol.max_routes else len(queue)
    if room <= 0:
        return None
    acts, labels = [], []
    for i, q in enumerate(queue):
        if i in done_idx or len(acts) >= room:
            continue
        ch = claim_choices(board, route_map(q["steps"]), pol).choices
        if ch is None:
            continue
        # Two claims in one call must not be the same route, and the board would
        # reject the second -- so they are compared here, by the same identity the
        # board uses.
        key = frozenset((board.mols[m].smiles,
                         frozenset(by_index(board.mols[m])[c].reactants))
                        for m, c in ch.items())
        if any(k == key for k in state.setdefault("claim_keys", set())):
            done_idx.add(i)
            continue
        if any(r.steps(board) == key for r in board.routes):
            done_idx.add(i)
            continue
        state["claim_keys"].add(key)
        done_idx.add(i)
        acts.append(Done(ch))
        labels.append(str(i + 1))
    if not acts:
        return None
    return Step(acts, note=f"claiming route {', '.join(labels)} of {len(queue)}: "
                           f"the board can already make {'them' if len(acts) > 1 else 'it'}")


def _swap_choices(board: Board, route, mid: str, cand: int) -> Optional[dict]:
    """`route` with `cand` at `mid`: molecule id -> candidate number.

    Walks down from the target so the result covers exactly the molecules the new
    route uses.  Carrying the old subtree below `mid` would name molecules the
    route no longer reaches, and the board rejects those -- correctly, since a
    claim has to say what it makes and nothing else.
    """
    out: dict[str, int] = {}
    stack, seen = [board.root], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == mid:
            out[cur] = cand
            continue                     # its pieces are purchasable: stop here
        rid = route.choices.get(cur)
        if rid is None:
            continue                     # a leaf of the old route
        rxn = board.rxns[rid]
        out[cur] = rxn.cand
        stack.extend(rxn.pieces)
    return out if board.root in out else None


def free_route_step(board: Board, dp: "DP", pol: Policy, state: dict) -> Optional[Step]:
    """Claim the best routes the graph can already make, in one call.

    Only candidates whose pieces are ALL purchasable qualify: those close their
    molecule on application, so the route is complete without opening anything.
    Ordered by the deciding axis, so what gets taken when there is room for three
    of twelve is the best three and not the first three the walk happens to reach.
    """
    if not pol.free_routes or not board.routes:
        return None
    room = (pol.max_routes - len(board.routes)) if pol.max_routes else pol.free_routes
    room = min(room, pol.free_routes - state.get("free_taken", 0))
    if room <= 0:
        return None
    have = {r.steps(board) for r in board.routes}
    seen_keys = set(state.setdefault("claim_keys", set()))
    found: dict = {}
    for rt in list(board.routes):
        for mid in list(rt.choices):
            m = board.mols[mid]
            if m.menu is None or m.status == "dead":
                continue
            for c in m.menu:
                if c.idx in m.failed or c.idx in m.applied:
                    continue
                if pol.window and c.idx >= pol.window:
                    continue
                if any(not board.world.info(sm)[0] for sm in c.reactants):
                    continue
                if not _acyclic(board, mid, c):
                    continue
                ch = _swap_choices(board, rt, mid, c.idx)
                if ch is None:
                    continue
                key = frozenset(
                    (board.mols[om].smiles,
                     frozenset(by_index(board.mols[om])[oc].reactants))
                    for om, oc in ch.items())
                if key in have or key in seen_keys or key in found:
                    continue
                val = c.signals.get(dp.axis)
                found[key] = (val if val is not None else -INF, ch, mid, c.idx)
    if not found:
        return None
    best = sorted(found.values(), key=lambda t: -t[0])[:room]
    acts, notes = [], []
    for val, ch, mid, cidx in best:
        acts.append(Done(ch))
        notes.append(f"{mid} c{cidx}")
        state["claim_keys"].add(frozenset(
            (board.mols[om].smiles,
             frozenset(by_index(board.mols[om])[oc].reactants))
            for om, oc in ch.items()))
    state["free_taken"] = state.get("free_taken", 0) + len(acts)
    return Step(acts, note=(
        f"claiming {len(acts)} route{'s' if len(acts) > 1 else ''} the graph "
        f"already holds: {', '.join(notes)} land on purchasable material, so no "
        f"molecule has to be opened"))


def route_open_step(board: Board, pol: Policy, state: dict) -> Optional[Step]:
    """Open exactly the molecules the next unclaimed route still needs a menu for.

    This is what replaces ranking down to a route.  Ranking the divergence applies
    a reaction the claim would have made itself, and the piece it creates then
    needs opening anyway -- so the rank was a turn spent on nothing.  Opening
    straight to what the route needs, then claiming, is the same result one turn
    per level cheaper.
    """
    queue = pol.route_queue or []
    if not queue:
        return None
    done_idx = state.setdefault("claimed", set())
    want: list[str] = []
    for i, q in enumerate(queue):
        if i in done_idx:
            continue
        probe = claim_choices(board, route_map(q["steps"]), pol)
        if probe.blocked or not probe.need_open:
            continue
        for mid in probe.need_open:
            if mid not in want and board.mols[mid].menu is None:
                want.append(mid)
        # No break: gathering across every waiting route fills the call, and one
        # call of four opens lets FOUR routes become claimable together instead of
        # one per turn.  Turn count is the whole cost of this format -- per-turn
        # size already went down -- so the batching is where it is paid back.
        if pol.max_open and len(want) >= pol.max_open:
            break
    if not want:
        return None
    take = want[: pol.max_open] if pol.max_open else want
    return Step([Open(mid) for mid in take],
                note=f"opening what the remaining routes still need a menu for")


def next_route_step(board: Board, pol: Policy, state: dict) -> Optional[Step]:
    """One move towards the next unrealised route in the queue.

    Picks the SHALLOWEST molecule where the route disagrees with what the board
    has already applied, because that is where the routes actually diverge; the
    pieces it opens are then driven by the ordinary route-following branch.  A
    route whose divergence point is not on the menu, or which the board would
    refuse there, is dropped from the queue with a note rather than retried.
    """
    idx = state.get("route_index", 0)
    queue = pol.route_queue or []
    while idx < len(queue):
        want_map = route_map(queue[idx]["steps"])
        state["route_map"] = want_map
        target = _divergence(board, want_map)
        if target is None:
            idx += 1                      # already realised; try the next one
            state["route_index"] = idx
            continue
        mid, want = target
        m = board.mols[mid]
        if m.menu is None:
            return Step([Open(mid)],
                        note=f"route {idx + 1} of {len(queue)}: opening {mid} to "
                             f"reach its own disconnection")
        head = match_candidate(m.menu, want)
        if head is None or (pol.window and head >= pol.window) \
                or not _acyclic(board, mid, by_index(m).get(head)):
            state["route_index"] = idx + 1
            idx += 1
            continue
        acts = [Rank(mid, [head])]
        labels = [str(idx + 1)]
        if pol.batch_divergence:
            taken = {mid}
            for j in range(idx + 1, len(queue)):
                if j in state.get("claimed", ()):
                    continue
                other = _divergence(board, route_map(queue[j]["steps"]))
                if other is None:
                    continue
                omid, owant = other
                om = board.mols[omid]
                if omid in taken:
                    # SAME molecule, a different candidate: two queued routes diverge right
                    # here. Skipping it is where the serialisation lives -- the second route's
                    # cut waits for a later turn, so the board never has two
                    # subtrees of one molecule open at once and `open` never has a choice.
                    # `branch_queue` merges it into the rank that is already going out, which
                    # is the faithful breadth-first replay: same reactions, same DAG, laid out
                    # in space. Only for a molecule the turn is already ranking, so nothing
                    # speculative is added -- the routes on the queue are the ones the search
                    # found, not candidates the DP merely likes.
                    if not getattr(pol, "branch_queue", False) or om.menu is None:
                        continue
                    oh = match_candidate(om.menu, owant)
                    if oh is None or (pol.window and oh >= pol.window):
                        continue
                    if not _acyclic(board, omid, by_index(om).get(oh)):
                        continue
                    for a in acts:
                        if isinstance(a, Rank) and a.mid == omid and oh not in a.order:
                            a.order = list(a.order) + [oh]
                            a.take = len(a.order)
                            labels.append(str(j + 1))
                            break
                    continue
                if om.menu is None:
                    continue
                ohead = match_candidate(om.menu, owant)
                if ohead is None or (pol.window and ohead >= pol.window):
                    continue
                if not _acyclic(board, omid, by_index(om).get(ohead)):
                    continue
                acts.append(Rank(omid, [ohead]))
                taken.add(omid)
                labels.append(str(j + 1))
                if pol.max_open and len(acts) >= pol.max_open:
                    break
            return Step(acts, note=(
                f"routes {', '.join(labels)} of {len(queue)}: taking every "
                f"divergence the board can rank now"))
        # ONE divergence per turn, deliberately.  Ranking every rankable
        # divergence at once saves few turns but builds more reactions and costs
        # more tokens, because the extra ranks are speculative (routes later
        # dropped on the window or a cycle) and a board widened early is paid for
        # on every remaining turn.  Opening the same molecule later is cheaper
        # than opening it sooner, and the molecule SHARED by two routes is
        # already opened once -- one node, one menu -- so the saving that matters
        # is in the graph, not in the scheduling: the shared graph spends far
        # fewer opens than independent runs would.
        return Step([Rank(mid, [head])],
                    note=f"route {idx + 1} of {len(queue)}: {mid} c{head} is where "
                         f"it diverges from what is already built")
    state["route_map"] = None
    return None


def _divergence(board: Board, want_map: dict) -> Optional[tuple]:
    """The shallowest molecule whose route step is not on the board yet."""
    best = None
    for mid in board.order:
        m = board.mols[mid]
        if m.status == "dead" or m.smiles not in want_map:
            continue
        want = want_map[m.smiles]
        realised = any(
            sorted(board.mols[p].smiles for p in board.rxns[r].pieces) == sorted(want)
            for r in m.rxns)
        if realised:
            continue
        if any(board.rxns[r].status == "open" for r in m.rxns):
            continue                      # still owes a piece; not rankable
        if best is None or m.depth < board.mols[best[0]].depth:
            best = (mid, want)
    return best


def continuation(board: Board, dp: DP, pol: Policy):
    """The best untried candidate that could still beat the route in hand.

    Searched over every molecule with a menu, CLOSED ones included: a second route
    through a molecule that is already solved is exactly what the format calls
    looking for an alternative, and the board allows re-ranking it.

    Returns (mid, order, value, floor) or None.  A candidate at or below the floor
    is not returned: a route is a minimum, so it cannot improve on one.
    """
    floor = route_floor(board, dp)
    if floor == -INF:
        return None
    best = None
    for mid in board.order:
        m = board.mols[mid]
        if m.menu is None or m.status == "dead":
            continue
        if any(board.rxns[r].status == "open" for r in m.rxns):
            continue          # still owes a piece; not rankable
        for idx, value in dp.order(board, mid):
            if value <= floor:
                break         # dp.order is descending; nothing below helps
            if pol.window and idx >= pol.window:
                continue
            if best is None or value > best[2]:
                best = (mid, [idx], value, floor)
            break
    return best


def _trap_pick(board: Board, mid: str, dp: DP, labels=None,
               depth: Optional[int] = None) -> Optional[int]:
    """The best-scoring candidate that does NOT close -- the readable-looking trap.

    Selecting for it rather than waiting for greedy to stumble into it is the
    difference between a dataset that contains backtracking and one that does
    not.  It stays an honest state: the same candidate a score-greedy policy
    takes on the menus where it fails.
    """
    m = board.mols[mid]
    # Two passes.  First over the candidates the RECORDED SEARCH expanded and
    # failed: those are dead ends it actually walked, so the recording holds their
    # subtree and `dead` can be declared for a reason that is true.  Only if there
    # are none does it fall back to any non-closing candidate, where `_in_data`
    # has to establish playability itself -- which it rarely manages, so without
    # the labels this function is effectively empty.
    for prefer_dead in (True, False):
        best, best_v = None, -INF
        for c in m.menu or []:
            if c.idx in m.failed or c.idx in m.applied:
                continue
            if dp.candidate_value(c, board.left(mid), parent=m.smiles) > -INF:
                continue
            if not _acyclic(board, mid, c):
                continue
            if prefer_dead and (labels is None
                                or labels.of(m.smiles, c.idx) != "dead"):
                continue
            # The label and the replay are two different claims.  `dead` says the
            # recorded SEARCH expanded this and failed; it does not say the draw
            # cache holds menus for what it reached -- the search can also fail a
            # branch on depth or budget without expanding it far.  Playing such a
            # trap runs the episode into NoMenu and the whole thing is dropped.
            # So both have to hold.
            if not _in_data(board, dp, c, left=board.left(mid), max_depth=depth):
                continue
            v = c.signals.get(dp.axis)
            if v is not None and v > best_v:
                best, best_v = c.idx, v
        if best is not None:
            return best
    return None


def _pick_mistake(board: Board, mid: str, dp: DP, pol: "Policy") -> Optional[int]:
    """Which candidate the rollout takes instead of the labelled one."""
    if pol.mistake_mode == "readable":
        return _readable_trap_pick(board, mid, dp, pol)
    if pol.mistake_mode == "trap":
        return _trap_pick(board, mid, dp, pol.labels, pol.trap_depth)
    return _greedy_pick(board, mid, dp)


def _in_data(board: Board, dp: DP, cand: Candidate, left: int, width: int = 3,
             max_depth: Optional[int] = None) -> bool:
    """Can the board play this candidate's whole subtree out of the recording?

    A trap is only usable as a label if every molecule the model could plausibly
    reach under it is either in stock, out of depth (a legitimate floor), or has
    a recorded menu.  Checking one level is not enough: the recording runs out
    two and three levels down, and on screen a data gap and a dead end look
    identical.  `width` bounds the check to the candidates a probing policy
    would actually take -- the top few -- so this stays affordable.
    """
    # The check guards against ONE thing: the recording running out under the
    # trap, which on screen looks identical to a dead end.  A live menu server has
    # no such gap -- any molecule can be expanded on demand -- so the premise is
    # false and the recursion is pure cost: `coverable` walks the remaining depth
    # at width 3, which offline is a dict lookup and live is an HTTP call per
    # molecule, far too slow for one episode.  Once the live budget is spent
    # the world falls back to the recording and the gap is real again, so the
    # exemption is tied to the budget rather than to the world's type.
    live = getattr(board.world, "menus", None)
    if isinstance(live, HttpMenus) and live.stats["live"] < live.budget:
        return True
    reach = left - 1 if max_depth is None else min(left - 1, max_depth)
    return all(dp.coverable(r, reach, width) for r in cand.reactants)


def _readable_trap_pick(board: Board, mid: str, dp: DP,
                        pol: "Policy") -> Optional[int]:
    """The best-scoring candidate whose failure is READABLE the turn after it lands.

    `_trap_pick` needs the recording to hold the trap's whole subtree, because
    `dead(exhausted)` may only be said after the ranked candidates came back
    failed -- and offline that test is nearly always empty.  Capping the check
    at depth 1 does not fix it, it only moves the failure: most trap episodes
    then walk into NoMenu and are dropped.  The search spent its budget
    elsewhere and a dead branch's children are not among the expanded
    molecules, so there is no recorded subtree to play.

    Two of the three `dead` reasons need no subtree at all.  `floor` is
    arithmetic -- no levels remain and nothing here is buyable.  `cutoff` needs
    one menu, the piece's own, which the cache has whenever the search expanded
    it.  So this selects for a candidate that lands the rollout on a piece dead
    for one of those two reasons: the mistake, the `dead` with a reason that is
    true on screen, and the correct re-rank, in three turns and entirely offline.

    What it deliberately does NOT produce is the deep trap -- the branch that
    looks fine for four levels and then runs out.  That one needs a live menu
    server, where any molecule can be expanded on demand.
    """
    m = board.mols[mid]
    left = board.left(mid)
    best, best_v = None, -INF
    for c in m.menu or []:
        if c.idx in m.failed or c.idx in m.applied:
            continue
        if dp.candidate_value(c, left, parent=m.smiles) > -INF:
            continue                      # it closes; not a trap
        if not _acyclic(board, mid, c):
            continue                      # the board would refuse it outright
        readable = False
        for r in c.reactants:
            if r in board.world.stock:
                continue                  # closed on arrival, says nothing
            if left - 1 <= 0:
                readable = True           # floor: arithmetic, no menu needed
                continue
            menu = board.world.menu(r)
            if not menu:
                readable = False          # a data gap, not a dead end
                break
            if pol.cutoff is not None:
                top = max((x.signals.get(dp.axis, 0.0) for x in menu), default=0.0)
                if top < pol.cutoff:
                    readable = True
        if not readable:
            continue
        v = c.signals.get(dp.axis)
        if v is not None and v > best_v:
            best, best_v = c.idx, v
    return best


def _greedy_pick(board: Board, mid: str, dp: DP) -> Optional[int]:
    """What a score-greedy policy would take: the top of the menu, untried."""
    m = board.mols[mid]
    best, best_v = None, -INF
    for c in m.menu or []:
        if c.idx in m.failed or c.idx in m.applied:
            continue
        # The probe is the one path that ignores the DP, so it has to carry the
        # board's own rules itself -- a probe the board refuses is not a probe.
        if not _acyclic(board, mid, c):
            continue
        v = c.signals.get(dp.axis)
        if v is not None and v > best_v:
            best, best_v = c.idx, v
    return best


# ---------------------------------------------------------------- the episode
@dataclass
class Turn:
    index: int
    env: str
    actions: list
    supervised: bool = True
    note: str = ""
    evidence: dict = field(default_factory=dict)
    thought: str = ""
    """The reasoning for THIS turn, written by the teacher.

    Empty until a distiller fills it. `harmony.episode_messages(analysis="text")` emits it as
    the analysis channel of this turn's assistant message, and skips turns where it is empty
    rather than emitting a blank channel."""


def _split_shape(evidence, product: str, precursors) -> dict:
    from .evidence import split_shape
    try:
        return split_shape(evidence.for_molecule(product),
                           [evidence.for_molecule(r) for r in precursors]) or {}
    except Exception:                                                  # noqa: BLE001
        return {}


BOND_SIDES = ("formed", "broken", "order_changed")


def _bond_rows(bond: dict, side: str) -> list:
    """One bond change, with the chemistry of BOTH ends rather than two atom-map indices.

    `reaction_get_bond_changes` returns, per changed bond: `atoms` ("C-Br"), `order` ("single"),
    and `fg1`/`fg2` -- what each end IS ("aromatic C", "Br (leaving group / halide)",
    "boronic/boronate B"). Only `atoms` and the two groups are reproducible chemistry; `at1`/
    `at2` are atom-map indices, an artefact of how the mapping ran.

    The indices alone lose the whole positional half of the fact. A teacher handed `C:1-C:2` can say WHICH
    ATOMS moved only by re-reading the SMILES and guessing; handed `C-C, aromatic C / carbonyl
    C` it can say where the bond sits without inventing anything, which is exactly what the
    reasoning is asked to write and exactly what no checker can verify if it is invented. The
    indices are kept alongside where the mapping produced them, because they still
    disambiguate two bonds of the same element pair.
    """
    out = []
    for x in (bond.get(side) or []):
        row = {"atoms": x.get("atoms"), "order": x.get("order"),
               "fg1": x.get("fg1"), "fg2": x.get("fg2")}
        if x.get("at1") and x.get("at2"):
            row["at"] = f"{x['at1']}-{x['at2']}"
        out.append({k: v for k, v in row.items() if v})
    return out


def _scope(board: Board, actions: list) -> list[str]:
    """Every molecule this turn's evidence has to cover: the open ones AND the ones it acts on.

    `open_mols()` alone is not enough. A rank does not only happen to an open molecule:
    re-ranking a molecule the board has already CLOSED is the normal way to build a second route
    through it, and such a molecule is not open, so the turn would come back with `mols`,
    `menus` and `facts` all empty -- and a teacher writing a row with no facts behind it is
    unconstrained, because every check here fails open when the fact is absent.
    """
    out = list(board.open_mols())
    for a in actions or []:
        for mid in ([getattr(a, "mid", None)] + list(getattr(a, "mids", None) or [])
                    + list(getattr(a, "choices", None) or {})):
            if mid and mid in board.mols and mid not in out:
                out.append(mid)
    return out


def evidence_for(board: Board, dp: DP, actions: list) -> dict:
    """The facts a <think> may cite on this turn, and nothing else.

    Collected here so that the teacher prompt and the faithfulness check read
    one table: a think block naming a number outside this dict is citing
    something the board never showed.
    """
    scope = _scope(board, actions)
    ev: dict = {"budget": [board.budget_used, board.budget_max], "menus": {}}
    for mid in scope:
        m = board.mols[mid]
        if m.menu is None:
            continue
        rows = []
        for c in m.menu:
            info = [board.world.info(r) for r in c.reactants]
            v = dp.candidate_value(c, board.left(mid), parent=m.smiles)
            rows.append({
                "c": c.idx,
                "signals": {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                            for k, v in c.signals.items()},
                "reactants": c.reactants,
                "buyable": [b for b, _ in info],
                "ln_price": [p for _, p in info],
                "dp": None if v == -INF else round(v, 4),
                "solves": v > -INF,
                "label": dp.labels.of(m.smiles, c.idx) if dp.labels else None,
            })
        ev["menus"][mid] = {"depth": m.depth, "left": board.left(mid),
                            "candidates": rows}

    untried = []
    for mid in board.order:
        m = board.mols[mid]
        for c in m.menu or []:
            if c.idx in m.applied or c.idx in m.failed:
                continue
            untried.append({"mol": mid, "c": c.idx, "turn": m.menu_turn,
                            "signals": {k: (round(v, 4) if isinstance(v, (int, float))
                                            else v)
                                        for k, v in c.signals.items()}})
    ev["untried"] = untried
    if board.routes:
        from .render import _route_rxns, signals_of_rxn, route_axes, route_front
        ev["routes"] = []
        # The AXES, not just the weakest step: `factcheck._c_numbers` builds its allowed set
        # of printed costs partly from the route records, so without them the ROUTES line's
        # own total is missing from the set and a draft quoting `$.218` off the screen is
        # refuted as having invented it.
        front, beaten = route_front(board)
        for route in board.routes:
            vals = [signals_of_rxn(board.rxns[r])[1]
                    for r in _route_rxns(board, route.root_rxn)]
            vals = [v for v in vals if v is not None]
            ax = route_axes(board, route)
            ev["routes"].append({"route": route.label,
                                 "weakest": min(vals) if vals else None,
                                 "steps": ax["steps"], "leaves": ax["leaves"],
                                 "p_pass": ax["p"], "p_min": ax["p_min"],
                                 "p_pass_n": ax["p_pass"], "rt_back": ax["rt"],
                                 "rt_worst": ax["rt_worst"], "cost": ax["d"],
                                 "cost_ln": ax["cost_ln"], "unpriced": ax["unpriced"],
                                 "front": route.label in front,
                                 "beaten_by": beaten.get(route.label) or []})
    # Which candidates have had evidence bought, and what came back. The distinction matters
    # to the faithfulness check as much as to the teacher: a thought citing an atom position
    # for a candidate nobody analysed is citing a fact the board never showed, and without
    # this map that is indistinguishable from a legitimate citation.
    ev["analyzed"] = {}
    if board.evidence is not None:
        for mid in scope:
            m = board.mols[mid]
            if m.analyzed_turn is None or not m.menu:
                continue
            rows = []
            for c in m.menu:
                if c.idx not in (m.analyzed or []):
                    continue
                fx = board.evidence.for_step(m.smiles, c.reactants)
                nm = fx.get("named") or {}
                bond = fx.get("bond") or {}
                rows.append({
                    "c": c.idx,
                    "named_tier": nm.get("tier"),
                    "named": nm.get("names") or [],
                    "named_match": nm.get("match"),
                    "formed": _bond_rows(bond, "formed"),
                    "broken": _bond_rows(bond, "broken"),
                    "changed": _bond_rows(bond, "order_changed"),
                    "center_fg": bond.get("center_fg") or {},
                    "bond_measured": bool(bond),
                })
            if rows:
                ev["analyzed"][mid] = {"turn": m.analyzed_turn, "candidates": rows}
    ev["actions"] = [type(a).__name__.lower() for a in actions]
    # ...and the same actions with their arguments, which the faithfulness check needs and the
    # type names alone cannot give: "which candidate does this turn actually TAKE" is the head
    # of a rank order and every choice of a done, and a claim about the step being taken can
    # only be checked against the candidate that was taken. Prefixed with `_` because it is
    # machinery for the checker rather than a fact the teacher is shown; `support_block` reads
    # the menus and facts, not this.
    ev["_actions_detail"] = [
        {"type": type(a).__name__.lower(),
         **({"mid": a.mid} if getattr(a, "mid", None) else {}),
         **({"order": list(a.order or [])} if getattr(a, "order", None) else {}),
         **({"take": int(a.take)} if int(getattr(a, "take", 1) or 1) > 1 else {}),
         **({"choices": dict(a.choices or {})} if getattr(a, "choices", None) else {})}
        for a in actions]

    # ---- teacher-private facts -------------------------------------------------------
    # Everything below is for the TEACHER only and never reaches the board. `ev["analyzed"]`
    # above is gated on the agent having taken `analyze`; with the action out of the tool
    # schema that map is always empty, so the mechanism half of the
    # reasoning brief would have nothing to draw on. These keys collect the same facts
    # unconditionally, so the teacher can write reasoning that CONTAINS them and the student
    # learns to produce them rather than to ask for them.
    ranked = {a.mid: list(getattr(a, "order", []) or []) for a in actions
              if type(a).__name__.lower() == "rank" and getattr(a, "mid", None)}
    if board.evidence is not None:
        ev["facts"] = {}
        # WHICH AXIS ACTUALLY SEPARATED THE TAKEN CANDIDATE, computed the way the DP
        # decides so the two cannot disagree: `p` unless the leaders tie within tie_eps,
        # then `rt`, then `price`, then `q`.
        #
        # Without this the teacher has to infer the reason for an action it did not take,
        # from a screen that shows the numbers but not which of them was decisive -- and it
        # infers it wrong, rarely naming a tie-break axis that did decide. Rewording the
        # guidance does not help when the prompt does not carry the fact.
        ev["decided_by"] = _decided_by(board, dp, ranked)
        # Off unless BOARD_PRICE_NOTE=1, so the price note can be ablated on its own.
        if os.environ.get("BOARD_PRICE_NOTE") == "1":
            ev["price_note"] = _price_note(board, ranked)
        for mid in scope:
            m = board.mols[mid]
            if not m.menu:
                continue
            # The candidates this turn is actually choosing between: the ones the ranking
            # names, plus the head of the menu for contrast. Capped, because a ten-wide menu
            # described in full is prompt the decision does not turn on.
            want = list(dict.fromkeys(list(ranked.get(mid) or []) + [c.idx for c in m.menu]))[:6]
            rows = []
            for c in m.menu:
                if c.idx not in want:
                    continue
                fx = board.evidence.for_step(m.smiles, c.reactants)
                nm = fx.get("named") or {}
                bond = fx.get("bond") or {}
                rows.append({
                    "c": c.idx,
                    "named_tier": nm.get("tier"), "named": nm.get("names") or [],
                    "named_match": nm.get("match"),
                    "formed": _bond_rows(bond, "formed"),
                    "broken": _bond_rows(bond, "broken"),
                    # A bond whose ORDER moved without being made or broken -- an oxidation, an
                    # imine reduction, a tautomerisation. Some steps carry ONLY this, so
                    # dropping it would render an alcohol-to-aldehyde oxidation as "no bond
                    # change measured".
                    "changed": _bond_rows(bond, "order_changed"),
                    # What each end of the changed bond IS, keyed by atom label. The reaction
                    # centre in the mapper's own vocabulary, and the only place a teacher can
                    # read a bond's position off something rather than infer it.
                    "center_fg": bond.get("center_fg") or {},
                    "bond_measured": bool(bond),
                    # for_step returns bond and named only; the split shape is derived from
                    # the descriptors of the product and its precursors, the same way
                    # render.evidence_block derived it when analyze put it on the board.
                    "shape": _split_shape(board.evidence, m.smiles, c.reactants),
                })
            if rows:
                ev["facts"][mid] = {"candidates": rows}

    # What an `open` turn has to reason from. An open turn has no candidates yet -- that is
    # why it is opening -- so its whole information set is the piece itself: what it is
    # structurally, that it is NOT purchasable (which is why it must be made at all), how
    # deep it sits, and what the route has already banked around it.
    ev["mols"] = {}
    for mid in scope:
        m = board.mols[mid]
        buyable, price = board.world.info(m.smiles)
        row = {"smiles": m.smiles, "depth": m.depth, "max_depth": board.max_depth,
               "buyable": buyable, "ln_price": price, "has_menu": m.menu is not None,
               "under": m.parent_rxn}
        if board.evidence is not None:
            row["desc"] = board.evidence.for_molecule(m.smiles) or {}
        ev["mols"][mid] = row
    ev["banked"] = [{"mol": mid, "smiles": board.mols[mid].smiles,
                     "ln_price": board.world.info(board.mols[mid].smiles)[1]}
                    for mid in board.closed_leaves()]
    return ev


def run_episode(target: str, world: DataWorld, dp: DP, pol: Policy = None, style=None,
                max_depth: int = 10, budget: int = 300, max_turns: int = 60,
                collect_evidence: bool = True, evidence=None):
    """Drive the board and render every turn.  Returns (turns, board)."""
    from . import render as R

    st = style or R.STYLE
    pol = pol or Policy()
    board = Board(target, world, max_depth=max_depth, budget=budget,
                  max_routes=pol.max_routes, evidence=evidence)
    state = {"mistakes": 0}
    turns: list[Turn] = []
    for t in range(max_turns):
        env = R.render_env(board, st)
        step = teacher_turn(board, dp, pol, state)
        turns.append(Turn(
            index=t, env=env, actions=step.actions,
            supervised=step.supervised, note=step.note,
            evidence=evidence_for(board, dp, step.actions) if collect_evidence else {},
        ))
        if any(isinstance(a, Terminate) for a in step.actions):
            break
        try:
            board.apply(step.actions)
        except NoMenu as exc:
            # NoMenu subclasses BoardError, so it has to be caught FIRST -- with
            # the broad clause above it, re-raising from inside that handler
            # escapes the try entirely and kills the build.
            turns[-1].note = (turns[-1].note + "; " if turns[-1].note else "") + \
                f"DROPPED: {exc}"
            board.invalid = str(exc)
            break
        except BoardError as exc:
            turns[-1].note = (turns[-1].note + "; " if turns[-1].note else "") + \
                f"DROPPED: the labeller proposed something the board refuses: {exc}"
            board.invalid = str(exc)
            break
    return turns, board
