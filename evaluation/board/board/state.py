#!/usr/bin/env python
"""The board: molecules, reactions, and the transitions four actions can cause.

This module owns SEMANTICS only -- no text.  render.py turns a Board into the
environment turn the model reads; parse.py turns the model's <act> block back
into the Action objects applied here.  Keeping the three apart is what makes the
format auditable: a rendering change cannot silently change the game, and a
parse bug shows up as a round-trip failure instead of a mislabelled episode.

The board is passive by construction.  Nothing moves without an Action:

  Open(mid)              fetch mid's ten candidate disconnections.  Costs budget.
  Rank(mid, [i, j, ...]) declare an ordering; the FIRST entry is applied now and
                         becomes a reaction, the rest are recorded as intent.
  Dead(mid, reason)      mid cannot be carried further; its parent reaction FAILS
                         and the grandparent molecule comes back open.
  Done(note)             stop and hand over a route.

A reaction is an AND: it closes only when every piece closes.  A molecule is an
OR: it may carry several reactions, and it closes when any one of them closes.
Closing and failing both cascade upward, and the cascade is recorded on the
event so the renderer can show it as one line instead of re-deriving it.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

# ---------------------------------------------------------------- molecule ids
# Two characters, assigned at first appearance and never reused.  Must not look
# like a candidate ref (c0..c9) or a reaction ref (r1, r2, ...), and must not be
# two digits -- a bare number in the transcript is always a score or a count.
_ID_LETTERS = "abdefghjkmnpqstuvwxyz"   # no c/r (reserved), no i/l/o (ambiguous)
_ID_DIGITS = "23456789"                 # no 0/1 (ambiguous with o/l)


def id_pool(seed: str) -> list[str]:
    """Every legal two-character id, shuffled deterministically from `seed`."""
    pool = [a + b for a in _ID_LETTERS for b in _ID_DIGITS]
    pool += [a + b for a in _ID_LETTERS for b in _ID_LETTERS if a != b]
    random.Random(seed).shuffle(pool)
    return pool


# -------------------------------------------------------------------- entities
@dataclass
class Candidate:
    """One disconnection on a molecule's menu.

    `idx` is the c-number: fixed at the moment the molecule was opened, in the
    single-step model's own order, so a signal is deliberately NOT monotone in
    the index.  `signals` is open-ended -- whatever the run actually scored.
    Known keys, all optional: q (model confidence), p (filter plausibility),
    rt (forward-model rank).  A key that is absent is not rendered.
    """

    idx: int
    reactants: list[str]
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class Molecule:
    mid: str
    smiles: str
    depth: int
    buyable: bool
    price_ln: Optional[float] = None
    status: str = "open"                       # open | closed | dead
    parent_rxn: Optional[str] = None           # the reaction that produced it
    menu: Optional[list[Candidate]] = None     # None until opened
    menu_turn: Optional[int] = None            # which turn the menu arrived in
    ranking: list[int] = field(default_factory=list)       # current declaration
    prev_ranking: Optional[list[int]] = None   # the one before it, for "(was ...)"
    ranking_turn: Optional[int] = None         # when the ranking was last declared
    failed: list[int] = field(default_factory=list)         # tried and came back dead
    applied: list[int] = field(default_factory=list)        # spent on a reaction
    rxns: list[str] = field(default_factory=list)           # reactions under it
    closed_via: Optional[str] = None           # which reaction closed it
    dead_reason: Optional[str] = None
    first_seen: int = 0
    analyzed_turn: Optional[int] = None        # when `analyze` last bought evidence for it
    analyzed: list = field(default_factory=list)
    """Candidate numbers evidence has been bought for. The renderer draws only these, so a
    partial analyze produces a partial EVIDENCE block -- which is the honest thing: a
    candidate nobody paid to analyse has no facts on screen and must not be reasoned about
    as if it did."""

    @property
    def untried(self) -> list[int]:
        return [i for i in self.ranking if i not in self.failed and i not in self.applied]


@dataclass
class Reaction:
    rid: str
    parent: str                # molecule id this reaction sits under
    cand: int                  # which candidate of the parent it applies
    pieces: list[str] = field(default_factory=list)
    status: str = "open"       # open | solved | failed
    signals: dict[str, float] = field(default_factory=dict)

    def n_closed(self, board: "Board") -> int:
        return sum(1 for p in self.pieces if board.mols[p].status == "closed")


@dataclass
class Route:
    """One complete solution: a reaction chosen at every molecule it uses.

    `root_rxn` alone was not enough.  With it, the tree below was read off each
    molecule's `closed_via` -- whichever reaction happened to close it FIRST --
    so two solutions that differ at an interior node were the same route as far
    as the board could tell, and the second one was never registered.  Carrying
    the choice per molecule is what lets one episode hold several routes that
    diverge anywhere, which is the point of continuing past the first solve.
    """

    label: str                 # A, B, C ...
    root_rxn: str
    turn: int
    choices: dict = field(default_factory=dict)   # molecule id -> reaction id

    def steps(self, board: "Board") -> frozenset:
        """(product, frozenset(reactants)) per step -- the route's identity."""
        out = []
        for mid, rid in self.choices.items():
            rxn = board.rxns[rid]
            out.append((board.mols[mid].smiles,
                        frozenset(board.mols[p].smiles for p in rxn.pieces)))
        return frozenset(out)


# --------------------------------------------------------------------- actions
@dataclass
class Open:
    mid: str


@dataclass
class Rank:
    mid: str
    order: list[int]
    # How many of `order` to commit NOW. 1 is the serial board: the head is applied and the
    # rest stay on the menu for a later turn to claim, so the OPEN frontier rarely holds more
    # than one molecule and `open` has no choice to reason about.
    # `take > 1` materialises the OR-branch instead of deferring it -- the same reactions, in
    # the same DAG, laid out in SPACE rather than in time (multi-route expansion): fewer
    # turns, and a frontier with several molecules to open at once.
    take: int = 1


@dataclass
class Dead:
    mid: str
    reason: str                # one of DEAD_REASONS


@dataclass
class Done:
    """Claim a route: one candidate number per molecule the route uses.

    A claim is a statement ABOUT THE GRAPH, not a walk through it.  Many of the
    routes an episode finds after its first need no molecule opened that is not
    open already -- they are sitting in the DAG, and walking them costs turns for
    nothing.  Naming the choice set makes such a route one action.

    `choices` is molecule id -> candidate number, which is how the reaction ledger
    already names reactions (`r1  qm·c0`).  Candidate numbers rather than reaction
    ids, for two reasons: a candidate can be named BEFORE its reaction exists, so
    a route needing a reaction the board has not made yet is still one action; and
    a JSON object cannot repeat a key, so "two reactions for one molecule" stops
    being a reachable error.
    """

    choices: dict = field(default_factory=dict)     # molecule id -> candidate no.
    note: str = ""


@dataclass
class Analyze:
    """Buy the structural and mechanistic evidence for molecules already opened.

    The board shows `q` and `ln$`. It does not show which bond a disconnection moves,
    whether the transformation has a name, or what the fragments structurally are -- none of
    which is derivable from board state, all of which a reason has to cite. This action asks
    for them, for every molecule named, in ONE call: the tools behind it are batched per
    product, so N separate analyze actions are the same work N times over.

    Charged nothing in COST because it calls no single-step model -- it is RDKit, RXNMapper
    and a template corpus. It is still an action and still has to be asked for: a model that
    cites a bond position without having requested the mapping is reciting a fact it was
    never given.

    Legal only on a molecule that HAS a menu. Analysing candidates that do not exist yet is a
    request for the expander, and that request is `open`.
    """

    mids: list = field(default_factory=list)
    """Molecules to analyse. Kept for the whole-menu form: `analyze mids=[qm, bj]`."""

    candidates: dict = field(default_factory=dict)
    """molecule id -> the candidate numbers to analyse, e.g. {"qm": [0, 2, 4]}.

    The per-candidate form, and the one an agent should reach for: analysing a ten-wide menu
    to choose between three offers pays for seven answers nobody reads, and each answer costs
    an atom mapping. `mids` and `candidates` may both be given -- a molecule in `mids` is
    analysed whole, one in `candidates` only where named."""

    note: str = ""

    def wanted(self, board) -> dict:
        """-> {mid: [candidate numbers]} with `mids` expanded to the whole menu."""
        out = {m: sorted(set(v)) for m, v in (self.candidates or {}).items()}
        for m in self.mids or []:
            mol = board.mols.get(m)
            if mol is not None and mol.menu:
                out[m] = [c.idx for c in mol.menu]
        return out


@dataclass
class Terminate:
    """Stop, and hand over the routes claimed so far.

    Separate from `done` because `done` no longer ends anything: an episode
    claims several routes and then stops, so the two decisions are two actions.
    This one is never sent as a tool call -- it renders as the final message.
    """

    note: str = ""


Action = Open | Rank | Dead | Done | Analyze | Terminate

# The three named reasons.  `dead` is only legal when one of them holds, and the
# label policy picks which -- so the strings live here, not in the renderer.
DEAD_REASONS = {
    "cutoff": "every candidate scores below the cut-off",
    "exhausted": "every candidate I ranked has come back failed",
    "floor": "no levels remain and nothing here is buyable",
}


# ---------------------------------------------------------------------- events
@dataclass
class Event:
    """What the environment reports back.  `kind` selects the renderer."""

    kind: str                                   # rank | dead | exhausted | solved
    mid: Optional[str] = None
    rxn: Optional[str] = None
    ranking: list[int] = field(default_factory=list)
    prev_ranking: Optional[list[int]] = None
    applied: Optional[int] = None
    pieces: list[str] = field(default_factory=list)
    cascade: list[str] = field(default_factory=list)   # rendered-ready fragments
    reason: Optional[str] = None
    route: Optional[str] = None
    reopened: Optional[str] = None


class World(Protocol):
    """Everything the board cannot know from its own state."""

    def menu(self, smiles: str) -> list[Candidate]: ...
    def info(self, smiles: str) -> tuple[bool, Optional[float]]: ...   # buyable, ln$


# ----------------------------------------------------------------------- board
COST = {"open": 1, "rank": 0, "dead": 0, "done": 0, "analyze": 0}
"""Budget charged per action.

Only `open` calls the single-step model, so only `open` costs: the budget counts
single-step calls.  Change it here if the budget line should count committed
reactions too; nothing else reads the numbers.
"""


class Board:
    def __init__(
        self,
        target: str,
        world: World,
        max_depth: int = 10,
        budget: int = 300,
        cost: Optional[dict[str, int]] = None,
        max_routes: int = 0,
        auto_routes: bool = False,
        evidence=None,
    ):
        self.world = world
        self.evidence = evidence
        """Structural/mechanistic facts, or None if this board cannot buy them.

        Injected like `world` and for the same reason: bond changes, earned reaction names and
        scaffolds are not derivable from board state. `None` makes `analyze` an error rather
        than a silent no-op -- a board that answers the action with nothing would teach the
        model to ask and then reason as if it had been told."""
        self.max_depth = max_depth
        self.budget_max = budget
        self.budget_used = 0
        self.cost = dict(cost or COST)
        self.turn = 0
        self.mols: dict[str, Molecule] = {}
        self.order: list[str] = []              # molecule ids, first-seen order
        self.rxns: dict[str, Reaction] = {}
        self.rxn_order: list[str] = []
        self.routes: list[Route] = []
        self.max_routes = max_routes
        """Stop registering routes past this many (0 = no cap).

        Not a rendering choice: the board keeps every route it registers for the
        rest of the episode, and one new reaction under a shared molecule spawns a
        sibling of every route that uses it, so the count runs ahead of what was
        asked for.  The cap keeps the episode's cost linear in the answers actually wanted."""
        self.routes_suppressed = 0
        # A route exists because the model CLAIMED it, not because the board
        # noticed the root could close.  With auto-registration on, `done` is
        # decorative: the board hands over the answer and the claim only agrees
        # with it.  What the board still reports is the graph -- the ledger, the
        # closed leaves, `vg has an alternative via r5` -- which is what a claim
        # is read off.  Left switchable so corpora built with it on stay
        # rebuildable.
        self.auto_routes = auto_routes
        self._claiming = False                   # see _settle: a claim registers
                                                 # its own choices, not closed_via
        self.events: list[Event] = []            # produced by the last apply()
        self.shown: dict[str, int] = {}          # id -> turn its SMILES was printed
        self.printed: set[str] = set()           # ids printed in the current render pass
        self._ids = id_pool(target)
        self._by_smiles: dict[str, str] = {}
        self.root = self.add_molecule(target, depth=0)

    # -- construction ------------------------------------------------------
    def add_molecule(self, smiles: str, depth: int, parent_rxn: str = None) -> str:
        """Return the id for `smiles`, creating it on first sight.

        A molecule seen twice is ONE node (the graph is a DAG, not a tree): the
        second parent gets the same id and, if it is already closed, gets it for
        free.  `depth` keeps the shallower of the two -- the piece is reachable
        at either, and the shallower one is the honest budget statement.
        """
        if smiles in self._by_smiles:
            mid = self._by_smiles[smiles]
            self.mols[mid].depth = min(self.mols[mid].depth, depth)
            return mid
        mid = self._ids.pop(0)
        buyable, price = self.world.info(smiles)
        self.mols[mid] = Molecule(
            mid=mid, smiles=smiles, depth=depth, buyable=buyable, price_ln=price,
            parent_rxn=parent_rxn, first_seen=self.turn,
            status="closed" if buyable else "open",
            closed_via="stock" if buyable else None,
        )
        self._by_smiles[smiles] = mid
        self.order.append(mid)
        return mid

    def mid_of(self, smiles: str) -> Optional[str]:
        return self._by_smiles.get(smiles)

    # -- views used by the renderer and the policy -------------------------
    def left(self, mid: str) -> int:
        """Levels of budget below this piece.  0 left => must already be buyable."""
        return self.max_depth - self.mols[mid].depth

    def open_mols(self) -> list[str]:
        """Every piece still owed a decision, in first-seen order."""
        return [
            m for m in self.order
            if self.mols[m].status == "open" and not self._has_unresolved_rxn(m)
        ]

    def _has_unresolved_rxn(self, mid: str) -> bool:
        return any(self.rxns[r].status == "open" for r in self.mols[mid].rxns)

    def closed_leaves(self) -> list[str]:
        """The running bill of materials: purchasable leaves banked so far."""
        return [m for m in self.order if self.mols[m].buyable and self.mols[m].status == "closed"]

    def dead_mols(self) -> list[str]:
        return [m for m in self.order if self.mols[m].status == "dead"]

    def solved(self) -> bool:
        return self.mols[self.root].status == "closed"

    # -- transitions -------------------------------------------------------
    def apply(self, actions: Iterable[Action]) -> list[Event]:
        """Apply one turn's actions IN ORDER and return the events to report.

        Order is load-bearing: `dead 9x` fails r3, which is what leaves 7h
        without an unresolved reaction and makes `rank 7h ...` legal on the next
        line.  Anything illegal raises -- the caller decides whether that is a
        builder bug (labels) or a model error (rollout).
        """
        self.turn += 1
        self.events = []
        touched: set[str] = set()
        for act in actions:
            mid = getattr(act, "mid", None)
            if mid is not None:
                if mid in touched:
                    raise BoardError(f"two actions on {mid} in one turn")
                touched.add(mid)
            if isinstance(act, Open):
                self._do_open(act)
            elif isinstance(act, Rank):
                self._do_rank(act)
            elif isinstance(act, Dead):
                self._do_dead(act)
            elif isinstance(act, Done):
                self._do_done(act)
            elif isinstance(act, Analyze):
                self._do_analyze(act)
            elif isinstance(act, Terminate):
                pass
            else:
                raise BoardError(f"unknown action {act!r}")
        return self.events

    def _do_analyze(self, act: Analyze) -> None:
        """Mark molecules as analysed; the renderer draws the evidence block.

        The board stores WHICH molecules were analysed, never the evidence itself. The facts
        come from `self.evidence` (an Evidence provider, injected like `world` because the
        board cannot know them from its own state), so a replay reads the same caches rather
        than a stale copy baked into a board snapshot.

        Analysing the same molecule twice is not an error and is not free of consequence: the
        board records the turn, so a second request is visible as a repeat rather than
        silently ignored.
        """
        want = act.wanted(self)
        if not want:
            raise BoardError("analyze needs mids=[...] or candidates={mid: [c, ...]}")
        if self.evidence is None:
            raise BoardError("analyze requested but this board has no evidence provider")
        seen = []
        for mid, cands in want.items():
            m = self._live(mid)
            if m.menu is None:
                raise BoardError(f"{mid} has no candidates to analyse; open it first")
            have = {c.idx for c in m.menu}
            bad = [c for c in cands if c not in have]
            if bad:
                raise BoardError(f"{mid} has no candidate {bad[0]} "
                                 f"(it offers {sorted(have)})")
            m.analyzed_turn = self.turn
            m.analyzed = sorted(set(m.analyzed) | set(cands))
            seen.append(f"{mid}·" + ",".join("c" + str(c) for c in cands))
        self.budget_used += self.cost["analyze"]
        self.events.append(Event(kind="analyzed", pieces=seen))

    def _do_open(self, act: Open) -> None:
        m = self._live(act.mid)
        if m.menu is not None:
            raise BoardError(f"{m.mid} already has candidates on screen")
        if m.buyable:
            raise BoardError(f"{m.mid} is purchasable; nothing to open")
        # With BOARD_MENU_FILTER=1, cycle candidates are never put on screen at all.
        #
        # Why: the search baselines on the same benchmark never see these candidates.
        # syntheseus drops reactions that have the root molecule as a precursor (with
        # prevent_repeat_mol_in_trees, the whole A->..->A), and RetroAgent filters cycle
        # reactants, then refills from the raw top-50 to keep top_k. Without the filter the
        # board shows cycle candidates as-is, and picking one is rejected by the guard in
        # _do_rank -- without the prompt ever stating that rule.
        #
        # Off by default. It changes which candidates are on screen, so results with and
        # without it are not comparable.
        keep = None
        if os.environ.get("BOARD_MENU_FILTER", "0") == "1":
            # With BOARD_MENU_DEDUP_RXN=1, (product, reactants) pairs already in the graph are
            # also removed from the screen. Same rule as RetroAgent's duplicate-reaction check.
            # Re-proposing the same reaction is a zero-information action
            # that spends budget without advancing the search -- the search arm does not expand
            # the same molecule twice either. Our board's "already been tried" is indexed within
            # one menu, so it could not stop reaching the same reaction via a different path.
            dedup = os.environ.get("BOARD_MENU_DEDUP_RXN", "0") == "1"
            seen_rxn = set()
            if dedup:
                for _rid, _rx in self.rxns.items():
                    _prod = self.mols[_rx.parent].smiles
                    seen_rxn.add((_prod, frozenset(self.mols[x].smiles for x in _rx.pieces)))

            def keep(reactants, _mid=m.mid, _smi=m.smiles, _seen=seen_rxn, _dd=dedup):
                if _dd and (_smi, frozenset(reactants)) in _seen:
                    return False
                for smi in reactants:
                    existing = self._by_smiles.get(smi)
                    if existing is not None and (existing == _mid
                                                 or self._reaches(existing, _mid)):
                        return False
                return True
        menu = self.world.menu(m.smiles, keep=keep) if keep is not None \
            else self.world.menu(m.smiles)
        if not menu:
            raise NoMenu(f"{m.mid} has no recorded menu ({m.smiles})")
        m.menu = menu
        m.menu_turn = self.turn
        self.budget_used += self.cost["open"]
        # No event: the menu simply appears in the OPEN block.

    def _do_rank(self, act: Rank) -> None:
        m = self._live(act.mid)
        if m.menu is None:
            raise BoardError(f"{m.mid} has no candidates on screen")
        if self._has_unresolved_rxn(m.mid):
            raise BoardError(f"{m.mid} has an unresolved reaction")
        if not act.order:
            raise BoardError(f"empty ranking on {m.mid}; use dead")
        known = {c.idx for c in m.menu}
        for i in act.order:
            if i not in known:
                raise BoardError(f"{m.mid} has no candidate c{i}")
            if i in m.failed or i in m.applied:
                raise BoardError(f"{m.mid} c{i} has already been tried")
        # A candidate may not bring in a molecule that is already ABOVE this one:
        # that makes the molecule its own precursor, and a route cannot contain
        # itself.  The DP never picks such a candidate (it carries a cycle guard),
        # so no training example has one -- but a model at inference can, and
        # without this the board accepts it, the ancestor is left with an
        # unresolved reaction, OPEN goes empty and there is no legal move left.
        take = max(1, min(int(getattr(act, "take", 1) or 1), len(act.order)))
        # The cycle guard runs on every candidate being COMMITTED, not just the head. It used
        # to check `order[0]` alone because only `order[0]` was applied; with `take > 1` a
        # second commit could walk back up the route and the board would accept it.
        for i in act.order[:take]:
            cand_check = next(c for c in m.menu if c.idx == i)
            for smi in cand_check.reactants:
                existing = self._by_smiles.get(smi)
                if existing is None:
                    continue
                if existing == m.mid or self._reaches(existing, m.mid):
                    raise BoardError(
                        f"c{i} would make {existing} its own precursor "
                        f"({existing} is already above {m.mid}); rank another candidate"
                    )

        m.prev_ranking = list(m.ranking) if m.ranking else None
        m.ranking = list(act.order)
        m.ranking_turn = self.turn
        self.budget_used += self.cost["rank"]

        for i in act.order[:take]:
            self._apply_candidate(m, i)

    def _apply_candidate(self, m: Molecule, take: int,
                         via_claim: bool = False) -> str:
        """Commit one candidate of `m` to a reaction and settle upwards.

        Shared by `rank` and by `done`: a claim that needs a reaction the board
        has not made yet makes it the same way ranking does, so the two paths
        cannot produce differently-shaped reactions.
        """
        cand = next(c for c in m.menu if c.idx == take)
        rid = f"r{len(self.rxn_order) + 1}"
        rxn = Reaction(rid=rid, parent=m.mid, cand=take, signals=dict(cand.signals))
        self.rxns[rid] = rxn
        self.rxn_order.append(rid)
        m.rxns.append(rid)
        m.applied.append(take)
        for smi in cand.reactants:
            rxn.pieces.append(self.add_molecule(smi, m.depth + 1, parent_rxn=rid))
        ev = Event(
            # A claim declares no ordering, so reporting one ("vg ranked []")
            # would describe a move the model did not make.
            kind="apply" if via_claim else "rank", mid=m.mid, rxn=rid,
            ranking=[] if via_claim else list(m.ranking),
            prev_ranking=None if via_claim else m.prev_ranking,
            applied=take, pieces=list(rxn.pieces),
        )
        self.events.append(ev)
        self._settle(rid, ev)
        return rid

    def rxn_for(self, mid: str, cand: int) -> Optional[str]:
        """The reaction that applies candidate `cand` of `mid`, if it exists."""
        for rid in self.mols[mid].rxns:
            r = self.rxns[rid]
            if r.cand == cand and r.status != "failed":
                return rid
        return None

    def _do_done(self, act: Done) -> None:
        """Claim a route the board can make from what is on screen.

        Every check runs BEFORE anything is applied, so a rejected claim leaves
        the board exactly as it was.  That matters more here than elsewhere: a
        claim can commit several reactions at once, and a half-applied rejected
        claim would leave the model looking at a board its own rejected action had
        changed.
        """
        if not act.choices:
            raise BoardError("done needs choices: a candidate number per molecule")

        # 1. every entry names something the model could actually read off the
        #    board.  A molecule with no menu has no candidate numbers at all, so
        #    naming one is a rejection rather than a fetch -- `open` is the action
        #    that fetches.
        for mid, c in act.choices.items():
            if mid not in self.mols:
                raise BoardError(f"{mid} is not a molecule on the board")
            m = self.mols[mid]
            if m.status == "dead":
                raise BoardError(f"{mid} is dead")
            if m.buyable:
                raise BoardError(f"{mid} is purchasable; it needs no reaction")
            if m.menu is None:
                raise BoardError(f"{mid} has no candidates on screen, so c{c} is "
                                 f"not a candidate you can name")
            if c not in {cd.idx for cd in m.menu}:
                raise BoardError(f"{mid} has no candidate c{c}")
            if c in m.failed:
                raise BoardError(f"{mid} c{c} has come back failed")

        # 2. walk down from the target.  The choices have to cover every molecule
        #    the route leaves to make, and every leaf has to be purchasable.
        plan: list[tuple[str, int, int]] = []          # (mid, candidate, depth)
        used: set[str] = set()
        stack: list[tuple[str, int, tuple]] = [(self.root, 0, ())]
        while stack:
            mid, depth, path = stack.pop()
            if mid in path:
                raise BoardError(f"these choices make {mid} its own precursor")
            m = self.mols[mid]
            if mid in act.choices:
                if mid in used:
                    continue                      # a shared piece, reached twice
                used.add(mid)
                c = act.choices[mid]
                cand = next(cd for cd in m.menu if cd.idx == c)
                plan.append((mid, c, depth))
                for smi in cand.reactants:
                    piece = self._by_smiles.get(smi)
                    if piece is not None:
                        stack.append((piece, depth + 1, path + (mid,)))
                        continue
                    # The piece is not on the board yet, so no choice could have
                    # been made for it: it has to be purchasable or the route
                    # does not close.
                    buyable, _ = self.world.info(smi)
                    if not buyable:
                        raise BoardError(
                            f"{smi} is left to make and is not purchasable; open "
                            f"it and choose a candidate for it first")
            elif m.buyable:
                continue                                       # a leaf
            else:
                raise BoardError(f"nothing chosen for {mid}, which this route "
                                 f"leaves to make")
        if self.root not in used:
            raise BoardError("these choices do not say how to make the target")
        extra = set(act.choices) - used
        if extra:
            raise BoardError(f"{', '.join(sorted(extra))} cannot be reached from "
                             f"the target through these choices")

        # 3. duplicates and the cap, still before applying.  The route's identity
        #    is (product, reactants) per step, which the plan already determines
        #    -- so this is answerable without committing anything.
        fresh = frozenset(
            (self.mols[mid].smiles,
             frozenset(next(cd for cd in self.mols[mid].menu
                            if cd.idx == c).reactants))
            for mid, c, _ in plan)
        for r in self.routes:
            if r.steps(self) == fresh:
                raise BoardError(f"that is route {r.label}")
        if self.max_routes and len(self.routes) >= self.max_routes:
            raise BoardError(f"the board is holding {self.max_routes} routes "
                             f"already; hand over instead")

        # 4. commit.  Deepest first, so every reaction settles once its own
        #    pieces are resolved.  A (molecule, candidate) already applied is
        #    reused -- that is what makes a route the graph already holds free.
        choices: dict[str, str] = {}
        self._claiming = True
        try:
            for mid, c, _ in sorted(plan, key=lambda t: -t[2]):
                rid = self.rxn_for(mid, c)
                if rid is None:
                    rid = self._apply_candidate(self.mols[mid], c, via_claim=True)
                choices[mid] = rid
            # Registration stays INSIDE the claim: outside it, _register emits its
            # own "ROUTE SOLVED" and the turn reports the same route twice.
            route = self._register(choices, choices[self.root])
        finally:
            self._claiming = False
        if route is None:                       # both causes were ruled out above
            raise BoardError("that route could not be registered")
        self.events.append(Event(kind="claim", mid=self.root,
                                 rxn=choices[self.root], route=route.label,
                                 pieces=[choices[m] for m, _, _ in plan]))

    def _do_dead(self, act: Dead) -> None:
        m = self._live(act.mid)
        if act.reason not in DEAD_REASONS:
            raise BoardError(f"unknown dead reason {act.reason!r}")
        m.status = "dead"
        m.dead_reason = act.reason
        ev = Event(kind="dead", mid=m.mid, reason=act.reason)
        self.events.append(ev)
        if m.parent_rxn:
            self._fail(m.parent_rxn, ev)

    # -- cascades ----------------------------------------------------------
    def _settle(self, rid: str, ev: Event) -> None:
        """Close upward as far as the new pieces allow, recording the chain."""
        rxn = self.rxns[rid]
        if rxn.status != "open":
            return
        if any(self.mols[p].status == "dead" for p in rxn.pieces):
            self._fail(rid, ev)
            return
        if rxn.n_closed(self) < len(rxn.pieces):
            return
        rxn.status = "solved"
        ev.cascade.append(f"{rid} {len(rxn.pieces)} of {len(rxn.pieces)} closed")
        parent = self.mols[rxn.parent]
        first_close = parent.status == "open"
        if first_close:
            parent.status = "closed"
            parent.closed_via = rid
            ev.cascade.append(f"{parent.mid} CLOSED")

        # A route is registered whenever a reaction DIRECTLY UNDER THE ROOT closes,
        # first time or not.  Registering only on the first close would silently
        # drop every route after the first: the second reaction under the root
        # solves, nothing is recorded, nothing appears on screen, and the episode
        # has no way to express that a second answer exists.  Stopping is the agent's decision to make, and it cannot make it
        # against a board that hides the alternative it just built.
        if parent.mid == self.root:
            # During a claim the choice set is the one the model named.  Reading it
            # off `closed_via` instead registers a DIFFERENT route -- whichever
            # reaction happened to close each molecule first -- and then the claim's
            # own registration looks like a duplicate.  Same for the sibling path
            # below: BOTH have to stand down while a claim is committing.
            if not self._claiming and self.auto_routes:
                self._register(self._choices_from(rid), rid)
        elif first_close and parent.parent_rxn:
            self._settle(parent.parent_rxn, ev)
        elif not first_close:
            # An alternative under a molecule that is already closed.  Every
            # registered route that uses this molecule gains a sibling: the same
            # route with this reaction swapped in, and the new subtree below it.
            # One new route per new reaction -- linear, not the combinatorial
            # enumeration of every choice function.
            ev.cascade.append(f"{parent.mid} has an alternative via {rid}")
            if self._claiming or not self.auto_routes:
                return
            for route in list(self.routes):
                if parent.mid not in route.choices:
                    continue
                merged = {m: r for m, r in route.choices.items()
                          if not self._below(route, parent.mid, m)}
                merged.update(self._choices_from(rid))
                self._register(merged, route.root_rxn)

    def _choices_from(self, rid: str) -> dict:
        """The reaction choices of the subtree hanging off `rid`, taking each
        molecule's current `closed_via` as its default below the new reaction."""
        rxn = self.rxns[rid]
        out = {rxn.parent: rid}
        stack = list(rxn.pieces)
        while stack:
            mid = stack.pop()
            m = self.mols[mid]
            via = m.closed_via
            if via and via != "stock" and mid not in out:
                out[mid] = via
                stack.extend(self.rxns[via].pieces)
        return out

    def _below(self, route: "Route", top: str, mid: str) -> bool:
        """Is `mid` inside the subtree `route` hangs under `top` (top excluded)?"""
        if mid == top:
            return True
        seen, stack = set(), [top]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            rid = route.choices.get(cur)
            if not rid:
                continue
            for p in self.rxns[rid].pieces:
                if p == mid:
                    return True
                stack.append(p)
        return False

    def _register(self, choices: dict, root_rxn: str) -> Optional["Route"]:
        """Add a route unless an identical one is already on the board."""
        if self.root not in choices:
            return None
        if self.max_routes and len(self.routes) >= self.max_routes:
            self.routes_suppressed += 1
            return None
        candidate = Route(label="?", root_rxn=choices[self.root],
                          turn=self.turn, choices=dict(choices))
        fresh = candidate.steps(self)
        for existing in self.routes:
            if existing.steps(self) == fresh:
                return None
        candidate.label = chr(ord("A") + len(self.routes))
        self.routes.append(candidate)
        if not self._claiming:
            self.events.append(Event(kind="solved", route=candidate.label))
        return candidate

    def _fail(self, rid: str, ev: Event) -> None:
        """A piece died, so the reaction dies and hands its parent back."""
        rxn = self.rxns[rid]
        if rxn.status == "failed":
            return
        rxn.status = "failed"
        ev.cascade.append(f"{rid} FAILED")
        # A route that used this reaction is no longer a solution.
        dropped = [r.label for r in self.routes if rid in r.choices.values()]
        if dropped:
            self.routes = [r for r in self.routes
                           if rid not in r.choices.values()]
            ev.cascade.append(f"route {', '.join(dropped)} withdrawn")
        parent = self.mols[rxn.parent]
        parent.failed.append(rxn.cand)
        if rxn.cand in parent.applied:
            parent.applied.remove(rxn.cand)
        if parent.status == "closed" and parent.closed_via == rid:
            parent.status = "open"                    # it was only closed by this
            parent.closed_via = None
        if parent.status == "open":
            ev.reopened = parent.mid
            if parent.menu is not None and not parent.untried:
                self.events.append(Event(
                    kind="exhausted", mid=parent.mid,
                    ranking=list(parent.ranking), reopened=parent.mid,
                ))

    def _reaches(self, start: str, target: str) -> bool:
        """Is `target` anywhere in the subtree under `start`?

        Walks the reactions already committed, so it answers "is start above
        target" for the graph as it stands.  Iterative and visited-guarded: the
        graph is a DAG only because this check keeps it one.
        """
        seen: set[str] = set()
        stack = [start]
        while stack:
            mid = stack.pop()
            if mid in seen:
                continue
            seen.add(mid)
            for rid in self.mols[mid].rxns:
                for piece in self.rxns[rid].pieces:
                    if piece == target:
                        return True
                    stack.append(piece)
        return False

    # -- helpers -----------------------------------------------------------
    def _live(self, mid: str) -> Molecule:
        if mid not in self.mols:
            raise BoardError(f"no molecule {mid}")
        m = self.mols[mid]
        if m.status == "dead":
            raise BoardError(f"{mid} is dead")
        if m.status == "closed":
            if m.buyable:
                raise BoardError(f"{mid} is purchasable")
            # A solved molecule may be re-ranked: that is how a second route
            # through it is looked for.
        return m


class BoardError(Exception):
    pass


class NoMenu(BoardError):
    """Offline only: the draw cache never recorded this molecule.

    At inference the environment would just call the single-step model, so this
    is a hole in the recorded data and NOT a dead end.  Labelling it `dead`
    would teach the model that a molecule with nothing on screen is exhausted,
    which is the one thing an empty menu does not mean.
    """
