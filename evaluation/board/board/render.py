#!/usr/bin/env python
"""Board -> the text the model reads.  Every formatting decision lives here.

One function per screen part, and each one is callable on its own so a format
change can be diffed in isolation.  Nothing in this file touches state; nothing
outside it decides what the transcript looks like.

The decisions, in the order they appear on screen -- this is the list to argue
with, and each has a knob in RenderStyle:

  1  budget line      "budget 4 of 300"                     off by default: see
                                                           RenderStyle.show_budget
  2  event block      "[Event] e3 ranked [c1] · c1 applied → r1 · pieces:"
                      pieces indented, each with the status it ARRIVES in,
                      then one closing line: the cascade, or "r1 1 of 2 closed"
  3  routes block     shown on the turn a route closes; chained left-to-right
                      so a linear stretch stays on one line
  4  ledger row       "r1  e3·c1  1 of 2 closed        e3 ranked: c1   (was —)"
                      the ONLY memory the board keeps of a declaration
  5  open block       "  n4   under r1 · depth 1"  + its menu, or
                      "· no candidates yet" + the SMILES on first appearance
  6  candidate line   "<c0 q.374>A*(ln$5.92) + B</c0>"      signals, then pieces
  7  menu tail        "... 5 more, best q .004"             the window is not a lie
  8  closed line      "CLOSED  k2* · d7*(ln$5.66)"          running bill of materials
  9  dead line        "DEAD    9x · best q .047"
 10  molecule tag     "<mol e3>SMILES</mol>" once, "<mol e3/>" forever after

Two conventions carry weight beyond looks:

  *  A score is written without its leading zero (q.374, p.951) so that every
     bare number in the transcript is a score and every integer is a count.
  *  ln$ appears on purchasable fragments and nowhere else.  A price on a
     molecule still to be made would be a claim about a route that does not
     exist yet, and the model is told to treat it that way.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

from .state import Board, Candidate, Event, Molecule

# --------------------------------------------------------------------- style
@dataclass
class RenderStyle:
    # -- separators and marks
    sep: str = " · "                    # between peers everywhere on the board
    buyable_mark: str = "*"             # on any purchasable fragment
    dash: str = "—"                     # "no value": (was —), unpriced leaf
    rt_unrecovered: str = "✗"           # the forward model ran and did not recover it
    rt_unknown: str = "?"               # no cache entry -- nothing was asked

    # -- signals: which ones, in what order, and how each is written
    signal_order: tuple[str, ...] = ("q", "p", "rt")
    route_signal: Optional[str] = None
    """The axis a ROUTE is scored on -- its weakest step.  Must be the axis the
    DP ranks on, which is NOT necessarily the first one displayed: the display
    order puts q first because the candidate numbers are q's order.  None falls
    back to the first available signal, which is only right for a q-only run."""
    decimals: int = 3                   # q.374 ; set 2 for q.37
    drop_leading_zero: bool = True
    rt_as_int: bool = True              # rt1, not rt1.000
    price_fmt: str = "({})"
    """The candidate/molecule price, in the same unit the ROUTE line already uses.

    Both go through `dollars()`, so the board carries one price scale: a per-piece price and
    the candidate or route total it contributes to are directly comparable, which is what
    price needs to be an axis the model optimises rather than a mark that a branch closed."""

    # -- the menu window
    menu_show: int = 5                  # candidate lines printed in full
    menu_cutoff: Optional[float] = None  # e.g. 0.05 -> "all below p .05" tails
    cutoff_signal: str = "q"            # which signal the tail summarises

    # -- layout
    indent_open: str = "  "             # before a molecule id in OPEN
    indent_cand: str = "    "           # before a candidate line
    indent_event: str = "          "    # before a piece inside an event
    indent_event_tail: str = "        "  # before an event's closing line
    id_col: int = 5                     # id column width in OPEN
    ledger_gap: int = 8                 # spaces before the "ranked:" column
    routes_layout: str = "chain"        # chain | indent
    routes_when: str = "on_solve"       # on_solve | always
    routes_detail: str = "summary"
    """How the OBSERVATION renders the routes found so far.

    "full" draws every route's tree every time the block appears.  On a
    multi-route episode that makes the ROUTES blocks dominate the observation
    stream -- cost that grows with route count on every later turn, so the
    episode is quadratic in the number of answers found.

    "summary" gives one line per route: its length, its weakest step, its leaf
    count, its cost, and which reactions it shares with the first.  Nothing is
    lost that the board did not already hold -- the ledger carries every reaction
    and each piece arrived with its SMILES in an event -- so a tree is
    reconstructible from history, which is the retrieval this format is built
    around.  The FULL tree still appears exactly once, in the answer, for the
    route actually handed over."""

    # -- what to repeat
    show_was_on_first: bool = True      # "(was —)" on a first declaration
    show_ranked_column: bool = True     # the ledger's "e3 ranked: ..." column
    price_in_closed: bool = True
    space_in_weakest: bool = True       # "weakest q .290" vs "weakest q.290"
    show_candidate_cost: bool = True
    """A candidate-level dollar total in the signal group.

    On, and not gated on purchasability. It is the aggregate over the candidate's pieces,
    printed whenever every piece carries a price -- MolPrice prices non-buyable molecules
    too. Gating it on buyability would make the number appear exactly when the branch
    closes, so the model could only learn `$` as a solve marker; ordering candidates by cost
    needs the cost of the ones that do NOT close it. A partial sum is still refused: one unpriced
    piece would make the total understate by an unknown amount while looking complete."""

    show_budget: bool = False
    """The `budget N of M` header line.

    Off. The call ceiling is a CONSTRAINT, and the search is meant to be scaled by raising N
    until the model reasons its way to a route rather than by fitting inside a number it was
    shown. A board that prints the ceiling teaches the model to spend against it: an agent that
    reads "budget 287 of 300" reasons about how many calls are left instead of about the
    chemistry, and the reasoning corpus inherits that -- which is why the developer message no
    longer documents the line either. The budget still EXISTS and still stops a runaway episode;
    it is simply not a fact the agent reasons from.

    Set True to put it back for a run that is deliberately about the ceiling."""

    # -- headings
    h_open: str = "OPEN"
    h_closed: str = "CLOSED"
    h_dead: str = "DEAD"
    h_routes: str = "ROUTES"
    h_evidence: str = "EVIDENCE"
    handover: str = "flow"
    """How the FINAL message renders the answer.  The default is the form the
    training corpus uses, so a run that forgets the flag serves what the checkpoint
    was trained on.

    tree:  one route drawn chained, the rest as one-line differences.
    steps: every step of every claimed route, one flat self-contained line, then
           the routes as the reactions they choose.  See steps_block for why flat
           beats chained for something the model has to generate.
    """


STYLE = RenderStyle()


# ------------------------------------------------------------------ numbers
def num(v: float, st: RenderStyle = STYLE, decimals: int = None) -> str:
    """.374 -- a score, never with its leading zero."""
    s = f"{v:.{decimals if decimals is not None else st.decimals}f}"
    if st.drop_leading_zero and s.startswith("0."):
        s = s[1:]
    return s


PRICE_SATURATE = 15.0
"""ln(USD/mmol) past which the price stops being reported as a number.

MORetro* normalises the same MolPrice output as `min(1, price / 15)` in its price
heuristic, i.e. it saturates the axis at this point rather than letting a molecule nobody
would ever buy set the scale for the search. The same ceiling is applied here so an outlier
cannot drive an ordering decision: prices are heavy-tailed, and a few routes can carry most
of the total cost."""

# The 0-1 cost's window on ln(USD/mmol), for BOARD_COST_NORM. Dividing ln$ by PRICE_SATURATE
# puts nearly every candidate in the middle of the scale, so cost would be the one axis that
# never reaches either bound while `q` and `p` use most of theirs. Anchoring the window on the
# range MolPrice predictions actually occupy spreads the values within a menu. Ties that
# remain are real: candidates sharing one dear piece cost the same and should print the same.
# The bounds are round numbers bracketing that range ($12 to $99k per mmol), not fitted ones.
COST_LO, COST_SPAN = 2.5, 9.0
PLAUS_CUTOFF = 0.05                 # the reaction filter's, as the developer block states it


def COST_NORM() -> bool:
    """Whether the board is printing the normalised 0-1 cost rather than dollar strings.

    Read here and nowhere else. Four places had grown their own dollar formatting -- the ROUTES
    total, the leaves summary, the handover table, and factcheck's set of board-printed values
    -- and each has to follow `dollars()` onto the 0-1 scale. Otherwise one board carries a
    dollar total beside a normalised candidate cost, and factcheck's allowed set holds a string
    the board does not show, so a correct citation reads as invented: the number on the screen
    and the number the checker expects have to be produced by the same function."""
    return os.environ.get("BOARD_COST_NORM") == "1"


def cost_of_usd(usd: float) -> str:
    """A cost already in dollars -> the string the board would print for it."""
    return dollars(math.log(usd)) if usd and usd > 0 else "$0"


def dollars(v: float) -> str:
    """ln(USD/mmol) -> a cost on the SAME 0-1 SCALE as the other axes, `c.187`.

    BOARD_COST_NORM=1 selects this: the `$` signal, the log-sum-exp of the precursor
    prices normalised to [0, 1], lower is cheaper. The default is the dollar string below.

    WHY. On a candidate line the other axes are small bounded numbers and a dollar figure is
    not: `q.282 p.979 rt1 $2.8k`. Price then becomes the most salient figure on every line
    even where it does not separate the candidates, and a teacher trace starts ordering by
    price where the search did not. On the 0-1 scale the four signals read alike.

    The window is COST_LO / COST_SPAN on ln(USD/mmol), not MORetro*'s `min(1, price / 15)`:
    over dollars that saturates nearly every MolPrice prediction to a flat 1.0 and throws away
    the ordering, and dividing ln$ by 15 keeps the ordering but crowds the values into the
    middle of the scale. The log keeps the ordering; the window keeps the spread.

    Additivity is not needed: the board prints the candidate total itself, so nothing
    downstream has to add anything."""
    if COST_NORM():
        # `$` kept, value normalised. A letter prefix such as `c.529` reads as an identifier
        # rather than a number, and `c3`/`c4` (candidate numbers) sit in the same character
        # space. `$` says price and cannot collide with a candidate label.
        c = min(1.0, max(0.0, (v - COST_LO) / COST_SPAN))
        return f"${c:.3f}".replace("$0.", "$.")
    if v >= PRICE_SATURATE:
        return "$3.3M+"
    d = math.exp(v)
    if d >= 1e6:
        return f"${d/1e6:.1f}M"
    if d >= 1000:
        return f"${d/1000:.0f}k" if d >= 10000 else f"${d/1000:.1f}k"
    if d >= 10:
        return f"${d:.0f}"
    if d >= 1:
        return f"${d:.1f}"
    return f"${d:.2f}"


def price(v: Optional[float], st: RenderStyle = STYLE) -> str:
    return "" if v is None else st.price_fmt.format(dollars(v))


# ---------------------------------------------------------------- score guard
class UnscoredBoard(Exception):
    """A candidate reached the board carrying a signal no cache ever scored.

    An unscored candidate is not a low one. `rt?` means nothing was asked; a missing
    `$` means one fragment has no price and the total is refused; a missing `p` means
    the filter never ran. Rendered into a corpus these read as facts about chemistry
    when they are facts about the cache, and the model learns the cache's gaps.

    A search run after the caches were last filled produces exactly such candidates,
    and nothing downstream notices: the board renders them without complaint.

    So the corpus path asks for the check. `require_scores(True)` -- or
    BOARD_REQUIRE_SCORES=1 -- makes every render that would print an unscored signal
    collect it and raise at the end of the board, with the missing keys attached and
    written to BOARD_UNSCORED_OUT if that is set. The payload is what the fill scripts
    read, so a raise is the start of the labelling job rather than a dead end.

    It is OFF by default. A live eval legitimately runs cache-only and the agent has
    to cope with `rt?` on a board it has never seen; raising there would turn a normal
    condition into a crash.
    """

    def __init__(self, missing: dict):
        self.missing = missing
        n = {k: len(v) for k, v in missing.items() if v}
        parts = ", ".join(f"{v} {k}" for k, v in n.items())
        super().__init__(
            f"board rendered unscored candidates: {parts}. "
            f"Score them into analysis/cache/node_scores and re-render; "
            f"set BOARD_UNSCORED_OUT to dump the keys.")


_REQUIRE = [os.environ.get("BOARD_REQUIRE_SCORES", "") not in ("", "0")]
_MISSING: dict[str, set] = {"rt": set(), "price": set(), "plaus": set()}


def require_scores(on: bool = True) -> None:
    """Turn the guard on for this process (the corpus path does; eval does not)."""
    _REQUIRE[0] = bool(on)


def _note_missing(axis: str, key) -> None:
    if _REQUIRE[0] and key:
        _MISSING[axis].add(key)


def unscored_seen() -> dict:
    return {k: set(v) for k, v in _MISSING.items()}


def raise_if_unscored(clear: bool = True) -> None:
    """Raise UnscoredBoard if anything rendered without a score since the last call."""
    if not _REQUIRE[0] or not any(_MISSING.values()):
        if clear:
            for v in _MISSING.values():
                v.clear()
        return
    missing = {k: sorted(v) for k, v in _MISSING.items()}
    out = os.environ.get("BOARD_UNSCORED_OUT")
    if out:
        import json as _json
        with open(out, "a") as fh:
            fh.write(_json.dumps(missing) + "\n")
    if clear:
        for v in _MISSING.values():
            v.clear()
    raise UnscoredBoard(missing)


def _rxn_key(product: str, c: Candidate) -> str:
    """The caches' own reaction key: product, then precursors as a sorted set.

    Mirrors evidence.rxn_key -- the strings are used as the board
    holds them, never re-canonicalised, because the scoring envs do not share an
    rdkit and re-canonicalising per env writes keys the other envs cannot look up.
    """
    return f"{product}>>{'.'.join(sorted(set(c.reactants)))}"


def signals(c: Candidate, st: RenderStyle = STYLE) -> str:
    """q.374 -- or "q.31 p.951 rt1" when the run scored more than one axis.

    Only signals actually present are shown.  A menu where every candidate is
    missing p renders without a p column at all rather than with a hole in it.
    """
    out = []
    for k in st.signal_order:
        if k not in c.signals or k == "rt_keyed":
            continue
        v = c.signals[k]
        if k == "rt":
            out.append(_rt(v, c.signals.get("rt_keyed"), st))
        else:
            out.append(f"{k}{num(v, st)}")
    return " ".join(out)


def _rt(v, keyed, st) -> str:
    """rt as one of THREE things, because it IS one of three things.

    `rt1` the forward model recovered the product at rank 1. `rt✗` it ran and did not.
    `rt?` there is no cache entry, so nothing was asked and the board knows nothing. The
    last two must not print alike: a candidate with no entry is a different population from
    one the forward model failed on, and a trace reading an absence as "the model could not
    recover it" would be citing evidence that does not exist.
    """
    if v is not None:
        return f"rt{int(v)}" if st.rt_as_int else f"rt{v}"
    return f"rt{st.rt_unknown}" if keyed is False else f"rt{st.rt_unrecovered}"


# ---------------------------------------------------------------- molecules
def mol_tag(b: Board, mid: str, st: RenderStyle = STYLE, mark_price: bool = False) -> str:
    """<mol e3>SMILES</mol> the first time, <mol e3/> every time after.

    The board owns `shown`, so the tag is a side-effecting call by design: a
    SMILES appears exactly once in a transcript and the model is told so.
    """
    m = b.mols[mid]
    first = b.shown.get(mid)
    if first is not None and (first < b.turn or mid in b.printed):
        return f"<mol {mid}/>"
    b.shown.setdefault(mid, b.turn)
    b.printed.add(mid)
    tail = st.buyable_mark if (mark_price and m.buyable) else ""
    return f"<mol {mid}>{m.smiles}{tail}</mol>"


def mol_ref(b: Board, mid: str, st: RenderStyle = STYLE, with_price: bool = False) -> str:
    """A bare reference: k2, k2*($25) when buyable, k2($60) when only priced.

    The star and the dollar are independent: `*` says a catalogue sells it, `$` says
    what it is predicted to cost. Printing the price only for buyable pieces would make
    the two the same signal."""
    m = b.mols[mid]
    s = mid + (st.buyable_mark if m.buyable else "")
    if with_price and m.price_ln is not None:
        s += price(m.price_ln, st)
    return s


def piece_status(b: Board, mid: str, st: RenderStyle = STYLE) -> str:
    """The status a piece ARRIVES in -- the only place it is spelled out."""
    m = b.mols[mid]
    if m.buyable:
        p = price(m.price_ln, st)
        return (f"buyable{st.sep}{dollars(m.price_ln)}" if m.price_ln is not None
                else "buyable")
    if m.status == "closed":
        return f"already CLOSED{st.sep}via {m.closed_via}"
    if m.status == "dead":
        return "DEAD"
    busy = [x for x in m.rxns if b.rxns[x].status == "open"]
    if busy:
        # Open, but not a piece the model can act on: it already carries a
        # reaction of its own, so it is not in the OPEN block either.
        return f"already under {busy[0]}"
    return f"open{st.sep}depth {m.depth}"


# --------------------------------------------------------------- 1 · budget
def header(b: Board, st: RenderStyle = STYLE) -> str:
    """The budget line, or nothing. See RenderStyle.show_budget for why nothing by default."""
    return f"budget {b.budget_used} of {b.budget_max}" if st.show_budget else ""


# ------------------------------------------------------------ 6/7 · the menu
def candidate_line(b: Board, c: Candidate, st: RenderStyle = STYLE,
                   product: str = "") -> str:
    """<c0 q.374 $85>A*($25) + B($60)</c0>

    Pieces are written as SMILES, not ids: a candidate is a proposal, and its
    precursors do not exist as board molecules until the candidate is applied.
    Each piece carries its own mark and its own price, and the two are separate
    facts: `*` says a catalogue sells it, `($X)` says what MolPrice predicts it
    costs. Pricing only the purchasable pieces would make the dollar sign a
    second way of writing the star -- and then price could not be an axis,
    because it would never vary except where the branch already closed.
    """
    parts = []
    for smi in c.reactants:
        buyable, pr = b.world.info(smi)
        parts.append(f"{smi}{st.buyable_mark if buyable else ''}{price(pr, st)}")
    sig = signals(c, st)
    if _REQUIRE[0] and product:
        key = _rxn_key(product, c)
        if c.signals.get("rt_keyed") is False:
            _note_missing("rt", key)
        if "p" in st.signal_order and "p" not in c.signals:
            _note_missing("plaus", key)
    if st.show_candidate_cost:
        info = [b.world.info(s) for s in c.reactants]
        # Every piece must be PRICED -- it does not have to be BUYABLE. Printing a cost
        # only when the candidate closes the molecule outright would make `$` read as
        # "done" instead of "costs this much"; a candidate whose
        # pieces still need expanding has a cost too, and that is the number a cost-to-go
        # decision is made on. A partial sum is still refused: one unpriced piece and the
        # total would understate by an unknown amount while looking complete.
        prs = [pr for _, pr in info]
        for smi, pr in zip(c.reactants, prs):
            if pr is None:
                _note_missing("price", smi)
        if prs and all(pr is not None for pr in prs):
            sig = (sig + " " if sig else "") + dollars(math.log(sum(math.exp(x) for x in prs)))
    head = f"<c{c.idx} {sig}>" if sig else f"<c{c.idx}>"
    return f"{head}{' + '.join(parts)}</c{c.idx}>"


def menu_tail(hidden: list[Candidate], st: RenderStyle = STYLE) -> Optional[str]:
    """... 5 more, best q .004 -- so the window is never a lie by omission."""
    if not hidden:
        return None
    key = st.cutoff_signal
    vals = [c.signals[key] for c in hidden if key in c.signals]
    n = len(hidden)
    if not vals:
        return f"... {n} more"
    best = max(vals)
    if st.menu_cutoff is not None and best < st.menu_cutoff:
        return f"... {n} more, all below {key} {num(st.menu_cutoff, st, 2)}"
    return f"... {n} more, best {key} {num(best, st)}"


def menu_block(b: Board, mid: str, st: RenderStyle = STYLE) -> list[str]:
    m = b.mols[mid]
    shown = (m.menu or [])[: st.menu_show]
    hidden = (m.menu or [])[st.menu_show :]
    prod = getattr(m, "smiles", "") or ""
    lines = [st.indent_cand + candidate_line(b, c, st, prod) for c in shown]
    tail = menu_tail(hidden, st)
    if tail:
        lines.append(st.indent_cand + tail)
    return lines


# ------------------------------------------------------------- 5 · the frontier
def open_line(b: Board, mid: str, st: RenderStyle = STYLE) -> str:
    """  n4   under r1 · depth 1 [· no candidates yet | · ranked c0 ✗ ...]"""
    m = b.mols[mid]
    bits = []
    if m.parent_rxn:
        bits.append(f"under {m.parent_rxn}")
    # Depth alone, with no ceiling beside it. `K left` was max_depth - depth and `depth D/MAX`
    # put the same ceiling on screen in the other direction; both make the search's depth cap
    # part of what the model reads, and the cap is a property of how this corpus was built
    # rather than of the task -- evaluation bounds an episode by turns, not by levels. A model
    # taught to reason from "ten levels remain" has learnt an artefact, and teacher traces do
    # cite the ceiling when it is shown, so the number has to leave the board before the
    # reasoning is written and not after.
    bits.append(f"depth {m.depth}")
    if m.menu is None:
        bits.append("no candidates yet")
    elif not m.menu:
        bits.append("no candidates returned")
    elif m.failed:
        bits.append("ranked " + "  ".join(f"c{i} ✗" for i in m.ranking if i in m.failed))
    return st.indent_open + mid.ljust(st.id_col) + st.sep.join(bits)


def open_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    """Every piece still owed a decision, listed every turn.

    A molecule with no menu carries its SMILES on the next line the first time
    it is listed; after that the id is enough.
    """
    ids = b.open_mols()
    if not ids:
        return []
    out = [st.h_open]
    for mid in ids:
        out.append(open_line(b, mid, st))
        m = b.mols[mid]
        if b.shown.get(mid) is None or (b.shown[mid] == b.turn and mid not in b.printed):
            out.append(" " * (len(st.indent_open) + st.id_col) + mol_tag(b, mid, st))
        if m.menu is not None:
            out += menu_block(b, mid, st)
    return out


# --------------------------------------------------------------- 4 · the ledger
def ledger_row(b: Board, rid: str, st: RenderStyle = STYLE) -> str:
    """r1  e3·c1  1 of 2 closed        e3 ranked: c1   (was —)

    The board never re-lists a menu.  This row is all it remembers: which
    candidate was spent, how much of the AND is closed, and the ordering the
    model declared.  Recovering what a candidate was WORTH means going back to
    the turn it arrived in -- which is the retrieval the format is testing.
    """
    r = b.rxns[rid]
    m = b.mols[r.parent]
    left = f"{rid}  {r.parent}·c{r.cand}  "
    if r.status == "failed":
        state = "FAILED"
    else:
        state = f"{r.n_closed(b)} of {len(r.pieces)} closed"
        if r.status == "solved":
            state += f"{st.sep}solved"
    row = left + state
    if st.show_ranked_column and m.ranking:
        ranked = st.sep.join(f"c{i}" for i in m.ranking)
        row += " " * st.ledger_gap + f"{m.mid} ranked: {ranked}"
        if m.ranking_turn == b.turn:
            was = (st.sep.join(f"c{i}" for i in m.prev_ranking)
                   if m.prev_ranking else st.dash)
            if m.prev_ranking or st.show_was_on_first:
                row += f"   (was {was})"
    return row


def ledger_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    """Every reaction ever created, in creation order, failures included.

    Failed rows stay on the board on purpose: they are the record of what has
    already been disproved, and `done` is a claim about exactly that history.
    """
    return [ledger_row(b, rid, st) for rid in b.rxn_order]


# --------------------------------------------------------- 8/9 · closed, dead
def closed_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    ids = b.closed_leaves()
    if not ids:
        return []
    body = st.sep.join(mol_ref(b, m, st, with_price=st.price_in_closed) for m in ids)
    return [f"{st.h_closed}  {body}"]


def dead_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    ids = b.dead_mols()
    if not ids:
        return []
    out = []
    for mid in ids:
        m = b.mols[mid]
        best = None
        if m.menu:
            vals = [c.signals.get(st.cutoff_signal) for c in m.menu]
            vals = [v for v in vals if v is not None]
            best = max(vals) if vals else None
        tag = mid if best is None else f"{mid}{st.sep}best {st.cutoff_signal} {num(best, st)}"
        out.append(tag)
    return [f"{st.h_dead}    {st.sep.join(out)}"]


# ------------------------------------------------------ 1b · the evidence block
def evidence_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    """Facts bought by `analyze`, per candidate, for every molecule that asked.

    Drawn as its own block rather than folded into the candidate lines for two reasons:
    evidence can be bought long after a molecule opens, so a candidate line cannot carry it;
    and only some molecules are analysed, so inlining would make the menu format conditional
    on an unrelated action.

    Drawn ONLY on the board that answers the `analyze` call that bought it. It is a tool
    response, not board state: the facts reach the agent once, in the reply to the call, and
    stay available thereafter because they are in the conversation -- not because the board
    keeps re-printing them. `evidence_block` walks `b.order`, which is every molecule ever
    opened rather than the open frontier, so without the turn check a molecule that had been
    ranked and closed would keep publishing its evidence for the rest of the episode, and the
    block would dominate the board text. That is prompt the teacher and the student both pay
    for on every later turn to re-read facts they already have.

    A candidate whose mapping failed prints `bond not measured`. It must never print "no bond
    change" -- RXNMapper fails on some reactions, and a failure rendered as a
    measurement is a false fact the model has no way to distinguish from a true one.
    """
    if b.evidence is None:
        return []
    rows: list[str] = []
    for mid in b.order:
        m = b.mols[mid]
        if m.analyzed_turn is None or not m.menu:
            continue
        if m.analyzed_turn != b.turn:      # answered once, in the reply to its own call
            continue
        rows.append(f"  {mid}")
        pd = b.evidence.for_molecule(m.smiles)
        if pd:
            rows.append(f"    {mid} itself: {_desc_line(pd)}")
        # Only what was paid for. An unanalysed candidate has no facts on screen, and drawing
        # one anyway would let a trace cite evidence no action bought.
        for c in [x for x in m.menu if x.idx in (m.analyzed or [])]:
            fx = b.evidence.for_step(m.smiles, c.reactants)
            nm = fx.get("named") or {}
            bond = fx.get("bond")
            bits = []
            if nm.get("tier") == "applies+makes":
                bits.append(", ".join(nm.get("names") or [])
                            + (" (skeleton match)" if nm.get("match") == "skeleton" else ""))
            elif nm.get("tier") == "applies":
                bits.append("no named reaction reproduces this")
            elif nm.get("tier") == "none":
                bits.append("no named template applies")
            if bond:
                f = st.sep.join(_bond_atoms(x) for x in (bond.get("formed") or [])) or "-"
                k = st.sep.join(_bond_atoms(x) for x in (bond.get("broken") or [])) or "-"
                bits.append(f"formed {f}; broken {k}")
            else:
                bits.append("bond not measured")
            from .evidence import split_shape
            sp = split_shape(pd, [b.evidence.for_molecule(r) for r in c.reactants])
            if sp:
                bits.append(("skeleton kept" if sp["scaffold_kept"] else "skeleton changes")
                            + f"; rings {sp['rings_product']}<-{sp['rings_precursors']}"
                            + f"; fragments {sp['fragments']} ratio {sp['size_ratio']}")
            rows.append(f"    c{c.idx}  " + st.sep.join(bits))
    return [st.h_evidence] + rows if rows else []


def _bond_atoms(x: dict) -> str:
    """`C:13-N:14` when the atom labels are there, `C-N` when they are not.

    Entries written before the atom labels existed fall back to the element pair rather than
    claiming a position that was never measured.
    """
    a, b = x.get("at1"), x.get("at2")
    return f"{a}-{b}" if a and b else str(x.get("atoms", "?"))


def _desc_line(d: dict) -> str:
    return (f"rings {d.get('n_rings')} in {d.get('n_ring_systems')} system(s)"
            f"; heavy {d.get('n_heavy')}; Fsp3 {d.get('frac_csp3')}"
            f"; stereo {d.get('n_stereocentres')}"
            f"; scaffold {d.get('scaffold') or '(acyclic)'}")


# --------------------------------------------------------------- 2 · the events
def event_block(b: Board, ev: Event, st: RenderStyle = STYLE) -> list[str]:
    if ev.kind in ("rank", "apply"):
        return _event_rank(b, ev, st)
    if ev.kind == "claim":
        rxns = st.sep.join(ev.pieces)
        return [f"[Event] ROUTE CLAIMED{st.sep}{ev.route} = {rxns}"]
    if ev.kind == "dead":
        head = f"[Event] {ev.mid} declared dead"
        if ev.cascade:
            head += st.sep + st.sep.join(ev.cascade)
        if ev.reopened:
            head += f"{st.sep}<mol {ev.reopened}/> open again"
        return [head]
    if ev.kind == "exhausted":
        return [f"[Event] {ev.mid} out of ranked candidates{st.sep}"
                f"every candidate tried"]
    if ev.kind == "solved":
        return [f"[Event] ROUTE SOLVED{st.sep}route {ev.route}"]
    if ev.kind == "analyzed":
        # The event says WHICH molecules were bought, not what came back; the facts are
        # drawn in the EVIDENCE block, where they sit next to the candidates they belong to.
        return [f"[Event] analysed{st.sep}" + st.sep.join(ev.pieces)]
    raise ValueError(f"no renderer for event {ev.kind!r}")


def _event_rank(b: Board, ev: Event, st: RenderStyle = STYLE) -> list[str]:
    m = b.mols[ev.mid]
    if ev.kind == "apply":
        # A claim commits a candidate without declaring an ordering, so there is
        # no ranking to report and printing an empty one would describe a move
        # the model did not make.
        head = (f"[Event] {ev.mid} c{ev.applied} applied → {ev.rxn}"
                f"{st.sep}pieces:")
    else:
        verb = "re-ranked" if ev.prev_ranking else "ranked"
        order = ", ".join(f"c{i}" for i in ev.ranking)
        head = (f"[Event] {ev.mid} {verb} [{order}]{st.sep}"
                f"c{ev.applied} applied → {ev.rxn}{st.sep}pieces:")
    tags = [(mid, mol_tag(b, mid, st)) for mid in ev.pieces]
    width = max((len(t) for _, t in tags), default=0)
    lines = [head]
    for mid, tag in tags:
        lines.append(st.indent_event + tag.ljust(width) + "   " + piece_status(b, mid, st))
    r = b.rxns[ev.rxn]
    if ev.cascade:
        all_buyable = all(b.mols[p].buyable for p in r.pieces)
        lead = "all buyable → " if all_buyable else ""
        lines.append(st.indent_event_tail + lead + st.sep.join(ev.cascade))
    else:
        lines.append(st.indent_event_tail
                     + f"{ev.rxn} {r.n_closed(b)} of {len(r.pieces)} closed")
    return lines


def events_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    out = []
    for ev in b.events:
        out += event_block(b, ev, st)
    return out


# --------------------------------------------------------------- 3 · the routes
def signals_of_rxn(r, st: RenderStyle = STYLE) -> tuple[str, Optional[float]]:
    """The one signal a route is scored on, as (text, value).

    Whichever of signal_order the run actually has comes first, so a q-only run
    reads "q.290" and a run with plausibility reads "p.871" without a format
    change.  The value comes back too, because `weakest` is a min over it.
    """
    order = ((st.route_signal,) + st.signal_order if st.route_signal
             else st.signal_order)
    for k in order:
        if k in r.signals and r.signals[k] is not None and k != "rt":
            return f"{k}{num(r.signals[k], st)}", float(r.signals[k])
    return "", None


def signals_text(r, st: RenderStyle = STYLE) -> str:
    """Every objective the run carries for one reaction, e.g. `p.923 rt1`.

    `signals_of_rxn` deliberately returns ONE number and skips rt, because it
    feeds `weakest`, which is a min and would be wrong on a rank (for rt, lower is
    better).  That is the right rule for the aggregate and the wrong one for the
    display: the board shows q, p and rt on every candidate, so an answer that
    reports only p hides two of the three axes the route was chosen on.
    """
    bits = []
    for k in st.signal_order:
        v = r.signals.get(k)
        if k == "q":
            continue                      # model confidence, not an objective
        if k == "rt":
            bits.append(_rt(v, r.signals.get("rt_keyed"), st))
        elif v is not None:
            bits.append(f"{k}{num(v, st)}")
    return " ".join(bits)


def route_objectives(b: Board, route, st: RenderStyle = STYLE) -> str:
    """The three axes for a whole route, aggregated the way each one works.

    plausibility is a min (the route is as good as its weakest step), round-trip
    is a max (the worst rank any step got, and how many steps have no rank at
    all), and price is a sum over what has to be bought.
    """
    rxns = _route_rxns(b, route.root_rxn, route)
    ps = [(rid, b.rxns[rid].signals.get("p")) for rid in rxns]
    ps = [(rid, v) for rid, v in ps if v is not None]
    rts = [b.rxns[rid].signals.get("rt") for rid in rxns]
    known = [v for v in rts if v is not None]
    leaves = _route_leaves(b, route.root_rxn, route)
    priced = [b.mols[m].price_ln for m in leaves if b.mols[m].price_ln is not None]
    bits = [f"{len(rxns)} steps"]
    if ps:
        rid, v = min(ps, key=lambda t: t[1])
        if ROUTE_AXES():
            # ONE p FIGURE ON THE LINE, AND IT IS THE AXIS. The share of steps that clear the
            # filter, printed the way `rt` already is -- `p 5/6 pass` beside `rt 4/6 back`:
            # two axes aggregated the same way, each saying which way. The minimum came off.
            # A line carrying both makes the reader guess which one the front mark is on, and
            # the one that is NOT the axis gets promoted to being it, the same way a salient
            # cost figure gets cited where it decided nothing. The weakest step keeps its POINTER,
            # `worst at r12`, so "work on the step that IS the weakest" still has a referent;
            # it just no longer carries a rival number.
            npass = sum(1 for _, x in ps if x >= PLAUS_CUTOFF)
            bits.append(f"p {npass}/{len(ps)} pass"
                        + (f", worst at {rid}" if npass < len(ps) else ""))
        else:
            bits.append(f"weakest p{num(v, st)} at {rid}")
    if rts:
        # rt is a top-5 recovery test: a step with no rank is one the forward
        # model FAILED to return the product for, not one nobody measured.  So
        # the route-level number is a recovery fraction, which is also what the
        # route-quality metric uses, and the worst rank describes only what came back.
        bits.append(f"rt {len(known)}/{len(rts)} back"
                    + (f", worst {int(max(known))}" if known else ""))
    bits.append(f"{len(leaves)} leaves")
    # Through dollars(), not formatted here: this is the same axis the candidate lines carry,
    # and under BOARD_COST_NORM they are normalised while a literal f-string is not. One board,
    # one axis, one scale -- otherwise a trace cites a number that is on no screen.
    bits.append(dollars(math.log(sum(math.exp(v) for v in priced))) if priced else "$?")
    return st.sep.join(bits)


def ROUTE_AXES() -> bool:
    """Whether the ROUTES block prints the three axes -- `p N/M pass` in place of the weakest
    step, beside `rt N/M back` and the cost.

    SEPARATE FROM THE MARKS ON PURPOSE. The axes are what makes a front DERIVABLE from the
    screen; the marks are the derivation already done. An arm that wants the model to compute
    dominance itself needs the first without the second, and an arm that wants it to read a
    verdict needs both. What is not a coherent arm is marks without axes (a verdict over
    numbers the reader cannot see) or NEITHER while the reasoning still talks about beaten
    routes -- with only `weakest p.401 at r7` on the line, `J is beaten by C` is an assertion
    no reader can check, and the model learns to imitate the claim rather than to make it.
    """
    return os.environ.get("BOARD_ROUTE_AXES") == "1" or ROUTE_FRONT()


def ROUTE_FRONT() -> bool:
    """Whether the ROUTES block marks its own Pareto front (`A ▲`, `E ≺C`, `· front A C H`).

    Read here and nowhere else, the way COST_NORM is. Off keeps the unmarked board
    byte-for-byte, so a checkpoint trained on it is still served the screen it learned."""
    return os.environ.get("BOARD_ROUTE_FRONT") == "1"


def ROUTES_FULL() -> bool:
    """Whether the ROUTES block lists EVERY claimed route or only the ones this turn added.

    Only-this-turn is the default because redrawing every route on every claim was quadratic
    in the number of answers. That reasoning holds for the block in general and fails for the
    LAST board: the hand-over orders the whole claimed set, and with one row on screen it has
    to recover the others from turns further back in the context, and ends up arguing about
    routes that are not on its screen. The cap this trades against is small -- ten routes at one line each, on the few
    turns that claim anything -- so the flag prints them all rather than trying to detect
    which board is last, which `render_env` has no way to know.
    """
    return os.environ.get("BOARD_ROUTES_FULL") == "1"


def route_axes(b: Board, route) -> dict:
    """The three route axes AS NUMBERS, aggregated exactly as `route_objectives` prints them.

    Same function, two consumers: the line the model reads and the dominance test that marks
    it. Computing the front from anything else would let the screen and the mark disagree,
    which is the failure `COST_NORM` documents -- a number on the board that the checker does
    not hold. `p` is the route's weakest step, `rt` the fraction of steps the forward model
    recovers, `d` the normalised cost of the leaves. Every one of them is on the printed line.

    Values are rounded to what the board PRINTS (three decimals for p, the exact fraction for
    rt, three for the cost), so the front is reproducible from the screen by eye. A route
    missing an axis gets None there and is compared only on the axes it shares -- see
    `route_front`.
    """
    rxns = _route_rxns(b, route.root_rxn, route)
    ps = [b.rxns[rid].signals.get("p") for rid in rxns]
    ps = [v for v in ps if v is not None]
    rts = [b.rxns[rid].signals.get("rt") for rid in rxns]
    known = [v for v in rts if v is not None]
    leaves = _route_leaves(b, route.root_rxn, route)
    priced = [b.mols[m].price_ln for m in leaves if b.mols[m].price_ln is not None]
    return {
        "route": route.label,
        "steps": len(rxns),
        # THE SHARE THAT CLEARS THE FILTER, not the minimum. The evaluation's plausibility
        # objective is the share of steps with p >= 0.05, and a front computed on the weakest
        # step is a front on a different quantity -- the board would mark a route dominated
        # that the evaluation's test does not.
        # The minimum stays on the line and stays what a route is WORTH ("a route is worth its
        # weakest step"); it is simply not the axis the front is over. The two answer
        # different questions: the minimum says how bad the worst step is, the share says how
        # much of the route the filter accepts, and two routes can disagree on them.
        "p": round(sum(1 for v in ps if v >= PLAUS_CUTOFF) / len(ps), 4) if ps else None,
        "p_min": round(min(ps), 3) if ps else None,
        "p_pass": (sum(1 for v in ps if v >= PLAUS_CUTOFF), len(ps)) if ps else None,
        "rt": round(len(known) / len(rts), 4) if rts else None,
        "rt_worst": int(max(known)) if known else None,
        "leaves": len(leaves),
        "d": (round(min(1.0, max(0.0, (math.log(sum(math.exp(v) for v in priced)) - COST_LO)
                                 / COST_SPAN)), 3) if priced else None),
        "cost_ln": math.log(sum(math.exp(v) for v in priced)) if priced else None,
        "unpriced": len(leaves) - len(priced),
    }


# HIGHER IS BETTER once signed: the route is worth its weakest step, the forward model should
# recover as much of it as possible, and cost is a cost.
_FRONT_AXES = (("p", 1), ("rt", 1), ("d", -1))


def _dominates(a: dict, b: dict) -> bool:
    """Does `a` dominate `b`? `all(<=) and any(<)` on the printed figures.

    The standard Pareto test on the same rounded figures the evaluation uses, so a route the
    evaluation calls dominated is the one the board marks. Axes
    absent from either route are skipped rather than defaulted: a route whose leaves are not
    all priced has no cost, and inventing a worst-case one there would let an unpriced route
    be marked beaten on a number it does not carry.
    """
    shared = [(k, s) for k, s in _FRONT_AXES if a.get(k) is not None and b.get(k) is not None]
    if not shared:
        return False
    return (all((a[k] - b[k]) * s >= -1e-9 for k, s in shared)
            and any((a[k] - b[k]) * s > 1e-9 for k, s in shared))


def route_front(b: Board) -> tuple[list[str], dict[str, list[str]]]:
    """(labels on the front, {beaten label: the labels that beat it}).

    Why the board carries this at all: an episode typically ends with several non-dominated
    routes, and a claimed set can hold routes beaten outright by another route in the SAME
    handover. A board that prints every number the comparison needs but never makes it gives
    `best first` no basis on the screen, and the handover cannot be judged against one.
    """
    axes = [route_axes(b, r) for r in b.routes]
    beaten: dict[str, list[str]] = {}
    for x in axes:
        by = [o["route"] for o in axes if o is not x and _dominates(o, x)]
        if by:
            beaten[x["route"]] = by
    # A LONE ROUTE IS ON THE FRONT. Nothing beats it, so `front: false` beside it -- which is
    # what a caller that only computes this for len(routes) > 1 records -- reads as "beaten"
    # to every consumer downstream, and the hand-over brief would be told its only answer is
    # dominated. The board still prints no mark there, because a front of one is not a
    # comparison worth a column; the FACT is what has to be right.
    front = [x["route"] for x in axes if x["route"] not in beaten]
    # NAME A DOMINATOR THAT IS ITSELF ON THE FRONT where one exists. `E ≺B` beside `B ≺C`
    # points the reader at a route that is itself beaten, so following the marks takes two
    # hops to reach an answer worth having; dominance is transitive, so a front member always
    # exists to name and it is the one the handover will actually offer.
    for lbl, by in beaten.items():
        beaten[lbl] = sorted(by, key=lambda r: (r not in front, r))
    return front, beaten


def _rxn_of(b: Board, mid: str, route=None) -> Optional[str]:
    """The reaction THIS ROUTE uses at `mid`, or the one that closed it.

    A route carries its own choice per molecule, so the tree has to be walked
    with the route in hand: reading `closed_via` instead would draw every route
    the same below its first divergence.
    """
    if route is not None:
        return route.choices.get(mid)
    m = b.mols[mid]
    return m.closed_via if (m.closed_via and m.closed_via != "stock") else None


def _route_lines(b: Board, rid: str, st: RenderStyle, route=None) -> list[str]:
    """The route tree, chained.

        e3 ←r1(q.290)— k2*
                       n4 ←r2(q.530)— b0 ←r3(q.535)— d7* · a5*
                                      m1 ←r4(q.889)— f8 ←r5(q.622)— h2* · j6*

    A linear stretch stays on one line; a new line starts only where an AND
    actually branches, and it is aligned under the first piece of that AND so
    the column tells you which reaction the pieces belong to.  Lines come back
    relative to column 0 and the caller pads them.
    """
    r = b.rxns[rid]
    sig, _ = signals_of_rxn(r, st)
    head = f"{r.parent} ←{rid}({sig})— " if sig else f"{r.parent} ←{rid}— "

    items: list[list[str]] = []
    leaves: list[str] = []
    for p in r.pieces:
        sub = _rxn_of(b, p, route)
        if sub is None:
            leaves.append(mol_ref(b, p, st))
            continue
        if leaves:
            items.append([st.sep.join(leaves)])
            leaves = []
        items.append(_route_lines(b, sub, st, route))
    if leaves:
        items.append([st.sep.join(leaves)])

    pad = " " * len(head)
    out: list[str] = []
    for k, item in enumerate(items):
        for j, ln in enumerate(item):
            out.append((head if (k == 0 and j == 0) else pad) + ln)
    return out or [head.rstrip()]


def _route_rxns(b: Board, rid: str, route=None) -> list[str]:
    out = [rid]
    for p in b.rxns[rid].pieces:
        sub = _rxn_of(b, p, route)
        if sub:
            out += _route_rxns(b, sub, route)
    return out


def _route_leaves(b: Board, rid: str, route=None) -> list[str]:
    out = []
    for p in b.rxns[rid].pieces:
        sub = _rxn_of(b, p, route)
        out += _route_leaves(b, sub, route) if sub else [p]
    return out


def route_block(b: Board, route, st: RenderStyle = STYLE) -> list[str]:
    """One route: the tree, its weakest step, and its bill of materials."""
    out = [f"  {route.label}"]
    out += ["     " + ln for ln in _route_lines(b, route.root_rxn, st, route)]

    scored = [(rid,) + signals_of_rxn(b.rxns[rid], st)
              for rid in _route_rxns(b, route.root_rxn, route)]
    scored = [(rid, txt, val) for rid, txt, val in scored if val is not None]
    if scored:
        rid, txt, _ = min(scored, key=lambda t: t[2])
        sig_key = txt[: len(txt) - len(txt.lstrip("abcdefghijklmnopqrstuvwxyz"))]
        out.append(f"      weakest {sig_key} {txt[len(sig_key):]} at {rid}"
                   if st.space_in_weakest else f"      weakest {txt} at {rid}")

    leaves = _route_leaves(b, route.root_rxn, route)
    shown = st.sep.join(
        f"{m} {b.mols[m].price_ln:.2f}" if b.mols[m].price_ln is not None else f"{m} {st.dash}"
        for m in leaves
    )
    line = f"      leaves  {shown}"
    priced = [b.mols[m].price_ln for m in leaves if b.mols[m].price_ln is not None]
    unpriced = [m for m in leaves if b.mols[m].price_ln is None]
    if priced:
        # dollars(), and the unit only where the figure is one: `/mmol` is meaningless on a
        # normalised 0-1 cost, and printing it there states a unit the number does not have.
        _t = dollars(math.log(sum(math.exp(v) for v in priced)))
        line += f"   ≈ {_t}" + ("" if COST_NORM() else "/mmol")
    if unpriced:
        line += f"{' +' if priced else '   '} {','.join(unpriced)} unpriced"
    out.append(line)
    return out


def route_summary_line(b: Board, route, st: RenderStyle = STYLE,
                       reference=None, mark: str = "") -> str:
    """One route on one line: what it costs, where it is weakest, what it shares.

    The shared-reaction note is what makes the list readable as a set of
    alternatives rather than a list of unrelated answers: two routes that share
    every reaction but one are a different proposition from two that share none.
    """
    rxns = _route_rxns(b, route.root_rxn, route)
    leaves = _route_leaves(b, route.root_rxn, route)
    priced = [b.mols[m].price_ln for m in leaves if b.mols[m].price_ln is not None]
    unpriced = len(leaves) - len(priced)
    # The mark sits between the label and the axes, in its own fixed column, so the numbers
    # still line up down the block. `▲` is on the front, `≺C` is beaten and says BY WHICH --
    # a bare "dominated" would make the reader re-derive the comparison the board just made.
    line = (f"  {route.label:<3} " + (f"{mark:<5}" if ROUTE_FRONT() else "")
            + route_objectives(b, route, st))
    if unpriced:
        line += f" ({unpriced} unpriced)"
    if reference is not None and reference is not route:
        # Where the two routes DISAGREE is shorter to say than what they share,
        # and it locates the merge point: everything not named is common.
        diff = [(mid, rid) for mid, rid in sorted(route.choices.items())
                if reference.choices.get(mid) != rid]
        shared = set(rxns) & set(_route_rxns(b, reference.root_rxn, reference))
        if diff and shared:
            head = st.sep.join(f"{rid} at {mid}" for mid, rid in diff[:2])
            more = f" +{len(diff) - 2}" if len(diff) > 2 else ""
            line += f"   = {reference.label} with {head}{more}"
        elif not shared:
            line += f"   shares nothing with {reference.label}"
    return line


def routes_block(b: Board, st: RenderStyle = STYLE) -> list[str]:
    if not b.routes:
        return []
    if st.routes_detail == "summary":
        ref = b.routes[0]
        word = "found" if b.auto_routes else "claimed"
        head = f"{st.h_routes}  {len(b.routes)} {word}"
        marks: dict[str, str] = {}
        if ROUTE_FRONT() and len(b.routes) > 1:
            front, beaten = route_front(b)
            # Named on the header as well as marked on the rows. The handover has to order
            # the front and it is a set, not a property of one line; a reader that has to
            # collect it by scanning eight rows is doing the board's arithmetic.
            if front:
                head += f" · front {' '.join(front)}"
            marks = {lbl: "\u25b2" for lbl in front}
            marks.update({lbl: "\u227a" + by[0] for lbl, by in beaten.items()})
        out = [head]
        out += [route_summary_line(b, rt, st, reference=ref, mark=marks.get(rt.label, ""))
                for rt in b.routes]
        return out
    out = [st.h_routes]
    for route in b.routes:
        out += route_block(b, route, st)
    return out


def _nest_lines(b: Board, rid: str, st: RenderStyle, route, depth: int = 0) -> list[str]:
    """The route as a tree, indented two spaces per level.

        pv \u2190r8(p.078)
          gn \u2190r9(p.149)
            kv*
            tq*
          xu \u2190r10(p.673)

    Same shape as the chained drawing, but the indent is a FIXED two spaces per
    level instead of the print width of whatever came before it.  That is the
    whole difference and it is the one that matters for something the model has to
    generate: depth is a small integer it walks up and down, not the length of a
    reaction id plus the digits of a signal.
    """
    r = b.rxns[rid]
    sig = signals_text(r, st)
    pad = "  " * depth
    head = f"{pad}{mol_ref(b, r.parent, st)} \u2190{rid}({sig})" if sig else \
           f"{pad}{mol_ref(b, r.parent, st)} \u2190{rid}"
    out = [head]
    for piece in r.pieces:
        sub = _rxn_of(b, piece, route)
        if sub is None:
            out.append("  " * (depth + 1) + mol_ref(b, piece, st))
        else:
            out += _nest_lines(b, sub, st, route, depth + 1)
    return out


def _sexp(b: Board, rid: str, st: RenderStyle, route) -> str:
    """The route as one bracketed line: `pv\u2190r8(gn\u2190r9(kv* tq*) xu\u2190r10(...))`.

    Nesting is carried by brackets rather than by position, so the whole route is
    one line no matter how deep it goes and nothing has to line up.  Balanced
    delimiters are the one structural convention a language model has seen more of
    than any other.
    """
    r = b.rxns[rid]
    sig, _ = signals_of_rxn(r, st)
    inner = []
    for piece in r.pieces:
        sub = _rxn_of(b, piece, route)
        inner.append(mol_ref(b, piece, st) if sub is None else _sexp(b, sub, st, route))
    tag = f"{r.parent}\u2190{rid}({sig})" if sig else f"{r.parent}\u2190{rid}"
    return f"{tag}[{' '.join(inner)}]"


def _weak_p(b: Board, route) -> float:
    """The route's weakest plausibility, or -1 when no step carries one."""
    vals = [b.rxns[rid].signals.get("p")
            for rid in _route_rxns(b, route.root_rxn, route)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else -1.0


def _route_cost(b: Board, route) -> Optional[float]:
    """What the route's leaves cost, or None when any of them is unpriced."""
    leaves = _route_leaves(b, route.root_rxn, route)
    vals = [b.mols[m].price_ln for m in leaves]
    if any(v is None for v in vals):
        return None
    return sum(math.exp(v) for v in vals)


def _flow_lines(b: Board, route, st: RenderStyle = STYLE) -> list[str]:
    """The route the way a chemist writes it: forward, materials first.

        ks* + xf* \u2192r7 au
        au + qp* \u2192r6 u3
        u3 \u2192r5 zw
        zw + g6* \u2192r3 qm

    A retrosynthesis tree read in the direction it will be RUN.  Nothing is
    carried by position: the order is a topological one, each line names its own
    inputs and output, and a convergent step is just a line whose inputs were
    made by earlier lines.  For a model that has to generate the answer this is
    the easiest of the forms here -- a sequence, not a nesting -- and for a
    chemist it is the form the route would be written in anyway.
    """
    rids = _route_rxns(b, route.root_rxn, route)
    depth = {}

    def mark(rid, d=0):
        depth[rid] = max(depth.get(rid, 0), d)
        for piece in b.rxns[rid].pieces:
            sub = _rxn_of(b, piece, route)
            if sub:
                mark(sub, d + 1)

    mark(route.root_rxn)
    out = []
    for rid in sorted(rids, key=lambda r: -depth.get(r, 0)):
        r = b.rxns[rid]
        sig = signals_text(r, st)
        # WITH THE PRICE. The route header carries the total (`$137`), and without the
        # per-leaf figure there is no way to see which leaf that total is: a route whose
        # cost sits in one expensive fragment and a route that is uniformly cheap read
        # identically. `nest` already prints it on its leaves, so `flow` -- the default --
        # was the one form that dropped it. The reaction parenthetical stays p and rt only:
        # a price belongs to a purchasable molecule and a reaction does not have one.
        ins = " + ".join(mol_ref(b, pc, st, with_price=True) for pc in r.pieces)
        tag = f"\u2192{rid}({sig})" if sig else f"\u2192{rid}"
        out.append(f"  {ins} {tag} {r.parent}")
    return out


def steps_block(b: Board, routes, st: RenderStyle = STYLE) -> list[str]:
    """Every step the claimed routes use, one self-contained line each.

        pv \u2190r8(p.078)\u2014 gn \u00b7 xu
        gn \u2190r9(p.149)\u2014 kv* \u00b7 tq*

    Chosen over the chained drawing because the final message is SUPERVISED: the
    model has to produce it.  In the chained form the AND structure is carried by
    column alignment -- `xu` is a sibling of `gn` because both start at column 15
    -- and the column depends on the text width of the head, so it moves with the
    reaction id and the number of digits in the signal.  That asks the model to do
    positional arithmetic that carries no information, and to reproduce exact runs
    of spaces, which tokenise inconsistently.  Here each line names its own
    product, so nothing is implied by position and the lines can come in any
    order.

    Written once for the whole set, not once per route: the routes are choice sets
    over one graph and the shared steps are literally the same lines.
    """
    seen: dict[str, str] = {}
    for rt in routes:
        for mid, rid in rt.choices.items():
            if rid in seen:
                continue
            r = b.rxns[rid]
            sig, _ = signals_of_rxn(r, st)
            pieces = st.sep.join(mol_ref(b, pc, st, with_price=True) for pc in r.pieces)
            head = f"{mid} \u2190{rid}({sig})\u2014 " if sig else f"{mid} \u2190{rid}\u2014 "
            seen[rid] = f"  {head}{pieces}"
    return ["STEPS"] + [seen[rid] for rid in b.rxn_order if rid in seen]


def route_line(b: Board, route, st: RenderStyle = STYLE) -> str:
    """One route as the reactions it chooses, and all three objectives."""
    rxns = _route_rxns(b, route.root_rxn, route)
    return f"  {route.label} = {' '.join(rxns)}   " + route_objectives(b, route, st)


def graph_block(b: Board, routes, st: RenderStyle = STYLE) -> list[str]:
    """The claimed routes as ONE graph, with the merge points marked.

    Drawing each route as its own tree says the wrong thing about the object: the
    routes are choice sets over a shared DAG, two of them typically differ by a
    single reaction, and redrawing the shared part once per route both hides that
    and costs the tokens.  So a molecule is drawn where it is first reached and
    referred to as `vg^` afterwards, and a molecule reached by several claimed
    reactions lists them one under the other.
    """
    used: dict[str, list[str]] = {}
    for rt in routes:
        for mid, rid in rt.choices.items():
            used.setdefault(mid, [])
            if rid not in used[mid]:
                used[mid].append(rid)

    out: list[str] = []
    drawn: set[str] = set()

    def walk(mid: str, indent: int) -> None:
        pad = " " * indent
        m = b.mols[mid]
        if mid in drawn:
            out.append(f"{pad}{mid}^")
            return
        drawn.add(mid)
        rids = used.get(mid, [])
        if not rids:
            out.append(f"{pad}{mol_ref(b, mid, st, with_price=True)}")
            return
        for k, rid in enumerate(rids):
            sig = signals_of_rxn(b.rxns[rid], st)[0]
            head = f"{pad}{mid if k else mol_ref(b, mid, st)} \u2190{rid}({sig})\u2014"
            out.append(head)
            for piece in b.rxns[rid].pieces:
                walk(piece, indent + 4)

    walk(b.root, 2)
    return ["GRAPH"] + out


def handover_block(b: Board, chosen, st: RenderStyle = STYLE) -> list[str]:
    """The answer: every route claimed, best first, over one drawing of the graph.

    `chosen` is the best of them and leads the list; it is no longer the only one
    handed over, because claiming several routes is the point of the format and
    handing over one of them would throw the rest away.
    """
    others = [rt for rt in b.routes if rt is not chosen]
    n = len(others) + 1
    if st.handover == "table":
        # One row per route, so the objectives line up in columns and the routes
        # can be COMPARED rather than read one after another -- which is what a
        # multi-objective answer is for.  Markdown pipes need no padding to parse,
        # so nothing here depends on width.
        ranked = [chosen] + others
        out = [f"Handing over {n} route{'s' if n != 1 else ''}, best first.", "",
               "| route | steps | weakest p | rt back | leaves | $ | sequence |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for rt in ranked:
            rids = _route_rxns(b, rt.root_rxn, rt)
            ps = [(rid, b.rxns[rid].signals.get("p")) for rid in rids]
            ps = [(rid, v) for rid, v in ps if v is not None]
            rts = [b.rxns[rid].signals.get("rt") for rid in rids]
            known = [v for v in rts if v is not None]
            leaves = _route_leaves(b, rt.root_rxn, rt)
            priced = [b.mols[m].price_ln for m in leaves
                      if b.mols[m].price_ln is not None]
            weak = min(ps, key=lambda t: t[1]) if ps else None
            seq = "; ".join(
                ln.strip().replace(" \u2192", " \u2192") for ln in _flow_lines(b, rt, st))
            out.append(
                f"| {rt.label} | {len(rids)} | "
                f"{num(weak[1], st) + ' at ' + weak[0] if weak else st.dash} | "
                f"{len(known)}/{len(rts)} | {len(leaves)} | "
                f"{dollars(math.log(sum(math.exp(v) for v in priced)))} | {seq} |")
        return out
    if st.handover == "flow":
        # Claim order, NOT best-first.  What a claim guarantees is that the route
        # is complete -- every step a reaction the board applied, every leaf
        # purchasable -- and that is a fact about the search.  Which of them a
        # chemist should run is not: the axis that would do the sorting here is
        # the weakest step, and ranking by it can put the literature route LAST
        # because a min over steps is mostly a proxy for how many steps there are.  So the objectives go on every line and the
        # ordering claims nothing.
        out = [f"Handing over {n} route{'s' if n != 1 else ''}. Each is complete: "
               f"every step is a reaction on the board and every leaf is "
               f"purchasable."]
        # One recommendation per NAMED axis rather than a single ranking.  A total
        # order over routes is not something the data supports -- sorted by the
        # weakest step, the literature route comes last -- but "which of these is
        # best on plausibility" is a fact about the set, so it can be said, and
        # checked.  Ties go to the route claimed first.
        picks = []
        best = max(b.routes, key=lambda rt: _weak_p(b, rt), default=None)
        if best is not None and _weak_p(b, best) > -1:
            picks.append(f"best on plausibility: {best.label}")
        short = min(b.routes, key=lambda rt: len(_route_rxns(b, rt.root_rxn, rt)),
                    default=None)
        if short is not None:
            picks.append(f"fewest steps: {short.label}")
        cheap = min((rt for rt in b.routes if _route_cost(b, rt) is not None),
                    key=lambda rt: _route_cost(b, rt), default=None)
        if cheap is not None:
            picks.append(f"cheapest: {cheap.label}")
        if picks:
            # NOT .capitalize(): it lowercases the rest of the string, and the
            # rest of the string is route labels.
            head = picks[0][0].upper() + picks[0][1:]
            out.append(st.sep.join([head] + picks[1:]) + ".")
        out.append("")
        for rt in b.routes:
            out.append(f"  {rt.label}   {route_objectives(b, rt, st)}")
            out += _flow_lines(b, rt, st)
        return out
    if st.handover in ("nest", "sexp"):
        ranked = [chosen] + others
        out = [f"Handing over {n} route{'s' if n != 1 else ''}, best first.", ""]
        for rt in ranked:
            out.append(f"  {rt.label}   {route_objectives(b, rt, st)}"
                       if st.handover == "nest" else "")
            if st.handover == "nest":
                out += ["  " + ln for ln in _nest_lines(b, rt.root_rxn, st, rt)]
            else:
                out.append(f"  {rt.label} = {_sexp(b, rt.root_rxn, st, rt)}")
                out.append("      " + route_line(b, rt, st).split("   ", 1)[-1])
        return [x for x in out if x != ""] if st.handover == "sexp" else out
    if st.handover == "steps":
        ranked = [chosen] + others
        return ([f"Handing over {n} route{'s' if n != 1 else ''}, best first.", ""]
                + steps_block(b, ranked, st) + ["", st.h_routes]
                + [route_line(b, rt, st) for rt in ranked])
    out = [f"Handing over {n} route{'s' if n != 1 else ''}.", ""]
    out += [st.h_routes]
    out += route_block(b, chosen, st)
    if others:
        out.append("")
        out.append(f"  the other {len(others)}, as differences from "
                   f"{chosen.label}:")
        out += [route_summary_line(b, rt, st, reference=chosen) for rt in others]
    return out


# ------------------------------------------------------------------- the turn
def render_env(b: Board, st: RenderStyle = STYLE) -> str:
    """One environment turn, top to bottom.

    Order is fixed and load-bearing: what CHANGED (the events, and a route if
    one closed) comes before what IS (the ledger, the frontier, the materials).
    The model reads the diff first and the state second.
    """
    b.printed = set()          # a render pass: each SMILES prints at most once
    head = header(b, st)
    blocks: list[list[str]] = [[head]] if head else []
    ev = events_block(b, st)
    if ev:
        blocks.append(ev)
    claimed_now = [e.route for e in b.events if e.kind == "claim"]
    if b.routes and (st.routes_when == "always"
                     or any(e.kind == "solved" for e in b.events)):
        blocks.append(routes_block(b, st))
    elif claimed_now:
        # A claim turn reports what it just claimed, not the whole list again:
        # redrawing every route on every claim made the observations grow
        # quadratically in the number of answers the episode collects.
        ref = b.routes[0]
        # The front is over EVERY route claimed so far, not over the ones this turn added:
        # the handover has to order the whole set, and a route claimed six turns ago is still
        # an answer being offered. Only the turn's own rows are redrawn (see above), so the
        # header names the front to keep it readable without re-listing the block.
        rhead = f"{st.h_routes}  {len(b.routes)} claimed"
        rmarks: dict[str, str] = {}
        if ROUTE_FRONT() and len(b.routes) > 1:
            rfront, rbeaten = route_front(b)
            if rfront:
                rhead += f" · front {' '.join(rfront)}"
            rmarks = {lbl: "\u25b2" for lbl in rfront}
            rmarks.update({lbl: "\u227a" + by[0] for lbl, by in rbeaten.items()})
        rows = b.routes if ROUTES_FULL() else [r for r in b.routes if r.label in claimed_now]
        blocks.append([rhead] +
                      [route_summary_line(b, rt, st,
                                          reference=(None if rt is ref else ref),
                                          mark=rmarks.get(rt.label, ""))
                       for rt in rows])
    for part in (ledger_block(b, st), open_block(b, st), evidence_block(b, st),
                 closed_block(b, st), dead_block(b, st)):
        if part:
            blocks.append(part)
    return "\n\n".join("\n".join(bl) for bl in blocks)


# ------------------------------------------------------------ the action block
def format_act(actions, st: RenderStyle = STYLE) -> str:
    """The assistant's <act> block -- one action per line, applied in order.

    Written here rather than in the label builder so that the text the model is
    trained to produce and the text parse.py accepts come from one place.
    """
    from .state import DEAD_REASONS, Analyze, Dead, Done, Open, Rank, Terminate

    lines = []
    for a in actions:
        if isinstance(a, Open):
            lines.append(f"open {a.mid}")
        elif isinstance(a, Rank):
            lines.append(f"rank {a.mid} " + " ".join(f"c{i}" for i in a.order))
        elif isinstance(a, Dead):
            lines.append(f"dead {a.mid}   reason: {DEAD_REASONS[a.reason]}")
        elif isinstance(a, Done):
            # `qm·c0 vg·c9` -- the ledger's own way of naming a reaction, so the
            # text form of a claim reads like the board it was read off.
            pairs = " ".join(f"{m}\u00b7c{c}" for m, c in a.choices.items())
            lines.append(f"done {pairs}")
        elif isinstance(a, Analyze):
            bits = [f"{m} " + " ".join("c" + str(c) for c in v)
                    for m, v in (a.candidates or {}).items()]
            bits += list(a.mids or [])
            lines.append("analyze " + "   ".join(bits))
        elif isinstance(a, Terminate):
            lines.append("terminate")
        else:
            raise ValueError(f"cannot format {a!r}")
    return "\n".join(lines)


def format_act_json(actions) -> dict:
    """The same actions as the board_act argument.

    One object per action, in order, so the JSON says exactly what the text form
    says -- `dead` carries the reason KEY rather than its sentence, because the
    sentence is prose for the reader and the key is what the board checks.
    """
    from .state import Analyze, Dead, Done, Open, Rank, Terminate

    out = []
    for a in actions:
        if isinstance(a, Open):
            out.append({"type": "open", "mid": a.mid})
        elif isinstance(a, Rank):
            # `take` ALWAYS, including when it is 1. Leaving defaults off, the way a person
            # types a call, is wrong for a corpus whose whole subject is this field: `take` is
            # how many of `order` the call commits now, so it is the difference between
            # expanding one subtree and expanding three at one depth. Left implicit, most rank
            # actions would omit it and a student would learn that the key is optional rather
            # than that the count is a decision every rank makes.
            # Emitting it changes nothing the board does -- `parse_act_json` reads
            # `a.get("take", 1)`, so the two forms build an identical `Rank` -- and it makes
            # every rank action in training state how many cuts it takes.
            out.append({"type": "rank", "mid": a.mid, "order": list(a.order),
                        "take": int(getattr(a, "take", 1) or 1)})
        elif isinstance(a, Dead):
            out.append({"type": "dead", "mid": a.mid, "reason": a.reason})
        elif isinstance(a, Done):
            out.append({"type": "done", "choices": dict(a.choices)})
        elif isinstance(a, Analyze):
            d = {"type": "analyze"}
            if a.candidates:
                d["candidates"] = {m: list(v) for m, v in a.candidates.items()}
            if a.mids:
                d["mids"] = list(a.mids)
            out.append(d)
        elif isinstance(a, Terminate):
            out.append({"type": "terminate"})
        else:
            raise ValueError(f"cannot serialise {a!r}")
    return {"actions": out}


def format_turn(think: str, actions, st: RenderStyle = STYLE) -> str:
    """<think>...</think> + <act>...</act> -- the whole assistant turn."""
    act = format_act(actions, st)
    one_line = "\n" not in act and len(think) < 80 and "\n" not in think
    if one_line:
        return f"<think>{think}</think>\n<act>{act}</act>"
    return f"<think>\n{think}\n</think>\n<act>\n{act}\n</act>"
