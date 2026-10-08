#!/usr/bin/env python
"""Write the analysis channel of a board episode, turn by turn, as one continuous line of thought.

The episode replay emits the Harmony conversation with the literal `{THINK}` in each analysis
channel. This fills it with what a teacher writes, and the shape of the problem is the reason
this is a separate script rather than a flag.

WHY PER-TURN AND SEQUENTIAL, NOT PER-DECISION AND PARALLEL.

An episode is not a decision. It opens a molecule, ranks it, opens a piece the ranking
produced, sets a second piece aside, comes back to it several turns later, kills a branch and
re-ranks its parent. Several routes are under construction at once and `OPEN` lists every
molecule still awaiting one. So the reasoning at turn k is not a fresh argument: it has to know
what turn k-1 decided and why, or it restates the board every turn and contradicts itself
across branches -- the failure that makes multi-turn traces read as N unrelated monologues.

Each turn is therefore asked for with the previous turns' own thoughts in context, and asked
to CONTINUE: to refer back to what it set aside, to say which branch it is on when several are
open, and to not re-derive what it already established. That forces the calls to be sequential
within an episode. They are parallel ACROSS episodes, which is where the throughput is.

WHAT IS CITABLE, AND WHAT IS NOT.

`Turn.evidence` (episode.evidence_for) is the table of facts the board actually showed on that
turn -- and it also carries `label`, the DP's oracle answer. The oracle steers; it must never
be quoted. Same rule as the route-level distiller: the teacher sees which action is right so
its conclusion lands there, and `verify` searches the text for the tells.

The board's own notation is what the reasoning may cite: `q`, `ln$`, the ledger, and -- after
an `analyze` action -- the EVIDENCE block's bond positions, reaction names and scaffolds. A
turn that has not analysed anything has no bond facts to cite, and the prompt says so, because
a thought that names a bond the board never showed is the same defect as one that names the
answer.

Usage (episodes with evidence and per-turn facts, from the board-episode replay):
  # 1. look at one turn's prompt before spending anything
  python scripts/traj_route_reasoning_board.py --in eps.jsonl --print-prompt 0:2
  # 2. write the channel
  TEACHER_BASE_URL=http://127.0.0.1:8000/v1 TEACHER_MODEL=Qwen/Qwen3.8-27B \\
    python scripts/traj_route_reasoning_board.py --in eps.jsonl \\
      --out eps_reasoned.jsonl --episode-workers 8 --samples 2
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
import zlib
import time
import urllib.request
from pathlib import Path

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", str(Path(__file__).resolve().parents[3] / "config" / "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)
sys.path.insert(0, RP.BOARD)                          # the board/ package lives with the harness

# The faithfulness half of the check. `verify` below reads a draft for REGISTER -- interface
# vocabulary, an oracle quote, an atom-map index -- all of which are properties of the words
# alone. `factcheck` asks whether the sentences are TRUE of the turn, against the same
# `evidence` dict `support_block` writes the prompt from. Imported softly so this script still
# runs where rdkit is missing; without it the factual gate is simply off.
try:
    from board import factcheck as FC
except Exception as _e:                                                # noqa: BLE001
    FC = None
    _FC_ERR = str(_e)

CORE = """You are an expert synthetic organic chemist writing the private reasoning of a \
retrosynthesis search agent, one turn at a time.

The agent operates a board. Several molecules are open at once, on several partly-built \
routes, and each turn it applies actions to some of them. You are writing the reasoning for \
ONE turn: the thinking that produced the actions that turn took.

This is a continuation, not an essay. The turns before this one are in the conversation with \
the reasoning you already wrote for them. Carry it forward:
- Do not restate the board. The reader has it.
- Say which molecule or branch you are working on when more than one is open, and name the \
others only when they bear on the choice.
- Refer back to what you decided earlier by what it was, not by turn number: "the aniline \
piece I left open", "the ranking I declared on the amide".
- When a branch just failed, say what that taught you before choosing the next thing.

What you may argue from:
%(BRIEF)s

Say it in chemistry, not in interface. The words `menu`, `candidate list`, `the list`,
`offered set` and `on screen` name the machinery you are reading, not the chemistry you are
reasoning about, and a trace built out of them teaches a reader to narrate a UI. Write about the
disconnections, the fragments and the molecules. Candidate numbers (c0, c2) are fine -- they are
how a specific disconnection gets named -- but "c2 cuts the benzyl ether" is reasoning and "the
menu offers c2" is not.

Substitute, do not just avoid. Where you would write "the candidate list shows three amide
cuts", write "three of the cuts break the amide". Where you would write "the menu is in for the
acid", write "the acid can come apart three ways". Name the chemistry that the list contains.

Never describe your own information state in terms of data fields. "I have no bond changes or
reaction names" is a sentence about a table, not about chemistry. If you need to know what a
disconnection does before choosing, say so in chemical terms -- which two candidates you cannot
separate, and what about the bond being made would separate them.

Name the three actions in backticks when you mean the action and not the English word:
`open`, `rank`, `done`. "I `open` mx" and "before I `rank` m2 I need the bond detail" mark the
decision points, while "the only piece left open" and "the ranking I declared" are ordinary
prose and take no backticks. Only you know which use is which, which is why this cannot be
applied afterwards.

Hard rules:
- Reason only from what this turn's brief and its supporting information give you. There is no
action that fetches facts: an open turn has the piece and nothing about its cuts, a rank turn
has the cuts and what each one does. Write within that, and never say that you are about to
look something up -- there is nothing to look up.
- Never mention: which action is correct, that you were told, a route's final cost or length, \
or anything about how the episode ends. You are inside it.
- The board gives a piece's depth and no ceiling. Do not invent one, do not count levels
remaining, and do not argue from how much room is left -- no "ten-level budget", no "with
seven levels below it". Depth is only ever an argument about how much a cut has to earn: a
piece sitting deep has less of a molecule left to simplify, so a cut that merely shuffles
substituents is worth less there than near the target.
- Say what is NEW this turn. Your earlier turns are above you so that you can build on them,
not so that you can repeat them: a sentence you have already written, with the molecule name
swapped, carries no information the reader did not have a turn ago. Refer back in a clause --
"the aniline I banked earlier" -- and spend the paragraph on what changed. If the honest answer
is that this turn is routine, say the one thing that makes it this piece rather than the last
one, and stop.
- No headings, no lists, no restating the actions as a table. Prose, 80-220 words.
- Do not end with a choice line. The actions follow separately."""


# The board has three actions and they are three different questions, so the teacher is given
# three different briefs and three different information sets. Turns are single-typed, so the
# split is exact rather than a heuristic over mixed turns.
BRIEF_OPEN = """You are writing an OPEN turn. AT MOST TWO SENTENCES AND 45 WORDS, plus one
short sentence and 20 more words for each of the clauses in 3 that this call earns.

That is a hard ceiling, not a style note, and the turn is refused for going past it. Everything
past the facts below is the turn restating the board back to itself.

1. This piece cannot be bought as it stands. That is the entire reason it is still open, and
requesting its disconnections is the only move available.
2. WHICH PIECES ARE ALREADY ABOVE IT. They are listed for you, by board id. Whatever comes back
must not hand any of them back: a cut whose precursor is a piece already above this one makes
that piece its own precursor, the board refuses the call, and the search ends there. Name the
ones above this piece so the next turn is choosing with them in view.

3. THE DECISION, where the call makes one. An open turn makes at most two, and each gets its
own clause on its own line. Write a clause ONLY when the call earns it; written where it is not
earned it is narration, and the turn is refused for it.

3a. WHERE THE CALL OPENS MORE THAN ONE PIECE -- always, then:

    opening together: <mid> under <rN>, <mid> under <rN> | <why these, now>

This is the most important sentence an open turn writes. Requesting several pieces in one call
is not bookkeeping, it is the move that makes the search breadth-first: the rank turn that
follows a multi-open compares several molecules, where one after a single open rarely does.
Opening them together is what puts rival disconnections in
front of the ranker AT THE SAME DEPTH, so it can choose between branches instead of finishing
one and discovering later that it was the wrong one. Say that, in the board's own terms.

THE REASON MUST BE THIS TURN'S, NOT A TEMPLATE. Four clauses in one episode reading "two rival
branches, one per parent, so opening both keeps each disconnection alive at one depth" with only
the count changed is one sentence written four times, and it is refused. What separates one
batch from another is on the ledger line of each reaction involved -- `r1  h9·c4  1 of 2 closed`
-- and it is printed for every one of them, on this turn, before you act:

  * `1 of 2 closed` means the step made two pieces and one of them arrived purchasable, so the
    piece being opened is the only thing still blocking it.
  * `0 of 1 closed` means the step owes a single piece, and this is it.
  * `0 of 2 closed` means neither piece is home and the step needs both.

Say which of these you have. It is a reading of the board in front of you, so state it in the
present tense and never as a forecast: `r5 is 0 of 2 closed and owes both` is the board;
`opening these will close r5` is a claim about a turn that has not happened, and it is refused
like any other appeal to how the search turns out.

There are exactly two reasons to batch and the board tells you which one you have. Read each
piece's `under rN` in the OPEN block:

  * THE REACTIONS DIFFER -- the usual case. The pieces descend from different cuts of the
    same parent, so they are rival strategies, not partners: opening both keeps two disconnec-
    tions alive at one depth, and the next turn ranks them against each other with both menus
    in view. `wt under r1 and na under r2 are the two cuts h9 offered; taking both to one depth
    lets the next turn rank them side by side rather than commit to r1 blind.`
  * ONE REACTION OWES THEM ALL -- the rare case. Then they are co-precursors: that step is a
    route only if EVERY piece it owes terminates, so opening one and not the others measures
    nothing about it. The ledger line `r1 1 of 2 closed` is what tells you this.

Name the pieces with their reactions, say which of the two relations you have, and give the
ledger state that makes THIS batch the one to take now. Do not guess a relation the `under` ids
do not support -- calling rival cuts "pieces of the same step" is refused, and so is the
reverse. Count what you name: a clause that says `both` while the call opens three is refused
on the count alone.

3b. ONLY WHERE THE CALL LEAVES SOMETHING OPEN, then:

    leaving open: <board id, or ids> | <why not this turn>

Where the call passes over a piece it could have opened, that is a decision about where the
search does NOT go and it has to be on the page: name the ids it leaves and why, from what the
board shows -- depth, which reaction each sits under, how many pieces that reaction still owes.
Do not name a molecule the call opens, and do not name one that could not have been opened.


NAME NO CHEMISTRY. Not the ring system, not the functional groups, not what the piece "is" as a
compound, not where a good cut would fall. Refer to a piece by its board id -- `u6`, `h9` --
and to nothing else. This is not brevity for its own sake: at an open turn the only measured
facts are purchasability, depth and parentage, so every structural word would be read off a
SMILES string by eye, and a name arrived at that way is a guess that reads like a fact. A turn
that says `u6` cannot be bought and sits under `r2`, whose branch already carries `h9`, is
completely true. A turn that calls u6 a chloro-substituted benzimidazole may not be, and nothing
on this turn can tell the difference.

Do not name a candidate number, a confidence, a plausibility, a round-trip rank or a price
either: none of them exist for a piece that has not been opened."""

BRIEF_RANK = """You are writing a RANK turn. The disconnections are in, and the turn chooses
which of them to take and in what order:
1. STRUCTURE. How each candidate you weigh cuts the molecule -- what it does to the ring
skeleton, how it divides the heavy atoms, what happens to the stereocentres. This is the body.
2. MECHANISM. Which bond is made and which is broken, and the reaction's name only when the
chemistry earns one. Name the bond by its CHEMISTRY -- "the amide C-N, at the carbonyl carbon",
"the C=C between the aldehyde carbon and the ylide carbon" -- and never by an atom-map index
like C:13. An index is an artefact of how the mapping was computed; you will not have it in
front of you next time, and a bond named by its chemical position is one anybody can find by
comparing the product with its precursors. If nothing reproduces the transformation, say that
rather than inventing a name.
3. THE SIGNALS, where they decide something. The board carries four and they answer different
questions, so name the one that actually carries your argument:
   - p, the reaction filter's plausibility, cut-off 0.05. Below it the filter says the step
does not go, and no other score rescues it. Above it, it says only that THIS step goes --
nothing about whether the precursors have anywhere to go, which is where most dead ends are.
   - rt, the rank at which the forward model recovers the product from these precursors. 1
reads cleanly forwards; a dash means it could not be recovered at all, which is a real warning
even when p is high.
   - q, the single-step model's own confidence. It is what a conventional search would expand
on, and it cannot see past a precursor. Reason with it, but never contradict it: if you name q
as your reason, the candidate you take has to be the one it points at.
   - $, MolPrice's estimate of what a fragment costs, in USD per mmol, printed on EVERY
priced fragment and totalled for the cut in its signal group. Dollars, not logs, so they add:
a cut into $25 + $60 costs $85. The star and the dollar are independent -- `*` says a
catalogue sells it, `$` says what it costs -- and a price on a piece you still have to make
is what the cut costs if you stop and buy here.
   Among candidates that tie on p, rt and price are what decide, so a ranking is never a
p-ranking alone. `structure` is for when NO number decided; price IS a number, so a tie on p
and rt with different dollar totals is a price decision, not a structural one.
   When two cuts read the same on p and on rt, price is the axis to NAME, with
both dollar figures: a menu's cheapest and dearest cut usually differ several-fold, so it is
a real ranking there. That a cut's pieces are all
purchasable is a DIFFERENT argument -- it closes the branch -- and belongs under the axis
that decided, not under price. A route is worth its weakest step: a candidate scoring below your best route's
weakest step cannot produce a better route, whatever sits behind it. Cite the axis that
decides; do not recite all four."""

BRIEF_DONE = """You are writing a DONE turn. AT MOST TWO SENTENCES AND 45 WORDS.

That is a hard ceiling, not a style note. A third sentence is a defect and the turn is refused
for it. There are exactly two facts to state and they fit in two sentences; everything past
them is the turn restating the board back to itself.

1. EVERY PIECE THIS ROUTE ENDS ON IS PURCHASABLE. That is what makes it a route and not a plan,
and it is why it is being claimed now. The purchasable leaves are listed for you with their
prices; say that they are what the route ends on.
2. WHICH PIECES ARE NOT. A piece this route resolves by MAKING it at an earlier step is built,
not bought. Never call one of those purchasable -- it is the mistake that turns an unresolved
leaf into a false commit, and it is the one this turn exists to get right. If a piece is not in
the banked list with a price beside it, it is not purchasable, whatever else is true of it.

Name the weakest step in one clause if it is worth saying -- either it is sound enough, or it is
the best any branch reached.

NAME NO CHEMISTRY. Not the transformations, not the ring systems, not the functional groups, not
what any piece "is" as a compound. Refer to a piece by its board id and to a step by the
reaction id the ledger gives it. A route recited as chemistry is a route described from SMILES
by eye, and the descriptions are what go wrong: the ledger already carries which reaction sits
under which molecule, so an id says the same thing and cannot be a guess.

Claiming does not end the episode: if good material sits unopened elsewhere, that is a fact
about the search, not something this turn argues against, so leave it unsaid. Do not total the
route's cost, count its steps as a score, or compare it with a route you have not built."""

BRIEF_FINAL = """You are writing the HAND-OVER. The search is finished and this is the last
thing the agent says: the routes it is handing over, in the order it wants them read. It is the
only message that is not a tool call, so it is the one piece of free text the agent has to
produce on its own, and it needs a reason behind it rather than a format to fill in:
1. WHAT WAS BUILT, in chemistry. The target came apart in a small number of ways that matter --
name them: which bond each family of routes cuts, and what that leaves to buy. Two or three
clauses, not a route-by-route walk; the listing below the message already does that.
2. WHY THIS ORDER. The routes differ, and the difference is the whole content of a hand-over.
Say which route you would run and why: the weakest step it has to survive, how many steps it
takes, what its leaves cost. A route is worth its weakest step, so a route that is short and
expensive can still beat a long cheap one -- say which trade you took.
3. WHAT YOU WOULD NOT CLAIM, when it is worth saying: a branch that stayed open and why it was
not worth the calls, or a route you claimed knowing one step is thin. Do not apologise for the
search and do not summarise your own process -- this is a report on chemistry, not on a session.
Never total a route's cost as a score or rank routes on a number the listing does not carry."""

BRIEFS = {"open": BRIEF_OPEN, "rank": BRIEF_RANK, "done": BRIEF_DONE, "final": BRIEF_FINAL}

# THE BRIEF HAS TO DESCRIBE THE BOARD THE TEACHER IS LOOKING AT. Two board variants differ in
# one thing -- whether the ROUTES block marks its own front -- and a brief that talks about marks
# to a teacher reading an unmarked block is the same defect as a board printing a figure the
# checker does not hold: it asks for a claim the screen cannot support. So there are two
# versions, chosen by the flag that chose the board, and everything in them EXCEPT the reading
# vs deriving instruction is word-for-word identical, so the variants differ only in that.
_FRONT_SHARED = """
Order the front and say what each of its routes buys against the others: the one you would run
first, and what it gives up to the ones behind it. A beaten route is not a second answer -- it
is worse than one you are already handing over on every axis, so leading with it, or offering
it as an alternative without saying it is beaten, claims the search found more than it did.
NAMING ONE WITHOUT SAYING SO IS THE COMMON FAILURE: a beaten route left unmarked reads as an alternative while being worse on every axis than a route
already offered. `J is beaten by C on every axis and is not a second answer` is the shape that
works. Either say what beats it, or leave it out.
Two routes that read the SAME on all three are one answer wearing two labels; say so and hand
over one. Quote the figures the lines carry and total nothing."""

FRONT_BRIEF_MARKS = """

THE BOARD MARKS WHICH ROUTES ARE WORTH HANDING OVER, and the hand-over has to use it. Each
ROUTES line carries three axes -- `p 5/6 pass` (steps clearing the 0.05 filter), `rt 4/6 back`
(steps the forward model recovers), `$.238` (what the leaves cost). A route marked `\u25b2` is
beaten by nothing else on the board; `E \u227aC` is beaten by C on all three at once. The
header names the unbeaten set.""" + _FRONT_SHARED

FRONT_BRIEF_AXES = """

WHICH ROUTES ARE WORTH HANDING OVER IS SOMETHING YOU WORK OUT, and the block gives you
everything the comparison needs. Each ROUTES line carries three axes -- `p 5/6 pass` (steps
clearing the 0.05 filter), `rt 4/6 back` (steps the forward model recovers), `$.238` (what the
leaves cost). One route BEATS another when it is at least equal on all three and better on one.
Nothing on the block marks that for you: read the three figures off each line and say which
routes nothing beats. The comparison is over the routes THIS BOARD HAS CLAIMED -- a route
nothing here beats is not thereby the best that exists.""" + _FRONT_SHARED


def system_for(kind: str) -> str:
    """The shared core with this action's brief spliced into it.

    The front paragraph is appended only where the BOARD ACTUALLY MARKS a front. A brief that
    describes marks the screen does not carry teaches the model to look for something it will
    never see, which is the same defect as a board carrying marks the message never explains.
    """
    brief = BRIEFS.get(kind) or BRIEF_RANK
    if kind in ("final", "done"):
        if os.environ.get("BOARD_ROUTE_FRONT") == "1":
            brief += FRONT_BRIEF_MARKS
        elif os.environ.get("BOARD_ROUTE_AXES") == "1":
            brief += FRONT_BRIEF_AXES
    return CORE % {"BRIEF": brief}


def turn_kind(turn: dict) -> str:
    """Which action this turn takes. Mixed turns fall back to the most demanding brief."""
    acts = {a.get("type") for a in (turn.get("actions") or []) if isinstance(a, dict)}
    if "final" in acts:
        return "final"
    for k in ("rank", "open", "done"):
        if k in acts:
            return k
    return "rank"


def _fmt_menu(ev_menus: dict, mid: str) -> str:
    m = ev_menus.get(mid) or {}
    rows = []
    for c in m.get("candidates") or []:
        sig = " ".join(f"{k}{v}" for k, v in (c.get("signals") or {}).items())
        buy = "".join("*" if b else "" for b in (c.get("buyable") or []))
        rows.append(f"      c{c['c']} {sig}  {' + '.join(c.get('reactants') or [])}{buy}")
    return "\n".join(rows)


def _desc_phrase(d: dict) -> str:
    """A descriptor row as a chemist would read it, not as a field dump."""
    if not d:
        return ""
    bits = []
    n = d.get("n_rings")
    if n is not None:
        sys_ = d.get("n_ring_systems")
        arom = d.get("n_arom_rings")
        bits.append(f"{n} ring(s) in {sys_} system(s)" + (f", {arom} aromatic" if arom else ""))
    if d.get("n_heavy") is not None:
        bits.append(f"{d['n_heavy']} heavy atoms")
    if d.get("mw"):
        bits.append(f"MW {d['mw']:.0f}")
    if d.get("frac_csp3") is not None:
        bits.append(f"Fsp3 {d['frac_csp3']:.2f}")
    for key, lab in (("n_heteroatoms", "heteroatoms"), ("n_halogens", "halogens"),
                     ("n_rot_bonds", "rotatable bonds"), ("n_bridgehead", "bridgeheads"),
                     ("n_spiro", "spiro centres")):
        if d.get(key):
            bits.append(f"{d[key]} {lab}")
    st = d.get("n_stereo_specified")
    un = d.get("n_stereo_unspecified")
    if st or un:
        bits.append(f"stereocentres {st or 0} set" + (f", {un} unspecified" if un else ""))
    if d.get("scaffold"):
        bits.append("Murcko " + d["scaffold"])
    return "; ".join(bits)


def _bond_phrase(row: dict) -> str:
    """The bond change, with the atom labels the mapping produced.

    The labels are given to the teacher because they say WHICH bond; the brief forbids quoting
    them in that form, because an atom-map index is an artefact of how the mapping ran and the
    student has no way to reproduce it. Naming the same bond by its chemical position is
    something anybody can recover by comparing the product with its precursors.
    """
    if not row.get("bond_measured"):
        return "bond not measured (the mapping failed here -- say nothing about which bond)"
    out = []
    if row.get("formed"):
        out.append("formed " + ", ".join(row["formed"]))
    if row.get("broken"):
        out.append("broken " + ", ".join(row["broken"]))
    return "; ".join(out) or "no bond change measured"


def _shape_phrase(sh: dict) -> str:
    if not sh:
        return ""
    bits = []
    if sh.get("scaffold_kept") is not None:
        bits.append("skeleton kept" if sh["scaffold_kept"] else "skeleton changes")
    if sh.get("rings_product") is not None:
        bits.append(f"rings {sh['rings_product']} <- {sh.get('rings_precursors')}")
    if sh.get("fragments"):
        bits.append(f"fragments {sh['fragments']}"
                    + (f" ratio {sh['size_ratio']:.2f}" if sh.get("size_ratio") is not None
                       else ""))
    return "; ".join(bits)


def support_block(turn: dict, kind: str) -> str:
    """The teacher's private supporting information for ONE turn.

    Never rendered onto the board and never shown to the student: with `analyze` out of the
    action space these facts reach the student only through the reasoning the teacher writes.
    Each action gets what its decision turns on and nothing else -- an open turn has no
    candidates to score, a done turn is not choosing a bond.
    """
    ev = turn.get("evidence") or {}
    L: list[str] = []

    if kind == "open":
        # No descriptor line (ring counts, heavy atoms, MW, Fsp3): a brief that says "name no
        # chemistry" beside a block that lists ring systems is asking and forbidding at once,
        # and the counts are the raw material a name gets guessed from. What an open turn
        # decides on is purchasability, depth and parentage, and that is all that is printed.
        for mid, m in (ev.get("mols") or {}).items():
            if m.get("has_menu"):
                continue                       # not one of the pieces this turn opens
            L.append(f"{mid}  {m.get('smiles')}")
            L.append(f"    depth {m.get('depth')}/{m.get('max_depth')}"
                     + (f", from {m['under']}" if m.get("under") else "")
                     + ("  NOT purchasable" if not m.get("buyable") else
                        f"  purchasable at ln${m.get('ln_price')}"))
        banked = ev.get("banked") or []
        if banked:
            L.append("already banked on this route:")
            for b in banked[:8]:
                pr = f" ln${b['ln_price']:.2f}" if b.get("ln_price") is not None else " unpriced"
                L.append(f"    {b['mol']}{pr}  {b['smiles']}")

    elif kind == "rank":
        facts = ev.get("facts") or {}
        for mid, mm in (ev.get("menus") or {}).items():
            rows = {r["c"]: r for r in ((facts.get(mid) or {}).get("candidates") or [])}
            if not rows:
                continue
            L.append(f"{mid}  depth {mm.get('depth')}")
            for c in mm.get("candidates") or []:
                r = rows.get(c["c"])
                if r is None:
                    continue
                sig = c.get("signals") or {}
                buy = "".join("*" if b else "-" for b in (c.get("buyable") or []))
                # A missing round-trip is a dash, the way the board writes it -- "rtNone" is
                # a Python repr and reads as a value rather than as "could not be recovered".
                rt = sig.get("rt")
                rt = "\u2014" if rt is None else rt
                # Prices only for the fragments that have one; an empty slot printed as a bare
                # comma invites the model to read it as a price of zero.
                price = "  ".join(f"ln${p:.2f}" for p in (c.get("ln_price") or [])
                                  if p is not None)
                L.append(f"    c{c['c']}  q{sig.get('q')} p{sig.get('p')} rt{rt}"
                         f"  buyable {buy}" + (f"  {price}" if price else ""))
                L.append(f"         {' + '.join(c.get('reactants') or [])}")
                nm = r.get("named") or []
                tier = r.get("named_tier")
                if nm and tier in ("applies+makes", "applies"):
                    L.append(f"         named: {', '.join(nm)}"
                             f" ({tier}, match {r.get('named_match')})")
                else:
                    L.append("         named: nothing reproduces this transformation")
                L.append(f"         {_bond_phrase(r)}")
                sh = _shape_phrase(r.get("shape") or {})
                if sh:
                    L.append(f"         {sh}")

    elif kind == "final":
        # The whole picture, because the hand-over is a judgement across routes rather than
        # about one molecule: every route with the step it is worth, what the leaves cost, and
        # what was left open.
        for r in (ev.get("routes") or []):
            w = r.get("weakest")
            L.append(f"route {r.get('route')}  weakest step p"
                     + ("--" if w is None else f"{w:.4f}"))
        banked = ev.get("banked") or []
        if banked:
            L.append("leaves banked:")
            for b in banked[:16]:
                pr = f" ln${b['ln_price']:.2f}" if b.get("ln_price") is not None else " unpriced"
                L.append(f"    {b['mol']}{pr}  {b['smiles']}")
        still = sorted(ev.get("mols") or {})
        if still:
            L.append("still open at hand-over: " + ", ".join(still))

    elif kind == "done":
        for r in (ev.get("routes") or []):
            w = r.get("weakest")
            L.append(f"route {r.get('route')}  weakest step p"
                     + ("--" if w is None else f"{w:.4f}"))
        banked = ev.get("banked") or []
        if banked:
            L.append("leaves banked:")
            for b in banked[:12]:
                pr = f" ln${b['ln_price']:.2f}" if b.get("ln_price") is not None else " unpriced"
                L.append(f"    {b['mol']}{pr}  {b['smiles']}")
        still = [mid for mid, m in (ev.get("mols") or {}).items()]
        if still:
            L.append("still open: " + ", ".join(sorted(still)))

    if not L:
        return ""
    return ("Supporting information for this turn. It is yours, not the agent's: write "
            "reasoning that CONTAINS what matters here, never a reference to having been "
            "given it.\n\n" + "\n".join(L))


def turn_prompt(ep: dict, k: int, full_boards: int = 4) -> list[dict]:
    """The conversation for turn k: prior turns with their thoughts, then this observation."""
    kind = turn_kind(ep["turns"][k])
    msgs = [{"role": "system", "content": system_for(kind)}]
    dev = ep.get("developer") or ""
    if dev:
        msgs.append({"role": "user",
                     "content": "The board's rules, as the agent has them:\n\n" + dev})
    turns = ep["turns"]
    # Carry the last `full_boards` observations verbatim and FOLD the ones before them.
    # Replaying every observation grows the prompt linearly and a long episode's late turns
    # exceed the context. What continuity needs is the reasoning and the actions -- the old
    # boards are re-derivable from them, and the current board is on screen in full.
    # Fold at a BLOCK boundary, not a sliding one. A window that moves one turn at a time
    # rewrites the prompt's prefix every turn, so vllm's prefix cache never hits and prefill
    # dominates. Snapping the boundary to a multiple of `block` keeps the prefix
    # byte-identical for `block` consecutive turns, so only one turn in `block` pays full
    # prefill.
    block = 8
    fold_before = max(0, ((k - full_boards) // block) * block)
    for j in range(k):
        t = turns[j]
        if j < fold_before:
            msgs.append({"role": "user", "content": f"[turn {j}] (board folded)"})
        else:
            msgs.append({"role": "user", "content": f"[turn {j}] board:\n{t['env']}"})
        if t.get("thought"):
            msgs.append({"role": "assistant", "content": t["thought"]})
        msgs.append({"role": "user",
                     "content": f"[turn {j}] the actions you took: "
                                f"{json.dumps(t['actions'], ensure_ascii=False)}"})
    t = turns[k]
    tail = [f"[turn {k}] board:\n{t['env']}"]
    rej = t.get("reject")
    if rej:
        # A real, board-verified refusal sits in front of this exact turn so the corpus
        # contains recovery, not just route-following. The call and the
        # board's answer are both true and already happened -- neither is invented for the
        # teacher, so citing them is not a leak the way an oracle number would be.
        tail.append(
            "\nBefore this turn's own actions, this same board refused a call:\n"
            f"  tried: {json.dumps(rej.get('actions'), ensure_ascii=False)}\n"
            f"  the board said: {rej.get('reason') or ''}\n"
            "That happened and is not hypothetical. If it bears on what this turn takes, "
            "account for it in one clause -- what it ruled out or what it revealed -- and "
            "move on; if it does not, do not mention it.")
    # The facts, as the teacher's own information rather than as something the board showed.
    # What each action is allowed to cite is decided by which brief it got and by what
    # support_block chose to hand over.
    sup = support_block(t, kind)
    if sup:
        tail.append("\n" + sup)
    if kind == "final":
        # The hand-over takes no action; what the reasoning has to lead to is the message
        # itself. Handing the teacher the listing keeps the two consistent -- reasoning that
        # recommends a route the listing does not put first would teach a contradiction.
        tail.append("\nThe message this turn hands over:\n"
                    + (t.get("final_text") or ""))
        tail.append("\nWrite the reasoning that leads to exactly that hand-over, continuing "
                    "from your earlier turns. Do not restate the listing -- it is below your "
                    "message already -- and do not say you were told what to do.")
    else:
        tail.append(f"\nThe actions this turn takes: "
                    f"{json.dumps(t['actions'], ensure_ascii=False)}")
        tail.append("\nWrite the reasoning that leads to exactly those actions, continuing "
                    "from your earlier turns. Do not say you were told what to do.")
    msgs.append({"role": "user", "content": "\n".join(tail)})
    return msgs


# ------------------------------------------------------------------ verify
# Interface vocabulary. A ban in the prompt alone does not stop it, so it is enforced here: a
# draft that narrates the UI is rejected and another is drawn. The words name the machinery
# being read, not the chemistry being reasoned about, and a student trained on them learns to
# describe a screen.
# `board` is NOT here on purpose: the developer block says "operating on the board", so it is
# the agent's own vocabulary and banning it would reject good drafts. What is banned is
# narration of the INTERFACE -- the words for the list being read rather than the chemistry in
# it.
# `budget` is not banned here. The board does not print the call budget
# (RenderStyle.show_budget), so a draft reasoning about calls remaining is reasoning from a
# number it was never shown; the schema module bans it.
# A depth ceiling the board does not print. Enforced rather than only asked for, because a
# prose ban does not shift the habit of reaching for the number.
_CEILING = [r"\b(ten|nine|eight|seven|six|five|four|three|two|\d+)[- ]level\b",
            r"\b(levels?|plies|layers?)\s+(left|remaining|below|to (go|spare))\b",
            r"\bdepth (budget|ceiling|cap|limit)\b",
            r"\bof (ten|\d+) (levels?|deep)\b",
            r"\bdepth \d+/\d+"]

_JARGON = [r"\bmenu\b", r"\bcandidate list\b", r"\bthe list\b", r"\bon screen\b",
           # `offered set` is interface vocabulary wearing a different coat and spreads as a
           # stock phrase by propagation, so it is banned on the same terms.
           r"\boffered set\b",
           # and the inventory-of-missing-fields sentence
           r"\bno bond changes\b", r"\breaction names or scaffolds\b",
           r"\bhave no (?:bond|mechanis|evidence)",
           # Anthropomorphic tool-talk. The board carries `q`
           # and the brief names it; "the model believes in c0 most strongly" is the same
           # claim wearing a personality, and it teaches the student to describe an oracle it
           # will not have. Each has a substitution below, because a jargon pattern with no
           # replacement makes salvage() a no-op that keeps the draft unchanged.
           r"\bthe (?:single[- ]step )?model believes(?: in)?\b",
           r"\bthe (?:single[- ]step )?model's own pick\b",
           r"\bthe (?:single[- ]step )?model's pick\b",
           r"\bthe (?:single[- ]step )?model points at\b",
           r"\bwhether the model routes through\b"]

# Claims no turn may make: the oracle answer, the instruction, knowledge of the outcome.
_BAD = [r"\bI was told\b", r"\bthe correct (action|answer|choice)\b", r"\boracle\b",
        r"\bgold\b", r"\bthe label\b", r"\bin hindsight\b",
        r"\bwe know that this leads\b", r"\bas instructed\b", r"\bper the guidance\b"]

# Post-hoc MID-EPISODE and ordinary language at the HAND-OVER. A turn in the middle of a search
# that names a route's total cost or calls something "the final route" is reading the end of an
# episode it is still inside; the hand-over IS the end, the routes there are final, and what
# they cost is the subject of the message. Applying these to the hand-over would reject correct
# summaries, and a model that reaches for "total cost" once reaches for it again on the retry. A
# mid-episode turn may cite a PRICE on a purchasable fragment -- that is `ln$`, the board's own
# notation -- and may not total a route or count how many the board can make. Both are hand-over
# judgements the turn has not earned yet. The patterns match the shapes drafts actually use
# ("$74", "six distinct routes"), not only the words `total cost`. The `$` pattern excludes
# `ln$` explicitly, or every legitimate fragment price is a rejection.
_BAD_MIDEPISODE = [r"\beventually\b", r"\bthe final route\b", r"\btotal cost\b",
                   r"(?<!ln)\$\s?\d",
                   r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
                   r"(?:distinct\s+|different\s+|complete\s+)?routes\b",
                   r"\bthe board can (?:now )?make\b"]


def verify(text: str, ep: dict, k: int) -> dict:
    """Everything a turn's thought must not contain. Recorded, never silently dropped."""
    # Search the TEXT case-insensitively, never a lowercased copy: `\bI was told\b` carries a
    # capital and can never match a lowercased haystack, so that check silently never fired.
    kind = turn_kind(ep["turns"][k])
    phrases = [p for p in _BAD if re.search(p, text, re.I)]
    if kind != "final":
        phrases += [p for p in _BAD_MIDEPISODE if re.search(p, text, re.I)]
    ev = (ep["turns"][k].get("evidence") or {})
    # The DP label is the oracle. A thought quoting it is not reasoning, it is transcription.
    labels = {str((c or {}).get("label")) for m in (ev.get("menus") or {}).values()
              for c in (m.get("candidates") or []) if (c or {}).get("label") is not None}
    low = text.lower()
    label_hits = sorted(x for x in labels if x and len(x) > 3 and x.lower() in low)
    # An atom-map index is never citable, on any turn. The student never sees one and cannot
    # compute one, so a trace containing "formed N:6-C:7" trains it to state an identifier it
    # has no way to produce. The same bond named by its chemical position -- the amide C-N at
    # the carbonyl carbon -- is recoverable by comparing the product with its precursors, which
    # is why the rank brief asks for that form and this rejects the other.
    bond_claim = bool(re.search(r"\b[A-Z][a-z]?:\d+\b", text))
    jargon = [j for j in _JARGON if re.search(j, text, re.I)]
    ceiling = [c for c in _CEILING if re.search(c, text, re.I)]
    # An OPEN turn has no candidates, so it cannot have candidate numbers or their scores. A
    # draft that names them is reasoning from a table that does not exist for this piece yet.
    premature = []
    if kind == "open":
        # What is premature is a score for THIS piece, which has none yet -- not every string
        # that looks like one. Two exemptions:
        #
        #   * a MOLECULE ID. Ids are two characters from a pool that includes `p4` and `q9`, and
        #     `\bp[\s.]?\d` and `\bq[\s.]?\d` match those exactly. The open brief asks the
        #     turn to name the pieces above it BY ID, so without this the prompt would require
        #     what this forbids.
        #   * a candidate the LEDGER has already spent. `The c0 cut I declared on dm has landed`
        #     is a backward reference to a decision on screen -- the env prints `r1  dm·c0` --
        #     not a score read off a table that does not exist. The check's own docstring says
        #     "for this piece"; a candidate applied on an ancestor is not for this piece.
        # Over every turn UP TO this one, not just this one. An id is exempt because the board
        # printed it, and it prints each id once -- in the turn the molecule first appeared. The
        # ancestors an open turn is asked to name are by definition older than it, so scoping
        # this to the current env would exempt none of them.
        known: set[str] = set()
        spent: set[str] = set()
        for j in range(k + 1):
            e = ep["turns"][j].get("env") or ""
            known |= set(re.findall(r"<mol (\w+)>", e))
            known |= set(((ep["turns"][j].get("evidence") or {}).get("mols") or {}).keys())
            spent |= {f"c{n}" for n in re.findall(r"\b\w+·c(\d+)\b", e)}
        for pat in (r"\bc\d\b", r"\bq[\s.]?\d", r"\bp[\s.]?\d", r"\brt\s?\d",
                    r"\bln\$"):
            for m in re.finditer(pat, text):
                hit = m.group(0).strip()
                if hit in known or hit in spent:
                    continue
                premature.append(pat)
                break
    # Which of this turn's actions the writer did NOT mark in backticks. Reported so the
    # normaliser below knows what to look for, and NOT part of `ok`: making it a rejection
    # multiplies the draws, because a model that missed the mark once mostly misses it on
    # retry too. Only the writer can tell `open` the action from "left
    # open" the adjective, so the prompt asks and the normaliser cleans up the safe half.
    acts = {a.get("type") for a in (ep["turns"][k].get("actions") or []) if isinstance(a, dict)}
    acts = {a for a in acts if a in ("open", "analyze", "rank", "done")}
    unmarked = sorted(a for a in acts if not re.search(rf"`{a}`", text))
    # OPEN and DONE are prose, on purpose: the rank turns around them carry a header-block
    # schema, and a model conditioned on that context imitates the shape regardless of what the
    # brief for THIS turn asks for -- a draft can open with a bare `MOLECULE` line and write
    # the rank form's blocks for a turn that has no candidates to put in them.
    # The brief already says "no headings, no lists"; this is what enforces it.
    schema_mimicry = []
    if kind in ("open", "done"):
        # Block headers AND the rank form's SLOT LINES. Headers alone are not enough: an open
        # turn can carry an `ordered by:` line, or `carried over:` / `risk:` (fields not in the
        # form), or name `route` as the measure, which is an oracle-lookahead axis that leaks
        # the answer. An open turn reciting `ordered by: route | ...` teaches the student a
        # measure the board does not carry.
        schema_mimicry = [ln for ln in re.findall(
            r"^\s*(MOLECULE|DISCONNECTIONS(?:\s+OF\b.*)?|DECIDING\b.*|WHY)\s*$",
            text, re.MULTILINE)]
        schema_mimicry += [m.group(1) for m in re.finditer(
            r"^\s*(ruled out|ordered by|leaves to make|carried over|risk)\s*:",
            text, re.MULTILINE | re.IGNORECASE)]
        # And the same slots run INTO a sentence, which the line anchor above cannot see:
        # `...nothing to rule out this turn. ruled out: none | no route | nothing has been
        # opened...`. The pipe is what makes it unambiguous -- prose does not use one -- so a
        # slot keyword whose colon is followed by a `|` on the same line is the form, wherever
        # it sits. `no route` in such a slot leaks the search's own lookahead.
        schema_mimicry += [m.group(1) for m in re.finditer(
            r"\b(ruled out|ordered by|leaves to make|carried over|risk)\s*:[^\n|]*\|",
            text, re.IGNORECASE)]
    return {"ok": not (phrases or label_hits or bond_claim or jargon or premature or ceiling
                       or schema_mimicry),
            "depth_ceiling": ceiling,
            "phrases": phrases, "label_quotes": label_hits,
            "atom_index": bond_claim, "premature_signals": premature, "jargon": jargon,
            "schema_mimicry": schema_mimicry,
            "unbackticked_actions": unmarked,
            "words": len(text.split())}


# ------------------------------------------------------------------ salvage
# Ordered longest-first so "the candidate list" is rewritten before "the list". The replacement
# has to be CHEMISTRY, not a synonym for the interface. Turn k's prompt carries turns 0..k-1 with their
# reasoning, so one clause salvage rewrites in an early turn becomes the episode's house style
# and propagates to every turn after it -- the same conditioning that makes a first turn without
# backticks produce an episode without backticks. A substitution therefore has to be a phrase
# that is CORRECT to imitate.
_SUB = [(r"\bthe candidate list\b", "the disconnections"),
        (r"\bcandidate list\b", "disconnections"),
        (r"\bthe menu\b", "the disconnections"),
        (r"\bmenu\b", "disconnections"),
        (r"\bthe list\b", "the disconnections"),
        (r"\bon screen\b", "available"),
        (r"\bthe single-step model believes in\b", "q favours"),
        (r"\bthe model believes in\b", "q favours"),
        (r"\bthe single-step model believes\b", "q favours"),
        (r"\bthe model believes\b", "q favours"),
        (r"\bthe single-step model's own pick\b", "q's own pick"),
        (r"\bthe model's own pick\b", "q's own pick"),
        (r"\bthe single-step model's pick\b", "q's pick"),
        (r"\bthe model's pick\b", "q's pick"),
        (r"\bthe single-step model points at\b", "q points at"),
        (r"\bthe model points at\b", "q points at"),
        (r"\bwhether the model routes through\b", "whether q routes through")]


def factcheck(text: str, ep: dict, k: int, a) -> dict:
    """Does the draft agree with the facts this turn was given? See board/factcheck.py.

    Returns a `check`-shaped dict, or an always-passing stub when the module or the flag is
    off, so the caller needs no branch. The turn's own action list is passed through because
    "which candidate does this turn TAKE" is what most of the claims are about, and it is not
    in the evidence dict.
    """
    if FC is None or not getattr(a, "factcheck", True):
        return {"ok": True, "violations": [], "codes": [], "fatal": [], "failed_checks": []}
    prev = [ep["turns"][i].get("thought") or "" for i in range(k)]
    return FC.check(text, ep["turns"][k].get("evidence") or {}, turn_kind(ep["turns"][k]),
                    prev_thoughts=[p for p in prev if p],
                    actions=ep["turns"][k].get("actions"),
                    # every molecule the episode has already shown, so a backward reference to
                    # a fragment banked earlier is not read as a claim about this turn
                    seen_smiles=FC.smiles_seen(ep["turns"][i].get("evidence") or {}
                                               for i in range(k)))


def draft_rank(v: dict, fv: dict) -> int:
    """How bad a rejected draft is, lowest kept. Three tiers, and the order is the point.

    A LEAK is worst: the oracle answer or a post-hoc number is a claim the student must never
    learn to make and no rewrite repairs it. A false CLAIM about the chemistry is next -- also
    unsalvageable, but a re-prompt naming the number that refutes it usually fixes it, which
    blind resampling does not. Register defects are last, because `salvage` can rewrite them.
    """
    if v.get("phrases") or v.get("label_quotes") or v.get("atom_index") \
            or v.get("premature_signals"):
        return 2
    if fv.get("fatal"):
        return 1
    # `schema_mimicry` stays in tier 0 with jargon: the chemistry is intact and `salvage` strips
    # the header lines off it, so ranking it with the leaks would refuse a draft that is one
    # substitution away from correct.
    return 0


def salvage(text: str, v: dict) -> tuple[str, list] | tuple[None, None]:
    """Rewrite the interface words out of a draft, or refuse.

    Two classes of defect and they are not alike. A LEAK -- the oracle answer, a post-hoc
    number, a bond position on a turn that analysed nothing -- is a claim the student must
    never learn to make, and no rewrite fixes it; those drafts are refused however many draws
    have been spent. Interface vocabulary is a register problem: the sentence says the right
    thing about the wrong subject, and swapping the noun leaves the chemistry intact.

    This is a last resort, taken only after --max-draws, and it is recorded on the turn
    (`thought_salvaged`) so a downstream pass can find every sentence a human, not the
    teacher, chose the words for.
    """
    if v.get("phrases") or v.get("label_quotes") or v.get("atom_index") \
            or v.get("premature_signals"):
        return None, None
    if not (v.get("jargon") or v.get("schema_mimicry")):
        return None, None
    # A block header on an open or done turn is FORMATTING, not a false claim, so it is
    # salvaged rather than refused. Refusing it would leave the first turn of an episode without
    # reasoning -- the opening move unexplained, and the position that conditions everything
    # after it. The cause is imitation with nothing else to imitate: at turn 0 the only form in
    # context is the developer block's own MOLECULE/DISCONNECTIONS/DECIDING listing, so the
    # model copies that. Stripping the header lines leaves the prose the brief asked for.
    if v.get("schema_mimicry"):
        text = re.sub(r"^[ \t]*(MOLECULE|DISCONNECTIONS(?:\s+OF\b.*)?|DECIDING\b.*|WHY)"
                      r"[ \t]*$\n?", "", text, flags=re.M)
        # The slot lines go WHOLE, not just their labels: the value is what carries the defect
        # ("ordered by: route | ..." leaks the measure; "risk: ..." is a field with no honest
        # null, not part of the form). A stripped draft that ends
        # up under --min-words is dropped, which is the right trade against shipping either.
        text = re.sub(r"^[ \t]*(?:ruled out|ordered by|leaves to make|carried over|risk)"
                      r"[ \t]*:.*$\n?", "", text, flags=re.M | re.I)
        # mid-sentence: cut from the slot keyword to the end of its line
        text = re.sub(r"\b(?:ruled out|ordered by|leaves to make|carried over|risk)"
                      r"\s*:[^\n|]*\|[^\n]*", "", text, flags=re.I)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    def _keep_case(m, word):
        # "The menu" must not become "the offered set" mid-sentence-start; match the case of
        # what was there rather than hard-coding the replacement's own.
        return word[0].upper() + word[1:] if m.group(0)[0].isupper() else word

    out, hits = text, []
    for pat, wf in _SUB:
        new = re.sub(pat, lambda m, w=wf: _keep_case(m, w), out, flags=re.I)
        if new != out:
            hits.append(pat)
        out = new
    # The inventory-of-fields sentences have no substitution: drop the sentence rather than
    # paraphrase a claim about the data layout into a claim about chemistry.
    keep = [x for x in re.split(r"(?<=[.;!?])\s+", out)
            if not re.search(r"\bno bond changes\b|\breaction names or scaffolds\b"
                             r"|\bhave no (?:bond|mechanis|evidence)", x, re.I)]
    out = " ".join(keep).strip()
    return (out, hits) if out else (None, None)


_ACT_ALIAS = {"open": ["open"], "analyze": ["analyze", "analyse"],
              "rank": ["rank", "re-rank"], "done": ["done"]}


def turn_action_targets(turn: dict) -> dict:
    """{action name: the molecule ids that action was applied to} for one turn."""
    out: dict[str, set[str]] = {}
    for a in (turn.get("actions") or []):
        if not isinstance(a, dict):
            continue
        t = a.get("type")
        if t not in _ACT_ALIAS:
            continue
        ids = set()
        if a.get("mid"):
            ids.add(str(a["mid"]))
        ids |= {str(x) for x in (a.get("mids") or [])}
        ids |= {str(x) for x in (a.get("candidates") or {})}
        ids |= {str(x) for x in (a.get("choices") or {})}
        out.setdefault(t, set()).update(ids)
    return out


def mark_actions(text: str, targets: dict) -> tuple[str, list[str]]:
    """Backtick an action word only where it governs an id the turn actually acted on.

    The governing position is the one place the word cannot be ordinary English -- "I open mx",
    "before I rank m2" -- while "the only piece left open" and "the ranking I declared" are
    prose and must survive untouched. Matching on the turn's REAL ids is what makes that safe:
    accepting any two-character token after the verb would wrap "left open is n4", because `is`
    is two characters. `done` is matched bare, since it names the hand-over rather than a
    molecule. Only governing positions are marked; mentions that were never actions are left.
    """
    hits = []
    for act, ids in sorted(targets.items()):
        # Longest alias first, so "re-rank" is wrapped whole instead of leaving "re-`rank`".
        for word in sorted(_ACT_ALIAS[act], key=len, reverse=True):
            if act == "done":
                pat = re.compile(rf"(?<!`)\b(done)\b", re.I)
            elif not ids:
                continue
            else:
                alt = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
                pat = re.compile(
                    rf"(?<!`)\b({re.escape(word)})\b(?=\s+(?:on\s+|up\s+)?(?:{alt})\b)",
                    re.I)
            new_text = pat.sub(lambda m: f"`{m.group(1)}`", text)
            if new_text != text:
                hits.append(f"{act}:{word}")
            text = new_text
    return text, hits


# ------------------------------------------------------------------ teacher
_RR = {"i": 0, "lock": threading.Lock()}
_STICKY = threading.local()


# A replica that has failed several times in a row is stepped over until the cooldown expires.
# WHY THIS EXISTS. Pinning makes a dead replica much more expensive than round-robin: the
# episodes homed on it pay a failed attempt and a backoff on EVERY request, and their load
# lands entirely on one neighbour, serialised behind its own retries.
_HEALTH = {"fail": collections.Counter(), "until": {}, "lock": threading.Lock()}
_DOWN_AFTER = 3          # consecutive failures before a replica is considered down
_DOWN_FOR = 120.0        # seconds before it is tried again


def _live(urls: list[str]) -> list[str]:
    """The replicas worth sending to. Never empty -- if all are down, try them all anyway."""
    now = time.time()
    with _HEALTH["lock"]:
        live = [u for u in urls if _HEALTH["until"].get(u, 0.0) <= now]
    return live or list(urls)


def _mark(url: str, ok: bool) -> None:
    with _HEALTH["lock"]:
        if ok:
            _HEALTH["fail"][url] = 0
            _HEALTH["until"].pop(url, None)
            return
        _HEALTH["fail"][url] += 1
        if _HEALTH["fail"][url] >= _DOWN_AFTER:
            _HEALTH["until"][url] = time.time() + _DOWN_FOR


_HOME = {"i": 0, "lock": threading.Lock()}


def stick_to(key) -> None:
    """Pin this thread's teacher requests to one replica, taken in turn.

    One episode is one worker thread, so setting this at the top of an episode routes every
    turn of it -- and every draw of every turn -- to the same replica.

    WHY A COUNTER AND NOT A HASH OF `key`. Each process owns a disjoint set of episodes, so
    processes have nothing to agree about and a hash buys only variance. Worse, it CORRELATES
    the assignment with the id, and the assignment is permanent -- so when one replica runs
    hot, the episodes hashed to it fall behind together and all end up in the tail. A counter
    cannot degenerate that way -- consecutive episodes go to consecutive replicas -- and
    stickiness, which is what the prefix cache needs, is preserved either way.
    """
    with _HOME["lock"]:
        _STICKY.home = _HOME["i"]
        _HOME["i"] += 1
    _STICKY.key = key


def _next_url(urls: list[str], attempt: int = 0) -> str:
    """The replica this request goes to.

    The teacher fits on one card, so the fleet is INDEPENDENT replicas rather than one
    tensor-parallel replica: tensor parallelism shortens a single request and this workload is
    many independent ones.

    WHY IT IS PINNED PER EPISODE rather than rotated per request. This workload is not a stream
    of independent prompts, whatever it looks like: turn k's prompt is turn k-1's prompt plus a
    board observation and a thought, and a turn's draws share their prompt exactly. So almost
    all of it is a prefix some replica has already seen, and prefill dominates. Round-robin
    makes every replica re-prefill the same growing episode prefix, because consecutive
    requests from one episode land on different cards. Pinning the episode makes that prefix
    land where it is already cached; the rotation survives only as the retry path, so a dead
    replica is still stepped over.
    """
    live = _live(urls)
    home_i = getattr(_STICKY, "home", None)
    if home_i is None:
        with _RR["lock"]:
            u = live[_RR["i"] % len(live)]
            _RR["i"] += 1
        return u
    u = urls[(home_i + attempt) % len(urls)]
    if u in live:
        return u
    # The home is down. Spread the displaced episodes over what is LEFT rather than stepping to
    # the next card: walking forward would put every one of the dead card's episodes on the same
    # neighbour.
    # Episodes whose own card is healthy never reach this line, so they keep their cached
    # prefix, and the displaced ones go home when the card returns -- `home_i` does not move.
    return live[(home_i + attempt) % len(live)]


def call_teacher(messages, a, retries: int = 4, temperature: float | None = None):
    payload = {"model": a.model, "messages": messages,
               "temperature": a.temperature if temperature is None else temperature,
               "max_tokens": a.max_tokens}
    if a.no_thinking:
        # A thinking model can spend its whole budget deliberating and return an EMPTY answer,
        # with every token in `reasoning` and `content` "". We are not distilling the teacher's
        # private deliberation -- we want the turn's thought as the agent would think it -- so
        # the thinking channel is turned off.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json"}
    if a.api_key:
        hdr["Authorization"] = f"Bearer {a.api_key}"
    last = None
    for i in range(retries):
        url = _next_url(a.urls, i)   # a retry lands on the NEXT replica, not the dead one
        try:
            req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                         data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=a.timeout) as fh:
                out = json.loads(fh.read())
            msg = out["choices"][0]["message"]
            # gpt-oss puts the analysis channel in `reasoning_content` and the visible answer
            # in `content`; qwen3 does the same under its own parser. We want the ANSWER --
            # the thought we are writing is the board turn's reasoning, not the teacher's own
            # private deliberation about how to write it.
            # vllm exposes the parsed thinking channel as `reasoning` on some parsers and
            # `reasoning_content` on others; `content` is the answer. Prefer the answer and
            # fall back only so a thinking-only reply is not silently recorded as empty.
            text = (msg.get("content") or msg.get("reasoning")
                    or msg.get("reasoning_content") or "")
            _mark(url, True)
            return text, out.get("usage") or {}
        except Exception as e:                                         # noqa: BLE001
            last = e
            _mark(url, False)
            # No backoff when another replica is still up: the next attempt goes to a different
            # card, so sleeping only holds the worker while healthy capacity sits idle.
            if len(_live(a.urls)) <= 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"teacher failed after {retries}: {type(last).__name__}: {last}")


def reason_episode(ep: dict, a) -> dict:
    """Fill every turn's `thought`, in order, each with the ones before it in context.

    Sequential BY CONSTRUCTION: turn k's prompt contains turn k-1's thought. Sampling several
    draws per turn and keeping the best is still worth it -- a turn whose only draw leaks would
    otherwise poison every turn after it, since they are written knowing it.
    """
    stats = collections.Counter()
    fill = ep.get("_fill")
    for k in range(len(ep["turns"])):
        if fill is not None and k not in fill:
            stats["kept"] += bool(ep["turns"][k].get("thought"))
            continue
        msgs = turn_prompt(ep, k, a.full_boards)
        base_msgs = msgs
        best, best_v, best_f = None, None, {}
        repairs = 0
        # Draw until something passes, up to --max-draws. A turn left empty is not a local
        # loss: every turn after it is written with the turns before it in context, so a hole
        # propagates as a discontinuity through the rest of the episode. So the budget is
        # escalating rather than fixed -- --samples draws at the requested temperature, then
        # up to --max-draws more at a rising one, because a model that keeps producing the
        # same rejected sentence needs a different sample, not another identical try.
        draws = 0
        fails = 0
        best_rank = 9
        while draws < max(a.max_draws, a.samples, 1):
            bump = 0.0 if draws <= a.samples else min(0.5, 0.12 * (draws - a.samples))
            try:
                text, usage = call_teacher(msgs, a, temperature=a.temperature + bump)
            except Exception as e:                                     # noqa: BLE001
                # A transport failure is not a bad sample, so it must not be charged against
                # the draw budget, or a turn unlucky enough to hit several gets zero drafts.
                # Retry it, with its own cap so a dead replica cannot spin here forever.
                stats["request_failed"] += 1
                ep["turns"][k]["thought_error"] = str(e)[:200]
                fails += 1
                if fails > max(a.max_draws, a.samples, 1) * 3:
                    break
                time.sleep(min(8.0, 0.5 * fails))
                continue
            draws += 1
            v = verify(text, ep, k)
            fv = factcheck(text, ep, k, a)
            stats["drawn"] += 1
            for c in fv.get("codes") or []:
                stats["fact_" + c] += 1
            fact_ok = fv["ok"] if a.fact_strict else not fv.get("fatal")
            if v["ok"] and fact_ok and a.min_words <= v["words"] <= a.max_words:
                best, best_v, best_f = text, v, fv
                break
            # A false claim is not a bad sample, it is a wrong belief: the same prompt at a
            # higher temperature returns the same sentence with different adjectives. So the
            # draft is handed back with the number that refutes it and redrawn ONCE per
            # violation set, before the blind-resample budget is touched. Repairs are capped
            # because a teacher that will not take the correction will not take it on the
            # fourth pass either, and each round is a full-context call.
            if (not fact_ok) and v["ok"] and repairs < a.fact_repair:
                repairs += 1
                stats["fact_repair_attempted"] += 1
                msgs = base_msgs + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": FC.describe(fv)},
                ]
                # Not charged as a draw: the redraw is answering a different prompt, and
                # charging it spends the sampling budget on the correction rather than on the
                # turn.
                draws -= 1
                continue
            msgs = base_msgs
            # Keep the LEAST-BAD draft, not the first one. These defects are not equally
            # fatal: jargon is a substitution salvage() can make, a leak is a claim no
            # rewrite repairs. Holding the first draft would refuse a turn whose opening draw
            # leaked even when a later draw was jargon-only and fixable.
            rank = draft_rank(v, fv)
            if rank < best_rank:
                best, best_v, best_f, best_rank = text, v, fv, rank
            stats["rejected"] += 1
            if fv.get("fatal"):
                stats["rejected_facts"] += 1
        if best is None:
            continue
        # Which actions this turn took, for the backtick normaliser.
        _acts = turn_action_targets(ep["turns"][k])
        if best_f:
            ep["turns"][k]["thought_facts"] = {
                kk: best_f[kk] for kk in ("codes", "fatal", "repeat_ratio", "failed_checks")
                if best_f.get(kk)}
        _fact_ok = (best_f.get("ok", True) if a.fact_strict
                    else not (best_f or {}).get("fatal"))
        # Salvage rewrites the interface words out of a draft. It cannot rewrite a false claim
        # about the chemistry -- swapping a noun leaves "takes c6 because q points at it" just
        # as wrong -- so a draft carrying a fatal factual violation is refused here on the same
        # terms as a leak, however many draws were spent.
        if not (best_v or {}).get("ok") and _fact_ok and a.salvage:
            fixed, hits = salvage(best, best_v or {})
            if fixed and a.min_words <= len(fixed.split()) <= a.max_words:
                fixed, marked = mark_actions(fixed, _acts)
                if marked:
                    hits = list(hits or []) + marked
                ep["turns"][k]["thought"] = fixed
                ep["turns"][k]["thought_salvaged"] = hits
                ep["turns"][k]["thought_draft"] = best
                ep["turns"][k]["thought_check"] = best_v
                stats["salvaged"] += 1
                stats["kept"] += 1
                continue
        if (best_v or {}).get("ok") and _fact_ok:
            text, marked = mark_actions(best, _acts)
            ep["turns"][k]["thought"] = text
            if marked:
                ep["turns"][k]["thought_marked"] = marked
                stats["marked"] += 1
        else:
            ep["turns"][k]["thought"] = ""
            if not _fact_ok:
                stats["dropped_facts"] += 1
        ep["turns"][k]["thought_draft"] = best
        ep["turns"][k]["thought_check"] = best_v
        stats["kept"] += bool(ep["turns"][k]["thought"])
    ep["reasoning_stats"] = dict(stats)
    return ep


# ------------------------------------------------------------------ io
def load_episodes(path: Path) -> list[dict]:
    """One row per episode from the board-episode replay.

    Reads the fields the replay already writes: `harmony_messages` for the observations, and
    `evidence` for the per-turn fact table. Turns are reconstructed from them rather than from
    a pickled Board, so this stays independent of the board module's in-memory types.
    """
    out = []
    for row in (json.loads(l) for l in open(path) if l.strip()):
        msgs = row.get("harmony_messages") or []
        dev = ((row.get("messages") or [{}])[0] or {}).get("content", "")
        turns, cur = [], None
        for m in msgs:
            role = m.get("role")
            # A rejection pair is a `supervised: false` assistant call and the board's real
            # rejection of it, inserted in front of the turn it shares an env with --
            # no env message of its own. Naively parsed, the fake call overwrites `cur`'s
            # actions (role == assistant, recipient set) and the fake tool reply reads as
            # a fresh env (role == tool), spawning a turn `evidence` never accounted for
            # and shifting every turn after it out of alignment with `row["evidence"]`.
            # So the pair is captured on `cur["reject"]` instead of being let through the
            # ordinary branches below.
            if (role == "assistant" and m.get("recipient") and m.get("supervised") is False
                    and cur is not None):
                try:
                    bad = json.loads(m.get("content") or "{}").get("actions")
                except Exception:                                      # noqa: BLE001
                    bad = None
                cur["reject"] = {"actions": bad, "reason": None}
                continue
            if (role == "tool" and cur is not None and cur.get("reject")
                    and cur["reject"].get("reason") is None):
                cur["reject"]["reason"] = m.get("content", "")
                continue
            if role in ("user", "tool"):
                cur = {"env": m.get("content", ""), "actions": None, "thought": ""}
                turns.append(cur)
            elif role == "assistant" and m.get("recipient") and cur is not None:
                try:
                    cur["actions"] = json.loads(m.get("content") or "{}").get("actions")
                except Exception:                                      # noqa: BLE001
                    cur["actions"] = None
        evs = row.get("evidence") or []
        for i, t in enumerate(turns):
            t["evidence"] = evs[i] if i < len(evs) else {}
        turns = [t for t in turns if t["actions"]]
        # The hand-over is an EXTRA assistant message on the `final` channel, appended after
        # the turn loop and sharing the last board rather than getting one of its own -- so it
        # is not a turn and the filter above never sees it. It is the one message the student
        # produces as free text at inference, everything else being a tool call, and without
        # this it would carry no analysis channel: a route listing the model has to reproduce
        # without ever being given a reason for its order.
        #
        # Appended as one extra pseudo-turn typed `final`, reasoning over the last board it
        # actually followed. Its thought is written to `handover_thought`, NOT to
        # `turn_thoughts`, because turn_thoughts is indexed against the BOARD's turn list and
        # an extra entry there would either be dropped by the renderer's bounds check or, worse,
        # land on the wrong turn.
        hand = next((m for m in reversed(msgs)
                     if m.get("role") == "assistant" and m.get("channel") == "final"
                     and (m.get("content") or "").strip()), None)
        if hand is not None and turns:
            turns.append({"env": turns[-1]["env"], "actions": [{"type": "final"}],
                          "thought": "", "final_text": hand["content"],
                          "evidence": turns[-1].get("evidence") or {}})
        if turns:
            out.append({"row": row, "developer": dev, "turns": turns,
                        "target": row.get("target"), "id": row.get("id") or row.get("target")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="board-episode jsonl from the episode replay")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-prompt", default=None,
                    help="EPISODE:TURN, e.g. 0:2 -- render one prompt and exit")
    ap.add_argument("--base-url", default=os.getenv("TEACHER_BASE_URL",
                                                    "http://127.0.0.1:8000/v1"),
                    help="one URL, or several comma-separated: requests rotate over them, "
                         "and a retry goes to the next replica rather than the one that "
                         "just failed")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "teacher"))
    ap.add_argument("--api-key", default=os.getenv("TEACHER_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2400,
                    help="too small a cap truncates the reply (finish_reason=length), "
                         "which reads as an empty thought")
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false. Required for "
                         "Qwen3.x: with thinking on it returns reasoning and no content")
    # A replica can stop scheduling while still answering `/v1/models` and `/health` with 200;
    # a frozen replica does not fail -- its requests hang. `_mark` only demotes a replica that
    # ERRORS, so every request to it burns the full timeout, and `stick_to` means the episodes
    # homed there cannot go anywhere else on their own. The ceiling is set just above the
    # longest real request, so a real request cannot trip it and a hang costs as little as
    # possible. This bounds the damage; it does not detect the freeze.
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--salvage", action="store_true", default=True,
                    help="after --max-draws, rewrite interface vocabulary out of the best "
                         "draft and keep it, marked `thought_salvaged`. NEVER applied to a "
                         "leak: an oracle quote or a post-hoc number is refused however many "
                         "draws were spent")
    ap.add_argument("--no-salvage", dest="salvage", action="store_false")
    ap.add_argument("--fill", action="store_true",
                    help="re-run only the turns that came back empty in --out, keeping every "
                         "thought already written. Resume is per episode and cannot reach "
                         "them.")
    ap.add_argument("--fill-from", default=None,
                    help="with --fill: a JSON {episode id: [turn, ...]} of turns to redo even "
                         "though they are not empty, e.g. turns a factual audit flagged. "
                         "Repairs those without paying for the turns that were already true")
    ap.add_argument("--max-draws", type=int, default=8,
                    help="hard cap on draws for ONE turn. Past --samples the temperature is "
                         "raised, because a model repeating the same rejected sentence needs "
                         "a different sample rather than another identical try. An empty turn "
                         "is not a local loss: later turns are written knowing it, so a hole "
                         "propagates through the rest of the episode")
    ap.add_argument("--samples", type=int, default=2,
                    help="draws per TURN. A turn whose only draw leaks poisons every turn "
                         "after it, because they are written knowing it")
    ap.add_argument("--full-boards", type=int, default=4,
                    help="how many recent observations to replay in full; older ones are "
                         "folded to a placeholder. Their thoughts and actions are kept, "
                         "which is what continuity needs")
    ap.add_argument("--factcheck", action="store_true", default=True,
                    help="reject drafts whose claims the turn's own facts contradict: a q "
                         "appeal naming the wrong candidate, a below-cut-off candidate taken "
                         "as sound, a named reaction no template reproduces, a ring or group "
                         "word no molecule on the turn matches, an unpriced fragment called "
                         "purchasable, a completeness claim with a piece still open. See "
                         "board/factcheck.py for the full list and what each one is checked "
                         "against")
    ap.add_argument("--no-factcheck", dest="factcheck", action="store_false")
    ap.add_argument("--fact-repair", type=int, default=2,
                    help="how many times a factually-wrong draft is handed back WITH the "
                         "number that refutes it and redrawn, before the blind-resample "
                         "budget is used. A false claim is a wrong belief rather than an "
                         "unlucky sample, so the same prompt at a higher temperature returns "
                         "the same sentence; a repair round is not charged as a draw")
    ap.add_argument("--fact-strict", action="store_true",
                    help="treat every factual violation as a rejection. Off by default: only "
                         "the FATAL ones (a false claim) reject, while a mostly-recycled "
                         "draft is recorded and kept, because a hole in the trace propagates "
                         "to every turn written after it")
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=260)
    ap.add_argument("--episode-workers", type=int, default=4,
                    help="episodes in flight. Turns WITHIN an episode are sequential and "
                         "cannot be parallelised -- that is the point of the design")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.urls = [u.strip() for u in a.base_url.split(",") if u.strip()]
    if not a.urls:
        raise SystemExit("--base-url is empty")
    print(f"  teacher {a.model} over {len(a.urls)} replica(s)", flush=True)
    # Said out loud rather than degraded quietly: a run whose factual gate silently did nothing
    # looks exactly like a run whose drafts were all true, and the two are not the same result.
    if a.factcheck and FC is None:
        print(f"  !! --factcheck requested but board.factcheck did not import ({_FC_ERR}); "
              f"claims will NOT be checked", flush=True)
    elif a.factcheck:
        print(f"  factcheck on: {len(FC.CHECKS)} checks, repair rounds {a.fact_repair}"
              + (", strict" if a.fact_strict else ""), flush=True)

    eps = load_episodes(Path(a.inp))
    print(f"  {len(eps)} episodes, {sum(len(e['turns']) for e in eps)} turns", flush=True)
    if a.limit:
        eps = eps[:a.limit]

    if a.print_prompt:
        i, k = (int(x) for x in a.print_prompt.split(":"))
        for m in turn_prompt(eps[i], k):
            print(f"\n=============== {m['role'].upper()} ===============\n")
            print(m["content"])
        return 0

    out = Path(a.out) if a.out else Path(a.inp).with_suffix(".reasoned.jsonl")
    done = set()
    if out.exists():
        for r in (json.loads(l) for l in open(out) if l.strip()):
            # Same identity rule load_episodes() uses. The replay writes no `id`,
            # so both sides fall back to the target -- which is unique per episode, since
            # multi-route expansion happens WITHIN one episode rather than across several.
            # Reading only `r["id"]` here collected {None} and re-did every finished episode.
            done.add(r.get("id") or r.get("target"))
        print(f"  resuming: {len(done)} episodes already written", flush=True)

    # --fill re-runs only the TURNS that came back empty, in episodes already written. Resume
    # works per episode, so an episode that finished with holes in it is never revisited and its
    # holes would be permanent. A hole is not a local loss either -- every later turn is written
    # with the earlier ones in context, so one missing thought is a discontinuity the rest of
    # the episode inherits. Filling is therefore the cheapest correctness pass available: it
    # costs the empty turns' draws and nothing else, and the thoughts already kept are left
    # alone.
    if a.fill:
        prev = {}
        for r in (json.loads(l) for l in open(out) if l.strip()):
            prev[r.get("id") or r.get("target")] = r
        # A turn can be a hole for two reasons and they need the same treatment. It came back
        # EMPTY, which this pass has always found by itself; or it came back FULL OF A FALSE
        # CLAIM, which only shows up when something checks, and which a factual audit writes
        # out as {episode id: [turn, ...]}.
        # Naming those here forces them back into the hole set, so one --fill run repairs both
        # kinds and the thoughts that passed are left alone.
        forced: dict = {}
        if a.fill_from:
            forced = json.load(open(a.fill_from))
            print(f"  fill-from {a.fill_from}: "
                  f"{sum(len(v) for v in forced.values()):,} turns named in "
                  f"{len(forced)} episodes", flush=True)
        todo, holes = [], 0
        for e in eps:
            r = prev.get(e["id"])
            if r is None:
                todo.append(e)
                continue
            tt = list(r.get("turn_thoughts") or [])
            # The hand-over lives in its own field, not in turn_thoughts, so append it before
            # deciding what is empty. Indexing tt by position without this marks the hand-over
            # pseudo-turn -- always the last one, always one past the end of turn_thoughts --
            # empty even when it was written, which would put EVERY episode in the hole set and
            # make each fill pass re-pay for the most expensive prompt in every one of them.
            if e["turns"] and turn_kind(e["turns"][-1]) == "final":
                tt.append(r.get("handover_thought") or "")
            for i in forced.get(str(e["id"]), ()):
                if 0 <= i < len(tt):
                    tt[i] = ""
            empty = [i for i in range(len(e["turns"])) if not (tt[i] if i < len(tt) else None)]
            if not empty:
                continue
            # Seed the turns that already have a thought, so the refill continues the same
            # line of reasoning instead of starting a new one mid-episode.
            for i, t in enumerate(e["turns"]):
                if i < len(tt) and tt[i]:
                    t["thought"] = tt[i]
            e["_fill"] = set(empty)
            holes += len(empty)
            todo.append(e)
        print(f"  fill: {len(todo)} episodes carry {holes:,} empty turns", flush=True)
        # Write to a SIDE FILE and swap it in at the end, never truncating the real output up
        # front. Truncating is unsafe in exactly the case that matters: a full sweep puts every
        # episode in `todo`, so the file is reduced to almost nothing until the pass completes,
        # and a pass killed midway destroys the other episodes' reasoning. The side file
        # starts with the episodes this pass is NOT touching, the refilled ones are appended to
        # it as they land, and the original is only replaced once the pass is done.
        keep = [r for k, r in prev.items() if k not in {e["id"] for e in todo}]
        out_final, out = out, out.with_suffix(out.suffix + ".fill")
        with open(out, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  fill: writing to {out.name}, swapped in when the pass completes", flush=True)
        prev = None                                # released; the side file holds it now
    else:
        out_final = None
        todo = [e for e in eps if e["id"] not in done]
    # Longest first. Episode cost is turns x draws and the tail is heavy. Handing the longest
    # out last leaves one worker grinding a long chain while the others sit idle, so the
    # makespan is the tail rather than the mean; handing them out first fills the idle slots
    # with short episodes instead.
    todo.sort(key=lambda e: -len(e.get("turns") or []))

    import concurrent.futures as cf
    lock = threading.Lock()
    agg = collections.Counter()
    t0 = time.time()
    with open(out, "a") as fh:
        with cf.ThreadPoolExecutor(max_workers=a.episode_workers) as ex:
            futs = {ex.submit(reason_episode, e, a): e for e in todo}
            for fut in cf.as_completed(futs):
                # as_completed, not ex.map: map yields in SUBMISSION order, so one slow
                # episode buffers every finished episode behind it. Nothing reaches the file
                # until it lands, which both hides progress and makes --resume useless after
                # a kill -- the completed work is in memory only.
                try:
                    ep = fut.result()
                except Exception as exc:                                   # noqa: BLE001
                    print(f"  ! episode {futs[fut].get('id')}: {exc}", flush=True)
                    agg["failed"] += 1
                    continue
                row = ep["row"]
                # Put the thoughts back where harmony reads them: the analysis channel of
                # each assistant call. Rewriting the row here rather than re-rendering keeps
                # this script independent of the board module.
                # The hand-over pseudo-turn is last and is NOT a board turn; its thought
                # travels in its own field so turn_thoughts stays index-aligned with the
                # board's turns.
                tl = ep["turns"]
                if tl and turn_kind(tl[-1]) == "final":
                    row["handover_thought"] = tl[-1].get("thought", "")
                    tl = tl[:-1]
                thoughts = [t.get("thought", "") for t in tl]
                row["turn_thoughts"] = thoughts
                row["reasoning_stats"] = ep.get("reasoning_stats")
                row["thought_checks"] = [t.get("thought_check") for t in tl]
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                agg["ep"] += 1
                agg["turns"] += len(thoughts)
                agg["kept"] += sum(1 for x in thoughts if x)
                if agg["ep"] % 5 == 0:
                    print(f"  {agg['ep']}/{len(todo)} episodes  "
                          f"{agg['kept']}/{agg['turns']} turns written  "
                          f"{agg['ep'] / max(time.time() - t0, 1e-9):.2f} ep/s", flush=True)
    if out_final is not None:
        # The pass finished; only now does the real output change.
        out.replace(out_final)
        out = out_final
    print(f"\n  wrote {out}", flush=True)
    print(f"  {agg['kept']}/{agg['turns']} turns carry a thought "
          f"({agg['kept'] / max(agg['turns'], 1):.1%})", flush=True)
    print("  re-render with the episode replay's --analysis text "
          "(reads turn_thoughts)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
