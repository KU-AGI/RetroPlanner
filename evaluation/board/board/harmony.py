#!/usr/bin/env python
"""The board as a gpt-oss Harmony conversation.

The episode is emitted as STRUCTURED messages -- role, channel, recipient,
content_type, content -- and never as Harmony token text.  The token layer is
version-dependent (0.0.8 renders the constrain marker as a literal " json"
after the channel, puts the recipient on the `assistant` header, and drops
`to=assistant` from the tool header), so hand-writing it would bake one
library version into the dataset.  tools/reaction-mcp/scripts/verify_harmony.py
renders these messages
with openai_harmony itself and round-trips the tool calls back through the
parser; that script is the only place that has to know the token format.

Shape of one turn:

    user                      the first board  (see first_board)
    assistant -> board_act    {"actions":[...]}          commentary, json
    functions.board_act       the next board             commentary
    ...
    assistant                 the route it hands over    final

`analysis` is left out by default.  The non-reasoning set comes first, and a
channel that carries {THINK} placeholders would train the model to emit them.
analysis="placeholder" emits the literal {THINK}; analysis="text" emits each
turn's own `Turn.thought`, which is what the teacher writes -- see
tools/reaction-mcp/scripts/traj_route_reasoning_board.py.  The reasoning is per TURN and continuous
across the episode: one episode expands several molecules over several routes,
and a turn's thought is written knowing the thoughts before it, so it can say
"the piece I set aside two turns ago" instead of restating the board.
"""
from __future__ import annotations

import importlib.util
import os

import json
from dataclasses import dataclass
from typing import Optional

from . import render as R
from .state import DEAD_REASONS

# ------------------------------------------------------------------ the tool
ACT_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            # The element shape is spelled out HERE, in prose, and not only in `items` below,
            # because gpt-oss's own chat_template.jinja renders an array of objects as
            # `actions: any[]` -- it does not descend into `items.properties`, so without this
            # the developer block would never say the call is `{"actions": [...]}` rather than
            # a bare action object. A property's `description` DOES survive the template, as
            # the comment above the field, so this is where the shape has to live.
            # OPENING SEVERAL MOLECULES IN ONE CALL is a move (the `open` stage of multi-route
            # expansion), stated outright rather than implied by "at most one per molecule":
            # the rank turn that follows a multi-open compares rival disconnections of several
            # molecules at one depth, and a student that does not generalise from the examples
            # needs an instruction to fall back on.
            "description": "Actions applied in the order given. At most one per molecule -- "
                           "so a single call may open several molecules at once, and usually "
                           "should: the pieces waiting on the board are the frontier, and "
                           "opening them together brings their disconnections back at the "
                           "same depth, where the next turn can rank rival cuts against each "
                           "other instead of committing to one branch and discovering later "
                           "that it was the wrong one. "
                           "Every call is wrapped: {\"actions\": [ACTION, ...]}, never a "
                           "bare action. Each ACTION is an object "
                           "{\"type\": \"open\"|\"rank\"|\"done\", \"mid\"?: string, "
                           "\"order\"?: number[], \"take\"?: number, "
                           "\"choices\"?: {molecule id: candidate number}} -- "
                           "open and rank take `mid`, rank also takes `order` and "
                           "optionally `take` (default 1), "
                           "done takes `choices` and no `mid`.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        # `dead` is COMMENTED OUT of the enum, not deleted: the board still
                        # implements it (state.Dead, _do_dead, the cascade and the reopen) and
                        # the labeller can still emit it, but no episode in the current corpus
                        # does, so declaring it teaches a tool call that never appears in a
                        # single example. Put it back in the enum and the description together
                        # -- a schema that allows an action the prose does not explain is
                        # worse than one that omits both.
                        # "dead: stop pursuing a molecule. "
                        #
                        # `analyze` is COMMENTED OUT on the same terms, and for a different
                        # reason than dead. The board still implements it end to end
                        # (state.Analyze, _do_analyze, render.evidence_block, parse), but the
                        # policy does not spend a turn on it: the structural and mechanistic
                        # facts it buys are given to the TEACHER as private input
                        # and reach the student only through the reasoning it learns to write.
                        # A student that cannot call the tool must not be shown the tool.
                        # "analyze: buy the bond change, reaction name and scaffolds for "
                        # "named candidates of a molecule already opened. "
                        "enum": ["open", "rank", "done"],
                        # `rank` says "apply the first `take`", not "apply the first": the
                        # schema below says `take` decides how many, and two descriptions of
                        # one field disagreeing is worse than either.
                        "description": "open: fetch a molecule's candidates -- one action "
                                       "per molecule, so several molecules can be opened in "
                                       "the same call. "
                                       "rank: order them and apply the first `take` of them. "
                                       "done: claim a route the board can make.",
                    },
                    "mid": {"type": "string",
                            "description": "Molecule id. Required except for done."},
                    # `candidates` and `mids` were analyze's arguments and leave with it.
                    # Kept here as a comment so putting analyze back is one edit in one place.
                    # "candidates": {mid: [candidate numbers]}  -- what to analyse, per molecule
                    # "mids":       [mid]                       -- molecules to analyse in full
                    "choices": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": "done only: molecule id -> candidate number, "
                                       "one entry for every molecule the route "
                                       "leaves to make. Each molecule named has to "
                                       "be open already and the candidate has to be "
                                       "on its menu; every piece the route ends on "
                                       "has to be purchasable.",
                    },
                    "order": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "rank only: candidate numbers, best first. "
                                       "The first `take` of them are applied now; "
                                       "the rest are the other routes this molecule "
                                       "can give, kept on the board to claim later.",
                    },
                    "take": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "rank only, default 1: how many of `order` to "
                                       "expand NOW. 1 opens one subtree under this "
                                       "molecule and leaves the rest for a later turn. "
                                       "Above 1 opens that many at once, so several "
                                       "branches are live together and the next turn "
                                       "chooses among them. Use it where the cuts you "
                                       "rank are genuinely different chemistry and both "
                                       "are worth having on the board; do not use it to "
                                       "hedge, because every subtree opened has to be "
                                       "finished or the route it belongs to is lost.",
                    },
                    # `reason` belongs to `dead`, which is commented out of the enum above.
                    # Left declared but unreachable is worse than either extreme: the model
                    # sees an argument for an action it cannot take. Renaming the key would
                    # not undeclare it, so it is commented out; restore it together with dead's
                    # enum entry.
                    # "reason": {"type": "string", "enum": list(DEAD_REASONS),
                    #            "description": "dead only: which condition holds. " + ...}
                },
                "required": ["type"],
            },
        }
    },
    "required": ["actions"],
}

TOOL_NAME = "board_act"
TOOL_DESC = ("Apply actions to the retrosynthesis board and receive the board that "
             "results. Nothing on the board moves without a call.")


# ------------------------------------------------------- the developer message
TASK = "Solve the retrosynthesis search by operating on the board."

SEMANTICS = """\
# Board semantics

A molecule is an OR: any one reaction under it closes it. A reaction is an AND:
every piece under it has to reach purchasable material, or the reaction dies and
takes its branch with it. A piece that is purchasable is closed on arrival.

- OPEN lists every molecule still awaiting a decision, every turn.
- open requests the candidate disconnections for a molecule that has no menu.
- rank names the candidates worth taking at a molecule and applies the first `take`
  of them now -- `take` defaults to 1, so `order` alone applies one and keeps the rest.
  The rest are the OTHER routes this molecule can give: each one the board has
  checked reaches purchasable material, so each is a route you can come back and
  claim. The board carries them by number only.
- done claims a route: a candidate number for every molecule the route leaves to
  make, as one object. The board makes any reaction in it that does not exist
  yet, so a route already sitting in the graph is one action rather than a walk
  back through it. Claiming does not end anything -- claim as many routes as the
  board can make, then hand over.
- Handing over is the last message, not a call. It ends the episode and names the
  routes you claimed, best first. It is valid only when nothing left on the board
  can beat them.

Molecules are shared: the same molecule reached two ways is one node with one id,
so two routes can differ at one reaction and agree everywhere else. That is why
claiming is cheap -- the second route is mostly the first route's graph.

Every call carries ONE argument, `actions`, and it is always a list:

    {"actions": [{"type": "open", "mid": "u6"}]}
    {"actions": [{"type": "open", "mid": "u6"}, {"type": "open", "mid": "n4"}]}
    {"actions": [{"type": "rank", "mid": "u6", "order": [0, 2], "take": 1}]}
    {"actions": [{"type": "rank", "mid": "u6", "order": [0, 2, 5], "take": 2}]}
    {"actions": [{"type": "done", "choices": {"u6": 0, "n4": 2}}]}

`take` ABOVE 1 EXPANDS SEVERAL SUBTREES UNDER ONE MOLECULE IN THE SAME CALL, so more than
one of its cuts is live at once and a later turn chooses among them instead of committing to
the first and finding out. It is an ordinary move, not a rare one.

A bare action object is not a call. This is stated here as well as in the tool
schema because the schema renders the list as `any[]` and says nothing about what
an element looks like.

Actions apply in the order you write them, and a later one may depend on an
earlier one: opening two pieces of the same reaction in one call, or ranking a
molecule whose parent you ranked a moment ago. At most one action per molecule
per call.

THE ONLY ACTIONS ARE `open`, `rank` AND `done`. There is no way to declare a
piece dead and none is needed: a branch that cannot close is left alone, and the
board stops offering it. Neither a `dead` action nor a DEAD section exists in the
tool schema, so reaching for one spends a call on a refusal.

Nothing moves without a call. A failed branch does not fall through to your
next-ranked candidate; it hands the molecule back and waits.

The board never repeats a menu you have already been shown. It carries the
rankings you declared and nothing else, so recovering what a candidate was
worth means going back to the turn it arrived in."""

NOTATION_HEAD = """\
# Board notation

<mol e3>SMILES</mol>   a molecule at first appearance; <mol e3/> afterwards. The
                       id is two characters, assigned once and never reused.
c0 .. c9               candidate numbers, fixed when the molecule is opened, in
                       the model's own order -- so a score is not monotone in the
                       index. Outside its own block a candidate is e3·c1.
depth D                how far below the target this piece sits. No ceiling is given:
                       how deep a route may go is not part of what you are judged
                       on, and the deeper a piece sits the more a cut has to earn.
rN  e3·c1  a of b closed
                       the ledger: which candidate was spent, how much of the AND
                       is closed, and the ordering you declared.
CLOSED                 every purchasable leaf banked so far, with its price. The
                       running bill of materials.
ROUTES                 a finished route, its weakest step, and its leaves."""

ROUTE_AXES_DOC = """\
ROUTES, and which of them are worth handing over
     The block carries three axes for every route it has claimed, all of them on the line:
     `p 5/6 pass`   how many of its steps clear the 0.05 filter. This is the plausibility
                    axis and it is the ONLY p figure on the line -- not the weakest step,
                    which answers a different question. `worst at r12` follows it when some
                    step fails and names WHICH step to work on: a pointer, not a measure.
     `rt 4/6 back`  how many of its steps the forward model recovers.
     `$.218`        what its leaves cost, on the same 0-1 scale as everything else.
     `6 steps` is printed too and is NOT one of the three. A shorter route can be beaten on
     all three and still be the one to run -- fewer steps is fewer chances to fail, and that
     is a real argument. It is simply not a dominance argument, so say it as what it is: `A is
     beaten by C on all three but takes two steps to C's four, which is the trade I would
     take`, not `A is the best`. The comparison below is over the three axes only.
     A route BEATS another when it is at least equal on all three and better on one, and the
     three figures are all on the line, so which routes are beaten is something the block lets
     you work out rather than something you have to take on trust.
     WHAT THAT COMPARISON IS OVER: the routes THIS BOARD HAS CLAIMED. A route being beaten by
     nothing here is not a statement that nothing better exists -- a route the search never
     found is not in the comparison, and neither is one you have not claimed yet.
     A beaten route is not a second answer. It is worse on every axis than one you are already
     handing over, so handing it over as well says the search found more than it did; two
     routes that read the SAME on all three are one answer wearing two labels. Hand over the
     ones nothing beats, order them by what each buys, and say what the ordering costs."""

ROUTE_MARK_DOC = """\
     The block does that comparison for you. `A ▲` is a route nothing else on the board beats;
     `E ≺C` is beaten, and by C. The header names the whole unbeaten set. The marks are the
     arithmetic above already done -- they add no fact the three figures do not carry, so a
     line's mark and its numbers can never disagree."""

SIGNAL_DOC = {
    "q": "q    the single-step model's own confidence in this disconnection. It is\n"
         "     what a conventional search would expand on. It cannot see what is\n"
         "     behind a precursor.",
    "p": "p    the reaction filter's plausibility; 0.05 is its cut-off. Below it the\n"
         "     filter says the reaction does not go. Above it, it says only that this\n"
         "     one step goes -- nothing about whether the precursors have anywhere to\n"
         "     go, and that is where most dead ends are.",
    "rt": "rt   the rank at which the forward model returns the product from these\n"
          "     precursors. 1 reads cleanly forwards; — means it could not be\n"
          "     recovered.",
}

COST_DOC = """\
*    this fragment is already purchasable. It says nothing about what it costs.
($N)
     What the fragment costs, on a 0-to-1 scale: `$.000` is the cheap end, where
     material is so cheap the price stops mattering, and `$1.000` is the dear end,
     past which differences stop being information. LOWER IS BETTER -- it is a COST, so
     `$.128` is cheap and `$.812` is dear. That is the opposite direction from `q`
     and `p`, where higher is better, and the same direction as `rt`, where rank 1
     beats rank 5.
     Carried on EVERY priced fragment, purchasable or not, and the two marks are
     independent: `k2*($.128)` is purchasable and cheap, `k2($.128)` is cheap and
     still has to be made, `k2*` is purchasable at an unknown cost.
     THE LINE ALREADY CARRIES THE TOTAL. `<c1 q.282 p.979 rt1 $.604>` is the cost
     of that whole cut, computed from its pieces; the per-piece figures are there
     so you can see where it comes from. Nothing needs adding up -- quote the
     figure the board prints, never one you worked out.
     The cost of a piece you still have to make is an ESTIMATE OF WHAT IT WOULD
     COST TO GET, i.e. what this cut costs if you stop and buy here. Open that
     piece and its precursors replace it, so the number moves -- down when the
     pieces are cheaper than the whole, up when they are not. Cite it as the cost
     of stopping here, never as the route's final cost.
     Cost is the deciding axis only where cost is what separated the leaders,
     which is rare: a route's cost is unknown unless every piece is both
     purchasable and priced."""

# Appended to COST_DOC only when the DECIDING form actually asks for `ordered by`.
# Kept out of COST_DOC itself so an arm served without that slot is not told to
# write a clause its form has no line for.
COST_ORDERED_SENTENCE = (" Say `ordered by: price` only where WHAT DECIDED IT\n"
                         "     says so.")

PRICE_DOC = """\
*    this fragment is already purchasable. It says nothing about the price.
($X)
     MolPrice's estimate of what the fragment costs, in USD per mmol. Carried on
     EVERY priced fragment, purchasable or not: MolPrice predicts a price for an
     arbitrary molecule, and what a piece you still have to make would cost is
     exactly the number a decision to keep expanding it is made against.
     The two marks are independent. `k2*($25)` is purchasable and cheap, `k2($25)`
     is cheap and still has to be made, `k2*` is purchasable at an unknown price.
     Dollars, not logs, so they ADD: a cut into $25 + $60 costs $85, and that is
     the number the ROUTES line reports when the route closes.
     A price on a piece you still have to make is an ESTIMATE OF WHAT IT WOULD
     COST TO GET, and it is what the cut costs if you stop and buy here. Open that
     piece and its own precursors replace it, so the number moves -- down when the
     pieces are cheaper than the whole, up when they are not. Cite it as the cost
     of stopping here, never as the route's final cost.
     Above $3.3M the estimate saturates and prints as $3.3M+: differences past
     that point are not information the model should be ordering on."""

DEFAULT_TIEBREAK = ("rt", "price")
"""What breaks a tie on the deciding axis, in order.

Mirrors DP's `--tiebreak`, minus q: q is the single-step model's confidence in
its own guess, not a property of the route, and the prompt already says it is
context rather than a reason.  Kept here so the developer message and the
labeller cannot describe different rules -- the sentence is GENERATED from this.
"""

# --- the reasoning form, one constant per part ------------------------------
# Split so an ablation can serve a SUBSET of the form without the prompt
# describing a block the model is not asked to write.  `analysis_form_text()`
# with its defaults rebuilds the full form byte-for-byte -- the arrival check
# (prefix_lockstep.py) is what holds that invariant.

_FORM_PREAMBLE = """\
# How to think, before each call

Your reasoning is a form with three parts, in this order. Each opens with its NAME ALONE ON A
LINE, in capitals, and runs until the next name. They answer three different questions, and
keeping them apart is what stops the thinking becoming a paragraph that restates the board.
"""

_FORM_MOLECULE = """\
MOLECULE
    What you are taking apart, and how far it is from the target. Build every line from the
    groups and ring systems you are given -- not from reading the SMILES yourself, which
    invents structure that is not there. Once a phrase for `target` has been established
    earlier in this episode, copy it exactly; do not re-derive it.
      target             what the target is, as a chemist names a compound. The same phrase all
                         the way down; it is the fixed point everything else is measured from.
      this piece         what the molecule in front of you is, named the same way.
      not in the target  the groups this piece carries that the target does NOT -- a bromide
                         put there for a coupling, an ester standing in for the acid, a
                         phosphonium salt. They are the handles the route installed on the way
                         down, and they are exactly what the next disconnection has to consume.
                         `none` when the piece has nothing the target lacks.
"""

_FORM_DISCONNECT = """\
DISCONNECTIONS OF <mid>
    What each cut IS, as chemistry -- ascending candidate number, no ranking on the row:
        c<n> | bond: ... | reaction: ... | what it does: ...
    Which cuts the call takes, and in what order, is said once, by DECIDING's route groups.
    These rows go in the board's own order, not the ranking's: a description block written in
    rank order announces the answer in its first line and everything after it is post-hoc.
    A row that argues for the winner stops describing -- so write the most GENERAL true account of each step,
    the one that would still be right if another candidate had been ranked first. Give a row
    to every candidate the call ranks and to at least one it passes over: a block describing
    only what was taken leaves nothing to compare against.
      bond          which bond the step makes and which it breaks, as an element pair and then
                    where it sits: "forms C-N at the carbonyl carbon; breaks C-Cl at the acyl
                    chloride". A bond can also change ORDER without being made or broken --
                    that is an oxidation, a reduction, a tautomerisation.
      reaction      the named reaction, when the step is one, and `none` when it is not.
      what it does  how the cut divides the molecule -- two comparable halves, one piece
                    carrying most of it, or a small piece shaved off the edge -- and whether
                    the ring skeleton survives it.
"""

_FORM_DECIDING_HD = """\
DECIDING <mid>
    Why that order and not another -- and never a claim about a precursor's own future
    disconnections, which have not happened and cannot be known. A LOOP, one group per route
    this call declares, two lines each. The call EXPANDS the first `take` of `order` on this
    turn -- several subtrees under one molecule, coming into being together -- and keeps the
    rest on the board for a later turn to claim. A group is arguing for a branch that will
    exist before the next observation, or for one the board is holding; those are different
    claims and the group must make the right one."""

_FORM_ROUTE_LINE = """\
        route <i> -- c<n>"""

_FORM_SLOT_DIV = """\
          diverging from: <routes already expanded> | <what they took> | <how THIS cut differs>
          diverging from: <routes already expanded> | same step, <the kind> | <why this one anyway>
          diverging from: none | <why there is nothing yet to diverge from>"""

_FORM_SLOT_ORD = """\
          ordered by: <measure> | <what it says that the number alone does not>"""

_FORM_DIV_DOC = """\
      `diverging from` points at EARLIER CONTEXT -- the steps this board has already expanded,
      which the observation lists as a reaction ledger (`r2  y7\u00b7c0`), each shown to you with
      the kind of reaction it was. Not at the other candidates of this same ranking: those are
      one decision seen three ways and the order already says which you prefer. Two things are
      worth saying and both are about the ledger: that this branch is worth opening TOO and how
      its chemistry differs, or that something of the same kind is ALREADY expanded -- said with
      `same step, <the kind>` -- and why this candidate is worth taking anyway. Claiming
      different chemistry where the board groups both under one kind is refused, and so is the
      reverse. Nothing expanded yet means `diverging from: none | <why>`."""

_FORM_CYCLE_GATE = """\
      BEFORE ranking anything, read each candidate's precursors against the pieces already
      ABOVE this molecule on its own branch -- the target, and every piece between it and
      here. Their SMILES are on the board: the target's from the first observation, each
      other from its own `<mol xx>` tag, and the chain is in the ledger (`rN  mid\u00b7cK` says
      reaction rN sits under mid). A candidate that puts one of them back makes that piece
      its own precursor; the board REFUSES that call and the search ends there. So this is
      not one consideration among five -- it is a gate that comes before the scores, it
      outranks every one of them, and a candidate that fails it is out however well it
      reads. Two cuts can look identical on p, rt, q and price and differ only in that one
      of them walks back up the route you already built. Do not rank such a candidate, and
      do not give it a DISCONNECTIONS row either: a row beside the real cuts says a refused
      step was on the menu. You are told which candidates these are.
        This is the most common way an episode ends on a refused call, which is why it
        is stated at length.
"""

_FORM_ORD_DOC = """\
      The measure on `ordered by` is one of five: `plausibility` (the filter on this step
      alone), `round-trip` (whether the forward model recovers the product from these
      precursors), `confidence` (the single-step model's own), `price`, or `structure` (the
      candidate's own bond change or named reaction, used only when none of the four numbers is
      what actually decided it). The first four have to name a measure on which the candidate
      put FIRST among what is still unclaimed actually comes first; one that reads the same on
      all of them ordered nothing. `structure` has no ranking to contradict, but it has to point
      at a fact the candidate carries. There is no measure for how good a candidate's eventual ROUTE
      will be -- that number does not exist until its pieces are opened. The candidate's `$`
      total is not that number: it is what the cut costs if you stop and buy its pieces here,
      and it moves once you open them. The last group has
      one cut left and nothing to order against, so it may leave `ordered by` off.
"""

_FORM_TAIL = """\
There is no fourth block. Do not add one, and do not write what the turn "means for the route"
past that line: the ranking is what the turn is for, and the ranking is above it."""


_FORM_COUNT = {1: "one part", 2: "two parts", 3: "three parts"}
_FORM_QUESTIONS = {1: "one question", 2: "two different questions",
                   3: "three different questions"}
_FORM_NEXT_ORDINAL = {1: "second", 2: "third", 3: "fourth"}
_SLOT_LINES = {0: "no line", 1: "one line each", 2: "two lines each"}

FORM_BLOCKS = ("molecule", "disconnections", "deciding")
FORM_SLOTS = ("diverging", "ordered")



def _cycle_gate(blocks, slots, standalone: bool) -> str:
    """The `own precursor` gate, fitted to the parts actually being served.

    Three things in it point at the rest of the form: its indentation (it sits
    under DECIDING), `among five` (the five `ordered by` measures) and the
    DISCONNECTIONS row it tells you not to write.  An arm served without those
    parts would otherwise be told about text it never sees -- the same failure the
    README calls out for a glossary that describes marks the board does not print.
    """
    t = _FORM_CYCLE_GATE
    if "ordered" not in slots:
        t = t.replace("not one consideration among five",
                      "not one consideration among the rest")
    if "disconnections" not in blocks:
        t = t.replace(
            "Do not rank such a candidate, and\n"
            "      do not give it a DISCONNECTIONS row either: a row beside the real cuts says a refused\n"
            "      step was on the menu. You are told",
            "Do not rank such a candidate. You are told")
    if standalone:
        # it is no longer nested under a block header
        t = "\n".join(ln[6:] if ln.startswith("      ") else ln for ln in t.split("\n"))
    return t

def analysis_form_text(blocks=FORM_BLOCKS, slots=FORM_SLOTS) -> str:
    """The `How to think` section, with only the parts asked for.

    `blocks` selects MOLECULE / DISCONNECTIONS / DECIDING; `slots` selects the two
    lines under a DECIDING route group.  Defaults rebuild the full form exactly,
    so a checkpoint keeps the message it was trained on.

    THE CYCLE GATE IS NOT PART OF THE FORM.  It is a rule about which candidates may
    be ranked at all -- a cycle candidate is refused by the board and ends the episode --
    so it is emitted whether or not DECIDING is, and an arm without DECIDING would
    otherwise be handicapped for a reason that has nothing to do with its reasoning.
    Its POSITION inside DECIDING is kept when DECIDING is present, so the full form
    stays byte-identical.
    """
    blocks = tuple(b for b in FORM_BLOCKS if b in blocks)
    slots = tuple(s for s in FORM_SLOTS if s in slots)
    if not blocks:
        raise ValueError("analysis_form_text needs at least one block")
    n = len(blocks)

    pre = _FORM_PREAMBLE
    if n != 3:
        pre = pre.replace("three parts", _FORM_COUNT[n])
        pre = pre.replace("three different questions", _FORM_QUESTIONS[n])
    if n == 1:
        # "Each opens with its NAME ... until the next name" needs a next name.
        pre = "\n".join(pre.split("\n")[:2] + [
            "Your reasoning is a form with one part.  It opens with its NAME ALONE ON A LINE,",
            "in capitals.", ""])

    parts = [pre]
    if "molecule" in blocks:
        parts.append(_FORM_MOLECULE)
    if "disconnections" in blocks:
        parts.append(_FORM_DISCONNECT)
    if "deciding" in blocks:
        head = _FORM_DECIDING_HD
        if len(slots) != 2:
            head = head.replace("two lines each", _SLOT_LINES[len(slots)])
        parts.append(head)
        parts.append(_FORM_ROUTE_LINE)
        if "diverging" in slots:
            parts.append(_FORM_SLOT_DIV)
        if "ordered" in slots:
            parts.append(_FORM_SLOT_ORD)
        if "diverging" in slots:
            parts.append(_FORM_DIV_DOC)
        parts.append(_cycle_gate(blocks, slots, standalone=False))
        if "ordered" in slots:
            parts.append(_FORM_ORD_DOC)
    else:
        parts.append(_cycle_gate(blocks, slots, standalone=True))

    tail = _FORM_TAIL
    if n != 3:
        tail = tail.replace("fourth", _FORM_NEXT_ORDINAL[n])
    parts.append(tail)
    return "\n".join(parts)


ANALYSIS_FORM = analysis_form_text()


RULES = """\
# How a route is judged

A route is worth its weakest step, and a minimum never rises above its lowest
term: a candidate scoring below your best route's weakest step cannot produce a
better route, whatever is behind it. The converse is not true -- a high score is
permission to look, not evidence that anything is there.

To improve a route, work on the step that IS the weakest. Everything else is
slack. A solved route is a report, not an ending: stopping with an unopened piece
still worth opening is the expensive mistake, not the safe one.

The routes you hand over are judged on three things together: how much of each
route clears the plausibility cut-off, how much of it the forward model recovers,
and what its leaves cost. Two routes that differ on those are two answers and
both are worth claiming; two that agree are one answer wearing two labels."""


def developer_instructions(style: R.RenderStyle = R.STYLE,
                           deciding: Optional[str] = None,
                           tiebreak: tuple = DEFAULT_TIEBREAK,
                           analysis_form: bool = False,
                           form_blocks: tuple = FORM_BLOCKS,
                           form_slots: tuple = FORM_SLOTS) -> str:
    """Task, semantics, notation -- with a glossary of the signals ACTUALLY shown.

    The glossary is generated from style.signal_order so the prompt cannot
    describe an axis the board does not render.  `deciding` names the one signal
    the ranking is scored on -- as ADMISSIBILITY, not as the only preference:
    below the cut-off a candidate cannot carry a route at all, and among
    candidates that tie there `tiebreak` decides.  Saying "only p decides" was
    wrong in both directions: it overstated p (ties are broken on rt and price)
    and it understated the rest (the answer is scored on three axes, not one).
    """
    shown = [k for k in style.signal_order if k in SIGNAL_DOC]
    lines = [TASK, "", SEMANTICS, "", NOTATION_HEAD, "",
             "# The signals", ""]
    lines += [SIGNAL_DOC[k] for k in shown]
    # BOARD_COST_NORM prints a normalised 0-1 cost, and PRICE_DOC describes dollars that add.
    # Serving the wrong one puts a glossary beside a board it does not describe, and traces
    # then cite numbers that are on no screen. The screen and its description move together.
    _wants_ordered = analysis_form and "ordered" in form_slots
    if os.environ.get("BOARD_COST_NORM") == "1":
        lines.append(COST_DOC + (COST_ORDERED_SENTENCE if _wants_ordered else ""))
    else:
        lines.append(PRICE_DOC)
    # Only where the board prints them. A message describing marks that are not on the screen
    # teaches the model to look for something it will never see, which is the same failure as
    # a message that omits marks the screen does carry: serve the message the model was
    # trained on.
    if os.environ.get("BOARD_ROUTE_AXES") == "1" or os.environ.get("BOARD_ROUTE_FRONT") == "1":
        lines.append(ROUTE_AXES_DOC)
    if os.environ.get("BOARD_ROUTE_FRONT") == "1":
        lines.append(ROUTE_MARK_DOC)
    if deciding:
        others = [k for k in shown if k != deciding and k not in tiebreak]
        ties = [k for k in tiebreak if k in shown or k == "price"]
        say = [f"{deciding} decides whether a candidate is worth taking at all: "
               f"below the cut-off it cannot carry a route, whatever else it "
               f"scores."]
        if ties:
            say.append(f"Among candidates that tie on {deciding}, "
                       + " then ".join(ties) + " decide, so the ranking you "
                       "declare is not a " + deciding + "-ranking alone.")
            if "price" in ties and _wants_ordered:
                # Stated as its own sentence because the clause above is not enough.
                # Prices within a menu differ widely, but once price is shown on every
                # fragment it no longer doubles as the closure argument ("every piece is
                # buyable, so this closes"), which leaves price with no occasion of its
                # own in a trace. This gives it one.
                say.append(f"When two cuts read the same on {deciding} and on "
                           "round-trip, price is the axis to NAME -- not a remark to "
                           "add under another one. A menu's cheapest and dearest cut "
                           "usually differ several fold, so `ordered by: price` with "
                           "the two dollar figures is a real ranking there, and "
                           "picking the dearer one without saying why is not. That a "
                           "cut's pieces are all purchasable is a DIFFERENT argument "
                           "-- it closes the branch -- and belongs under the axis that "
                           "actually decided, not under price.")
        if others:
            say.append(f"{', '.join(others)} is context you may reason with and "
                       "must not contradict: if you name it as your reason, the "
                       "candidate you take has to be the one it points at.")
        lines += ["", " ".join(say)]
    lines += ["", RULES]
    if analysis_form:
        lines += ["", analysis_form_text(form_blocks, form_slots)]
    return "\n".join(lines)


# ------------------------------------------------------------------- messages
@dataclass
class HarmonyEpisode:
    system: dict
    developer: dict
    messages: list[dict]

    def as_list(self) -> list[dict]:
        return [self.system, self.developer] + self.messages


def episode_messages(turns, board, style: R.RenderStyle = R.STYLE, handover_think=None,
                     analysis: str = "none", think: str = "{THINK}",
                     first_board: str = "user", final: str = "route",
                     done: str = "final", deciding: Optional[str] = None,
                     reasoning: str = "low") -> HarmonyEpisode:
    """Turns -> Harmony messages.

    first_board="user"  the opening board is a user message.  A tool message with
                        no call before it is malformed Harmony, and the opening
                        board has no call before it -- this is the honest shape.
    first_board="tool"  emit an explicit board_state call for turn 0 instead, so
                        that every observation is a tool result.  Costs one call
                        and needs the second tool declared.

    done="final"        `terminate` is NOT sent as a tool call: the last turn is the
                        assistant's final message.  In Harmony a <|call|> is a
                        promise that a tool result follows, so a done call at the
                        end of an episode trains a dangling call -- at inference
                        the server waits for a result that never comes.  Choosing
                        to answer instead of calling again IS the stop decision,
                        and it is supervised like any other turn.
    done="call"         keep {"type":"terminate"} as a call.  Only coherent if the
                        board answers it with a closing observation.
    """
    sysmsg = {"role": "system", "reasoning": reasoning}
    devmsg = {"role": "developer",
              # The reasoning form is described to the AGENT whenever the episode carries an
              # analysis channel. It was described only in the teacher's prompt, which is where
              # the traces come FROM and not where they are used: at inference the student sees
              # bare `MOLECULE` and `DECIDING ja` headers and has to infer the schema from
              # examples. A form the reader has not been told the shape of is a form it can
              # only imitate.
              "content": developer_instructions(style, deciding=deciding,
                                                analysis_form=analysis != "none"),
              "tools": [{"name": TOOL_NAME, "description": TOOL_DESC,
                         "parameters": ACT_SCHEMA}]}
    out: list[dict] = []
    for i, t in enumerate(turns):
        if i == 0 and first_board == "user":
            out.append({"role": "user", "content": t.env})
        else:
            out.append({"role": "tool", "name": f"functions.{TOOL_NAME}",
                        "channel": "commentary", "content": t.env})
        if analysis == "placeholder":
            out.append({"role": "assistant", "channel": "analysis", "content": think})
        elif analysis == "text":
            # Per-turn reasoning, carried ON the turn rather than passed as one string, so it
            # cannot drift out of alignment with the turn it explains. A turn with no thought
            # emits no analysis message at all: an empty analysis channel would train the
            # model to open the channel and say nothing, and a placeholder would train it to
            # emit the placeholder. Skipping is the honest shape for an unwritten turn.
            th = getattr(t, "thought", None)
            if th:
                out.append({"role": "assistant", "channel": "analysis", "content": th})
        payload = R.format_act_json(t.actions)
        # Only `terminate` becomes the final message.  `done` is a claim and stays
        # a tool call: the board answers it with the route it registered, and the
        # episode carries on.
        is_stop = all(a.get("type") == "terminate" for a in payload["actions"])
        if is_stop and done == "final":
            continue        # the final message below IS this turn
        out.append({
            "role": "assistant", "channel": "commentary",
            "recipient": f"functions.{TOOL_NAME}", "content_type": "json",
            "content": json.dumps(payload, ensure_ascii=False),
            "supervised": t.supervised,
        })
    if final == "route" and board.routes:
        # Every claimed route is handed over, ranked by the deciding axis, over
        # one drawing of the graph they share.
        def weakest(rt):
            vals = [R.signals_of_rxn(board.rxns[r], style)[1]
                    for r in R._route_rxns(board, rt.root_rxn, rt)]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else -1.0

        chosen = max(board.routes, key=weakest)
        # The hand-over gets its own analysis channel when one was written for it. It is the
        # only free-text message in the episode, so training it without a reason teaches the
        # listing's format and nothing about why the routes are in that order.
        if handover_think:
            out.append({"role": "assistant", "channel": "analysis",
                        "content": handover_think})
        out.append({"role": "assistant", "channel": "final",
                    "content": "\n".join(R.handover_block(board, chosen, style)),
                    "supervised": True})
    return HarmonyEpisode(sysmsg, devmsg, out)


def to_chat_messages(ep: HarmonyEpisode) -> list[dict]:
    """The episode as `messages` for apply_chat_template -- the training shape.

    openai/gpt-oss's own chat_template.jinja builds the system message itself
    (identity, knowledge cutoff, date, `Reasoning: <effort>`) and takes the
    DEVELOPER instructions from `messages[0].content` when that first message has
    role developer or system.  The tool namespace comes from the separate `tools`
    kwarg, not from a message.  So the list starts with one developer message and
    the row carries `tools` and `reasoning_effort` alongside it:

        tok.apply_chat_template(row["messages"], tools=row["tools"],
                                reasoning_effort=row["reasoning_effort"],
                                add_generation_prompt=False, tokenize=False)

    Field-by-field the template wants different names from Harmony's own:

      * a tool call is `tool_calls[0]` (`.function` unwrapped if present) and its
        `arguments` go through `|tojson`, so arguments must be an OBJECT -- a JSON
        string would double-encode into a quoted string.
      * analysis goes in `thinking`, never in `content` with channel tags; the
        template raises if it finds a <|channel|> tag in either field.
      * `content` and `thinking` together on a tool-call message is an error.
      * the final assistant message is plain `content`, and the template
        terminates the LAST one with <|return|> so the model learns to emit it.

    One thing it does that Harmony does not: it renders a tool result as
    `content|tojson`, html-safe, so the board's `<c0 ...>` becomes `\u003cc0 ...`
    and every newline an escape -- more tokens, and not the text vLLM feeds the
    model at inference.  `harmony_messages` on the same row is the vLLM-identical
    form; see tools/reaction-mcp/scripts/verify_harmony.py --compare-hf.
    """
    return [{"role": "developer", "content": ep.developer["content"]}] + \
        _assistant_shape(ep)


def _assistant_shape(ep: HarmonyEpisode) -> list[dict]:
    """The same episode in the shape openai/gpt-oss's chat_template.jinja wants.

    The template reads different fields from the ones Harmony itself uses:

      * a tool call comes from `tool_calls[0]` (`.function` unwrapped if present)
        and its `arguments` go through `|tojson`, so arguments must be an OBJECT.
        Handing it the JSON string would double-encode into a quoted string.
      * analysis goes in `thinking`, never in `content` with channel tags -- the
        template raises if it finds a <|channel|> tag in either field.
      * content + thinking together on a tool-call message is an error.
      * the final message is plain `content` and the template terminates the LAST
        one with <|return|> instead of <|end|>.

    And one thing it does that Harmony does not: it renders a tool result as
    `content|tojson`, html-safe.  The board's `<c0 ...>` becomes `\u003cc0 ...`
    and every newline an escape -- more tokens, and not what vLLM feeds the model
    at inference.  If you train through this template, patch that branch or train
    on pre-rendered text instead.  See tools/reaction-mcp/scripts/verify_harmony.py
    --compare-hf.
    """
    out: list[dict] = []
    pending_think: Optional[str] = None
    for m in ep.messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "name": m["name"].split(".", 1)[-1],
                        "content": m["content"]})
        elif m.get("channel") == "analysis":
            pending_think = m["content"]
        elif m.get("recipient"):
            msg = {"role": "assistant",
                   "tool_calls": [{"type": "function", "function": {
                       "name": m["recipient"].split(".", 1)[-1],
                       "arguments": json.loads(m["content"]),
                       "content_type": m.get("content_type", "json")}}]}
            if pending_think is not None:
                msg["thinking"] = pending_think
                pending_think = None
            msg["supervised"] = m.get("supervised", True)
            out.append(msg)
        else:
            msg = {"role": "assistant", "content": m["content"]}
            if pending_think is not None:
                msg["thinking"] = pending_think
                pending_think = None
            out.append(msg)
    return out


def preview(ep: HarmonyEpisode) -> str:
    """A readable, NON-canonical dump.  For eyeballing only -- see the docstring."""
    out = []
    for m in ep.as_list():
        head = m["role"]
        if m.get("recipient"):
            head += f" to={m['recipient']}"
        if m.get("name"):
            head = m["name"]
        ch = f"<|channel|>{m['channel']}" if m.get("channel") else ""
        if m.get("content_type"):
            ch += f" {m['content_type']}"
        body = m.get("content", "")
        if m["role"] == "developer":
            body = "# Instructions\n\n" + body + "\n\n# Tools\n\n(namespace functions)"
        if m["role"] == "system":
            body = f"(system boilerplate)\n\nReasoning: {m['reasoning']}"
        end = "<|call|>" if m.get("recipient") else "<|end|>"
        out.append(f"<|start|>{head}{ch}<|message|>{body}{end}")
    return "\n\n".join(out)


# ------------------------------------------------------- the rendered text
# config/paths.py, loaded by file location under its own name (the analysis pipeline owns
# the module name `paths`).
_spec = importlib.util.spec_from_file_location(
    "rp_paths", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "config", "paths.py"))
_RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RP)
DEFAULT_TEMPLATE_GLOB = os.path.join(
    _RP.CACHE, "huggingface", "hub", "models--openai--gpt-oss-120b",
    "snapshots", "*", "chat_template.jinja")


def find_chat_template(pattern: str = DEFAULT_TEMPLATE_GLOB) -> Optional[str]:
    import glob
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


class ChatRenderer:
    """apply_chat_template for gpt-oss, with the date pinned.

    The template is compiled with TRANSFORMERS' own jinja environment, not a
    plain one.  A plain `jinja2.Environment` gets two things wrong and both
    corrupt the text silently:

      * its `tojson` returns `Markup`, so concatenating the tool header with the
        json'd body escapes the header -- `<|channel|>` comes out as
        `&lt;|channel|&gt;` and the episode is no longer Harmony at all;
      * its default json policy is `sort_keys=True`, which reorders the action
        object to `{"mid": ..., "type": ...}` while the Harmony path keeps the
        order it was written in -- the same call, two different strings.

    So: compile through transformers when it is importable (identical to what
    `tokenizer.apply_chat_template` does), and fall back to an environment
    configured the same way.  `strftime_now` is supplied here rather than left to
    transformers, because a template that stamps today's date renders a different
    dataset tomorrow.
    """

    def __init__(self, template_path: Optional[str] = None,
                 model: str = "openai/gpt-oss-120b",
                 date: str = "2024-06-01"):
        self.date = date
        src = None
        try:
            from transformers import AutoTokenizer
            src = AutoTokenizer.from_pretrained(model).chat_template
        except Exception:
            path = template_path or find_chat_template()
            if path is None:
                raise RuntimeError("no tokenizer and no chat_template.jinja found")
            src = Path(path).read_text()
        self.tmpl = self._compile(src)

    @staticmethod
    def _compile(src: str):
        try:
            from transformers.utils.chat_template_utils import _compile_jinja_template
            return _compile_jinja_template(src)
        except Exception:
            import json as _json

            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            env = ImmutableSandboxedEnvironment(
                trim_blocks=True, lstrip_blocks=True,
                extensions=[jinja2.ext.loopcontrols])
            env.filters["tojson"] = lambda v, **kw: _json.dumps(v, **kw)
            env.globals["raise_exception"] = _raise
            return env.from_string(src)

    def render(self, messages: list[dict], tools: list[dict],
               reasoning_effort: str = "low") -> str:
        return self.tmpl.render(
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools],
            reasoning_effort=reasoning_effort,
            add_generation_prompt=False,
            strftime_now=lambda fmt: self.date,
            raise_exception=_raise,
        )


def _raise(msg):
    raise AssertionError(msg)
