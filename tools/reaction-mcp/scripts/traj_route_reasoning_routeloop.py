#!/usr/bin/env python
"""Write the analysis channel of a board episode to a FORM, one rank turn at a time.

`traj_route_reasoning_board.py` asks the teacher for a paragraph and reads the paragraph back
for defects. That works, and it costs: every check has to find the claim in free prose before it
can decide whether the claim is true, so a wrong reaction name and a wrong skeleton claim are
both "somewhere in the paragraph". This module asks for the same argument in slots, so a
violation has an address -- which is also what lets a draft be scored per MOLECULE rather than
all-or-nothing.

THE FORM.  Three blocks, and the third is a LOOP.

    MOLECULE
    target: ... / this piece: ... / not in the target: ...

    DISCONNECTIONS OF <mid>
    c<n> | bond: ... | reaction: ... | what it does: ...

    DECIDING <mid>
    route <i> -- c<n>
      diverging from: <what it is not repeating> | <what that took> | <how this one differs>
      ordered by: <measure> | value: <this cut's reading> | <what it means here>

DISCONNECTIONS carries NO ranking and goes in the board's own order, ascending candidate number.
The ranking is said once, by the route groups, and a description block written in rank order
would put the answer in its first line -- the reasoning is generated BEFORE the tool call, so a
block ordered by the answer means the answer was already decided and everything after it is
post-hoc. The board's order is one the model is already looking at, so it gives nothing away.

DECIDING is a loop because a rank call does not make one decision. It ranks several cuts and
EXPANDS the first `take` of them on this turn -- `take` is a field on the action, added with
this form -- so several subtrees come into being together and the next turn chooses among their
pieces. Groups after `take` are ranked and held. Which is which is printed in the support block,
never inferred.

`diverging from` is the slot that makes a second route worth declaring, and WHICH LEVEL it may
cite is decided by the board, not by preference:
  * a step already on the board's reaction ledger is the same KIND as this cut -> cite that
    ledger line and say so.
  * nothing on the ledger relates -> cite the sibling candidates and say what makes them
    different BETS. They usually are: a multi-candidate group typically ranks more than one
    reaction family.
Left free the slot takes whichever is easier to write, and on the turns where nothing relates
the ledger form degrades into "these two reactions differ" -- almost always true, and therefore
saying nothing.

`value:` exists because a number in free text has no stated axis and no stated candidate, so
whether it is RIGHT cannot be decided, and checking numbers inside prose flags true sentences
as false. Positional, the same figure is a fact. The argument beside it carries no digits at all.

WHAT IS CARRIED OVER UNCHANGED.

The chain, the verify gate and the factual gate are the same design as the freeform module and
most of them are the same code, imported rather than copied:

  * CHAIN. Turn k is written with the turns before it in context and asked to CONTINUE, so the
    calls are sequential within an episode and parallel across episodes. What changed is only
    how much history is replayed: a monotonically growing one-line LEDGER of everything decided
    so far, plus the last `--history-turns` turns as (reasoning, decision) pairs. The ledger
    grows by appending, so the prompt prefix stays byte-identical turn to turn and vllm's
    prefix cache hits it; the freeform module's block-folded board replay carries a whole board
    per remembered turn to say what one ledger line says.
  * VERIFY. Register is still a property of the words: interface vocabulary, an oracle quote,
    an atom-map index, a depth ceiling, a post-hoc number. Those patterns are imported from
    `traj_route_reasoning_board` so there is one copy, and they are applied to block BODIES so
    a hit comes back with the block it fired in. On top of them sit the schema checks, which
    the freeform prompt could not have: blocks present and in order, rows well-formed, the
    `decides:` axis real, `<order>` a permutation of candidates that exist.
  * FACTCHECK. `board.factcheck.check` runs over the whole draft exactly as before -- same
    checks, same FATAL set, same repair message -- and its violations are then ATTRIBUTED to a
    block by re-running the safe subset per block. Alongside it run the checks the schema makes
    exact and the paragraph made hard: `<order>` against the call's own order, a `class:` field
    against the template tier that earned it, a `bond:` field against whether the mapping
    measured anything, `decides: q` against the argmax of q.
  * THE DRAW LOOP. Escalating temperature past `--samples`, repair rounds that hand the draft
    back with the correction rather than resampling blind, least-bad draft kept, salvage for
    register-only defects, resume, `--fill`, side-file swap.

WHAT IS DELIBERATELY DIFFERENT.

The DECIDING route groups restate the decision the tool call also carries. The freeform prompt
forbade that ("do not end with a choice line") to stop the reasoning collapsing into a header
for the call. Here it is required, because it is the cheapest exact check in the whole
pipeline: the worst freeform defect is prose arguing for one candidate while the call takes
another, and in prose it needs a clause-level heuristic to find. Against the route groups it
is a list comparison.

Only RANK turns get the schema. A rank turn is never mixed with another action type and
usually ranks a single molecule -- so the form addresses one molecule and carries its id, and a
multi-rank turn repeats the DISCONNECTIONS/DECIDING pair per molecule. `open`, `done` and the
hand-over keep the freeform briefs from `traj_route_reasoning_board` until their own schemas
are designed; `--kinds rank` writes only rank turns and `--seed-from` fills the rest from a
file already written, so a rank-only run still has an unbroken chain behind it.

Usage:
  # 1. episodes with evidence, from render_board_episode.py
  python scripts/render_board_episode.py --n 20 --analysis none --out eps.jsonl
  # 2. read the contract, then one turn's prompt, before spending anything
  python scripts/traj_route_reasoning_routeloop.py --print-schema
  python scripts/traj_route_reasoning_routeloop.py --in eps.jsonl --print-prompt 0:1
  # 3. prove the gates fire, offline, no teacher
  python scripts/traj_route_reasoning_routeloop.py --in eps.jsonl --selftest
  # 4. write the channel
  TEACHER_BASE_URL=http://127.0.0.1:8000/v1 TEACHER_MODEL=Qwen/Qwen3.8-27B \\
    python scripts/traj_route_reasoning_routeloop.py --in eps.jsonl \\
      --out eps_schema.jsonl --episode-workers 8 --samples 2
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where another module of that name could shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", str(Path(__file__).resolve().parents[3] / "config" / "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)
sys.path.insert(0, RP.BOARD)                          # the board/ package lives with the harness

# The freeform module is the shared half, not a fallback: the register patterns, the teacher
# transport, the episode reader and the open/done/hand-over briefs all live there and are used
# from here. Copying them would give the corpus two definitions of "interface vocabulary".
import traj_route_reasoning_board as LEGACY                            # noqa: E402
from board import render as R                                          # noqa: E402
"""Prices are formatted through the RENDERER, not re-derived here.

If the prompt formats price one way and the board another, the teacher cites the prompt's
figure, a number on no screen the model would ever see. One formatter means the two cannot
drift."""

try:
    from board import factcheck as FC
except Exception as _e:                                                # noqa: BLE001
    FC = None
    _FC_ERR = str(_e)

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception as _e:                                                # noqa: BLE001
    Chem = None

CUTOFF = 0.05                       # the reaction filter's, as the developer block states it


# ============================================================================ the schema
# FOUR BLOCKS, AND THE LINE BETWEEN THEM IS LEDGER vs ARGUMENT.
#
#   <frame>      where the search stands: the target, this molecule, and the difference
#   <cuts MID>   the ranked disconnections, one row each, in the order the call declares
#   <weigh MID>  which axis eliminated and which axis ordered, with the values
#   <why>        the one prose block -- what the three ledgers cannot say
#
# The first three are a LEDGER: fixed fields, phrases, every value checkable against the turn's
# evidence. The fourth is an ARGUMENT: the cross-turn judgement, the risk being taken, what a
# dead branch taught. Keeping them apart is not tidiness. A structured slot invites boilerplate
# -- `state:` and `delta:` become a house style within a few turns -- and prose that restates
# the slots doubles the tokens to say the same thing and drives the repeat ratio with it. So
# `<why>` is checked for restating the ledger, and the repetition check runs on `<why>` alone.
#
# There is no <order> block. It used to exist so the ranking could be compared with the call,
# and it was duplication: the <cuts> rows are already required to be the ranked candidates in
# the call's order, so the ROW ORDER is the decision and the same comparison is exact without a
# second copy. Candidates weighed and rejected belong in <why>, which is what prose is for.
BLOCKS = ("molecule", "disconnections", "deciding")
ADDRESSED = {"disconnections", "deciding"}
# `route` (the DP's own weakest-step value, `dp` in the evidence) was a fifth measure here and
# it is gone. It reads like the other four -- a number, four decimal places, "the search's own
# ordering" -- and that is exactly the problem: p, rt, q and ln$ are each a property of ONE step
# or ONE purchasable fragment that a real tool call returns right now. `dp` is the value of the
# best route all the way down to purchasable material, computed by recursing through menus the
# board has not opened yet. No live process can hand an agent that number at decision time;
# producing it requires having already solved the very search this turn is in the middle of. In
# substance it is the same object as the label the rest of this pipeline refuses to let a turn
# quote -- a number instead of a word, but the same oracle foresight. p/rt/q/ln$ explain almost
# every recorded ranking on their own; where only `route` explains one, it does so by telling
# the model the answer to a question it had not asked the board yet.
AXES = ("p", "rt", "ln$", "q")
# How each measure is NAMED in English -- the prompt's own words, and what the fixtures write.
AXIS_SAY = {"p": "plausibility", "rt": "the round-trip",
            "q": "the single-step confidence", "ln$": "the price"}
# The slot spelling of each, for the fixtures and for anything that has to WRITE one.
AXIS_TOKEN = {"p": "plausibility", "rt": "round-trip", "q": "confidence", "ln$": "price"}

# Word ceilings, per LINE, not per block. The form is keyword-first: a ledger line is a set of
# labelled phrases, not a paragraph with tags around it.
#   * A phrase is checkable where a sentence is inferable. "delta: benzyl bromide, aldehyde" is
#     two set-membership claims decidable against two SMILES; the same content inside "the piece
#     carries a benzyl bromide handle the target does not have, along with the aldehyde" is the
#     same two claims wrapped in words no check can use.
#   * A word FLOOR produces padding: a model asked for a fixed number of words about a routine
#     turn writes that many, mostly recycled sentences. Asked for a noun phrase, it writes the
#     noun phrase.
# `why` is the exception and gets the largest budget, because it is the block that is allowed to
# be an argument.
CAP = {"molecule": 25, "row": 55, "deciding": 45, "leaves": 30, "weighed": 3}

SCHEMA_RANK = """Your answer is a FORM. Four blocks, always in this order, each opened by its
NAME ALONE ON A LINE in capitals and running until the next name. There are no closing tags and
nothing outside the blocks -- no preamble, no closing line, no explanation of the form itself:

MOLECULE
target: ...
this piece: ...
not in the target: ...
DISCONNECTIONS OF <mid>
c<n> | bond: ... | reaction: ... | what it does: ...
DECIDING <mid>
route <i> -- c<n>
  diverging from: <a route already expanded> | <what it took> | <how this cut differs>
  ordered by: <measure name> | value: <this cut's reading> | <what it says that the number alone does not>

Every label and every value is ENGLISH a reader can follow without a legend. Write "splits it
into fragments of 34 and 14 heavy atoms", never "28 -> 34 + 14"; write "the plausibility filter
puts it below the cut-off", never "p .001 < .05". A value that has to be decoded is a value
nobody can check and nobody can learn from. Short is good; cryptic is not, and they are not the
same thing.

Every block is a LEDGER: labelled lines, each one a phrase, no hedging and no restating the
label inside the value. The reason clauses in DECIDING are the one place an ARGUMENT lives, and
they must not repeat the rows above them.

<mid> is the molecule this turn ranks, spelled exactly as the board spells it, on the header
line: `DISCONNECTIONS OF ja`. If the turn ranks more than one molecule, write one DISCONNECTIONS
and one DECIDING per molecule -- all the DISCONNECTIONS blocks together, then all the DECIDING
blocks -- and let MOLECULE cover both.


MOLECULE  --  WHAT IS BEING TAKEN APART, AND HOW FAR IT IS FROM THE TARGET.

    target:            what the TARGET is: its ring systems and the groups that decide how it
                       comes apart. The same phrase every turn of an episode; it is the fixed
                       point everything else is measured against.
    this piece:        what the molecule being ranked is, named the same way. If it IS the
                       target, say so.
    not in the target: the groups and ring systems that are in this piece and are NOT in the
                       target -- the handles this route installed on the way down: a bromide put
                       there for a coupling, an ester standing in for the acid, a phosphonium
                       salt. They are exactly what the next disconnection has to consume. A
                       group that is in both is not one of these; it is carried, and it belongs
                       on the `this piece` line. If there are none, write `none`.

Name chemistry, never measurements. "a benzyl bromide and an aldehyde", not "28 heavy atoms, 3
ring systems, Fsp3 0.11". The counts you are given are there so the phrase is TRUE.

BUILD THIS ONLY FROM THE DESCRIPTORS YOU ARE GIVEN -- the groups and ring systems listed, the
ring/heavy-atom/heteroatom counts, the Murcko scaffold. Do not read the SMILES yourself and name
what you see in it: you will name something that is not there. Every ring system and functional
group each molecule contains is listed for you, and a name you reach for instead of one on that
list is a guess -- toward the commoner word ("a stilbene" for a styryl compound, one side of
which is not even an aryl; "2-phenylethyl" for a 2-(naphthalen-2-yl)ethyl group), or toward a
substituent that simply is not there (a hydroxycyclopropyl, an azabicyclohexyl, a piperazine, on
a molecule with none of them). Do NOT invent a substituent, a stereocentre, a specific ring
fusion, or a specific heterocycle name that is not on the list you were given. When something is
in the molecule but not nameable from that list, describe it by what the descriptors DO say --
"a ring bearing two heteroatoms", "a halogenated ring", "an aliphatic substituent" -- rather than
guessing a specific name for it. This is the ONE line every later turn is handed back to copy
(see below), so a name invented here is not a one-turn mistake -- it becomes the episode's fixed
point, wrong, for every turn after it.
Ceiling %(molecule)d words per line.


DISCONNECTIONS OF <mid>  --  WHAT EACH CUT IS, AS CHEMISTRY.

    c<n> | bond: ... | reaction: ... | what it does: ...

One row per candidate, in the BOARD's order -- ascending candidate number, the order the
menu is printed in -- and NO ranking on it. Which cuts the call takes, and in what order, is
said once, by the route groups in DECIDING; printing it here as well turned these rows into an
argument for the winner, and a row that is arguing is a row that stops describing. So write each
one as the most GENERAL true account of the step it is. Writing them in the order the CALL
ranks them would put the answer in the first line of a block whose job is to describe, and
every sentence after it would be justifying a choice already made; the board's order is one
you are looking at, so it gives nothing away -- the bond change, the reaction class if
a template earns one, what it does to the skeleton -- the account that would still be right if
another candidate had been ranked first.

Give a row to every candidate the call ranks, and to at least one it passes over -- up to
%(weighed)d of those. You are shown every disconnection of this molecule grouped by the kind of
step it is, with the same facts behind each, and a block describing only what was taken leaves
nothing to compare against. Describe the ones that would have been the real alternative: the
same transformation from the other side, a different reaction class, the one whose leaf is
cheaper.

  bond:          the bond the step MAKES and the bond it BREAKS, as an element pair and then
                 where it sits. You are given both. The pair is measured (`C-Br`, `C-N`, `O-C`)
                 and so is the position -- each end of each changed bond comes with what that
                 end IS: "aromatic C", "carbonyl C", "Br (leaving group / halide)",
                 "boronic/boronate B", "amine N". Write the pair, then the position IN THOSE
                 TERMS:

                     bond: forms C-C between an aromatic carbon and the carbon on the boronic
                           ester; breaks C-Br at the aryl halide

                 THE ELEMENT PAIR IS NOT OPTIONAL. Every bond you mention carries one, written
                 with a hyphen -- `C-C`, `C-Br`, `O-C` -- and a field that says only where a
                 bond sits ("the bond at the benzylic position") names a place and no bond, so
                 there is nothing in it for the mapping to agree or disagree with. Pair first,
                 position after.

                     bond: forms C-N at the carbonyl carbon; breaks C-Cl at the acyl chloride
                     bond: the C-O bond changes order at the carbinol carbon
                     bond: not measured

                 A bond can also move in ORDER without being made or broken -- an oxidation, an
                 imine reduction, a tautomerisation -- and when it does you are told so. Say
                 "changes order" for it and do not guess a direction; such a step has nothing
                 formed and nothing broken, and calling it `not measured` would be false.

                 Do not read the position off the SMILES and do not invent it. It is the one
                 half of this field no check can refute, so a guess here is a guess that
                 survives into the corpus -- use the groups you are given. NEVER an atom-map
                 index like `C:13`: an index is an artefact of how the mapping ran and you will
                 not have it in front of you next time. If the mapping failed on a candidate the
                 whole field is exactly: not measured.
  reaction:      the named reaction, and ONLY when a template actually reproduces the step. When
                 it does not -- including when the groups fit a named reaction but the outcome
                 is not what that reaction gives -- the field is exactly: none. Not a hedge, not
                 a guess, not the name with a qualifier. A name you cannot earn is the claim a
                 reader carries to the next molecule.
  what it does:  how the cut divides the molecule, and what happens to the skeleton:

                     what it does: splits it into fragments of 21 and 8 heavy atoms, one piece
                                   carrying most of it; the ring skeleton is kept
                     what it does: two comparable halves of 15 and 14 heavy atoms; the ring
                                   skeleton changes

                 Both judgements are computed for you and both are COPIED, not decided. You are
                 told whether the cut is convergent (two comparable halves), unbalanced (one
                 piece carries most of it), a decoration (it shaves a small piece off the edge),
                 or no split at all (one precursor). And you are told whether the ring skeleton
                 survives -- which asks whether a precursor still carries the product's
                 ring-and-linker topology, and is not judgeable by eye from fragment sizes.
                 Where the precursors carry atoms the product never sees -- a phosphine, an
                 auxiliary, a leaving group -- you are told how many, and it is worth a clause,
                 because it is why the two fragment sizes add up to more than the product.
At most %(row)d words for `bond` and `what it does` together; the reaction name is quoted, not
written, and does not count.


DECIDING <mid>  --  WHAT WAS ELIMINATED, AND WHAT ORDERED THE REST.

NEVER ARGUE FROM WHAT YOU HAVE NOT LOOKED AT. Every measure named here is a property of ONE step
or ONE purchasable fragment, true right now -- not a claim about how a precursor's OWN
disconnections will turn out, because you have not seen them and neither has the agent. "nothing
behind it reaches purchasable material" and "its route is worth 0.35" both claim to know the
outcome of an exploration that has not happened. Say only what THIS candidate's own numbers show.

DECIDING is a LOOP, one group per route this call declares, and each group has exactly two
lines. The call does not make one decision. It ranks several cuts and EXPANDS the first `take`
of them on this turn -- `take` is a field on the call and you are told what it is -- so those
subtrees all come into being together and the next turn chooses among their pieces. The groups
after the first `take` are the cuts you rank and do not expand yet: they stay on the board for
a later turn to claim.

    route <i> -- c<n>
      diverging from: <routes already expanded> | <what they took> | <how THIS cut differs>
      diverging from: <routes already expanded> | same step, <the kind both are> | <why this candidate anyway>
      diverging from: none | <why there is nothing yet to diverge from>
      ordered by: <measure name> | value: <this cut's own reading, or none for structure> | <what it says that the number alone does not>

One group per ranked cut, in ranking order, no more and no fewer; the candidate after `--` is
that route's own cut, and the two lines come in the order shown. Which of them this call
EXPANDS is not for you to decide or to guess -- it is printed for you, and a group that is
being expanded now is arguing for a branch that will exist before the next observation, while
one that is not is arguing for a branch the board is keeping. Do not write that a cut is "for
later" when the call is expanding it, and do not write that one is being taken when it is
being kept.

`diverging from` is the whole reason this block is a loop, and what it points at is EARLIER
CONTEXT: the steps this board has already expanded, which the observation lists as a reaction
ledger -- `r2  y7\u00b7c0`, `r6  q4\u00b7c2`. You are shown that ledger with the kind of reaction each
of those steps was. It does NOT point at the other candidates of this same ranking: they are one
decision seen three ways, and the route order has already said which you prefer, so arguing
between them adds nothing.

WHICH of the two you owe is not a choice -- the board decides it, and it tells you per ledger
line whether that step is the same kind as a cut you are ranking:

  * a cut MARKED as the same kind as something already expanded -> say that, against that
    ledger line. It outranks anything about the other candidates: the board is about to spend
    a call repeating chemistry it already has, and that is the fact worth writing down.
  * no ledger line relates to this cut -> argue at the CANDIDATE level. Say what makes the cuts
    this call ranks different BETS from one another: `diverging from: c3, c7 | both are
    Suzukis needing a boronic acid | this one makes the same bond by Negishi, so it needs no
    boron partner`. A call that ranks more than one candidate usually ranks distinct kinds of
    step, and the ranking says which is preferred, never what makes them different.

Writing the candidate level where a ledger line matches, or the ledger level where none does,
is refused. Left free the slot takes whichever is easier, and on the turns where nothing
relates the ledger form degrades into "these two reactions differ" -- almost always true, and
therefore saying nothing.

There are two things worth saying, and both are about that ledger:

  * this branch is worth opening TOO. Name the route that went another way, name what it took,
    and say how this cut differs -- a different bond, a different class, the other end of the
    molecule. That is a claim that the board is exploring two real alternatives, not one idea
    twice.
  * something like this is ALREADY expanded. When the kind of step is the same as one already
    on the ledger, say so with `same step, <the kind>` -- and then why this candidate is still
    worth taking, or which other candidate you took because of it. You are told, per ledger
    line, when its kind matches a cut you are ranking, so this is never a guess.

Claiming a difference in chemistry where the board groups both steps under one kind is refused,
and so is the reverse. On a board where nothing has been expanded yet the honest line is
`diverging from: none | <why>`; writing one there invents a history the board does not have.

`ordered by` is on EVERY group, the last one included, and it is the OBJECTIVE that route is
worth keeping for -- not "the measure that puts it first among what is left". Route 1 leads
outright and names the measure it leads on. A later route is on the board because it wins
somewhere route 1 does not, and that is what its line says: name a measure this cut actually
BEATS the head on. When it beats the head on no number, say
`structure` and what the cut DOES that the head's does not. Two routes naming the same measure
at the same reading are one route written twice and are refused; that is the whole difference
between a set of routes and a ranking written three times.

NO FIGURES IN THE ARGUMENT. `value:` holds the number -- this route's own cut's reading on the
measure named, one figure, or `none` when the measure is `structure` and there is none. The
argument beside it carries no digits at all. This is not a style rule: a number written into
free text has no stated axis and no stated candidate, so whether it is RIGHT cannot be decided:
a round-trip line quoting a plausibility is indistinguishable from a wrong round-trip. In `value:` the same figure is a fact that can be
compared. Candidate labels (`c7`) and the measure's own name are not figures.

The measure named on `ordered by` is one of these five words and nothing else:

  plausibility   the reaction filter on THIS step, cut-off %(cutoff)s. Below it the step does
                 not go and no other score rescues it -- USUALLY. Above it, it says only that
                 this one step goes -- nothing about whether the precursors have anywhere to go.
  round-trip     the rank at which the forward model recovers the product from these
                 precursors, and it prints as one of THREE things, not two. `1` reads
                 cleanly forwards. `rt X` means the forward model RAN and did not recover
                 it. `rt ?` means nothing was asked and the board knows nothing. AN
                 ABSENCE IS NOT A FINDING: never write that a `?` cut "could not be
                 recovered" -- it has not been tried, and a sentence that treats the two
                 alike is false on the half of them that would recover. High plausibility
                 with `rt X` is the shape of a step that looks fine and leads nowhere;
                 high plausibility with `rt ?` is a step nobody has checked yet.
  confidence     the single-step model's own confidence: what a conventional search would
                 expand on, and it cannot see past a precursor. IT RANKS BELOW THE OTHER
                 THREE. plausibility, round-trip and price are the axes this work is scored
                 on; confidence is the model's opinion of itself and is not one of them. If
                 the hint below offers any of those three, name one of them and not this --
                 `confidence` is for the turns where it is the only number that separates
                 anything; it is a last resort.
  price          cost, as the board prints it: a `($X)` on each priced fragment and the cut's
                 own total in the signal group. Quote the figure on the screen, never one you
                 derived -- a `value: 2.91` cites nothing, and under a normalised cost the
                 pieces do not add up to the total.
                 A SMALL PRINTED GAP IS A LARGE REAL ONE. The figure is not dollars: it is the
                 log of the price, squeezed onto 0-to-1, so the whole range from about $12 to
                 about $99,000 per mmol fits between `$.000` and `$1.000`. Read the gaps on
                 that scale, not as if they were dollars:
                     .05 apart on screen  ~   1.6x in money
                     .10 apart           ~   2.5x
                     .26 apart           ~  10x
                     .50 apart           ~  90x
                 So `$.72` beside `$.46` is not a near-tie to be waved through on some other
                 axis -- it is ten times the money, and it is worth `ordered by: price` and
                 saying so. Treat two costs as level only when the printed figures are equal. Every priced fragment carries one,
                 purchasable or not, so the total is what the cut costs if you stop and buy its
                 pieces here. A cut whose every fragment is purchasable also CLOSES the piece
                 outright, which is a separate argument from its cost and is worth making even
                 when the number is high.
  structure      the candidate's own bond change or named reaction, when NONE of the four
                 numbers above is what actually decided it. THE HINT BELOW TELLS YOU WHETHER
                 THAT IS THE CASE: it prints the measures that put this candidate first, and
                 while that list is non-empty `structure` is refused -- a number did decide,
                 and naming the shape instead throws away the one thing a reader can check.
                 Do not reach for this slot as the easy answer. PRICE IS ONE OF THOSE NUMBERS:
                 if the cuts tie on plausibility and round-trip but their dollar totals
                 differ, a number did decide and `structure` is the wrong name for it.
                 The top cuts often tie on plausibility and round-trip, and then price
                 is the measure that decides. Most often on a candidate whose
                 plausibility reads low but whose bond change matches a well-precedented named
                 reaction, or whose split is otherwise the clean one, and the numbers alone
                 would make the taken candidate look like the wrong choice. Ground it in a fact
                 this row actually carries (its `reaction:` name, its `bond:` line) -- never in
                 an opinion with nothing behind it.

`ordered by` has to name a measure on which the candidate you put FIRST actually comes first.
You are told which measures do and which do not, so this is a choice between true answers, not
a guess: naming one that ranks another candidate ahead of the one taken contradicts the call
(this does not apply to `structure`, which has no ranking to contradict). A measure reading the
same on all of them ordered nothing and is also wrong, however true it is.

There is no measure here for "the route below this candidate is worth X" and there never should
be one you invent. That number does not exist for a piece nobody has opened -- producing it
would mean already having solved the disconnection you are being asked to reason about. Naming
`structure` is not that: it points at a fact already on this row, not at an outcome you cannot
see. If the taken candidate has neither a matching named reaction nor a measured bond either,
`structure` is not available and you are looking at a candidate none of the five measures
explains -- in that rare case, name plausibility and say plainly that it reads low here, rather
than inventing an argument.

The `|` slots are the whole point: they say which candidates, which reason, which number,
without anyone having to read a sentence to find out. Do not put the reason or the number in the
candidate slot, and do not name two measures on one line.

A route is worth its weakest step: a cut scoring below your best route's weakest step cannot
produce a better route, whatever sits behind it.
Ceiling %(deciding)d words for the reason on each line.

There is no block after this one. Do not write what the turn "means for the route" beyond this
line, do not carry a history forward, and do not name a risk: two such fields existed and both
had no honest null to take, so they were filled whether or not there was anything to say. What this
turn is for is the ranking; the ranking is above."""

REGISTER = """Rules that hold inside every block:

- Chemistry, not interface. The words `menu`, `candidate list`, `the list`, `offered set` and
`on screen` name the machinery you are reading, not the chemistry you are reasoning about, and a
trace built out of them teaches a reader to narrate a UI. Substitute, do not merely avoid: where
you would write "the list shows three amide cuts", write "three of the cuts break the amide".
Candidate numbers (c0, c2) are how a specific disconnection gets named and are fine.
- Never describe your own information state in terms of data fields. "I have no bond changes for
c4" is a sentence about a table. The row's bond field says `not measured`, and that is the whole
of what there is to say.
- Never mention which action is correct, that you were told anything, a route's final cost or
length, or anything about how the episode ends. You are inside it.
- The board gives a piece's depth and no ceiling, and it shows no call budget. Do not invent
either: no levels remaining, no calls to spare. Depth is only ever an argument about how much a
cut has to earn -- a piece sitting deep has less of a molecule left to simplify, so a cut that
merely shuffles substituents is worth less there than near the target.
- Every claim comes from this turn's board and its supporting information. There is no action
that fetches facts and nothing to look up."""

CORE_RANK = """You are an expert synthetic organic chemist writing the private reasoning of a \
retrosynthesis search agent, one turn at a time.

The agent operates a board. Several molecules are open at once, on several partly-built routes, \
and each turn it applies actions to some of them. You are writing the reasoning for ONE turn, \
and it is a RANK turn: the disconnections of a piece are in, and the turn declares which of them \
to take and in what order.

This is a continuation, not an essay. What you decided earlier is above you, with the reasoning \
you wrote for it. Build on it: do not restate the board, name the branch you are on when more \
than one is open, and refer back to what you set aside by what it was rather than by turn \
number.

%(SCHEMA)s

%(REGISTER)s"""


def system_rank(order_block: bool = True) -> str:
    """The system prompt for a rank turn. `order_block` is accepted and ignored -- see BLOCKS."""
    if NAIVE["on"]:
        return CORE_RANK % {"SCHEMA": SCHEMA_NAIVE, "REGISTER": REGISTER}
    schema = SCHEMA_RANK % {"molecule": CAP["molecule"], "row": CAP["row"],
                            "deciding": CAP["deciding"], "leaves": CAP["leaves"],
                            "weighed": CAP["weighed"], "cutoff": CUTOFF}
    return CORE_RANK % {"SCHEMA": schema, "REGISTER": REGISTER}


# ============================================================================ parsing
# One regex for the block, one for the row, one for a labelled line. All three are deliberately
# forgiving about WHITESPACE and strict about everything else: a teacher that indents its rows is
# writing the right thing in a slightly different shape, and rejecting that costs a draw for
# nothing. A teacher that writes `class = amide coupling` is writing a different shape, and that
# IS worth a redraw, because a field the parser cannot find is a field no check can look at.
# A block is a HEADER LINE and everything under it, up to the next header. There are no closing
# tags: with paired tags the teacher writes the content correctly and puts the slash on the
# wrong tag -- `</weigh ja>` where `</cuts>` then `<weigh ja>` belonged -- which the parser
# then reads as two missing blocks and a page of stray text, a rejection that is not a
# reasoning failure. A closing tag carries no information here: the next header ends the block,
# and a header is unambiguous because it is a bare keyword alone on its line.
_HEAD_RE = re.compile(r"^[ \t]*(MOLECULE|DISCONNECTIONS|DECIDING|WHY)"
                      r"(?:[ \t]+(?:OF[ \t]+)?([A-Za-z0-9_.\-]+))?[ \t]*$", re.M)
# A row now says WHAT IT IS as well as what it does: `rank 1` for a cut the call takes, in the
# position it takes it, and `weighed` for one described and then set aside.
#
# Without the role slot the rows WERE the ranking, and that had a cost the checks could not see:
# the answer sat at the top of the trace and everything below it was justification. A student
# reading a corpus of those learns to explain an order it is given, not to compare cuts and
# arrive at one. Giving the rejected candidates a row -- they have bond and reaction facts too
# -- turns the block into the comparison it was supposed to
# be, and `ruled out` into the pruning step that follows from it.
# No `rank N` / `weighed` role. The ranking is carried by DECIDING's route groups, and printing
# it here as well made DISCONNECTIONS an argument about which cut wins -- so the rows drifted
# into justifying the head instead of describing the chemistry. Without the role the block is
# what it should be: one general, reaction-level description per candidate, and whether a
# candidate is ranked is read off the CALL, which is where that fact actually lives.
_ROW_RE = re.compile(r"^\s*c(\d+)\s*"
                     r"\|\s*bond\s*:\s*(.*?)\s*"
                     r"\|\s*reaction\s*:\s*(.*?)\s*"
                     r"\|\s*what it does\s*:\s*(.*?)\s*$", re.I)
# A label may be several words. `delta:` and `gate:` were one word each and neither was English:
# a reader had to be told what they meant before the line said anything, which is the opposite of
# what a field is for. The cost of spelling them out is a looser key pattern and nothing else.
_KEYED_RE = re.compile(r"^\s*([a-z][a-z $-]*?)\s*:\s*(.*)$", re.I)
MOL_KEYS = ("target", "this piece", "not in the target")
# WHY is fielded too. Left as free prose it drifted back into a paragraph that recapped the
# candidates -- the one thing it is not for -- because a blank block invites a summary and a
# summary of what is directly above it is always available. Three lines, and each asks for
# something none of the other blocks can hold.

# What a field says when it asserts NOTHING. Kept apart from the assertion path because the two
# are checked against opposite facts: an empty class must be empty when no template earned a
# name, and a filled one must be earned.
_NOT_MEASURED = re.compile(r"^\s*(not measured|unmeasured|not mapped|unmapped|"
                           r"mapping failed|none|-|--|—)\s*\.?\s*$", re.I)
_NO_CLASS = re.compile(r"^\s*(none|no name|unnamed|nothing|nothing reproduces[^.]*|"
                       r"no named reaction[^.]*|-|--|—)\s*\.?\s*$", re.I)
_NO_VALUE = re.compile(r"^\s*(none|nothing|-|--|—|n/?a)\s*\.?\s*$", re.I)

# The element pair in a `bond:` field: `C-C`, `C-Br`, `O-C`, `P-C`, and `C=C` for a double bond.
# Two element symbols joined by a bond mark, and NOTHING that looks like an atom-map index -- the
# negative lookahead on `:` keeps `C:13-C:7` out of the pair set so it is reported as an index
# rather than silently read as a pair.
_PAIR_RE = re.compile(r"\b([A-Z][a-z]?)(?![a-z:])\s*[-=#]\s*([A-Z][a-z]?)(?![a-z:])\b")
# The verb, in any order a writer puts it. `order changes` and `changes order` and `changes the
# order of` are one claim, and matching only the first spelling rejects drafts that wrote the
# element pair correctly and phrased the verb the other way round. Classified by what the matched
# text CONTAINS rather than by which alternative fired, so a new spelling lands in the right bucket
# instead of silently in `formed`.
_VERB_RE = re.compile(
    r"\b(?:(?:changes?|changed|change)\s+(?:the\s+)?order(?:\s+of)?"
    r"|order\s+(?:changes?|changed|is\s+raised|is\s+lowered)"
    r"|raised|lowered|reduced|oxidi[sz]ed"
    r"|broken|breaks|break|cleaved|cleaves"
    r"|formed|forms|form|made|makes|make)\b", re.I)


def _verb_side(text: str) -> str:
    """Which of the three sides a matched verb phrase names."""
    t = text.lower()
    if "order" in t or t in ("raised", "lowered", "reduced", "oxidised", "oxidized"):
        return "changed"
    if any(w in t for w in ("brok", "break", "cleav")):
        return "broken"
    return "formed"


BOND_SIDES = ("formed", "broken", "changed")


def parse_document(text: str) -> dict:
    """The draft as blocks. Never raises; what it could not parse it reports.

    Returns {blocks, by_name, names, stray} where each block is {name, mid, body, start, end}.
    `stray` is every non-whitespace character OUTSIDE a block, which is the preamble-and-sign-off
    failure mode and is invisible if you only look at what parsed.
    """
    text = text or ""
    heads = list(_HEAD_RE.finditer(text))
    blocks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        blocks.append({"name": m.group(1).lower(), "mid": (m.group(2) or "").strip() or None,
                       "body": text[m.end():end].strip(),
                       "start": m.start(), "end": end,
                       "body_start": m.end()})
    # Only what sits BEFORE the first header is stray -- the preamble, which is the failure that
    # matters. There is no "after the last block": the last block runs to the end of the draft.
    stray = [text[:heads[0].start()]] if heads else [text]
    by_name: dict[str, list[dict]] = {}
    for b in blocks:
        by_name.setdefault(b["name"], []).append(b)
    return {"blocks": blocks, "by_name": by_name,
            "names": [b["name"] for b in blocks],
            "stray": "".join(stray).strip()}


def parse_keyed(body: str, keys: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    """({key: value} for the keys asked for, lines that are not one of them).

    Keys match in ANY order and a repeat keeps the first, because "which lines are present" and
    "are they in the right order" are two rejections and only the first is worth a draw. A line
    with no `key:` at all is reported: that is the prose-instead-of-fields failure, and it is the
    one the ledger blocks exist to prevent.
    """
    out, bad = {}, []
    for line in (body or "").splitlines():
        if not line.strip():
            continue
        m = _KEYED_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        k = m.group(1).lower()
        if k not in keys:
            bad.append(line.strip())
            continue
        out.setdefault(k, m.group(2).strip())
    return out, bad


def parse_cuts(body: str) -> tuple[list[dict], list[str]]:
    """(rows, lines that are not rows). A row is {c, bond, cls, split, raw}."""
    rows, bad = [], []
    for line in (body or "").splitlines():
        if not line.strip():
            continue
        m = _ROW_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        rows.append({"c": int(m.group(1)),
                     "bond": m.group(2), "cls": m.group(3),
                     "split": m.group(4), "raw": line.strip()})
    return rows, bad


def parse_bond_pairs(field: str) -> dict[str, list[tuple[str, str]]]:
    """{"formed": [(el, el), ...], "broken": [...]} out of a `bond:` field.

    The checkable half of the field. Free text says WHERE a bond sits and no checker can decide
    that; the element pair says WHICH ATOMS, and the mapping gives exactly that -- `O:29-C:1` is
    the pair (C, O) however the phrase around it is worded. Splitting the field here is what
    turns "a C-Br bond formation that severs the indole" (the mapping says C-C) from an
    unverifiable sentence into a refutable claim.

    Attribution is by CLAUSE and by the verb in it, so `C-C formed at the aldehyde carbon; C-O
    and P-C broken` yields formed {(C,C)} and broken {(C,O), (C,P)}. A clause with no verb is
    attributed to the previous clause's verb, which is how "C-O and P-C broken" reads when a
    writer splits it across two clauses. Pairs are sorted within themselves so `O-C` and `C-O`
    are one pair.
    """
    out: dict[str, list[tuple[str, str]]] = {s: [] for s in BOND_SIDES}
    if _NOT_MEASURED.match(field or ""):
        return out
    clauses = re.split(r"[;,]|\band\b", field or "")
    verbs = []
    prefix = []                # does the verb come BEFORE the pairs in its own clause?
    for c in clauses:
        v = _VERB_RE.search(c)
        verbs.append(_verb_side(v.group(0)) if v else None)
        first_pair = _PAIR_RE.search(c)
        prefix.append(bool(v and (first_pair is None or v.start() < first_pair.start())))
    # Which way a verbless clause inherits depends on where the verb SITS in the clause that
    # has one, and the two orders propagate opposite ways:
    #
    #   postfix -- "C-O and P-C broken"      the list shares its TRAILING verb  -> look forward
    #   prefix  -- "forms C-C and C-N"       the list shares its LEADING verb   -> look back
    #
    # Only the postfix rule was implemented, and the prompt teaches the PREFIX form ("forms C-N
    # at the carbonyl carbon; breaks C-Cl at the acyl chloride") -- so the model writes prefix
    # and every conjoined pair after a leading verb was handed the NEXT clause's verb instead:
    # `forms C-C and C-N at the ring; breaks C-Br ...` parsed C-N as BROKEN, the mapping refuted
    # it, and a correct draft was rejected as `bond_pair_wrong`.
    for i, v in enumerate(verbs):
        if v:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if verbs[j]), None)
        nxt = next((j for j in range(i + 1, len(verbs)) if verbs[j]), None)
        if prev is not None and prefix[prev]:
            verbs[i] = verbs[prev]
        elif nxt is not None and not prefix[nxt]:
            verbs[i] = verbs[nxt]
        else:
            verbs[i] = (verbs[nxt] if nxt is not None
                        else verbs[prev] if prev is not None else None)
    for clause, verb in zip(clauses, verbs):
        if verb is None:
            continue
        for a, b in _PAIR_RE.findall(clause):
            out[verb].append(tuple(sorted((a, b))))
    return out


# The measure, as a CLOSED SET of tokens rather than a phrase to recognise. Everything else in
# this block is slots, and this is what makes them worth having: with the measure named by a
# token, attributing a claim is parsing, and with it named in English it was a regex over prose,
# which produces false positives -- an axis missed because the writer said "changes order"
# instead of "order changes", a purchasability claim read out of a negation, a whole line's
# measure applied to a clause that named a different one -- that slot-level checks do not.
MEASURE_TOKENS = {
    "plausibility": "p", "p": "p",
    "round-trip": "rt", "round trip": "rt", "roundtrip": "rt", "rt": "rt",
    "confidence": "q", "q": "q",
    "price": "ln$", "cost": "ln$", "ln$": "ln$",
    # A fifth measure, for the rank turns whose taken candidate is not the winner on p, rt, q OR
    # price -- the label ranked it for a reason none of the four exposed numbers carries (the
    # search's own recursive lookahead, which is the oracle leak `route`/`dp` was removed for).
    # Forcing one of the four numeric measures onto such a turn either produces a false claim about
    # which number wins (the check catches it) or leaves the turn permanently unwritable, usually
    # with a clearing candidate sitting right there in the same menu. `structure` is not a number,
    # so `_c_deciding` grounds it in the candidate's own bond/named-reaction facts instead of a
    # value comparison -- see `_c_deciding`'s `dax == "struct"` branch.
    "structure": "struct", "mechanism": "struct", "named reaction": "struct",
}
_RULED_RE = re.compile(r"^\s*ruled out\s*:\s*(.*?)\s*$", re.I)
_ORDERED_RE = re.compile(r"^\s*ordered by\s*:\s*(.*?)\s*$", re.I)
_LEAVES_RE = re.compile(r"^\s*leaves to make\s*:\s*(.*?)\s*$", re.I)
_ROUTE_RE = re.compile(r"^\s*route\s+(\d+)\s*(?:--|-|\u2014)\s*c(\d+)\s*:?\s*$", re.I)
_DIVERGING_RE = re.compile(r"^\s*diverging from\s*:\s*(.*?)\s*$", re.I)
_SPENT_ON_RE = re.compile(r"^\s*(.*?)\s+spent\s+on\s+(.+?)\s*$", re.I)


# The number a `ruled out` line quotes used to live inside the free `why` text, and that put it
# beyond checking twice over: which candidate a number belonged to had to be guessed, and whether
# it was RIGHT was never checked at all -- only whether the ranking's ORDER agreed with it. That
# leaves room to invent plausible-looking route values for candidates the teacher was never shown
# one for, or to call a candidate "outranked" by a route worth MORE than the one taken, because
# nothing compares the two numbers. `value:` is its own slot now, aligned to `<candidates>`
# position for position, so both defects become a lookup: does this number match the board's, and
# does it sit on the right side of the one taken.
_DASHY = {"-", "--", "\u2014", "\u2013", "dash", "none", "n/a", "na", ""}


def _parse_values(s: str) -> list[float] | None:
    """Comma-separated readings, in order, POSITIONALLY. None if nothing parses.

    A dash holds its slot. The board prints a missing round-trip as an em dash and the prompt
    teaches that spelling ("a dash means it could not be recovered at all"), so a turn ruling
    three candidates out on round-trip writes `value: 2, --, --` -- and a regex that only finds
    numbers returns one reading for three candidates and the count check refuses the line. The
    dash is not a missing value, it IS the value for that candidate, and the checks
    downstream already treat `None` on the rt axis as "not recovered".
    """
    if _NO_VALUE.match(s or ""):
        return []
    parts = [x.strip() for x in (s or "").split(",")]
    out: list[float | None] = []
    for x in parts:
        if x.lower() in _DASHY:
            out.append(None)
            continue
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", x)
        if m is None:
            return None
        out.append(float(m.group(0)))
    return out if out else None


def parse_routes(body: str) -> tuple[list[dict], str | None, list[str]]:
    """DECIDING as the route loop it now is: (routes, the leaves-to-make line, malformed lines).

        route 1 -- c9
          ruled out: c0 | below cut-off | value: 0.009 | the filter says it does not go
          diverging from: none | nothing is spent on this piece yet
          ordered by: price | c9 closes the piece on the cheaper leaf
        route 2 -- c7
          ruled out: c1 | outranked on plausibility | value: 0.052 | reads below c7
          diverging from: c9 | spent on the acyl chloride | c7 takes the alkene C-C instead
          ordered by: plausibility | among what c9 left, c7 is the only cut that still splits it
        leaves to make: the brominated hydroxybenzaldehyde

    WHY a loop and not three slots once. A rank turn does not make one decision: `order[0]` is
    applied now and `order[1:]` stay on the board as the other routes through this molecule --
    the developer block says so in as many words. The single-pass form could only argue for the
    head, so the routes the turn was actually declaring went unexplained, and nothing in the
    text said they were different EXPANSIONS rather than a ranked list of the same idea. One
    `route i -- c<n>` group per entry of `order`, in order, is that statement, and every part of
    it is on the page already: the labels come from `order`, the candidates from the menu, and
    what each route diverges FROM is the earlier routes in this same block plus whatever the
    board's `untried` says is already spent.

    `leaves to make` stays molecule-level and outside the loop -- it is what the FIRST cut owes
    the route, and there is no check that grounds its content, so multiplying it per route would
    multiply an ungrounded claim, and the copies come to disagree with each other.
    """
    routes: list[dict] = []
    leaves: str | None = None
    bad: list[str] = []
    cur: dict | None = None
    for line in (body or "").splitlines():
        if not line.strip():
            continue
        m = _ROUTE_RE.match(line)
        if m:
            cur = {"label": int(m.group(1)), "cand": int(m.group(2)),
                   "ruled": [], "diverging": None, "ordered": None, "raw": line.strip(),
                   "raw_lines": []}
            routes.append(cur)
            continue
        if _LEAVES_RE.match(line) or _RULED_RE.match(line):
            # Both slots are gone. `ruled out` went because the route order already carries
            # which cuts the call prefers -- restating it as a per-candidate scoring argument
            # made the block an argument about the ranking instead of about the chemistry, and
            # it drew many refused drafts. `leaves to make` went with it: no check grounds its
            # CONTENT against the board, and its copies came to disagree with each other.
            bad.append(line.strip())
            continue
        m = _DIVERGING_RE.match(line)
        if m:
            if cur is not None:
                cur["raw_lines"].append(line.strip())
            d = _parse_diverging(m.group(1))
            if d is None:
                bad.append(line.strip())
            elif cur is None:
                bad.append(line.strip())          # a slot outside any route group
            else:
                cur["diverging"] = d
            continue

        m = _ORDERED_RE.match(line)
        if m:
            if cur is not None:
                cur["raw_lines"].append(line.strip())
            parts = [x.strip() for x in m.group(1).split("|")]
            # THREE slots. `value:` is what makes a quoted number checkable: verifying numbers
            # from the free text produces flags that cannot be stood behind -- a round-trip line
            # quoting a plausibility reads as a mismatch, because which axis a number belongs to
            # is a guess. Positional, it is a fact. And `why` may then carry no digits at all, so
            # there is nowhere left for an unchecked number to sit.
            if len(parts) < 3 or cur is None \
                    or not parts[1].lower().lstrip().startswith("value"):
                bad.append(line.strip())
                continue
            cur["ordered"] = {"token": parts[0].lower(),
                              "values": _parse_values(parts[1].split(":", 1)[-1]),
                              "value_raw": parts[1].split(":", 1)[-1].strip(),
                              "why": " | ".join(parts[2:]).strip(), "raw": line.strip()}
            continue
        bad.append(line.strip())
    return routes, leaves, bad


_REF_RE = re.compile(r"\br(\d+)\b|\b(\w+)[\u00b7.]c(\d+)\b")


def _parse_diverging(v: str) -> dict | None:
    """`none | <why>` or `<history refs> | <what that route took> | <how this one differs>`.

    The references are to the board's REACTION LEDGER -- `r2`, or `y7\u00b7c0` -- not to the other
    candidates of this turn's own ranking. That was the first design and it was pointed the
    wrong way: sibling candidates are one decision seen three ways, and the route order already
    says which the call prefers, so arguing between them added nothing the ranking had not said.
    What a route has to be argued against is what the board has already EXPANDED elsewhere --
    either "that branch went this way, and this one is worth opening too because it differs
    here", or "something like this is already expanded, so take the other candidate". Both are
    statements about earlier context, and attending to earlier context is the point.

    `none` still takes a reason: a null that costs nothing to write is a null the model writes
    when it should not.
    """
    parts = [x.strip() for x in v.split("|")]
    if parts and _NO_VALUE.match(parts[0]):
        return {"none": True, "why": " | ".join(parts[1:]).strip(), "raw": v.strip()}
    if len(parts) < 3:
        return None
    refs = []
    for rid, mid, ci in _REF_RE.findall(parts[0]):
        if rid:
            refs.append(("r", int(rid)))
        elif mid and ci:
            refs.append(("m", mid, int(ci)))
    if not refs:
        # A bare `c3, c7` is the CANDIDATE level: the other cuts of this same ranking. Which
        # level a route is allowed to cite is decided by the board, not by preference -- see
        # `_c_deciding`. Parsed here without judgement; judged there.
        refs = [("c", int(x)) for x in re.findall(r"\bc(\d+)\b", parts[0])]
    if not refs:
        return None
    mid_slot = parts[1]
    # Two honest shapes, because the board produces both. Where the route already expanded and
    # this one are the same KIND of step, saying so IS the finding -- it is the "something like
    # this is already expanded" case, and dressing it up as different chemistry would be false.
    same = bool(re.match(r"^\s*same\b", mid_slot, re.I))
    m = _SPENT_ON_RE.match(mid_slot) or re.match(r"^\s*spent\s+on\s+(.+)$", mid_slot, re.I) \
        or re.match(r"^\s*same\s+step\s*[,:]?\s*(.+)$", mid_slot, re.I)
    on = (m.group(m.lastindex) if m else mid_slot).strip()
    return {"none": False, "same": same, "refs": refs, "on": on,
            "instead": " | ".join(parts[2:]).strip(), "raw": v.strip()}


# `split_axis` and its AXIS_WORDS table lived here: a regex that tried to read which measure a
# line of English was arguing from. It is gone. The DECIDING block names its measure in a slot
# from a closed set, so the question it answered is now a dict lookup (MEASURE_TOKENS). The
# regex produced false rejections -- it missed "changes order" for "order changes", missed
# "reach no stock", missed "best route is worth", and fell back to the wrong measure whenever
# it missed. Nothing that reads prose decides anything here.


_MOL_TAG_RE = re.compile(r"<mol (\w+)>([^<]+)</mol>")
_LEDGER_RE = re.compile(r"^\s*(r\d+)\s+(\w+)·c(\d+)\s", re.M)


def ancestors_of(ep: dict, k: int, mid: str) -> dict[str, str]:
    """{molecule id: SMILES} for every piece STRICTLY ABOVE `mid` on its own branch.

    Reconstructed from what the board already printed, not from a new evidence field: the
    `<mol xx>SMILES</mol>` tag at first appearance gives the id -> SMILES map, the ledger line
    `rN  mid·cK` gives each reaction's parent, and `evidence["mols"][mid]["under"]` gives the
    reaction a piece hangs from. Walking `under` -> parent -> `under` reaches the root, so this
    needs no re-render of the episodes, which is the whole reason it is done this way.

    The target is always included: it is the ancestor of everything and the piece a cycle most
    often regenerates.
    """
    smi: dict[str, str] = {}
    parent: dict[str, str] = {}
    under: dict[str, str] = {}
    for j in range(k + 1):
        t = ep["turns"][j]
        env = t.get("env") or ""
        for m in _MOL_TAG_RE.finditer(env):
            smi.setdefault(m.group(1), m.group(2))
        for m in _LEDGER_RE.finditer(env):
            parent.setdefault(m.group(1), m.group(2))
        for xid, row in ((t.get("evidence") or {}).get("mols") or {}).items():
            if row.get("smiles"):
                smi.setdefault(xid, row["smiles"])
            if row.get("under"):
                under.setdefault(xid, row["under"])
    out: dict[str, str] = {}
    cur, seen = under.get(mid), {mid}
    while cur and cur in parent:
        p = parent[cur]
        if p in seen:
            break
        seen.add(p)
        if p in smi:
            out[p] = smi[p]
        cur = under.get(p)
    root = ep.get("target")
    if root:
        rid = next((x for x, s in smi.items() if s == root), None)
        if rid and rid != mid:
            out.setdefault(rid, root)
    return out


_LEDGER_RE = re.compile(r"^(r\d+)\s+(\w+)[\u00b7.]c(\d+)", re.M)


def history_reactions(ep: dict, k: int) -> list[dict]:
    """Every step already applied on this board, each with the KIND of reaction it was.

    This is what `diverging from` attends to. The slot used to point at the other candidates of
    the turn's own ranking, which was the wrong thing entirely: sibling candidates are one
    decision seen three ways, and the ranking already says which the call prefers. What a route
    has to be argued against is the chemistry the board has ALREADY expanded somewhere else --
    "this branch is worth opening too, and here is how it differs" or "something like this is
    already expanded, so take the other candidate".

    The source is the board's own reaction ledger, parsed out of the turn's observation:

        r1  tu·c5  2 of 2 closed · solved        tu ranked: c5 · c7 · c2
        r6  q4·c2  2 of 2 closed · solved
        r7  q4·c0  2 of 2 closed · solved

    -- the exact lines the model is looking at, so a claim about history is checkable against
    the page rather than against state only this process can see. `r6`/`r7` above are two routes
    through ONE molecule, which is the case the slot exists for.

    The reaction KIND is not in that text, so it comes from the turn where the candidate's menu
    was shown: `facts_of` there gives the mapping, and `family_key` reduces it to the same
    identity the support block groups candidates by.
    """
    out: list[dict] = []
    for rid, mid, ci_s in _LEDGER_RE.findall(ep["turns"][k].get("env") or ""):
        ci = int(ci_s)
        key = ("unmeasured",)
        for j in range(k):
            f = facts_of(ep["turns"][j].get("evidence") or {}, mid)
            if ci in f:
                key = family_key(f[ci])
                break
        out.append({"rid": rid, "mid": mid, "c": ci, "family": key,
                    "ref": f"{mid}\u00b7c{ci}"})
    return out


def cycle_traps(ep: dict, k: int, mid: str) -> dict[int, list[str]]:
    """{candidate: the ancestor ids it would regenerate} for one ranked molecule.

    A candidate is a trap when any of its precursors IS a piece already above `mid`, OR is `mid`
    itself: taking it makes that piece its own precursor and the board refuses the call outright.

    `mid` itself has to be in the set. The board's own guard is
    `existing == m.mid or self._reaches(existing, m.mid)`, and the first half was missing here --
    `ancestors_of` walks STRICTLY above the molecule, so a candidate handing back the very piece
    being ranked would come out clean although the board refuses it ("c9 would make wt its own
    precursor"), and at the root -- where there are no ancestors -- it is the only kind of trap
    there is.
    """
    ev = ep["turns"][k].get("evidence") or {}
    anc = ancestors_of(ep, k, mid)
    by_smi = {s: x for x, s in anc.items()}
    own = ((ev.get("mols") or {}).get(mid) or {}).get("smiles")
    if not own:
        for j in range(k + 1):
            for m in _MOL_TAG_RE.finditer(ep["turns"][j].get("env") or ""):
                if m.group(1) == mid:
                    own = m.group(2)
    if own:
        by_smi.setdefault(own, mid)
    if not by_smi:
        return {}
    out: dict[int, list[str]] = {}
    for c in ((ev.get("menus") or {}).get(mid) or {}).get("candidates") or []:
        hit = sorted({by_smi[s] for s in (c.get("reactants") or []) if s in by_smi})
        if hit:
            out[c["c"]] = hit
    return out


def ranked_orders(turn: dict) -> dict[str, list[int]]:
    """{molecule id: the order the call declares} for every rank action on this turn."""
    out: dict[str, list[int]] = {}
    for a in (turn.get("actions") or []):
        if not isinstance(a, dict) or (a.get("type") or "").lower() != "rank":
            continue
        if not a.get("mid"):
            continue
        try:
            out[str(a["mid"])] = [int(x) for x in (a.get("order") or [])]
        except (TypeError, ValueError):
            out[str(a["mid"])] = []
    return out


def takes_of(turn: dict) -> dict[str, int]:
    """{molecule id: how many of its ranking the call EXPANDS on this turn}.

    `take` is a field on the rank action, default 1. The form has to say which groups are
    arguing for a branch that will exist and which for one the board is keeping, and that is a
    fact about the call -- so it is printed in the support block rather than left to be
    inferred. A support block that withholds a fact the form demands leaves turns unwritable.
    """
    out: dict[str, int] = {}
    for a in (turn.get("actions") or []):
        if not isinstance(a, dict) or (a.get("type") or "").lower() != "rank":
            continue
        if not a.get("mid"):
            continue
        try:
            out[str(a["mid"])] = max(1, int(a.get("take", 1) or 1))
        except (TypeError, ValueError):
            out[str(a["mid"])] = 1
    return out


def menu_of(ev: dict, mid: str) -> dict[int, dict]:
    return {c["c"]: c for c in (((ev.get("menus") or {}).get(mid) or {}).get("candidates") or [])}


def facts_of(ev: dict, mid: str) -> dict[int, dict]:
    return {r["c"]: r for r in (((ev.get("facts") or {}).get(mid) or {}).get("candidates") or [])}


# ==================================================================== molecule descriptors
# The evidence dict carries a descriptor for every OPEN molecule, and never one for the target
# once the target is closed -- which is most of an episode. The <gap> block is a comparison
# against the target, so the target's descriptor has to come from somewhere: the shared cache
# first (it is the same table `board.evidence` reads, so the two paths cannot disagree), RDKit
# second, and nothing at all third -- in which case the delta line is simply absent and the
# teacher writes the gap from the two SMILES, which is still enough.
_DESC_CACHE: dict[str, dict] = {}
_CACHE_EV = None


def _cache_evidence():
    global _CACHE_EV
    if _CACHE_EV is None:
        try:
            from board.evidence import CacheEvidence
            _CACHE_EV = CacheEvidence()
        except Exception:                                              # noqa: BLE001
            _CACHE_EV = False
    return _CACHE_EV or None


def _ring_systems(mol) -> int:
    """Connected components of rings sharing atoms -- what a chemist means by 'tricyclic'.

    The usual ring-system definition, computed here so this module needs only RDKit.
    """
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    parent = list(range(len(rings)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return len({find(i) for i in range(len(rings))})


def describe(smiles: str) -> dict:
    """The cached descriptor for a molecule, or a recomputed one, or {}. Never raises."""
    if not smiles:
        return {}
    if smiles in _DESC_CACHE:
        return _DESC_CACHE[smiles]
    d = {}
    ce = _cache_evidence()
    if ce is not None:
        try:
            d = ce.for_molecule(smiles) or {}
        except Exception:                                              # noqa: BLE001
            d = {}
    if not d and Chem is not None:
        try:
            from rdkit.Chem import Descriptors, rdMolDescriptors
            from rdkit.Chem.Scaffolds import MurckoScaffold
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                scaf = MurckoScaffold.GetScaffoldForMol(m)
                gen = (Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.Mol(scaf)))
                       if scaf.GetNumAtoms() else "")
                centres = Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                                    useLegacyImplementation=False)
                d = {"scaffold": Chem.MolToSmiles(scaf) if scaf.GetNumAtoms() else "",
                     "scaffold_generic": gen,
                     "n_rings": int(m.GetRingInfo().NumRings()),
                     "n_arom_rings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
                     "n_ring_systems": _ring_systems(m),
                     "n_heavy": m.GetNumHeavyAtoms(),
                     "mw": round(Descriptors.MolWt(m), 2),
                     "n_heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(m)),
                     "n_halogens": sum(1 for a in m.GetAtoms()
                                       if a.GetSymbol() in ("F", "Cl", "Br", "I")),
                     "frac_csp3": round(rdMolDescriptors.CalcFractionCSP3(m), 4),
                     "n_rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(m)),
                     "n_stereocentres": len(centres),
                     "n_unspec_stereocentres": sum(1 for _, t in centres if t == "?")}
        except Exception:                                              # noqa: BLE001
            d = {}
    _DESC_CACHE[smiles] = d
    return d


def _desc_phrase(d: dict) -> str:
    """A descriptor row as a chemist would read it. Same shape the freeform module uses."""
    if not d:
        return ""
    bits = []
    n = d.get("n_rings")
    if n is not None:
        bits.append(f"{n} ring(s) in {d.get('n_ring_systems')} system(s)"
                    + (f", {d['n_arom_rings']} aromatic" if d.get("n_arom_rings") else ""))
    if d.get("n_heavy") is not None:
        bits.append(f"{d['n_heavy']} heavy atoms")
    if d.get("mw"):
        bits.append(f"MW {d['mw']:.0f}")
    if d.get("frac_csp3") is not None:
        bits.append(f"Fsp3 {d['frac_csp3']:.2f}")
    for key, lab in (("n_heteroatoms", "heteroatoms"), ("n_halogens", "halogens"),
                     ("n_rot_bonds", "rotatable bonds")):
        if d.get(key):
            bits.append(f"{d[key]} {lab}")
    st = d.get("n_stereocentres")
    un = d.get("n_unspec_stereocentres")
    if st or un:
        bits.append(f"stereocentres {st or 0} set"
                    + (f", {un} unspecified" if un else ""))
    if d.get("scaffold"):
        bits.append("Murcko " + d["scaffold"])
    return "; ".join(bits)


def _substruct(part: str, whole: str) -> bool | None:
    """Is `part` literally still inside `whole`? None when it cannot be decided.

    This is the one fact that says whether a piece is a PIECE of the target or something the
    route has already transformed -- a bromide installed as a coupling handle is not a
    substructure of the target, and that is exactly the sentence the gap block should be making.
    """
    if Chem is None or not part or not whole:
        return None
    a, b = Chem.MolFromSmiles(part), Chem.MolFromSmiles(whole)
    if a is None or b is None:
        return None
    try:
        return bool(b.HasSubstructMatch(a))
    except Exception:                                                  # noqa: BLE001
        return None


# =============================================================== the teacher's private facts
def _worse_on(c: dict, head: dict) -> str:
    """Which axes this candidate reads WORSE on than the one taken -- computed, not derived.

    `outranked on <axis>` asserts exactly this, and left to the draft it rules a candidate out
    as outranked on an axis it is actually BETTER on. The axes the head wins are already
    printed for `ordered by`; this is the same fact per candidate, for the line that names one.
    A candidate that loses on nothing has no honest `outranked` axis at all -- `structure` is
    what that turn should reach for -- so saying so is as useful as the list.
    """
    win = [AXIS_TOKEN[ax] for ax in AXES
           if (k := _axis_key(c, ax)) is not None
           and (h := _axis_key(head, ax)) is not None and k > h]
    return ("  worse than the taken cut on: " + ", ".join(win) if win
            else "  worse on NO number -- rule it out with `outranked on structure` and say what "
                 "the taken cut does that it does not")


def _cut_side(c: dict) -> str:
    """` BELOW the cut-off` / ` clears the cut-off`, computed, next to the number it is about.

    The comparison, not left to the reader. Left to the draft, `gate_wrong` is the model quoting
    the board's own p correctly and then classifying it wrongly (`c5 | below cut-off | value:
    0.392`, above the 0.05 cut-off): the number is read, the comparison against the bar is not
    made. It is a fact about one candidate that the board
    already determines, so it is handed over the same way the split verdict and the deciding
    axes are, rather than being a step the draft has to perform and can fail.
    """
    p = (c.get("signals") or {}).get("p")
    if p is None:
        return "  (no plausibility on record)"
    return "  BELOW the cut-off" if p < CUTOFF else "  clears the cut-off"


def _signals(sig: dict) -> str:
    """`q.493  p1.000  rt1`, exactly as the board writes them.

    Not the raw values out of the evidence dict. The board rounds to three places and drops the
    leading zero, and the student never sees anything else -- so a support line printing
    `p0.9999` teaches it to quote a number the board will not have shown it, and every check
    that compares a quoted score against the evidence then has to accept two spellings.
    """
    out = []
    for k in ("q", "p", "rt"):
        v = sig.get(k)
        if k == "rt":
            out.append("rt" + ("—" if v is None else str(int(v))))
        elif v is None:
            out.append(k + "—")
        else:
            out.append(k + f"{v:.3f}".lstrip("0"))
    return "  ".join(out)


def bond_pairs_of(row: dict) -> dict[str, list[tuple[str, str]]]:
    """The element pairs the mapping measured: {"formed": [(C,C)], "broken": [(C,O), (C,P)]}.

    Reads the evidence in either shape. `episode._bond_rows` now writes a dict per changed bond
    -- `atoms` ("C-Br"), `order`, `fg1`/`fg2`, and `at` where the mapping produced indices -- and
    older rows carry only the string `C:1-C:2`. The element pair is what the `bond:` field is
    checked against and it is present in both.
    """
    out: dict[str, list[tuple[str, str]]] = {"formed": [], "broken": [], "changed": []}
    for side in BOND_SIDES:
        for x in row.get(side) or []:
            if isinstance(x, dict):
                els = re.findall(r"[A-Z][a-z]?", str(x.get("atoms") or ""))
            else:
                els = re.findall(r"([A-Z][a-z]?):\d+", str(x))
            if len(els) == 2:
                out[side].append(tuple(sorted(els)))
    return out


def bond_where(row: dict) -> dict[str, list[str]]:
    """What each end of each changed bond IS, in the mapper's own words.

    This is the positional half of the bond fact, and it was being thrown away. The tool reports
    `fg1`/`fg2` per changed bond -- "aromatic C", "carbonyl C", "Br (leaving group / halide)",
    "boronic/boronate B" -- and that is a POSITION a chemist can act on, recoverable from the
    product and its precursors, unlike the atom-map index that was kept in its place. Without it
    the teacher had to read the SMILES and guess where a bond sat, which is the one half of the
    field no checker can refute and therefore the one place a guess survives into the corpus.
    """
    out: dict[str, list[str]] = {"formed": [], "broken": [], "changed": []}
    for side in BOND_SIDES:
        for x in row.get(side) or []:
            if not isinstance(x, dict):
                continue
            a, b = x.get("fg1"), x.get("fg2")
            if not a and not b:
                continue
            order = f" ({x['order']})" if x.get("order") and x["order"] != "single" \
                else ""
            out[side].append(f"{a or '?'} / {b or '?'}{order}")
    return out


def _pair_str(pairs) -> str:
    return ", ".join("-".join(p) for p in dict.fromkeys(pairs)) or "—"


def _bond_phrase(row: dict) -> str:
    """The bond change: the element pairs first, then the atom-map labels that locate them.

    The labels are given because they say WHICH atoms; the form forbids quoting them, because an
    atom-map index is an artefact of how the mapping ran and the student has no way to reproduce
    one. The element pair in front of them is the part the row must carry verbatim -- it is what
    `bond_pair_wrong` compares against -- and the phrase after it is the teacher's own, recovered
    by reading the SMILES.
    """
    if not row.get("bond_measured"):
        return ("NOT MEASURED -- no atom mapping is on record for this step; the row's bond "
                "field reads exactly `not measured`, and nothing anywhere may claim which "
                "bond moves")
    pair, where = bond_pairs_of(row), bond_where(row)
    bits = []
    for side in BOND_SIDES:
        if not pair[side] and not where[side]:
            continue
        w = ("   " + "; ".join(where[side])) if where[side] else ""
        verb = {"formed": "forms", "broken": "breaks",
                "changed": "changes the order of"}[side]
        bits.append(f"{verb} {_pair_str(pair[side])}{w}")
    return "\n                       ".join(bits) or \
        "the mapping ran and found no bond change"


def _class_phrase(row: dict) -> str:
    """What the template run actually earned, in the words the `class:` field is checked on.

    The weak tier is spelled out rather than hidden, because "the groups fit a Suzuki and the
    outcome is not what a Suzuki gives" is informative and is NOT a name -- and naming it anyway
    is a false claim. So the sentence says what the row
    must contain, not merely what the tier is.
    """
    tier, names = row.get("named_tier"), row.get("named") or []
    if tier == "applies+makes" and names:
        return (f"EARNED: {', '.join(names)}  (a template reproduces the product, "
                f"match {row.get('named_match')})")
    if tier == "applies":
        if names:
            return ("NOT EARNED: the groups fit " + ", ".join(names[:3])
                    + ", but no template reproduces the product, so that is not what this step "
                      "IS -- the class field must read `none`")
        return "NOT EARNED: a template applies and none reproduces the product -- `none`"
    if tier == "none":
        return "NOT EARNED: no named template even applies -- the class field must read `none`"
    # tier is absent: the templates were never RUN on this step. That is not the same fact and
    # it must not be dressed as one: many steps are simply absent from the named cache, so "no
    # named template even applies" would be a false negative there, and a teacher that argues
    # FROM the absence of a name is arguing from our coverage.
    return ("NOT CHECKED: no template run has been recorded for this step. Write `none` -- the "
            "field records what is earned -- and do not argue anywhere that this step has no "
            "name, because that has not been tested")


# What a split IS, as a word rather than as three numbers to compare in your head.
#
# A field reading `28 -> 34 + 14` says nothing a reader can use: the product has 28 heavy atoms
# and the precursors 34 and 14, which looks like broken arithmetic until you notice the extra
# atoms are a triphenylphosphine that never reaches the product.
# The number a chemist actually wants is the RATIO of the two pieces -- does this cut halve the
# molecule or shave something off its edge -- and that is a verdict, so it is computed here and
# handed over as one, the same way the skeleton verdict is.
SPLIT_WORDS = {
    "no split": "no split at all -- there is one precursor, so this rearranges or decorates "
                "rather than divides",
    "convergent": "convergent -- two comparable halves",
    "unbalanced": "unbalanced -- one piece carries most of the molecule",
    "decoration": "a decoration -- it shaves a small piece off the edge",
}


# A row may carry the verdict as the word or as the phrase the prompt shows, because the field
# is English and "one piece carries most of it" is the same claim as "unbalanced". Both spellings
# are recognised so the check reads what a chemist would write rather than a token.
SPLIT_PATTERNS = {
    "convergent": r"\bconvergent\b|comparable halves|two halves|halves it",
    "unbalanced": r"\bunbalanced\b|one piece carr(?:ies|ying)|most of (?:it|the molecule)",
    "decoration": r"\bdecorat|shaves|off the edge|small piece off",
    "no split": r"\bno split\b|(?:one|a single) precursor|does not divide",
}


def split_verdict(sh: dict) -> str:
    """Which of the four shapes this cut is. The word is the answer; the gloss is for reading."""
    frags = sh.get("fragments") or []
    if len(frags) < 2:
        return "no split"
    r = sh.get("size_ratio")
    if r is None:
        r = min(frags) / max(max(frags), 1)
    return "convergent" if r >= 0.5 else "unbalanced" if r >= 0.2 else "decoration"


def _shape_phrase(sh: dict) -> str:
    if not sh:
        return ""
    v = split_verdict(sh)
    bits = [SPLIT_WORDS.get(v, v).upper()]
    frags = sh.get("fragments") or []
    if len(frags) > 1:
        bits.append("fragments of " + " and ".join(str(x) for x in frags) + " heavy atoms")
    elif frags:
        bits.append(f"one precursor of {frags[0]} heavy atoms")
    # Atoms the precursors carry that the product never sees: a leaving group, an auxiliary, a
    # phosphine. Saying it explicitly is what stops `28 -> 34 + 14` reading as arithmetic that
    # does not add up.
    bal = sh.get("heavy_balance")
    if bal is not None and bal <= -3:
        bits.append(f"{-bal} of those atoms are reagent and never reach the product, "
                    f"which has {sh.get('heavy_product')}")
    if sh.get("scaffold_kept") is not None:
        # The one field of the three that cannot be read off the SMILES by eye, so a teacher
        # left to judge it guesses, and guesses in one direction.
        bits.append("THE RING SKELETON IS KEPT" if sh["scaffold_kept"]
                    else "THE RING SKELETON CHANGES")
    if sh.get("rings_product") is not None:
        bits.append(f"{sh['rings_product']} rings in the product against "
                    f"{sh.get('rings_precursors')} across the precursors")
    st, sp = sh.get("stereo_product"), sh.get("stereo_precursors")
    if st or sp:
        bits.append(f"{st} stereocentres in the product against {sp} across the "
                    f"precursors")
    return "; ".join(bits)


# ------------------------------------------------------------------ the delta, as a set
def group_words(smiles: str) -> set[str]:
    """Every ring class and functional group `board.factcheck` can name in one molecule.

    The vocabulary is the checker's own, not a second list: a word this returns is a word
    `ring_identity`/`fg_identity` can decide, and a word it does not return is one no check can
    adjudicate. That is what makes `delta` a set difference rather than an impression -- it is
    computed in the same terms it will be verified in.
    """
    if FC is None or Chem is None or not smiles:
        return set()
    out = set()
    try:
        sigs = FC._ring_sigs(smiles)
        for word, sig in FC.RING_SIG.items():
            if sig in sigs:
                out.add(word)
        pairs = FC._fused_pairs(smiles)
        for word, sig in FC.FUSED_SIG.items():
            if sig in pairs:
                out.add(word)
        for word, sma in FC.FG_SMARTS.items():
            if FC._has_fg(smiles, sma):
                out.add(word)
    except Exception:                                                  # noqa: BLE001
        return set()
    return out


def delta_facts(state_smi: str, target_smi: str) -> tuple[list[str], list[str]]:
    """(in the piece and NOT in the target, in both) -- the `delta` and `state` fields' evidence.

    `delta` was `todo` in the first schema and it drifted, because "what still has to be made"
    admits two readings -- a handle the route installed that the target does not have, and a
    part of the target this piece has not reached yet -- and the field slid between them turn to
    turn. A set difference has one reading, is computed here rather than judged, and is checkable
    both ways afterwards: a word in `delta` must be in the piece AND absent from the target.
    """
    if not state_smi or not target_smi:
        return [], []
    s, t = group_words(state_smi), group_words(target_smi)
    return sorted(s - t), sorted(s & t)


# ------------------------------------------------------------------ structural annotations
# A sidecar of structural notes, keyed by molecule SMILES or by rxn_key, merged into the support
# block as GIVEN facts.
#
# The `bond:` field asks the teacher to say where a bond sits -- "at the aldehyde carbon", "at
# the benzylic position" -- and that is the half of the field no checker can decide. The element
# pair is checked; the position phrase is trusted. Trusting it is only safe if the teacher can
# read it off something rather than invent it, and a teacher reading a SMILES string invents
# more than it reads. So the position can be supplied: whatever produced the note (a stronger
# model, a cheminformatics pass, a hand-written table) writes it once per molecule or per step,
# and the teacher quotes rather than guesses.
#
# Format -- JSON object or JSONL, either of:
#     {"<SMILES>": "the aldehyde carbon bears the propyl chain to the naphthalene"}
#     {"key": "<product>>><reactant>.<reactant>", "note": "the bond formed is the styryl C=C"}
# The rxn_key spelling is `board.evidence.rxn_key`: product, then ">>", then the reactants
# sorted and joined by ".". A key that matches nothing is ignored rather than reported, because a
# note file is written against one pool and reused across several.
def load_annotations(path) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--annotations: no such file {p}")
    out: dict = {}
    txt = p.read_text().strip()
    if txt.startswith("{") and "\n" not in txt.strip().rstrip("}").rstrip():
        pass
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            for k, v in obj.items():
                out[k] = v if isinstance(v, str) else (v or {}).get("note", "")
            return {k: v for k, v in out.items() if v}
    except json.JSONDecodeError:
        pass
    for line in txt.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        k = r.get("key") or r.get("smiles") or r.get("rxn_key")
        v = r.get("note") or r.get("text") or ""
        if k and v:
            out[str(k)] = str(v)
    return out


def _rxn_key(product: str, reactants) -> str:
    """Byte-identical to board.evidence.rxn_key, so a note file keyed there joins here."""
    return f"{product}>>" + ".".join(sorted(set(reactants or [])))


# ------------------------------------------------------------------ candidate families
NAIVE = {"on": False}

# The ablation's control arm: the teacher gets the SAME supporting information the qualified
# arm gets -- the axis that separated the leaders, the per-candidate signals, the families --
# and is asked to write prose instead of the four-block form, with no gate reading what comes
# back. What that arm adds over this one is the FORM and its verification, so those are the two
# things this removes and the only two.
SCHEMA_NAIVE = """Write the reasoning as plain prose. There is no form to fill in and no block
structure: say what this call does and why, in the order that makes the argument, and stop.

Use the supporting information above -- it is the same information the search itself had. Name
the cuts by their c-numbers, say what each one does to the molecule, and say what made the one
this call takes better than the ones it did not. Where a measure decided it, say which measure
and what it read.

Do not invent facts the information above does not carry."""


def family_key(row: dict) -> tuple:
    """What makes two disconnections the same KIND of step.

    Earned reaction names first: two candidates a template calls a Wittig are one family however
    their fragments differ. Failing that, the bond-change signature -- which element pairs are
    made and which broken -- which is the level a reaction generalises at and the level the
    `bond:` field is written at. Failing both, everything unmeasured sits together, which is
    itself informative: it is the group nothing is known about.
    """
    if row.get("named_tier") == "applies+makes" and row.get("named"):
        return ("named", tuple(sorted(row["named"])))
    if row.get("bond_measured"):
        p = bond_pairs_of(row)
        return ("bond", tuple(sorted(p["formed"])), tuple(sorted(p["broken"])))
    return ("unmeasured",)


def family_label(key: tuple) -> str:
    if key[0] == "named":
        return ", ".join(key[1])
    if key[0] == "bond":
        f = ", ".join("-".join(p) for p in key[1]) or "—"
        b = ", ".join("-".join(p) for p in key[2]) or "—"
        return f"unnamed: {f} formed / {b} broken"
    return "no mapping"


# There was a `_route_phrase` here, printing `dp` -- "best route through it is worth 0.9957" --
# for every candidate. It is gone along with the measure it existed to justify: see the note by
# `AXES`. Nothing in the support block now says what the DP's own recursive value is, because
# nothing here should be arguing from it.


def rank_support(ep: dict, turn: dict, ann: dict | None = None, k: int | None = None) -> str:
    """Everything a RANK turn is allowed to argue from, in the order the form asks for it.

    Never rendered onto the board and never shown to the student: with no `analyze` action in the
    space, these facts reach the student only through the reasoning the teacher writes.

    Candidates are printed BY FAMILY rather than in menu order. Ranking is a choice between kinds
    of step -- two Wittigs and a Williamson is a different question from three Williamsons -- and
    a flat list ordered by candidate number hides that, which is how a trace ends up arguing from
    c-numbers instead of from chemistry. Grouping also stops the support block reading as a
    ranking already made: the ranked candidates are not printed first, they are printed inside
    whichever family they belong to, alongside the alternatives that were not taken.
    """
    ev = turn.get("evidence") or {}
    ann = ann or {}
    tgt = ep.get("target") or ""
    orders = ranked_orders(turn)
    L: list[str] = []

    if tgt:
        L.append(f"TARGET  {tgt}")
        ph = _desc_phrase(describe(tgt))
        if ph:
            L.append(f"    {ph}")
        # The ring classes and functional groups actually PRESENT, computed. The counts alone
        # left the NAME to a guess, and a guess from a SMILES string is a frequency prior: a
        # styryl compound becomes "a stilbene" (one side is an sp3 chain, not an aryl) and a
        # 2-(naphthalen-2-yl)ethyl group "2-phenylethyl", with every count correct. Both err
        # toward the commoner word, and this is the line every later turn refers back to.
        # It is also the vocabulary the identity checks decide in, so a line written from it is
        # a line that can be checked.
        gw = group_words(tgt)
        if gw:
            L.append(f"    groups and ring systems present: {', '.join(sorted(gw))}")
        if ann.get(tgt):
            L.append(f"    note: {ann[tgt]}")
        canon = ep.get("_target_canon")
        if canon:
            # A phrase to COPY, not a fact to re-derive. The groups line above is recomputed
            # correctly every turn -- it is the free-written NAME built from it that drifts: each
            # turn is a fresh guess from the same SMILES, and nothing forces the guesses to agree
            # unless later turns are asked to copy the first turn's answer. So the first accepted
            # turn's phrase is handed back verbatim from here on; `target:` becomes a copy task,
            # which a teacher does far more reliably than a naming task repeated every turn
            # without ever seeing its own earlier answer.
            L.append(f"    ESTABLISHED PHRASE for `target:` -- copy this exactly, do not "
                     f"reword it: {canon!r}")
        L.append("")

    for mid, order in orders.items():
        mm = (ev.get("menus") or {}).get(mid) or {}
        mrow = (ev.get("mols") or {}).get(mid) or {}
        smi = mrow.get("smiles") or ""
        L.append(f"RANKING  {mid}   depth {mm.get('depth', mrow.get('depth'))}"
                 + (f", under {mrow['under']}" if mrow.get("under") else ""))
        if smi:
            L.append(f"    {smi}")
            ph = _desc_phrase(mrow.get("desc") or describe(smi))
            if ph:
                L.append(f"    {ph}")
            if ann.get(smi):
                L.append(f"    note: {ann[smi]}")
            gone, shared = delta_facts(smi, tgt)
            L.append("    groups and ring systems present: "
                     + (", ".join(sorted(group_words(smi))) or "none this vocabulary names"))
            if shared:
                L.append(f"      of those, also in the target: {', '.join(shared)}")
            L.append("      of those, NOT in the target: "
                     + (", ".join(gone) if gone else
                        "none -- every group here is one the target also has"))
        banked = ev.get("banked") or []
        if banked:
            L.append("    already banked on this route: "
                     + ", ".join(f"{b['mol']}"
                                 + (f" {R.dollars(b['ln_price'])}"
                                    if b.get("ln_price") is not None else " unpriced")
                                 for b in banked[:8]))
        L.append("")

        menu, facts = menu_of(ev, mid), facts_of(ev, mid)
        fams: dict[tuple, list[int]] = {}
        for ci in sorted(facts):
            if ci in menu:
                fams.setdefault(family_key(facts[ci]), []).append(ci)
        # Families with a ranked candidate first, then the rest; inside a family, menu order.
        def _fam_rank(kv):
            key, cs = kv
            taken = [c for c in cs if c in order]
            return (0 if taken else 1, min(order.index(c) for c in taken) if taken else 0,
                    min(cs))
        # What `diverging from` must attend to, spelled out. A support block that withholds a fact
        # the form demands leaves the turn unwritable, and this slot demands a claim about EARLIER
        # CONTEXT -- which route already expanded what. The ledger lines are in the observation the
        # model is reading; the reaction KIND beside each is not, and that is what this adds.
        hist = history_reactions(ep, k) if k is not None else []
        L.append("ALREADY EXPANDED ON THIS BOARD -- what `diverging from` has to attend to")
        if hist:
            mine_keys = {family_key(facts.get(ci) or {}) for ci in order}
            for h in hist:
                key = h["family"]
                if key[0] == "unmeasured":
                    say = "the board records no mapping for it -- name it by number only"
                else:
                    say = family_label(key)
                same = "   << SAME KIND as a cut you are ranking" if key in mine_keys \
                    and key[0] != "unmeasured" else ""
                L.append(f"  {h['rid']:<4} {h['ref']:<9} [{say}]{same}")
            # WHICH LEVEL, PER ROUTE GROUP. The rule is not a preference and not a judgement
            # call: `diverging_wrong_level` refuses a group that cites candidates when a
            # same-kind reaction IS on the ledger, and refuses one that cites the ledger when
            # none is. Both directions are fatal and both are decidable from the family keys
            # already computed above -- so the brief says which, per group, instead of stating
            # the rule and leaving the match to be found.
            L.append("  WHICH LEVEL EACH GROUP MUST USE -- citing the wrong one is refused "
                     "either way:")
            for pos, ci in enumerate(order):
                mine_k = family_key(facts.get(ci) or {})
                hits = [h["rid"] for h in hist
                        if h["family"] == mine_k and mine_k[0] != "unmeasured"]
                if hits:
                    L.append(f"    route {pos + 1} (c{ci}): the LEDGER -- "
                             f"`diverging from: {hits[0]} | same step, "
                             f"{family_label(mine_k)} | <why this candidate anyway>`")
                elif len(order) > 1:
                    others = [f"c{x}" for x in order if x != ci][:2]
                    L.append(f"    route {pos + 1} (c{ci}): the CANDIDATES -- nothing expanded "
                             f"is this kind, so `diverging from: {', '.join(others)} | <what "
                             f"they are> | <what this one does instead>`")
                else:
                    L.append(f"    route {pos + 1} (c{ci}): `diverging from: none | <why there "
                             f"is nothing yet to diverge from>`")
        else:
            L.append("  nothing yet: this is the first step on the board, so no route can cite "
                     "the ledger. Argue at the CANDIDATE level -- what makes the cuts this "
                     "call ranks different bets -- or `diverging from: none | <why>` when it "
                     "ranks only one.")
        L.append("")

        # WHAT ACTUALLY SEPARATED THE HEAD OF THIS RANKING. Without it the reason for an
        # action the teacher did not take has to be inferred from a screen that shows every
        # number and marks none of them decisive -- and the inference goes wrong in a
        # specific, one-sided way: where the search broke a tie on PRICE, the draft names
        # `structure`, whose own definition is "no number decided this".
        dec = ((ev.get("decided_by") or {}).get(mid) or {})
        if dec.get("axis") and dec.get("values"):
            AX = {"p": "plausibility", "rt": "round-trip", "price": "price",
                  "q": "confidence"}
            ax = AX.get(dec["axis"], dec["axis"])
            lead = ", ".join(f"c{i}" for i in dec.get("leaders") or [])
            taken = dec.get("taken")
            # ONE SHAPE, ONE LINE, AND IT ALWAYS ENDS IN A MEASURE THE FORM ACCEPTS.
            #
            # A brief that branches, some branches carrying a negative -- "LEAVE `ordered by`
            # OFF", "Do NOT name round-trip" -- is readable on a turn that ranks ONE molecule.
            # On a turn that ranks two it is not: molecule A is told to name an axis and
            # molecule B is told to name none, and the draft has to apply different rules to two
            # blocks of one document. Such turns then fail to fill, and multi-molecule rank
            # turns are the ones the breadth-first corpus exists to teach. The fault is the
            # instruction being inconsistent ACROSS the molecules of one turn.
            #
            # So every case below emits one line of the same shape and names a measure that
            # `MEASURE_TOKENS` accepts. Where no number decided it the answer is `structure`,
            # which exists for exactly that -- "used only when none of the four numbers is what
            # actually decided it" -- and which has no ranking to contradict.
            say = None
            if dec["axis"] == "p":
                reads = ", ".join(f"c{k} {v:g}" for k, v in dec["values"].items())
                say = f"plausibility separated the leaders on its own -- {reads}."
            elif not dec.get("taken_in_leaders", True) and len(dec.get("leaders") or []) > 1:
                # The action is not a choice on this menu at all: the molecule is usually
                # already solved and this rank gives it another route. The reason is on the
                # ledger, so `diverging from` carries it and `ordered by` says `structure`.
                say = (f"nothing on this menu decided it: {lead} tied and the call takes "
                       f"c{taken}, which is not among them -- this rank gives the molecule "
                       f"ANOTHER route. Put the reason in `diverging from`, against the "
                       f"reaction the ledger already has here, and order by `structure`.")
            elif dec["axis"] == "closure":
                got = ", ".join(f"c{k}" for k, v in dec["values"].items() if v)
                say = (f"of the tied leaders {lead}, only {got} has every piece already "
                       f"purchasable -- that, not price and not confidence, is what put "
                       f"c{taken} first. Order by `structure` and say in the `why` that its "
                       f"pieces are all bought, so the branch closes here.")
            elif dec["axis"] == "rt_missing":
                got = ", ".join(f"c{k}" for k, v in dec["values"].items() if v)
                say = (f"of the tied leaders {lead}, only {got} has a round-trip rank at all; "
                       f"the rest were measured and did not come back. Quote a rank only for "
                       f"the candidate that has one.")
            elif not dec.get("earned", True):
                say = (f"{lead} tie on plausibility and {ax} separates them, but {ax} does not "
                       f"point at c{taken}, which is the cut taken. Order by `structure` and "
                       f"argue the chemistry -- naming {ax} here is a claim the checks refuse.")
            else:
                fmt = (lambda v: R.dollars(v)) if dec["axis"] == "price" else (
                    lambda v: f"{v:g}")
                reads = ", ".join(f"c{k} {fmt(v)}" for k, v in dec["values"].items())
                say = (f"the leaders {lead} tie on plausibility, and {ax} separates them: "
                       f"{reads}. c{taken} is the one taken.")
            # NO AXIS RECOMMENDATION HERE. This block explains what separated the DP's TIED
            # LEADERS; `ordered by` is judged against `field_of` -- the whole described field --
            # and the two sets can disagree, so a recommendation from this one can be a
            # recommendation the checks refuse (plausibility decides the tied leaders while
            # `field_of` puts the head third on plausibility). So the PER ROUTE GROUP list below,
            # computed over the set the check actually uses, is the only place a measure is
            # recommended.
            L.append(f"WHAT DECIDED IT at {mid}: {say}"
                     "  (this is what separated the tied leaders -- the measure `ordered by` "
                     "may NAME is the per-group list further down, which is computed over the "
                     "whole field the check compares against)")
            L.append("")

        # The cut taken is also the cheapest on the menu, and by enough that the comparison
        # is worth making. Handed over rather than left to be found: left alone, the teacher
        # mentions a price only under `ordered by: price`, so the board's candidate prices
        # produce a corpus that argues from none of them.
        pn = ((ev.get("price_note") or {}).get(mid) or {})
        # Episodes rendered before the note carried piece prices have only the totals, and the
        # totals are the thing a draft may not name. Skip them rather than emit an instruction
        # whose figures would be refused on sight.
        if pn and "taken_piece" not in pn:
            pn = {}
        if pn:
            # WHERE THIS MAY BE WRITTEN IS NOT A STYLE NOTE -- it is the difference between a
            # draft that survives and one that is refused: a figure in the `why` of `ordered by:`
            # is what `deciding_number_in_why` refuses, so the figures go in `what it does:`.
            # THE BOARD ALREADY ADDS IT UP. Every candidate line carries its own total --
            # `<c1 q.282 p.979 rt1 $2.8k>` -- so the draft has nothing to compute and nothing
            # to invent; it quotes what the environment printed. `leak_posthoc` tells a quoted
            # total from an invented one -- `_printed_prices` exempts every price this turn
            # actually printed -- so the instruction does not have to steer around it.
            L.append(f"COST ON THE MENU: c{pn['taken']} totals "
                     f"{R.dollars(math.log(pn['taken_cost']))} against "
                     f"{R.dollars(math.log(pn['dearest']))} for the dearest cut here, "
                     f"{pn['spread']}x. Both are printed on their own candidate lines -- quote "
                     f"them exactly as the board writes them, in the `what it does:` of the "
                     f"candidate they belong to. Do NOT put a figure in the `why` of "
                     f"`ordered by:` (the number goes in `value:`), and do NOT change "
                     f"`ordered by:` to price unless WHAT DECIDED IT above says price is what "
                     f"separated the leaders.")
            L.append("")

        _tk = takes_of(turn).get(mid, 1)
        if order:
            now = [f"c{c}" for c in order[:_tk]]
            keep = [f"c{c}" for c in order[_tk:]]
            L.append(f"THIS CALL EXPANDS {', '.join(now)} on this turn"
                     + (f" and KEEPS {', '.join(keep)} on the board for a later turn"
                        if keep else " and keeps nothing back")
                     + f" -- so route{'s' if _tk > 1 else ''} 1"
                     + (f"-{_tk}" if _tk > 1 else "")
                     + " argue for branches that will exist before the next observation"
                     + (f", and route{'s' if len(keep) > 1 else ''} {_tk + 1}"
                        + (f"-{len(order)}" if len(keep) > 1 else "")
                        + " for branches the board is holding" if keep else ""))
            L.append("")

        L.append(f"CANDIDATES of {mid}, grouped by the kind of step they are")
        ranked_fams = collections.Counter(
            family_key(facts[ci]) for ci in order if ci in facts)
        for key, cs in sorted(fams.items(), key=_fam_rank):
            dup = ranked_fams.get(key, 0) > 1 and key[0] != "unmeasured"
            L.append(f"  [{family_label(key)}]"
                     + ("   << the call ranks MORE THAN ONE of these. They are the same KIND "
                        "of step, so the route groups for them take `diverging from: c<the "
                        "earlier one> | same step, ... | <which fragment or position differs>` "
                        "-- not a claim of different chemistry" if dup else ""))
            for ci in cs:
                c, f = menu[ci], facts[ci]
                mark = f"  <- RANKED {order.index(ci) + 1}" if ci in order else "  (not ranked)"
                L.append(f"    c{ci}   {_signals(c.get('signals') or {})}"
                         f"{_cut_side(c)}{mark}")
                if ci not in order and order and order[0] in menu:
                    L.append(f"        {_worse_on(c, menu[order[0]]).strip()}")
                # One line per precursor, each carrying its OWN purchasability and price. The
                # old shape was a positional mask -- `buyable -*` above two SMILES -- and a
                # reader had to align the marks with the fragments to know which one was for
                # sale, and a misaligned reading calls the wrong fragment "a cheap leaf". A mask
                # that has to be aligned is a code, and this is the same defect as
                # `28 -> 34 + 14`.
                L.append("         precursors:")
                for smi, buy, pr in zip(c.get("reactants") or [],
                                        (c.get("buyable") or []) + [None] * 9,
                                        (c.get("ln_price") or []) + [None] * 9):
                    if pr is not None:
                        tag = (("PURCHASABLE, " if buy else "must be made -- not purchasable, ")
                               + f"{R.dollars(pr)} per mmol")
                    elif buy:
                        tag = "PURCHASABLE, no price on record"
                    else:
                        tag = "must be made -- not purchasable"
                    L.append(f"           {smi}\n               {tag}")
                L.append(f"         bond          {_bond_phrase(f)}")
                L.append(f"         reaction      {_class_phrase(f)}")
                sh = _shape_phrase(f.get("shape") or {})
                if sh:
                    L.append(f"         what it does  {sh}")
                note = ann.get(_rxn_key(smi, c.get("reactants") or []))
                if note:
                    L.append(f"         note          {note}")
        # Which candidates would put a piece back that is already ABOVE this one. Computed, not
        # asked for: the SMILES are all on screen, so a student can reproduce the comparison,
        # but a planner does not reliably make it, and `would make X its own precursor` ends
        # the call as a refusal. Naming the trap here
        # is what lets the turn's DECIDING block prune it by name instead of ignoring it.
        traps = ({} if ep.get("_no_ancestors")
                 else cycle_traps(ep, k, mid) if k is not None else {})
        if traps:
            anc = ancestors_of(ep, k, mid)
            L.append("  ALREADY ABOVE THIS PIECE on its own branch -- putting any of these back "
                     "makes it its own precursor and the board REFUSES the call:")
            for xid, smi_a in sorted(anc.items()):
                L.append(f"    {xid}  {smi_a}")
            for ci in sorted(traps):
                L.append(f"    c{ci} REGENERATES {', '.join(traps[ci])} -- it cannot be taken, "
                         f"so do NOT give it a DISCONNECTIONS row: describing it beside the "
                         f"real cuts says a refused step was on the menu")
        rest = [ci for ci in sorted(menu) if ci not in facts]
        if rest:
            # Named but not described. A support block that printed only the measured candidates
            # would read as if the piece had only those disconnections, and a ranking argued
            # against a field that size is a ranking against a field the board did not show.
            L.append("  [not measured -- also on this molecule, scores only]")
            for ci in rest:
                c = menu[ci]
                nb = sum(1 for b in (c.get("buyable") or []) if b)
                L.append(f"    c{ci}   {_signals(c.get('signals') or {})}{_cut_side(c)}"
                         f"   {nb} of {len(c.get('reactants') or [])} precursors purchasable")
                L.append(f"           {' + '.join(c.get('reactants') or [])}")
        L.append("")
        # THE ROW LIST, SPELLED OUT, IN THE ORDER THE CHECKER WANTS IT. A line reading "the
        # DISCONNECTIONS rows are exactly these, in this order" after the RANKING contradicts
        # the gates it is supposed to help satisfy, both ways:
        #
        #   `cuts_rank_order` wants rows in ASCENDING CANDIDATE NUMBER, because a block written
        #   in rank order announces the answer in its first line; `c3 > c7 > c9` is the ranking.
        #   `no_alternative_described` wants a row for at least one candidate the call PASSES
        #   OVER, because a block describing only what was taken leaves nothing to compare
        #   against; "exactly these" says the opposite.
        #
        # The row list is computable here, so it is printed as a list to copy rather than a
        # rule to apply.
        alt = [ci for ci in sorted(facts)
               if ci not in order and ci not in (traps or {})]
        rows_want = sorted(set(order) | set(alt[:1]))
        L.append(f"  the call ranks: " + " > ".join(f"c{c}" for c in order)
                 + "   -- that is the RANKING, and it is said once, by DECIDING")
        L.append(f"  the DISCONNECTIONS rows are exactly: "
                 + ", ".join(f"c{c}" for c in rows_want)
                 + "  -- in that order, ascending, NOT the ranking's order"
                 + (f" (c{alt[0]} is there because the call passes over it and a block that "
                    f"describes only what was taken is refused)" if alt else ""))
        # Which measures actually put the head first, computed. `ordered by` has to name one that
        # does, and a teacher left inferring which guesses wrong. The head is the best on some
        # measure on most turns, so there is usually a true answer sitting here already. Not
        # always: some rank turns have a head candidate none of the four numbers favours -- ranked
        # for a reason the search's own recursive lookahead saw and none of p/rt/q/price carries
        # (see MEASURE_TOKENS on `structure`). Left silent, a teacher reaches for the numeric axis
        # that reads closest anyway ("round-trip" on a candidate merely TIED for best round-trip,
        # still refused by `applied_subcutoff` for the plausibility nothing rescued) rather than
        # the one slot built for exactly this. So this is computed and handed over the same way:
        # the thing that is checkable, not inferred.
        # A tie counts as "puts the head first" for the ordinary contradiction check (below
        # cut-off is not the issue), but not here: a candidate below the cut-off is vetoed
        # regardless of what `ordered by` names UNLESS that name is `structure`, so a tied
        # numeric axis buys this head nothing -- naming it still gets the turn refused, just
        # one check later. Below cut-off, "good" therefore means STRICTLY ahead, not tied, so
        # the hint below fires and points at `structure` instead of a number that reads like an
        # answer and is not one.
        # PER ROUTE GROUP, NOT PER TURN. DECIDING writes one group for EVERY candidate in
        # `order` and each group's `ordered by` has to name a measure that puts ITS OWN
        # candidate first among what is still unclaimed. Computed for the head only, every
        # group after the first would be a guess -- and guessing is what
        # `ordered_not_differentiating` and `ordered_same_objective` refuse. The head's list is
        # kept below in full because the head carries the `structure` fallback reasoning; this
        # adds the tails.
        if len(order) > 1:
            # PER ROUTE GROUP, AND BY THE CHECK'S OWN RULE, WHICH IS NOT THE HEAD'S RULE.
            # `ordered_not_differentiating` holds a TAIL group to "does this candidate beat the
            # HEAD on the measure it names" -- head-to-tail, not tail-against-the-whole-field.
            # The first cut of this block used `field_of` for every group, which is the head's
            # comparison, and it is strictly harder: it pushed almost every tail to `structure`,
            # a slot that cannot be refused and teaches nothing. The check even carries the
            # answer it wants in its own repair text ("Name ... -- it wins there"), so this
            # computes exactly that and hands it over.
            #
            # The HEAD's list is not repeated here -- it is computed below against `field_of`,
            # which is the set the head is judged over.
            L.append("  PER ROUTE GROUP, routes 2 onward -- `ordered by` has to name a measure "
                     "on which THAT candidate beats the head:")
            for pos, ci in enumerate(order[1:], start=2):
                wins, loses = [], []
                for ax in AXES:
                    mk = _axis_key(menu.get(ci) or {}, ax)
                    hk = _axis_key(menu.get(order[0]) or {}, ax)
                    if mk is None or hk is None:
                        continue
                    (wins if mk < hk else loses).append(AXIS_TOKEN[ax])
                if wins:
                    L.append(f"    route {pos} (c{ci}): " + ", ".join(wins)
                             + (f"   -- NOT " + ", ".join(loses) if loses else ""))
                else:
                    L.append(f"    route {pos} (c{ci}): it beats c{order[0]} on no number "
                             f"-- `structure`, and say what its cut does that the head's does "
                             f"not")
            L.append("  Two groups may not be kept for the same measure AT THE SAME READING; "
                     "`structure` is exempt, so several groups may use it.")
        head_p = (menu.get(order[0]) or {}).get("signals", {}).get("p")
        head_subcutoff = head_p is not None and head_p < CUTOFF
        # Compared against every candidate the draft can TALK about, not just the ranked ones.
        # `_axis_values` covers `order` alone, and many rank turns rank exactly one candidate -- so
        # on those the head is trivially "first" on every axis and the hint would say "may name any
        # of these ... plausibility" for a head that is the LOWEST on its menu. The draft then
        # rules other candidates out as "outranked on plausibility" when they are better on it,
        # which is what `ruled_out_contradiction` refuses. The comparison the draft is held to is
        # head-against-the-field, so that is the comparison this hint has to make.
        field = field_of(ev, mid, order)     # the same set decides_flat is judged over
        good, bad, tied_useless = [], [], []
        for ax in AXES:
            vals = _axis_values(menu, field, ax)
            if order[0] not in vals:
                continue
            best = min(vals.values())
            tied = sorted(c for c, v in vals.items() if v == best)
            if len(set(vals.values())) <= 1 and len(vals) > 1:
                # Level across everything described: it orders nothing, and `decides_flat`
                # refuses it. Recommending it anyway is the other half of the contradiction the
                # shared `field_of` was introduced to end, so a fully-flat axis is never offered,
                # whether or not the head is below the cut-off.
                tied_useless.append(f"{AXIS_TOKEN[ax]} (level on all of them)")
            elif vals[order[0]] != best:
                bad.append(f"{AXIS_TOKEN[ax]} (c{tied[0]} leads)")
            elif head_subcutoff and len(tied) > 1:
                other = next(c for c in tied if c != order[0])
                tied_useless.append(f"{AXIS_TOKEN[ax]} (tied with c{other})")
            else:
                good.append(AXIS_TOKEN[ax])
        # Printed as the TOKENS the slot takes, so `ordered by` is a copy rather than a
        # translation. The English names were here first and the teacher had to convert them.
        if good:
            # Ordered p, rt, price, confidence by AXES, and said out loud, because the list
            # alone was read as a menu of equals: the three scored axes come first and
            # `confidence` last, and `structure` is off the table while anything is here.
            three = [g for g in good if g != AXIS_TOKEN["q"]]
            L.append(f"  `ordered by` may name any of these -- each puts c{order[0]} first: "
                     + ", ".join(good))
            if three:
                L.append(f"  name one of these three -- they are the scored axes: "
                         + ", ".join(three)
                         + ("  (`confidence` is available but ranks below them)"
                            if len(three) < len(good) else ""))
            L.append("  `structure` is NOT available on this turn: a number decided it.")
        if bad:
            L.append(f"  naming any of these would contradict the call: " + ", ".join(bad))
        if tied_useless:
            L.append(f"  these only tie, and a tie does not clear the cut-off veto for a "
                     f"below-cut-off candidate: " + ", ".join(tied_useless))
        if not good:
            hf = facts.get(order[0]) or {}
            grounds = []
            if hf.get("named_tier") in ("applies", "applies+makes"):
                grounds.append(f"the named reaction ({', '.join(hf.get('named') or [])})")
            if hf.get("bond_measured"):
                grounds.append("its measured bond change")
            if grounds:
                L.append(f"  no number above puts c{order[0]} first -- `ordered by: structure` "
                         f"is the honest slot here, grounded in " + " and ".join(grounds)
                         + f"; naming a number instead (even one that merely ties) will not "
                         f"clear this candidate if its plausibility is below cut-off")
            else:
                L.append(f"  no number above puts c{order[0]} first, and it carries no "
                         f"named-reaction match or measured bond either -- name plausibility "
                         f"and say plainly that it reads low here; do not invent a reason")
        L.append("")

    if not L:
        return ""
    return ("Supporting information for this turn. It is yours, not the agent's: write blocks "
            "that CONTAIN what matters here, never a reference to having been given it.\n\n"
            + "\n".join(L).rstrip())


# ==================================================================== the chain
def _decision_line(turn: dict) -> str:
    """One turn's call, as one line of ledger. `rank ja -> c0 > c7 > c2`, `open s7`."""
    bits = []
    for a in (turn.get("actions") or []):
        if not isinstance(a, dict):
            continue
        t = (a.get("type") or "").lower()
        if t == "rank":
            order = " > ".join(f"c{c}" for c in (a.get("order") or []))
            bits.append(f"rank {a.get('mid')} -> {order or '(nothing)'}")
        elif t == "done":
            bits.append("done " + ", ".join(f"{m}:c{c}"
                                            for m, c in (a.get("choices") or {}).items()))
        elif t == "final":
            bits.append("hand over")
        elif a.get("mid"):
            bits.append(f"{t} {a['mid']}")
        else:
            bits.append(t)
    return "; ".join(bits) or "(no call)"


def ledger(ep: dict, upto: int) -> str:
    """Turns 0..upto-1 as one line each. Grows by APPENDING, which is the whole point.

    The freeform module replays the last few BOARDS verbatim and folds the rest at a block
    boundary, so the prompt's prefix is rewritten at every fold and each remembered turn costs a
    whole board. A ledger line says what the board replay was being kept for -- what was decided
    -- in one short line, and because turn k's ledger is a literal prefix of turn k+1's, the
    tokens in front of it are byte-identical from one turn to the next and vllm's prefix cache
    hits every turn.
    """
    rows = [f"  t{j:<3} {_decision_line(ep['turns'][j])}" for j in range(upto)]
    return "\n".join(rows)


def turn_prompt(ep: dict, k: int, a=None) -> list[dict]:
    """The conversation for turn k: the ledger, then the last few turns, then this board.

    History is (reasoning, decision), not (board, reasoning, decision): the board of a turn four
    back is re-derivable from what it decided, and the reasoning written for it already CONTAINS
    the chemistry that mattered -- which is the property this whole pipeline is built around.
    `--window-boards` puts the boards back for the most recent turns when that is worth paying
    for.
    """
    win = getattr(a, "history_turns", 4) if a is not None else 4
    win_boards = getattr(a, "window_boards", 0) if a is not None else 0
    turns = ep["turns"]
    msgs = [{"role": "system", "content": system_rank()}]
    dev = ep.get("developer") or ""
    if dev:
        msgs.append({"role": "user",
                     "content": "The board's rules, as the agent has them:\n\n" + dev})

    start = max(0, k - win)
    if start:
        msgs.append({"role": "user",
                     "content": "What you have decided so far, one line per turn:\n"
                                + ledger(ep, start)})
    for j in range(start, k):
        t = turns[j]
        if j >= k - win_boards:
            msgs.append({"role": "user", "content": f"[turn {j}] board:\n{t['env']}"})
        if t.get("thought"):
            msgs.append({"role": "assistant", "content": t["thought"]})
        msgs.append({"role": "user",
                     "content": f"[turn {j}] your call: {_decision_line(t)}"})

    t = turns[k]
    tail = [f"[turn {k}] board:\n{t['env']}"]
    rej = t.get("reject")
    if rej:
        # See traj_route_reasoning_board.turn_prompt for what this is: a real,
        # board-verified refusal the rejection-injection pass put in front of this exact turn.
        # Both the call and the board's answer already happened, so citing them is not
        # the kind of foresight the rest of this module refuses.
        tail.append(
            "\nBefore this turn's own actions, this same board refused a call:\n"
            f"  tried: {json.dumps(rej.get('actions'), ensure_ascii=False)}\n"
            f"  the board said: {rej.get('reason') or ''}\n"
            "That happened and is not hypothetical. If it bears on this turn's ranking or "
            "deciding, account for it in one clause -- what it ruled out or what it "
            "revealed -- and move on; if it does not, do not mention it.")
    sup = rank_support(ep, t, getattr(a, "annotations_map", None), k)
    if sup:
        tail.append("\n" + sup)
    tail.append("\nWrite the form for this turn, continuing from your earlier turns. "
                "Blocks only, nothing outside them, and do not say you were told what to do.")
    msgs.append({"role": "user", "content": "\n".join(tail)})
    return msgs


# ==================================================================== verify
# A violation is a claim ABOUT A PLACE. Everything below carries `where` -- the block, the field
# inside it, and the row when there is one -- because that is the whole reason the form exists:
# the repair prompt hands back the line, not the paragraph.
def V(code: str, where: str, detail: str, fix: str = "", fatal: bool = False) -> dict:
    return {"code": code, "where": where, "detail": detail, "fix": fix, "fatal": fatal}


# Register, imported rather than restated: one definition of "interface vocabulary" for both
# modules. `_JARGON` is a register problem salvage can rewrite; `_BAD` is a leak; the mid-episode
# set is a claim a turn inside the search has not earned; `_CEILING` is a depth budget the board
# stopped printing.
_JARGON = LEGACY._JARGON
_BAD = LEGACY._BAD
_BAD_MIDEPISODE = LEGACY._BAD_MIDEPISODE
_CEILING = LEGACY._CEILING
_ATOM_INDEX = re.compile(r"\b[A-Z][a-z]?:\d+\b")

# The call ceiling, which the board no longer prints and the developer message no longer
# documents. It was the agent's own vocabulary while `budget N of M` sat on every observation;
# with the line gone, a draft that reasons about calls remaining is reasoning from a number it
# was never shown, and the whole point of removing it was that the search is scaled by raising N
# rather than by fitting inside it. Banned on the same terms as the depth ceiling.
# The call's own order, cited as a reason. The teacher is SHOWN the ranking -- it has to be, to
# write the rows in that order -- and a draft that argues "the board has c9 first, so I take it"
# is transcribing the answer rather than reasoning to it. Same defect as quoting the oracle
# label, one step removed.
_ORDER_APPEAL = [r"\bthe board (?:has|puts|ranks|already has)\b",
                 r"\bthe (?:call|ranking|order|declaration) (?:ranks|declares|puts|has|names)\b",
                 r"\b(?:since|because|as) c\d+ is (?:ranked )?first\b",
                 r"\bc\d+ is (?:ranked|declared) first\b",
                 r"\bfollow(?:ing)? the (?:declared )?(?:order|ranking)\b"]

_BUDGET_TALK = [r"\bbudget\b", r"\bcalls? (?:left|remaining|to spare)\b",
                r"\b(?:call|expansion) (?:ceiling|cap|limit|allowance)\b",
                r"\b\d+ of \d+ calls\b", r"\bspend(?:ing)? (?:my|the) (?:remaining )?calls\b"]


def _words(s: str) -> int:
    return len((s or "").split())


def _printed_prices(ev: dict) -> set:
    """Every price string the board PRINTED on this turn, candidate totals included.

    The mid-episode dollar ban exists to stop a draft naming a route total it worked out for
    itself. It was never meant to stop the draft quoting the screen -- and once the board began
    printing a candidate total on every line (`<c0 q.528 p.702 rt1 $1.0k>`), the two became the
    same string, so the teacher would be refused for reading back a number the environment had
    just computed for it. `factcheck` already builds this same set to check a cited price
    against; this reuses the rule so the two cannot disagree.
    """
    out = set()
    try:
        from board import factcheck as FC_, render as R_
        rows = FC_._rows(ev)
    except Exception:                                              # noqa: BLE001
        return out
    for _, c, _ in rows:
        lp = [x for x in (c.get("ln_price") or []) if x is not None]
        out.update(R_.dollars(x) for x in lp)
        if lp and len(lp) == len(c.get("ln_price") or []):
            out.add(R_.dollars(math.log(sum(math.exp(x) for x in lp))))
    for b in (ev.get("banked") or []):
        if b.get("ln_price") is not None:
            out.add(R_.dollars(b["ln_price"]))
    for m in (ev.get("mols") or {}).values():
        if m.get("ln_price") is not None:
            out.add(R_.dollars(m["ln_price"]))
    for r in (ev.get("routes") or []):
        if r.get("cost_usd") is not None:
            out.add(R_.cost_of_usd(r["cost_usd"]))
    return out


def _register(body: str, where: str, allowed=()) -> list[dict]:
    """The word-level defects, wherever they appear. Same patterns as the freeform gate."""
    out = []
    for p in _BAD:
        if re.search(p, body, re.I):
            out.append(V("leak_phrase", where, f"says {p!r}, which quotes the answer or the "
                                               f"instruction", "Argue it from the chemistry.",
                         fatal=True))
    # A price in dollars, per mmol, on one fragment. The support block prints it that way --
    # "PURCHASABLE, ln$4.94 (about $139 per mmol)" -- so a draft quoting it is quoting what it
    # was given, and the mid-episode dollar ban exists to stop a route TOTAL being named, not a
    # leaf's price. The unit is what separates them.
    # `[kM]?` is not decoration: `render.dollars` writes anything over a thousand as `$1.2k`
    # or `$273k`, which is exactly how a board prints the pieces this ban is meant to allow.
    # Without the suffix the scrub stops at the digits, the `$1.2k` survives into the check,
    # and a draft quoting a piece price the board had just shown it is refused as if it had
    # totalled a route.
    scrubbed = re.sub(r"\$\s?[\d,.]+[kMB]?\s*(?:per\s+mmol|/\s*mmol)", " ", body, flags=re.I)
    # ...and every price this turn actually printed, candidate totals included. Quoting the
    # screen is not inventing a total; see `_printed_prices`.
    for tok in sorted(allowed, key=len, reverse=True):
        if tok:
            scrubbed = scrubbed.replace(tok, " ")
    for p in _BAD_MIDEPISODE:
        if re.search(p, scrubbed, re.I):
            out.append(V("leak_posthoc", where, f"matches {p!r} -- a judgement about how the "
                                                f"episode ends, made inside it",
                         "A turn may cite a fragment's ln$ and may not total a route.",
                         fatal=True))
    for p in _CEILING:
        if re.search(p, body, re.I):
            out.append(V("depth_ceiling", where, f"matches {p!r}: the board gives a depth and "
                                                 f"no ceiling", "Argue from how much the cut "
                         "has to earn at this depth, not from levels left.", fatal=True))
    for p in _ORDER_APPEAL:
        if re.search(p, body, re.I):
            out.append(V("order_appeal", where,
                         f"matches {p!r}: it gives the ranking as the reason for the ranking",
                         "Say what about the chemistry puts that cut first.", fatal=True))
    for p in _BUDGET_TALK:
        if re.search(p, body, re.I):
            out.append(V("budget_ceiling", where,
                         f"matches {p!r}: the board shows no call ceiling",
                         "Argue from the chemistry, not from how many calls are left.",
                         fatal=True))
    for p in _JARGON:
        if re.search(p, body, re.I):
            out.append(V("jargon", where, f"narrates the interface ({p!r})",
                         "Name the chemistry the list contains, not the list.", fatal=True))
    if _ATOM_INDEX.search(body):
        out.append(V("atom_index", where, "names a bond by its atom-map index",
                     "Name the element pair and say where it sits -- `C-N formed at the "
                     "carbonyl carbon`.", fatal=True))
    return out


def _label_quotes(text: str, ev: dict) -> list[dict]:
    """The DP's oracle answer, quoted. Transcription, not reasoning."""
    labels = {str((c or {}).get("label")) for m in (ev.get("menus") or {}).values()
              for c in (m.get("candidates") or []) if (c or {}).get("label") is not None}
    low = (text or "").lower()
    return [V("label_quote", "document", f"quotes the oracle label {x!r}",
              "The label steers you; it is never citable.", fatal=True)
            for x in sorted(labels) if x and len(x) > 3 and x.lower() in low]


def verify(text: str, ep: dict, k: int, a=None) -> dict:
    """Is this draft the form, and is it written in the right register?

    Everything here is decidable from the WORDS -- shape, vocabulary, arithmetic on the document.
    Whether the sentences are TRUE of the turn is `factcheck` below, against the same evidence
    dict `rank_support` wrote the prompt from.
    """
    turn = ep["turns"][k]
    ev = turn.get("evidence") or {}
    orders = ranked_orders(turn)
    doc = parse_document(text)
    by = doc["by_name"]
    viol: list[dict] = []

    # ---- shape ---------------------------------------------------------------------------
    for name in BLOCKS:
        if name not in by:
            viol.append(V("schema_missing", f"<{name}>", f"has no <{name}> block",
                          "Every block is required, in order.", fatal=True))
    for name in by:
        if name not in BLOCKS:
            viol.append(V("schema_extra", f"<{name}>",
                          f"carries a <{name}> block that is not part of the form", fatal=True))
    # Order, allowing a multi-molecule turn to group PER MOLECULE. The spec asked for all the
    # DISCONNECTIONS blocks and then all the DECIDING blocks, and the sort below enforced exactly
    # that -- so a two-molecule turn written as
    #     MOLECULE / DISCONNECTIONS j2 / DECIDING j2 / DISCONNECTIONS gm / DECIDING gm
    # was refused for the ORDER while every block in it was right. That grouping is the more
    # readable of the two and the one the model reaches for; nothing downstream depends on the
    # global grouping, because every addressed block carries its own mid. So the rule is now:
    # MOLECULE first, and within each molecule its DISCONNECTIONS before its DECIDING.
    seq = [n for n in doc["names"] if n in BLOCKS]
    ok_order = all(n == "molecule" for n in seq[:1])
    if ok_order:
        for b in doc.get("blocks") or []:
            if b["name"] != "deciding" or not b["mid"]:
                continue
            before = [x for x in (doc.get("blocks") or [])
                      if x is not b and (doc["blocks"].index(x) < doc["blocks"].index(b))]
            if not any(x["name"] == "disconnections" and x["mid"] == b["mid"] for x in before):
                ok_order = False
                break
    if seq and (not ok_order or "molecule" in seq[1:]):
        viol.append(V("schema_order", "document",
                      "puts the blocks in the order " + " ".join(seq)
                      + " -- MOLECULE first, then each molecule's DISCONNECTIONS before its "
                        "own DECIDING", fatal=True))
    if doc["stray"]:
        viol.append(V("schema_stray", "document",
                      "writes text outside the blocks: " + doc["stray"][:120],
                      "The first line of the answer is `MOLECULE`.",
                      fatal=True))
    if not doc["blocks"]:
        return _verdict(viol, text, doc)

    # ---- addresses -----------------------------------------------------------------------
    # `<cuts ja>` is the whole defence against a draft that describes one molecule and ranks
    # another: with the id in the tag it is a set comparison rather than a clause-level
    # heuristic, and the heuristic is what the freeform gate needed.
    for name in [b for b in BLOCKS if b in ADDRESSED]:
        got = {b["mid"] for b in by.get(name, [])}
        if None in got:
            viol.append(V("schema_unaddressed", f"<{name}>",
                          f"writes <{name}> with no molecule id",
                          f"Write `{name.upper()}"
                          + (" OF" if name == "disconnections" else "")
                          + f" {next(iter(orders), 'MID')}`.",
                          fatal=True))
            got.discard(None)
        for mid in sorted(got - set(orders)):
            viol.append(V("address_unranked", f"<{name} {mid}>",
                          f"writes a <{name}> block for {mid}, which this turn does not rank",
                          "Only the molecules the call ranks get blocks.", fatal=True))
        for mid in sorted(set(orders) - got):
            viol.append(V("address_missing", f"<{name} {mid}>",
                          f"ranks {mid} and writes no <{name}> block for it", fatal=True))

    # ---- frame ---------------------------------------------------------------------------
    for b in by.get("molecule", []):
        kv, bad = parse_keyed(b["body"], MOL_KEYS)
        for line in bad:
            viol.append(V("molecule_malformed", "MOLECULE",
                          f"has a line that is not one of its fields: {line[:90]!r}",
                          "target: / this piece: / not in the target:, one each.", fatal=True))
        for key in MOL_KEYS:
            if key not in kv:
                viol.append(V("molecule_missing", f"MOLECULE {key}", f"has no `{key}:` line",
                              fatal=True))
            elif not kv[key]:
                viol.append(V("molecule_empty", f"MOLECULE {key}", f"leaves `{key}:` empty",
                              fatal=True))
            elif _words(kv[key]) > CAP["molecule"]:
                viol.append(V("molecule_long", f"MOLECULE {key}",
                              f"runs {_words(kv[key])} words against a {CAP['molecule']}-word "
                              f"ceiling -- it is a phrase, not a sentence"))
        viol += _register(b["body"], "MOLECULE", _printed_prices(ev))

    # ---- cuts ----------------------------------------------------------------------------
    for b in by.get("disconnections", []):
        where = f"DISCONNECTIONS {b['mid']}"
        rows, bad = parse_cuts(b["body"])
        for line in bad:
            viol.append(V("row_malformed", where, f"has a line that is not a row: {line[:90]!r}",
                          "c<n> | bond: ... | class: ... | split: ...", fatal=True))
        if not rows:
            viol.append(V("cuts_empty", where, "has no candidate rows", fatal=True))
        menu = menu_of(ev, b["mid"])
        seen = set()
        for r in rows:
            rw = f"{where} row c{r['c']}"
            if r["c"] in seen:
                viol.append(V("row_duplicate", rw, f"writes c{r['c']} twice", fatal=True))
            seen.add(r["c"])
            if menu and r["c"] not in menu:
                viol.append(V("row_unknown", rw,
                              f"writes a row for c{r['c']}, which is not a candidate of "
                              f"{b['mid']}", fatal=True))
            own = _words(r["bond"]) + _words(r["split"])
            if own > CAP["row"]:
                viol.append(V("row_long", rw, f"spends {own} words on bond and split against a "
                                              f"{CAP['row']}-word ceiling -- they are phrases, "
                                              f"not sentences"))
            for field, lab in (("bond", "bond"), ("cls", "class"), ("split", "split")):
                if not (r[field] or "").strip():
                    viol.append(V("row_field_empty", rw, f"leaves the {lab} field empty",
                                  fatal=True))
            # The bond field must carry an element pair or say it was not measured. A phrase
            # with neither -- "the bond at the benzylic position" -- names a place and no bond,
            # and there is nothing in it for the mapping to agree or disagree with.
            if r["bond"] and not _NOT_MEASURED.match(r["bond"]):
                pp = parse_bond_pairs(r["bond"])
                if not any(pp[side] for side in BOND_SIDES):
                    viol.append(V("bond_no_pair", rw,
                                  f"gives no element pair for c{r['c']}: {r['bond'][:70]!r}",
                                  "Write the pair first -- `C-N formed at the carbonyl "
                                  "carbon`.", fatal=True))
            viol += _register(r["raw"], rw, _printed_prices(ev))
        # Which candidates the block must cover, read off the CALL rather than off a role word
        # the rows no longer carry. Every ranked candidate needs a description -- a route group
        # below argues from it -- and at least one candidate the call passes over needs one too,
        # or the block only ever describes what was taken and there is nothing to compare.
        # Row ORDER is not checked: with the ranking gone from these lines their sequence makes
        # no claim, and the block is meant to read as a description of the menu.
        order = orders.get(b["mid"]) or []
        seen = {r["c"] for r in rows}
        missing = [c for c in order if c not in seen]
        if order and missing:
            viol.append(V("cuts_missing", where,
                          "describes " + (", ".join(f"c{c}" for c in sorted(seen)) or "nothing")
                          + "; the call ranks " + ", ".join(f"c{c}" for c in order)
                          + " and every one of them needs a row", fatal=True))
        # Rows in the BOARD's order, ascending candidate index -- not the call's ranking.
        # Rank order is derived from the ANSWER, so a description block written in it hands the
        # decision over in its first line and everything after is post-hoc. Menu order is
        # derived from the observation the model is already looking at, so it leaks nothing new:
        # rank order puts the taken cut first by construction, while ascending index puts it
        # first only where the board itself already prints it first.
        got_rows = [r["c"] for r in rows]
        if got_rows != sorted(got_rows):
            viol.append(V("cuts_rank_order", where,
                          "lists its rows " + ", ".join(f"c{c}" for c in got_rows)
                          + "; the block describes the menu, so it goes in the board's order, "
                          + ", ".join(f"c{c}" for c in sorted(got_rows)),
                          "Ascending candidate number. The ranking is said once, by DECIDING's "
                          "route groups.", fatal=True))
        others = sorted(c for c in seen if c not in order)
        if order and not others:
            viol.append(V("no_alternative_described", where,
                          "describes only the candidates the call ranks; give at least one it "
                          "passes over, so there is something to compare against",
                          fatal=True))
        if len(others) > CAP["weighed"]:
            viol.append(V("row_many", where,
                          f"describes {len(others)} candidates the call does not rank, against "
                          f"a ceiling of {CAP['weighed']}"))

    # ---- weigh ---------------------------------------------------------------------------
    for b in by.get("deciding", []):
        where = f"DECIDING {b['mid']}"
        routes, _, bad = parse_routes(b["body"])
        for line in bad:
            viol.append(V("deciding_malformed", where,
                          f"has a line that is not one of its fields: {line[:90]!r}",
                          "route <i> -- c<n>, then `diverging from: ... | ... | ...` and "
                          "`ordered by: <measure> | <why>`. `ruled out` and `leaves to make` "
                          "are not fields of this block any more.", fatal=True))
        if not routes:
            viol.append(V("deciding_missing", where,
                          "has no `route <i> -- c<n>` group",
                          "One group per candidate the call ranks, in its order.", fatal=True))
        for rt in routes:
            if rt["diverging"] is None:
                viol.append(V("deciding_missing", f"{where} route {rt['label']}",
                              "has no `diverging from:` line",
                              "`diverging from: none | <why>` while nothing is expanded.",
                              fatal=True))
        ordered = routes[0]["ordered"] if routes else None
        if ordered is not None:
            if ordered["token"] not in MEASURE_TOKENS:
                viol.append(V("axis_unknown", f"{where} ordered by",
                              f"names {ordered['token']!r} as the measure; it is one of "
                              + ", ".join(sorted(set(MEASURE_TOKENS))), fatal=True))
            if not ordered["why"]:
                viol.append(V("deciding_empty", f"{where} ordered by",
                              "names a measure and says nothing about what it means here",
                              fatal=True))
            elif _words(ordered["why"]) > CAP["deciding"]:
                viol.append(V("deciding_long", f"{where} ordered by",
                              f"runs {_words(ordered['why'])} words against a "
                              f"{CAP['deciding']}-word ceiling"))
        viol += _register(b["body"], where, _printed_prices(ev))

    viol += _label_quotes(text, ev)
    return _verdict(viol, text, doc)


def _verdict(viol: list[dict], text: str, doc: dict) -> dict:
    """One shape for every exit from `verify`.

    `ok` is "nothing at all was wrong" and is what decides whether a repair round is worth
    spending. `blocking` is "something was wrong that refuses the draft" and is what decides
    whether the turn is kept. They are different questions, and collapsing them would let an
    advisory defect such as a long row cost an episode a link.
    """
    block = [v for v in viol if v["fatal"]]
    return {"ok": not viol, "violations": viol,
            "codes": sorted({v["code"] for v in viol}),
            "fatal": sorted({v["code"] for v in block}),
            "blocking": bool(block),
            "advisory": sorted({v["code"] for v in viol if not v["fatal"]}),
            "jargon_only": bool(block) and {v["code"] for v in block} == {"jargon"},
            "words": _words(text), "doc": doc}


# ==================================================================== factcheck
# Two halves. The first is `board.factcheck` unchanged -- the same checks, the same FATAL set,
# the same repair text -- run over the whole draft and then ATTRIBUTED to a block, so a
# violation the freeform gate could only report gets an address here. The second is the set of
# checks the form makes exact: a field compared against the fact that field is about.

# Which of the shared checks can be run against a BLOCK BODY in isolation to find out where a
# whole-document violation came from. `action_match`, `handover`, `completeness`, `repetition`
# and `quantifier` are excluded on purpose: they are claims about the draft as a whole, and
# scoring them on a fragment invents violations the document does not have.
_ATTRIBUTABLE = {"numbers", "q_argmax", "cutoff", "named", "named_conflict", "ring_identity",
                 "fg_identity", "purchasable", "shape", "counts", "rt", "route_floor"}


def _attribute(doc: dict, ev: dict, kind: str, ctx: dict, codes: set) -> dict:
    """{code: the block it fires in}, by re-running the safe checks per block.

    Regex and cached RDKit only, so this is free next to the call that produced the draft. A
    code that fires nowhere in isolation keeps no address and is reported against the document.
    """
    if FC is None or not codes:
        return {}
    out: dict[str, str] = {}
    for b in doc.get("blocks") or []:
        where = f"<{b['name']}{(' ' + b['mid']) if b['mid'] else ''}>"
        body = b["body"]
        if b["name"] == "disconnections":
            body = re.sub(r"(reaction\s*:\s*)(.*?)(\s*\|)", r"\1\3", body, flags=re.I)
        try:
            v = FC.check(body, ev, kind, prev_thoughts=ctx.get("prev_thoughts") or (),
                         actions=ctx.get("actions"), observation=ctx.get("observation"),
                         seen_smiles=ctx.get("seen_smiles") or (), enable=_ATTRIBUTABLE)
        except Exception:                                              # noqa: BLE001
            continue
        for c in v.get("codes") or []:
            out.setdefault(c, where)
    return out


def mask_class_fields(text: str, doc: dict) -> str:
    """The draft with every `reaction:` field blanked, for the identity checks only.

    A `class:` field holds a reaction NAME, and a name legitimately contains words that are
    functional groups and ring classes: "Oxidation or Dehydrogenation of Alcohols to Aldehydes
    and Ketones" is what the template corpus calls the step, and `board.factcheck`'s group check
    reads the word `ketone` in it as a claim that some molecule on the turn has a ketone. On a
    paragraph the two are indistinguishable and the false rejection is the price of the check;
    with the name in its own field they are not, and the field is already verified where it
    should be -- against the template tier that earned it, by `class_unearned` and `class_wrong`
    below. So the identity checks see the draft without it.

    Blanked to the same LENGTH so every other check's offsets, and the repetition ratio, are
    unchanged.
    """
    out = list(text)
    for b in doc.get("blocks") or []:
        if b["name"] != "disconnections":
            continue
        for m in re.finditer(r"reaction\s*:\s*(.*?)\s*\|", b["body"], re.I):
            frag = m.group(1)
            i = text.find(frag, b["start"])
            if i >= 0 and frag.strip():
                out[i:i + len(frag)] = " " * len(frag)
    return "".join(out)


# The shared purchasability check fires on the WORD, not on the claim. "no route through them
# reaches purchasable material" is the board's own reason for ruling a candidate out, and it is
# read as an assertion that the pieces are for sale -- a draft saying the opposite of what the
# check hears. So a violation whose every trigger
# sits behind a negation is dropped. It is the same shape as the class-field masking: the check
# is right about what it measures and wrong about where it looked.
_BUY_WORD = re.compile(r"\b(purchasable|buyable|commercial|off the shelf|already banked)\b", re.I)
_NEGATED = re.compile(r"\b(no|not|never|cannot|can't|without|neither|nor|none|nothing)\b"
                      r"|\b(?:do|does|did)(?:es)?n?o?t\b|\bfails? to\b", re.I)


def _drop_negated_purchasable(shared: dict, text: str) -> None:
    if FC is None:
        return
    hits = [v for v in (shared.get("violations") or []) if v["code"] == "purchasable"]
    if not hits:
        return
    spans = list(_BUY_WORD.finditer(text or ""))
    if not spans:
        return
    # Every mention negated within the 60 characters before it -- the clause it sits in.
    if not all(_NEGATED.search(text[max(0, m.start() - 60):m.start()]) for m in spans):
        return
    shared["violations"] = [v for v in shared["violations"] if v["code"] != "purchasable"]
    shared["codes"] = sorted({v["code"] for v in shared["violations"]})
    shared["fatal"] = sorted(c for c in shared["codes"] if c in FC.FATAL)
    shared["ok"] = not shared["violations"]


_IDENTITY = {"fg_identity", "ring_identity"}


def _drop_named_field_identity(shared: dict, text: str, doc: dict, ev: dict, kind: str,
                               ctx: dict) -> None:
    """Remove identity violations whose only evidence is a word inside a `class:` field."""
    if FC is None:
        return
    hits = [v for v in (shared.get("violations") or []) if v["code"] in _IDENTITY]
    if not hits:
        return
    try:
        again = FC.check(mask_class_fields(text, doc), ev, kind,
                         prev_thoughts=ctx.get("prev_thoughts") or (),
                         actions=ctx.get("actions"), observation=ctx.get("observation"),
                         seen_smiles=ctx.get("seen_smiles") or (), enable=_IDENTITY)
    except Exception:                                                  # noqa: BLE001
        return
    real = {(v["code"], v["detail"]) for v in (again.get("violations") or [])}
    shared["violations"] = [v for v in shared["violations"]
                            if v["code"] not in _IDENTITY or (v["code"], v["detail"]) in real]
    shared["codes"] = sorted({v["code"] for v in shared["violations"]})
    shared["fatal"] = sorted(c for c in shared["codes"] if c in FC.FATAL)
    shared["ok"] = not shared["violations"]


def _asserts_bond(field: str) -> bool:
    return not _NOT_MEASURED.match(field or "")


def _asserts_class(field: str) -> bool:
    return not _NO_CLASS.match(field or "")


_STOP = {"with", "and", "the", "of", "to", "a", "an", "reaction", "synthesis", "type"}


def _names_match(field: str, names: list[str]) -> bool:
    """Does the class field name one of the reactions the template run actually earned?

    Token overlap, not string equality: `Williamson Ether Synthesis` is legitimately written
    `Williamson etherification` and `{Williamson ether}` in the corpus itself. One shared
    content word of four letters or more is the bar -- enough to separate a Williamson from a
    Suzuki, which is the confusion that matters, and loose enough not to reject a
    paraphrase.
    """
    ft = {w for w in re.findall(r"[a-z]{4,}", (field or "").lower()) if w not in _STOP}
    if not ft:
        return False
    for n in names or []:
        nt = {w for w in re.findall(r"[a-z]{4,}", n.lower()) if w not in _STOP}
        if nt & ft:
            return True
    return False


# A clause that names several candidates and asserts ONE thing about them asserts it about
# each. `board.factcheck` attributes a claim only when a clause names exactly one candidate --
# a deliberate conservatism, since "c0 at rt 1 beats c2" attaches the number to c0 alone -- so
# "c7 and c2 sit at rt dash" would go through with c2 at rt 1.
# Distribution is safe only over a CONJOINED RUN with no comparative in the clause, which is
# what separates "c7 and c2 sit at rt dash" from "c0 at rt 1 is better than c2".
_CONJOINED = re.compile(r"\bc\d\b(?:\s*,\s*c\d\b)*\s*(?:,\s*)?(?:and|or|nor)\s+c\d\b", re.I)
_COMPARATIVE = re.compile(r"\b(than|over|against|unlike|versus|vs\.?|compared|beats?|ahead of|"
                          r"rather than|instead of|where(?:as)?)\b", re.I)


def _distribute_group_claims(text: str) -> str:
    """Every multi-candidate clause rewritten into one clause per candidate.

    The result is not a draft and is never kept -- it exists so `board.factcheck`'s own
    clause-attributed checks can run on claims they otherwise skip, with their own patterns and
    their own thresholds. Anything they find in it is true of the original.
    """
    if FC is None:
        return ""
    out = []
    for cl in FC._clauses(text or ""):
        run = _CONJOINED.search(cl)
        if not run or _COMPARATIVE.search(cl):
            continue
        # The segment is the conjoined subject and the predicate that follows it, bounded on
        # BOTH sides by candidates outside the run. Either bound left off produces a false
        # positive that the other does not:
        #   left  -- "c0 reads forwards, c7 and c2 sit at rt —" taken whole distributes c0's
        #            verdict onto c7, the opposite of what the draft said;
        #   right -- "c0 and c2 read forwards, c7 rt —" taken to the end of the clause
        #            distributes the forward claim onto c7, which is the dash.
        cs = FC._cands_in(run.group(0))
        if len(cs) < 2:
            continue
        nxt = re.search(r"\bc\d\b", cl[run.end():])
        seg = cl[run.start():run.end() + (nxt.start() if nxt else len(cl))]
        for c in cs:
            out.append(re.sub(r"\bc\d\b", f"c{c}", seg).strip().rstrip(".;,") + ".")
    return " ".join(out)


def _c_group_claims(text: str, ev: dict, ctx: dict, where: str) -> list[dict]:
    """The shared clause checks, run over the distributed form of a grouped claim."""
    if FC is None:
        return []
    spread = _distribute_group_claims(text)
    if not spread:
        return []
    try:
        v = FC.check(spread, ev, ctx.get("kind") or "rank",
                     prev_thoughts=(), actions=ctx.get("actions"),
                     seen_smiles=ctx.get("seen_smiles") or (),
                     observation=ctx.get("observation"),
                     enable={"rt", "cutoff", "q_argmax", "numbers"})
    except Exception:                                                  # noqa: BLE001
        return []
    return [V("group_" + x["code"], where,
              x["detail"] + " -- said of several candidates at once, and it is not true of "
                            "every one of them",
              x.get("fix", ""), fatal=True)
            for x in (v.get("violations") or [])]


# ------------------------------------------------------------------ repetition, prose only
# `board.factcheck._c_repetition` splits a draft on sentence punctuation. A form has none: the
# whole document comes back as ONE unit, and because the field labels are the same eight words
# every turn the bag-of-words overlap with any earlier turn is structurally near 1, so on a form
# it fires as a false positive.
#
# It is replaced, and NOT by the same check over the ledger. A ledger is SUPPOSED to repeat: the
# target line is the same phrase every turn of an episode by design, `bond: not measured` is the
# prescribed answer whenever the mapping fails, and `split: 28 -> 21 + 8` recurs because the
# arithmetic recurs. Scoring those as boilerplate would punish the form for working. The block
# that must not repeat is the one that is an argument, so repetition is measured on `<why>`
# alone -- which is also the only block where a repeat means what the check thinks it means.
_REP_MIN_WORDS = 6           # content words before a `why` can count as boilerplate


def _why_of(text: str) -> str:
    """The ARGUMENT of a draft, without the labels that are the form itself.

    Was the WHY block's field values. WHY is gone, so this is now what carries the argument
    instead: the reason clause on each `ruled out` line, the one on `ordered by`, and
    `leaves to make`. Pointing these two checks at DECIDING rather than deleting them matters --
    with WHY removed and this left reading `by.get("why")`, both would have gone quietly to
    returning nothing on every draft, and `prose_repetition` is the gate that catches a draft
    recycling its own earlier sentences.

    The slot NAMES are excluded on the same grounds the field labels were: they repeat by
    design, and scoring them would make every turn a repeat of the last.
    """
    parts: list[str] = []
    for b in parse_document(text).get("blocks") or []:
        if b["name"] != "deciding":
            continue
        # Every clause of the block that is an ARGUMENT: what a route diverges from and why
        # it is kept. `ruled out` and `leaves to make` used to be collected here and are gone
        # from the form; `diverging from` took their place and is most of the block's prose,
        # so leaving it out would have let the repetition and restatement checks miss the part
        # that repeats.
        for rt in parse_routes(b["body"])[0]:
            d = rt.get("diverging") or {}
            parts += [x for x in (d.get("why"), d.get("on"), d.get("instead")) if x]
            if rt.get("ordered"):
                parts.append(rt["ordered"].get("why", ""))
    return " ".join(x for x in parts if x)


def _c_prose_repetition(text: str, prev_thoughts) -> list[dict]:
    """The argument block, against the argument blocks of earlier turns."""
    if not prev_thoughts:
        return []

    def bag(x):
        return collections.Counter(w for w in re.findall(r"[a-z]{4,}", x.lower()))
    mine = _why_of(text)
    b = bag(mine)
    if sum(b.values()) < _REP_MIN_WORDS:
        return []
    tot = sum(b.values())
    for t in prev_thoughts:
        ob = bag(_why_of(t))
        if not ob:
            continue
        if sum((b & ob).values()) / tot >= 0.75:
            return [V("prose_repetition", "DECIDING",
                      "says what an earlier turn's argument already said",
                      "This turn is a different decision: say what changed.", fatal=True)]
    return []


def _c_prose_restates(text: str) -> list[dict]:
    """`<why>` paraphrasing the ledger it sits under.

    The failure the two-part form invites: the fields say which bond each cut makes and which
    axis ordered them, and then the prose says it again in sentences. That doubles the tokens to
    carry the same claims, and it is what turns a form into a template -- the argument slot
    filled with a restatement because a restatement is always available and a real argument is
    not. Scored as the fraction of the argument's content words that already appear in the
    ledger; a cross-turn judgement ("the same dash I ignored on the aniline") shares almost
    nothing with it, and a paraphrase shares nearly all.
    """
    doc = parse_document(text)
    why = _why_of(text)
    if not why:
        return []
    ledger = " ".join(b["body"] for b in (doc.get("blocks") or [])
                      if b["name"] in ("molecule", "disconnections"))
    lb = set(re.findall(r"[a-z]{4,}", ledger.lower()))
    wb = [w for w in re.findall(r"[a-z]{4,}", why.lower())]
    if len(wb) < _REP_MIN_WORDS or not lb:
        return []
    shared = sum(1 for w in wb if w in lb) / len(wb)
    if shared >= 0.8:
        return [V("prose_restates", "DECIDING",
                  f"restates the fields above it ({shared:.0%} of its words are already in the "
                  f"ledger)",
                  "Spend it on what the fields cannot hold: the link to an earlier turn, the "
                  "risk being taken, what a dead branch taught.", fatal=True)]
    return []


# ------------------------------------------------------------------ the axes, as an order
def _axis_key(c: dict, axis: str) -> tuple | None:
    """A single candidate's comparable key on `axis`, lower is better, or None if it has none.

    One direction for every axis: q and p are better high, rt is better low with a dash worst of
    all, and ln$ is better when fewer fragments still have to be made -- a cut whose every piece
    is purchasable closes the molecule outright, which is a price argument the raw number does
    not carry. Split out from `_axis_values` so a candidate the call did NOT rank -- a `ruled
    out` line's own subject -- can be scored the same way a ranked one is, without pretending it
    belongs to an `order` list it is not in.
    """
    sig = c.get("signals") or {}
    if axis == "ln$":
        # THE COST THE BOARD PRINTS, AND NOTHING ELSE IN FRONT OF IT. This key used to lead
        # with `unmade` -- how many pieces still have to be made -- so a cut at $.100 with one
        # piece left to make lost to a cut at $.900 with none, while the board printed
        # `$.100` beside `$.900`. That is closure smuggled into price, and the developer
        # message forbids exactly it: "That a cut's pieces are all purchasable is a DIFFERENT
        # argument -- it closes the branch". `decided_by` now names that argument `closure`,
        # so price can be price.
        #
        # WHY IT MATTERS. The printed cost is comparable across the ranked field on many
        # rankings, while `_complete_cost`, the definition the `price_axis` gate used, is
        # finite for two or more candidates far more rarely -- so the screen hands over an
        # ordering the pipeline would otherwise refuse to let be named.
        lp = c.get("ln_price") or []
        if not lp or any(x is None for x in lp):
            return None
        import math as _m
        total = _m.log(sum(_m.exp(x) for x in lp))
        return (round(min(1.0, max(0.0, (total - R.COST_LO) / R.COST_SPAN)), 3),)
    if axis == "rt":
        v = sig.get("rt")
        return (1, 0.0) if v is None else (0, float(v))
    v = sig.get(axis)
    return None if v is None else (0, -float(v))


def field_of(ev: dict, mid: str, order: list[int]) -> list[int]:
    """The candidates a draft may argue about: the ranked ones plus the ones it is described.

    One definition, used by BOTH the `ordered by` hint in `rank_support` and the `decides_flat`
    check, because if they disagree the hint recommends an axis computed over this set and the
    check refuses it for being level over `order` alone -- a false `decides_flat`.
    """
    facts, menu = facts_of(ev, mid), menu_of(ev, mid)
    return list(dict.fromkeys(list(order) + [ci for ci in sorted(facts) if ci in menu]))


def _class_agrees(said: str, names: list[str]) -> bool:
    """Does the text name one of the classes the board matched this candidate to?

    Loose on purpose. The board carries `Ullmann-Goldberg Substitution aryl alcohol` and prose
    says "the Ullmann coupling"; requiring the string back is requiring a quotation, and the
    defect being caught is naming a class the candidate does NOT have. One content word in
    common is the bar. Stop words that appear in half the class names carry no evidence, so
    they do not count as the match.
    """
    stop = {"substitution", "synthesis", "reaction", "coupling", "aryl", "alcohol", "amine",
            "acid", "general", "with", "from", "type"}
    words = {w for w in re.findall(r"[a-z]{4,}", (said or "").lower())} - stop
    for n in names:
        for w in re.findall(r"[a-z]{4,}", n.lower()):
            if w not in stop and w in words:
                return True
    return False


def _bond_sig(cf: dict) -> tuple:
    """(formed, broken) atom pairs -- the identity of the expansion when nothing names it."""
    def one(kk):
        return tuple(sorted(str((b or {}).get("atoms") or "").upper().replace("-", "")
                            for b in (cf.get(kk) or []) if (b or {}).get("atoms")))
    return one("formed"), one("broken")


def _bond_words(said: str, cf: dict) -> bool:
    """For a candidate that earns no class name: does the text name a bond it measures?"""
    formed, broken = _bond_sig(cf)
    bonds = set(formed) | set(broken)
    if not bonds:
        return bool(re.search(r"[a-z]{4,}", said or ""))   # nothing to check it against
    said_n = (said or "").upper().replace("-", "").replace("\u2013", "").replace("\u2014", "")
    return any(b in said_n for b in bonds)


def _axis_values(menu: dict, order: list[int], axis: str) -> dict[int, tuple]:
    """{candidate: its `_axis_key`} over the RANKED candidates. See `_axis_key` for the scale."""
    out: dict[int, tuple] = {}
    for ci in order:
        c = menu.get(ci)
        if c is None:
            continue
        key = _axis_key(c, axis)
        if key is not None:
            out[ci] = key
    return out


def _c_deciding(doc: dict, ev: dict, turn: dict, ep: dict = None,
                k: int = None) -> list[dict]:
    """`diverging from` and `ordered by` against the board. Slots, so nothing is inferred.

    Every claim here is now a lookup: which candidates a line names, which measure it names, and
    what that measure reads on them. The earlier version had to find the measure in English and
    attribute a clause's candidates to it, and it was wrong far more often than the drafts were.
    """
    orders = ranked_orders(turn)
    out: list[dict] = []
    for b in (doc.get("by_name") or {}).get("deciding", []):
        mid = b["mid"]
        where = f"DECIDING {mid}"
        order, menu = orders.get(mid) or [], menu_of(ev, mid)
        if not order or not menu:
            continue
        facts = facts_of(ev, mid)
        routes, _, _ = parse_routes(b["body"])

        # ---- the route loop itself: one group per entry of `order`, in order.
        # `order` IS the multi-route statement -- the developer block tells the model that the
        # first is applied now and the rest stay on the board as the other routes through this
        # molecule -- so the labels are not a stylistic choice and are checked against it.
        want = [(i + 1, ci) for i, ci in enumerate(order)]
        got = [(rt["label"], rt["cand"]) for rt in routes]
        if got != want:
            out.append(V("route_labels_wrong", f"{where}",
                         f"declares routes {got or 'none'}; the call ranks {want}",
                         "One `route i -- c<n>` group per ranked candidate, in the order the "
                         "call ranks them.", fatal=True))
        for rt in routes:
            # Slot order inside a group, ADVISORY. Reading order is decision order -- what this
            # route is not repeating, then the measure that put it here -- and the spec shows it
            # that way. A group whose two lines arrive swapped still says two true things, so
            # this earns a repair round (`v["ok"]` goes false) and never drops a turn on its
            # own. Dropping a turn over line order would pay an episode-wide cost for a
            # cosmetic defect.
            seq = [ln.split(":", 1)[0].strip().lower() for ln in (rt.get("raw_lines") or [])]
            rank = {"diverging from": 0, "ordered by": 1}
            got_seq = [rank[x] for x in seq if x in rank]
            if got_seq != sorted(got_seq):
                out.append(V("deciding_slot_order", f"{where} route {rt['label']}",
                             "puts `ordered by` above `diverging from`",
                             "diverging from, then ordered by.", fatal=False))

        # ---- `diverging from`, judged against the board's REACTION LEDGER
        # `ruled_out_twice` stood here and is gone with the slot it policed.
        hist = history_reactions(ep, k) if (ep is not None and k is not None) else []
        by_r = {int(h["rid"][1:]): h for h in hist}
        by_m = {(h["mid"], h["c"]): h for h in hist}
        for rt in routes:
            d = rt["diverging"]
            if d is None:
                out.append(V("diverging_missing", f"{where} route {rt['label']}",
                             "has no `diverging from` line", fatal=True))
                continue
            # Required exactly when there IS earlier context to attend to, refused when there
            # is not. A slot that can be filled with narration on a board where nothing has
            # been expanded gets filled with narration, as `carried over` and `risk` were.
            if d.get("none") and hist and len(order) > 1:
                out.append(V("diverging_unearned", f"{where} route {rt['label']}",
                             f"says there is nothing to diverge from; the board has already "
                             f"expanded {', '.join(h['rid'] for h in hist[:4])}"
                             + (" and more" if len(hist) > 4 else ""),
                             "Name the route whose chemistry this one is not repeating.",
                             fatal=True))
                continue
            # Only a LEDGER citation is unearned on an empty ledger. The candidate level
            # needs no history at all -- it compares the cuts this call ranks against each
            # other -- and this branch used to refuse any non-`none` line, which killed every
            # honest c*-level sentence on a first rank turn. `lvl` is read before the level
            # gate below because that gate is about which level is OWED, and this is about
            # which level is POSSIBLE.
            # `.get`, not `[]`: the `none` shape carries no `refs` at all, and reading it
            # unconditionally raised a KeyError that `factcheck` caught and turned into a
            # silently EMPTY `own` -- every schema check on that draft disappeared, not just
            # this one.
            lvl0 = {r[0] for r in (d.get("refs") or [])}
            if (not d.get("none")) and not hist and lvl0 != {"c"}:
                out.append(V("diverging_unearned", f"{where} route {rt['label']}",
                             "cites the reaction ledger; nothing has been expanded on this "
                             "board yet",
                             "Compare the cuts this call ranks against each other, or "
                             "`diverging from: none | <why>` when it ranks only one.",
                             fatal=True))
                continue
            if d.get("none"):
                continue
            mine = family_key(facts.get(rt["cand"]) or {})

            # ---- WHICH LEVEL this route is allowed to cite, decided by the board.
            # Two things can be said and only one of them is the finding on any given turn:
            #   r*  a step already on the reaction ledger is the SAME KIND as this cut. Then
            #       "something like this is already expanded" outranks anything about the
            #       siblings, and it must be said.
            #   c*  nothing on the ledger relates. Then what is worth saying is what makes the
            #       ranked cuts DIFFERENT BETS -- and they genuinely are: a multi-candidate
            #       group typically ranks more than one reaction family.
            # Left free, the slot would take whichever is easier to write, and on the turns
            # where the ledger relates to nothing the r* form degrades into "these two
            # reactions differ", which is almost always true and therefore says nothing. The
            # gate is the whole reason this slot can be factchecked at all.
            overlap = [h for h in hist
                       if h["family"] == mine and mine[0] != "unmeasured"]
            lvl = {r[0] for r in d["refs"]}
            if overlap and lvl == {"c"}:
                out.append(V("diverging_wrong_level", f"{where} route {rt['label']}",
                             f"argues against the other candidates; {overlap[0]['rid']} on the "
                             f"ledger is already the same kind of step as c{rt['cand']} "
                             f"([{family_label(mine)}]), which is the fact that outranks it",
                             f"`diverging from: {overlap[0]['rid']} | same step, <the kind> | "
                             f"<why this candidate anyway>`.", fatal=True))
                continue
            if not overlap and lvl and lvl != {"c"} and len(order) > 1:
                out.append(V("diverging_wrong_level", f"{where} route {rt['label']}",
                             "cites the reaction ledger; nothing already expanded is the same "
                             "kind as this cut, so there is no repetition to report",
                             "Say what makes the cuts this call ranks different bets from each "
                             "other -- `diverging from: c<n>, c<m> | <what they are> | <what "
                             "this one does instead>`.", fatal=True))
                continue

            # ---- the CANDIDATE level
            if lvl == {"c"}:
                for _lvl, ci in d["refs"]:
                    if ci == rt["cand"]:
                        out.append(V("diverging_self", f"{where} route {rt['label']}",
                                     f"diverges c{ci} from itself", fatal=True))
                        continue
                    if ci not in menu:
                        out.append(V("diverging_not_claimed", f"{where} route {rt['label']}",
                                     f"diverges from c{ci}; it is not on this molecule's menu",
                                     fatal=True))
                        continue
                    theirs = family_key(facts.get(ci) or {})
                    if theirs[0] == "unmeasured" or mine[0] == "unmeasured":
                        continue
                    if theirs == mine and not d.get("same"):
                        out.append(V("diverging_same_class", f"{where} route {rt['label']}",
                                     f"calls c{ci} a different bet; the support block groups it "
                                     f"and c{rt['cand']} under [{family_label(mine)}]",
                                     "Say `same step, <the kind>` and then what differs -- "
                                     "which fragment, or where on the skeleton.", fatal=True))
                    elif theirs != mine and d.get("same"):
                        out.append(V("diverging_same_claimed", f"{where} route {rt['label']}",
                                     f"calls c{ci} the same step; the board groups them under "
                                     f"[{family_label(theirs)}] and [{family_label(mine)}]",
                                     fatal=True))
                    elif theirs != mine and not _class_agrees(d.get("on") or "",
                                                              list(theirs[1])
                                                              if theirs[0] == "named"
                                                              else [family_label(theirs)]) \
                            and not _bond_words(d.get("on") or "", facts.get(ci) or {}):
                        out.append(V("diverging_class_wrong", f"{where} route {rt['label']}",
                                     f"says c{ci} is {(d.get('on') or '')[:40]!r}; the board "
                                     f"maps it to {family_label(theirs)!r}", fatal=True))
                continue

            # ---- the ROUTE level
            for ref in d["refs"]:
                h = by_r.get(ref[1]) if ref[0] == "r" else by_m.get((ref[1], ref[2]))
                if h is None:
                    said = f"r{ref[1]}" if ref[0] == "r" else f"{ref[1]}\u00b7c{ref[2]}"
                    out.append(V("diverging_not_claimed", f"{where} route {rt['label']}",
                                 f"diverges from {said}; the board's reaction ledger has no "
                                 f"such step", fatal=True))
                    continue
                # What that route TOOK has to be named as the board maps it -- one definition,
                # `family_key`, shared with the support block that prints the ledger and with
                # the same-kind comparison below.
                key = h["family"]
                if key[0] == "named":
                    if not _class_agrees(d.get("on") or "", list(key[1])):
                        out.append(V("diverging_class_wrong", f"{where} route {rt['label']}",
                                     f"says {h['rid']} took {(d.get('on') or '')[:40]!r}; the "
                                     f"board maps it to {family_label(key)!r}", fatal=True))
                elif key[0] == "bond":
                    if not _bond_words(d.get("on") or "", {"formed": [], "broken": []}) \
                            and not _class_agrees(d.get("on") or "", [family_label(key)]):
                        out.append(V("diverging_ungrounded", f"{where} route {rt['label']}",
                                     f"says {h['rid']} took {(d.get('on') or '')[:40]!r}; it "
                                     f"earns no reaction name, so name what it does -- "
                                     f"{family_label(key)}", fatal=True))
                if mine[0] == "unmeasured" or key[0] == "unmeasured":
                    continue
                if key == mine and not d.get("same"):
                    out.append(V("diverging_same_class", f"{where} route {rt['label']}",
                                 f"claims a different expansion from {h['rid']}; the board "
                                 f"groups that step and this cut under [{family_label(mine)}]",
                                 "That IS the finding -- say `same step, <the kind>` and then "
                                 "why this candidate is worth taking anyway, or take a cut "
                                 "from another group.", fatal=True))
                elif key != mine and d.get("same"):
                    out.append(V("diverging_same_claimed", f"{where} route {rt['label']}",
                                 f"calls this the same step as {h['rid']}; the board groups "
                                 f"them under [{family_label(key)}] and [{family_label(mine)}]",
                                 fatal=True))

        # `ordered by` is per ROUTE now, so this is judged once per route group. Route i
        # chooses among `order[i:]` -- its own cut plus the ones still unclaimed after it --
        # because the earlier routes in this same block have already taken theirs. Judging
        # route 2's measure against the whole of `order` would ask it to put a candidate first
        # that route 1 already took.
        seen_obj: dict[tuple, int] = {}
        for ri, rt in enumerate(routes):
            ordered = rt["ordered"]
            scope = order[ri:] if ri < len(order) else []
            if not scope:
                continue
            if ordered is None:
                # Required on EVERY route, the last one included. `ordered by` was written as "the
                # measure that puts this cut first among what is left", which made the final group
                # -- a field of one -- exempt, and made the middle groups repeat the head's axis:
                # `route 1 ... plausibility | value: 1.000` and `route 2 ... plausibility | value:
                # 1.000` is the same argument twice, saying nothing about why route 2 is worth
                # keeping. It is the OBJECTIVE this route wins on, and a route with no objective of
                # its own is not a route worth declaring. Once episodes expand several branches per
                # molecule, a non-head route can beat the head on no number AND have no grounded
                # `structure`. For those there is no objective to name. What must NOT be written
                # there is "the search found a route through this cut": that is an appeal to the
                # oracle, the same defect `order_appeal` and `leak_posthoc` refuse everywhere else,
                # and it is not chemistry. A second route earns its place by being a chemically
                # different way in -- another bond, another class, another division of the skeleton
                # -- and all three of those are grounded and checkable. Where the board measured
                # none of them there is nothing to say and the line is left off -- silence, not a
                # justification. With the evidence caches complete this branch is a guard rather
                # than a live path: it exists so a future change to the caches cannot silently
                # create a turn nobody can write.
                mine_ = menu.get(rt["cand"]) or {}
                head_ = menu.get(order[0]) or {}
                beats_ = any(_axis_key(mine_, a) is not None
                             and _axis_key(head_, a) is not None
                             and _axis_key(mine_, a) < _axis_key(head_, a) for a in AXES)
                cf_ = facts.get(rt["cand"]) or {}
                struct_ok = (cf_.get("named_tier") in ("applies", "applies+makes")
                             or cf_.get("bond_measured"))
                if rt["label"] == 1 or beats_ or struct_ok:
                    out.append(V("ordered_missing", f"{where} route {rt['label']}",
                                 "has no `ordered by` line; every route that HAS an objective "
                                 "names the measure it is worth keeping for", fatal=True))
                continue
            dax = MEASURE_TOKENS.get(ordered["token"])
            if dax is None:
                continue
            # The reason slot must be a reason, not the score column copied back. Counted over the
            # tokens that are neither a candidate label nor a number: an argument has words in it.
            toks = [t for t in re.findall(r"[A-Za-z$][A-Za-z0-9$.\-]*|\d[\d.]*", ordered["why"])
                    if not re.fullmatch(r"c?\d[\d.]*|the|a|an|of|to|is|are|it|and|or|at|vs|s",
                                        t, re.I)]
            # No digits in `why`. The slot beside it holds the number, so a figure here is a
            # second, unpositioned copy -- and an unpositioned number is one whose axis and
            # candidate have to be guessed, which is exactly what made the old free-text form
            # uncheckable. `c9`-style labels are not digits in this sense and are allowed.
            stray = re.sub(r"\bc\d+\b|\bln\$|\brt\d\b", "", ordered["why"])
            if re.search(r"\d", stray):
                out.append(V("deciding_number_in_why", f"{where} ordered by",
                             f"puts a figure in the argument: {ordered['why'][:70]!r}",
                             "Numbers go in `value:`, one per line, where they can be checked. "
                             "The argument says what the measure MEANS.", fatal=True))
            # ...and the number that IS positioned has to be this route's cut's own reading.
            vals = ordered.get("values") or []
            stated = vals[0] if vals else None
            head0 = rt["cand"]
            # An EMPTY `value:` used to skip every check below it, because all of them are guarded
            # on `stated is not None`. So a draft could name a numeric axis and write `value: none`
            # while the board carries the number, and the argument beside it then carries the
            # comparison with nothing positioned to check it against. `structure` is the one axis
            # with no number by definition and is exempt; the other four are printed on the
            # candidate line or on its fragments.
            if stated is None and dax != "struct" and head0 in menu:
                cd = menu[head0] or {}
                if dax == "ln$":
                    priced = [x for x in (cd.get("ln_price") or []) if x is not None]
                    if priced:
                        out.append(V("ordered_value_missing", f"{where} ordered by",
                                     f"orders by price and leaves `value:` empty; c{head0}'s "
                                     f"fragments carry "
                                     f"{', '.join(R.dollars(x) for x in priced)}",
                                     "Put the reading in `value:`; it is what the argument is "
                                     "about.", fatal=True))
                    # A cut with no purchasable fragment has no price, and `value: none` is
                    # then the honest reading. Naming price as the objective there is arguably the
                    # wrong axis, but `_route_axis` picks it the same way when it builds a
                    # clean fixture, so refusing it here would refuse the module's own notion
                    # of a valid draft. Left alone deliberately.
                elif (cd.get("signals") or {}).get(dax) is not None:
                    out.append(V("ordered_value_missing", f"{where} ordered by",
                                 f"orders by {AXIS_SAY[dax]} and leaves `value:` empty; the "
                                 f"board prints "
                                 f"{(cd.get('signals') or {}).get(dax)} for c{head0}",
                                 "Put the reading in `value:`; it is what the argument is "
                                 "about.", fatal=True))
            if stated is not None and dax in ("p", "q") and head0 in menu:
                real = (menu[head0].get("signals") or {}).get(dax)
                if real is not None and abs(stated - real) > 0.0015:
                    out.append(V("ordered_value_wrong", f"{where} ordered by",
                                 f"gives {stated:g} for c{head0}'s {AXIS_SAY[dax]}; the board "
                                 f"has {real:.3f}", fatal=True))
            elif stated is not None and dax == "rt" and head0 in menu:
                real = (menu[head0].get("signals") or {}).get("rt")
                if real is None:
                    out.append(V("ordered_value_wrong", f"{where} ordered by",
                                 f"gives a round-trip rank for c{head0}; the forward model "
                                 f"does not recover it", fatal=True))
                elif int(stated) != int(real):
                    out.append(V("ordered_value_wrong", f"{where} ordered by",
                                 f"gives round-trip {int(stated)} for c{head0}; the board has "
                                 f"{int(real)}", fatal=True))
            if len(toks) <= 2:
                out.append(V("deciding_recites", f"{where} ordered by",
                             f"gives the score column back instead of what it means: "
                             f"{ordered['why'][:70]!r}",
                             "Say what that measure tells you about these cuts that the number "
                             "alone does not.", fatal=True))
            if dax == "struct":
                # No number to check an order against -- ground it in the taken candidate's own
                # bond or named-reaction fact instead, so "structure" cannot become a way to skip
                # grounding a claim in anything at all.
                head = rt["cand"]
                hf = facts.get(head) or {}
                # `shape` is not accepted as a third grounding. A non-head route with no named
                # match and no mapped bond is a CACHE MISS in `bond_changes.json`, not a fact
                # about the chemistry, and with the mapping and template caches filled for the
                # menu reactions nearly every one has something to point at. Accepting `shape`
                # would loosen this check for a case the complete caches do not produce.
                if not (hf.get("named_tier") in ("applies", "applies+makes")
                        or hf.get("bond_measured")):
                    out.append(V("structure_ungrounded", f"{where} ordered by",
                                 f"orders c{head} on structure; it has no named-reaction match and "
                                 f"no measured bond to point to",
                                 "Name plausibility, round-trip, confidence or price instead, or "
                                 "point to a fact this candidate actually carries.", fatal=True))
                continue
            # ---- the OBJECTIVE this route is kept for.
            # Route 1 leads outright. A later route exists because it wins somewhere the head
            # does not -- that is what makes the set multi-objective rather than one ranking
            # written three times. So a non-head route naming a NUMBER has to name one it
            # actually beats the head on, and one that beats the head on no number has to say
            # `structure` and point at what its cut DOES.
            if ri > 0 and dax != "struct":
                mine_k = _axis_key(menu.get(rt["cand"]) or {}, dax)
                head_k = _axis_key(menu.get(order[0]) or {}, dax)
                if mine_k is not None and head_k is not None and not (mine_k < head_k):
                    beats = [ax for ax in AXES
                             if (_axis_key(menu.get(rt["cand"]) or {}, ax) is not None
                                 and _axis_key(menu.get(order[0]) or {}, ax) is not None
                                 and _axis_key(menu.get(rt["cand"]) or {}, ax)
                                 < _axis_key(menu.get(order[0]) or {}, ax))]
                    out.append(V("ordered_not_differentiating",
                                 f"{where} route {rt['label']}",
                                 f"is kept for {AXIS_SAY[dax]}; c{rt['cand']} does not beat "
                                 f"c{order[0]} on it, so that is the head's reason, not this "
                                 f"route's",
                                 ("Name " + ", ".join(AXIS_SAY[x] for x in beats)
                                  + " -- it wins there.") if beats else
                                 "It wins on no number, so say `structure` and what its cut "
                                 "does that the head's does not.", fatal=True))
            key_obj = (dax, ordered.get("value_raw", "").strip().lower())
            if key_obj in seen_obj and dax != "struct":
                out.append(V("ordered_same_objective", f"{where} route {rt['label']}",
                             f"is kept for {AXIS_SAY[dax]} at the same reading as route "
                             f"{seen_obj[key_obj]}; two routes with one objective are one "
                             f"route written twice", fatal=True))
            seen_obj.setdefault(key_obj, rt["label"])

            vals = _axis_values(menu, scope, dax)
            if not vals:
                out.append(V("decides_absent", f"{where} ordered by",
                             f"orders on a measure no ranked candidate of {mid} carries",
                             fatal=True))
                continue
            # Flat against the FIELD the hint was computed over, not against `order` alone. The
            # support block tells the draft which measures put the head first, and it computes that
            # head-against-the-described-field (see `field` in rank_support) -- so a measure can
            # separate the head from the alternatives and still read level across the two or three
            # candidates the call happens to rank. Checking only `order` would refuse the axis the
            # hint had just recommended as valid. Recommending X and refusing X is not a gate, it
            # is a contradiction, and the draft has no way to satisfy both.
            field_vals = _axis_values(menu, field_of(ev, mid, order), dax)
            if len(set(vals.values())) == 1 and len(scope) > 1 \
                    and len(set(field_vals.values())) <= 1:
                out.append(V("decides_flat", f"{where} ordered by",
                             "orders on a measure that reads the same on every candidate described, "
                             "so it separated nothing",
                             "Name the measure that splits them -- you are told which ones do.",
                             fatal=True))
                continue
            # Route 1 only: it is the cut being APPLIED, so the measure it names has to put
            # it first. A later route is expected to beat the head on its own axis -- checking
            # it the same way would refuse exactly the statement the slot is for.
            head = rt["cand"]
            if ri == 0 and head in vals and vals[head] > min(vals.values()):
                winner = sorted(c for c, v in vals.items() if v == min(vals.values()))[0]
                out.append(V("decides_order", f"{where} ordered by",
                             f"orders on a measure that puts c{winner} ahead of the c{head} the "
                             f"call takes",
                             "You are told which measures put the head first; name one of them.",
                             fatal=True))
    return out


# ------------------------------------------------------------------ the delta, both ways
def _norm_target(s: str) -> str:
    """Loose enough that formatting noise doesn't fail a copy that IS faithful."""
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(". ")).lower()


def _c_molecule_target(doc: dict, ep: dict) -> list[dict]:
    """`target:` checked two ways -- against the vocabulary the target actually has, and, once
    one turn's phrasing has been accepted, against ITSELF.

    The spec calls `target:` "the same phrase every turn... the fixed point everything else is
    measured against," and nothing enforced that. The field is free English re-derived from the
    SMILES on every turn, and a teacher re-deriving the same name on every turn of a long episode
    does not produce identical names -- sooner or later it invents a ring system or substituent
    (a saturated chain called "a stilbene", a piperazine on a molecule that has none), and every
    turn after it inherits the wrong fixed point, unless something compares a turn's target line
    to another turn's, or to the molecule.

    Two defects, not one check, because they need different evidence:
      * IDENTITY -- a ring or group named that the target does not contain at all. Decidable
        from the target's own SMILES via the same vocabulary `not in the target` is checked
        against, so it fires from turn one, before anything has been established to copy.
      * DRIFT -- once a turn's phrase has been accepted, `ep["_target_canon"]` holds it and every
        later turn is handed it back to COPY (see `rank_support`). A turn that does not
        reproduce it is not making a fresh naming error, it is failing to copy text it was
        handed -- which is the more useful fact to gate on, because copying is a task a teacher
        is good at even when re-deriving the same name from a SMILES string is not.
    """
    tgt = ep.get("target") or ""
    canon = ep.get("_target_canon")
    vocab = (set(FC.RING_SIG) | set(FC.FUSED_SIG) | set(FC.FG_SMARTS)) \
        if (FC is not None and Chem is not None) else set()
    in_target = group_words(tgt) if (vocab and tgt) else set()
    out = []
    for b in (doc.get("by_name") or {}).get("molecule", []):
        kv, _ = parse_keyed(b["body"], MOL_KEYS)
        val = kv.get("target", "")
        if not val:
            continue
        if canon is not None:
            if _norm_target(val) != _norm_target(canon):
                out.append(V("target_drift", "MOLECULE target",
                             "does not match the phrase this episode already established for "
                             "the target", f"Copy it exactly: {canon!r}", fatal=True))
            continue                      # matches (or was just flagged); nothing more to check
        if not vocab:
            continue
        low = val.lower()
        named = {w for w in vocab if re.search(rf"\b{re.escape(w)}s?\b", low)}
        for w in sorted(named - in_target):
            out.append(V("target_identity", "MOLECULE target",
                         f"names a {w} the target does not contain", fatal=True))
    return out


def _c_delta(doc: dict, ep: dict, turn: dict) -> list[dict]:
    """`not in the target` is a set difference, and it is wrong in two ways -- neither of them
    "mentions a word the target also has".

    The first version flagged every vocabulary word in the line that the target shares, and most
    of what it caught was POSITION: "a bromine on the phenyl, installed as the coupling handle"
    names the bromine as the difference and the phenyl as where it sits, and both molecules have
    a phenyl, so the check refuted a line that was right. It also read only the FIRST ranked
    molecule, while a turn ranking two writes one line covering both.

    So what is checked now is the CLAIM, not the vocabulary:
      * a group named there that neither the piece nor the target contains is invented;
      * a line that names groups and not ONE of them is a genuine difference has not answered
        the question -- that is the field sliding back into "what the piece is".
    A line naming a true difference plus the ring it sits on is correct, and is left alone.
    """
    if FC is None or Chem is None:
        return []
    ev = turn.get("evidence") or {}
    tgt = ep.get("target") or ""
    mids = list(ranked_orders(turn))
    if not mids or not tgt:
        return []
    have = set()
    for mid in mids:
        smi = ((ev.get("mols") or {}).get(mid) or {}).get("smiles")
        if smi:
            have |= group_words(smi)
    if not have:
        return []
    in_target = group_words(tgt)
    vocab = set(FC.RING_SIG) | set(FC.FUSED_SIG) | set(FC.FG_SMARTS)
    out = []
    for b in (doc.get("by_name") or {}).get("molecule", []):
        kv, _ = parse_keyed(b["body"], MOL_KEYS)
        val = kv.get("not in the target", "")
        if not val or _NO_VALUE.match(val):
            continue
        low = val.lower()
        named = {w for w in vocab if re.search(rf"\b{re.escape(w)}s?\b", low)}
        if not named:
            continue                    # named nothing this vocabulary knows: nothing to decide
        for w in sorted(named - have - in_target):
            out.append(V("delta_absent", "MOLECULE not-in-the-target",
                         f"names a {w} that neither this piece nor the target contains",
                         fatal=True))
        # There WAS a check here for a line that names only groups the target also has -- "you
        # named no difference". It is gone, because the vocabulary cannot support the negative.
        # `group_words` knows ring classes and a fixed list of functional groups; it does not know
        # halogens, protecting groups or plain substituents, and those are most of what a route
        # actually installs. "a bromine on the phenyl, installed as the coupling handle" names
        # the true difference and one word of position, and the vocabulary sees only the
        # position -- so the check concluded the line named nothing and refused a correct
        # answer. What survives is the claim that can be decided: a group named there that
        # NEITHER molecule contains is invented, whatever else the line says.
    return out


# ------------------------------------------------------------------ the bond, element pairs
def _c_bond_pairs(doc: dict, ev: dict, turn: dict) -> list[dict]:
    """The element pairs a `bond:` field declares, against the ones the mapping measured.

    The field is half free text and half fact, and this is the fact half. `C-Br formed` where the
    mapping says `C:1-C:2` is refutable without reading a word of the phrase around it -- and it
    is exactly the class of error freeform prose makes and cannot gate: "a C-Br bond
    formation that severs the indole" on a step that forms a C-C, "c9 forms the C-Br" on a
    benzylic C-C. Where the bond SITS stays free, because no checker can decide it; which atoms
    it joins does not.
    """
    orders = ranked_orders(turn)
    out: list[dict] = []
    for b in (doc.get("by_name") or {}).get("disconnections", []):
        mid = b["mid"]
        facts = facts_of(ev, mid)
        for r in parse_cuts(b["body"])[0]:
            f = facts.get(r["c"])
            if f is None or not f.get("bond_measured"):
                continue                              # nothing measured: nothing to contradict
            said, real = parse_bond_pairs(r["bond"]), bond_pairs_of(f)
            rw = f"<cuts {mid}> row c{r['c']}"
            for side in BOND_SIDES:
                # Sets, not multisets. A step that breaks two C-O bonds is one claim when the
                # row says "C-O broken", and counting the second as an omission would ask the
                # writer to enumerate identical pairs.
                sd, rl = set(said[side]), set(real[side])
                for p in sorted(sd - rl):
                    other = next((o for o in BOND_SIDES
                                  if o != side and p in set(real[o])), None)
                    hint = (f" -- it is {p[0]}-{p[1]} {other} here" if other else "")
                    out.append(V("bond_pair_wrong", rw,
                                 f"says {p[0]}-{p[1]} is {side} at c{r['c']}; the mapping has "
                                 f"{side} " + _pair_str(sorted(rl)) + hint, fatal=True))
                missing = sorted(rl - sd)
                if missing and sd:
                    out.append(V("bond_pair_missing", rw,
                                 f"leaves out {_pair_str(missing)} {side} at c{r['c']}"))
    return out


def _c_cycle(doc: dict, ep: dict, k: int) -> list[dict]:
    """`regenerates an ancestor` against the branch the board actually built.

    Two directions, and both matter. A turn that does NOT prune a trap teaches the student to
    read past the one thing that kills its episode -- that is the defect this check exists for.
    A turn that pins the label on a candidate which regenerates nothing teaches the check as a
    verbal tic, which is worse than silence, so the claim is verified as well as required.

    Only fires where a trap exists, so most turns are untouched by it.
    """
    turn = ep["turns"][k]
    orders = ranked_orders(turn)
    out: list[dict] = []
    for b in (doc.get("by_name") or {}).get("deciding", []):
        mid = b["mid"]
        if mid not in orders:
            continue
        # This used to require every cycle trap to be named on a `ruled out` line. That slot is
        # gone, and with it the only place the form could say "the board would refuse this one"
        # -- so `cycle_unpruned` and `cycle_false` are unsatisfiable claims now and are not
        # asked for. What survives is expressible and still matters: a trap must not be
        # DESCRIBED as though it were an available alternative. The call cannot rank one (the
        # board refuses it outright, so no episode contains that), but a draft can hand one a
        # DISCONNECTIONS row beside the real cuts, and a reader learns from that row that a
        # refused step was on the menu.
        traps = cycle_traps(ep, k, mid)
        if not traps:
            continue
        rowed = {r["c"] for db in (doc.get("by_name") or {}).get("disconnections", [])
                 if db["mid"] == mid for r in parse_cuts(db["body"])[0]}
        for ci in sorted(set(traps) & rowed):
            out.append(V("cycle_described", f"DISCONNECTIONS {mid} row c{ci}",
                         f"gives c{ci} a row beside the real cuts; it puts "
                         f"{', '.join(traps[ci])} back, which is already above {mid}, so the "
                         f"board refuses that call outright",
                         "Describe a candidate that could actually be taken.", fatal=True))
    return out


def schema_facts(doc: dict, ep: dict, turn: dict, k: int = None) -> list[dict]:
    """Field against fact. Every one of these was a whole-paragraph inference before the form.

    All fatal except the ceilings and `class_dropped`: each is a false claim about the chemistry
    or a contradiction between the reasoning and the call, and neither is repaired by rewriting a
    noun.
    """
    ev = turn.get("evidence") or {}
    by = doc.get("by_name") or {}
    out: list[dict] = []

    for b in by.get("disconnections", []):
        mid = b["mid"]
        where = f"DISCONNECTIONS {mid}"
        facts, menu = facts_of(ev, mid), menu_of(ev, mid)
        for r in parse_cuts(b["body"])[0]:
            rw = f"{where} row c{r['c']}"
            f = facts.get(r["c"])
            if f is None:
                continue                                   # no fact to contradict: fail open
            # bond ------------------------------------------------------------------
            if not f.get("bond_measured") and _asserts_bond(r["bond"]):
                out.append(V("bond_unmeasured", rw,
                             f"states a bond change for c{r['c']}, whose mapping failed",
                             "That row's bond field reads: not measured.", fatal=True))
            elif (f.get("bond_measured") and not _asserts_bond(r["bond"])
                  and (f.get("formed") or f.get("broken") or f.get("changed"))):
                # Only when the mapping actually recorded a CHANGE. A cache entry can be valid
                # with no bonds in it -- the mapper ran and found nothing moved -- and
                # for those `not measured` is the honest field and there is nothing to withhold.
                out.append(V("bond_withheld", rw,
                             f"says the bond for c{r['c']} was not measured; it was: "
                             + _bond_phrase(f).replace("\n", " "),
                             "Name the element pair and say where it sits.", fatal=True))
            # class -----------------------------------------------------------------
            # Only `applies+makes` earns a name. `applies` means the groups fit some named
            # reaction and the outcome is NOT what that reaction gives, which is a different
            # claim and not a name -- so it takes `none` like the rest.
            tier, names = f.get("named_tier"), f.get("named") or []
            if _asserts_class(r["cls"]):
                if tier != "applies+makes" or not names:
                    out.append(V("reaction_unearned", rw,
                                 f"names a reaction for c{r['c']} that no template reproduces"
                                 + (f" (the groups fit {names[0]}, which is not the same claim)"
                                    if names else ""),
                                 "That row's class field reads: none.", fatal=True))
                elif not _names_match(r["cls"], names):
                    out.append(V("reaction_wrong", rw,
                                 f"calls c{r['c']} a {r['cls'].strip()!r}; the template that "
                                 f"reproduces it is {', '.join(names)}", fatal=True))
            elif tier == "applies+makes" and names:
                out.append(V("reaction_dropped", rw,
                             f"leaves c{r['c']} unnamed although {names[0]} reproduces it",
                             "Name it -- an earned name is what transfers to the next "
                             "molecule."))
            # split -----------------------------------------------------------------
            sh = f.get("shape") or {}
            want = split_verdict(sh) if sh else None
            if want:
                said = {k for k, pat in SPLIT_PATTERNS.items()
                        if re.search(pat, r["split"], re.I)}
                wrong = said - {want}
                if wrong:
                    out.append(V("split_verdict", rw,
                                 f"describes c{r['c']} as {sorted(wrong)[0]}; the fragments make "
                                 f"it {want}"
                                 + (f" ({' and '.join(str(x) for x in sh['fragments'])} heavy "
                                    f"atoms)" if sh.get("fragments") else ""),
                                 fatal=True))
            frags = sh.get("fragments") or []
            if len(frags) >= 2:
                m = re.search(r"\b(\d+)\s*(?:\+|and)\s*(\d+)\b", r["split"])
                if m and sorted(int(x) for x in m.groups()) != sorted(frags[:2]):
                    out.append(V("split_sizes", rw,
                                 f"gives c{r['c']} fragments of {m.group(1)} and "
                                 f"{m.group(2)} heavy atoms; they are "
                                 + " and ".join(str(x) for x in frags), fatal=True))
            kept = sh.get("scaffold_kept")
            if kept is not None:
                said_kept = re.search(r"\b(skeleton|scaffold|framework|core)\b[^.|]{0,40}"
                                      r"\b(kept|survives|intact|unchanged|preserved|stays)\b",
                                      r["split"], re.I)
                said_broken = re.search(r"\b(skeleton|scaffold|framework|core)\b[^.|]{0,40}"
                                        r"\b(changes|broken|breaks|lost|rebuilt|built)\b",
                                        r["split"], re.I)
                if said_kept and not kept:
                    out.append(V("split_shape", rw,
                                 f"says c{r['c']} keeps the skeleton; the precursors do not "
                                 f"carry the product's ring-and-linker topology", fatal=True))
                if said_broken and kept:
                    out.append(V("split_shape", rw,
                                 f"says c{r['c']} changes the skeleton; a precursor carries the "
                                 f"product's topology unchanged", fatal=True))
            # buyable ---------------------------------------------------------------
            c = menu.get(r["c"]) or {}
            if re.search(r"\b(purchasable|buyable|in stock|off the shelf|commercial)\b",
                         r["split"] + " " + r["bond"], re.I) and c:
                if not any(c.get("buyable") or []):
                    out.append(V("purchasable", rw,
                                 f"calls a fragment of c{r['c']} purchasable; none of its "
                                 f"precursors is marked buyable", fatal=True))

    # ---- described candidates: real ones, and a real comparison ---------------------------
    # `weighed_ranked` and `weighed_unpruned` lived here and are gone with the role word and the
    # `ruled out` slot: there is no longer a claim that a row is an ALTERNATIVE, so there is
    # nothing to contradict, and nothing to name it on. What survives is the pair of facts that
    # were never about the role -- a described candidate has to be one the board has facts for,
    # and the block has to describe something the call passed over, or the comparison is
    # one-sided. The cycle-prune exemption goes with `ruled out` too; a trap is now argued in
    # `diverging from` if it is argued at all.
    orders_ = ranked_orders(turn)
    for b_ in by.get("disconnections", []):
        mid = b_["mid"]
        where = f"DISCONNECTIONS {mid}"
        rows, _ = parse_cuts(b_["body"])
        facts, menu = facts_of(ev, mid), menu_of(ev, mid)
        order = orders_.get(mid) or []
        for r in rows:
            if r["c"] not in facts:
                out.append(V("row_no_facts", f"{where} row c{r['c']}",
                             f"describes c{r['c']}, which carries no bond or reaction facts on "
                             f"this turn -- there is nothing to describe it FROM", fatal=True))
        spare = [ci for ci in facts if ci not in order and ci in menu]
        described_spare = [r for r in rows if r["c"] not in order]
        if order and spare and not described_spare:
            out.append(V("no_alternative_described", where,
                         f"describes only what the call ranks; c{sorted(spare)[0]} is described "
                         f"for you and is not ranked",
                         "Give at least one candidate the call passes over its own row.",
                         fatal=True))

    out += _c_molecule_target(doc, ep)
    out += _c_bond_pairs(doc, ev, turn)
    out += _c_deciding(doc, ev, turn, ep, k)
    out += _c_delta(doc, ep, turn)

    # ---- grouped claims, in any block's prose ----------------------------------------------
    ctx = {"kind": "rank", "actions": turn.get("actions"), "seen_smiles": ()}
    for b in doc.get("blocks") or []:
        where = f"<{b['name']}{(' ' + b['mid']) if b['mid'] else ''}>"
        out += _c_group_claims(b["body"], ev, ctx, where)
    return out



def _struct_exempt(doc: dict, ev: dict, turn: dict) -> set[tuple[str, int]]:
    """(mid, c) pairs `_c_deciding` already accepted as grounded on `structure`.

    `applied_subcutoff` (board/factcheck.py) fires on the structural fact alone -- the call
    took a candidate below the plausibility cut-off -- with no way to see that this turn's
    DECIDING block named a reason for it that has nothing to do with p. Re-deriving that same
    grounding here (rather than trusting the draft's own claim) means a candidate only escapes
    the cut-off veto when it genuinely has a named-reaction match or a measured bond to point
    to, the same bar `_c_deciding` already holds it to.
    """
    orders = ranked_orders(turn)
    out: set[tuple[str, int]] = set()
    for b in (doc.get("by_name") or {}).get("deciding", []):
        mid = b["mid"]
        order = orders.get(mid) or []
        if not order:
            continue
        rts = parse_routes(b["body"])[0]
        ordered = rts[0]["ordered"] if rts else None
        if ordered is None or MEASURE_TOKENS.get(ordered["token"]) != "struct":
            continue
        head = order[0]
        hf = facts_of(ev, mid).get(head) or {}
        if hf.get("named_tier") in ("applies", "applies+makes") or hf.get("bond_measured"):
            out.add((mid, head))
    return out


def factcheck(text: str, ep: dict, k: int, doc: dict, a) -> dict:
    """The shared checks, addressed, plus the checks the form makes exact.

    Returns a dict shaped like `board.factcheck.check` with two additions: `where` on every
    violation, and `schema` violations alongside the shared ones. Fails open in both halves --
    no `board.factcheck` and no RDKit means fewer checks, never a rejected draft.
    """
    turn = ep["turns"][k]
    ev = turn.get("evidence") or {}
    kind = LEGACY.turn_kind(turn)
    prev = [ep["turns"][i].get("thought") or "" for i in range(k)]
    ctx = {"prev_thoughts": [p for p in prev if p], "actions": turn.get("actions"),
           # The rendered board for THIS turn. `factcheck` builds its set of quotable prices
           # from it rather than re-deriving them, so a figure the model was shown can never
           # be refuted as invented -- see `_c_numbers`.
           "observation": turn.get("env") or "",
           "seen_smiles": (FC.smiles_seen(ep["turns"][i].get("evidence") or {}
                                          for i in range(k)) if FC is not None else ())}
    shared = {"ok": True, "violations": [], "codes": [], "fatal": [], "failed_checks": []}
    if FC is not None and getattr(a, "factcheck", True):
        shared = FC.check(text, ev, kind, prev_thoughts=ctx["prev_thoughts"],
                          actions=ctx["actions"], seen_smiles=ctx["seen_smiles"],
                          observation=ctx.get("observation"))
        _drop_named_field_identity(shared, text, doc, ev, kind, ctx)
        _drop_negated_purchasable(shared, text)
        # The shared skeleton check attributes NOTHING: it searches the whole draft for a
        # "keeps" phrase and a "breaks" phrase and blames the head candidate for whichever it
        # finds. On a form with one row per candidate that is unusable -- any draft mixing a
        # kept row with a changed row fires it -- and its break pattern misses the word order
        # this prompt teaches ("the ring skeleton changes" against `changes? the skeleton`), so
        # the mix reads to it as keep-only, refusing drafts that got no skeleton wrong.
        # `split_shape` above does the same comparison per row, against
        # that row's own measured verdict, which is what the form makes possible.
        shared["violations"] = [v for v in (shared.get("violations") or [])
                                if v["code"] != "shape"]
        shared["codes"] = sorted({v["code"] for v in shared["violations"]})
        shared["fatal"] = sorted(c for c in shared["codes"] if c in FC.FATAL)
        shared["ok"] = not shared["violations"]
        exempt = _struct_exempt(doc, ev, turn)
        if exempt:
            shared["violations"] = [
                v for v in (shared.get("violations") or [])
                if not (v["code"] == "applied_subcutoff"
                       and any(v["detail"].startswith(f"takes {mid}·c{c} ")
                               for mid, c in exempt))]
            shared["codes"] = sorted({v["code"] for v in shared["violations"]})
            shared["fatal"] = sorted(c for c in shared["codes"] if c in FC.FATAL)
            shared["ok"] = not shared["violations"]
        # The shared repetition check cannot see a form. `_c_prose_repetition` replaces
        # it, on the one block where a repeat means what the check thinks it means.
        shared["violations"] = [v for v in (shared.get("violations") or [])
                                if v["code"] != "repetition"]
        shared["codes"] = sorted({v["code"] for v in shared["violations"]})
        shared["fatal"] = sorted(c for c in shared["codes"] if c in FC.FATAL)
        shared["ok"] = not shared["violations"]
        addr = _attribute(doc, ev, kind, ctx, set(shared.get("codes") or []))
        for v in shared.get("violations") or []:
            v["where"] = addr.get(v["code"], "document")
            v["fatal"] = v["code"] in FC.FATAL
    own = []
    if getattr(a, "factcheck", True):
        try:
            own = schema_facts(doc, ep, turn, k)
            if not ep.get("_no_ancestors"):
                own += _c_cycle(doc, ep, k)
            own += _c_prose_repetition(text, ctx["prev_thoughts"])
            own += _c_prose_restates(text)
        except Exception as e:                                         # noqa: BLE001
            shared.setdefault("failed_checks", []).append(f"schema_facts: {type(e).__name__}: {e}")
    viol = list(shared.get("violations") or []) + own
    codes = sorted({v["code"] for v in viol})
    return {"ok": not viol, "violations": viol, "codes": codes,
            "fatal": sorted({v["code"] for v in viol if v.get("fatal")}),
            "repeat_ratio": shared.get("repeat_ratio"),
            "failed_checks": shared.get("failed_checks") or []}


# ==================================================================== repair and salvage
def describe_repair(v: dict, fv: dict, limit: int = 8) -> str:
    """The violations as the correction to hand back, addressed by block.

    Written as instructions about the form and the chemistry, never as a report about a checker:
    a message that mentions a gate invites a draft that narrates having been corrected. The
    address is the whole gain over the freeform module's version -- "your <cuts ja> row for c3"
    is a line the teacher can rewrite without touching anything else, and rewriting one line is
    what keeps the rest of the draft (and the episode's voice) stable.
    """
    items = [x for x in (v.get("violations") or []) + (fv.get("violations") or [])]
    if not items:
        return ""
    lines = []
    for x in items[:limit]:
        where = x.get("where") or "document"
        detail = x.get("detail") or x.get("code")
        fix = (x.get("fix") or "").rstrip(".")
        lines.append(f"- {where}: {detail}." + (f" {fix}." if fix else ""))
    return ("Your draft breaks the form or states something this turn's facts contradict. "
            "Rewrite it so every point below is corrected, changing nothing else -- same "
            "decision, same voice, same blocks:\n" + "\n".join(lines))


def draft_rank(v: dict, fv: dict) -> int:
    """How bad a rejected draft is, lowest kept. Four tiers, and the order is the point.

    A LEAK is worst: the oracle answer, a post-hoc number, an atom-map index is a claim the
    student must never learn to make and no rewrite repairs it. A false CLAIM about the
    chemistry is next -- also unsalvageable, but a re-prompt naming the field that refutes it
    usually fixes it, which blind resampling does not. A broken FORM is third: nothing is wrong
    with the chemistry, the shape is wrong, and that is the most repairable defect there is.
    Register is last, because `salvage` can rewrite it.
    """
    leaks = {"leak_phrase", "leak_posthoc", "depth_ceiling", "atom_index", "label_quote"}
    if set(v.get("fatal") or []) & leaks:
        return 3
    if fv.get("fatal"):
        return 2
    if set(v.get("fatal") or []) - {"jargon"}:
        return 1
    return 0


def salvage(text: str, v: dict) -> tuple[str, list] | tuple[None, None]:
    """Rewrite the interface words out of a draft, or refuse.

    Same substitutions as the freeform module and the same refusal rule: a leak, a false claim
    or a broken form is never rewritten, only a register defect where the sentence says the
    right thing about the wrong subject. Applied to block BODIES only -- a substitution that
    touched a tag would turn a well-formed draft into an unparseable one, which is the failure
    mode a text-level rewrite has and a block-level one does not.
    """
    if not v.get("jargon_only"):
        return None, None

    def _keep_case(m, word):
        return word[0].upper() + word[1:] if m.group(0)[0].isupper() else word

    # Rewrite BODIES in place and leave the header lines untouched. Splicing on `body_start`
    # rather than rebuilding the document is what keeps this safe: a rebuild has to re-emit
    # every header, and a substitution that touched one would turn a well-formed draft into an
    # unparseable one -- the failure a text-level rewrite has and a block-level one does not.
    hits, edits = [], []
    for b in (v.get("doc") or {}).get("blocks") or []:
        body = text[b["body_start"]:b["end"]]
        for pat, wf in LEGACY._SUB:
            new = re.sub(pat, lambda m, w=wf: _keep_case(m, w), body, flags=re.I)
            if new != body:
                hits.append(f"{b['name']}:{pat}")
            body = new
        edits.append((b["body_start"], b["end"], body))
    if not hits:
        return None, None
    buf, prev = [], 0
    for start, end, body in edits:
        buf.append(text[prev:start])
        buf.append(body)
        prev = end
    buf.append(text[prev:])
    return "".join(buf), hits


# ==================================================================== the draw loop
BRIEF_KINDS = ("open", "done")
BRIEF_SENTENCES = 2


def _axis_say(menu: dict, ci: int, ax: str | None) -> str:
    """The candidate's own reading on `ax`, spelled as `value:` wants it.

    `structure` has no number and takes `none`; a round-trip is an integer rank; the rest are
    three decimals. The fixture prints what the check reads, so a baseline cannot fail its own
    numeric gate.
    """
    if not ax or ax == "struct":
        return "none"
    v = ((menu.get(ci) or {}).get("signals") or {}).get(ax)
    if v is None:
        return "none"
    return str(int(v)) if ax == "rt" else f"{v:.3f}"


def brief_cap(a, kind: str) -> int:
    """The word ceiling for THIS kind of turn.

    `--max-words` is sized for a rank turn's three blocks and is far too loose for an open turn.
    The brief asks for "two sentences", and without a matching acceptance window the model
    writes well past it. An instruction nothing enforces is a suggestion.
    """
    if kind in BRIEF_KINDS:
        return min(getattr(a, "max_words_brief", 45), a.max_words)
    return a.max_words


def rank_cap(a, ep: dict, k: int) -> int:
    """The word ceiling for a RANK turn, scaled by how many molecules it ranks.

    `--max-words` is one number and a rank turn is not one size. The form is MOLECULE once
    plus DISCONNECTIONS and DECIDING per molecule, so a turn that ranks three needs roughly
    three times the words of one that ranks one -- and a flat cap applied to all of them makes
    the gate, not the chemistry, decide whether a multi-molecule turn fills: a complete
    multi-molecule thought runs over it, and one that fits has had to compress two or three
    full accounts into one molecule's budget.

    So the budget is ONE MOLECULE'S WORTH PER MOLECULE, which is what the form asks for: a
    turn ranking three writes three DISCONNECTIONS blocks and three DECIDING blocks, and there
    is no sense in which that is the same document as a one-molecule turn's.

    A SINGLE-MOLECULE TURN KEEPS THE PLAIN CEILING. A one-molecule rank thought fits within
    `--max-words`, so that discipline is real and nothing is gained by loosening it. What this
    changes is the turns where the cap was measuring the wrong thing: multi-molecule thoughts,
    where a flat cap refuses the shape the corpus is made of.

    LENGTH IS ONE CAUSE, NOT THE ONLY ONE. `dropped_length` is what this removes;
    `no_alternative_described`, `cuts_rank_order`, `diverging_wrong_level` and
    `ordered_not_differentiating` are untouched by it. Raising this ceiling is necessary for a
    high rank fill, not sufficient.
    """
    n = len(ranked_orders(ep["turns"][k]) or {}) or 1
    return a.max_words * n


def brief_room(a, kind: str, ep: dict, k: int) -> tuple[int, int]:
    """(word ceiling, sentence ceiling) for THIS open/done turn.

    The two-sentence bound is what keeps open turns short, and it stays the default. But an open
    turn that opens several pieces owes an `opening together` line and one that passes over a piece
    owes a `leaving open` line, and the tight bound would refuse the very thing the form asks for
    -- so each clause the call EARNS buys one sentence and twenty words, and nothing else does. A
    turn with no decision to report keeps 2 and 45, because there is nothing for a third sentence
    to carry.
    """
    cap = brief_cap(a, kind)
    sents = BRIEF_SENTENCES
    if kind == "open" and FC is not None:
        try:
            turn = ep["turns"][k]
            ev = turn.get("evidence") or {}
            openable, opened = FC.open_choice(ev, turn.get("actions"))
            earned = (len(opened) >= 2) + bool([m for m in openable if m not in opened])
            if earned:
                cap = min(cap + 20 * earned, a.max_words)
                sents = BRIEF_SENTENCES + earned
        except Exception:                                              # noqa: BLE001
            pass
    return cap, sents


def sentence_count(text: str) -> int:
    return len([x for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x.strip()])


def _legacy_turn(ep: dict, k: int, a, stats) -> None:
    """open / done / hand-over, on the freeform briefs, until their own schemas are designed.

    Kept in the same pass rather than in a second run so one command produces a complete
    episode: a rank turn is written with the open turn before it in context, and an open turn
    with no thought is a hole every later turn inherits.
    """
    msgs = LEGACY.turn_prompt(ep, k, a.full_boards)
    # The pieces already above each molecule this turn opens. BRIEF_OPEN asks for them by id,
    # and they cannot come from `support_block`: `ancestors_of` lives in this module and
    # traj_route_reasoning_board is imported BY it, so computing them there would be a cycle.
    # This is where the cycle discipline is planted -- at the turn before any candidate exists,
    # so the rank turn that follows already has the set in context rather than meeting it for
    # the first time next to a menu.
    if LEGACY.turn_kind(ep["turns"][k]) == "open":
        lines = []
        for act in (ep["turns"][k].get("actions") or []):
            if not isinstance(act, dict) or (act.get("type") or "").lower() != "open":
                continue
            mid = act.get("mid")
            anc = ancestors_of(ep, k, mid) if mid else {}
            if anc:
                lines.append(f"  above {mid}, on its own branch: "
                             + ", ".join(f"{x} ({s})" for x, s in sorted(anc.items())))
            elif mid:
                lines.append(f"  above {mid}: nothing -- it is the target itself")
        if lines:
            msgs = msgs[:-1] + [{
                "role": "user",
                "content": msgs[-1]["content"]
                + "\n\nAlready above the piece(s) this turn opens. A disconnection that hands "
                  "one of these back would make it its own precursor and the board would refuse "
                  "the call:\n" + "\n".join(lines)}]
    base = msgs
    best, best_v, best_f, best_rank = None, None, {}, 9
    draws = fails = repairs = 0
    while draws < max(a.max_draws, a.samples, 1):
        bump = 0.0 if draws <= a.samples else min(0.5, 0.12 * (draws - a.samples))
        try:
            text, _ = LEGACY.call_teacher(msgs, a, temperature=a.temperature + bump)
        except Exception as e:                                         # noqa: BLE001
            stats["request_failed"] += 1
            fails += 1
            if fails > max(a.max_draws, a.samples, 1) * 3:
                break
            time.sleep(min(8.0, 0.5 * fails))
            continue
        draws += 1
        v = LEGACY.verify(text, ep, k)
        fv = LEGACY.factcheck(text, ep, k, a)
        stats["drawn"] += 1
        fact_ok = fv["ok"] if a.fact_strict else not fv.get("fatal")
        # The freeform gate is all-or-nothing and stays that way: its violations carry no
        # advisory/blocking split, so there is nothing here to be lenient about.
        cap, sent_cap = brief_room(a, LEGACY.turn_kind(ep["turns"][k]), ep, k)
        n_sent = sentence_count(text)
        brief_ok = n_sent <= sent_cap or cap >= a.max_words
        if NAIVE["on"]:
            # What this arm removes is the schema and the verification. verify/factcheck
            # are still computed to fill the fields the code below uses; their verdict is ignored.
            best, best_v, best_f = text, v, fv
            break
        if v["ok"] and fact_ok and brief_ok and a.min_words <= v["words"] <= cap:
            best, best_v, best_f = text, v, fv
            break
        if v["ok"] and fact_ok and not (brief_ok and v["words"] <= cap) \
                and repairs < a.repair:
            # A turn that is too long has an ADDRESS -- the number of words and sentences it
            # actually wrote -- so it earns a repair round rather than a fresh draw.
            repairs += 1
            stats["repair_length"] += 1
            msgs = base + [{"role": "assistant", "content": text},
                           {"role": "user",
                            # NOT "say it in two sentences": on an open turn that owes an
                            # `opening together` or `leaving open` clause the ceiling is three
                            # or four, and telling it to cut to two asks it to drop the clause
                            # -- which the open gates then refuse. The ceiling is computed for
                            # this turn; quote that, and name what may not be cut.
                            "content": f"That is {v['words']} words over {n_sent} sentences. "
                                       f"The ceiling for this turn is {sent_cap} sentences and "
                                       f"{cap} words. Cut the restatement of the board, not "
                                       f"the facts and not any clause the call earns; keep "
                                       f"every line the form requires and stop there."}]
            draws -= 1
            continue
        if (not fact_ok) and v["ok"] and repairs < a.repair and FC is not None:
            repairs += 1
            msgs = base + [{"role": "assistant", "content": text},
                           {"role": "user", "content": FC.describe(fv)}]
            draws -= 1
            continue
        msgs = base
        r = LEGACY.draft_rank(v, fv)
        if r < best_rank:
            best, best_v, best_f, best_rank = text, v, fv, r
        stats["rejected"] += 1
    if best is None:
        return
    acts = LEGACY.turn_action_targets(ep["turns"][k])
    fact_ok = (best_f.get("ok", True) if a.fact_strict else not (best_f or {}).get("fatal"))
    if not (best_v or {}).get("ok") and fact_ok and a.salvage:
        fixed, hits = LEGACY.salvage(best, best_v or {})
        if fixed and a.min_words <= len(fixed.split()) <= brief_room(
                a, LEGACY.turn_kind(ep["turns"][k]), ep, k)[0]:
            fixed, marked = LEGACY.mark_actions(fixed, acts)
            ep["turns"][k].update({"thought": fixed,
                                   "thought_salvaged": list(hits or []) + list(marked or []),
                                   "thought_draft": best, "thought_check": best_v})
            stats["salvaged"] += 1
            return
    # Same leak as the rank path: `best` is the least-bad draft and arrives unchecked against
    # the ceiling `brief_room` computed for this turn. An open turn written at rank length is
    # the specific thing brief_cap exists to stop, so re-apply the turn's own ceiling here.
    b_cap = brief_room(a, LEGACY.turn_kind(ep["turns"][k]), ep, k)[0]
    over_len = not (a.min_words <= _words(best) <= b_cap)
    if NAIVE["on"] or ((best_v or {}).get("ok") and fact_ok and not over_len):
        text, marked = LEGACY.mark_actions(best, acts)
        ep["turns"][k]["thought"] = text
        if marked:
            ep["turns"][k]["thought_marked"] = marked
        _lock_target(ep, text)
    else:
        ep["turns"][k]["thought"] = ""
        if over_len:
            stats["dropped_length"] += 1
        elif not fact_ok:
            stats["dropped_facts"] += 1
    ep["turns"][k]["thought_draft"] = best
    ep["turns"][k]["thought_check"] = best_v
    # The FACTUAL verdict too, not only in `_schema_turn`: otherwise an open or done turn refused
    # for a false claim leaves `thought_check` clean and `thought_facts` empty, a dropped turn
    # with no recorded cause. A gate whose firing is not written down is a gate nobody can
    # audit.
    if best_f:
        ep["turns"][k]["thought_facts"] = {kk: best_f[kk] for kk in
                                          ("codes", "fatal", "repeat_ratio", "failed_checks")
                                          if best_f.get(kk)}


def _reject_draft(ep: dict, k: int, a, stats) -> None:
    """Reasoning for the call the board REFUSED, written without the ancestry it missed.

    Only for a refused RANK on a molecule that has a menu -- a cycle trap. The point is what
    makes that trap teachable: the refused candidate's defect is an OMISSION, not a false claim.
    For a typical trap, "it clears the filter and splits the molecule in two" is entirely TRUE;
    what is missing is that one of its precursors is the target. So the
    draft here is generated under the ordinary gates -- every number checked, every ring name
    checked -- with exactly two things withheld: the ancestor listing (`_no_ancestors` suppresses
    it in `rank_support`) and the cycle check itself. What comes back is a faithful reproduction
    of the trap: a correct argument that never ran the one test that mattered.

    That is why this is generated rather than fabricated. A hand-written wrong draft would fail
    every check we have; this one passes them all and is still the wrong move, which is the
    lesson. The turn AFTER it -- written with the refusal in context and the ancestors shown --
    is the contrast.

    `rank_unopened` gets nothing: its refused call names candidate numbers for a piece with no
    menu, so there is no fact to ground an argument in and any text would be invention.
    """
    t = ep["turns"][k]
    rej = t.get("reject") or {}
    acts = [x for x in (rej.get("actions") or []) if isinstance(x, dict)]
    ranks = [x for x in acts if (x.get("type") or "").lower() == "rank" and x.get("order")]
    if not ranks:
        return
    ev = t.get("evidence") or {}
    if not all(menu_of(ev, x.get("mid")) for x in ranks):
        return                        # no menu behind it: nothing to reason from
    shadow_t = dict(t)
    shadow_t["actions"] = acts
    shadow_t.pop("reject", None)
    shadow = dict(ep)
    shadow["turns"] = list(ep["turns"])
    shadow["turns"][k] = shadow_t
    shadow["_no_ancestors"] = True
    msgs = turn_prompt(shadow, k, a)
    best = None
    for i in range(max(a.samples, 1)):
        try:
            text, _ = LEGACY.call_teacher(msgs, a, temperature=a.temperature + 0.1 * i)
        except Exception:                                              # noqa: BLE001
            stats["reject_request_failed"] += 1
            continue
        text = text.strip()
        v = verify(text, shadow, k, a)
        fv = factcheck(text, shadow, k, v.get("doc") or parse_document(text), a)
        stats["reject_drawn"] += 1
        if not v["blocking"] and not fv.get("fatal") \
                and a.min_words <= v["words"] <= rank_cap(a, shadow, k):
            best = text
            break
    if best is None:
        stats["reject_dropped"] += 1
        return
    rej["thought"] = best
    stats["reject_written"] += 1
    # Into the wire form, immediately before the call it argues for. Matched on the call's own
    # content so the position cannot drift: whatever has to recognise or strip this injected
    # pair keys on `supervised is False`.
    want = json.dumps({"actions": acts}, ensure_ascii=False)
    for j, m in enumerate((ep.get("row") or {}).get("harmony_messages") or []):
        if (m.get("role") == "assistant" and m.get("supervised") is False
                and m.get("content") == want):
            ep["row"]["harmony_messages"].insert(
                j, {"role": "assistant", "channel": "analysis",
                    "content": best, "supervised": False})
            break


def _lock_target(ep: dict, text: str) -> None:
    """Fix this episode's `target:` phrase to the first accepted one, and hand it back after.

    `ep["_target_canon"]` is READ in two places -- `rank_support`, to print the phrase back with
    "copy this exactly", and `_c_molecule_target`, to refuse a turn that does not reproduce it --
    and this is where a real run WRITES it. Without it `canon` stays None, the support block
    never shows the phrase, and the drift branch of the check never runs.

    The defect it stops: two turns of one episode naming the same target differently (the
    ethyl on a different position, a different saturated ring). Where the scaffold really
    contains both rings, each phrase names something the molecule has and each turn passes its
    own check against the molecule. Neither is checked against the OTHER, and that is the whole
    reason for a canon -- one episode teaching two names for one target is exactly the
    inconsistency a student learns to reproduce.
    """
    if ep.get("_target_canon"):
        return
    for b in (parse_document(text).get("by_name") or {}).get("molecule", []):
        kv, _ = parse_keyed(b["body"], MOL_KEYS)
        phrase = (kv.get("target") or "").strip()
        if phrase:
            ep["_target_canon"] = phrase
            return


_ADDR_MID = re.compile(r"^(?:DISCONNECTIONS|DECIDING)\s+(\S+)")


def mid_of(v: dict) -> str | None:
    """Which molecule a violation is ABOUT, or None when it is about the turn.

    Every block-level check addresses itself as `DISCONNECTIONS <mid> ...` or
    `DECIDING <mid> ...`, so the address already carries this. MOLECULE, the register checks and
    the document-level ones (`schema_*`, `address_*`, the leaks) are turn-level and gate the
    whole draft.
    """
    m = _ADDR_MID.match((v.get("where") or "").strip())
    return m.group(1) if m else None


def blocks_by_mid(text: str, doc: dict = None) -> dict[str, str]:
    """{mid: its DISCONNECTIONS block + its DECIDING block, verbatim}.

    The unit a draft is scored in. A rank turn can cover several molecules, and as one
    all-or-nothing draft one wrong slot on one molecule throws away every honest block beside
    it: the more candidates a draft has to get right, the lower its pass rate, even at an
    unchanged per-candidate error.
    """
    doc = doc or parse_document(text)
    out: dict[str, list[str]] = {}
    for b in (doc.get("blocks") or []):
        if b.get("name") not in ("disconnections", "deciding") or not b.get("mid"):
            continue
        # A parsed block carries offsets, not its own text: slice the source so the header
        # comes with it and the block can be pasted into an assembled draft verbatim.
        out.setdefault(b["mid"], []).append(text[b["start"]:b["end"]].rstrip("\n"))
    return {k: "\n".join(x for x in v if x) for k, v in out.items()}


def _schema_turn(ep: dict, k: int, a, stats) -> None:
    """One RANK turn, written to the form.

    The budget is escalating rather than fixed, exactly as in the freeform module and for the
    same reason: a turn left empty is not a local loss, because every turn after it is written
    with the turns before it in context, so a hole propagates through the rest of the episode.
    What is different is WHICH failures get a repair round. A broken form and a false claim are
    both answers to a specific correction -- hand the draft back with the line that is wrong and
    it comes back fixed -- while a leak or a register habit is a sampling problem that a higher
    temperature and a fresh draw address and a correction does not.
    """
    msgs = turn_prompt(ep, k, a)
    base = msgs
    best, best_v, best_f, best_rank = None, None, {}, 9
    draws = fails = repairs = 0
    alts: list[str] = []
    # Per-MOLECULE salvage. A rank turn is one draft that can cover several molecules, and a
    # single wrong slot on one of them would throw away every honest block beside it (see
    # `blocks_by_mid`). `kept` holds, per molecule, the first block
    # that came back with nothing addressed to it; `head` the MOLECULE block of a draw whose
    # turn-level checks were clean. When every ranked molecule has one they are assembled, and
    # the ASSEMBLY is re-verified whole: blocks from different draws have never been checked
    # against each other, so it clears the same gates as a single draw or it is thrown away.
    want_mids = list(ranked_orders(ep["turns"][k]) or {})
    kept: dict[str, str] = {}
    head: str | None = None
    while draws < max(a.max_draws, a.samples, 1):
        bump = 0.0 if draws <= a.samples else min(0.5, 0.12 * (draws - a.samples))
        try:
            text, _ = LEGACY.call_teacher(msgs, a, temperature=a.temperature + bump)
        except Exception as e:                                         # noqa: BLE001
            stats["request_failed"] += 1
            ep["turns"][k]["thought_error"] = str(e)[:200]
            fails += 1
            if fails > max(a.max_draws, a.samples, 1) * 3:
                break
            time.sleep(min(8.0, 0.5 * fails))
            continue
        draws += 1
        text = text.strip()
        v = verify(text, ep, k, a)
        fv = factcheck(text, ep, k, v.get("doc") or parse_document(text), a)
        stats["drawn"] += 1
        for c in v.get("codes") or []:
            stats["v_" + c] += 1
        for c in fv.get("codes") or []:
            stats["f_" + c] += 1
        fact_ok = fv["ok"] if a.fact_strict else not fv.get("fatal")
        if NAIVE["on"]:
            # What this arm removes is the schema and the verification. verify/factcheck
            # are still computed to fill the fields the code below uses; their verdict is ignored.
            best, best_v, best_f, best_rank = text, v, fv, 0
            break
        # Two bars, not one. `keep` is "this draft may be written to the corpus" -- nothing
        # BLOCKING is wrong with it -- and `v["ok"]` is "nothing at all is wrong with it". A
        # draft that clears the first and not the second is worth one repair round for the
        # nudge, and is kept if the round is not available: dropping a turn over a long row
        # costs the whole episode a link, since every later turn is written knowing this one.
        # ---- harvest whatever this draw got right, molecule by molecule
        if want_mids:
            bad_mid = collections.Counter()
            turn_level = 0
            for vv in list(v.get("violations") or []) + list(fv.get("violations") or []):
                m_ = mid_of(vv)
                if m_ is None:
                    turn_level += 1
                else:
                    bad_mid[m_] += 1
            if not turn_level and head is None:
                hb = [b for b in ((v.get("doc") or {}).get("blocks") or [])
                      if b.get("name") == "molecule"]
                if hb:
                    head = text[hb[0]["start"]:hb[0]["end"]].rstrip("\n")
            for m_, blk in blocks_by_mid(text, v.get("doc")).items():
                if m_ not in kept and blk and not bad_mid.get(m_):
                    kept[m_] = blk
                    stats["block_kept"] += 1
            if head is not None and all(m_ in kept for m_ in want_mids):
                asm = head + "\n" + "\n".join(kept[m_] for m_ in want_mids)
                av = verify(asm, ep, k, a)
                af = factcheck(asm, ep, k, av.get("doc") or parse_document(asm), a)
                a_ok = af["ok"] if a.fact_strict else not af.get("fatal")
                if ((not av["blocking"]) and a_ok and av["ok"]
                        and a.min_words <= av["words"] <= rank_cap(a, ep, k)):
                    stats["assembled"] += 1
                    if asm not in alts:
                        alts.append(asm)
                    best, best_v, best_f, best_rank = asm, av, af, 0
                    break
                # KEEP THE HARVEST. Clearing it would throw away every block that had come back
                # clean because the ASSEMBLY of them failed one gate, and the loop would discard
                # good blocks faster than it finds them. Only the blocks the refusal names are
                # dropped; the rest stand and the next draw fills the gaps.
                stats["assembly_refused"] += 1
                bad = {mid_of(vv) for vv in
                       (list(av.get("violations") or []) + list(af.get("violations") or []))}
                bad.discard(None)
                if bad:
                    for m_ in bad:
                        kept.pop(m_, None)
                else:
                    head = None

        keep = ((not v["blocking"]) and fact_ok
                and a.min_words <= v["words"] <= rank_cap(a, ep, k))
        if keep and v["ok"]:
            if best is None:
                # best_rank 0, not the 9 sentinel. Without it a fully clean winner is left at
                # rank 9 and the `if r < best_rank` fallback at the bottom -- which exists to
                # keep the least-bad draft for diagnostics when NOTHING passed -- overwrites it
                # with a FAILING draw on the next iteration. The loop only reaches that
                # iteration because --variants keeps drawing after the winner is found. The turn
                # would then be dropped on the failing draft's verdict while its clean winner
                # survives only in `thought_variants`.
                best, best_v, best_f, best_rank = text, v, fv, 0
            # EVERY fully clean draw is kept, not just the first. The loop already pays for up
            # to --max-draws of them, and keeping one would throw the rest away. A discarded draw
            # is verified against the same gates as the winner and explains the SAME action, so it
            # is another true rationale for one decision -- which is exactly what a corpus with
            # one rationale per instance cannot teach. `--variants 1` keeps one per turn, and
            # only the FIRST is written to `thought`; the rest ride along in
            # `thought_variants` for the wire builder to fan out into sibling rows.
            if text not in alts:
                alts.append(text)
            if len(alts) >= max(a.variants, 1):
                break
            continue
        if keep and repairs >= a.repair:
            if best is None:
                best, best_v, best_f = text, v, fv
            break
        # A draft refused ONLY for length has an address too -- its own word count -- and the
        # rank path had no way to say so: `repairable` below requires a schema or a fact
        # defect, so an otherwise-clean overlong draft would never be corrected, it would just
        # burn a draw and come back overlong, and then drop. Those are rank turns, so dropping
        # them starves the corpus of the very decision the model has to learn. Repairing keeps
        # the exemplar and cuts it to the ceiling instead.
        too_long = (not v["blocking"]) and fact_ok and v["words"] > a.max_words
        if too_long and repairs < a.repair:
            repairs += 1
            stats["repair_length"] += 1
            msgs = base + [{"role": "assistant", "content": text},
                           {"role": "user",
                            "content": f"That is {v['words']} words. The ceiling for this "
                                       f"turn is {a.max_words}. Cut the restatement of the "
                                       f"board and the argument you repeat across routes, "
                                       f"not the facts; keep every line the form requires -- "
                                       f"each `route i -- cN`, its `diverging from:` and the "
                                       f"`ordered by:` line all stay -- and stop there."}]
            draws -= 1                     # a correction is a different prompt, not a draw
            continue
        # A repairable rejection is one where every fatal defect has an ADDRESS: a field the
        # correction can name. Those come back fixed. A leak has no address worth naming --
        # telling a model not to quote the answer in the sentence it just quoted it in mostly
        # produces the same sentence with a synonym -- so it goes to a fresh draw instead.
        repairable = (draft_rank(v, fv) <= 2) and (not v["ok"] or not fact_ok)
        if repairable and repairs < a.repair:
            # an accepted-but-nudged draft is the fallback if the repair comes back worse
            if keep and best_rank > 0:
                best, best_v, best_f, best_rank = text, v, fv, 0
            repairs += 1
            stats["repair_attempted"] += 1
            msgs = base + [{"role": "assistant", "content": text},
                           {"role": "user", "content": describe_repair(v, fv)}]
            draws -= 1                     # a correction is a different prompt, not a draw
            continue
        msgs = base
        r = draft_rank(v, fv)
        if r < best_rank:
            best, best_v, best_f, best_rank = text, v, fv, r
        stats["rejected"] += 1
        if fv.get("fatal"):
            stats["rejected_facts"] += 1
    if best is None:
        return
    fact_ok = (best_f.get("ok", True) if a.fact_strict else not (best_f or {}).get("fatal"))
    if best_f:
        ep["turns"][k]["thought_facts"] = {kk: best_f[kk] for kk in
                                           ("codes", "fatal", "repeat_ratio", "failed_checks")
                                           if best_f.get(kk)}
    if (best_v or {}).get("blocking") and fact_ok and a.salvage:
        fixed, hits = salvage(best, best_v or {})
        if fixed:
            v2 = verify(fixed, ep, k, a)
            if not v2["blocking"] and a.min_words <= v2["words"] <= a.max_words:
                ep["turns"][k].update({"thought": fixed, "thought_salvaged": hits,
                                       "thought_draft": best, "thought_check": _slim(best_v)})
                stats["salvaged"] += 1
                stats["kept"] += 1
                return
    # `best` here is the least-bad draft the loop kept for diagnostics, and it reaches this
    # line WITHOUT having passed the length gate -- the `if r < best_rank` fallback stores a
    # draft whatever its word count. Written unconditionally, a turn whose every draw was refused
    # for running long would reach the corpus at full length; such turns are mostly `rank`, so
    # the leak would teach length and ranking together -- a model trained on them over-ranks,
    # ranking molecules it has never opened. An exemplar that failed a gate is worse than no
    # exemplar, so a failing draft is dropped.
    # LENGTH IS RECORDED, NOT ENFORCED, ABOVE THE CEILING. A draft that is too SHORT is not a
    # turn and still drops -- there is nothing to salvage from forty words that should have
    # been three blocks. A draft that is too LONG is a complete, checked account of the turn
    # that happens to be verbose, and dropping it throws away the only thing in this pipeline
    # that costs GPU. Trimming is a post-process: `thought_words` is on every turn, so `text`
    # and `wire` can drop or shorten by any rule chosen later, with the whole corpus in view
    # rather than one turn at a time. The risk is long thoughts reaching TRAINING unnoticed;
    # here they reach the corpus file labelled, and the label is what makes the decision
    # reversible.
    n_words = _words(best) if best else 0
    too_short = best is None or n_words < a.min_words
    over_len = (not too_short) and n_words > rank_cap(a, ep, k)
    if NAIVE["on"] or (not (best_v or {}).get("blocking") and fact_ok and not too_short):
        ep["turns"][k]["thought"] = best
        ep["turns"][k]["thought_words"] = n_words
        ep["turns"][k]["thought_over_len"] = bool(over_len)
        stats["kept"] += 1
        if over_len:
            stats["kept_over_length"] += 1
        for c in (best_v or {}).get("advisory") or []:
            stats["kept_with_" + c] += 1
    else:
        ep["turns"][k]["thought"] = ""
        if too_short:
            stats["dropped_short"] += 1
        elif not fact_ok:
            stats["dropped_facts"] += 1
        else:
            stats["dropped_schema"] += 1
    ep["turns"][k]["thought_draft"] = best
    ep["turns"][k]["thought_check"] = _slim(best_v)
    # Every OTHER clean draw for this turn: same board, same action, a different true argument
    # for it. Compared against `best`, NOT against the written `thought` -- the accept path runs
    # `mark_actions` over the winner, which backticks the action words, so the stored text is no
    # longer string-equal to the draft it came from and alts[0] would come back as a variant of
    # itself. Each variant gets the same marking when the wire builder fans it out.
    extra = [x for x in alts if x != best]
    if extra:
        ep["turns"][k]["thought_variants"] = [
            LEGACY.mark_actions(x, LEGACY.turn_action_targets(ep["turns"][k]))[0]
            for x in extra]
        stats["variants_kept"] += len(extra)


def _slim(v: dict | None) -> dict:
    """The check result as it is written to the row: no parsed document, it is re-derivable."""
    if not v:
        return {}
    return {kk: v[kk] for kk in ("ok", "codes", "fatal", "advisory", "blocking",
                                 "words", "violations") if kk in v}


def reason_episode(ep: dict, a) -> dict:
    """Fill every turn's `thought`, in order, each with the ones before it in context."""
    # Every request this episode makes goes to one replica. Turn k's prompt extends turn k-1's
    # and a turn's draws share theirs exactly, so pinning is what lets the prefix cache carry
    # them -- see `LEGACY._next_url`. It is set here because one episode is one worker thread.
    LEGACY.stick_to(ep.get("id"))
    stats = collections.Counter()
    fill = ep.get("_fill")
    # In a FILL pass the kept turns never reach the accept path, so nothing would establish the
    # episode's target phrase and the canon would be set by the first REDRAWN turn instead --
    # locking onto whichever phrasing that turn happens to use, which on a drift-repair pass is
    # precisely the phrasing being repaired. Seed it here from the earliest turn that already
    # has one, so the redraws are held to the phrase the episode opened with.
    if fill is not None:
        for t in ep["turns"]:
            if t.get("thought"):
                _lock_target(ep, t["thought"])
                if ep.get("_target_canon"):
                    break
    for k in range(len(ep["turns"])):
        kind = LEGACY.turn_kind(ep["turns"][k])
        if fill is not None and k not in fill:
            stats["kept"] += bool(ep["turns"][k].get("thought"))
            continue
        if kind not in a.kinds:
            stats["skipped_" + kind] += 1
            continue
        stats["turn_" + kind] += 1
        if kind == "rank":
            _schema_turn(ep, k, a, stats)
        else:
            _legacy_turn(ep, k, a, stats)
            stats["kept"] += bool(ep["turns"][k].get("thought"))
        # After the turn's own reasoning, not before: the refused draft borrows this episode's
        # established target phrase and must not be the draft that sets it.
        if ep["turns"][k].get("reject") and getattr(a, "reject_reasoning", True):
            _reject_draft(ep, k, a, stats)
    ep["reasoning_stats"] = dict(stats)
    return ep


# ==================================================================== selftest
# A gate nobody has seen fire is a gate that might not. Each case below is a draft written to
# break exactly one thing about a REAL turn out of the input file, so the checks run against real
# menus, real mappings and real template verdicts rather than a fixture that agrees with them by
# construction. `--selftest` prints which code each one produced; a case whose `want` code is
# missing is a check that has stopped working, and it exits non-zero.
#
# The fixture prose is deliberately colourless: no ring name, no functional group, no claim about
# the skeleton. A "clean" draft that says "the benzyl ether is still on it" is only clean on the
# molecules that have one, and a selftest whose baseline fires the identity checks cannot tell a
# broken gate from a badly written fixture.
_MOL = ("target: the molecule this route is for\n"
        "this piece: a fragment that still has to be built\n"
        "not in the target: none")
_DOES = "divides it in two"        # overridden per candidate from its own measured shape
def _mk_draft(mid: str, rows: list[str], molecule: str = None, deciding: str = None,
              leaves: str = None) -> str:
    """The three blocks. `leaves to make` rides on DECIDING now -- WHY is gone.

    DECIDING is a route loop, and every case below was written against the single-pass form. The
    selftest installs `_DECIDING_WRAP` -- a closure over the fixture turn's own `order`, `menu`
    and `untried` -- and it wraps a bare body into a truthful loop here, so each case still
    tests the one thing it was written for instead of all of them failing on the shape.
    """
    dec = deciding or "ruled out: none\nordered by: plausibility | unstated"
    w = globals().get("_DECIDING_WRAP")
    if w is not None:
        dec = w(dec) or dec
    return ("MOLECULE\n" + (molecule or _MOL) + "\n"
            f"DISCONNECTIONS OF {mid}\n" + "\n".join(rows) + "\n"
            f"DECIDING {mid}\n" + dec
            + (("\n" + leaves) if leaves else ""))


def _absent_ring_word(ev: dict, k: int, ep: dict) -> str | None:
    """A ring-class name whose signature no molecule in scope has, for the attribution case.

    Picked from `board.factcheck`'s own vocabulary against the episode's own molecules rather
    than hard-coded, so the case exercises the shared check on this turn instead of asserting
    that some fixture disagrees with itself.
    """
    if FC is None or Chem is None:
        return None
    try:
        seen = FC.smiles_seen(ep["turns"][i].get("evidence") or {} for i in range(k + 1))
        have = set()
        for smi in seen:
            have |= FC._ring_sigs(smi)
        for word, sig in FC.RING_SIG.items():
            if sig not in have and len(word) > 6:
                return word
    except Exception:                                                  # noqa: BLE001
        return None
    return None


def selftest(eps: list[dict], a) -> int:
    """Run every gate against a real rank turn. Returns a shell exit status."""
    # Not the first suitable turn -- the RICHEST one. A case whose precondition the turn does not
    # meet is a case that silently does not run, and a gate that silently does not run is exactly
    # what this exists to catch.
    best, pick = -1, None
    for ep in eps:
        for k, t in enumerate(ep["turns"]):
            if LEGACY.turn_kind(t) != "rank":
                continue
            orders = ranked_orders(t)
            ev = t.get("evidence") or {}
            if len(orders) != 1:
                continue
            mid = next(iter(orders))
            order, facts, menu = orders[mid], facts_of(ev, mid), menu_of(ev, mid)
            if len(order) < 2 or not facts:
                continue
            qs = {ci: (c.get("signals") or {}).get("q") for ci, c in menu.items()}
            qs = {ci: v for ci, v in qs.items() if v is not None}
            score = (any(f.get("named_tier") != "applies+makes" for f in facts.values())
                     + any(not f.get("bond_measured") for f in facts.values())
                     + any(f.get("bond_measured") for f in facts.values())
                     + bool(qs and order[0] != max(qs, key=lambda c: qs[c])))
            if score > best:
                best, pick = score, (ep, k, mid, order)
            if score == 4:
                break
        if best == 4:
            break
    if not pick:
        print("  selftest: no single-molecule rank turn with >=2 ranked candidates and facts")
        return 1
    ep, k, mid, order = pick
    ev = ep["turns"][k].get("evidence") or {}
    facts, menu = facts_of(ev, mid), menu_of(ev, mid)
    head = order[0]

    def _row(ci, bond=None, cls=None, split=None, role=None):
        """A row built FROM this candidate's own facts, so the baseline contradicts nothing.

        `role` is accepted and ignored: the rows carry no ranking any more. Kept in the
        signature because a handful of cases below pass it to say "this row is the head", which
        is now expressed by the route groups instead.
        """
        f = facts.get(ci) or {}
        if bond is None:
            if not f.get("bond_measured"):
                bond = "not measured"
            else:
                p = bond_pairs_of(f)
                bits = []
                if p["formed"]:
                    bits.append(_pair_str(p["formed"]) + " formed")
                if p["broken"]:
                    bits.append(_pair_str(p["broken"]) + " broken")
                # No fallback when `bits` is empty. A "not measured" default is a false claim
                # on a candidate the board DID map (`bond_withheld`), and spelling the
                # order-changed pair out in prose is not what `parse_bond_pairs` reads, so it
                # mis-attributes the pair and trips `bond_pair_wrong`. The empty field trips
                # `row_field_empty` on a fixture turn whose only change is a bond order, which
                # is a defect in the FIXTURE's choice of turn.
                bond = "; ".join(bits)
        if cls is None:
            cls = ((f.get("named") or ["none"])[0]
                   if f.get("named_tier") == "applies+makes" else "none")
        if split is None:
            sh = f.get("shape") or {}
            v = split_verdict(sh).split(" --")[0].strip() if sh else "unbalanced"
            fr = sh.get("fragments") or []
            sk = ("skeleton kept" if sh.get("scaffold_kept")
                  else "skeleton changes" if sh.get("scaffold_kept") is not None else "")
            split = ", ".join(x for x in
                              [v, (" + ".join(str(n) for n in fr[:2]) + " heavy") if len(fr) > 1
                               else "", sk] if x)
        return (f"c{ci} | bond: {bond} | reaction: {cls} "
                f"| what it does: {split or _DOES}")

    # The baseline's deciding axis is chosen so it CANNOT contradict this turn: the axis that
    # actually separates the ranked candidates and whose best is the head. If no axis does, the
    # baseline case is dropped rather than reported as a failure -- that is a fact about the
    # turn, not about the gates.
    safe_axis = None
    for ax in AXES:
        vals = _axis_values(menu, order, ax)
        if len(vals) == len(order) and len(set(vals.values())) > 1 \
                and vals[head] == min(vals.values()):
            safe_axis = ax
            break
    # A weighed alternative can only be honestly pruned once an axis is settled: "outranked"
    # means worse on THAT axis, so the spare candidate has to actually read worse on it -- the
    # direction check would otherwise refuse the fixture's own baseline for the same
    # contradiction it exists to catch.
    extra = None
    _traps0 = cycle_traps(ep, k, mid)
    if safe_axis is not None:
        head_key = _axis_key(menu[head], safe_axis)
        for ci in sorted(facts):
            if ci in order or ci not in menu or ci in _traps0:
                continue
            ci_key = _axis_key(menu[ci], safe_axis)
            if ci_key is not None and head_key is not None and ci_key > head_key:
                extra = ci
                break
    # Traps get no row: `cycle_described` refuses one, and rightly -- so a fixture that
    # described one would fail its own baseline for the defect the gate exists to catch.
    _traps = _traps0
    if extra is None:
        extra = next((ci for ci in sorted(facts)
                      if ci not in order and ci in menu and ci not in _traps), None)
    # Ascending candidate number, or the baseline fails `cuts_rank_order` -- the fixture
    # cannot be written in an order the form refuses.
    _cands = sorted([ci for ci in order if ci not in _traps]
                    + ([extra] if extra is not None else []))
    good_rows = [_row(ci) for ci in _cands]
    # Words, not a copy of the score column -- `weigh_recites` refuses the latter, and the
    # baseline has to be a draft the gates accept.
    # Every candidate that would regenerate an ancestor has to be pruned by name or `_c_cycle`
    # refuses the baseline for the very defect it exists to catch. Computed from the turn, like
    # `safe_axis`, so the fixture cannot disagree with the board it was built from.
    # `cyc` built a `ruled out: ... | regenerates an ancestor` line here. That slot is gone, so
    # the string was dead and any case still passing it would have been refused as malformed.
    # The trap facts are still used -- `good_rows` excludes them, because `cycle_described`
    # refuses a row for a candidate the board would refuse.
    # (axis, reading) pairs already claimed by a group in the draft being built.
    _used_obj: set = set()

    def _route_axis(ri: int) -> str | None:
        """The OBJECTIVE route `ri` is kept for, chosen the way the check judges it.

        Route 1 names a measure it leads on. A later route names one it BEATS THE HEAD on, and
        `structure` when it beats the head on nothing -- that is the multi-objective statement,
        and a fixture that reused route 1's axis would fail `ordered_not_differentiating` on
        its own baseline. Objectives are also not repeated across groups.
        """
        if ri == 0:
            _used_obj.add((safe_axis, _axis_say(menu, order[0], safe_axis).lower()))
            return safe_axis
        mine, head0 = menu.get(order[ri]) or {}, menu.get(order[0]) or {}
        for ax in AXES:
            a1, a0 = _axis_key(mine, ax), _axis_key(head0, ax)
            if not (a1 is not None and a0 is not None and a1 < a0):
                continue
            # ...and not an objective another group in this draft already claimed at the same
            # reading: `ordered_same_objective` refuses that, correctly -- two routes kept for
            # one objective at one value are one route written twice. Two branch-queue routes
            # of the same family routinely read identically, so the fixture has to notice.
            key = (ax, _axis_say(menu, order[ri], ax).lower())
            if key in _used_obj:
                continue
            _used_obj.add(key)
            return ax
        cf = facts.get(order[ri]) or {}
        if cf.get("named_tier") in ("applies", "applies+makes") or cf.get("bond_measured"):
            return "struct"
        return None       # no objective exists: the line is left off -- see ordered_missing

    def _diverge_line(ri: int) -> str:
        """A TRUE `diverging from` for route `ri`, at the level the BOARD requires.

        Same gate as the check: a cut whose family already appears on the reaction ledger owes
        the ledger line; one whose family does not owes the other ranked candidates. The
        fixture cannot pick the easier level or it fails its own baseline on
        `diverging_wrong_level`.
        """
        hist = history_reactions(ep, k)
        mine = family_key(facts.get(order[ri]) or {})
        overlap = next((h for h in hist
                        if h["family"] == mine and mine[0] != "unmeasured"), None)
        if overlap is not None:
            return (f"  diverging from: {overlap['rid']} | same step, {family_label(mine)} | "
                    f"that kind is already expanded there, and this candidate is still worth "
                    f"taking on this piece")
        sibs = [c for c in order if c != order[ri]]
        pick = next((c for c in sibs
                     if family_key(facts.get(c) or {}) != mine
                     and family_key(facts.get(c) or {})[0] != "unmeasured"), None)
        if pick is not None:
            return (f"  diverging from: c{pick} | it is "
                    f"{family_label(family_key(facts.get(pick) or {}))} | this cut changes "
                    f"different bonds, so the two are different bets on this piece")
        same_sib = next((c for c in sibs
                         if family_key(facts.get(c) or {}) == mine
                         and mine[0] != "unmeasured"), None)
        if same_sib is not None:
            return (f"  diverging from: c{same_sib} | same step, {family_label(mine)} | the "
                    f"same kind of cut taken on the other fragment of this piece")
        # No family overlap on the ledger and no sibling to compare against -- a call that
        # ranks ONE candidate. `none` is refused here (`diverging_unearned`) because the board
        # HAS expanded things, and the level gate only forces the candidate level where there
        # are candidates, so citing the ledger is what is left and what is true.
        if hist:
            h0 = next((h for h in hist if h["family"][0] != "unmeasured"), hist[0])
            lab0 = (family_label(h0["family"]) if h0["family"][0] != "unmeasured"
                    else "a step the board records no mapping for")
            return (f"  diverging from: {h0['rid']} | spent on {lab0} | this cut changes "
                    f"different bonds, so it opens the piece from another side")
        return ("  diverging from: none | nothing has been expanded on this board yet, so "
                "there is no earlier route to differ from")

    def D(body: str | None) -> str | None:
        """A bare DECIDING body, wrapped into the route loop the form now requires.

        Every case below was written against the single-pass form. Wrapping here keeps each one
        testing the single thing it was written for instead of all of them failing on the shape.
        The ruled-out lines belong to route 1: they name candidates the call does not rank, and
        a pruned candidate is argued once.
        """
        if body is None:
            return None
        if _ROUTE_RE.search(body) or re.search(r"(?m)^\s*route\s+\d", body):
            return body
        out_l = [f"route 1 -- c{order[0]}"]
        for ln in body.rstrip("\n").split("\n"):
            out_l.append("  " + ln.strip() if ln.strip() else ln)
        # A `ruled out` line in a case body is now a REMOVED slot, and leaving it in would make
        # every such case fail on `deciding_malformed` instead of the thing it tests. Dropped
        # here, and the group gets the two lines the form actually has.
        out_l = [x for x in out_l if not _RULED_RE.match(x.strip())
                 and not _LEAVES_RE.match(x.strip())]
        if not _DIVERGING_RE.search("\n".join(out_l)):
            out_l.insert(1, _diverge_line(0))
        for ri in range(1, len(order)):
            out_l.append(f"route {ri + 1} -- c{order[ri]}")
            out_l.append(_diverge_line(ri))
            ax = _route_axis(ri)
            # None means this route has no objective it can honestly be kept for -- it loses
            # on every number and carries no named match and no measured bond.
            # `ordered_missing` exempts exactly those routes, so
            # the line is left off rather than filled with an ungrounded `structure`.
            if ax:
                out_l.append(f"  ordered by: {AXIS_TOKEN.get(ax, 'structure')} | value: "
                             f"{_axis_say(menu, order[ri], ax)} | of what the routes "
                             f"above already took, this is the cut that still "
                             f"reads first here")
        return "\n".join(out_l)

    # Installed for `_mk_draft` to pick up, so no case below has to know about the loop.
    globals()["_DECIDING_WRAP"] = D

    ok_deciding = (f"ordered by: {AXIS_TOKEN[safe_axis]} | value: "
                   f"{_axis_say(menu, order[0], safe_axis)} | it is the measure "
                   f"that actually separates these cuts, and the others leave them "
                   f"level") if safe_axis else None
    ok_draft = _mk_draft(mid, good_rows, deciding=ok_deciding)
    # The first row's own words, for the restatement fixture: an argument built out of these is
    # by construction a paraphrase of the ledger sitting above it.
    _row_words = " ".join(good_rows[0].split("|")[1:]).replace(":", " ")

    unearned = next((ci for ci in order if (facts.get(ci) or {}).get("named_tier")
                     != "applies+makes"), None)
    unmeasured = next((ci for ci in order if not (facts.get(ci) or {}).get("bond_measured")),
                      None)
    measured = next((ci for ci in order if (facts.get(ci) or {}).get("bond_measured")), None)
    qvals = {ci: (c.get("signals") or {}).get("q") for ci, c in menu.items()}
    qhave = {ci: v for ci, v in qvals.items() if v is not None}
    qbest = max(qhave, key=lambda c: qhave[c]) if qhave else None

    cases: list[tuple[str, str, str]] = [
        *([("clean draft", "", ok_draft)] if safe_axis else []),
        ("missing block", "schema_missing",
         ok_draft.split(f"DECIDING {mid}")[0]),
        ("text outside the blocks", "schema_stray",
         "Here is my analysis of the turn.\n\n" + ok_draft),
        ("unaddressed block", "schema_unaddressed",
         ok_draft.replace(f"DISCONNECTIONS OF {mid}", "DISCONNECTIONS")),
        ("malformed row", "row_malformed",
         _mk_draft(mid, [f"c{head} cuts the amide and looks good"], deciding=ok_deciding)),
        ("row for a candidate that does not exist", "row_unknown",
         _mk_draft(mid, good_rows + [f"c{max(menu) + 7} | bond: not measured | "
                                     f"reaction: none | what it does: halves it"],
                   deciding=ok_deciding)),
        ("a ranked candidate with no row", "cuts_missing",
         _mk_draft(mid, good_rows[:1], deciding=ok_deciding)),
        ("prose where a field belongs", "molecule_malformed",
         _mk_draft(mid, good_rows, deciding=ok_deciding,
                   molecule="The piece still has to be built and the cut has to earn its place.")),
        ("no route group in DECIDING", "deciding_missing",
         _mk_draft(mid, good_rows,
                   deciding="route 0 -- this piece is the one I am ranking")),
        ("an axis the board does not carry", "axis_unknown",
         _mk_draft(mid, good_rows,
                   deciding="ordered by: SCScore | value: none | it is the more "
                            "tractable fragment")),
        ("the score column recited back", "deciding_recites",
         _mk_draft(mid, good_rows,
                   deciding=f"ordered by: {AXIS_TOKEN[safe_axis or 'p']} | value: "
                            f"{_axis_say(menu, order[0], safe_axis or 'p')} | "
                            + ", ".join(f"c{c}" for c in order))),
        ("a WHY block, which the form no longer has", "schema_extra",
         _mk_draft(mid, good_rows, deciding=ok_deciding)
         + "\nWHY\nleaves to make: nothing\nrisk: none"),
        # The whole argument has to be the restatement, not one line of it: `_why_of` now
        # collects every reason clause in DECIDING as well as `leaves to make`, so a single
        # restating line sitting beside honest ones scores below the threshold -- correctly.
        # WHOLE argument, not one line of it. `_why_of` collects `diverging from` as well now,
        # because that slot is most of DECIDING's prose and a repetition check that could not
        # see it would miss the part that repeats -- so a case that restates only in
        # `ordered by` is diluted by an honest diverging line and correctly scores below the
        # threshold. Both lines restate here.
        ("prose that restates the ledger", "prose_restates",
         _mk_draft(mid, good_rows, deciding="\n".join([
             f"route 1 -- c{order[0]}",
             f"  diverging from: none | {_row_words} {_row_words}",
             f"  ordered by: {AXIS_TOKEN[safe_axis or 'p']} | value: "
             f"{_axis_say(menu, order[0], safe_axis or 'p')} | {_row_words} {_row_words}"]
             + [x for i, ci in enumerate(order) if i > 0
                for x in (f"route {i + 1} -- c{ci}",
                          f"  diverging from: none | {_row_words} {_row_words}")]))),
        ("the ranking as its own reason", "order_appeal",
         _mk_draft(mid, good_rows,
                   deciding=f"ordered by: {AXIS_TOKEN[safe_axis or 'p']} | value: "
                            f"{_axis_say(menu, order[0], safe_axis or 'p')} | the board has "
                            f"c{head} first, which is why it goes first")),
        ("interface vocabulary", "jargon",
         _mk_draft(mid, good_rows, deciding=ok_deciding,
                   leaves="leaves to make: whatever the menu offers")),
        ("the oracle", "leak_phrase",
         _mk_draft(mid, good_rows, deciding=ok_deciding,
                   leaves="leaves to make: I was told which cut is right")),
        ("a depth ceiling", "depth_ceiling",
         _mk_draft(mid, good_rows, deciding=ok_deciding,
                   leaves="leaves to make: with seven levels remaining there is room")),
        ("a call ceiling", "budget_ceiling",
         _mk_draft(mid, good_rows, deciding=ok_deciding,
                   leaves="leaves to make: the last of the budget")),
    ]
    if measured is not None:
        cases.append(("atom-map index", "atom_index",
                      _mk_draft(mid, [_row(ci, bond="formed C:1-C:2") if ci == measured
                                      else _row(ci) for ci in order], deciding=ok_deciding)))
        cases.append(("a bond field with no element pair", "bond_no_pair",
                      _mk_draft(mid, [_row(ci, bond="the bond at the benzylic position")
                                      if ci == measured else _row(ci) for ci in order],
                                deciding=ok_deciding)))
        # A pair the mapping does not have, chosen so it cannot coincide with the real one.
        real = bond_pairs_of(facts[measured])
        # Sorted, and against every side. The pairs in `real` are sorted tuples, so an unsorted
        # candidate never matched one and the fixture happily picked a pair the step really has.
        seen = {p for side in BOND_SIDES for p in real[side]}
        fake = next((p for p in (tuple(sorted(x)) for x in
                                 (("C", "Br"), ("C", "N"), ("C", "S"), ("B", "C"), ("C", "I")))
                     if p not in seen), ("Br", "C"))
        cases.append((f"an element pair the mapping refutes ({fake[0]}-{fake[1]})",
                      "bond_pair_wrong",
                      _mk_draft(mid, [_row(ci, bond=f"forms {fake[0]}-{fake[1]} at the "
                                                    f"benzylic carbon")
                                      if ci == measured else _row(ci) for ci in order],
                                deciding=ok_deciding)))
    if unearned is not None:
        cases.append(("a reaction no template reproduces", "reaction_unearned",
                      _mk_draft(mid, [_row(ci, cls="Suzuki coupling") if ci == unearned
                                      else _row(ci) for ci in order], deciding=ok_deciding)))
    if unmeasured is not None:
        cases.append(("a bond the mapping never measured", "bond_unmeasured",
                      _mk_draft(mid, [_row(ci, bond="C-O formed at the benzylic carbon")
                                      if ci == unmeasured else _row(ci) for ci in order],
                                deciding=ok_deciding)))
    absent = _absent_ring_word(ev, k, ep)
    if absent:
        cases.append((f"a ring class no molecule has ({absent})", "ring_identity",
                      _mk_draft(mid, good_rows, deciding=ok_deciding,
                                molecule=f"target: a {absent} bearing one substituent\n"
                                      f"this piece: an unbuilt fragment\n"
                                      f"not in the target: none")))
    # delta, both directions -- a group the piece does not have, and one the target shares.
    smi = ((ev.get("mols") or {}).get(mid) or {}).get("smiles") or ""
    tgt = ep.get("target") or ""
    if smi and tgt and FC is not None and Chem is not None:
        have, in_t = group_words(smi), group_words(tgt)
        vocab = set(FC.RING_SIG) | set(FC.FUSED_SIG) | set(FC.FG_SMARTS)
        gone = sorted(vocab - have)
        shared = sorted(have & in_t)
        if gone:
            cases.append((f"a delta the piece does not have ({gone[0]})", "delta_absent",
                          _mk_draft(mid, good_rows, deciding=ok_deciding,
                                    molecule=f"target: the molecule this route is for\n"
                                          f"this piece: an unbuilt fragment\n"
                                          f"not in the target: {gone[0]}")))
        if shared:
            cases.append((f"a delta the target also has ({shared[0]})", "delta_in_target",
                          _mk_draft(mid, good_rows, deciding=ok_deciding,
                                    molecule=f"target: the molecule this route is for\n"
                                          f"this piece: an unbuilt fragment\n"
                                          f"not in the target: {shared[0]}")))
    # an axis that reads the same on every ranked candidate
    flat = next((ax for ax in AXES
                 if len(_axis_values(menu, order, ax)) == len(order)
                 and len(set(_axis_values(menu, order, ax).values())) == 1), None)
    if flat and len(order) > 1:
        cases.append((f"an axis that separates nothing ({flat})", "decides_flat",
                      _mk_draft(mid, good_rows,
                                deciding=f"ordered by: {AXIS_TOKEN[flat]} | value: "
                                        f"{_axis_say(menu, order[0], flat)} | it reads the same on all of "
                                         f"them and still orders them")))
    if qbest is not None and head != qbest and qbest in order:
        cases.append(("q named while pointing elsewhere", "decides_order",
                      _mk_draft(mid, good_rows,
                                deciding=f"ordered by: confidence | value: "
                                         f"{_axis_say(menu, order[0], 'q')} | the expander is "
                                         f"most confident in the one I put first")))
    # ---------------------------------------------------------- the route loop's own gates
    # Each draft is the fixture's own VALID loop with one thing broken, so what fires is
    # attributable. `_DECIDING_WRAP` passes these through untouched: they already carry headers.
    def _grp(label: int, ci: int, diverging: str = None, ordered: bool = True) -> str:
        L = [f"route {label} -- c{ci}"]
        L.append("  " + diverging.strip() if diverging is not None
                 else _diverge_line(order.index(ci)))
        ax = _route_axis(order.index(ci))
        if ordered:
            L.append(f"  ordered by: {AXIS_TOKEN.get(ax, 'structure')} | value: "
                     f"{_axis_say(menu, ci, ax)} | of what the routes above already took, this "
                     f"is the cut that still reads first here")
        return "\n".join(x for x in L if x.strip())

    def _loop(*groups: str) -> str:
        return "\n".join(groups)

    hist_sel = history_reactions(ep, k)
    full = [_grp(i + 1, ci) for i, ci in enumerate(order)]

    if len(order) > 1:
        cases.append(("a route group missing for a ranked cut", "route_labels_wrong",
                      _mk_draft(mid, good_rows, deciding=_loop(full[0]))))
        cases.append(("route groups in the wrong order", "route_labels_wrong",
                      _mk_draft(mid, good_rows,
                                deciding=_loop(*[_grp(i + 1, ci) for i, ci in
                                                 enumerate(reversed(order))]))))
        cases.append(("no `diverging from` on a route", "diverging_missing",
                      _mk_draft(mid, good_rows,
                                deciding=_loop(full[0],
                                               f"route 2 -- c{order[1]}\n  ordered by: "
                                               f"{AXIS_TOKEN.get(_route_axis(1) or 'p', 'plausibility')}"
                                               f" | of what is left this one reads first"))))
        cases.append(("`ordered by` dropped where a field is still open", "ordered_missing",
                      _mk_draft(mid, good_rows,
                                deciding=_loop(_grp(1, order[0], ordered=False), *full[1:]))))
        cases.append(("the two lines of a group out of order", "deciding_slot_order",
                      _mk_draft(mid, good_rows, deciding=_loop(
                          "\n".join([f"route 1 -- c{order[0]}"]
                                    + list(reversed(full[0].split("\n")[1:]))),
                          *full[1:]))))
    # The ledger-level cases need a route whose cut IS the same kind as something already
    # expanded -- otherwise the level gate fires first and the case tests the gate instead of
    # what it was written for.
    _ov = [(i, ci, next((h for h in hist_sel
                         if h["family"] == family_key(facts.get(ci) or {})
                         and family_key(facts.get(ci) or {})[0] != "unmeasured"), None))
           for i, ci in enumerate(order)]
    _ov = [(i, ci, h) for i, ci, h in _ov if h is not None]
    if not _ov and hist_sel:
        print("  -- no ranked cut matches a ledger family on this turn: the route-level "
              "`diverging` gates are exercised on the separate turn below")
    if hist_sel and _ov:
        _ri, _ci, _h = _ov[0]

        def _swap(diverging: str) -> str:
            """The valid loop with ONE group's `diverging from` replaced."""
            return _loop(*[_grp(i + 1, ci, diverging=diverging) if i == _ri
                           else full[i] for i, ci in enumerate(order)])

        cases.append(("divergence from a step the ledger has no record of",
                      "diverging_not_claimed",
                      _mk_draft(mid, good_rows, deciding=_swap(
                          "diverging from: r999 | same step, whatever it took | it is already "
                          "expanded there"))))
        cases.append(("a kind called different where the ledger matches it",
                      "diverging_same_class",
                      _mk_draft(mid, good_rows, deciding=_swap(
                          f"diverging from: {_h['rid']} | spent on "
                          f"{family_label(_h['family'])} | this cut changes different bonds"))))
        _diff = next((h for h in hist_sel if h["family"][0] != "unmeasured"
                      and h["family"] != family_key(facts.get(_ci) or {})), None)
        if _diff is not None:
            cases.append(("a class the expanded step does not earn", "diverging_class_wrong",
                          _mk_draft(mid, good_rows, deciding=_swap(
                              f"diverging from: {_h['rid']} | same step, a Grignard addition | "
                              f"it is already expanded there"))))
    if safe_axis:
        cases.append(("a figure in the argument instead of `value:`", "deciding_number_in_why",
                      _mk_draft(mid, good_rows,
                                deciding=f"ordered by: {AXIS_TOKEN[safe_axis]} | value: "
                                         f"{_axis_say(menu, order[0], safe_axis)} | it reads "
                                         f"0.873 here, which is what puts it first")))
        _real = ((menu.get(order[0]) or {}).get("signals") or {}).get(safe_axis)
        if _real is not None and safe_axis in ("p", "q"):
            cases.append(("a `value:` the board does not carry", "ordered_value_wrong",
                          _mk_draft(mid, good_rows,
                                    deciding=f"ordered by: {AXIS_TOKEN[safe_axis]} | value: "
                                             f"{_real + 0.2:.3f} | it is the measure that "
                                             f"actually separates these cuts")))
    if len(order) > 1:
        # a later route kept for the HEAD's reason -- no objective of its own
        cases.append(("a later route kept for the head's own measure",
                      "ordered_not_differentiating",
                      _mk_draft(mid, good_rows, deciding=_loop(
                          full[0],
                          *[_grp(i + 1, ci).replace(
                              f"ordered by: {AXIS_TOKEN.get(_route_axis(i), 'structure')}",
                              f"ordered by: {AXIS_TOKEN[safe_axis or 'p']}")
                            .replace(f"value: {_axis_say(menu, ci, _route_axis(i))}",
                                     f"value: {_axis_say(menu, ci, safe_axis or 'p')}")
                            for i, ci in enumerate(order) if i > 0]))))
        # two routes on one objective, at the same reading -- one route written twice
        cases.append(("two routes kept for the same objective", "ordered_same_objective",
                      _mk_draft(mid, good_rows, deciding=_loop(
                          *[f"route {i + 1} -- c{ci}\n" + _diverge_line(i)
                            + f"\n  ordered by: {AXIS_TOKEN[safe_axis or 'p']} | value: "
                              f"{_axis_say(menu, order[0], safe_axis or 'p')} | it is the "
                              f"measure that separates these cuts"
                            for i, ci in enumerate(order)]))))
    if hist_sel:
        cases.append(("a route claiming nothing is expanded yet", "diverging_unearned",
                      _mk_draft(mid, good_rows, deciding=_loop(
                          _grp(1, order[0],
                               diverging="diverging from: none | nothing has been expanded on "
                                         "this board yet"), *full[1:]))))
        # `diverging_same_claimed` needs a ledger step of a DIFFERENT kind from the route's
        # own cut, and the level gate permits citing the ledger only where a kind MATCHES --
        # so this one is built on the overlapping route too, naming the wrong ledger line.
        if _ov:
            _ri2, _ci2, _h2 = _ov[0]
            mine2 = family_key(facts.get(_ci2) or {})
            diff2 = next((h for h in hist_sel if h["family"][0] != "unmeasured"
                          and h["family"] != mine2), None)
            if diff2 is not None:
                cases.append(("two different kinds called the same step",
                              "diverging_same_claimed",
                              _mk_draft(mid, good_rows, deciding=_loop(
                                  *[_grp(i + 1, ci,
                                         diverging=f"diverging from: {diff2['rid']} | same "
                                                   f"step, {family_label(diff2['family'])} | "
                                                   f"on the other fragment")
                                    if i == _ri2 else full[i]
                                    for i, ci in enumerate(order)]))))

    print(f"  selftest on episode {ep['id'][:30]} turn {k}, molecule {mid}, "
          f"call {' > '.join(f'c{c}' for c in order)}\n")
    bad = 0
    for label, want, draft in cases:
        v = verify(draft, ep, k, a)
        fv = factcheck(draft, ep, k, v.get("doc") or parse_document(draft), a)
        got = set(v.get("codes") or []) | set(fv.get("codes") or [])
        hit = (not got) if not want else (want in got)
        if not hit:
            bad += 1
        print(f"  {'ok ' if hit else '!! '}{label:<46} want {want or '(nothing)':<18} got "
              f"{', '.join(sorted(got)) or '(nothing)'}")
    if a.show_selftest:
        print("\n" + "=" * 74 + "\n  a clean draft, and a repair message\n" + "=" * 74)
        print(ok_draft)
        v = verify(cases[5][2], ep, k, a)
        fv = factcheck(cases[5][2], ep, k, v.get("doc"), a)
        print("\n--- repair ---\n" + describe_repair(v, fv))
    # The target-lock mechanism gets its own inline check: `ep["_target_canon"]` has to be set
    # ONLY for this one verification, not for the whole list above it, or every other fixture's
    # `molecule:` line trips it too (they were not written to match an arbitrary locked
    # phrase). Set immediately before, read immediately after, restored immediately after that.
    prior_canon = ep.get("_target_canon")
    ep["_target_canon"] = "a completely different established phrase for the target"
    drift_draft = _mk_draft(mid, good_rows, deciding=ok_deciding,
                            molecule="target: something else entirely\nthis piece: an "
                                     "unbuilt fragment\nnot in the target: none")
    v = verify(drift_draft, ep, k, a)
    fv = factcheck(drift_draft, ep, k, v.get("doc") or parse_document(drift_draft), a)
    if prior_canon is None:
        ep.pop("_target_canon", None)
    else:
        ep["_target_canon"] = prior_canon
    got = set(v.get("codes") or []) | set(fv.get("codes") or [])
    hit = "target_drift" in got
    if not hit:
        bad += 1
    print(f"  {'ok ' if hit else '!! '}{'a target that drifts from the established phrase':<46} "
          f"want target_drift         got {', '.join(sorted(got)) or '(nothing)'}")

    # `diverging_same_class` fires only where a cut being ranked is the SAME KIND as a step
    # already on the board's reaction ledger. The turn picked above need not be one of those, so
    # this gate gets its own turn, the way `target_drift` gets its own canon. Extra codes do not
    # matter: the assertion is that this one is among them.
    same_pick = None
    for ep2 in eps:
        for k2, t2 in enumerate(ep2["turns"]):
            if LEGACY.turn_kind(t2) != "rank":
                continue
            h2 = history_reactions(ep2, k2)
            if not h2:
                continue
            ev2 = t2.get("evidence") or {}
            for mid2, ord2 in (ranked_orders(t2) or {}).items():
                if not ord2:
                    continue
                f2 = facts_of(ev2, mid2)
                for ci in ord2:
                    key = family_key(f2.get(ci) or {})
                    if key[0] == "unmeasured":
                        continue
                    hit = next((h for h in h2 if h["family"] == key), None)
                    if hit is not None:
                        same_pick = (ep2, k2, mid2, ord2, ci, hit, key)
                        break
                if same_pick:
                    break
            if same_pick:
                break
        if same_pick:
            break
    if same_pick:
        ep2, k2, mid2, ord2, ci_same, hit, key = same_pick
        lab = family_label(key)
        rows2 = [f"c{c} | bond: not measured | reaction: none | "
                 f"what it does: it divides the piece" for c in ord2]
        dec2 = []
        for i, c in enumerate(ord2, start=1):
            dec2.append(f"route {i} -- c{c}")
            if c == ci_same:
                # the one thing the slot refuses: a claim of different chemistry where the board
                # groups this cut and that ledger step under one kind.
                dec2.append(f"  diverging from: {hit['rid']} | spent on {lab} | this cut "
                            f"changes different bonds, so it opens the piece elsewhere")
            else:
                dec2.append(f"  diverging from: {hit['rid']} | same step, {lab} | the same "
                            f"kind is already expanded there")
        d2 = _mk_draft(mid2, rows2, deciding="\n".join(dec2))
        v2 = verify(d2, ep2, k2, a)
        fv2 = factcheck(d2, ep2, k2, v2.get("doc") or parse_document(d2), a)
        got2 = set(v2.get("codes") or []) | set(fv2.get("codes") or [])
        hit2 = "diverging_same_class" in got2
        if not hit2:
            bad += 1
        print(f"  {'ok ' if hit2 else '!! '}"
              f"{'one kind called a different expansion (c%d vs %s)' % (ci_same, hit['rid']):<46} "
              f"want diverging_same_class got "
              f"{', '.join(sorted(c for c in got2 if c.startswith('diverging'))) or '(none)'}")
    else:
        print("  -- no rank turn shares a family with the ledger: `diverging_same_class` NOT "
              "exercised")

    # The open slot. It is checked on real open turns rather than on a synthetic board because
    # both clauses are claims about parentage -- which reaction each opened piece sits under --
    # and a hand-built `mols` would prove nothing about the boards that are actually generated.
    # A run is expected to contain both shapes; where one is absent it is reported, not passed.
    n_open = 0
    if FC is not None:
        def _open_turn(pred):
            for ep3 in eps:
                for t3 in ep3["turns"]:
                    if LEGACY.turn_kind(t3) != "open":
                        continue
                    ev3 = t3.get("evidence") or {}
                    oa3, op3 = FC.open_choice(ev3, t3.get("actions"))
                    if pred(ev3, oa3, op3):
                        return t3, ev3, oa3, op3
            return None

        def _open_case(label: str, want: str, text: str, t3: dict, ev3: dict) -> None:
            nonlocal bad, n_open
            n_open += 1
            r3 = FC.check(text, ev3, "open", actions=t3.get("actions"))
            got3 = set(r3.get("codes") or [])
            ok3 = want in got3 if want else not any(c.startswith("open_") for c in got3)
            if not ok3:
                bad += 1
            print(f"  {'ok ' if ok3 else '!! '}{label:<46} "
                  f"want {want or 'no open code':<24} got "
                  f"{', '.join(sorted(c for c in got3 if c.startswith('open_'))) or '(none)'}")

        picked = _open_turn(lambda ev3, oa3, op3: len(op3) >= 2 and all(m in op3 for m in oa3))
        if picked:
            t3, ev3, _oa3, op3 = picked
            mols3 = ev3.get("mols") or {}
            pair = ", ".join(f"{m} under {mols3[m].get('under')}" for m in op3 if m in mols3)
            base = "Neither piece can be bought as it stands. "
            _open_case("a batch opened with no reason given", "open_together_missing",
                       base + "Their disconnections are the only move.", t3, ev3)
            _open_case("the batch named with its reactions", "",
                       base + f"opening together: {pair} | rival cuts of one parent, held at "
                              f"one depth so the next turn can rank them side by side",
                       t3, ev3)
            _open_case("a piece put under the wrong reaction", "open_together_under_wrong",
                       base + f"opening together: {op3[0]} under r99, "
                              f"{op3[1]} under {mols3[op3[1]].get('under')} | rival cuts",
                       t3, ev3)
            _open_case("rival cuts called one step", "open_together_relation",
                       base + f"opening together: {pair} | both pieces of the same step, "
                              f"which owes them all", t3, ev3)
            _open_case("the batch listed short one piece", "open_together_incomplete",
                       base + f"opening together: {op3[0]} under "
                              f"{mols3[op3[0]].get('under')} | a cut worth taking to depth",
                       t3, ev3)
            _open_case("`leaving open` where nothing is left", "open_choice_unearned",
                       base + f"opening together: {pair} | rival cuts at one depth\n"
                              f"leaving open: {op3[0]} | not this turn", t3, ev3)
        else:
            print("  -- no multi-open turn: the `opening together` gates NOT exercised")

        picked = _open_turn(lambda ev3, oa3, op3: len(op3) == 1
                            and [m for m in oa3 if m not in op3])
        if picked:
            t3, ev3, oa3, op3 = picked
            left3 = [m for m in oa3 if m not in op3]
            base = f"{op3[0]} cannot be bought as it stands. "
            _open_case("a piece passed over in silence", "open_choice_missing",
                       base + "Asking for its disconnections is the only move available.",
                       t3, ev3)
            _open_case("the piece left, named", "",
                       base + f"leaving open: {left3[0]} | it sits deeper and its reaction "
                              f"still owes another piece", t3, ev3)
            _open_case("the piece it opens called the one it leaves", "open_choice_wrong",
                       base + f"leaving open: {op3[0]} | not this turn", t3, ev3)
            # `as`, `at`, `an`, `be`, `by`, `we`, `us` and `up` are all live board ids, so the
            # clause's prose half must not be read for ids. This is the gate on that.
            _open_case("english words in the reason, not ids", "",
                       base + f"leaving open: {left3[0]} | as it is at an early depth and we "
                              f"can be by it later", t3, ev3)
        else:
            print("  -- no narrowing open turn: the `leaving open` gates NOT exercised")

    print(f"\n  {len(cases) - bad + 2 + n_open}/{len(cases) + 2 + n_open} gates behaved "
          f"as expected")
    return 1 if bad else 0


# ==================================================================== io
def seed_thoughts(eps: list[dict], path: Path) -> int:
    """Fill non-rank turns from a file already written, so a rank-only run keeps its chain.

    `--kinds rank` writes rank turns and leaves the rest alone, which is the right experiment
    when the rank form is what is being iterated on -- and it is only the right experiment if
    the open turns before each rank turn still carry their reasoning. This copies them across
    from a previous run's output, matched on the same episode identity `load_episodes` uses.
    """
    prev = {}
    for r in (json.loads(l) for l in open(path) if l.strip()):
        prev[r.get("id") or r.get("target")] = r
    n = 0
    for e in eps:
        r = prev.get(e["id"])
        if r is None:
            continue
        tt = list(r.get("turn_thoughts") or [])
        if e["turns"] and LEGACY.turn_kind(e["turns"][-1]) == "final":
            tt.append(r.get("handover_thought") or "")
        for i, t in enumerate(e["turns"]):
            if i < len(tt) and tt[i] and not t.get("thought"):
                t["thought"] = tt[i]
                n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    """The CLI, factored out of `main` so another pass can rebuild the exact same `a`.

    Re-checking a draft against anything other than the settings it was drawn under is
    not a re-check. A pass that re-checks stored drafts needs `--fact-strict`, `--min-words`
    and the rest to mean what they meant during the run that produced the draft.
    """
    ap = argparse.ArgumentParser(
        description="board reasoning written to a schema: rank turns as a form, with the "
                    "register gate, the factual gate and the repair loop addressed per block")
    ap.add_argument("--in", dest="inp", default=None,
                    help="jsonl from render_board_episode.py --out. Required for everything "
                         "except --print-schema, which needs no episodes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-schema", action="store_true",
                    help="the system prompt a rank turn gets, and exit")
    ap.add_argument("--print-prompt", default=None,
                    help="EPISODE:TURN, e.g. 0:1 -- render one prompt and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="run every gate against a real rank turn out of --in, with drafts "
                         "written to break one thing each. No teacher, no GPU. Exits non-zero "
                         "if a gate did not fire")
    ap.add_argument("--show-selftest", action="store_true",
                    help="with --selftest: also print a clean draft and a repair message")
    ap.add_argument("--kinds", default="rank,open,done,final",
                    help="which turn kinds to write. `rank` gets the schema; open, done and "
                         "the hand-over keep the freeform briefs from "
                         "traj_route_reasoning_board until their own forms are designed. "
                         "`--kinds rank` with --seed-from is the rank-only experiment")
    ap.add_argument("--seed-from", default=None,
                    help="a previous run's output: copy the thoughts of the turns this run is "
                         "not writing, so the chain behind each rank turn is unbroken")
    ap.add_argument("--annotations", default=None,
                    help="a JSON or JSONL sidecar of STRUCTURAL NOTES, keyed by molecule SMILES "
                         "or by rxn_key (`product>>reactant.reactant`, reactants sorted). Each "
                         "note is printed with its molecule or its candidate as a given fact. "
                         "This is where a stronger model's reading of the structure goes: the "
                         "`bond:` field asks where a bond sits, that half of the field is "
                         "trusted rather than checked, and a note lets the teacher quote it "
                         "instead of inventing it")
    ap.add_argument("--history-turns", type=int, default=4,
                    help="how many recent turns are replayed as (reasoning, decision). "
                         "Everything older is one ledger line per turn, appended, so the "
                         "prompt prefix stays byte-identical turn to turn")
    ap.add_argument("--window-boards", type=int, default=0,
                    help="how many of those turns also get their board replayed verbatim. 0 by "
                         "default: the reasoning written for a turn already contains the "
                         "chemistry, and a whole board says what one ledger line says")
    ap.add_argument("--full-boards", type=int, default=4,
                    help="the freeform path's own board window, for open/done/hand-over turns")
    # -- teacher
    ap.add_argument("--base-url", default=os.getenv("TEACHER_BASE_URL",
                                                    "http://127.0.0.1:8000/v1"),
                    help="one URL, or several comma-separated: requests rotate over them and a "
                         "retry lands on the next replica rather than the one that just failed")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "teacher"))
    ap.add_argument("--api-key", default=os.getenv("TEACHER_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false. Required for "
                         "Qwen3.x: with thinking on it returns reasoning and no content")
    # A replica can stop scheduling while still answering `/v1/models` and `/health`, and a
    # frozen replica does not fail -- its requests hang. `_mark` only demotes a replica that
    # ERRORS, so every request to it burns the full timeout, several of them to reach
    # `_DOWN_AFTER`, and `stick_to` means the episodes homed there cannot go anywhere else on
    # their own. The ceiling is set above what a real request needs, so a real request cannot
    # trip it and a hang costs as little as possible. This bounds the damage; it does not
    # detect the freeze.
    ap.add_argument("--timeout", type=float, default=240.0)
    # -- draws
    ap.add_argument("--samples", type=int, default=2,
                    help="draws per turn at the requested temperature")
    ap.add_argument("--max-draws", type=int, default=12,
                    help="hard cap on draws for ONE turn. Past --samples the temperature is "
                         "raised, because a model repeating the same rejected draft needs a "
                         "different sample rather than another identical try. The numeric "
                         "checks fire PER CANDIDATE, so the more candidates a draft names the "
                         "lower the chance one draft gets all of them right. More claims per "
                         "draft needs more draws per turn, or the turn is dropped for a defect "
                         "the next sample would not have had")
    ap.add_argument("--variants", type=int, default=3,
                    help="how many fully clean drafts to KEEP per rank turn, not just how many "
                         "to draw. The draw budget is already being spent, so every extra kept "
                         "here is a rationale the run already paid for. They "
                         "explain the SAME action and pass the SAME gates, which is what makes "
                         "them usable: one instance with k true rationales instead of one, "
                         "which is the shape a corpus needs if reasoning is to add anything "
                         "over the non-reasoning baseline rather than diluting it. Written to "
                         "`thought_variants`; the wire builder fans them out into sibling rows "
                         "under the same target, which prepare_board_dataset already splits by. "
                         "1 keeps a single rationale per turn")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["low", "medium", "high", "keep"],
                    help="the `reasoning_effort` stamped on every output row, which is what the "
                         "chat template turns into the `Reasoning: <x>` line of the system "
                         "message. `low` sets a prior toward emitting no analysis channel, "
                         "which a corpus that carries reasoning on only some turns cannot "
                         "overcome. `medium` matches the data: the analysis blocks are a "
                         "paragraph or two of structured argument -- not the long chain `high` "
                         "sets a prior for. `keep` leaves whatever the input row carried. "
                         "WHATEVER IS CHOSEN HERE, SERVE THE EVAL WITH THE SAME VALUE: a "
                         "mismatch suppresses the reasoning the corpus teaches")
    ap.add_argument("--no-reject-reasoning", dest="reject_reasoning",
                    action="store_false", default=True,
                    help="leave a refused call with no analysis block. On by default: a refused "
                         "cycle candidate's argument is TRUE and merely incomplete, so it is "
                         "generated under the ordinary gates with the ancestor listing and the "
                         "cycle check withheld -- see `_reject_draft`. Costs one draw per "
                         "refused rank")
    ap.add_argument("--repair", type=int, default=2,
                    help="how many times a draft with an ADDRESSABLE defect -- a broken block, "
                         "a field the facts contradict -- is handed back with the line that is "
                         "wrong and redrawn, before the blind-resample budget is touched. A "
                         "repair round is not charged as a draw")
    ap.add_argument("--salvage", action="store_true", default=True,
                    help="after the draws, rewrite interface vocabulary out of the best draft "
                         "and keep it, marked `thought_salvaged`. Never applied to a leak, a "
                         "false claim or a broken form")
    ap.add_argument("--no-salvage", dest="salvage", action="store_false")
    ap.add_argument("--naive", action="store_true",
                    help="ablation control: SAME supporting information as the qualified arm, "
                         "but no output schema and no gate. The rank turn asks for prose "
                         "instead of the four-block form, and the first draft is kept whatever "
                         "verify and factcheck say. What the qualified arm adds over this is "
                         "the form and its verification, so those are the only two things it "
                         "removes. --samples 1 --max-draws 1 --repair 0 costs nothing extra "
                         "here: with no gate there is nothing for a redraw to fix.")
    ap.add_argument("--factcheck", action="store_true", default=True)
    ap.add_argument("--no-factcheck", dest="factcheck", action="store_false")
    ap.add_argument("--fact-strict", action="store_true",
                    help="treat every factual violation as a rejection. Off by default: only "
                         "the FATAL ones reject, because a hole propagates to every turn "
                         "written after it")
    ap.add_argument("--min-words", type=int, default=45)
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--max-words-brief", type=int, default=45,
                    help="word ceiling for open and done turns, where the brief asks for two "
                         "sentences. --max-words is sized for a rank turn's three blocks and "
                         "is far too loose here: without its own ceiling the brief is not "
                         "enforced")
    # -- run
    ap.add_argument("--episode-workers", type=int, default=4,
                    help="episodes in flight. Turns WITHIN an episode are sequential and "
                         "cannot be parallelised -- that is the point of the design")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fill", action="store_true",
                    help="re-run only the turns that came back empty in --out, keeping every "
                         "thought already written. Resume is per episode and cannot reach them")
    ap.add_argument("--fill-only-forced", action="store_true",
                    help="with --fill-from: redraw ONLY the listed turns, not every empty turn "
                         "in the episode. Without it a targeted pass silently carries a full "
                         "refill with it, and the effect of the two cannot be separated")
    ap.add_argument("--fill-from", default=None,
                    help="with --fill: a JSON {episode id: [turn, ...]} of turns to redo even "
                         "though they are not empty, as an audit pass over the stored drafts "
                         "writes it")
    return ap


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    # The prompt builders take no arguments and read module globals -- there are many call
    # sites, and setting it once here leaves less room to diverge than threading an argument.
    NAIVE["on"] = bool(getattr(a, "naive", False))
    if NAIVE["on"] and a.factcheck:
        print("# --naive: arm with no schema and no gate -- factcheck is off",
              file=sys.stderr)
        a.factcheck = False
    a.urls = [u.strip() for u in a.base_url.split(",") if u.strip()]
    a.kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}
    # `--repair` is what the freeform module calls `--fact-repair`; the legacy path reads that
    # name off the same namespace, so it is aliased rather than duplicated on the command line.
    a.fact_repair = a.repair

    a.order_block = True                      # accepted and ignored: there is no <order> block
    a.annotations_map = load_annotations(a.annotations)
    if a.print_schema:
        print(system_rank())
        return 0
    if not a.inp:
        raise SystemExit("--in is required (only --print-schema runs without episodes)")
    if not a.urls:
        raise SystemExit("--base-url is empty")

    eps = LEGACY.load_episodes(Path(a.inp))
    n_rank = sum(1 for e in eps for t in e["turns"] if LEGACY.turn_kind(t) == "rank")
    print(f"  {len(eps)} episodes, {sum(len(e['turns']) for e in eps)} turns, "
          f"{n_rank} of them rank", flush=True)
    if a.limit:
        eps = eps[:a.limit]
    if a.seed_from:
        print(f"  seeded {seed_thoughts(eps, Path(a.seed_from))} thoughts from "
              f"{a.seed_from}", flush=True)

    if a.selftest:
        return selftest(eps, a)
    if a.print_prompt:
        i, k = (int(x) for x in a.print_prompt.split(":"))
        for m in (turn_prompt(eps[i], k, a) if LEGACY.turn_kind(eps[i]["turns"][k]) == "rank"
                  else LEGACY.turn_prompt(eps[i], k, a.full_boards)):
            print(f"\n=============== {m['role'].upper()} ===============\n")
            print(m["content"])
        return 0

    if a.reasoning_effort != "keep":
        print(f"  reasoning_effort stamped on every row: {a.reasoning_effort.upper()}  "
              f"-- SERVE THE EVAL WITH THE SAME VALUE", flush=True)
    print(f"  teacher {a.model} over {len(a.urls)} replica(s); kinds "
          f"{', '.join(sorted(a.kinds))}", flush=True)
    if a.factcheck and FC is None:
        print(f"  !! --factcheck requested but board.factcheck did not import ({_FC_ERR}); "
              f"the shared claims will NOT be checked (the schema checks still run)", flush=True)
    elif a.factcheck:
        print(f"  factcheck on: {len(FC.CHECKS)} shared checks + the schema's own, "
              f"repair rounds {a.repair}" + (", strict" if a.fact_strict else ""), flush=True)
    if Chem is None:
        print("  !! rdkit missing: the frame block's delta is off", flush=True)
    if a.annotations_map:
        print(f"  structural notes: {len(a.annotations_map):,} keys from {a.annotations}",
              flush=True)

    out = Path(a.out) if a.out else Path(a.inp).with_suffix(".schema.jsonl")
    done = set()
    if out.exists():
        for r in (json.loads(l) for l in open(out) if l.strip()):
            done.add(r.get("id") or r.get("target"))
        print(f"  resuming: {len(done)} episodes already written", flush=True)

    out_final = None
    if a.fill:
        prev = {}
        for r in (json.loads(l) for l in open(out) if l.strip()):
            prev[r.get("id") or r.get("target")] = r
        forced = json.load(open(a.fill_from)) if a.fill_from else {}
        todo, holes = [], 0
        for e in eps:
            r = prev.get(e["id"])
            if r is None:
                todo.append(e)
                continue
            tt = list(r.get("turn_thoughts") or [])
            if e["turns"] and LEGACY.turn_kind(e["turns"][-1]) == "final":
                tt.append(r.get("handover_thought") or "")
            forced_here = [i for i in forced.get(str(e["id"]), ()) if 0 <= i < len(tt)]
            for i in forced_here:
                tt[i] = ""
            empty = [i for i in range(len(e["turns"]))
                     if not (tt[i] if i < len(tt) else None)
                     and LEGACY.turn_kind(e["turns"][i]) in a.kinds]
            # `--fill` means "every hole", and with `--fill-from` that quietly becomes "every hole
            # PLUS the listed turns" -- a full refill riding along with whatever targeted pass was
            # intended, and stacking a refill under a second change makes the effect of either
            # unattributable. This keeps the two separable: with the flag, ONLY the listed turns
            # are redrawn and the corpus's other holes are left exactly as they are.
            if a.fill_only_forced:
                empty = [i for i in empty if i in set(forced_here)]
            if not empty:
                continue
            for i, t in enumerate(e["turns"]):
                if i < len(tt) and tt[i]:
                    t["thought"] = tt[i]
            e["_fill"] = set(empty)
            holes += len(empty)
            todo.append(e)
        print(f"  fill: {len(todo)} episodes carry {holes:,} empty turns", flush=True)
        # A side file, swapped in at the end. Truncating the real output up front is unsafe in
        # exactly the case that matters: a full sweep puts every episode in `todo`, and a pass
        # killed part-way would have destroyed the episodes it had not reached.
        keep = [r for kk, r in prev.items() if kk not in {e["id"] for e in todo}]
        out_final, out = out, out.with_suffix(out.suffix + ".fill")
        with open(out, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  fill: writing to {out.name}, swapped in when the pass completes", flush=True)
    else:
        todo = [e for e in eps if e["id"] not in done]

    # Longest first: episode cost is turns x draws and the tail is heavy, so handing the long
    # episodes out last leaves one worker grinding a long chain while the rest sit idle.
    todo.sort(key=lambda e: -len(e.get("turns") or []))

    import concurrent.futures as cf
    lock = threading.Lock()
    agg = collections.Counter()
    t0 = time.time()
    with open(out, "a") as fh:
        with cf.ThreadPoolExecutor(max_workers=a.episode_workers) as ex:
            futs = {ex.submit(reason_episode, e, a): e for e in todo}
            for fut in cf.as_completed(futs):
                try:
                    ep = fut.result()
                except Exception as exc:                               # noqa: BLE001
                    print(f"  ! episode {futs[fut].get('id')}: {exc}", flush=True)
                    agg["failed"] += 1
                    continue
                row = ep["row"]
                tl = ep["turns"]
                if tl and LEGACY.turn_kind(tl[-1]) == "final":
                    row["handover_thought"] = tl[-1].get("thought", "")
                    tl = tl[:-1]
                row["turn_thoughts"] = [t.get("thought", "") for t in tl]
                row["reasoning_stats"] = ep.get("reasoning_stats")
                row["thought_checks"] = [t.get("thought_check") for t in tl]
                row["thought_facts"] = [t.get("thought_facts") for t in tl]
                # The best draft of a turn that ended EMPTY, and only those. A dropped turn is
                # the one thing this file cannot otherwise explain -- `thought_checks` says the
                # register was clean and says nothing about which claim was false -- and
                # carrying the rejected text for every turn would double the file to say it.
                row["thought_rejected"] = {str(i): t.get("thought_draft")
                                           for i, t in enumerate(tl)
                                           if not t.get("thought") and t.get("thought_draft")}
                # The reasoning written for each REFUSED call, keyed by the turn it sits in
                # front of. Also inlined into harmony_messages as an unsupervised analysis
                # block; carried here as well so a downstream pass can find every one without
                # walking the wire form.
                row["reject_thoughts"] = {str(i): (t.get("reject") or {}).get("thought")
                                          for i, t in enumerate(tl)
                                          if (t.get("reject") or {}).get("thought")}
                # {turn index: [the other clean drafts]}. Every one passed the same gates as
                # the winner and explains the same action, so the wire builder fans them out
                # into sibling rows under this target -- which is the unit prepare_board_dataset
                # splits by, so no variant of an episode can land on the other side of the split
                # from its siblings.
                row["thought_variants"] = {str(i): t["thought_variants"]
                                           for i, t in enumerate(tl)
                                           if t.get("thought_variants")}
                # The tag is the ONLY field that tells corpora written in different forms apart
                # downstream. This form -- DECIDING as a route loop, `take` groups meaning
                # "expanding now", `ordered by` with a `value:` slot, breadth-first episodes --
                # is not interchangeable with earlier ones, and rows of two forms in one
                # training set would show the student two answers to "what does DECIDING look
                # like", so the tag changes whenever the form does.
                row["reasoning_format"] = "route-loop-v6"
                if a.reasoning_effort != "keep":
                    row["reasoning_effort"] = a.reasoning_effort
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                for kk, vv in (ep.get("reasoning_stats") or {}).items():
                    agg[kk] += vv
                agg["ep"] += 1
                agg["turns"] += len(row["turn_thoughts"])
                agg["written"] += sum(1 for x in row["turn_thoughts"] if x)
                if agg["ep"] % 5 == 0:
                    print(f"  {agg['ep']}/{len(todo)} episodes  "
                          f"{agg['written']}/{agg['turns']} turns written  "
                          f"{agg['ep'] / max(time.time() - t0, 1e-9):.2f} ep/s", flush=True)
    if out_final is not None:
        out.replace(out_final)
        out = out_final
    print(f"\n  wrote {out}", flush=True)
    print(f"  {agg['written']}/{agg['turns']} turns carry a thought "
          f"({agg['written'] / max(agg['turns'], 1):.1%})", flush=True)
    rej = {k[2:]: v for k, v in agg.items() if k.startswith(("v_", "f_"))}
    if rej:
        print("  what the gates caught, worst first:")
        for k, v in sorted(rej.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {k:<24} {v:,}")
    print("  re-render with: render_board_episode.py --analysis text "
          "(reads turn_thoughts)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
