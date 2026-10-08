"""Does a drafted thought agree with the facts the turn was given?

`traj_route_reasoning_board.verify` reads a draft for REGISTER: interface vocabulary, an
oracle quote, an atom-map index, a signal named on a turn that has no candidates. Every one
of those is a property of the words alone. None of them looks at whether the sentence is
TRUE of the turn it was written for, and that is where a teacher's errors are:

  * "it is what q points at", written about a candidate that is not the argmax of q.
    `verify` sees a well-formed sentence.
  * a candidate whose own p is below the 0.05 cut-off applied in the same paragraph that
    says "the other cuts fall below the plausibility cut-off, so they cannot carry a route".
  * a wrong name: a Williamson etherification called a Suzuki coupling, a chloroformate an
    acetyl chloride, a thiophene a thiazole -- while `support_block` hands the teacher the
    earned verdict from `named_reactions.classify` and the SMILES the ring claim contradicts.
  * "this variant is complete: every piece reaches purchasable material", written on a turn
    whose evidence lists an open molecule.
  * and the one that needs no chemistry at all to see: the prose argues for one candidate and
    the tool call takes another. "c5 is the route to take" / "c7's transformation reproduces
    nothing" -> `order [7, 0, 5]`. "c0 ... sits second as a fallback" -> `I declare 0, 6`.

So this module asks the other question. Every check is a function of (draft text, the turn's
`evidence` dict) and returns violations with the number that refutes them, so a rejection can
be re-prompted with the specific correction rather than resampled blind -- resampling a claim
the teacher believes costs draws and mostly returns the same sentence.

WHAT IS AND IS NOT CHECKABLE HERE.

Checkable, and checked: any number quoted from the board (q, p, rt, ln$, heavy atoms, ring
counts), any comparative claim about those numbers, any named reaction (`ev["facts"]` carries
`named_tier` from a template run FORWARD), any ring-class or functional-group word (the
candidate SMILES are right there), the purchasability of a fragment, whether the skeleton
survives, and whether a completeness claim is true of the state.

Not checkable, and deliberately not attempted: whether the chemistry is a good idea, whether
the strategy is sound, whether the prose reads well. Those are what a teacher is for. A check
that cannot be decided from the evidence dict does not belong here, because a false rejection
costs a draw and teaches nothing.

FAIL-OPEN, ALWAYS. A check that cannot evaluate -- the field is absent, RDKit will not parse,
the regex found nothing to compare -- returns no violation. `bond: None` means "not measured"
everywhere else in this pipeline and it means the same thing here: silence is not evidence of
agreement, but neither is it grounds to throw away a draft. Only a claim that CONTRADICTS a
fact present in the evidence is a violation.

Usage:

    from board.factcheck import check, describe
    v = check(text, ep["turns"][k].get("evidence") or {}, kind, prev_thoughts=earlier)
    if not v["ok"]:
        print(describe(v))          # the correction to re-prompt with
"""
from __future__ import annotations

import difflib
import math
import re

from . import render as R
from collections import Counter

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:                                                      # noqa: BLE001
    Chem = None

CUTOFF = 0.05                       # the reaction filter's, as the developer block states it
REPEAT_RATIO = 0.6                  # a draft this much recycled is boilerplate -- see _c_repetition

# --------------------------------------------------------------------------- ring vocabulary
# A ring class is identified by (size, multiset of ring heteroatoms). That is all a name like
# "thiazole" asserts structurally, and it is exactly what a draft gets wrong: a
# 5-ring with one S called a thiazole (which needs S and N), a 5-ring with O and N called a
# 1,3,4-oxadiazole (which needs O and two N), a 6-5 bicycle with two N called purine-like
# (which needs four). Isomers that share a signature -- thiazole/isothiazole,
# imidazole/pyrazole -- are NOT separated here: the signature cannot tell them apart and a
# check that cannot decide must stay silent, so they share an entry and only cross-signature
# claims are refuted.
#
# The third slot is AROMATICITY, and it is the difference between a check that works and one
# that cannot fire. Without it (size, heteroatoms) makes benzene ~ cyclohexane, pyridine ~
# piperidine, furan ~ tetrahydrofuran and pyrrole ~ pyrrolidine the same signature, with three
# consequences:
#   * `group_words` -- the "groups and ring systems present" line the teacher builds every
#     MOLECULE block from -- would offer names contradicting the real aromaticity
#     ("cyclohexane" on an aromatic six-ring, "piperidine" on a pyridine). The line written to
#     PREVENT naming hallucination would induce it.
#   * a saturated ring whose only offered word is the aromatic one -- tetrahydrofuran mapped to
#     `furan` and nothing else, pyrrolidine to `pyrrole` -- would have no true name available.
#   * and this check would share the blindness, so it could never refute any of it.
# Saturated counterparts are named here too: a word this table lacks is a word `group_words`
# cannot offer and `ring_identity` cannot adjudicate, so leaving them out forces the aromatic
# name.
RING_SIG: dict[str, tuple[int, tuple[str, ...], bool]] = {
    # -- 3/4/5 carbocycles, saturated
    "cyclopropane": (3, (), False), "cyclobutane": (4, (), False),
    "cyclopentane": (5, (), False), "cycloheptane": (7, (), False),
    # -- five-membered heteroaromatics
    "furan": (5, ("O",), True),
    "thiophene": (5, ("S",), True),
    "pyrrole": (5, ("N",), True),
    "imidazole": (5, ("N", "N"), True), "pyrazole": (5, ("N", "N"), True),
    "thiazole": (5, ("N", "S"), True), "isothiazole": (5, ("N", "S"), True),
    "oxazole": (5, ("N", "O"), True), "isoxazole": (5, ("N", "O"), True),
    "oxadiazole": (5, ("N", "N", "O"), True), "thiadiazole": (5, ("N", "N", "S"), True),
    "triazole": (5, ("N", "N", "N"), True), "tetrazole": (5, ("N", "N", "N", "N"), True),
    # -- and their saturated counterparts
    "tetrahydrofuran": (5, ("O",), False), "oxolane": (5, ("O",), False),
    "thiolane": (5, ("S",), False),
    "pyrrolidine": (5, ("N",), False),
    "imidazolidine": (5, ("N", "N"), False), "pyrazolidine": (5, ("N", "N"), False),
    "oxazolidine": (5, ("N", "O"), False), "isoxazolidine": (5, ("N", "O"), False),
    "thiazolidine": (5, ("N", "S"), False),
    # -- six-membered, aromatic
    "benzene": (6, (), True), "phenyl": (6, (), True),
    "pyridine": (6, ("N",), True),
    "pyrimidine": (6, ("N", "N"), True), "pyrazine": (6, ("N", "N"), True),
    "pyridazine": (6, ("N", "N"), True),
    "triazine": (6, ("N", "N", "N"), True),
    # -- six-membered, saturated
    "cyclohexane": (6, (), False),
    "piperidine": (6, ("N",), False),
    "pyran": (6, ("O",), False), "tetrahydropyran": (6, ("O",), False),
    "dioxane": (6, ("O", "O"), False),
    "piperazine": (6, ("N", "N"), False),
    "morpholine": (6, ("N", "O"), False),
    "thiomorpholine": (6, ("N", "S"), False),
    # -- seven-membered; none of these is aromatic
    "oxepine": (7, ("O",), False), "oxepane": (7, ("O",), False),
    "azepine": (7, ("N",), False), "azepane": (7, ("N",), False),
}
# Words that assert a ring signature only in combination -- "purine" is a 6-5 fused pair with
# four N between them, not a 6-5 pair with two.
# The third slot counts how many of the TWO rings are fully aromatic (0, 1 or 2), for the same
# reason RING_SIG carries aromaticity: without it indole and indoline share a signature, as do
# chromene and chroman, and a benzene fused to a piperidine has no name but `quinoline`. One ring
# aromatic and one saturated is the commonest drug-like fused system there is.
FUSED_SIG: dict[str, tuple[tuple[int, int], tuple[str, ...], int]] = {
    "purine": ((6, 5), ("N", "N", "N", "N"), 2),
    "indole": ((6, 5), ("N",), 2),
    "indoline": ((6, 5), ("N",), 1),
    "indane": ((6, 5), (), 1), "indene": ((6, 5), (), 1),
    "benzimidazole": ((6, 5), ("N", "N"), 2),
    "benzofuran": ((6, 5), ("O",), 2),
    "dihydrobenzofuran": ((6, 5), ("O",), 1),
    "benzothiophene": ((6, 5), ("S",), 2),
    "benzothiazole": ((6, 5), ("N", "S"), 2),
    "benzoxazole": ((6, 5), ("N", "O"), 2),
    "thienothiophene": ((5, 5), ("S", "S"), 2),
    "thienopyridine": ((6, 5), ("N", "S"), 2),
    "quinoline": ((6, 6), ("N",), 2),
    "isoquinoline": ((6, 6), ("N",), 2),
    "tetrahydroquinoline": ((6, 6), ("N",), 1),
    "tetrahydroisoquinoline": ((6, 6), ("N",), 1),
    "quinazoline": ((6, 6), ("N", "N"), 2),
    "quinoxaline": ((6, 6), ("N", "N"), 2),
    "naphthalene": ((6, 6), (), 2),
    "tetralin": ((6, 6), (), 1),
    "chromene": ((6, 6), ("O",), 1),
    "chroman": ((6, 6), ("O",), 1),
    "benzodioxole": ((6, 5), ("O", "O"), 1),
    "benzoxepine": ((7, 6), ("O",), 1),
    "benzazepine": ((7, 6), ("N",), 1),
}

# --------------------------------------------------------------------- functional-group SMARTS
# Only groups drafts are known to confuse. Each entry is (SMARTS, what the word
# claims), and a word is refuted only when NO molecule on the turn matches it.
FG_SMARTS: dict[str, str] = {
    "lactone": "[#6;R][OX2;R][CX3;R](=O)",       # cyclic ester: the O and the C=O in one ring
    "lactam": "[NX3;R][CX3;R](=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    "anhydride": "[CX3](=O)[OX2][CX3]=O",
    "acid chloride": "[CX3](=O)[Cl]",
    "acyl chloride": "[CX3](=O)[Cl]",
    "chloroformate": "[Cl][CX3](=O)[OX2][#6]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitrile": "[NX1]#[CX2]",
    "amidine": "[NX2]=[CX3][NX3]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "oxime": "[CX3]=[NX2][OX2H1]",
    "azide": "[NX2]=[NX2+]=[NX1-]",
    "boronic acid": "[#6][BX3]([OX2H1])[OX2H1]",
    "phenol": "[OX2H1]c",
    "primary amine": "[NX3H2][#6]",
    "thiourea": "[NX3][CX3](=[SX1])[NX3]",
    "benzyl ether": "[CX4H2](c)[OX2][#6]",
    "morpholine": "C1COCCN1",
}

# ------------------------------------------------------------------------- reaction lexicon
# The names a draft may use, and the evidence name they have to be backed by. `named_reactions`
# returns Rxn-INSIGHT names; a draft writes English. The value is the substring that must
# appear (case-insensitively) in some evidence name for the claim to stand.
RXN_ALIAS: dict[str, str] = {
    "suzuki": "suzuki",
    "negishi": "negishi",
    "stille": "stille",
    "sonogashira": "sonogashira",
    "heck": "heck",
    "buchwald": "buchwald",
    "hartwig": "buchwald",
    "ullmann": "ullmann",
    "chan-lam": "chan",
    "grignard": "grignard",
    "wittig": "wittig",
    "horner": "horner",
    "friedel-crafts": "friedel",
    "friedel crafts": "friedel",
    "williamson": "williamson",
    "mitsunobu": "mitsunobu",
    "sandmeyer": "sandmeyer",
    "wohl-ziegler": "wohl",
    "appel": "appel",
    "reductive amination": "reductive amination",
    "esterification": "esterification",
    "fischer": "fischer",
    "saponification": "hydrolysis",
    "amidation": "amid",
    "hantzsch": "hantzsch",
    "biginelli": "biginelli",
    "knoevenagel": "knoevenagel",
    "aldol": "aldol",
    "michael": "michael",
    "mannich": "mannich",
    "curtius": "curtius",
    "hofmann": "hofmann",
    "staudinger": "staudinger",
    "mizoroki": "heck",
    "snar": "nucleophilic aromatic",
    "boc protection": "boc",
    "boc deprotection": "boc",
}
# Classes that cannot both describe one step. A draft that calls the same candidate a
# Friedel-Crafts acylation and then "the same cross-coupling" has contradicted itself, and no
# chemistry fact is needed to see it.
RXN_CLASS = {
    "cross-coupling": ["suzuki", "negishi", "stille", "sonogashira", "heck", "buchwald",
                       "ullmann", "cross-coupling", "cross coupling"],
    "friedel-crafts": ["friedel-crafts", "friedel crafts"],
    "acylation": ["acylation", "amidation", "esterification"],
    "alkylation": ["alkylation", "etherification", "williamson"],
    "reduction": ["reduction", "hydrogenation", "hydrogenolysis"],
    "oxidation": ["oxidation"],
    "cyclization": ["cyclization", "cyclisation", "ring closure", "ring-closing",
                    "annulation", "condensation"],
    "protection": ["protection", "deprotection", "saponification", "hydrolysis"],
}
EXCLUSIVE = [("cross-coupling", "friedel-crafts"), ("cross-coupling", "alkylation"),
             ("cross-coupling", "acylation"), ("friedel-crafts", "alkylation"),
             ("reduction", "oxidation"), ("acylation", "alkylation")]

# ------------------------------------------------------------------------------ score phrases
# "q points at it", "the model points hardest at it", "the highest confidence": all the same
# claim, that the candidate being taken is the argmax of q.
Q_APPEAL = re.compile(
    r"(what q points at|q points (?:at|to)|points hardest|highest (?:single[- ]step )?"
    r"(?:model )?confidence|the model points|q is highest|highest q\b)|the (?:single[- ]step )?model's own pick|the model's pick|the model believes(?: in)?|the model's own choice", re.I)
COMPLETE = re.compile(
    r"(every piece reaches purchasable|this variant is complete|the route is complete|"
    r"nothing left on the board can beat|nothing (?:else )?can beat (?:it|them))", re.I)
ONLY_CLEARS = re.compile(
    r"the only candidate that clears|only c(\d) (?:remains|clears)", re.I)


# ------------------------------------------------------------------------------- small helpers
def _num(tok: str) -> float | None:
    """`.977` and `0.977` and `977` from a board score. Returns None on anything else."""
    tok = tok.strip()
    try:
        return float("0" + tok) if tok.startswith(".") else float(tok)
    except ValueError:
        return None


def _rows(ev: dict) -> list[tuple[str, dict, dict]]:
    """(mid, menu candidate row, facts row) for every candidate this turn can reason about."""
    out = []
    facts = ev.get("facts") or {}
    for mid, mm in (ev.get("menus") or {}).items():
        fr = {r["c"]: r for r in ((facts.get(mid) or {}).get("candidates") or [])}
        for c in mm.get("candidates") or []:
            out.append((mid, c, fr.get(c["c"]) or {}))
    return out


def _ambiguous_cands(ev: dict) -> set[int]:
    """Candidate indices that sit on MORE THAN ONE molecule in this turn's evidence.

    `c0` is a different disconnection on every molecule -- the numbering is per-menu -- so on a
    turn whose evidence spans several of them, a bare `c0` in prose names nothing in particular
    and no check can decide what it claims. Collapsing the index anyway -- `max(p)` over every
    molecule's c0, or a plain dict where the LAST molecule's c0 overwrites the rest -- is nearly
    harmless on a rank turn, whose evidence usually carries one molecule, but a DONE turn
    carries several, and there `cutoff_class` and `rt_claim` would fire on drafts whose numbers
    are right for the molecule they are talking about.

    So an ambiguous index is left alone, on every kind of turn: it is the module's own rule that
    a check which cannot decide has to stay silent. A candidate written the unambiguous way,
    `mid·cN`, is not affected -- nothing here reads that form yet, and it is what a draft should
    use when the turn spans more than one molecule.
    """
    seen: dict[int, set[str]] = {}
    for mid, c, _ in _rows(ev):
        seen.setdefault(c["c"], set()).add(mid)
    return {ci for ci, mids in seen.items() if len(mids) > 1}


def _sig(ev: dict, key: str) -> list[float]:
    return [v for _, c, _ in _rows(ev)
            if (v := (c.get("signals") or {}).get(key)) is not None]


def _applied(ev: dict, actions=None) -> list[tuple[str, int]]:
    """(mid, candidate) the turn actually takes: the head of each rank, every done choice.

    The FIRST of a rank order is the one applied now; the rest are alternatives the ranking
    merely declares, so a claim about "the step" is a claim about the head. A `done` names one
    candidate per molecule and all of them are taken.

    `actions` is the turn's own action list (`load_episodes` parses it straight out of the
    harmony tool call, so this works on rows written before `_actions_detail` existed) and
    `ev["_actions_detail"]` is the fallback for a caller holding only the evidence.
    """
    src = actions if actions is not None else (ev.get("_actions_detail") or [])
    out = []
    for a in src or []:
        if not isinstance(a, dict):
            continue
        t = (a.get("type") or "").lower()
        if t == "rank" and a.get("order"):
            try:
                out.append((a.get("mid"), int(a["order"][0])))
            except (TypeError, ValueError, IndexError):
                pass
        elif t == "done":
            for mid, c in (a.get("choices") or {}).items():
                try:
                    out.append((mid, int(c)))
                except (TypeError, ValueError):
                    pass
    return [(m, c) for m, c in out if m]


def _mols_in_scope(ev: dict, ctx: dict) -> list[str]:
    """Every SMILES a draft may legitimately name a ring or a group in.

    NOT just this turn's molecules. A turn continues a branch and is asked to refer back to
    what it set aside -- "the route I built through the azide, the bromide and the Williamson
    ether" is the continuity the whole design is for -- so a group named in a backward
    reference is on the board's history even when it is absent from the current menu. Scoping
    the identity checks to the current turn alone would make every such sentence a violation
    (an azide banked several turns earlier, say).

    So the scope is the union of every molecule the EPISODE has shown up to and including this
    turn. That still refutes what these checks are for -- a thiazole in an episode whose only
    5-ring is a thiophene, a piperidine in an episode whose only N-ring is a pyrrolidine -- and
    it stops punishing a draft for remembering.
    """
    out = list(ctx.get("seen_smiles") or ())
    out.extend(_mols_on_turn(ev))
    return list(dict.fromkeys(out))


def _mols_on_turn(ev: dict) -> list[str]:
    """Every SMILES the turn legitimately talks about: open pieces and candidate fragments."""
    out = []
    for m in (ev.get("mols") or {}).values():
        if m.get("smiles"):
            out.append(m["smiles"])
    for b in ev.get("banked") or []:
        if b.get("smiles"):
            out.append(b["smiles"])
    for _, c, _ in _rows(ev):
        out.extend(c.get("reactants") or [])
    return list(dict.fromkeys(out))


def smiles_seen(evidences) -> list[str]:
    """Every SMILES a sequence of turn-evidences has shown, for `check(seen_smiles=...)`."""
    out: list[str] = []
    for ev in evidences or ():
        if isinstance(ev, dict):
            out.extend(_mols_on_turn(ev))
    return list(dict.fromkeys(out))


_MOL_CACHE: dict[str, object] = {}


def _mol(smi: str):
    if Chem is None:
        return None
    if smi not in _MOL_CACHE:
        try:
            _MOL_CACHE[smi] = Chem.MolFromSmiles(smi)
        except Exception:                                              # noqa: BLE001
            _MOL_CACHE[smi] = None
    return _MOL_CACHE[smi]


def _ring_sigs(smi: str) -> set[tuple[int, tuple[str, ...], bool]]:
    """(size, sorted heteroatoms, is fully aromatic) for every ring. See RING_SIG."""
    m = _mol(smi)
    if m is None:
        return set()
    out = set()
    for r in m.GetRingInfo().AtomRings():
        het = tuple(sorted(m.GetAtomWithIdx(i).GetSymbol() for i in r
                           if m.GetAtomWithIdx(i).GetSymbol() != "C"))
        arom = all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        out.add((len(r), het, arom))
    return out


def _fused_pairs(smi: str) -> list[tuple[tuple[int, int], tuple[str, ...], int]]:
    """((size_a, size_b), heteroatoms of the union, how many of the two are aromatic).

    One entry per pair of rings sharing >= 2 atoms. Sizes descending, so `(6, 5)` is the only
    spelling of a six-five and a caller writing a signature does not have to guess the order.
    The heteroatom multiset is over the UNION of the two rings, which is what a fused-system
    name asserts: "purine" claims four ring nitrogens across the pair, "thienothiophene" two
    sulfurs, "benzoxepine" one oxygen. The aromatic count separates indole from indoline and
    chromene from chroman -- see FUSED_SIG.
    """
    m = _mol(smi)
    if m is None:
        return []
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    arom = [all(m.GetAtomWithIdx(a).GetIsAromatic() for a in r) for r in rings]
    out = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) < 2:
                continue
            het = tuple(sorted(m.GetAtomWithIdx(a).GetSymbol()
                               for a in rings[i] | rings[j]
                               if m.GetAtomWithIdx(a).GetSymbol() != "C"))
            sizes = tuple(sorted((len(rings[i]), len(rings[j])), reverse=True))
            out.append((sizes, het, int(arom[i]) + int(arom[j])))
    return out


_SMARTS_CACHE: dict[str, object] = {}


def _has_fg(smi: str, smarts: str) -> bool | None:
    m = _mol(smi)
    if m is None or Chem is None:
        return None
    if smarts not in _SMARTS_CACHE:
        _SMARTS_CACHE[smarts] = Chem.MolFromSmarts(smarts)
    q = _SMARTS_CACHE[smarts]
    return None if q is None else m.HasSubstructMatch(q)


def _V(code: str, detail: str, fix: str = "") -> dict:
    return {"code": code, "detail": detail, "fix": fix}


_LEAVING_RE = re.compile(r"(?im)leaving open\s*:\s*(.+?)\s*$")
_TOGETHER_RE = re.compile(r"(?im)opening together\s*:\s*(.+?)\s*$")
_UNDER_RE = re.compile(r"\b([a-z]{1,2}[0-9]*)\s+under\s+(r[0-9]+)\b", re.I)

# Phrases that commit the clause to ONE of the two relations. Only an explicit commitment is
# checked -- a clause that says neither is left alone, because the ids and their `under` are
# already verified and a relation gate that guesses would refuse honest prose.
_SAME_STEP = re.compile(r"(?i)\b(same (?:step|reaction|disconnection|cut)|one (?:step|reaction|"
                        r"disconnection)|both pieces of|co-?precursors?|the step (?:owes|still "
                        r"owes)|pieces (?:that )?r[0-9]+ (?:still )?owes)\b")
_DIFF_STEP = re.compile(r"(?i)\b(different (?:steps?|reactions?|disconnections?|cuts?|branch(?:es)?)"
                        r"|two (?:different )?(?:disconnections?|cuts?|routes?|branches)"
                        r"|separate branch(?:es)?|rival|competing|alternative (?:cuts?|"
                        r"disconnections?|routes?))\b")


def _claimed_ids(clause: str, ev: dict) -> set[str]:
    """The board ids a clause names -- and nothing else.

    Two traps. The clause carries a reason after `|` and the reason is prose, so only the half
    before the pipe is read. And the board's id alphabet overlaps English: `as`, `at`, `an`,
    `be`, `by`, `we`, `us` and `up` are all live mids in this corpus, so a bare two-letter token
    is only taken as an id when the board actually has that molecule. A token that cannot be a
    word -- a letter followed by digits -- is taken as an id whether or not the board has it,
    which is what catches an invented one, except for `rN` and `cN`: those name a reaction and
    a candidate, and `wt under r1` names exactly one molecule.
    """
    head = (clause or "").split("|")[0]
    mols = (ev or {}).get("mols") or {}
    out = set()
    for tok in re.findall(r"\b([a-z]{1,2}[0-9]*)\b", head):
        if tok in mols:
            out.add(tok)
        elif re.fullmatch(r"[rc][0-9]+", tok):
            continue          # a reaction or candidate id -- `wt under r1` names one molecule
        elif re.fullmatch(r"[a-z]{1,2}[0-9]+", tok):
            out.add(tok)      # id-shaped and not on the board: an invented molecule
    return out


def open_choice(ev: dict, actions) -> tuple[list[str], list[str]]:
    """(openable, opened) at an open turn.

    Openable is a molecule on the board that cannot be bought and has no menu yet -- the board
    prints exactly those under OPEN with `no candidates yet`. Opened is what the call names.
    """
    mols = (ev or {}).get("mols") or {}
    openable = [m for m, d in mols.items()
                if not d.get("buyable") and not d.get("has_menu")]
    opened = [a.get("mid") for a in (actions or [])
              if isinstance(a, dict) and (a.get("type") or "").lower() == "open" and a.get("mid")]
    return sorted(openable), [m for m in opened if m]


_QTY = {"both": 2, "the two": 2, "all two": 2, "two": 2,
        "all three": 3, "the three": 3, "three": 3,
        "all four": 4, "the four": 4, "four": 4,
        "all five": 5, "the five": 5, "five": 5}
_QTY_RE = re.compile(r"(?i)\b(both|all (?:two|three|four|five)|the (?:two|three|four|five))\b")


def _argument_of(clause: str) -> str:
    """A clause's reason half with its particulars removed, for comparing ARGUMENTS.

    Board ids, reaction ids and counts are stripped, so `two rival branches, one per parent`
    and `three rival branches, one per parent` normalise to the same string. That is the point:
    the ids differing is not the argument differing, and a clause whose only new content is a
    number is the same sentence written again.
    """
    tail = (clause or "").split("|", 1)
    tail = (tail[1] if len(tail) > 1 else tail[0]).lower()
    # The ledger state survives normalisation, and it is the ONE particular that does. `1 of 2
    # closed` and `0 of 1 closed` are different situations -- a two-piece step with one piece
    # already bought against a step that owes a single piece -- and two clauses that cite
    # different states are giving different reasons, not the same reason with a number swapped.
    tail = re.sub(r"\b([0-9]+) of ([0-9]+) closed\b", r" led\1_\2 ", tail)
    tail = re.sub(r"\b[rc][0-9]+\b", " ", tail)
    tail = re.sub(r"\b(?:both|all|the)?\s*(?:one|two|three|four|five|six|seven|eight)\b",
                  " ", tail)
    tail = re.sub(r"\b[a-z]{1,2}[0-9]+\b", " ", tail)
    tail = re.sub(r"[^a-z0-9_ ]+", " ", tail)
    return re.sub(r"\s+", " ", tail).strip()


def _c_open_named(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A reaction name at an OPEN turn has to be one the episode already established.

    The open brief forbids naming chemistry, and for a good reason: the OPEN block prints
    purchasability, depth and parentage, so a ring or a functional group written there was read
    off a SMILES string by eye. A REACTION name is the one exception worth allowing, because a
    ledger step can honestly be referred to by the name the turn that ranked it already earned
    from the evidence -- `the N-arylation branch` for a step whose menu row said
    `Buchwald-Hartwig N-arylation`. That is memory, and it is better writing than `r6`.

    What it must not be is invention. Most reaction names at an open turn appear in an earlier
    turn of the same episode; the rest -- `the reductive amination closed eq`, `the Suzuki I
    took on qb` -- name a step the episode never called that. Rare, and exactly the failure
    this corpus cannot afford, since a student learns that an open turn is a place to assert
    chemistry it cannot see.

    The rule is therefore: cite, do not coin. A name is earned here if an earlier turn of this
    episode used it, or if it appears in this turn's own evidence.
    """
    if ctx.get("kind") != "open":
        return []
    low = (text or "").lower()
    said = {k for k in RXN_ALIAS if k in low}
    if not said:
        return []
    prior = " ".join(t or "" for t in (ctx.get("prev_thoughts") or [])).lower()
    # The turn's own evidence counts too: a name printed on this board is not a coinage.
    for mid, f in ((ev or {}).get("facts") or {}).items():
        for cd in (f or {}).get("candidates") or []:
            for nm in (cd.get("names") or []) or ([cd.get("named")] if cd.get("named") else []):
                prior += " " + str(nm).lower()
    out = []
    for k in sorted(said):
        alias = RXN_ALIAS[k]
        if k in prior or alias in prior:
            continue
        out.append(_V("open_named_uncited",
                      f"calls a step a {k!r} at an open turn; no earlier turn of this episode "
                      f"and nothing on this board names it that",
                      "At an open turn a reaction may only be called what the episode has "
                      "already called it. Use the board id."))
    return out


def _c_open_together(text: str, ev: dict, ctx: dict) -> list[dict]:
    """`opening together:` -- required wherever the call opens more than one molecule.

    WHY this is a decision and not bookkeeping. Opening several pieces at once is the move that
    makes the search breadth-first, and it is the move the corpus exists to teach: the rank turn
    that follows a multi-open usually compares two or more molecules, and the one that follows
    a single open rarely does. So the batch is what puts rival disconnections in front of the
    ranker at the same depth (multi-route expansion) -- it is where the
    `diverging from` slot's material comes from. A turn that opens three pieces and says nothing
    about why they belong together teaches the student to batch by habit.

    WHAT IT MAY SAY. Only what the board prints for an open piece: its parent reaction (`under
    rN`), its depth, and how many pieces that reaction still owes. Most multi-opens are pieces
    under DIFFERENT reactions -- rival cuts of the same parent, held at one depth instead of
    committing to one branch -- and the rest are pieces the SAME reaction owes, where the step
    is a route only if every one of them terminates. Those are the
    two chemically real reasons to batch, they are distinguishable from the board alone, and the
    clause has to name which one it is by citing each piece's reaction.
    """
    if ctx.get("kind") != "open":
        return []
    mols = (ev or {}).get("mols") or {}
    _openable, opened = open_choice(ev, ctx.get("actions"))
    m = _TOGETHER_RE.search(text or "")
    if len(opened) < 2:
        if m:
            return [_V("open_together_unearned",
                       f"has an `opening together` line while the call opens "
                       f"{len(opened)} molecule(s)",
                       "Say it only where two or more are opened in one call.")]
        return []
    if not m:
        return [_V("open_together_missing",
                   f"opens {', '.join(opened)} in one call without saying why they belong "
                   f"together",
                   "`opening together: <mid> under <rN>, ... | <why these, now>`")]

    clause = m.group(1)
    out = []
    said = _claimed_ids(clause, ev)
    for mid in sorted(said - set(opened)):
        out.append(_V("open_together_wrong",
                      f"lists {mid} among the molecules being opened together; the call does "
                      f"not open it",
                      "Name the molecules this call opens."))
    missing = [x for x in opened if x not in said]
    if missing:
        out.append(_V("open_together_incomplete",
                      f"opens {', '.join(opened)} but the line names only "
                      f"{', '.join(sorted(said)) or 'none of them'}",
                      "Name every molecule the call opens."))
    for mid, rid in _UNDER_RE.findall(clause):
        if mid in mols:
            real = mols[mid].get("under")
            if real and rid.lower() != str(real).lower():
                out.append(_V("open_together_under_wrong",
                              f"puts {mid} under {rid.lower()}; the board has it under {real}",
                              f"{mid} is under {real}."))
    # A quantity word is a claim about how many pieces the call opens, and it can go wrong:
    # `the two cuts a5 offered ... so opening all three`. Cheap to check.
    for q in _QTY_RE.findall(clause):
        want = _QTY.get(q.lower())
        if want is not None and want != len(opened):
            out.append(_V("open_together_count",
                          f"says {q!r} where the call opens {len(opened)}",
                          f"The call opens {len(opened)} pieces."))
            break

    # The reason half has to earn its place. Left alone, the clause becomes the same sentence
    # with a number swapped -- "two rival branches, one per parent, so opening both keeps each
    # disconnection alive at one depth", turn after turn. Each is true, and none tells the
    # turn's reader anything the previous one had not. The comparison strips ids and counts on purpose: a new number is
    # not a new argument, and what the slot is for is the reason THESE pieces go together NOW
    # -- which reaction offered them, how many pieces each still owes, at what depth.
    arg = _argument_of(clause)
    if len(arg) > 30:
        for prev in (ctx.get("prev_thoughts") or []):
            pm = _TOGETHER_RE.search(prev or "")
            if not pm:
                continue
            if difflib.SequenceMatcher(None, arg, _argument_of(pm.group(1))).ratio() > 0.80:
                out.append(_V("open_together_recycled",
                              "gives the same reason for the batch as an earlier open turn, "
                              "with only the ids and counts changed",
                              "Say what makes THIS batch the right one: which reaction offered "
                              "these cuts, how many pieces each still owes, at what depth."))
                break

    unders = {mols.get(x, {}).get("under") for x in opened if x in mols}
    unders.discard(None)
    if len(unders) > 1 and _SAME_STEP.search(clause) and not _DIFF_STEP.search(clause):
        out.append(_V("open_together_relation",
                      f"calls them pieces of one step; they sit under "
                      f"{', '.join(sorted(unders))} -- different reactions",
                      "These are rival cuts held at one depth, not co-precursors."))
    elif len(unders) == 1 and _DIFF_STEP.search(clause) and not _SAME_STEP.search(clause):
        out.append(_V("open_together_relation",
                      f"calls them different steps; every one of them sits under "
                      f"{next(iter(unders))}",
                      "These are pieces the same reaction owes."))
    return out


def _c_open_choice(text: str, ev: dict, ctx: dict) -> list[dict]:
    """`leaving open:` -- required where the call NARROWS, refused where it does not.

    On most open turns of the breadth-first corpus the call opens everything openable; there is
    nothing left and a sentence about a choice would be narration, so it is refused there. Where
    the call DOES leave something, that is a decision about where the search does not go, and
    without the clause it is invisible.

    Under the serial board this choice hardly exists: an open turn almost always has a single
    openable molecule. Breadth-first replay is what creates the choice.
    """
    if ctx.get("kind") != "open":
        return []
    openable, opened = open_choice(ev, ctx.get("actions"))
    left = [m for m in openable if m not in opened]
    m = _LEAVING_RE.search(text or "")
    said = _claimed_ids(m.group(1), ev) if m else set()
    out = []
    if not left:
        if m:
            out.append(_V("open_choice_unearned",
                          f"says it is leaving {m.group(1)[:40]!r} open; the call opens every "
                          f"molecule that could be opened",
                          "Say it only where something is being left."))
        return out
    if not m:
        out.append(_V("open_choice_missing",
                      f"opens {', '.join(opened) or 'nothing'} and leaves "
                      f"{', '.join(left)} open without saying so",
                      "`leaving open: <mid> | <why not now>`"))
        return out
    for mid in sorted(said - set(left)):
        if mid in opened:
            out.append(_V("open_choice_wrong",
                          f"says it leaves {mid} open; the call opens it",
                          "Name the ones it does not open."))
        elif mid not in openable:
            out.append(_V("open_choice_wrong",
                          f"says it leaves {mid} open; {mid} could not have been opened "
                          f"-- it is bought, or its menu is already on screen",
                          "Name a molecule that was openable and was not opened."))
    return out


# ============================================================================== the checks
# Each takes (text, ev, ctx) and returns a list of violations. Registered below; a check that
# raises is reported as a check failure and never as a violation of the draft.

def _c_numbers(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Every board number quoted must be a number this turn actually carries.

    The board writes scores without a leading zero (`q.374`), so a bare decimal in a draft is
    a score; both spellings are accepted here because teachers write `p 0.977` as often as
    `p.977`. A cited value is refuted only when the turn holds no candidate within 5e-4 of it,
    which is the rounding `evidence_for` applies.
    """
    out = []
    for key in ("q", "p"):
        have = _sig(ev, key)
        if not have:
            continue
        for m in re.finditer(rf"\b{key}\s*=?\s*(0?\.\d+|1\.000|1\.0\b)", text):
            v = _num(m.group(1))
            if v is None:
                continue
            if not any(abs(v - h) <= 5e-4 for h in have):
                near = min(have, key=lambda h: abs(h - v))
                out.append(_V(f"{key}_value",
                              f"cites {key} {m.group(1)}, which no candidate on this turn "
                              f"has (nearest is {near:.3f})",
                              f"quote {key} exactly as this turn carries it"))
    # Prices are compared as the STRINGS the board printed, not as floats. The board
    # rounds to two significant figures ($1.2k, $120k), so a numeric tolerance would either
    # accept a value no line carries or reject the line's own rounding; the model is being
    # asked to quote what is on screen, and that is a string.
    shown = set()
    for _, c, _ in _rows(ev):
        lp = [p for p in (c.get("ln_price") or []) if p is not None]
        shown.update(R.dollars(p) for p in lp)
        # the candidate-level total, printed whenever every piece is priced
        if lp and len(lp) == len(c.get("ln_price") or []):
            shown.add(R.dollars(math.log(sum(math.exp(x) for x in lp))))
    for b in (ev.get("banked") or []):
        if b.get("ln_price") is not None:
            shown.add(R.dollars(b["ln_price"]))
    for m in (ev.get("mols") or {}).values():
        if m.get("ln_price") is not None:
            shown.add(R.dollars(m["ln_price"]))
    for r in (ev.get("routes") or []):
        if r.get("cost_usd") is not None:
            shown.add(f"${r['cost_usd']:,.0f}")
    # `decided_by` says what actually separated the leaders. A draft handed the
    # "ALSO THE CHEAPEST" note sometimes promotes it to the axis, which is a different and
    # false claim: the cut being cheapest does not mean cheapness is what the ranking is on,
    # and a corpus that says so teaches the student to read a coincidence as a reason.
    dec = (ev.get("decided_by") or {})
    if dec and re.search(r"ordered by:\s*price", text):
        axes = {d.get("axis") for d in dec.values() if isinstance(d, dict)}
        if axes and "price" not in axes:
            out.append(_V("price_axis",
                          "orders by price on a turn the search did not decide on price "
                          f"(it separated on {', '.join(sorted(a for a in axes if a))})",
                          "name the axis that decided; a cut being cheapest belongs in the "
                          "`why`, not in `ordered by:`"))

    # A price axis whose `value:` is a bare number cites nothing: the board prints prices
    # only as dollar strings. A glossary that describes price as a log while the board prints
    # `$18` yields `ordered by: price | value: 2.91`, a number on no screen. The dollar-token
    # check below cannot see it, because there is no dollar token.
    for m in re.finditer(r"ordered by:\s*price\s*\|\s*value:\s*([^|\n]+)", text):
        val = m.group(1).strip()
        if val.lower() not in ("none", "-", "") and "$" not in val:
            out.append(_V("price_value",
                          f"orders by price with `value: {val[:24]}`, which is not a price "
                          f"the board prints",
                          "quote the dollar string from the line -- a fragment's ($X) or "
                          "the cut's own total; the board carries no log"))
    if shown:
        # commas are part of the ROUTES line's own format ($4,917); without them the
        # pattern stops at the comma and rejects the route cost as "$4"
        for m in re.finditer(r"\$\s?[0-9][0-9,.]*[kM]?\+?", text):
            tok = m.group(0).replace(" ", "").rstrip(".")
            if tok not in shown:
                out.append(_V("price_value",
                              f"cites {tok}, which is not a price on this turn",
                              "quote a price exactly as the board prints it -- a piece's "
                              "own $, a candidate's total, or a finished route's cost"))
    return out


def _c_q_argmax(text: str, ev: dict, ctx: dict) -> list[dict]:
    """If the draft argues from q, the candidate it takes has to be q's own choice.

    The developer block states this as a rule rather than a preference -- "if you name it as
    your reason, the candidate you take has to be the one it points at" -- and it is a
    frequently violated claim.
    """
    if not Q_APPEAL.search(text):
        return []
    out = []
    for mid, c in _applied(ev, ctx.get("actions")):
        rows = [(r["c"], (r.get("signals") or {}).get("q"))
                for r in ((ev.get("menus") or {}).get(mid) or {}).get("candidates") or []]
        rows = [(i, q) for i, q in rows if q is not None]
        if len(rows) < 2:
            continue
        top, tq = max(rows, key=lambda x: x[1])
        mine = dict(rows).get(c)
        if mine is None or top == c or abs(mine - tq) <= 1e-9:
            continue
        out.append(_V("q_argmax",
                      f"argues from q but takes {mid}·c{c} (q {mine:.3f}) while q is highest "
                      f"at c{top} ({tq:.3f})",
                      f"either take c{top} or argue from p, rt or price instead of q"))
    return out


def _c_cutoff(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Cut-off claims about named candidates, and about the candidate being taken.

    Three distinct errors, all decidable: calling an above-cut-off
    candidate below it, asserting only one candidate clears when six do, and taking a
    below-cut-off candidate in a paragraph that says below-cut-off cannot carry a route.
    """
    out = []
    pmap = {(mid, c["c"]): (c.get("signals") or {}).get("p") for mid, c, _ in _rows(ev)}
    fuzzy = _ambiguous_cands(ev)          # see _ambiguous_cands: a bare cN naming two menus
    byc: dict[int, list[float]] = {}
    for (_, ci), p in pmap.items():
        if p is not None and ci not in fuzzy:
            byc.setdefault(ci, []).append(p)

    # "(c2, c4, c5)" following a below/above-CUT-OFF phrase. The cut-off noun is REQUIRED and
    # the span may not cross a line, because without either the sense word alone was enough:
    # `its second precursor is dj, which sits above this piece` -- the wording the cycle-prune
    # reason naturally takes -- matched `above`, then reached across the newline into the NEXT
    # `ruled out` line and captured its candidates, reporting "has c0 clearing the cut-off" for
    # a line that said `c0, c3 | below cut-off | value: 0.014, 0.033` with both numbers right.
    # The clause-level pass below is the careful one and keeps its own reach; this list form only
    # ever meant to catch "below the cut-off (c2, c4, c5)", and now that is all it catches.
    for m in re.finditer(r"(below|above|clears?|under|beneath)\s+(?:the\s+)?"
                         r"(?:filter|cut-?off|plausibility|bar)[^.;\n]{0,40}?"
                         r"((?:c\d\s*(?:,|and|or|\s)+)*c\d)", text, re.I):
        sense = m.group(1).lower()
        want_below = sense in ("below", "under", "beneath")
        for ci in {int(x) for x in re.findall(r"c(\d)", m.group(2))}:
            ps = byc.get(ci) or []
            if not ps:
                continue
            p = max(ps)
            if want_below and p >= CUTOFF:
                out.append(_V("cutoff_class",
                              f"puts c{ci} below the {CUTOFF} cut-off; its p is {p:.3f}",
                              f"c{ci} clears the cut-off"))
            if (not want_below) and p < CUTOFF:
                out.append(_V("cutoff_class",
                              f"has c{ci} clearing the cut-off; its p is {p:.3f}",
                              f"c{ci} is below the cut-off"))

    # The list form above only reads a candidate that FOLLOWS the marker -- "below the cut-off
    # (c2, c4, c5)". A draft writes it the other way round at least as often: "c0 is the
    # acylation and it clears the plausibility cut-off comfortably", where the number is the
    # subject and the marker is the predicate. Attributing by clause catches both orders, and
    # a clause with one candidate is the same unambiguous unit the action markers use.
    for cl in _clauses(text):
        cs = _cands_in(cl)
        if len(cs) != 1:
            continue
        ci = cs[0]
        ps = byc.get(ci) or []
        if not ps:
            continue
        pv = max(ps)
        if re.search(r"\b(clears?|above|passes|goes|is over)\b[^.;]{0,40}?"
                     r"(?:the )?(?:filter|cut-?off|plausibilit|bar)", cl, re.I) and pv < CUTOFF:
            out.append(_V("cutoff_class",
                          f"has c{ci} clearing the cut-off; its p is {pv:.3f}, below {CUTOFF} "
                          f"-- the filter says the step does not go",
                          f"c{ci} is below the cut-off"))
        elif re.search(r"\b(below|under|beneath|fails?|does not (?:clear|pass|go))\b"
                       r"[^.;]{0,40}?(?:the )?(?:filter|cut-?off|plausibilit|bar)", cl,
                       re.I) and pv >= CUTOFF:
            out.append(_V("cutoff_class",
                          f"puts c{ci} below the cut-off; its p is {pv:.3f}",
                          f"c{ci} clears the cut-off"))

    if ONLY_CLEARS.search(text):
        n = sum(1 for ps in byc.values() if max(ps) >= CUTOFF)
        if n > 1:
            out.append(_V("only_clears",
                          f"says one candidate clears the cut-off; {n} of them do",
                          "count them, or drop the word 'only'"))

    # `applied_subcutoff` asks a RANK turn not to take a step the filter says does not go. A
    # DONE turn takes nothing: it claims a route whose every step was chosen on an earlier turn,
    # and it cannot choose otherwise -- the choice set is what makes the route that route. So on
    # a done turn this would fire on a p the draft has no say in and NO text could satisfy it,
    # dropping most done drafts. Cited, not silenced -- the sub-cut-off step is worth SAYING on a done turn, and the brief
    # asks for the weakest step by name; it is just not a defect in the sentence.
    if ctx.get("kind") != "done":
        for mid, c in _applied(ev, ctx.get("actions")):
            p = pmap.get((mid, c))
            if p is None or p >= CUTOFF:
                continue
            out.append(_V("applied_subcutoff",
                          f"takes {mid}·c{c} at p {p:.3f}, below the {CUTOFF} cut-off",
                          "the filter says this step does not go: say so, or take another"))
    return out


_DIVERGING_ON = re.compile(r"(?im)^(\s*diverging from\s*:[^|\n]*\|)([^|\n]*)(\|?)")


def _mask_diverging(text: str) -> str:
    """Blank the middle slot of every `diverging from` line.

    That slot names the reaction ANOTHER route already expanded, on another molecule, in an
    earlier turn. The name checks below judge names against THIS turn's candidates, so they
    refuse it for not being earned here -- and it never could be. The claim is not unchecked:
    `_c_deciding`'s `diverging_class_wrong` holds it against the board's own reaction ledger,
    which is the only place that knows what that step was. Masking it here keeps each claim
    with the checker that can actually adjudicate it.
    """
    return _DIVERGING_ON.sub(lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3), text)


def _c_named(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A named reaction has to be one the template corpus reproduced on THESE precursors.

    `ev["facts"][mid]["candidates"][i]["named_tier"]` is the verdict from running the SMIRKS
    forward: `applies+makes` means the name is earned, `applies` means the groups are right
    and the outcome is not, `none` means no template even applies. The brief already says to
    say nothing when nothing reproduces the transformation; this refuses the drafts that say
    a name anyway.
    """
    # Only the candidates this turn's claim can be ABOUT: the ones it takes, plus any it names
    # by number. `ev["facts"]` caps at six rows of a ten-wide menu, so judging a name against
    # every row would refute a claim about c7 with the silence of a row that was never
    # computed -- a false rejection, which costs a draw and teaches nothing.
    text = _mask_diverging(text)
    want = {c for _, c in _applied(ev, ctx.get("actions"))}
    want |= {int(x) for x in re.findall(r"\bc(\d)\b", text)}
    earned, applies_only, judged = set(), set(), False
    for _, cand, f in _rows(ev):
        if cand["c"] not in want or not f.get("named_tier"):
            continue
        judged = True
        names = [n.lower() for n in (f.get("named") or [])]
        if f["named_tier"] == "applies+makes":
            earned.update(names)
        elif f["named_tier"] == "applies":
            applies_only.update(names)
    if not judged:
        return []                                       # no verdict was measured: stay silent
    applied_only = applies_only
    out = []
    # The WORD counts as well as its key. The table maps a colloquial name to a canonical
    # fragment of the corpus's label -- "hartwig" to "buchwald", "mizoroki" to "heck" -- and for
    # some entries the word is not inside its own key, so matching on the key alone would
    # refuse a draft quoting the corpus's OWN earned label verbatim. `saponification` maps to
    # `hydrolysis`, and the label the corpus actually earns is "Ester saponification (alkyl
    # deprotection)". The two conditions cover each other -- "snar" appears in no label and is
    # caught by its key, "saponification" matches no key and is caught by itself -- and a
    # genuinely invented name satisfies neither, so it still fires.
    def _has(name_set, word, key):
        return any(key in n or word in n for n in name_set)

    for word, key in RXN_ALIAS.items():
        if not re.search(rf"\b{re.escape(word)}\b", text, re.I):
            continue
        if _has(earned, word, key):
            continue
        if _has(applied_only, word, key):
            out.append(_V("named_applies_only",
                          f"calls this a {word}; that template applies to these precursors "
                          f"but does not reproduce the product",
                          "the functional groups are right and the outcome is not -- do not "
                          "give it the name"))
        else:
            have = sorted(earned)[:3]
            out.append(_V("named_unearned",
                          f"calls this a {word}; nothing on this turn reproduces that "
                          + (f"(earned here: {', '.join(have)})" if have
                             else "(nothing reproduces the transformation)"),
                          "name it only when a template makes the product, else describe the "
                          "bond"))
    return out


def _c_named_conflict(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Two reaction classes that cannot both describe THE SAME step, in one sentence.

    Per sentence, not per draft. A rank turn weighs several candidates and they are routinely
    of different classes -- "c0 is an acylation, c3 an N-alkylation" is a correct sentence pair
    and a draft-level check would call it a contradiction on every comparison turn. The defect
    is narrower and stays caught: one clause calling a step a Friedel-Crafts
    acylation and the next calling it "the same cross-coupling".
    """
    out = []
    # Same masking as `_c_named`, and for a sharper reason: a `diverging from` line names the
    # class of the step already expanded AND the class of this cut, in one line, on purpose --
    # that contrast IS the slot's content. Read unmasked, every correct one of them looks like
    # a step called two classes at once.
    for sent in re.split(r"(?<=[.;!?])\s+", _mask_diverging(text)):
        seen = {cls for cls, words in RXN_CLASS.items()
                if any(re.search(rf"\b{re.escape(w)}\b", sent, re.I) for w in words)}
        for a, b in EXCLUSIVE:
            if a in seen and b in seen:
                out.append(_V("named_conflict",
                              f"calls one step both a {a} and a {b}: {sent.strip()[:110]!r}",
                              "one step is one class: pick the one the bond change supports"))
    return out


def _c_ring_identity(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A ring-class word must match a ring actually present on this turn.

    Signature is (size, sorted ring heteroatoms) -- all a name of this kind asserts, and
    enough to refute thiazole-for-thiophene, oxadiazole-for-isoxazole,
    triazine-for-imidazole and purine-for-imidazopyridine.
    """
    if Chem is None:
        return []
    mols = _mols_in_scope(ev, ctx)
    if not mols:
        return []
    have = set()
    for s in mols:
        have |= _ring_sigs(s)
    if not have:
        return []
    out = []
    for word, sig in RING_SIG.items():
        if not re.search(rf"\b{re.escape(word)}s?\b", text, re.I):
            continue
        if sig in have:
            continue
        size, het, want_arom = sig
        # What the same-size rings on this turn actually are, aromaticity included: the whole
        # point of the third slot is that "no such ring" is often "the ring is there and it is
        # saturated", and naming the difference is what makes the correction actionable.
        same_size = sorted({("aromatic " if a else "saturated ") + ("".join(h) or "all-C")
                            for n, h, a in have if n == size})
        out.append(_V("ring_identity",
                      f"names a {word} ({size}-ring, heteroatoms {''.join(het) or 'none'}, "
                      f"{'aromatic' if want_arom else 'not aromatic'}); no such ring is on "
                      f"this turn"
                      + (f" -- the {size}-rings here are {', '.join(same_size)}"
                         if same_size else ""),
                      "read the ring off the SMILES before naming it"))
    fused = {p for s in mols for p in _fused_pairs(s)}
    for word, (sizes, het, want_n) in FUSED_SIG.items():
        if not re.search(rf"\b{re.escape(word)}", text, re.I):
            continue
        if (sizes, het, want_n) in fused:
            continue
        near = sorted({("".join(h) or "all-C") + f", {n} aromatic"
                       for sz, h, n in fused if sz == sizes})
        out.append(_V("ring_identity",
                      f"names a {word} ({sizes[0]}-{sizes[1]} fused, heteroatoms "
                      f"{''.join(het) or 'none'}, {want_n} of the two aromatic); this turn's "
                      f"{sizes[0]}-{sizes[1]} systems "
                      + (f"carry {', '.join(near)}" if near else "include no such pair"),
                      "read the fused system off the SMILES before naming it"))
    return out


def _c_functional_group(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A functional-group word must match a group present on this turn.

    Catches the lactone that is a ketone with a ring ether, the acetyl chloride that is a
    chloroformate, the anhydride that is a ketoester, the guanidine that is an amidine.
    """
    if Chem is None:
        return []
    mols = _mols_in_scope(ev, ctx)
    if not mols:
        return []
    out = []
    for word, smarts in FG_SMARTS.items():
        if not re.search(rf"\b{re.escape(word)}s?\b", text, re.I):
            continue
        hits = [_has_fg(s, smarts) for s in mols]
        if any(h is True for h in hits) or all(h is None for h in hits):
            continue
        out.append(_V("fg_identity",
                      f"names a {word}; no molecule on this turn contains one",
                      "describe the group the SMILES actually shows"))
    return out


def _c_purchasable(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A purchasability claim against the prices the turn carries.

    Two shapes of the claim. The WEAK one names the pieces purchasable when none of them is:
    an unpriced fragment called "a single, purchasable X" that the board opens as a new
    problem on the next line. The STRONG one
    says EVERY piece is -- "its two precursors are both cheap and purchasable, so the AND
    closes on arrival" -- while one of the two carries no price, which is the claim that turns
    an unsolved branch into a finished route in the reader's head. A price is carried on
    purchasable fragments only, so its absence is the fact that refutes both.
    """
    # Not on a DONE turn. There, a purchasability claim is about the route's LEAVES, and the
    # board has already proved it: `_do_done` walks the choice set down and refuses the claim
    # outright if any piece it ends on is unpurchasable, so a done turn that exists at all has
    # purchasable leaves. What this check compares against instead is the immediate precursors
    # of the chosen candidates -- and a multi-step route's interior steps have unpriced
    # precursors by definition, because those are the pieces it MAKES. So it would refute the
    # one sentence a done turn is for ("every leaf is purchasable, so I commit").
    if ctx.get("kind") == "done":
        return []
    weak = re.search(r"\b(purchasable|buyable|commercial|off the shelf|already banked)\b",
                     text, re.I)
    strong = re.search(r"(both .{0,40}(?:purchasable|buyable|cheap)"
                       r"|all (?:of them |the pieces |fragments )?(?:are )?"
                       r"(?:purchasable|buyable)"
                       r"|every (?:piece|fragment|leaf) (?:is |reaches )"
                       r"(?:purchasable|buyable|purchasable material)"
                       r"|the AND closes on arrival"
                       r"|closes? on arrival)", text, re.I)
    if not (weak or strong):
        return []
    out = []
    for mid, c in _applied(ev, ctx.get("actions")):
        row = next((r for r in ((ev.get("menus") or {}).get(mid) or {}).get("candidates") or []
                    if r["c"] == c), None)
        if row is None:
            continue
        buy = row.get("buyable") or []
        prices = row.get("ln_price") or []
        if not buy:
            continue
        if not any(buy):
            out.append(_V("purchasable",
                          f"calls the pieces purchasable; no fragment of {mid}·c{c} carries a "
                          f"price",
                          "an unpriced fragment still has to be made"))
        elif strong and not all(buy):
            missing = [i for i, b in enumerate(buy) if not b]
            frags = row.get("reactants") or []
            names = ", ".join((frags[i][:34] if i < len(frags) else f"fragment {i}")
                              for i in missing)
            have = [f"ln${x:.2f}" for x in prices if x is not None]
            out.append(_V("purchasable",
                          f"has every piece of {mid}·c{c} purchasable; {len(missing)} of "
                          f"{len(buy)} carries no price ({names}) -- only {', '.join(have)} "
                          f"is banked",
                          "the AND does not close: that fragment is the next open piece"))
    return out


def _c_shape(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Skeleton and ring-count claims against `split_shape`.

    `scaffold_kept` is the computed answer to "does this build the skeleton or decorate it",
    and `rings A <- B` is handed to the teacher verbatim by `_shape_phrase`; a draft that
    writes "rings 4 <- 4" and "changes the skeleton" in one sentence has contradicted the
    number it just quoted.
    """
    out = []
    for mid, c in _applied(ev, ctx.get("actions")):
        f = next((r for r in ((ev.get("facts") or {}).get(mid) or {}).get("candidates") or []
                  if r["c"] == c), None)
        sh = (f or {}).get("shape") or {}
        kept = sh.get("scaffold_kept")
        if kept is None:
            continue
        says_break = re.search(r"\b(opens? the (?:\w+ )?ring|changes? the skeleton|"
                               r"breaks? the (?:ring|skeleton)|builds? the (?:ring|skeleton)|"
                               r"ring[- ]forming|ring[- ]opening)\b", text, re.I)
        says_keep = re.search(r"\b(keeps? (?:the |all )?(?:\w+ )?(?:rings?|skeleton) intact|"
                              r"skeleton (?:is )?kept|leaves the skeleton|"
                              r"without touching the (?:\w+ )?(?:ring|skeleton|framework))\b",
                              text, re.I)
        if kept and says_break and not says_keep:
            out.append(_V("shape",
                          f"has {mid}·c{c} building or opening a ring; the skeleton is kept "
                          f"(rings {sh.get('rings_product')} <- {sh.get('rings_precursors')})",
                          "this step decorates a skeleton that already exists"))
        if (not kept) and says_keep and not says_break:
            out.append(_V("shape",
                          f"has {mid}·c{c} keeping the skeleton; it changes "
                          f"(rings {sh.get('rings_product')} <- {sh.get('rings_precursors')})",
                          "this step builds or breaks the skeleton"))
        for m in re.finditer(r"rings?\s+(\d+)\s*(?:<-|←|from)\s*(\d+)", text):
            a, b = int(m.group(1)), int(m.group(2))
            rp, rr = sh.get("rings_product"), sh.get("rings_precursors")
            if rp is not None and rr is not None and (a, b) != (rp, rr):
                out.append(_V("shape",
                              f"writes rings {a} <- {b}; this candidate is {rp} <- {rr}",
                              f"rings {rp} <- {rr}"))
    return out


def _c_counts(text: str, ev: dict, ctx: dict) -> list[dict]:
    """"N-atom", "N heavy atoms", "C1/C2 unit" against the descriptors and the SMILES."""
    if Chem is None:
        return []
    heavies = set()
    for m in (ev.get("mols") or {}).values():
        d = m.get("desc") or {}
        if d.get("n_heavy"):
            heavies.add(int(d["n_heavy"]))
    for _, _, f in _rows(ev):
        sh = f.get("shape") or {}
        for k in ("heavy_product",):
            if sh.get(k):
                heavies.add(int(sh[k]))
        for n in sh.get("fragments") or []:
            heavies.add(int(n))
    for s in _mols_in_scope(ev, ctx):
        mm = _mol(s)
        if mm is not None:
            heavies.add(mm.GetNumHeavyAtoms())
    out = []
    if heavies:
        for m in re.finditer(r"\b(\d{1,3})[- ](?:heavy[- ]|)atom\b", text):
            n = int(m.group(1))
            if n not in heavies:
                near = min(heavies, key=lambda h: abs(h - n))
                out.append(_V("count",
                              f"calls something a {n}-atom fragment; nothing on this turn has "
                              f"{n} heavy atoms (nearest {near})",
                              f"count the heavy atoms: {sorted(heavies)[:8]}"))
    # "a cheap C2 unit" -- the carbon count of a small reagent
    for m in re.finditer(r"\bC(\d)\s+(?:unit|building block|fragment|source)\b", text):
        n = int(m.group(1))
        carbs = set()
        for s in _mols_in_scope(ev, ctx):
            mm = _mol(s)
            if mm is not None and mm.GetNumHeavyAtoms() <= 8:
                carbs.add(sum(1 for a in mm.GetAtoms() if a.GetSymbol() == "C"))
        if carbs and n not in carbs:
            out.append(_V("count",
                          f"calls a reagent a C{n} unit; the small reagents here have "
                          f"{sorted(carbs)} carbons",
                          "count the carbons"))
    return out


def _c_completeness(text: str, ev: dict, ctx: dict) -> list[dict]:
    """"complete", "every piece purchasable", "nothing can beat it" against the state.

    Two facts decide it and both are in the evidence: `mols` lists the molecules still open,
    and `untried` lists candidates the board has not spent. A hand-over is valid only when
    nothing left can beat the routes, so an untried candidate whose p exceeds the best route's
    weakest step refutes the claim outright.
    """
    if not COMPLETE.search(text):
        return []
    out = []
    still = sorted(ev.get("mols") or {})
    if still:
        out.append(_V("completeness",
                      f"claims the route is complete; {len(still)} molecule(s) are still open "
                      f"({', '.join(still[:5])})",
                      "an open piece is unsolved: say what is left"))
    weakest = [r["weakest"] for r in (ev.get("routes") or []) if r.get("weakest") is not None]
    if weakest:
        floor = max(weakest)
        better = [u for u in (ev.get("untried") or [])
                  if ((u.get("signals") or {}).get("p") or 0) > floor]
        if better and re.search(r"nothing (?:left on the board |else )?can beat", text, re.I):
            b = better[0]
            out.append(_V("completeness",
                          f"says nothing can beat the routes; {len(better)} untried candidate(s) "
                          f"score above the best floor {floor:.3f} "
                          f"(e.g. {b.get('mol')}·c{b.get('c')} at p "
                          f"{(b.get('signals') or {}).get('p')})",
                          "an untried candidate above the floor can still beat it"))
    bud = ev.get("budget") or []
    if (len(bud) == 2 and bud[1] and bud[0] < 0.5 * bud[1]
            and re.search(r"nothing (?:left on the board |else )?can beat", text, re.I)):
        out.append(_V("completeness",
                      f"closes the search with {bud[1] - bud[0]} calls still available",
                      "an unopened piece worth opening is a reason to keep looking, not to "
                      "hand over"))
    return out


def _c_route_floor(text: str, ev: dict, ctx: dict) -> list[dict]:
    """A route's weakest step, when the draft names one, is a number the evidence carries."""
    routes = {r.get("route"): r.get("weakest") for r in (ev.get("routes") or [])
              if r.get("weakest") is not None}
    if not routes:
        return []
    out = []
    for m in re.finditer(r"\broute ([A-Z])\b[^.;]{0,60}?(0?\.\d+|1\.0+)", text):
        lab, v = m.group(1), _num(m.group(2))
        w = routes.get(lab)
        if w is not None and v is not None and abs(v - w) > 5e-3:
            out.append(_V("route_floor",
                          f"gives route {lab} a floor of {m.group(2)}; its weakest step is "
                          f"{w:.3f}",
                          f"route {lab} is worth {w:.3f}"))
    # A probability above 1 is never a misquote worth arguing about; it is a misread decimal.
    for m in re.finditer(r"\b(?:weakest|floor|rates? (?:it )?at|sits at)\D{0,12}(\d\.\d+)", text):
        v = _num(m.group(1))
        if v is not None and v > 1.0:
            out.append(_V("route_floor",
                          f"quotes a plausibility of {m.group(1)}; p is a probability and "
                          f"cannot exceed 1",
                          "the board writes .103, not 1.03"))
    return out


def _c_repetition(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Sentences copied from earlier turns of the same episode.

    Turn k's prompt carries turns 0..k-1 with their reasoning, which is what makes the trace
    continuous and also what lets one paragraph become the episode's house style: the same
    few sentences, turn after turn, with the molecule name swapped.

    A sentence is a repeat when it shares 85% of its content words with one already written,
    and by that rule most turns contain at least one -- some of it legitimate, because the
    board's own vocabulary is fixed and a turn that continues a branch has to refer back to
    it. So the VIOLATION is not "contains a repeat" but "is mostly repeat": `REPEAT_RATIO` of
    the draft's sentences or more, which catches whole-draft boilerplate and leaves a draft
    with a couple of recycled sentences among fresh ones alone.
    """
    prev = ctx.get("prev_thoughts") or ()
    if not prev:
        return []
    # Not on an OPEN or DONE turn any more, and this is a real trade rather than a loosened
    # gate. Those two briefs were narrowed to two facts each with no chemistry in them -- the
    # piece is not purchasable, here is what sits above it; the leaves are purchasable, here is
    # what is not -- precisely so there is nothing left to invent. What that leaves is a turn
    # whose FRAME is prescribed and whose only varying content is board ids, and a threshold
    # calibrated on free prose ("is mostly repeat") is met by nearly every such turn after the
    # first few, and it would drop open turns for nothing else. The repeated sentence is not
    # boilerplate standing in for an argument -- it IS the argument, and the facts inside it
    # differ every turn. Rank keeps the check; its own `_c_prose_repetition` scores the DECIDING
    # reason clauses, where a repeat still means what this docstring says it means.
    if ctx.get("kind") in ("open", "done"):
        return []

    def bag(s: str) -> Counter:
        return Counter(w for w in re.findall(r"[a-z]{4,}", s.lower()))
    mine = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text) if len(s.split()) >= 8]
    old = [s.strip() for t in prev for s in re.split(r"(?<=[.;!?])\s+", t)
           if len(s.split()) >= 8]
    if not mine or not old:
        return []
    obags = [bag(s) for s in old]
    hits = []
    for s in mine:
        b = bag(s)
        tot = sum(b.values()) or 1
        for ob in obags:
            shared = sum((b & ob).values())
            if shared / tot >= 0.85:
                hits.append(s[:90])
                break
    ratio = len(hits) / len(mine)
    ctx["repeat_ratio"] = round(ratio, 2)
    if ratio >= REPEAT_RATIO:
        return [_V("repetition",
                   f"is {len(hits)} of {len(mine)} sentences repeated from an earlier turn "
                   f"({hits[0]!r}...)",
                   "this turn is a different decision: say what changed, not what held")]
    return []


# --------------------------------------------------------------- reasoning vs the action
# The one class of defect that needs no chemistry at all to see: the prose argues for one
# candidate and the tool call takes another. For example:
#
#   (a)  "c5 is the route to take" / "c7's transformation reproduces nothing"  -> order [7,0,5]
#   (b)  "the real disconnection ... which c6 offers" / "c0 ... sits second"   -> order [0,6]
#   (c)  "the cleanest way to close it is c5" / "c0 ... sits second"           -> order [0,5,1]
#
# (a) is the worst shape: one sample contains both the sentence refuting c7 and the call that
# applies it. A student trained on that learns the prose and the action are unrelated channels,
# which is the opposite of what the whole trace is for. (b) and (c) do not even need the
# action -- "c0 sits second" and "I declare 0, 6" contradict each other inside one paragraph.
#
# Markers, not parsing. A draft says "so it leads" or "sits second as a fallback" in a small
# number of ways, and a sentence carrying one of those with exactly ONE candidate number in it
# is unambiguous about which candidate it means. Sentences with two numbers are skipped rather
# than guessed at: fail open is worth more here than coverage, because this check gates.
# The conclusion, with the candidate it concludes on: "taking c2 now", "I apply c0".
_TAKES = re.compile(r"(?:tak(?:ing|es?)|apply(?:ing)?|applies)\s+c(\d)\b", re.I)
_LEAD = re.compile(
    r"(is the (?:route|cut|one|disconnection) to take|so it leads\b|it leads\b"
    r"|earns the top slot|the top slot|is the cleanest|cleanest (?:way|cut|seam)"
    r"|I take it first|take it first|is the strongest|the one to take"
    r"|the real disconnection|is the cut I|I take c|I apply c|apply c\d now"
    # The concluding idiom the drafts actually use -- "taking c2 now and holding c0 as the
    # fallback". Without it the LAST endorsement in a draft that argues past its first choice
    # is the one it argued past, and the check reads a correct conclusion as a contradiction.
    r"|tak(?:ing|es) c\d|and (?:take|apply) c\d"
    r"|leads,? with|first choice)|is the stronger|the stronger of|is the one I take", re.I)
# `below`/`behind` are not here either: they are RELATIONAL and invert the reading. "the methyl
# is the worse handle, so it sits behind c2" demotes the methyl candidate and promotes c2, but
# the clause names only c2, so a marker matching there attributes the demotion to the winner.
# `second` and `third` are NOT here bare. They match route ordering -- "B second, and the
# spiro-built route third", "gives the board a second handle" -- on turns whose candidate
# ranking is fine, rejecting them for a word about something else. What is
# demotion is the candidate being placed BELOW another one, so the word has to be attached to
# that: "sits second", "ranks third", "as the second choice".
_DEMOTE = re.compile(
    r"(sits? (?:second|third)(?!\s+c\d)|ranks? (?:second|third)"
    r"|(?:is|as) the (?:second|third) (?:choice|option|cut|candidate)"
    r"|as a fallback|the fallback\b|is the fallback|is my fallback"
    r"|held (?:as|in reserve)|in reserve|if (?:it|that|this) (?:fails|dies)"
    r"|sits? last|is last|comes? last)", re.I)
# The candidate a clause names as the thing something else is ranked against, rather than as
# the thing being ranked. "behind c0", "after c0", "below c0", "ahead of c0" all put the named
# candidate on the other side of the comparison from the one being placed.
_REFERENCE = re.compile(r"\b(?:behind|after|below|under|ahead of|above)\s+c\d\b", re.I)
_REFUTE = re.compile(
    r"(reproduces nothing|nothing reproduces|cannot carry|does not divide|divides? nothing"
    r"|changes no bond|is out\b|dead end|invents? a|fails the|is a weak\b|buys? nothing"
    r"|do(?:es)? not simplif)", re.I)
# "I declare 0, 6" / "I rank yq as c4, c2, c0" -- the order stated in the prose itself.
_DECLARED = re.compile(
    r"(?:I )?(?:declare|rank|order)\b[^.;\n]{0,60}?"
    r"((?:c?\d\s*(?:,|and|·|then|\s)+)*c?\d)\b", re.I)


def _cands_in(sent: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\bc(\d)\b", sent)})


# Clauses, not sentences. Real drafts put the verdict on two candidates in one sentence --
# "c7's transformation reproduces nothing, while c5 clears the filter at 0.998" -- and a
# sentence-level split sees two numbers, cannot attribute either, and skips the turn. Splitting
# on the contrastive joins as well recovers both halves with one candidate each, which is what
# makes (a) above (the sample that refutes c7 and then applies it) visible at all.
_CLAUSE = re.compile(r"(?<=[.;!?])\s+|\bwhile\b|\bwhereas\b|,\s+but\b|\bbut\b(?=\s+c\d)"
                     r"|:\s+|\s+[—–]\s+|\s+--\s+")


def _clauses(text: str) -> list[str]:
    return [c for c in _CLAUSE.split(text) if c and c.strip()]


def _c_action_match(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Does the prose argue for the candidate the call actually takes?

    Scoped to turns with exactly ONE rank or done action. A turn taking several actions can
    legitimately praise a candidate of one molecule while ranking another, and disentangling
    which sentence belongs to which molecule is guesswork; those turns get the molecule check
    only, which is unambiguous either way.
    """
    acts = [a for a in (ctx.get("actions") or [])
            if isinstance(a, dict) and (a.get("type") or "").lower() in ("rank", "done")]
    out = []

    # -- the molecule. Naming the id at all is a house style, not a duty -- most drafts
    # describe the piece instead ("the aldehyde piece I opened last turn"), and gating on a
    # missing id would reject those. What is a defect is naming a DIFFERENT one: an earlier
    # turn's wording about `yq` reused while ranking `z2`, a fragment of `tm` described while
    # ranking `u7`. So the check needs a wrong id present, not the right id absent.
    known = set(ev.get("mols") or {}) | set(ev.get("menus") or {})
    known |= {b.get("mol") for b in (ev.get("banked") or []) if b.get("mol")}
    for a in acts:
        # `rank` only. A `done` names a candidate for EVERY molecule the route leaves to make
        # -- several ids for one claim -- and no draft enumerates them, so requiring each in
        # the prose would reject turns for describing the route instead of listing it.
        # A rank has exactly one subject, which is what this check is about.
        if (a.get("type") or "").lower() != "rank":
            continue
        mids = [a["mid"]] if a.get("mid") else []
        for mid in mids:
            if not mid or re.search(rf"\b{re.escape(mid)}\b", text):
                continue
            others = sorted(m for m in known
                            if m != mid and re.search(rf"\b{re.escape(m)}\b", text))
            if others:
                out.append(_V("action_molecule",
                              f"names {', '.join(others)} but the call acts on {mid}",
                              f"this turn is about {mid}: say what {mid} is and why this cut"))
    if len(acts) != 1:
        return out
    a = acts[0]
    order = list(a.get("order") or [])
    if not order and a.get("choices"):
        # A `done` names a candidate for EVERY molecule the route uses, and all but one of
        # them are reactions already sitting in the graph. Only one is a DECISION this turn:
        # the piece whose menu is on the board, which is the one the paragraph argues about.
        # Taking `values()[0]` instead would pick whichever molecule the dict happens to list
        # first and silently agree with the prose whenever that number matched -- a done turn
        # would come back clean while the prose argued for c0 and the call sent
        # `{'xm': 0, ..., 'ju': 5}`.
        live = [m for m in a["choices"] if m in (ev.get("menus") or {})]
        if len(live) != 1:
            return out                    # no single subject: cannot attribute, stay silent
        subject = live[0]
        try:
            order = [int(a["choices"][subject])]
        except (TypeError, ValueError):
            return out
    if not order:
        return out
    applied = int(order[0])

    # -- the order the prose states, against the order the call sends
    for m in _DECLARED.finditer(text):
        nums = [int(x) for x in re.findall(r"\d", m.group(1))]
        if len(nums) < 2:
            continue
        if nums != order[:len(nums)]:
            out.append(_V("action_order",
                          f"states the ranking as {', '.join('c%d' % n for n in nums)} but the "
                          f"call sends {', '.join('c%d' % n for n in order)}",
                          f"the call is {', '.join('c%d' % n for n in order)}: argue for that"))
        break

    # -- per sentence: which candidate is praised, which is demoted, which is refuted
    lead, demoted, refuted = [], set(), set()
    for sent in _clauses(text):
        cs = _cands_in(sent)
        # An explicit conclusion names its own candidate, so it can be attributed even in a
        # clause that mentions others: "taking c2 now and holding c0 as the fallback" concludes
        # c2 and merely mentions c0. Without this the clause is skipped for ambiguity and the
        # last endorsement in the draft is whichever candidate it argued past on the way.
        m = _TAKES.search(sent)
        if m:
            lead.append(int(m.group(1)))
            continue
        if len(cs) != 1:
            continue                      # two numbers in one clause: cannot attribute
        c = cs[0]
        if _REFUTE.search(sent):
            refuted.add(c)
        elif _DEMOTE.search(sent) and not _REFERENCE.search(sent):
            # `_REFERENCE` is the same trap the `sits second` guard above was written for, in
            # its other form. "c7 could not be recovered, so it sits BEHIND c0 as the
            # fallback" names only c0, and c0 is what the fallback is measured against -- the
            # candidate being demoted is c7, which the clause does not name. Attributing the
            # demotion to the reference marks the winner as demoted and rejects a draft that
            # argues correctly for the candidate it takes.
            demoted.add(c)
        elif _LEAD.search(sent):
            lead.append(c)

    if applied in refuted:
        out.append(_V("action_refuted",
                      f"refutes c{applied} and then applies it",
                      f"c{applied} is the candidate this call takes: argue for it or take "
                      f"another"))
    if applied in demoted:
        out.append(_V("action_demoted",
                      f"puts c{applied} second or in reserve and then applies it first",
                      f"c{applied} leads this ranking: say why it leads"))
    # The LAST endorsement, not the set of them. A paragraph is allowed to name the obvious
    # cut and then argue past it -- "the model points at c0 and it reads cleanly forwards, but
    # the weakest step is the acetal, and c2 is the only cut that addresses it, so I take c2"
    # is the reasoning we want, and scoring every endorsement in the draft rejects it for the
    # sentence it spends refuting. A draft that changes its mind ends on its conclusion, so
    # the conclusion is what has to agree with the call. A draft that ends "taking c0 now"
    # against a call that sends c4 still fails, correctly.
    if lead and lead[-1] != applied:
        p = lead[-1]
        out.append(_V("action_endorsed",
                      f"ends by arguing for c{p}, but the call applies c{applied}",
                      f"either lead with c{p} or make the case for c{applied}"))
    return out


# ------------------------------------------------------------------ the rt column
# `rt` is the rank at which the forward model returns the product from these precursors, and a
# dash means it could not be recovered AT ALL. "c1 ... reads cleanly forwards (rt 1)" against
# a board that prints rt-dash for c1 is a false claim on one of the three axes a route is
# judged on. The
# claim is attributed the way the action markers are -- by clause, one candidate per clause.
# Leading word boundaries are load-bearing, not decoration. Without them `recover\w*` matched
# the tail of "UNrecovered" and `back\b` matched the tail of "fallBACK", so
# "c0 ... plausible but unrecovered, so it is the fallback" -- a correct sentence saying the
# exact opposite -- was read as a forward-recovery claim.
_READS_FWD = re.compile(
    r"\b(?:reads?|recovers?|recovered|comes?|returns?)\b[^.;]{0,40}?"
    r"(?:back\s+)?(?:cleanly|clean|well)?\s*\b(?:forwards?|back)\b|\brt\s?1\b", re.I)
_NO_FWD = re.compile(
    r"(could not be recovered|cannot be recovered|fails? the forward|no forward recovery"
    r"|the forward model (?:could not|cannot|fails)|rt\s?[—–-]\B|not recovered"
    r"|\bunrecovered\b|\bnever recovered\b|\bwithout (?:a )?forward recovery)", re.I)
_RT_CITE = re.compile(r"\brt\s?(\d)\b", re.I)


def _c_rt(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Forward-recovery claims against the rt column.

    Three shapes, all decidable: a candidate said to read forward that the forward model never
    recovered; a candidate said NOT to be recovered that reads back at rank 1; and a cited
    `rt N` that is not the N the board printed for that candidate.
    """
    fuzzy = _ambiguous_cands(ev)          # a plain dict here let the last molecule win
    rt = {c["c"]: (c.get("signals") or {}).get("rt")
          for _, c, _ in _rows(ev) if c["c"] not in fuzzy}
    if not rt:
        return []
    out = []
    for cl in _clauses(text):
        cs = _cands_in(cl)
        if len(cs) != 1 or cs[0] not in rt:
            continue
        c = cs[0]
        v = rt[c]
        cite = _RT_CITE.search(cl)
        if cite:
            n = int(cite.group(1))
            if v is None:
                out.append(_V("rt_claim",
                              f"gives c{c} rt {n}; the forward model did not recover it at all",
                              f"c{c} is a dash in the rt column: say the forward model could "
                              f"not recover it"))
                continue
            if v != n:
                out.append(_V("rt_claim",
                              f"gives c{c} rt {n}; the board prints rt {v}",
                              f"c{c} reads back at rank {v}"))
                continue
        if _NO_FWD.search(cl):
            if v == 1:
                out.append(_V("rt_claim",
                              f"says c{c} was not recovered forwards; it reads back at rank 1",
                              f"c{c} is rt 1"))
        elif _READS_FWD.search(cl) and v is None:
            out.append(_V("rt_claim",
                          f"has c{c} reading cleanly forwards; the forward model could not "
                          f"recover it (rt is a dash)",
                          f"c{c} was not recovered: a dash is a real warning even when p is "
                          f"high"))
    return out


# ------------------------------------------------------------------ quantifiers
# "the only cut that reads forward" and "all three read forward at rt 1" are claims about a
# COUNT, and the count is on the board. They are often wrong, and the worst are self-refuting
# inside one paragraph: "c0 ... the only cut that reads forward cleanly" followed by "c2 ...
# also rt 1", or "All three read forward at rt 1" on a molecule whose every candidate is
# rt-dash.
_ONLY_FWD = re.compile(
    r"the only (?:cut|one|candidate|disconnection|step)\b[^.;]{0,80}?"
    r"(?:reads?|recover\w*|comes?) (?:back |it )?(?:cleanly |clean )?forward", re.I)
# "both" has to be about CANDIDATES. "it scores well on both p and rt" is about two SIGNALS,
# and counting it as two candidates reading forward is a false positive -- so a `both`
# followed by a signal name is not this claim.
_ALL_FWD = re.compile(
    r"\b(all (?:two|three|four|five|six|\d+)|both|every one)\b"
    r"(?!\s+(?:p\b|q\b|rt\b|signals|scores|axes|of (?:its|the) (?:signals|scores)))"
    r"[^.;]{0,60}?(?:reads?|recover\w*) (?:back )?(?:cleanly |clean )?forward", re.I)
_WORD_N = {"both": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
# A claim can be a CONJUNCTION -- "the only cut that both goes and reads cleanly forwards",
# "the only cut that passes the filter and reads forward". Counting rt alone then refutes it
# with candidates the speaker already excluded on plausibility, which is a false rejection of
# a correct sentence. When the claim names the filter, the count has to carry both terms.
_ALSO_FILTER = re.compile(r"\b(?:passes|clears?|goes|above|both goes)\b[^.;]{0,40}?"
                          r"(?:the )?(?:filter|cut-?off|plausibilit)|"
                          r"\bboth (?:goes|passes|clears)\b", re.I)


def _c_quantifier(text: str, ev: dict, ctx: dict) -> list[dict]:
    """"the only X that reads forward" / "all three read forward" against the rt column."""
    rows = [(c["c"], (c.get("signals") or {}).get("rt"), (c.get("signals") or {}).get("p"))
            for _, c, _ in _rows(ev)]
    if not rows:
        return []
    out = []
    m = _ONLY_FWD.search(text)
    if m:
        sent = next((s for s in re.split(r"(?<=[.;!?])\s+", text) if _ONLY_FWD.search(s)), text)
        conj = bool(_ALSO_FILTER.search(sent))
        qual = sorted(i for i, rt, pp in rows
                      if rt == 1 and (not conj or (pp is not None and pp >= CUTOFF)))
        if len(qual) > 1:
            what = ("clear the filter and read forward" if conj else "read forward")
            out.append(_V("quantifier",
                          f"calls one candidate the only one that reads forward; "
                          f"{len(qual)} {what} ({', '.join('c%d' % i for i in qual)})",
                          f"say which of {', '.join('c%d' % i for i in qual)} you take and "
                          f"why, not that it is alone"))
    m = _ALL_FWD.search(text)
    if m:
        word = (m.group(1) or "").lower().replace("all ", "").strip()
        want = _WORD_N.get(word, 2)
        clean = sorted(i for i, rt, _ in rows if rt == 1)
        if len(clean) < want:
            out.append(_V("quantifier",
                          f"has {want} candidates reading forward; {len(clean)} do"
                          + (f" ({', '.join('c%d' % i for i in clean)})" if clean
                             else " -- none was recovered at all"),
                          "a dash in the rt column means the forward model could not recover "
                          "it: say so rather than counting it in"))
    return out


def _c_quantifier(text: str, ev: dict, ctx: dict) -> list[dict]:
    """"the only X that reads forward" / "all three read forward" against the rt column."""
    rts = [(c["c"], (c.get("signals") or {}).get("rt")) for _, c, _ in _rows(ev)]
    if not rts:
        return []
    clean = sorted(i for i, rt in rts if rt == 1)
    out = []
    if _ONLY_FWD.search(text) and len(clean) > 1:
        out.append(_V("quantifier",
                      f"calls one candidate the only one that reads forward; "
                      f"{len(clean)} do ({', '.join('c%d' % i for i in clean)})",
                      f"rt 1 is on {', '.join('c%d' % i for i in clean)}: say which you take "
                      f"and why, not that it is alone"))
    m = _ALL_FWD.search(text)
    if m:
        word = (m.group(1) or m.group(2) or "").lower().replace("all ", "").strip()
        want = _WORD_N.get(word, 2)
        if len(clean) < want:
            out.append(_V("quantifier",
                          f"has {want} candidates reading forward; {len(clean)} do"
                          + (f" ({', '.join('c%d' % i for i in clean)})" if clean
                             else " -- none was recovered at all"),
                          "a dash in the rt column means the forward model could not recover "
                          "it: say so rather than counting it in"))
    return out


# ------------------------------------------------------------------ the hand-over
_CHEAPEST = re.compile(r"\broute ([A-Z])\b[^.;]{0,70}?\b(?:is )?(?:also )?the cheapest"
                       r"|\bthe cheapest[^.;]{0,30}?\bis route ([A-Z])\b", re.I)
_CHEAPER = re.compile(r"\b([A-Z]) is cheaper\b|\bcheaper still[^.;]{0,20}?\b([A-Z])\b"
                      r"|\b([A-Z])\b[^.;]{0,20}?\bis cheaper still", re.I)
_WOULD_NOT = re.compile(r"\b([A-Z]) is the one I would not claim"
                        r"|\bI would not claim ([A-Z])\b", re.I)


def _c_handover(text: str, ev: dict, ctx: dict) -> list[dict]:
    """Two contradictions the hand-over kept making, both decidable without any chemistry.

    A superlative that the next line takes back -- "I would run F ... and it is also the
    cheapest at $62", then "C is cheaper still" two lines down, with the listing's own header
    saying cheapest: C. And naming a route as one the agent would NOT claim when that route is
    claimed and sitting in the listing underneath: `ev["routes"]` IS the claimed set.
    """
    out = []
    ch = _CHEAPEST.search(text)
    if ch:
        lab = next(g for g in ch.groups() if g)
        for m in _CHEAPER.finditer(text):
            other = next(g for g in m.groups() if g)
            if other.upper() != lab.upper():
                out.append(_V("handover_contradiction",
                              f"calls route {lab.upper()} the cheapest and then says "
                              f"{other.upper()} is cheaper",
                              "one route is the cheapest: name it once and keep it"))
                break
    claimed = {str(r.get("route")).upper() for r in (ev.get("routes") or []) if r.get("route")}
    for m in _WOULD_NOT.finditer(text):
        lab = next(g for g in m.groups() if g).upper()
        if lab in claimed:
            out.append(_V("handover_contradiction",
                          f"says route {lab} is one it would not claim; {lab} is claimed and "
                          f"is in the listing",
                          f"either do not claim {lab} or do not disown it"))
    return out


# ------------------------------------------------------------------ stock closing lines
# Distinct from `repetition`, which asks whether a WHOLE draft is recycled and so never fires
# on a turn that reasons freshly and then ends with the same sentence every time -- "Nothing
# about its cuts exists yet, so the move is to lay them out. I open X." on turn after turn. A
# closing line repeated across most of an episode is a
# format the student learns to emit rather than a decision it learns to make.
def _c_stock_close(text: str, ev: dict, ctx: dict) -> list[dict]:
    prev = ctx.get("prev_thoughts") or ()
    if len(prev) < 6:
        return []
    def tail(t):
        ss = [x.strip() for x in re.split(r"(?<=[.;!?])\s+", t or "") if x.strip()]
        return " ".join(ss[-2:]) if ss else ""
    def bag(t):
        return Counter(w for w in re.findall(r"[a-z]{4,}", t.lower()))
    mine = bag(tail(text))
    if sum(mine.values()) < 5:
        return []
    hits = 0
    for t in prev:
        ob = bag(tail(t))
        if not ob:
            continue
        shared = sum((mine & ob).values())
        if shared / max(sum(mine.values()), 1) >= 0.8:
            hits += 1
    if hits >= max(4, int(0.5 * len(prev))):
        return [_V("stock_close",
                   f"ends with the sentence {hits} of {len(prev)} earlier turns end with",
                   "close on what this turn decided, not on a formula")]
    return []


CHECKS = {
    "action_match": _c_action_match,
    "rt": _c_rt,
    "quantifier": _c_quantifier,
    "handover": _c_handover,
    "stock_close": _c_stock_close,
    "numbers": _c_numbers,
    "q_argmax": _c_q_argmax,
    "cutoff": _c_cutoff,
    "named": _c_named,
    "named_conflict": _c_named_conflict,
    "open_choice": _c_open_choice,
    "open_together": _c_open_together,
    "open_named": _c_open_named,
    "ring_identity": _c_ring_identity,
    "fg_identity": _c_functional_group,
    "purchasable": _c_purchasable,
    "shape": _c_shape,
    "counts": _c_counts,
    "completeness": _c_completeness,
    "route_floor": _c_route_floor,
    "repetition": _c_repetition,
}

# Which codes no rewrite can repair, so a draft carrying one is refused rather than salvaged.
# Everything here is a false CLAIM about the chemistry or the state; `repetition` is not, it is
# a register problem like jargon, and a redraw fixes it.
# `action_molecule` gates. `repetition` does not catch its defect: a paragraph written about
# `tm` while the call ranks `u7` is not a repeat of any single earlier turn, so repetition
# scores it clean, and the draft teaches the model to describe one molecule and act on
# another. The check requires a WRONG id to be present rather than the right id to be absent,
# which is what makes it safe to gate -- a draft that names no molecule at all, the common
# terse case, never fires it. It fires rarely, so the cost is small and the defect is one of
# the worst in the set.
FATAL = {"rt_claim", "quantifier", "handover_contradiction",
         "action_molecule",
         "action_endorsed", "action_refuted", "action_demoted", "action_order",
         "q_argmax", "cutoff_class", "only_clears", "applied_subcutoff", "named_unearned",
         "named_applies_only", "named_conflict", "ring_identity", "fg_identity",
         "purchasable", "shape", "count", "completeness", "route_floor",
         "q_value", "p_value", "price_value",
         # `repetition` gates as well. Left advisory, recycled drafts accumulate, because the
         # prompt shows the model its own earlier turns and a house style, once started, is
         # cheaper for it than new prose. The threshold already tolerates the
         # legitimate case -- it fires only when 60% or more of a draft's sentences repeat --
         # so what it rejects is boilerplate, not continuity.
         "repetition",
         # The two open-turn clauses are claims about which molecules the call touches, and
         # every part of both is checkable against the board: what was openable, what was
         # opened, and which reaction each opened piece sits under. `opening together` is the
         # batch decision -- the one that makes the search breadth-first and feeds the next
         # rank turn rival cuts at one depth -- and `leaving open` is where the search does
         # not go. Said where they are not earned they are narration; left out where they are,
         # the only decisions an open turn makes go unwritten.
         "open_together_missing", "open_together_unearned", "open_together_wrong",
         "open_together_incomplete", "open_together_under_wrong", "open_together_relation",
         "open_together_count", "open_together_recycled", "open_named_uncited",
         "open_choice_missing", "open_choice_unearned", "open_choice_wrong",
          # A price axis on a turn the search did not decide on price is a false
          # reason, not a weak one: it teaches the coincidence as the cause.
          "price_axis"}


def check(text: str, ev: dict, kind: str = "rank", prev_thoughts=(),
          actions=None, seen_smiles=(), enable: set[str] | None = None,
          observation=None) -> dict:
    """Run every check against one draft. Never raises; a broken check is reported as one.

    `observation` is the board screen for this turn. Nothing here reads it: the callers in
    traj_route_reasoning_routeloop pass it, so the argument is accepted and ignored.

    `seen_smiles` is every molecule the episode has shown BEFORE this turn. The identity and
    count checks widen their scope with it, so a draft that refers back to a fragment banked
    several turns ago is not refuted for remembering -- see `_mols_in_scope`. Omitting it makes
    those checks turn-local and noisy; it is not required for the rest.
    """
    if not text or not ev:
        return {"ok": True, "violations": [], "codes": [], "failed_checks": []}
    ctx = {"kind": kind, "prev_thoughts": list(prev_thoughts), "actions": actions,
           "seen_smiles": list(seen_smiles)}
    viol, failed = [], []
    for name, fn in CHECKS.items():
        if enable is not None and name not in enable:
            continue
        try:
            viol.extend(fn(text, ev, ctx) or [])
        except Exception as e:                                         # noqa: BLE001
            failed.append(f"{name}: {type(e).__name__}: {e}")
    # One violation per (code, detail): a word repeated three times is one mistake.
    seen, uniq = set(), []
    for v in viol:
        k = (v["code"], v["detail"])
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    codes = sorted({v["code"] for v in uniq})
    return {"ok": not uniq, "violations": uniq, "codes": codes,
            "fatal": sorted(c for c in codes if c in FATAL),
            # recorded whether or not it crossed the bar: the trend over an episode is the
            # signal that a house style is forming, and it is invisible in a pass/fail flag
            "repeat_ratio": ctx.get("repeat_ratio"),
            "failed_checks": failed}


def describe(v: dict, limit: int = 6) -> str:
    """The violations as the correction to re-prompt with.

    Written as instructions about the chemistry, not as a report about a checker: the teacher
    is being told what is wrong with its claim, and a message about "the fact-checker" invites
    a draft that narrates having been corrected.
    """
    if v.get("ok"):
        return ""
    lines = []
    for x in v["violations"][:limit]:
        lines.append(f"- Your draft {x['detail']}."
                     + (f" {x['fix'].rstrip('.')}." if x.get("fix") else ""))
    return ("The draft states things this turn's facts contradict. Rewrite it so every claim "
            "below is corrected, keeping the same decision and the same voice:\n"
            + "\n".join(lines))
