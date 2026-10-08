#!/usr/bin/env python
"""Text -> structure.  The inverse of render.py, and the reason it is trusted.

Two jobs, and they are different:

  parse_act(text)   the model's <act> block -> Action objects.  This one runs in
                    anger: it is what an RL rollout uses to move the board, so
                    every rejection carries a message the model could act on.

  parse_env(text)   the environment turn -> StateView.  Nothing in training
                    needs this; it exists so that check_roundtrip() can assert
                    that every fact the board holds survived the trip to text
                    and back.  A field that renders but does not parse is a
                    field the model is being asked to infer, and that should be
                    a decision, not an accident.

Run the round-trip over a real episode with

  python -m board.parse --selftest
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .state import (Analyze, Board, Dead, Done, Open, Rank, Terminate,
                    DEAD_REASONS)

# --------------------------------------------------------------- the act block
_ACT_WRAP = re.compile(r"<act>(.*?)</act>", re.S)
_OPEN = re.compile(r"^open\s+(\S+)$")
_RANK = re.compile(r"^rank\s+(\S+)\s+((?:c\d+\s*)+)$")
_DEAD = re.compile(r"^dead\s+(\S+)\s*(?:reason\s*:\s*(.+))?$")
_DONE = re.compile(r"^done\s+(.*)$")
_TERM = re.compile(r"^(?:terminate|hand over)\b\s*(.*)$")
_REASON_BY_TEXT = {v: k for k, v in DEAD_REASONS.items()}


class ActError(Exception):
    """Rejected model output.  The message is meant to be shown back."""


def parse_act(text: str) -> list:
    """Parse one assistant action block.  Order is preserved and it matters."""
    m = _ACT_WRAP.search(text)
    body = m.group(1) if m else text
    out = []
    for raw in body.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if mo := _OPEN.match(line):
            out.append(Open(mo.group(1)))
        elif mo := _RANK.match(line):
            idxs = [int(x[1:]) for x in mo.group(2).split()]
            if len(set(idxs)) != len(idxs):
                raise ActError(f"repeated candidate in: {line}")
            out.append(Rank(mo.group(1), idxs))
        elif mo := _DEAD.match(line):
            reason = (mo.group(2) or "").strip()
            key = _REASON_BY_TEXT.get(reason)
            if key is None:
                raise ActError(
                    "dead needs one of the three named reasons, got "
                    f"{reason!r}; expected one of: "
                    + " | ".join(DEAD_REASONS.values())
                )
            out.append(Dead(mo.group(1), key))
        elif mo := _TERM.match(line):
            out.append(Terminate(mo.group(1).strip()))
        elif mo := _DONE.match(line):
            out.append(Done(_choices_from_text(mo.group(1))))
        else:
            raise ActError(f"not an action: {line!r} "
                           f"(expected open/rank/dead/done/terminate)")
    if not out:
        raise ActError("empty act block")
    return out


def _choices_from_text(text: str) -> dict:
    """`qm·c0 vg·c9` -- the ledger's own way of naming a reaction."""
    out: dict[str, int] = {}
    for tok in text.split():
        if "\u00b7" not in tok:
            raise ActError(f"done takes molecule\u00b7candidate pairs, got {tok!r}")
        mid, _, c = tok.partition("\u00b7")
        if not c.startswith("c") or not c[1:].isdigit():
            raise ActError(f"done takes molecule\u00b7candidate pairs, got {tok!r}")
        if mid in out:
            raise ActError(f"done names {mid} twice")
        out[mid] = int(c[1:])
    if not out:
        raise ActError("done needs a candidate per molecule the route uses")
    return out


def parse_act_json(payload) -> list:
    """Parse a board_act argument.  Same rejections as parse_act, same messages.

    The tool-call form is the one an rl rollout of gpt-oss produces, so it has to
    reject exactly what the text form rejects -- an unknown reason key, a
    repeated candidate, an action with no molecule.
    """
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except Exception as exc:
            raise ActError(f"board_act argument is not JSON: {exc}")
    if not isinstance(payload, dict) or "actions" not in payload:
        raise ActError('board_act needs an object with an "actions" array')
    acts = payload["actions"]
    if not isinstance(acts, list) or not acts:
        raise ActError('"actions" must be a non-empty array')
    out = []
    for a in acts:
        if not isinstance(a, dict) or "type" not in a:
            raise ActError(f'each action needs a "type": {a!r}')
        kind = a["type"]
        if kind == "terminate":
            out.append(Terminate(str(a.get("note", ""))))
            continue
        if kind == "done":
            ch = a.get("choices")
            if not isinstance(ch, dict) or not ch:
                raise ActError('done needs "choices": a candidate number per '
                               'molecule, e.g. {"qm": 0, "vg": 9}')
            choices = {}
            for mid, c in ch.items():
                if not isinstance(mid, str) or not mid:
                    raise ActError(f'done has a molecule that is not an id: {mid!r}')
                # bool is an int in python, and `true` in this slot is a typo that
                # would otherwise be read as candidate 1.
                if isinstance(c, bool) or not isinstance(c, int):
                    raise ActError(f'done needs a candidate NUMBER for {mid}, '
                                   f'got {c!r}')
                if c < 0:
                    raise ActError(f'done has a negative candidate for {mid}: {c}')
                choices[mid] = c
            out.append(Done(choices, str(a.get("note", ""))))
            continue
        if kind == "analyze":
            # Handled before the `mid` check because analyze names SEVERAL molecules, and
            # per molecule the CANDIDATES it wants: {"qm": [0, 2, 4]}. One action covers the
            # whole request -- the tools behind it are batched per product, so N calls are the
            # same work N times over. `mids` is the whole-menu shorthand.
            mids = a.get("mids") or []
            cands = a.get("candidates") or {}
            if not isinstance(mids, list):
                raise ActError(f'"mids" must be a list of molecule ids: {mids!r}')
            if any(not isinstance(x, str) or not x for x in mids):
                raise ActError(f'"mids" must be molecule ids: {mids!r}')
            if not isinstance(cands, dict):
                raise ActError('analyze "candidates" must be a map of molecule id to '
                               'candidate numbers, got ' + repr(cands))
            clean = {}
            for m, v in cands.items():
                if not isinstance(m, str) or not m:
                    raise ActError(f'analyze has a molecule that is not an id: {m!r}')
                if not isinstance(v, list) or not v:
                    raise ActError(f'analyze needs a non-empty candidate list for {m}')
                if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in v):
                    raise ActError(f'analyze needs candidate NUMBERS for {m}, got {v!r}')
                clean[m] = list(dict.fromkeys(v))
            if not mids and not clean:
                raise ActError('analyze needs "candidates" {mid: [c, ...]} or "mids" [...]')
            if len(set(mids)) != len(mids):
                raise ActError(f"repeated molecule in {mids!r}")
            out.append(Analyze(list(mids), clean))
            continue
        mid = a.get("mid")
        if not isinstance(mid, str) or not mid:
            raise ActError(f'{kind} needs "mid": {a!r}')
        if kind == "open":
            out.append(Open(mid))
        elif kind == "rank":
            order = a.get("order")
            if not isinstance(order, list) or not order:
                raise ActError(f'rank needs a non-empty "order": {a!r}')
            if any(not isinstance(i, int) for i in order):
                raise ActError(f'"order" must be candidate numbers: {order!r}')
            if len(set(order)) != len(order):
                raise ActError(f"repeated candidate in {order!r}")
            take = a.get("take", 1)
            if not isinstance(take, int) or take < 1:
                raise ActError(f'"take" must be a positive integer: {take!r}')
            if take > len(order):
                raise ActError(f'"take" is {take} and the ranking has {len(order)}')
            out.append(Rank(mid, list(order), take=take))
        elif kind == "dead":
            reason = a.get("reason")
            if reason not in DEAD_REASONS:
                raise ActError(
                    f"dead needs one of {sorted(DEAD_REASONS)}, got {reason!r}"
                    " -- " + " | ".join(f"{k}: {v}" for k, v in DEAD_REASONS.items()))
            out.append(Dead(mid, reason))
        else:
            raise ActError(f"unknown action type {kind!r}")
    return out


# ------------------------------------------------------------- the environment
@dataclass
class MenuView:
    idx: int
    signals: dict[str, float]
    pieces: list[str]
    buyable: list[bool]


@dataclass
class OpenView:
    mid: str
    under: Optional[str]
    depth: int
    left: int
    has_menu: bool
    menu: list[MenuView] = field(default_factory=list)
    hidden: Optional[int] = None
    smiles: Optional[str] = None
    failed: list[int] = field(default_factory=list)


@dataclass
class LedgerView:
    rid: str
    parent: str
    cand: int
    closed: Optional[int]
    total: Optional[int]
    status: str                     # open | solved | failed
    ranked: list[int] = field(default_factory=list)
    was: Optional[list[int]] = None


@dataclass
class StateView:
    budget_used: int = 0
    budget_max: int = 0
    events: list[str] = field(default_factory=list)
    ledger: list[LedgerView] = field(default_factory=list)
    open: list[OpenView] = field(default_factory=list)
    closed: dict[str, Optional[float]] = field(default_factory=dict)
    dead: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)


_BUDGET = re.compile(r"^budget (\d+) of (\d+)$")
_LEDGER = re.compile(
    r"^(r\d+)  (\S+)·c(\d+)  (?:FAILED|(\d+) of (\d+) closed(?: · solved)?)"
    r"(?:\s+(\S+) ranked: ([^(]+?)(?:\s+\(was (.+)\))?)?\s*$"
)
_OPENL = re.compile(r"^  (\S+)\s+(?:under (r\d+) · )?depth (\d+) · (\d+) left(?: · (.+))?$")
_CANDL = re.compile(r"^    <c(\d+)(?:\s+([^>]*))?>(.*)</c(\d+)>$")
_TAIL = re.compile(r"^    \.\.\. (\d+) more(?:,.*)?$")
_MOLTAG = re.compile(r"^\s*<mol (\S+)>(.*)</mol>\s*$")
_SIG = re.compile(r"([a-z]+)(\.\d+|\d+|—)")
_CLOSED_ITEM = re.compile(r"^(\S+?)\*(?:\(ln\$([\d.]+)\))?$")
_CN = re.compile(r"c(\d+)")


def _signals(blob: str) -> dict[str, float]:
    out = {}
    for k, v in _SIG.findall(blob or ""):
        if v == "—":
            continue
        out[k] = float("0" + v) if v.startswith(".") else float(v)
    return out


def parse_env(text: str) -> StateView:
    """Read back one rendered environment turn.

    Deliberately strict about the parts that carry state (budget, ledger, OPEN,
    CLOSED, DEAD) and deliberately loose about the parts that are prose for the
    model (the event lines, the route tree) -- those are kept verbatim so a
    change to their wording does not break the check.
    """
    view = StateView()
    section = None
    cur: Optional[OpenView] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if mo := _BUDGET.match(line):
            view.budget_used, view.budget_max = int(mo.group(1)), int(mo.group(2))
            continue
        if line.startswith("[Event]") or (section == "event" and line.startswith(" ")):
            section = "event"
            view.events.append(line)
            continue
        if line == "ROUTES" or line.startswith("ROUTES  "):
            # "ROUTES  21 found" in summary mode, bare "ROUTES" in full mode.
            section = "routes"
            continue
        if line == "OPEN":
            section, cur = "open", None
            continue
        if line == "EVIDENCE":
            # Read past, not into. The round trip exists to prove the STATE survives the
            # rendering -- budget, ledger, OPEN, CLOSED, DEAD. Evidence is not board state:
            # it is bought from a provider and looked up again on replay, so parsing it back
            # would assert that a cache read is reproducible, which is a different claim and
            # not the one this check is for.
            section, cur = "evidence", None
            continue
        if line.startswith("CLOSED  "):
            section = None
            for item in line[len("CLOSED  "):].split(" · "):
                if mo := _CLOSED_ITEM.match(item.strip()):
                    view.closed[mo.group(1)] = float(mo.group(2)) if mo.group(2) else None
                else:
                    raise ValueError(f"unparsed CLOSED item: {item!r}")
            continue
        if line.startswith("DEAD    "):
            section = None
            for item in line[len("DEAD    "):].split(" · "):
                item = item.strip()
                if item.startswith("best "):
                    continue                    # the score belongs to the id before it
                view.dead.append(item.split()[0])
            continue
        if mo := _LEDGER.match(line):
            section = None
            ranked = [int(x) for x in _CN.findall(mo.group(7) or "")]
            was = mo.group(8)
            view.ledger.append(LedgerView(
                rid=mo.group(1), parent=mo.group(2), cand=int(mo.group(3)),
                closed=int(mo.group(4)) if mo.group(4) else None,
                total=int(mo.group(5)) if mo.group(5) else None,
                status=("failed" if "FAILED" in line else
                        "solved" if "· solved" in line else "open"),
                ranked=ranked,
                was=None if was in (None, "—") else [int(x) for x in _CN.findall(was)],
            ))
            continue
        if section == "open":
            if mo := _OPENL.match(line):
                extra = mo.group(5) or ""
                cur = OpenView(
                    mid=mo.group(1), under=mo.group(2), depth=int(mo.group(3)),
                    left=int(mo.group(4)), has_menu="no candidates yet" not in extra,
                    failed=[int(x) for x in _CN.findall(extra)] if "ranked" in extra else [],
                )
                view.open.append(cur)
                continue
            if mo := _CANDL.match(line):
                if cur is None:
                    raise ValueError(f"candidate outside a molecule: {line!r}")
                if mo.group(1) != mo.group(4):
                    raise ValueError(f"unbalanced candidate tag: {line!r}")
                pieces, buyable = [], []
                for part in mo.group(3).split(" + "):
                    part = part.strip()
                    star = "*" in part
                    buyable.append(star)
                    pieces.append(re.sub(r"\*(\(ln\$[\d.]+\))?$", "", part))
                cur.menu.append(MenuView(int(mo.group(1)), _signals(mo.group(2)),
                                         pieces, buyable))
                continue
            if mo := _TAIL.match(line):
                if cur is not None:
                    cur.hidden = int(mo.group(1))
                continue
            if mo := _MOLTAG.match(line):
                if cur is not None:
                    cur.smiles = mo.group(2)
                continue
            raise ValueError(f"unparsed line in OPEN: {line!r}")
        if section == "evidence":
            continue
        if section == "routes":
            view.routes.append(line)
            continue
        raise ValueError(f"unparsed line: {line!r}")
    return view


# ------------------------------------------------------------- the round trip
def check_roundtrip(b: Board, text: str, style=None) -> StateView:
    """Assert the rendered turn still contains everything the board knows."""
    view = parse_env(text)
    # Only when the style renders it. The budget header is off by default -- the call ceiling is
    # a constraint the agent is not meant to reason from, see RenderStyle.show_budget -- and a
    # round trip cannot assert the survival of a line that was never written.
    from . import render as _R
    if (style or _R.STYLE).show_budget:
        assert (view.budget_used, view.budget_max) == (b.budget_used, b.budget_max), \
            f"budget: {view.budget_used}/{view.budget_max} vs {b.budget_used}/{b.budget_max}"

    seen = {r.rid: r for r in view.ledger}
    assert list(seen) == b.rxn_order, f"ledger ids: {list(seen)} vs {b.rxn_order}"
    for rid, row in seen.items():
        r = b.rxns[rid]
        assert (row.parent, row.cand) == (r.parent, r.cand), f"{rid} parent/cand"
        assert row.status == r.status, f"{rid} status {row.status} vs {r.status}"
        if row.status != "failed":
            assert (row.closed, row.total) == (r.n_closed(b), len(r.pieces)), f"{rid} counts"
        m = b.mols[r.parent]
        if row.ranked or m.ranking:
            assert row.ranked == m.ranking, f"{rid} ranked {row.ranked} vs {m.ranking}"

    got = {o.mid: o for o in view.open}
    assert list(got) == b.open_mols(), f"OPEN ids: {list(got)} vs {b.open_mols()}"
    for mid, o in got.items():
        m = b.mols[mid]
        assert o.depth == m.depth and o.left == b.left(mid), f"{mid} depth/left"
        assert o.under == m.parent_rxn, f"{mid} under {o.under} vs {m.parent_rxn}"
        assert o.has_menu == (m.menu is not None), f"{mid} menu presence"
        if m.menu is not None:
            n_shown = len(o.menu)
            assert [c.idx for c in o.menu] == [c.idx for c in m.menu[:n_shown]], f"{mid} c-numbers"
            assert (o.hidden or 0) == len(m.menu) - n_shown, f"{mid} hidden count"
            for cv, c in zip(o.menu, m.menu):
                assert cv.pieces == c.reactants, f"{mid} c{c.idx} pieces"
                for k, v in c.signals.items():
                    assert abs(cv.signals.get(k, -9) - round(v, 3)) < 5e-4, \
                        f"{mid} c{c.idx} {k}: {cv.signals.get(k)} vs {v}"

    assert list(view.closed) == b.closed_leaves(), \
        f"CLOSED: {list(view.closed)} vs {b.closed_leaves()}"
    assert view.dead == b.dead_mols(), f"DEAD: {view.dead} vs {b.dead_mols()}"
    return view


def _selftest() -> None:
    """A hand-built two-step board: render, parse, and move it with parse_act."""
    from . import render as R
    from .state import Candidate

    MENU = {
        "T": [(["A", "B"], 0.62), (["C", "D"], 0.31), (["E"], 0.02)],
        "B": [(["F", "G"], 0.55), (["H"], 0.04)],
    }
    STOCK = {"A": 3.03, "F": 1.90, "G": None, "C": 2.5}

    class W:
        def menu(self, smi):
            return [Candidate(i, r, {"q": q}) for i, (r, q) in enumerate(MENU[smi])]

        def info(self, smi):
            return (smi in STOCK, STOCK.get(smi))

    b = Board("T", W(), max_depth=4, budget=20)
    st = R.RenderStyle(menu_show=2)
    root = b.root
    plan = [f"open {root}", f"rank {root} c0", "open {B}", "rank {B} c0"]
    for step in plan:
        act = step.format(B=b.mid_of("B") or "?")
        acts = parse_act(f"<act>{act}</act>")
        as_json = R.format_act_json(acts)
        back = parse_act_json(json.dumps(as_json))
        assert [(type(x).__name__, getattr(x, "mid", None), getattr(x, "order", None),
                 getattr(x, "reason", None)) for x in acts] == \
               [(type(x).__name__, getattr(x, "mid", None), getattr(x, "order", None),
                 getattr(x, "reason", None)) for x in back], \
            f"text and json action forms disagree on {act!r}"
        b.apply(acts)
        text = R.render_env(b, st)
        check_roundtrip(b, text)
    assert b.solved(), "the toy board should close"
    print(R.render_env(b, st))
    print("\nroundtrip ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
