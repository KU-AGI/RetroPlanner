# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parsing and validation for retrosynthesis AND-OR-tree agent trajectories.

A raw record (one JSON line of ``conv_*.oss.jsonl``) holds a full multi-turn
episode: a system prompt, alternating user "state" messages and assistant
``<think>...</think>\\n<act>...</act>`` messages, plus the ground-truth
``actions`` list, the molecule table ``mols`` and the committed search
``tree``.

This module answers three questions about each record, which are the three
things worth logging before a single training token is consumed:

1. format      -- does every assistant message have the exact think/act shape
                  and does the action parse against the action grammar?
2. gold        -- does the committed disconnection the assistant ranked first
                  actually match the children recorded in the search tree?
                  ("gold adoption": if it does not, the trajectory teaches an
                  action that contradicts its own supervision target.)
3. state       -- is every user message the correct successor state of the
                  previous action (turn index, expansion budget, the echo of
                  an expansion, the pieces opened by a commit)?

Issue codes are stable strings so they can be counted, logged and diffed
across dataset versions.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Role ids carried alongside every supervised label token. Kept as small ints
# so they survive the dataloader / chunked-loss path as a plain int tensor.
ROLE_IGNORE = 0
ROLE_REASONING = 1
ROLE_DECISION = 2
ROLE_FORMAT = 3
ROLE_TERMINATE = 4
"""The stop signal: the assistant's `final` channel, carrying the route it hands
over.  Emitting it IS the decision to stop -- in Harmony a tool call promises a
result, so choosing to answer instead of calling again is how an agent terminates.

Named `terminate` rather than `done` because `done` is already an action type in
the board_act schema; `loss/done` would not say which of the two it meant.  It is
separate from ROLE_DECISION so that `action/exact_match` stays a statement about
the tool calls: this target is much the longest in an episode, and folding it in
would dilute the one number that says whether the model picks
the same disconnection."""

ROLE_NAMES: dict[int, str] = {
    ROLE_TERMINATE: "terminate",
    ROLE_REASONING: "reasoning",
    ROLE_DECISION: "decision",
    ROLE_FORMAT: "format",
}

# Assistant message shape. Anchored: anything else is a format violation.
TURN_RE = re.compile(r"\A<think>(.*?)</think>\n<act>(.*?)</act>\Z", re.DOTALL)

# Action grammar.
ACT_SINGLE_RE = re.compile(r"\Asingle (\S+)\Z")
ACT_RANK_RE = re.compile(r"\Arank (\S+) (c\d+(?:,c\d+)*)\Z")
ACT_DONE_RE = re.compile(r"\Adone\Z")

# User "state" message fragments.
# The remaining budget is allowed to be negative in the raw data: a trajectory
# can overspend its expansion allowance, and that is a state defect we want to
# see rather than a parse failure.
TURN_HEADER_RE = re.compile(
    r"^Turn (\d+)\. Expansion budget: (-?\d+) of (-?\d+) left\.$", re.MULTILINE
)
EXPANSION_ECHO_RE = re.compile(
    r"\A<mol (\S+)/> came back with (\d+) candidate disconnections\."
)
READY_TO_RANK_RE = re.compile(
    r"^  <mol (\S+)/> [^\n]*?(\d+) candidates:$", re.MULTILINE
)
CANDIDATE_RE = re.compile(r"^    <c(\d+)>(.*?)</c\1>$", re.MULTILINE)
ROUTE_OPEN_RE = re.compile(
    r"^Route \*\*(\S+)\*\* under <mol (\S+)/> is open", re.MULTILINE
)
# A molecule introduced with its SMILES, or referred to by bare id.
MOL_DECL_RE = re.compile(r"<mol (\S+?)>(.*?)</mol>")
MOL_REF_RE = re.compile(r"<mol (\S+?)/>")
NOTHING_OPEN_RE = re.compile(r"^Nothing is open\.$", re.MULTILINE)


@dataclass(frozen=True)
class Action:
    """One parsed assistant action."""

    kind: str  # "single" | "rank" | "done"
    mol: str | None = None
    choices: tuple[int, ...] = ()

    def render(self) -> str:
        if self.kind == "done":
            return "done"
        if self.kind == "single":
            return f"single {self.mol}"
        return f"rank {self.mol} " + ",".join(f"c{c}" for c in self.choices)


def parse_action(text: str) -> Action | None:
    """Parse the contents of an ``<act>`` tag, or return None if ungrammatical."""
    if ACT_DONE_RE.match(text):
        return Action(kind="done")
    m = ACT_SINGLE_RE.match(text)
    if m:
        return Action(kind="single", mol=m.group(1))
    m = ACT_RANK_RE.match(text)
    if m:
        choices = tuple(int(c[1:]) for c in m.group(2).split(","))
        return Action(kind="rank", mol=m.group(1), choices=choices)
    return None


@dataclass
class Turn:
    """One (state, reasoning, decision) triple of a trajectory."""

    index: int
    user: str
    think: str
    act: str
    action: Action | None


@dataclass
class Trajectory:
    """A raw record reshaped into the fields the tokenizer needs."""

    developer: str
    """The record's system message. Rendered into the harmony developer turn."""

    turns: list[Turn]
    route_idx: int
    combo_idx: int
    benchmark: str


@dataclass
class ValidationReport:
    """Per-record validation outcome."""

    route_idx: int
    combo_idx: int
    n_turns: int
    n_expansions: int
    issues: list[str] = field(default_factory=list)
    n_decisions: int = 0
    n_gold_checked: int = 0
    n_gold_adopted: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, code: str) -> None:
        # Keep one entry per code per record so a single malformed turn does not
        # dominate the aggregate counts.
        if code not in self.issues:
            self.issues.append(code)


def parse_candidates(user_msg: str) -> dict[str, dict[int, list[str]]]:
    """Return ``{mol_id: {candidate_index: [fragment_smiles, ...]}}``.

    Only the block that follows a "Ready to rank" header is parsed. A trailing
    ``*`` on a fragment marks it as already buyable and is stripped.
    """
    result: dict[str, dict[int, list[str]]] = {}
    headers = list(READY_TO_RANK_RE.finditer(user_msg))
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(user_msg)
        block = user_msg[header.end() : end]
        cands: dict[int, list[str]] = {}
        for m in CANDIDATE_RE.finditer(block):
            frags = [f.strip().rstrip("*") for f in m.group(2).split(" + ")]
            cands[int(m.group(1))] = frags
        result[header.group(1)] = cands
    return result


def open_molecule_ids(user_msg: str) -> set[str]:
    """Ids listed under "Needs an expansion" in a state message."""
    marker = "Needs an expansion"
    pos = user_msg.find(marker)
    if pos < 0:
        return set()
    block = user_msg[pos:]
    return set(MOL_REF_RE.findall(block)) | {
        m.group(1) for m in MOL_DECL_RE.finditer(block)
    }


def _piece_ids(route_block: str) -> list[str]:
    """Molecule ids of the pieces announced by a "Route ... is open" block."""
    ids = [m.group(1) for m in MOL_DECL_RE.finditer(route_block)]
    ids += MOL_REF_RE.findall(route_block)
    # The first reference in the block is the parent molecule being opened.
    header = ROUTE_OPEN_RE.search(route_block)
    parent = header.group(2) if header else None
    return [i for i in ids if i != parent]


def to_trajectory(record: dict[str, Any]) -> Trajectory:
    """Reshape a raw record into a Trajectory. Does not validate."""
    messages = record["messages"]
    developer = messages[0]["content"] if messages[0]["role"] == "system" else ""
    body = messages[1:] if messages[0]["role"] == "system" else messages

    turns: list[Turn] = []
    for i in range(0, len(body) - 1, 2):
        user_msg, asst_msg = body[i], body[i + 1]
        if user_msg["role"] != "user" or asst_msg["role"] != "assistant":
            break
        m = TURN_RE.match(asst_msg["content"])
        think, act = (m.group(1), m.group(2)) if m else ("", "")
        turns.append(
            Turn(
                index=len(turns),
                user=user_msg["content"],
                think=think,
                act=act,
                action=parse_action(act) if m else None,
            )
        )
    return Trajectory(
        developer=developer,
        turns=turns,
        route_idx=record.get("route_idx", -1),
        combo_idx=record.get("combo_idx", -1),
        benchmark=record.get("benchmark", ""),
    )


def identity_canonicalizer(smiles: str) -> str:
    """Default fragment comparison: exact string match."""
    return smiles


def inchikey14_canonicalizer() -> "Callable[[str], str]":
    """Compare fragments by the skeleton block of their InChIKey.

    The distilled trajectories and the search tree disagree on tautomers -- a
    2-pyridone is written as a 2-hydroxypyridine in one and not the other --
    so a raw string comparison reports gold-adoption failures that are only
    a difference in how the same molecule was written. The first InChIKey
    block ignores the tautomer/protonation layer, which is the comparison we
    actually want. Requires RDKit; raises if it is not installed.
    """
    from rdkit import Chem, RDLogger

    # pyrefly: ignore[missing-attribute]
    RDLogger.DisableLog("rdApp.*")
    cache: dict[str, str] = {}

    def canon(smiles: str) -> str:
        hit = cache.get(smiles)
        if hit is not None:
            return hit
        mol = Chem.MolFromSmiles(smiles)
        # Unparsable fragments fall back to their literal string so they can
        # still match an identical string on the other side.
        key = Chem.MolToInchiKey(mol).split("-")[0] if mol is not None else smiles
        cache[smiles] = key
        return key

    return canon


def validate_record(
    record: dict[str, Any],
    *,
    canonicalize: "Callable[[str], str]" = identity_canonicalizer,
) -> ValidationReport:
    """Run the format / gold / state checks over one raw record.

    ``canonicalize`` maps a fragment SMILES to the token used when comparing a
    committed candidate against the search tree.
    """
    messages = record["messages"]
    mols: dict[str, str] = record.get("mols", {})
    tree: dict[str, Any] = record.get("tree", {})
    gold_actions: list[str] = record.get("actions", [])

    report = ValidationReport(
        route_idx=record.get("route_idx", -1),
        combo_idx=record.get("combo_idx", -1),
        n_turns=record.get("n_turns", -1),
        n_expansions=record.get("n_expansions", -1),
    )

    if not messages or messages[0]["role"] != "system":
        report.add("format/no_system_message")

    traj = to_trajectory(record)
    turns = traj.turns

    # Every message must have been consumed by exactly one turn.
    body_len = len(messages) - (
        1 if messages and messages[0]["role"] == "system" else 0
    )
    if body_len != 2 * len(turns):
        report.add("format/message_role_sequence")

    # ---- format ----------------------------------------------------------
    for turn in turns:
        if turn.action is None:
            report.add("format/assistant_shape_or_grammar")
        elif not turn.think.strip():
            report.add("format/think_empty")

    if record.get("n_turns", len(turns)) != len(turns):
        report.add("format/n_turns_mismatch")

    # ---- action list agreement -------------------------------------------
    rendered = [t.action.render() for t in turns if t.action is not None]
    if len(rendered) == len(gold_actions):
        for got, want in zip(rendered, gold_actions, strict=True):
            if got != want:
                report.add("action/mismatch_actions_field")
                break
    else:
        report.add("action/mismatch_actions_field")

    n_single = sum(1 for t in turns if t.action and t.action.kind == "single")
    if record.get("n_expansions", n_single) != n_single:
        report.add("action/n_expansions_mismatch")

    if not turns or turns[-1].action is None or turns[-1].action.kind != "done":
        report.add("gold/unfinished")
    for turn in turns[:-1]:
        if turn.action is not None and turn.action.kind == "done":
            report.add("state/done_not_last")

    # ---- state transitions ------------------------------------------------
    edges = tree.get("edges", [])
    children: dict[str, list[str]] = {}
    for parent, child, _route in edges:
        children.setdefault(parent, []).append(child)
    nodes: dict[str, str] = tree.get("nodes", {})

    prev_budget: int | None = None
    for turn in turns:
        header = TURN_HEADER_RE.search(turn.user)
        if header is None:
            report.add("state/turn_header_missing")
            continue
        if int(header.group(1)) != turn.index:
            report.add("state/turn_index")
        budget = int(header.group(2))
        if budget < 0:
            report.add("state/budget_overspent")
        if prev_budget is not None:
            prev_action = turns[turn.index - 1].action
            expected = prev_budget - (
                1 if prev_action is not None and prev_action.kind == "single" else 0
            )
            if budget != expected:
                report.add("state/budget")
        prev_budget = budget

    for turn in turns:
        action = turn.action
        if action is None:
            continue
        nxt = turns[turn.index + 1] if turn.index + 1 < len(turns) else None

        if action.kind == "single":
            if action.mol not in open_molecule_ids(turn.user):
                report.add("state/expand_target_not_open")
            if nxt is not None:
                echo = EXPANSION_ECHO_RE.match(nxt.user)
                if echo is None or echo.group(1) != action.mol:
                    report.add("state/expansion_not_echoed")
            else:
                report.add("state/expansion_without_successor")

        elif action.kind == "rank":
            report.n_decisions += 1
            cands = parse_candidates(turn.user).get(action.mol or "")
            if cands is None:
                report.add("state/rank_target_not_ready")
                continue
            top = action.choices[0] if action.choices else None
            if top is None or top not in cands:
                report.add("state/candidate_missing")
                continue
            chosen = sorted(canonicalize(f) for f in cands[top])

            # Gold adoption: the fragments of the committed candidate must be
            # exactly the children the search tree recorded for this molecule.
            # The three near-misses are split out because only the last one
            # means the supervised action disagrees with the route it builds:
            #   - the tree repeats a reagent the candidate lists once
            #   - the tree carries a piece the candidate does not spell out
            #   - the two sides name different molecules
            report.n_gold_checked += 1
            tree_children = sorted(
                canonicalize(mols.get(child, child))
                for child in children.get(action.mol or "", [])
            )
            if chosen == tree_children:
                report.n_gold_adopted += 1
            elif set(chosen) == set(tree_children):
                report.add("gold/duplicate_piece")
            elif set(chosen) < set(tree_children):
                report.add("gold/tree_has_extra_children")
            else:
                report.add("gold/committed_children_mismatch")

            if nxt is not None:
                route = ROUTE_OPEN_RE.search(nxt.user)
                if route is None or route.group(2) != action.mol:
                    report.add("state/route_not_opened")
                else:
                    block = nxt.user[route.start() :]
                    stop = block.find("\n\nTurn ")
                    if stop >= 0:
                        block = block[:stop]
                    # Pieces already seen in an earlier turn appear as a bare
                    # ``<mol id/>`` reference, so resolve ids through the
                    # molecule table instead of reading SMILES off the message.
                    announced = sorted(
                        canonicalize(mols.get(mol_id, mol_id))
                        for mol_id in _piece_ids(block)
                    )
                    if announced and announced != sorted(tree_children):
                        report.add("state/route_pieces_mismatch")
            else:
                report.add("state/rank_without_successor")

        else:  # done
            unresolved = [
                mol
                for mol, status in nodes.items()
                if status not in ("buyable", "committed")
            ]
            if unresolved:
                report.add("gold/done_premature")
            leaves = [m for m in nodes if m not in children]
            if any(nodes.get(leaf) != "buyable" for leaf in leaves):
                report.add("gold/done_leaf_not_buyable")
            if NOTHING_OPEN_RE.search(turn.user) is None:
                report.add("state/done_while_open")

    return report


def aggregate(reports: list[ValidationReport]) -> dict[str, Any]:
    """Aggregate per-record reports into countable statistics."""
    issue_counts: Counter[str] = Counter()
    for r in reports:
        issue_counts.update(r.issues)
    n = len(reports)
    n_ok = sum(1 for r in reports if r.ok)
    gold_checked = sum(r.n_gold_checked for r in reports)
    gold_adopted = sum(r.n_gold_adopted for r in reports)
    return {
        "records": n,
        "records_clean": n_ok,
        "records_clean_frac": (n_ok / n) if n else 0.0,
        "decisions": sum(r.n_decisions for r in reports),
        "gold_checked": gold_checked,
        "gold_adopted": gold_adopted,
        "gold_adoption_rate": (gold_adopted / gold_checked) if gold_checked else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
    }
