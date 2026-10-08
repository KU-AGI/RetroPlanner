# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The board format: one episode of tool calls against the retrosynthesis board.

This is the successor to :mod:`trajectory`, whose records were
``<think>…</think>\\n<act>single 4f</act>`` in the ``final`` channel.  A board
episode is a Harmony **tool-call** conversation instead:

    user                     the opening board
    assistant -> board_act   {"actions":[{"type":"rank","mid":"vw","order":[1,0,4]}]}
    functions.board_act      the board that results
    ...
    assistant                the route it hands over            (final channel)

Two consequences for supervision.  The decision is now a JSON object, so a wrong
answer can be wrong in several distinguishable ways -- the verb, the molecule, the
order of a multi-action call, the head of a ranking (the candidate actually
applied) versus its tail (intent the board only records) -- and
:mod:`board_metrics` scores each separately.  And the observation arrives on the
``commentary`` channel from ``functions.board_act``, which the model must learn to
read but never writes.

Records come from ``tools/reaction-mcp/scripts/render_board_episode.py --format
harmony``.  The fields used here:

    harmony_messages   the Harmony-native turns (channel / recipient / content),
                       which is what vLLM renders at inference
    raw_text           the apply_chat_template rendering.  The system and
                       developer preambles are LIFTED from it rather than rebuilt,
                       so the tool namespace the model trains on is byte-identical
                       to the one it is served.
    supervised         per-turn flag; a turn produced by the rollout rather than
                       the labeller carries no loss
    route              which selected route this instance follows, and its axes
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .trajectory import ROLE_DECISION, ROLE_FORMAT, ROLE_IGNORE  # noqa: F401

SYSTEM_RE = re.compile(r"<\|start\|>system<\|message\|>(.*?)<\|end\|>", re.DOTALL)
DEVELOPER_RE = re.compile(r"<\|start\|>developer<\|message\|>(.*?)<\|end\|>", re.DOTALL)

# The frontier the model is allowed to name, straight off the board it was shown.
OPEN_BLOCK_RE = re.compile(r"^OPEN\n(.*?)(?:\n\n|\Z)", re.DOTALL | re.MULTILINE)
LEDGER_RE = re.compile(r"^r\d+  (\S+)·c\d+  ", re.MULTILINE)
OPEN_ID_RE = re.compile(r"^  (\S+)\s+(?:under r\d+ · )?depth \d+", re.MULTILINE)
CAND_RE = re.compile(r"^    <c(\d+)[ >]", re.MULTILINE)

TOOL_NAME = "board_act"
ACTION_TYPES = ("open", "rank", "dead", "done")


@dataclass(frozen=True)
class Item:
    """One action inside a call.  `order` is empty for anything but a rank."""

    type: str
    mid: str | None = None
    order: tuple[int, ...] = ()
    reason: str | None = None

    @property
    def head(self) -> int | None:
        """The candidate actually applied -- the only part that moves the tree."""
        return self.order[0] if self.order else None


@dataclass(frozen=True)
class BoardAction:
    """One ``board_act`` call: a sequence of items, applied in the order given."""

    items: tuple[Item, ...]

    @property
    def mids(self) -> tuple[str | None, ...]:
        return tuple(i.mid for i in self.items)

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(i.type for i in self.items)

    def payload(self) -> dict:
        out = []
        for i in self.items:
            d: dict[str, Any] = {"type": i.type}
            if i.mid is not None:
                d["mid"] = i.mid
            if i.order:
                d["order"] = list(i.order)
            if i.reason is not None:
                d["reason"] = i.reason
            out.append(d)
        return {"actions": out}

    def render(self) -> str:
        """The exact JSON the dataset trains on -- key order included.

        `sort_keys` stays off and the separators stay at json.dumps' defaults
        because that is what the builder wrote and what the chat template
        reproduces; a different spacing here would train the model to emit a
        string the board's own parser accepts but the tokenizer never saw.
        """
        return json.dumps(self.payload(), ensure_ascii=False)


def parse_board_action(text: str) -> BoardAction | None:
    """Parse a ``board_act`` argument, or None if it is not a legal call.

    Deliberately strict in the same places the environment is: an unknown verb,
    a missing molecule, a repeated candidate and a non-integer candidate are all
    rejected, because at inference each of them is a turn the board refuses.
    """
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        return None
    items: list[Item] = []
    for a in actions:
        if not isinstance(a, dict):
            return None
        kind = a.get("type")
        if kind not in ACTION_TYPES:
            return None
        mid = a.get("mid")
        if kind != "done" and not isinstance(mid, str):
            return None
        order = a.get("order") or []
        if not isinstance(order, list) or any(not isinstance(x, int) for x in order):
            return None
        if len(set(order)) != len(order):
            return None
        if kind == "rank" and not order:
            return None
        reason = a.get("reason")
        items.append(Item(type=kind, mid=mid if kind != "done" else None,
                          order=tuple(order),
                          reason=reason if isinstance(reason, str) else None))
    return BoardAction(items=tuple(items))


def board_open_ids(board: str) -> set[str]:
    """The molecule ids listed under OPEN."""
    m = OPEN_BLOCK_RE.search(board)
    if not m:
        return set()
    return set(OPEN_ID_RE.findall("\n" + m.group(1)))


def board_actionable_ids(board: str) -> set[str]:
    """Every id a call may legally name: OPEN plus the ledger's own molecules.

    OPEN alone is too narrow, and the difference is the whole continuation.  Once a
    route closes, the molecule the agent goes back to is CLOSED and so not in OPEN
    -- but its ledger row is on screen with the ordering it declared
    (`ye ranked: c4 · c0 · c8`), which is where the untried candidates come from.
    The board accepts a re-rank there; a validator that did not would report every
    continuation as a call on a molecule that is not open.
    """
    return board_open_ids(board) | set(LEDGER_RE.findall(board))


def board_menu_sizes(board: str) -> dict[str, int]:
    """id -> how many candidates are ON SCREEN for it.

    A ranking may only name what the board showed, so this is what turns "c7 on
    a five-line menu" from a ranking mistake into a hallucination.
    """
    m = OPEN_BLOCK_RE.search(board)
    if not m:
        return {}
    out: dict[str, int] = {}
    current: str | None = None
    for line in m.group(1).splitlines():
        head = OPEN_ID_RE.match("\n" + line) or OPEN_ID_RE.match(line)
        if line.startswith("  ") and not line.startswith("    "):
            ids = OPEN_ID_RE.findall("\n" + line)
            current = ids[0] if ids else None
            out.setdefault(current, 0) if current else None
        elif current and line.startswith("    <c"):
            out[current] = out.get(current, 0) + 1
    return out


@dataclass
class BoardTurn:
    """One (observation, call) pair, or the closing stop message."""

    index: int
    obs: str | None
    """The board.  None on the terminating turn, which follows a board already shown."""

    obs_kind: str = "tool"          # "user" for the opening board, else "tool"
    think: str | None = None        # analysis channel, when the set carries one
    payload: str | None = None      # the JSON of a board_act call
    action: BoardAction | None = None
    final: str | None = None        # the stop message, on the last turn
    supervised: bool = True
    """False for a turn the ROLLOUT produced -- a deliberate mistake or a doomed
    probe.  It stays in the sequence so the conversation is coherent, and out of
    the loss so the blunder is not taught."""


@dataclass
class BoardTrajectory:
    system: str
    developer: str
    turns: list[BoardTurn]
    target: str = ""
    route_hash: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return sum(1 for t in self.turns if t.payload is not None)


def to_board_trajectory(record: dict[str, Any]) -> BoardTrajectory:
    """Reshape one built record into the streams the tokenizer needs."""
    # The preamble is LIFTED, never rebuilt -- and it must be the one the
    # serving stack produces.  vLLM builds it with openai_harmony, whose tool
    # namespace differs from the HF chat template's, so `wire_system`/`wire_developer` (written by verify_harmony.py
    # --augment) win over anything extracted from raw_text.
    system = record.get("wire_system")
    developer = record.get("wire_developer")
    if system is None or developer is None:
        raw = record.get("raw_text") or ""
        sys_m = SYSTEM_RE.search(raw)
        dev_m = DEVELOPER_RE.search(raw)
        if not sys_m or not dev_m:
            raise ValueError(
                "no wire_system/wire_developer and no raw_text to fall back on. "
                "Run verify_harmony.py --augment over the built corpus: the "
                "preamble has to be the one vLLM renders, or the model trains on "
                "a tool namespace it is never served."
            )
        system, developer = sys_m.group(1), dev_m.group(1)
    msgs = record.get("harmony_messages")
    if not msgs:
        raise ValueError("record has no harmony_messages")

    turns: list[BoardTurn] = []
    pending_obs: str | None = None
    pending_kind = "tool"
    pending_think: str | None = None
    for m in msgs:
        role = m.get("role")
        if role in ("user", "tool"):
            content = m.get("content", "")
            # A refusal is not a board.  The board answers a refused action by
            # leaving itself unchanged and saying why, so the state the NEXT
            # action is judged against is still the last real board -- overwriting
            # it with the rejection line made every following mid look absent
            # and, worse, would hand the model's state
            # checks a string with no molecules in it.
            if content.startswith("rejected:"):
                continue
            pending_obs = content
            pending_kind = "user" if role == "user" else "tool"
            continue
        if role != "assistant":
            raise ValueError(f"unexpected role {role!r} in harmony_messages")
        if m.get("channel") == "analysis":
            pending_think = m.get("content")
            continue
        if m.get("recipient"):
            payload = m.get("content", "")
            turns.append(BoardTurn(
                index=len(turns), obs=pending_obs, obs_kind=pending_kind,
                think=pending_think, payload=payload,
                action=parse_board_action(payload),
                supervised=bool(m.get("supervised", True)),
            ))
        else:
            turns.append(BoardTurn(
                index=len(turns), obs=pending_obs, obs_kind=pending_kind,
                think=pending_think, final=m.get("content", ""),
                supervised=bool(m.get("supervised", True)),
            ))
        pending_obs = None
        pending_think = None

    route = record.get("route") or {}
    return BoardTrajectory(
        system=system,
        developer=developer,
        turns=turns,
        target=record.get("target", ""),
        route_hash=route.get("set_hash"),
        meta={"n_steps": route.get("n_steps"), "lls": route.get("lls"),
              "plaus_min": route.get("plaus_min"), "cost_usd": route.get("cost_usd"),
              "pareto_rank": route.get("pareto_rank"),
              "solved": record.get("solved"), "max_depth": record.get("max_depth")},
    )


# ------------------------------------------------------------------ validation
def validate_board_record(record: dict[str, Any]) -> dict[str, int]:
    """Pre-flight counts for one record.  Every key is a failure except `n_*`.

    Run over the corpus before training: these are the conditions that make a
    turn unlearnable rather than merely hard, and each one is silent at training
    time -- the loss on a call that names a molecule the board never showed looks
    exactly like the loss on a call that names the wrong one.
    """
    out: dict[str, int] = {}

    def bump(key: str, n: int = 1) -> None:
        out[key] = out.get(key, 0) + n

    try:
        traj = to_board_trajectory(record)
    except ValueError:
        bump("record/unparsed")
        return out

    bump("n_records")
    bump("n_turns", len(traj.turns))
    last_board: str | None = None
    for turn in traj.turns:
        if turn.obs is not None:
            last_board = turn.obs
        if turn.final is not None:
            bump("n_final")
            # "ROUTES" is one renderer's header, not the thing being checked.
            # The board has five handover forms and only some print that word, so
            # keying on it reported every flow-rendered episode as broken.  What
            # the check is for is that the last message actually hands routes
            # over, and every form says so in the same sentence.
            handed = re.search(r"Handing over (\d+) route", turn.final)
            if not handed or int(handed.group(1)) < 1:
                bump("final/no_route")
            continue
        bump("n_calls")
        if turn.action is None:
            bump("action/unparsed")
            continue
        bump("n_items", len(turn.action.items))
        if not turn.supervised:
            bump("n_unsupervised")
            # An unsupervised turn is a REFUSED action -- a rejection injected so
            # the corpus contains recovery.  Being illegal against the board it
            # was shown is the whole point of it, so checking it here reports the
            # injection itself as corruption ("mid_not_open").  It carries no
            # loss, so nothing downstream
            # depends on it being well formed.
            continue
        if last_board is None:
            bump("state/no_board")
            continue
        open_ids = board_actionable_ids(last_board)
        menus = board_menu_sizes(last_board)
        for item in turn.action.items:
            if item.type == "done":
                continue
            if item.mid not in open_ids:
                bump("state/mid_not_open")
            if item.order:
                size = menus.get(item.mid or "", 0)
                if size and any(c >= size for c in item.order):
                    bump("state/candidate_off_screen")
        mids = [i.mid for i in turn.action.items if i.mid]
        if len(set(mids)) != len(mids):
            bump("action/two_actions_one_molecule")
    return out


def aggregate_validation(reports: Iterable[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for rep in reports:
        for k, v in rep.items():
            total[k] = total.get(k, 0) + v
    return total
