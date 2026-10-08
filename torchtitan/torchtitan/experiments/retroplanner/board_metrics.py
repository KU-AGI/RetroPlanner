# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What went wrong with a predicted action, not just whether it was wrong.

``action/exact_match`` -- every token of the call is the argmax -- answers one
question and hides several.  A call can be wrong because the verb is wrong,
because it names a molecule that is not open, because two actions in one call are
in the wrong order, because the ranking's HEAD is wrong (the candidate actually
applied, which moves the tree) or only because its TAIL is (intent the board
merely records).  Those failures have different costs and different fixes, so
they are counted separately.

The two headline numbers:

``transition/exact``
    the whole payload matches the gold, so the board that comes back is
    byte-identical.  The environment is a function of the action, so this IS
    state-transition accuracy.

``transition/tree``
    the verbs, the molecules and every ranking's head match.  The AND-OR tree
    then evolves identically and only the ledger's recorded intent differs -- the
    model is in the same place, having declared a different fallback.

Everything is computed from teacher-forced argmax, so a number here is "would the
model have emitted this token given the gold prefix", not "would it have got here
on its own".  That is the same footing as ``acc/*``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch

from .board_trajectory import (
    BoardAction,
    board_menu_sizes,
    board_open_ids,
    parse_board_action,
)
from .trajectory import ROLE_DECISION

IGNORE_INDEX = -100


@dataclass
class Comparison:
    """One predicted call against its gold, field by field."""

    json_valid: bool
    exact: bool
    n_items_match: bool
    verbs_match: bool
    mids_set_match: bool
    mids_order_match: bool
    heads_match: bool
    orders_match: bool
    mid_hallucinated: bool | None = None
    """A named molecule is not in the board's OPEN block.  None when the board
    was not supplied, so it is never silently counted as a pass."""

    candidate_off_screen: bool | None = None

    @property
    def mid_misorder(self) -> bool:
        """Right molecules, wrong order -- a scheduling error, not a chemistry one."""
        return self.mids_set_match and not self.mids_order_match

    @property
    def rank_misorder(self) -> bool:
        """Right head, wrong tail: the same move, a different declared fallback."""
        return self.heads_match and not self.orders_match

    @property
    def rank_head_wrong(self) -> bool:
        """The applied candidate is wrong -- the tree moves somewhere else."""
        return not self.heads_match

    @property
    def rank_mismatch(self) -> bool:
        """ANY ranking that is not the gold one, head included.

        `rank_misorder` is CONDITIONAL on the head being right, so the call whose
        very first `order` entry is wrong -- the worst case, and the one that moves
        the tree -- is excluded from it. Read as an error rate it therefore reports
        a small number because the large failures are not in its numerator. This is
        the unconditional rate: invalid JSON, a different item count and a wrong
        head all count, so `1 - action/rank_order_match` and this agree, and a plot
        of it can be compared across arms without a second series to explain it.
        """
        return not self.orders_match

    @property
    def mid_mismatch(self) -> bool:
        """Any molecule list that is not the gold one, membership included.

        The counterpart of `mid_misorder`, which is conditional on the SET being
        right and so drops every call that named the wrong molecules outright.
        """
        return not self.mids_order_match

    @property
    def transition_tree(self) -> bool:
        return (self.json_valid and self.n_items_match and self.verbs_match
                and self.mids_order_match and self.heads_match)


def compare(gold: BoardAction, predicted_text: str,
            board: str | None = None) -> Comparison:
    """Compare one predicted call against the gold call it was scored against."""
    pred = parse_board_action(predicted_text)
    if pred is None:
        return Comparison(json_valid=False, exact=False, n_items_match=False,
                          verbs_match=False, mids_set_match=False,
                          mids_order_match=False, heads_match=False,
                          orders_match=False)

    n_ok = len(pred.items) == len(gold.items)
    verbs = pred.types == gold.types
    mids_order = pred.mids == gold.mids
    mids_set = Counter(pred.mids) == Counter(gold.mids)
    if n_ok:
        heads = all(p.head == g.head for p, g in zip(pred.items, gold.items))
        orders = all(p.order == g.order for p, g in zip(pred.items, gold.items))
    else:
        heads = orders = False

    halluc = off_screen = None
    if board is not None:
        open_ids = board_open_ids(board)
        menus = board_menu_sizes(board)
        halluc = any(i.mid is not None and i.mid not in open_ids for i in pred.items)
        off_screen = False
        for i in pred.items:
            size = menus.get(i.mid or "", 0)
            if size and any(c >= size for c in i.order):
                off_screen = True

    return Comparison(
        json_valid=True,
        exact=pred.payload() == gold.payload(),
        n_items_match=n_ok, verbs_match=verbs,
        mids_set_match=mids_set, mids_order_match=mids_order,
        heads_match=heads, orders_match=orders,
        mid_hallucinated=halluc, candidate_off_screen=off_screen,
    )


COUNTERS: tuple[tuple[str, str], ...] = (
    ("transition/exact", "exact"),
    ("transition/tree", "transition_tree"),
    ("action/json_valid", "json_valid"),
    ("action/verbs_match", "verbs_match"),
    ("action/n_items_match", "n_items_match"),
    ("action/mid_set_match", "mids_set_match"),
    ("action/mid_order_match", "mids_order_match"),
    ("action/mid_misorder", "mid_misorder"),
    ("action/rank_head_match", "heads_match"),
    ("action/rank_order_match", "orders_match"),
    ("action/rank_misorder", "rank_misorder"),
    # Appended, never reordered: these are the unconditional counterparts of the
    # two `*_misorder` series above, which exclude the head-wrong and set-wrong
    # calls and so understate the error rate they look like they report.
    ("action/rank_mismatch", "rank_mismatch"),
    ("action/mid_mismatch", "mid_mismatch"),
    ("action/rank_head_wrong", "rank_head_wrong"),
)
"""(logged name, Comparison attribute).  Order is the wire format of the count
vector the trainer all-reduces, so appending is safe and reordering is not."""


def count_vector(comparisons: Iterable[Comparison]) -> torch.Tensor:
    """Raw counts, plus the denominator last.

    Counts rather than rates because the trainer sums them ACROSS RANKS before
    dividing.  Averaging per-rank fractions would weight a rank that happened to
    see two actions the same as one that saw twenty -- and the capture is
    rank-local, so that is the normal case.
    """
    rows = list(comparisons)
    out = torch.zeros(len(COUNTERS) + 1, dtype=torch.float64)
    for i, (_, attr) in enumerate(COUNTERS):
        out[i] = float(sum(bool(getattr(r, attr)) for r in rows))
    out[-1] = float(len(rows))
    return out


def metrics_from_counts(vector: torch.Tensor) -> dict[str, float]:
    """Turn a (cross-rank reduced) count vector into loggable rates."""
    n = float(vector[-1].item())
    if n <= 0:
        return {}
    out = {"action/decoded": n}
    for i, (name, _) in enumerate(COUNTERS):
        out[name] = float(vector[i].item()) / n
    return out


def summarize(comparisons: Iterable[Comparison]) -> dict[str, float]:
    """Aggregate to the series that get logged.

    Rates are over the calls actually scored, and `action/decoded` says how many
    that was -- a rate over three calls is not a measurement, and without the
    count there is no way to see that from the chart.
    """
    rows = list(comparisons)
    n = len(rows)
    if n == 0:
        return {}
    # Derived from COUNTERS rather than restated. The two lists were separate and
    # a counter appended to one did not reach the other -- so a new series would be
    # all-reduced by the trainer and then dropped here, which reads as "the metric
    # is not logged" rather than as a bug.
    out = {"action/decoded": float(n)}
    for name, attr in COUNTERS:
        out[name] = sum(bool(getattr(r, attr)) for r in rows) / n
    graded = [r for r in rows if r.mid_hallucinated is not None]
    if graded:
        out["action/mid_hallucinated"] = (
            sum(r.mid_hallucinated for r in graded) / len(graded))
        out["action/candidate_off_screen"] = (
            sum(bool(r.candidate_off_screen) for r in graded) / len(graded))
    return out


class ActionCapture:
    """Collects the argmax and the gold ids of decision tokens, span by span.

    Armed only on steps that will be logged.  It copies the DECISION tokens of
    the microbatch to host -- a few hundred per episode, not the whole sequence
    -- so the sync it costs is bounded by the supervision budget rather than by
    seq_len.  Everything it needs is already in the loss: the gold call is the
    labels at those positions, so no gold text has to be plumbed alongside the
    batch.
    """

    def __init__(self) -> None:
        # (microbatch, chunk, span, position) -> (pred, gold).  All four are needed
        # in the key: span ids are numbered PER PACKED SEQUENCE, so every
        # microbatch reuses them, and so does every chunk's flat position.  Keying
        # on (span, pos) alone would let a later microbatch overwrite an earlier
        # one and silently throw away most of the actions; `action/decoded`
        # falling short of `action/count` is the symptom.
        self._rows: list[tuple[int, int, int, int, int, int]] = []
        self._microbatch = 0
        self.armed = False
        self.overflow = 0

    def arm(self, on: bool = True) -> None:
        self.armed = on

    def reset(self) -> None:
        self._rows.clear()
        self._microbatch = 0
        self.overflow = 0

    def end_microbatch(self) -> None:
        """Called where the accumulator's own end_microbatch is."""
        self._microbatch += 1

    @torch.no_grad()
    def add(self, *, role_ids: torch.Tensor, span_ids: torch.Tensor,
            argmax: torch.Tensor, labels: torch.Tensor, chunk: int = 0,
            max_tokens: int = 200_000) -> None:
        """Record this chunk's decision tokens.  Flat tensors, one entry per token."""
        if not self.armed:
            return
        keep = (role_ids == ROLE_DECISION) & (labels != IGNORE_INDEX) & (span_ids >= 0)
        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        if len(self._rows) + idx.numel() > max_tokens:
            self.overflow += int(idx.numel())
            return
        block = torch.stack(
            (span_ids[idx], idx, argmax[idx], labels[idx]), dim=1
        ).to("cpu", non_blocking=False)
        mb, ch = self._microbatch, chunk
        self._rows.extend(
            (mb, ch, int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            for row in block
        )

    def decode(self, tokenizer) -> list[tuple[str, str]]:
        """(gold_text, predicted_text) per span, in the order the tokens appeared."""
        # Activation checkpointing recomputes a chunk in backward, so the same
        # (span, position) can arrive twice.  Keyed rather than appended, so a
        # recomputed token overwrites instead of duplicating -- appending would
        # decode every action twice over and score the doubled string as wrong.
        cells: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        for mb, chunk, span, pos, pred, gold in self._rows:
            cells[(mb, chunk, span, pos)] = (pred, gold)
        # One action is (microbatch, span): it may straddle chunk boundaries, and
        # grouping without the chunk reunites it.
        by_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for (mb, chunk, span, pos), (pred, gold) in cells.items():
            by_span.setdefault((mb, span), []).append((chunk, pos, pred, gold))
        out: list[tuple[str, str]] = []
        for span in sorted(by_span):
            rows = [(p, pr, g) for _, p, pr, g in sorted(by_span[span])]
            rows = sorted(rows)
            gold = tokenizer.decode([g for _, _, g in rows])
            pred = tokenizer.decode([p for _, p, _ in rows])
            out.append((gold, pred))
        return out

    def comparisons(self, tokenizer) -> list[Comparison]:
        """Decode and compare.  The board is not available here, so the
        hallucination checks stay ungraded -- run them in the validator, which has
        the observation, or in prepare_board_dataset over the corpus."""
        rows = []
        for gold_text, pred_text in self.decode(tokenizer):
            gold = parse_board_action(gold_text)
            if gold is None:
                # The gold is the label stream; if that does not parse, the span
                # extraction is broken, not the model.
                rows.append(Comparison(False, False, False, False, False, False,
                                       False, False))
                continue
            rows.append(compare(gold, pred_text))
        return rows

    def counts(self, tokenizer) -> torch.Tensor:
        """The count vector for this rank, ready to be summed across ranks."""
        return count_vector(self.comparisons(tokenizer))

    def summarize(self, tokenizer) -> dict[str, float]:
        """Rank-local rates.  Used by tests and by anything single-process."""
        metrics = summarize(self.comparisons(tokenizer))
        if self.overflow:
            metrics["action/capture_overflow"] = float(self.overflow)
        return metrics
