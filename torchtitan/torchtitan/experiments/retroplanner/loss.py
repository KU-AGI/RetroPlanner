# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-entropy that reports itself split by the role each token plays.

The training loss is unchanged: it is the same sum-reduced cross-entropy over
the same labels that ``CrossEntropyLoss`` computes, so numerics and gradients
match the stock path token for token. What is added is bookkeeping, read off
the per-token NLL that the sum reduction is built from anyway:

    reasoning   analysis-channel tokens (the chain of thought)
    decision    final-channel tokens (the action)
    format      harmony channel headers and terminators

One cross-entropy per role, plus teacher-forced top-1 accuracy per role, plus
whole-action exact match. A run whose total loss falls while decision loss
does not is learning to narrate rather than to choose, and a single loss curve
cannot tell you that.

``ChunkedLossWrapper`` calls this once per sequence chunk, so the accumulator
sums across chunks; the trainer folds each microbatch and reduces across data
parallel ranks at logging time.
"""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torch.utils.checkpoint import checkpoint

from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.config import CompileConfig
from torchtitan.distributed.utils import get_spmd_backend

from .board_metrics import ActionCapture
from .trajectory import ROLE_NAMES

_NUM_ROLES = max(ROLE_NAMES) + 1


def per_token_cross_entropy(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token NLL, ``[B * L]``, zero at ignored positions."""
    return F.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )


class RoleMetricAccumulator:
    """Device-side running totals for role and action metrics.

    Lives across chunks and microbatches and is drained once per logged step.
    Everything stays on device until the drain so the hot path adds no
    host-device syncs.
    """

    def __init__(
        self,
        *,
        max_spans: int,
        num_slots: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        self.max_spans = max_spans
        self.num_slots = num_slots
        opts = {"dtype": torch.float64, "device": device}
        self.role_ce = torch.zeros(_NUM_ROLES, **opts)
        self.role_tokens = torch.zeros(_NUM_ROLES, **opts)
        self.role_correct = torch.zeros(_NUM_ROLES, **opts)
        # [num_correct_actions, num_actions]
        self.actions = torch.zeros(2, **opts)
        # Per-microbatch scratch, one row per sequence chunk. A row is
        # overwritten rather than added to, so a chunk that is executed twice
        # -- which is exactly what activation checkpointing does when it
        # recomputes the chunk in backward -- contributes once, not twice.
        # Rows are summed into the totals at end_microbatch.
        self._slot_ce = torch.zeros(num_slots, _NUM_ROLES, **opts)
        self._slot_tokens = torch.zeros(num_slots, _NUM_ROLES, **opts)
        self._slot_correct = torch.zeros(num_slots, _NUM_ROLES, **opts)
        # Span counts stay per chunk until the fold because one action can
        # straddle a chunk boundary; summing rows first reunites it.
        self._span_correct = torch.zeros(num_slots, max_spans, **opts)
        self._span_tokens = torch.zeros(num_slots, max_spans, **opts)
        self._device = torch.device(device)

    def ensure_slots(self, num_slots: int) -> None:
        """Grow to one scratch row per sequence chunk.

        Rows are overwritten, not added to, so too few rows would silently
        drop every chunk but the last rather than fail.
        """
        if num_slots <= self.num_slots:
            return
        self.num_slots = num_slots
        opts = {"dtype": torch.float64, "device": self._device}
        self._slot_ce = torch.zeros(num_slots, _NUM_ROLES, **opts)
        self._slot_tokens = torch.zeros(num_slots, _NUM_ROLES, **opts)
        self._slot_correct = torch.zeros(num_slots, _NUM_ROLES, **opts)
        self._span_correct = torch.zeros(num_slots, self.max_spans, **opts)
        self._span_tokens = torch.zeros(num_slots, self.max_spans, **opts)

    def ensure_capacity(self, num_spans: int) -> None:
        """Grow the span scratch so ``num_spans`` distinct spans fit.

        Span ids are numbered per packed sequence, so a batch of B sequences
        needs B times the per-sequence capacity or two different actions would
        share a slot and be scored as one.
        """
        if num_spans <= self.max_spans:
            return
        self.max_spans = num_spans
        opts = {"dtype": torch.float64, "device": self._device}
        self._span_correct = torch.zeros(self.num_slots, num_spans, **opts)
        self._span_tokens = torch.zeros(self.num_slots, num_spans, **opts)

    def _move_to(self, device: torch.device) -> None:
        if device == self._device:
            return
        self.role_ce = self.role_ce.to(device)
        self.role_tokens = self.role_tokens.to(device)
        self.role_correct = self.role_correct.to(device)
        self.actions = self.actions.to(device)
        self._slot_ce = self._slot_ce.to(device)
        self._slot_tokens = self._slot_tokens.to(device)
        self._slot_correct = self._slot_correct.to(device)
        self._span_correct = self._span_correct.to(device)
        self._span_tokens = self._span_tokens.to(device)
        self._device = device

    def add_chunk(
        self,
        *,
        role_ids: torch.Tensor,
        span_ids: torch.Tensor,
        nll: torch.Tensor,
        correct: torch.Tensor,
        valid: torch.Tensor,
        slot: int = 0,
    ) -> None:
        """Record one sequence chunk into ``slot``. Tensors are flat, detached.

        Writing (not adding) makes this idempotent, so a checkpointed chunk
        recomputed during backward overwrites its own row with the same
        numbers instead of counting them twice.
        """
        if not 0 <= slot < self.num_slots:
            raise ValueError(
                f"slot {slot} is outside the {self.num_slots} scratch rows; "
                "call ensure_slots(num_chunks) first, or chunks would "
                "overwrite each other's counts."
            )
        self._move_to(nll.device)
        zero = torch.zeros((), dtype=torch.int64, device=nll.device)

        # Ignored positions are parked in role slot 0, which is ROLE_IGNORE and
        # is never reported, so masking is a multiply rather than a gather.
        roles = torch.where(valid, role_ids, zero)
        weights = valid.to(torch.float64)
        ce_row = self._slot_ce[slot].zero_()
        tok_row = self._slot_tokens[slot].zero_()
        cor_row = self._slot_correct[slot].zero_()
        ce_row.scatter_add_(0, roles, nll.to(torch.float64) * weights)
        tok_row.scatter_add_(0, roles, weights)
        cor_row.scatter_add_(0, roles, correct.to(torch.float64) * weights)

        in_span = valid & (span_ids >= 0) & (span_ids < self.max_spans)
        slots = torch.where(in_span, span_ids, zero)
        span_weights = in_span.to(torch.float64)
        span_tok_row = self._span_tokens[slot].zero_()
        span_cor_row = self._span_correct[slot].zero_()
        span_tok_row.scatter_add_(0, slots, span_weights)
        span_cor_row.scatter_add_(0, slots, correct.to(torch.float64) * span_weights)

    def end_microbatch(self) -> None:
        """Fold the per-chunk scratch into the running totals."""
        self.role_ce += self._slot_ce.sum(0)
        self.role_tokens += self._slot_tokens.sum(0)
        self.role_correct += self._slot_correct.sum(0)

        # Sum over chunks first: an action split across a chunk boundary is
        # only whole once its rows are added together.
        span_tokens = self._span_tokens.sum(0)
        span_correct = self._span_correct.sum(0)
        seen = span_tokens > 0
        self.actions[0] += (seen & (span_correct == span_tokens)).sum()
        self.actions[1] += seen.sum()

        self._slot_ce.zero_()
        self._slot_tokens.zero_()
        self._slot_correct.zero_()
        self._span_correct.zero_()
        self._span_tokens.zero_()

    def reset(self) -> None:
        self.role_ce.zero_()
        self.role_tokens.zero_()
        self.role_correct.zero_()
        self.actions.zero_()
        self._slot_ce.zero_()
        self._slot_tokens.zero_()
        self._slot_correct.zero_()
        self._span_correct.zero_()
        self._span_tokens.zero_()

    def state(self) -> torch.Tensor:
        """Flatten the totals so the trainer can all-reduce them in one call."""
        return torch.cat(
            [self.role_ce, self.role_tokens, self.role_correct, self.actions]
        )

    @staticmethod
    def metrics_from_state(state: torch.Tensor) -> dict[str, float]:
        """Turn a (cross-rank reduced) state tensor into loggable scalars."""
        ce = state[:_NUM_ROLES]
        tokens = state[_NUM_ROLES : 2 * _NUM_ROLES]
        correct = state[2 * _NUM_ROLES : 3 * _NUM_ROLES]
        actions = state[3 * _NUM_ROLES :]

        metrics: dict[str, float] = {}
        # Slot 0 is ROLE_IGNORE and only ever receives zero weight, so this is
        # the supervised-token count.
        supervised = float(tokens.sum().item())
        for role_id, name in ROLE_NAMES.items():
            ntok = float(tokens[role_id].item())
            if ntok == 0:
                continue
            metrics[f"loss/{name}"] = float(ce[role_id].item()) / ntok
            metrics[f"acc/{name}"] = float(correct[role_id].item()) / ntok
            metrics[f"tokens/{name}"] = ntok
            if supervised > 0:
                metrics[f"tokens/{name}_frac"] = ntok / supervised

        n_actions = float(actions[1].item())
        if n_actions > 0:
            # "Gold adoption" under teacher forcing: every token of the action
            # is the argmax. Anything less would have produced a different
            # action string at inference.
            metrics["action/exact_match"] = float(actions[0].item()) / n_actions
            metrics["action/count"] = n_actions
        return metrics


class RoleCrossEntropyLoss(BaseLoss):
    """Sum-reduced cross-entropy that also fills a ``RoleMetricAccumulator``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        global_vocab_size: int | None = None
        """Full vocabulary size. Kept for parity with ``CrossEntropyLoss``."""

        max_spans: int = 256
        """Decision spans per packed sequence; must match the dataloader."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.global_vocab_size = config.global_vocab_size
        self.max_spans_per_sequence = config.max_spans
        self.fn = per_token_cross_entropy
        self._maybe_compile(compile_config)
        self.accumulator = RoleMetricAccumulator(max_spans=config.max_spans)
        self.capture = ActionCapture()
        """Field-level action metrics need the argmax IDS, not just whether they
        were right, so they cannot come out of the device-side accumulator.  The
        capture is armed only on steps that will be logged; see the trainer."""

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        *,
        role_ids: torch.Tensor | None = None,
        span_ids: torch.Tensor | None = None,
        chunk_index: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(pred, DTensor):
            raise ValueError(
                "RoleCrossEntropyLoss needs replicated logits to attribute a "
                "loss to a role; it does not support vocab-parallel (TP) "
                "logits. Run with tensor_parallel_degree=1, or swap in "
                "CrossEntropyLoss and give up the per-role split."
            )

        nll = self.fn(pred, labels)
        # reduction="none" then summed is exactly reduction="sum", so a single
        # softmax pass serves both the gradient and the metrics.
        loss = nll.sum()

        if role_ids is not None and span_ids is not None:
            with torch.no_grad():
                flat_labels = labels.flatten(0, 1)
                valid = flat_labels != IGNORE_INDEX
                argmax = pred.flatten(0, 1).argmax(dim=-1)
                correct = argmax == flat_labels

                # Spans are numbered within their packed sequence, so shift each
                # batch row into its own block before flattening.
                batch_size = span_ids.shape[0]
                self.accumulator.ensure_capacity(
                    batch_size * self.max_spans_per_sequence
                )
                offsets = (
                    torch.arange(batch_size, device=span_ids.device).unsqueeze(1)
                    * self.max_spans_per_sequence
                )
                global_spans = torch.where(span_ids >= 0, span_ids + offsets, span_ids)

                flat_roles = role_ids.flatten(0, 1)
                flat_spans = global_spans.flatten(0, 1)
                self.accumulator.add_chunk(
                    role_ids=flat_roles,
                    span_ids=flat_spans,
                    nll=nll.detach(),
                    correct=correct,
                    valid=valid,
                    slot=chunk_index,
                )
                self.capture.add(
                    role_ids=flat_roles, span_ids=flat_spans,
                    argmax=argmax, labels=flat_labels, chunk=chunk_index,
                )

        if global_valid_tokens is not None:
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
                if get_spmd_backend() == "spmd_types":
                    spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P, "tp": spmd.I})
        return loss, {}


class CheckpointedChunkedLoss(ChunkedLossWrapper):
    """``ChunkedLossWrapper``'s memory profile without its custom autograd.

    The stock wrapper runs a per-chunk ``backward()`` inside the forward and
    splices the accumulated gradient back onto the hidden states through a
    custom autograd Function. That Function does not survive torch 2.12 (see
    compat.py): the backward dies with "the tensor has a non-zero number of
    elements, but its data is not allocated yet", and it takes every stock
    config down with it, not just this one.

    This subclass keeps the reason the wrapper exists -- never materializing
    logits for the whole sequence at once, which at seq_len 16384 and a 201k
    vocabulary would be several gigabytes -- but gets there with stock
    ``torch.utils.checkpoint``: each chunk's lm_head and cross-entropy are
    recomputed during backward, so only one chunk of logits is ever live.

    Recomputation means the metric code runs twice per chunk, which is why
    ``RoleMetricAccumulator`` writes per-chunk rows instead of adding to a
    running total.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        # Declared, not inherited: Configurable binds the owning class to the
        # Config where it is defined, so an inherited Config would quietly
        # build a ChunkedLossWrapper and none of this class would run.
        pass

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        inner = self.loss_fn
        if not isinstance(inner, RoleCrossEntropyLoss):
            raise ValueError(
                "CheckpointedChunkedLoss exists to carry the role metrics "
                f"through chunking; wrapping {type(inner).__name__} gains "
                "nothing over ChunkedLossWrapper."
            )
        inner.accumulator.ensure_slots(self.num_chunks)

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Keep an FSDP lm_head gathered for the lifetime of the step.

        The chunks are recomputed during backward, long after FSDP would
        normally have resharded the lm_head, and a sharded DTensor weight
        against a plain hidden state is an immediate "mixed Tensor and
        DTensor" error. The stock wrapper suppresses the same reshard, but it
        can restore it at the end of its forward because it runs its backward
        there too; this one cannot, so the suppression is permanent. The cost
        is the unsharded lm_head resident per rank -- 1.2 GiB for gpt-oss-20b's
        201k x 2880 head.
        """
        super().set_lm_head(lm_head)

        from torch.distributed._composable.fsdp import FSDPModule

        if isinstance(lm_head, FSDPModule):
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **loss_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling the chunked loss"

        if isinstance(pred, DTensor) or any(
            isinstance(v, DTensor) for v in loss_inputs.values()
        ):
            raise ValueError(
                "CheckpointedChunkedLoss chunks plain tensors; it does not "
                "handle DTensor activations from tensor or context parallelism."
            )

        num_chunks = self.num_chunks
        seq_len = pred.shape[1]
        if seq_len % num_chunks != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by num_chunks "
                f"{num_chunks}."
            )
        chunk_len = seq_len // num_chunks

        h_chunks = torch.split(pred, chunk_len, dim=1)
        label_chunks = torch.split(labels, chunk_len, dim=1)
        input_chunks = {
            key: torch.split(value, chunk_len, dim=1)
            for key, value in loss_inputs.items()
        }

        def chunk_loss(index: int, hidden: torch.Tensor) -> torch.Tensor:
            loss, _ = self.loss_fn(
                lm_head(hidden),
                label_chunks[index],
                global_valid_tokens,
                # pyrefly: ignore[unexpected-keyword]
                chunk_index=index,
                **{key: chunks[index] for key, chunks in input_chunks.items()},
            )
            return loss

        total_loss = pred.new_zeros((), dtype=torch.float32)
        for index, hidden in enumerate(h_chunks):
            if hidden.requires_grad:
                total_loss = total_loss + checkpoint(
                    chunk_loss, index, hidden, use_reentrant=False
                )
            else:
                total_loss = total_loss + chunk_loss(index, hidden)
        return total_loss, {}
