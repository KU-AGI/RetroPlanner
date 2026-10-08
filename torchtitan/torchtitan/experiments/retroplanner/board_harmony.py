# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenize a board episode into token / label / role / span streams.

Same contract as :class:`harmony.HarmonyTrajectoryEncoder` -- and the same roles,
so the loss and the accumulator are reused unchanged -- but the segments are the
ones a tool-call conversation is made of:

    <|start|>user<|message|>BOARD<|end|>                                 ignore
    <|start|>functions.board_act to=assistant<|channel|>commentary
        <|message|>BOARD<|end|>                                          ignore
    <|start|>assistant                                                   ignore
        to=functions.board_act<|channel|>commentary json<|message|>      format
        {"actions": …}                                                   DECISION
        <|call|>                                                         format
    <|start|>assistant<|channel|>final<|message|>                        format
        the route it hands over                                          terminate
        <|return|>                                                       format

Three things are deliberate.

The TURN'S FIRST ``<|start|>assistant`` is not supervised: the harness emits it and
the model continues from there.  Everything after it is, INCLUDING the recipient and
the channel -- choosing to call the tool rather than to answer is the stop decision,
so it has to be in the loss.  The SECOND one, the one that follows an analysis
message, IS supervised: no harness emits that, the model has to, and a reasoning arm
that is not taught to write it cannot leave its own reasoning.

The observation header is the one vLLM produces (``to=assistant`` present, body
raw).  The HF chat template json-escapes that body; training on the escaped form
and serving on the raw one is a mismatch, so the raw form is what is encoded here.

A turn the rollout produced rather than the labeller is encoded with its decision
tokens set to ``ROLE_IGNORE``.  It stays in the sequence -- the conversation has
to remain coherent, and the model has to see the mistake it is recovering from --
and it contributes no gradient.
"""
from __future__ import annotations

from torchtitan.components.tokenizer import BaseTokenizer

from .board_trajectory import BoardTrajectory, TOOL_NAME
from .harmony import NO_SPAN, EncodedTrajectory
from .trajectory import (
    ROLE_DECISION,
    ROLE_FORMAT,
    ROLE_TERMINATE,
    ROLE_IGNORE,
    ROLE_REASONING,
)

IGNORE_INDEX = -100


class BoardHarmonyEncoder:
    """Turns a :class:`BoardTrajectory` into the four aligned streams."""

    def __init__(self, tokenizer: BaseTokenizer, *,
                 supervise_terminate: bool = True) -> None:
        self._tokenizer = tokenizer
        if tokenizer.eos_id is None:
            raise ValueError(
                "the gpt-oss tokenizer must expose an eos_id (<|return|>); got "
                "None -- check hf_assets_path points at the gpt-oss assets"
            )
        self.eos_id: int = tokenizer.eos_id
        self.supervise_terminate = supervise_terminate
        self._cache: dict[str, list[int]] = {}
        # Harmony control tokens are atomic to this tokenizer, so encoding
        # segment by segment and concatenating equals encoding the whole string.
        for atom in ("<|start|>", "<|end|>", "<|call|>", "<|return|>",
                     "<|message|>", "<|channel|>"):
            if len(self._encode(atom, cache=True)) != 1:
                raise ValueError(
                    f"{atom!r} did not encode to a single token; this tokenizer "
                    "is not the gpt-oss/harmony one and the role streams would "
                    "be misaligned"
                )

    def _encode(self, text: str, *, cache: bool = False) -> list[int]:
        if cache and text in self._cache:
            return self._cache[text]
        ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        if cache:
            self._cache[text] = ids
        return ids

    def encode(self, traj: BoardTrajectory, *,
               first_span_id: int = 0) -> EncodedTrajectory:
        tokens: list[int] = []
        roles: list[int] = []
        spans: list[int] = []

        def emit(text: str, role: int, span: int = NO_SPAN, *,
                 cache: bool = False) -> None:
            if not text:
                return
            ids = self._encode(text, cache=cache)
            tokens.extend(ids)
            roles.extend([role] * len(ids))
            spans.extend([span] * len(ids))

        emit(f"<|start|>system<|message|>{traj.system}<|end|>", ROLE_IGNORE)
        emit(f"<|start|>developer<|message|>{traj.developer}<|end|>", ROLE_IGNORE)

        span_id = first_span_id
        last = len(traj.turns) - 1
        for turn in traj.turns:
            if turn.obs is not None:
                if turn.obs_kind == "user":
                    emit(f"<|start|>user<|message|>{turn.obs}<|end|>", ROLE_IGNORE)
                else:
                    emit(f"<|start|>functions.{TOOL_NAME} to=assistant"
                         f"<|channel|>commentary<|message|>{turn.obs}<|end|>",
                         ROLE_IGNORE)

            emit("<|start|>assistant", ROLE_IGNORE, cache=True)
            body_role = ROLE_DECISION if turn.supervised else ROLE_IGNORE
            fmt_role = ROLE_FORMAT if turn.supervised else ROLE_IGNORE

            if turn.think is not None:
                emit("<|channel|>analysis<|message|>", fmt_role, cache=True)
                emit(turn.think, ROLE_REASONING if turn.supervised else ROLE_IGNORE)
                emit("<|end|>", fmt_role, cache=True)
                # SUPERVISED, unlike the turn's first `<|start|>assistant`. That one the
                # harness emits and the model continues from it; THIS one closes the
                # analysis and opens the tool call, and nothing emits it but the model.
                # Left on ROLE_IGNORE the label at the analysis's `<|end|>` is -100, so a
                # reasoning arm is never taught to leave its own reasoning, and the
                # more analyses an episode carries the less the base model's prior
                # carries it back to the tool call.
                emit("<|start|>assistant", fmt_role, cache=True)

            if turn.payload is not None:
                emit(f" to=functions.{TOOL_NAME}<|channel|>commentary json"
                     "<|message|>", fmt_role, cache=True)
                emit(turn.payload, body_role, span_id if turn.supervised else NO_SPAN)
                emit("<|call|>", fmt_role, cache=True)
                if turn.supervised:
                    span_id += 1
            else:
                term_role = (
                    ROLE_TERMINATE
                    if (turn.supervised and self.supervise_terminate)
                    else ROLE_IGNORE
                )
                emit("<|channel|>final<|message|>", fmt_role, cache=True)
                emit(turn.final or "", term_role)
                emit("<|return|>" if turn.index == last else "<|end|>",
                     fmt_role, cache=True)

        labels = [
            tokens[i + 1] if roles[i + 1] != ROLE_IGNORE else IGNORE_INDEX
            for i in range(len(tokens) - 1)
        ]
        return EncodedTrajectory(
            input_ids=tokens[:-1],
            labels=labels,
            roles=roles[1:],
            span_ids=spans[1:],
            num_decisions=span_id - first_span_id,
        )
