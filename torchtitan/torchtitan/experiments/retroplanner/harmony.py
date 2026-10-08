# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render a retrosynthesis trajectory into gpt-oss harmony tokens.

One trajectory becomes one training sequence. Every assistant turn is emitted
as the two harmony messages the model produces at inference time::

    <|start|>assistant<|channel|>analysis<|message|>THINK<|end|>
    <|start|>assistant<|channel|>final<|message|>ACT<|end|>

with the last turn closed by ``<|return|>`` instead of ``<|end|>``. The
``<think>``/``<act>`` tags of the source data are dropped: harmony already
separates reasoning from the answer by channel, so the channel *is* the tag.

Alongside ``input_ids``/``labels`` the encoder emits a per-token ``role`` and a
per-token ``decision span id``. Those two are what make the run legible:

    role         which of reasoning / decision / format a supervised token
                 belongs to, so the loss can be reported per role instead of
                 as one number that hides which half of the job is failing.
    span id      which action a decision token belongs to, so a whole action
                 can be scored as adopted-or-not under teacher forcing rather
                 than as a bag of independently correct tokens.

Both are aligned with ``labels`` (that is, with the *predicted* token), not
with ``input_ids``.

Generation at inference starts right after ``<|start|>assistant``, so those
tokens are left unsupervised; everything the model itself would emit -- the
channel header, the content, the closing token -- is supervised.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer

from .trajectory import (
    ROLE_DECISION,
    ROLE_FORMAT,
    ROLE_IGNORE,
    ROLE_REASONING,
    Trajectory,
)

NO_SPAN = -1

# Mirrors build_system_message() of the gpt-oss chat template. The date is a
# fixed field rather than "today" so that two runs over the same data produce
# the same tokens.
DEFAULT_MODEL_IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."
DEFAULT_KNOWLEDGE_CUTOFF = "2024-06"
VALID_CHANNELS_LINE = (
    "# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)


def build_system_message(
    *,
    model_identity: str = DEFAULT_MODEL_IDENTITY,
    knowledge_cutoff: str = DEFAULT_KNOWLEDGE_CUTOFF,
    current_date: str,
    reasoning_effort: str = "medium",
) -> str:
    return (
        f"{model_identity}\n"
        f"Knowledge cutoff: {knowledge_cutoff}\n"
        f"Current date: {current_date}\n\n"
        f"Reasoning: {reasoning_effort}\n\n"
        f"{VALID_CHANNELS_LINE}"
    )


@dataclass
class EncodedTrajectory:
    """A trajectory tokenized and aligned for next-token prediction.

    All four lists have the same length, ``len(full_tokens) - 1``. Entry ``i``
    means: given ``input_ids[:i + 1]``, predict ``labels[i]``, which plays role
    ``roles[i]`` and (for decision tokens) belongs to action ``span_ids[i]``.
    """

    input_ids: list[int]
    labels: list[int]
    roles: list[int]
    span_ids: list[int]
    num_decisions: int

    def __len__(self) -> int:
        return len(self.input_ids)


class HarmonyTrajectoryEncoder:
    """Turns a ``Trajectory`` into token / label / role / span streams."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        current_date: str,
        reasoning_effort: str = "medium",
        model_identity: str = DEFAULT_MODEL_IDENTITY,
    ) -> None:
        self._tokenizer = tokenizer
        if tokenizer.eos_id is None:
            raise ValueError(
                "The gpt-oss tokenizer must expose an eos_id (<|return|>); "
                "got None. Check hf_assets_path points at the gpt-oss assets."
            )
        self.eos_id: int = tokenizer.eos_id
        self._system_message = build_system_message(
            model_identity=model_identity,
            current_date=current_date,
            reasoning_effort=reasoning_effort,
        )
        # Harmony control tokens delimit every segment, and the tokenizer treats
        # them as atomic, so encoding segment-by-segment and concatenating is
        # identical to encoding the whole string at once.
        self._encode_cache: dict[str, list[int]] = {}

    def _encode(self, text: str, *, cache: bool = False) -> list[int]:
        if cache:
            hit = self._encode_cache.get(text)
            if hit is not None:
                return hit
        ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        if cache:
            self._encode_cache[text] = ids
        return ids

    def prefix_segments(self, developer: str) -> list[tuple[str, int]]:
        """The system + developer preamble. Never supervised."""
        return [
            (
                f"<|start|>system<|message|>{self._system_message}<|end|>",
                ROLE_IGNORE,
            ),
            (
                "<|start|>developer<|message|># Instructions\n\n"
                f"{developer}\n\n<|end|>",
                ROLE_IGNORE,
            ),
        ]

    def encode(
        self, trajectory: Trajectory, *, first_span_id: int = 0
    ) -> EncodedTrajectory:
        """Encode one trajectory, numbering its decisions from ``first_span_id``."""
        tokens: list[int] = []
        roles: list[int] = []
        spans: list[int] = []

        def emit(text: str, role: int, span: int = NO_SPAN, *, cache: bool = False):
            ids = self._encode(text, cache=cache)
            tokens.extend(ids)
            roles.extend([role] * len(ids))
            spans.extend([span] * len(ids))

        for text, role in self.prefix_segments(trajectory.developer):
            emit(text, role)

        span_id = first_span_id
        last = len(trajectory.turns) - 1
        for turn in trajectory.turns:
            emit(f"<|start|>user<|message|>{turn.user}<|end|>", ROLE_IGNORE)
            # The harness supplies this token at inference; the model does not
            # generate it, so it stays out of the supervision.
            emit("<|start|>assistant", ROLE_IGNORE, cache=True)

            emit("<|channel|>analysis<|message|>", ROLE_FORMAT, cache=True)
            emit(turn.think, ROLE_REASONING)
            emit("<|end|>", ROLE_FORMAT, cache=True)

            emit(
                "<|start|>assistant<|channel|>final<|message|>",
                ROLE_FORMAT,
                cache=True,
            )
            emit(turn.act, ROLE_DECISION, span_id)
            emit(
                "<|return|>" if turn.index == last else "<|end|>",
                ROLE_FORMAT,
                cache=True,
            )
            span_id += 1

        # Shift: position i predicts token i + 1, so labels/roles/spans are the
        # streams of the *next* token, and unsupervised targets are masked out.
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
