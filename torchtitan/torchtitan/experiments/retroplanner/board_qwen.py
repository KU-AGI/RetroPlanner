# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenize a board episode into token / label / role / span streams, Qwen3.5 flavor.

Same contract as :class:`board_harmony.BoardHarmonyEncoder` -- and the same
roles, so the loss and the accumulator are reused unchanged -- but the wire
syntax is Qwen3.5's chat template (``<|im_start|>``/``<|im_end|>``, a
``<think>`` block, and a ``<tool_call>``/``<function=...>``/``<parameter=...>``
tag soup) instead of harmony's control tokens:

    <|im_start|>user
    BOARD<|im_end|>
    <|im_start|>assistant
    <think>
    THINK
    </think>

    <tool_call>
    <function=board_act>
    <parameter=actions>
    [ACTION, ...]
    </parameter>
    </function>
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    BOARD
    </tool_response><|im_end|>
    ...
    <|im_start|>assistant
    <think>
    THINK
    </think>

    the route it hands over<|im_end|>

Four things carried over from ``BoardHarmonyEncoder``, and one that changes.

``<|im_start|>assistant\\n`` is not supervised: the harness emits it and the
model continues from there.  Everything after it is, INCLUDING the choice to
open a tool call rather than answer -- that choice is the stop decision, so it
has to be in the loss.

An unsupervised (rollout) turn is encoded with its decision tokens set to
``ROLE_IGNORE``.  It stays in the sequence -- the conversation has to remain
coherent, and the model has to see the mistake it is recovering from -- and it
contributes no gradient.

The tool-call argument is the bare JSON ARRAY, not the ``{"actions": [...]}``
object ``turn.payload`` carries.  Qwen's chat template renders one
``<parameter=NAME>`` block per top-level argument key (see
``chat_template.jinja``'s ``tool_calls`` loop); the schema declares a single
argument named ``actions`` whose value is the array, so the array -- not the
wrapper dict -- is what a real ``<parameter=actions>`` block would ever hold.
Training on the wrapper form would teach a call a served model's tool-call
parser cannot execute.

The system preamble is rendered through the REAL downloaded chat template
(``tokenizer.apply_chat_template`` with ``tools=[...]``), not hand-assembled,
for the same reason ``to_board_trajectory`` LIFTS ``wire_system``/
``wire_developer`` from the builder instead of rebuilding them: a hand-rolled
``<tools>`` block risks drifting from whatever the serving stack renders (as
harmony's own openai_harmony encoder drifts from the HF template).  Qwen has one chat template used for both training
and serving, so rendering the system turn once through it, off the real
downloaded tokenizer, is the byte-faithful version and the harmony-specific
tool-declaration tail of ``wire_developer`` (the ``# Tools\\n\\n## functions``
TypeScript block openai_harmony writes) is stripped and replaced with it.
``wire_system`` itself (ChatGPT identity, "Reasoning: low", the channel list)
is harmony plumbing with no Qwen equivalent and is dropped outright; the
model-agnostic board semantics/instructions prose that precedes the harmony
tools block in ``wire_developer`` is what is kept.

The one thing that changes from ``BoardHarmonyEncoder``: the ``<think>`` block
is ALWAYS present, never omitted.  Qwen3.5's chat template renders every
assistant turn after the last real user query with
``'<think>\\n' + reasoning_content + '\\n</think>\\n\\n'`` unconditionally
(``chat_template.jinja`` line ~101); a turn with nothing to think encodes as
the literal empty block ``<think>\\n\\n</think>\\n\\n``, byte-identical to what
the template emits for ``enable_thinking=False``.  There is no per-turn
"open the channel or not" choice the way harmony's separate analysis message
gives it -- so, unlike ``BoardHarmonyEncoder``, ``turn.think is None`` does NOT
skip the block, it just leaves its interior empty.  This matters: the whole
point of an optional reasoning channel is to avoid teaching the model to
treat "think or don't" as a coin flip on every turn, and for Qwen that coin
flip does not exist in the wire format to begin with, so encoding it as if it
did would train on a shape the model can never produce at inference.

The template's ``<think>``-stripping rule (keep reasoning only for assistant
turns after the LAST real ``user``-role message, where a ``tool``-role
observation does not count) is benign for this corpus: every board episode's
only genuine ``user`` message is the opening board at turn 0 (every later
observation arrives as a ``tool`` message), which is how the renderer writes
every episode. If a future corpus ever puts a
second real user turn mid-episode, this encoder would still emit reasoning on
every turn while the real chat template would strip it from everything before
that turn -- a train/serve mismatch this docstring flags rather than guards
against, since guarding against it would mean guessing at a shape no episode
in this corpus has ever produced.
"""

from __future__ import annotations

import json

from torchtitan.components.tokenizer import BaseTokenizer

from .board_trajectory import BoardTrajectory, TOOL_NAME
from .harmony import EncodedTrajectory, NO_SPAN
from .trajectory import (
    ROLE_DECISION,
    ROLE_FORMAT,
    ROLE_IGNORE,
    ROLE_REASONING,
    ROLE_TERMINATE,
)

IGNORE_INDEX = -100

# The harmony-specific tail of wire_developer (openai_harmony's own TypeScript
# tool declaration) starts here; everything before it is model-agnostic board
# instructions and is kept, everything from here on is replaced by Qwen's own
# <tools> block for the same schema.  See the module docstring.
_HARMONY_TOOLS_MARKER = "\n\n# Tools\n\n## functions"

# Duplicated from evaluation/board/board/harmony.py's ACT_SCHEMA / TOOL_DESC,
# not imported: that file lives outside this torchtitan checkout, and
# torchtitan/experiments code does not reach across repo
# boundaries with sys.path tricks (see board_trajectory.py's own TOOL_NAME,
# which is the same kind of duplication for the same reason). If the upstream
# schema changes, this copy has to change with it.
ACT_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": "Actions applied in the order given. At most one per molecule. "
            'Every call is wrapped: {"actions": [ACTION, ...]}, never a '
            "bare action. Each ACTION is an object "
            '{"type": "open"|"rank"|"done", "mid"?: string, '
            '"order"?: number[], '
            '"choices"?: {molecule id: candidate number}} -- '
            "open and rank take `mid`, rank also takes `order`, "
            "done takes `choices` and no `mid`.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["open", "rank", "done"],
                        "description": "open: fetch a molecule's candidates. "
                        "rank: order them and apply the first. "
                        "done: claim a route the board can make.",
                    },
                    "mid": {
                        "type": "string",
                        "description": "Molecule id. Required except for done.",
                    },
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
                        "The first is applied now; the rest are the "
                        "other routes this molecule can give, kept "
                        "on the board for you to claim later.",
                    },
                },
                "required": ["type"],
            },
        }
    },
    "required": ["actions"],
}

TOOL_DESC = (
    "Apply actions to the retrosynthesis board and receive the board that "
    "results. Nothing on the board moves without a call."
)


class BoardQwenEncoder:
    """Turns a :class:`BoardTrajectory` into the four aligned streams, Qwen3.5 wire format."""

    def __init__(
        self, tokenizer: BaseTokenizer, *, supervise_terminate: bool = True
    ) -> None:
        self._tokenizer = tokenizer
        if tokenizer.eos_id is None:
            raise ValueError(
                "the Qwen3.5 tokenizer must expose an eos_id (<|im_end|>); got "
                "None -- check hf_assets_path points at the Qwen3.5 assets"
            )
        self.eos_id: int = tokenizer.eos_id
        self.supervise_terminate = supervise_terminate
        self._cache: dict[str, list[int]] = {}
        self._system_cache: dict[str, str] = {}
        # These are the literal strings every emit() call splits segments on.
        # If any of them is not atomic to this tokenizer, encoding segment by
        # segment and concatenating stops being equal to encoding the whole
        # string at once, and the role/span streams silently misalign with
        # the tokens they are meant to label.
        for atom in (
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ):
            if len(self._encode(atom, cache=True)) != 1:
                raise ValueError(
                    f"{atom!r} did not encode to a single token; this tokenizer "
                    "is not the Qwen3.5 one this encoder was written against, "
                    "and the role streams would be misaligned"
                )

    def _encode(self, text: str, *, cache: bool = False) -> list[int]:
        if cache and text in self._cache:
            return self._cache[text]
        ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        if cache:
            self._cache[text] = ids
        return ids

    def _system_block(self, developer: str) -> str:
        """The Qwen system turn: model-agnostic prose plus a real <tools> block.

        Rendered once per distinct ``developer`` string through the tokenizer's
        own chat template, so the tool declaration this trains on is exactly
        what a Qwen serving stack (which uses the same template) would render
        for the same tool schema -- not a hand-typed approximation of it.
        """
        cached = self._system_cache.get(developer)
        if cached is not None:
            return cached
        idx = developer.find(_HARMONY_TOOLS_MARKER)
        if idx < 0:
            raise ValueError(
                "developer text has no harmony tool-declaration marker "
                f"({_HARMONY_TOOLS_MARKER!r}); expected wire_developer to end "
                "in openai_harmony's own '# Tools' block, which this encoder "
                "strips and replaces with Qwen's <tools> convention"
            )
        prose = developer[:idx]
        tool = {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": TOOL_DESC,
                "parameters": ACT_SCHEMA,
            },
        }
        # The template raises "No user query found in messages" for a message
        # list with no genuine user turn (see its reversed multi_step_tool
        # scan); the dummy user turn satisfies that, and is discarded below
        # along with everything from the system block's own <|im_end|> on.
        rendered = self._tokenizer.apply_chat_template(
            messages=[
                {"role": "system", "content": prose},
                {"role": "user", "content": "_"},
            ],
            tools=[tool],
            add_generation_prompt=False,
        )
        marker = "<|im_end|>\n"
        end = rendered.find(marker)
        if end < 0:
            raise ValueError(
                "chat template did not close the system message with "
                "'<|im_end|>\\n'; this tokenizer's chat_template.jinja has "
                "changed shape from the one this encoder was written against"
            )
        block = rendered[: end + len(marker)]
        self._system_cache[developer] = block
        return block

    @staticmethod
    def _actions_json(payload: str) -> str:
        """The bare JSON array a <parameter=actions> block holds.

        ``turn.payload`` is the full ``{"actions": [...]}`` object (harmony's
        function-call argument is the whole object); Qwen's tool-call syntax
        argues per named parameter, so only the array under "actions" belongs
        inside the tag -- see the module docstring.
        """
        try:
            payload_obj = json.loads(payload)
        except (ValueError, TypeError) as e:
            raise ValueError(f"board_act payload is not valid JSON: {payload!r}") from e
        if not isinstance(payload_obj, dict) or "actions" not in payload_obj:
            raise ValueError(f"board_act payload has no 'actions' key: {payload!r}")
        return json.dumps(payload_obj["actions"], ensure_ascii=False)

    def encode(
        self, traj: BoardTrajectory, *, first_span_id: int = 0
    ) -> EncodedTrajectory:
        tokens: list[int] = []
        roles: list[int] = []
        spans: list[int] = []

        def emit(
            text: str, role: int, span: int = NO_SPAN, *, cache: bool = False
        ) -> None:
            if not text:
                return
            ids = self._encode(text, cache=cache)
            tokens.extend(ids)
            roles.extend([role] * len(ids))
            spans.extend([span] * len(ids))

        emit(self._system_block(traj.developer), ROLE_IGNORE, cache=True)

        span_id = first_span_id
        for turn in traj.turns:
            if turn.obs is not None:
                if turn.obs_kind == "user":
                    emit(f"<|im_start|>user\n{turn.obs}<|im_end|>\n", ROLE_IGNORE)
                else:
                    emit(
                        f"<|im_start|>user\n<tool_response>\n{turn.obs}"
                        "\n</tool_response><|im_end|>\n",
                        ROLE_IGNORE,
                    )

            emit("<|im_start|>assistant\n", ROLE_IGNORE, cache=True)
            fmt_role = ROLE_FORMAT if turn.supervised else ROLE_IGNORE
            body_role = ROLE_DECISION if turn.supervised else ROLE_IGNORE
            reasoning_role = ROLE_REASONING if turn.supervised else ROLE_IGNORE

            # Always present -- see the module docstring on why this, unlike
            # BoardHarmonyEncoder's analysis channel, is never skipped.
            emit("<think>\n", fmt_role, cache=True)
            if turn.think:
                emit(turn.think, reasoning_role)
            emit("\n</think>\n\n", fmt_role, cache=True)

            if turn.payload is not None:
                emit(
                    f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=actions>\n",
                    fmt_role,
                    cache=True,
                )
                emit(
                    self._actions_json(turn.payload),
                    body_role,
                    span_id if turn.supervised else NO_SPAN,
                )
                emit("\n</parameter>\n</function>\n</tool_call>", fmt_role, cache=True)
                if turn.supervised:
                    span_id += 1
            else:
                term_role = (
                    ROLE_TERMINATE
                    if (turn.supervised and self.supervise_terminate)
                    else ROLE_IGNORE
                )
                emit(turn.final or "", term_role)
            emit("<|im_end|>\n", fmt_role, cache=True)

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
