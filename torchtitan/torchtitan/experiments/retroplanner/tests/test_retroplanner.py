# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the harmony encoder, the role accumulator and the validator.

Run with the gpt-oss tokenizer on hand::

    GPT_OSS_ASSETS=/path/to/gpt-oss-20b pytest \\
        torchtitan/experiments/retroplanner/tests/test_retroplanner.py
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.retroplanner.harmony import HarmonyTrajectoryEncoder, NO_SPAN
from torchtitan.experiments.retroplanner.loss import (
    RoleCrossEntropyLoss,
    RoleMetricAccumulator,
)
from torchtitan.experiments.retroplanner.trajectory import (
    parse_action,
    ROLE_DECISION,
    ROLE_FORMAT,
    ROLE_IGNORE,
    ROLE_REASONING,
    to_trajectory,
    validate_record,
)

# Run from an upstream torchtitan checkout, so the cache root comes from the
# environment (config/env.sh's RP_CACHE), not from this file's path.
_RP_CACHE = os.environ.get("RP_CACHE", "/mnt/data/.cache")

_DEFAULT_ASSET_GLOB = (
    f"{_RP_CACHE}/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*/"
)


def _assets_path() -> str:
    path = os.environ.get("GPT_OSS_ASSETS")
    if path:
        return path
    matches = glob.glob(_DEFAULT_ASSET_GLOB)
    if not matches:
        pytest.skip("gpt-oss assets not available; set GPT_OSS_ASSETS")
    return matches[0]


RECORD = {
    "route_idx": 0,
    "combo_idx": 0,
    "benchmark": "paroutes",
    "n_turns": 3,
    "n_expansions": 1,
    "actions": ["single aa", "rank aa c0,c1,c2", "done"],
    "mols": {"aa": "CCO", "bb": "CC=O", "cc": "O"},
    "tree": {
        "nodes": {"aa": "committed", "bb": "buyable", "cc": "buyable"},
        "edges": [["aa", "bb", "h1"], ["aa", "cc", "h1"]],
    },
    "messages": [
        {"role": "system", "content": "You are planning a retrosynthesis."},
        {
            "role": "user",
            "content": (
                "Turn 0. Expansion budget: 10 of 10 left.\n\n"
                "Needs an expansion (costs 1):\n  <mol aa>CCO</mol>"
            ),
        },
        {
            "role": "assistant",
            "content": "<think>Only one move.</think>\n<act>single aa</act>",
        },
        {
            "role": "user",
            "content": (
                "<mol aa/> came back with 3 candidate disconnections.\n\n"
                "Turn 1. Expansion budget: 9 of 10 left.\n\n"
                "Ready to rank (free) — commit one disconnection:\n"
                "  <mol aa/> — 3 candidates:\n"
                "    <c0>CC=O* + O*</c0>\n"
                "    <c1>CCBr* + O*</c1>\n"
                "    <c2>CCCl* + O*</c2>"
            ),
        },
        {
            "role": "assistant",
            "content": "<think>c0 splits cleanly.</think>\n<act>rank aa c0,c1,c2</act>",
        },
        {
            "role": "user",
            "content": (
                "Route **h1** under <mol aa/> is open — 2 pieces. "
                "Both must reach buyable material:\n"
                "  <mol bb>CC=O</mol> is buyable.\n"
                "  <mol cc>O</mol> is buyable.\n\n"
                "Turn 2. Expansion budget: 9 of 10 left.\n\nNothing is open."
            ),
        },
        {
            "role": "assistant",
            "content": "<think>All buyable.</think>\n<act>done</act>",
        },
    ],
}


@pytest.fixture(scope="module")
def encoder() -> HarmonyTrajectoryEncoder:
    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=_assets_path())
    return HarmonyTrajectoryEncoder(tokenizer, current_date="2025-06-01")


def test_validator_accepts_a_well_formed_record():
    report = validate_record(RECORD)
    assert report.issues == []
    assert report.n_gold_checked == 1
    assert report.n_gold_adopted == 1


def test_validator_flags_a_broken_state_transition():
    record = {**RECORD, "messages": [dict(m) for m in RECORD["messages"]]}
    # Spend an expansion without paying for it.
    record["messages"][3]["content"] = record["messages"][3]["content"].replace(
        "9 of 10 left", "10 of 10 left"
    )
    assert "state/budget" in validate_record(record).issues


def test_validator_flags_a_gold_adoption_failure():
    record = {**RECORD, "messages": [dict(m) for m in RECORD["messages"]]}
    # Commit a disconnection the tree does not record.
    record["messages"][4][
        "content"
    ] = "<think>c1 instead.</think>\n<act>rank aa c1,c0,c2</act>"
    record["actions"] = ["single aa", "rank aa c1,c0,c2", "done"]
    report = validate_record(record)
    assert "gold/committed_children_mismatch" in report.issues
    assert report.n_gold_adopted == 0


def test_action_grammar():
    done = parse_action("done")
    single = parse_action("single 4k")
    rank = parse_action("rank 99 c0,c1,c2")
    assert done is not None and done.kind == "done"
    assert single is not None and single.mol == "4k"
    assert rank is not None and rank.choices == (0, 1, 2)
    assert parse_action("rank 99 c0 c1") is None
    assert parse_action("expand 99") is None


def test_encoded_streams_are_aligned(encoder):
    encoded = encoder.encode(to_trajectory(RECORD))
    n = len(encoded.input_ids)
    assert len(encoded.labels) == n
    assert len(encoded.roles) == n
    assert len(encoded.span_ids) == n

    for label, role, span in zip(
        encoded.labels, encoded.roles, encoded.span_ids, strict=True
    ):
        # A masked label and an ignored role must always agree, otherwise the
        # per-role denominators would not add up to the loss denominator.
        assert (label == IGNORE_INDEX) == (role == ROLE_IGNORE)
        assert (span != NO_SPAN) == (role == ROLE_DECISION)

    assert encoded.num_decisions == 3
    assert sorted(set(s for s in encoded.span_ids if s != NO_SPAN)) == [0, 1, 2]


def test_decision_tokens_decode_to_the_action(encoder):
    encoded = encoder.encode(to_trajectory(RECORD))
    tokenizer = encoder._tokenizer
    for span, expected in enumerate(RECORD["actions"]):
        ids = [
            label
            for label, s in zip(encoded.labels, encoded.span_ids, strict=True)
            if s == span
        ]
        assert tokenizer.decode(ids) == expected


def test_reasoning_tokens_carry_the_chain_of_thought(encoder):
    encoded = encoder.encode(to_trajectory(RECORD))
    tokenizer = encoder._tokenizer
    ids = [
        label
        for label, role in zip(encoded.labels, encoded.roles, strict=True)
        if role == ROLE_REASONING
    ]
    text = tokenizer.decode(ids)
    assert "Only one move." in text
    assert "c0 splits cleanly." in text


def test_prompt_tokens_are_never_supervised(encoder):
    encoded = encoder.encode(to_trajectory(RECORD))
    tokenizer = encoder._tokenizer
    supervised = tokenizer.decode(
        [label for label in encoded.labels if label != IGNORE_INDEX]
    )
    # The 50-candidate listings are the bulk of the sequence and must not leak
    # into the target, or the reported loss is mostly copying the prompt.
    assert "candidate disconnections" not in supervised
    assert "Expansion budget" not in supervised


def test_trailing_return_token_closes_the_trajectory(encoder):
    encoded = encoder.encode(to_trajectory(RECORD))
    assert encoded.labels[-1] == encoder.eos_id
    assert encoded.roles[-1] == ROLE_FORMAT


def test_accumulator_scores_actions_as_all_or_nothing():
    accumulator = RoleMetricAccumulator(max_spans=8)
    # Two actions of two tokens each: the first fully correct, the second not.
    role_ids = torch.tensor([ROLE_DECISION] * 4)
    span_ids = torch.tensor([0, 0, 1, 1])
    accumulator.add_chunk(
        role_ids=role_ids,
        span_ids=span_ids,
        nll=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        correct=torch.tensor([True, True, True, False]),
        valid=torch.tensor([True, True, True, True]),
    )
    accumulator.end_microbatch()
    metrics = RoleMetricAccumulator.metrics_from_state(accumulator.state())

    assert metrics["action/count"] == 2
    assert metrics["action/exact_match"] == 0.5
    assert metrics["loss/decision"] == pytest.approx(0.75)
    assert metrics["acc/decision"] == pytest.approx(0.75)


def test_accumulator_splits_chunked_spans():
    """A span split across two chunks must still be scored as one action."""
    whole = RoleMetricAccumulator(max_spans=8)
    whole.add_chunk(
        role_ids=torch.tensor([ROLE_DECISION] * 4),
        span_ids=torch.tensor([0, 0, 0, 0]),
        nll=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        correct=torch.tensor([True, True, True, False]),
        valid=torch.tensor([True] * 4),
    )
    whole.end_microbatch()

    chunked = RoleMetricAccumulator(max_spans=8)
    chunked.ensure_slots(2)
    for slot, (lo, hi) in enumerate(((0, 2), (2, 4))):
        chunked.add_chunk(
            role_ids=torch.tensor([ROLE_DECISION] * (hi - lo)),
            span_ids=torch.tensor([0] * (hi - lo)),
            nll=torch.tensor([1.0] * (hi - lo)),
            correct=torch.tensor([True, True, True, False][lo:hi]),
            valid=torch.tensor([True] * (hi - lo)),
            slot=slot,
        )
    chunked.end_microbatch()

    assert torch.equal(whole.state(), chunked.state())
    assert (
        RoleMetricAccumulator.metrics_from_state(chunked.state())["action/exact_match"]
        == 0.0
    )


def test_loss_matches_stock_cross_entropy_and_keeps_batch_rows_apart():
    torch.manual_seed(0)
    batch, seq, vocab = 2, 6, 11
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, 0] = IGNORE_INDEX

    roles = torch.full((batch, seq), ROLE_DECISION)
    roles[:, 0] = ROLE_IGNORE
    spans = torch.zeros(batch, seq, dtype=torch.long)
    spans[:, 0] = NO_SPAN

    loss_fn = RoleCrossEntropyLoss.Config(max_spans=4).build()
    loss, _ = loss_fn(logits, labels, None, role_ids=roles, span_ids=spans)

    # The gradient-carrying value must be the stock sum-reduced CE, unchanged.
    reference = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )
    assert torch.allclose(loss, reference)

    loss_fn.accumulator.end_microbatch()
    metrics = RoleMetricAccumulator.metrics_from_state(loss_fn.accumulator.state())
    # Both rows number their single action 0; they must not collapse into one.
    assert metrics["action/count"] == 2
    assert metrics["tokens/decision"] == batch * (seq - 1)
    assert metrics["loss/decision"] == pytest.approx(
        reference.item() / (batch * (seq - 1))
    )


def test_accumulator_ignores_masked_positions():
    accumulator = RoleMetricAccumulator(max_spans=8)
    accumulator.add_chunk(
        role_ids=torch.tensor([ROLE_REASONING, ROLE_IGNORE, ROLE_FORMAT]),
        span_ids=torch.tensor([NO_SPAN, NO_SPAN, NO_SPAN]),
        nll=torch.tensor([2.0, 99.0, 1.0]),
        correct=torch.tensor([False, True, True]),
        valid=torch.tensor([True, False, True]),
    )
    accumulator.end_microbatch()
    metrics = RoleMetricAccumulator.metrics_from_state(accumulator.state())

    assert metrics["loss/reasoning"] == pytest.approx(2.0)
    assert metrics["loss/format"] == pytest.approx(1.0)
    assert "loss/decision" not in metrics
    assert metrics["tokens/reasoning_frac"] == pytest.approx(0.5)


def test_accumulator_rejects_an_unprovisioned_slot():
    """Too few scratch rows must fail loudly, not drop all but the last chunk."""
    accumulator = RoleMetricAccumulator(max_spans=8)
    with pytest.raises(ValueError, match="ensure_slots"):
        accumulator.add_chunk(
            role_ids=torch.tensor([ROLE_DECISION]),
            span_ids=torch.tensor([0]),
            nll=torch.tensor([1.0]),
            correct=torch.tensor([True]),
            valid=torch.tensor([True]),
            slot=1,
        )


def test_recomputing_a_chunk_does_not_double_count():
    """Activation checkpointing runs each chunk twice; metrics must not double."""
    once = RoleMetricAccumulator(max_spans=8)
    once.ensure_slots(2)
    twice = RoleMetricAccumulator(max_spans=8)
    twice.ensure_slots(2)

    def record(accumulator, slot):
        accumulator.add_chunk(
            role_ids=torch.tensor([ROLE_DECISION, ROLE_REASONING]),
            span_ids=torch.tensor([slot, NO_SPAN]),
            nll=torch.tensor([2.0, 3.0]),
            correct=torch.tensor([True, False]),
            valid=torch.tensor([True, True]),
            slot=slot,
        )

    for slot in (0, 1):
        record(once, slot)
    for slot in (0, 1):
        record(twice, slot)
        record(twice, slot)  # the backward-pass recompute

    once.end_microbatch()
    twice.end_microbatch()
    assert torch.equal(once.state(), twice.state())


def test_every_config_builds_its_own_class():
    """A Config must be declared on its subclass, not inherited.

    ``Configurable`` binds the owning class to wherever ``Config`` is defined,
    so a subclass that only inherits ``Config`` silently builds its parent --
    the subclass never runs and nothing says so.
    """
    from torchtitan.experiments.retroplanner.dataset import RetroTrajectoryDataLoader
    from torchtitan.experiments.retroplanner.loss import (
        CheckpointedChunkedLoss,
        RoleCrossEntropyLoss,
    )
    from torchtitan.experiments.retroplanner.trainer import RetroSFTTrainer, RetroValidator

    for cls in (
        RetroSFTTrainer,
        RetroValidator,
        CheckpointedChunkedLoss,
        RoleCrossEntropyLoss,
        RetroTrajectoryDataLoader,
    ):
        owner = cls.Config._owner
        assert owner is cls, (
            f"{cls.__name__}.Config builds {owner and owner.__name__}; "
            "declare a Config on the subclass"
        )
