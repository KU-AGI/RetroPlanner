# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Board-format SFT: parsing, encoding, and the field-level action metrics."""
from __future__ import annotations

import json
import os

import pytest

from torchtitan.experiments.retroplanner.board_metrics import compare, summarize
from torchtitan.experiments.retroplanner.board_trajectory import (
    board_menu_sizes,
    board_open_ids,
    parse_board_action,
    to_board_trajectory,
    validate_board_record,
)

# Run from an upstream torchtitan checkout, so the RetroPlanner roots come from
# the environment (config/env.sh names), not from this file's path.
_RP_ROOT = os.environ.get("RP_ROOT", "/mnt/data/RetroPlanner")
_RP_CACHE = os.environ.get("RP_CACHE", "/mnt/data/.cache")
_RP_DATA = os.path.join(_RP_ROOT, "tools", "reaction-mcp", "data")

BOARD = """budget 2 of 300

r1  vw·c1  1 of 2 closed        vw ranked: c1 · c0 · c4

OPEN
  kp   under r2 · depth 2 · 8 left
    <c0 q.535 p1.000 rt1>A*(ln$5.66) + B*(ln$2.99)</c0>
    <c1 q.451 p.997 rt1>C*(ln$2.52) + D</c1>
  ud   under r2 · depth 2 · 8 left · no candidates yet

CLOSED  w8*(ln$2.77)"""


def _payload(items):
    return json.dumps({"actions": items}, ensure_ascii=False)


def test_open_ids_and_menu_sizes():
    assert board_open_ids(BOARD) == {"kp", "ud"}
    assert board_menu_sizes(BOARD) == {"kp": 2, "ud": 0}


@pytest.mark.parametrize(
    "text",
    [
        '{"actions": []}',                                    # empty
        '{"actions": [{"type": "fly", "mid": "kp"}]}',         # unknown verb
        '{"actions": [{"type": "rank", "mid": "kp"}]}',        # rank with no order
        '{"actions": [{"type": "rank", "mid": "kp", "order": [0, 0]}]}',  # repeat
        '{"actions": [{"type": "open"}]}',                     # no molecule
        "not json",
    ],
)
def test_illegal_calls_are_rejected(text):
    assert parse_board_action(text) is None


def test_action_roundtrips_through_render():
    text = _payload([{"type": "rank", "mid": "kp", "order": [1, 0]}])
    action = parse_board_action(text)
    assert action is not None
    assert action.render() == text
    assert action.items[0].head == 1


def test_compare_distinguishes_the_failure_modes():
    gold = parse_board_action(
        _payload([{"type": "rank", "mid": "kp", "order": [0, 1]}])
    )
    assert gold is not None

    exact = compare(gold, gold.render(), BOARD)
    assert exact.exact and exact.transition_tree
    assert not exact.rank_misorder and not exact.mid_hallucinated

    # same head, different tail: the same move, a different declared fallback
    tail = compare(
        gold, _payload([{"type": "rank", "mid": "kp", "order": [0]}]), BOARD
    )
    assert tail.heads_match and not tail.orders_match
    assert tail.rank_misorder and tail.transition_tree and not tail.exact

    # different head: the tree diverges
    head = compare(
        gold, _payload([{"type": "rank", "mid": "kp", "order": [1, 0]}]), BOARD
    )
    assert not head.heads_match and not head.transition_tree

    # a molecule that is not on the board
    ghost = compare(
        gold, _payload([{"type": "rank", "mid": "zz", "order": [0, 1]}]), BOARD
    )
    assert ghost.mid_hallucinated and not ghost.mids_set_match

    # a candidate behind the fold
    off = compare(
        gold, _payload([{"type": "rank", "mid": "kp", "order": [0, 7]}]), BOARD
    )
    assert off.candidate_off_screen

    broken = compare(gold, "{oops", BOARD)
    assert not broken.json_valid and not broken.transition_tree


def test_mid_misorder_is_not_a_wrong_molecule():
    gold = parse_board_action(
        _payload([{"type": "open", "mid": "kp"}, {"type": "open", "mid": "ud"}])
    )
    assert gold is not None
    swapped = compare(
        gold,
        _payload([{"type": "open", "mid": "ud"}, {"type": "open", "mid": "kp"}]),
        BOARD,
    )
    assert swapped.mids_set_match and not swapped.mids_order_match
    assert swapped.mid_misorder and not swapped.transition_tree


def test_summarize_reports_counts_with_rates():
    gold = parse_board_action(_payload([{"type": "open", "mid": "kp"}]))
    assert gold is not None
    rows = [compare(gold, gold.render()), compare(gold, "{oops")]
    out = summarize(rows)
    assert out["action/decoded"] == 2
    assert out["transition/exact"] == 0.5
    assert out["action/json_valid"] == 0.5


def test_record_to_trajectory_and_validation():
    record = {
        "target": "CCO",
        "route": {"set_hash": "abc", "n_steps": 1},
        "raw_text": (
            "<|start|>system<|message|>SYS<|end|>"
            "<|start|>developer<|message|>DEV<|end|>rest"
        ),
        "harmony_messages": [
            {"role": "user", "content": BOARD},
            {"role": "assistant", "channel": "commentary",
             "recipient": "functions.board_act", "content_type": "json",
             "content": _payload([{"type": "open", "mid": "kp"}]),
             "supervised": True},
            {"role": "tool", "name": "functions.board_act",
             "channel": "commentary", "content": BOARD},
            {"role": "assistant", "channel": "commentary",
             "recipient": "functions.board_act", "content_type": "json",
             "content": _payload([{"type": "rank", "mid": "zz", "order": [0]}]),
             "supervised": False},
            {"role": "assistant", "channel": "final", "content": "ROUTES\n  A"},
        ],
    }
    traj = to_board_trajectory(record)
    assert traj.system == "SYS" and traj.developer == "DEV"
    assert traj.n_calls == 2
    assert [t.supervised for t in traj.turns] == [True, False, True]
    assert traj.turns[-1].final is not None

    report = validate_board_record(record)
    assert report["n_calls"] == 2
    assert report["n_unsupervised"] == 1
    # the unsupervised turn names a molecule that is not open -- counted, so a
    # rollout that wandered off the board cannot slip into training unseen
    assert report["state/mid_not_open"] == 1


def test_encoder_matches_the_inference_wire_text():
    """The training tokens must be the tokens vLLM feeds, one terminator apart.

    openai_harmony's ``render_conversation`` closes the last assistant message
    with ``<|end|>`` because it renders a conversation as CONTEXT.  Training has
    to close it with ``<|return|>`` -- that is the token the model must learn to
    emit to stop -- so the streams agree on everything except the final id.
    """
    import os

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.experiments.retroplanner.board_harmony import BoardHarmonyEncoder

    assets = os.environ.get(
        "GPT_OSS_ASSETS",
        f"{_RP_CACHE}/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
        "6cee5e81ee83917806bbde320786a8fb61efebee",
    )
    corpus = os.environ.get(
        "BOARD_WIRE_SAMPLE",
        f"{_RP_DATA}/sft_board/"
        "board_v1.wire.jsonl",
    )
    if not os.path.isdir(assets) or not os.path.exists(corpus):
        pytest.skip("gpt-oss assets or an augmented corpus are not available here")

    tokenizer = HuggingFaceTokenizer(tokenizer_path=assets)
    encoder = BoardHarmonyEncoder(tokenizer)
    with open(corpus) as handle:
        for _ in range(3):
            line = handle.readline()
            if not line:
                break
            record = json.loads(line)
            encoded = encoder.encode(to_board_trajectory(record))
            wire = tokenizer.encode(
                record["wire_text"], add_bos=False, add_eos=False
            )
            assert encoded.input_ids == wire[:-1]
            assert encoded.labels[-1] == tokenizer.eos_id
            assert encoded.num_decisions == sum(
                1 for m in record["harmony_messages"]
                if m.get("recipient") and m.get("supervised", True)
            )


def test_capture_turns_tensors_into_field_metrics():
    """The tensor -> host -> decode -> compare path, without a model.

    Two spans: one where the argmax is the gold call, one where it names the
    wrong molecule. Everything the trainer logs comes out of this path, so it is
    worth pinning independently of any forward pass.
    """
    import os

    import torch

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.experiments.retroplanner.board_metrics import ActionCapture
    from torchtitan.experiments.retroplanner.trajectory import (
        ROLE_DECISION,
        ROLE_FORMAT,
    )

    assets = os.environ.get(
        "GPT_OSS_ASSETS",
        f"{_RP_CACHE}/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
        "6cee5e81ee83917806bbde320786a8fb61efebee",
    )
    if not os.path.isdir(assets):
        pytest.skip("gpt-oss assets are not available here")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=assets)

    gold_a = _payload([{"type": "open", "mid": "kp"}])
    gold_b = _payload([{"type": "rank", "mid": "ud", "order": [0, 1]}])
    wrong_b = _payload([{"type": "rank", "mid": "zz", "order": [0, 1]}])

    def ids(text):
        return tokenizer.encode(text, add_bos=False, add_eos=False)

    gold_ids = ids(gold_a) + ids(gold_b)
    pred_ids = ids(gold_a) + ids(wrong_b)
    assert len(ids(gold_b)) == len(ids(wrong_b)), "test needs aligned lengths"

    spans = [0] * len(ids(gold_a)) + [1] * len(ids(gold_b))
    roles = [ROLE_DECISION] * len(spans)
    # a format token in the middle must be ignored by the capture
    gold_ids.append(199999)
    pred_ids.append(199999)
    spans.append(-1)
    roles.append(ROLE_FORMAT)

    capture = ActionCapture()
    capture.arm()
    capture.add(
        role_ids=torch.tensor(roles),
        span_ids=torch.tensor(spans),
        argmax=torch.tensor(pred_ids),
        labels=torch.tensor(gold_ids),
    )
    decoded = capture.decode(tokenizer)
    assert [g for g, _ in decoded] == [gold_a, gold_b]

    metrics = capture.summarize(tokenizer)
    assert metrics["action/decoded"] == 2
    assert metrics["transition/exact"] == 0.5
    assert metrics["action/json_valid"] == 1.0
    assert metrics["action/verbs_match"] == 1.0
    assert metrics["action/mid_order_match"] == 0.5
    assert metrics["transition/tree"] == 0.5

    # a recomputed chunk must not double-count
    capture.add(
        role_ids=torch.tensor(roles),
        span_ids=torch.tensor(spans),
        argmax=torch.tensor(pred_ids),
        labels=torch.tensor(gold_ids),
    )
    assert [g for g, _ in capture.decode(tokenizer)] == [gold_a, gold_b]

    capture.reset()
    capture.arm(False)
    capture.add(
        role_ids=torch.tensor(roles),
        span_ids=torch.tensor(spans),
        argmax=torch.tensor(pred_ids),
        labels=torch.tensor(gold_ids),
    )
    assert capture.summarize(tokenizer) == {}


def test_loss_feeds_the_capture():
    """RoleCrossEntropyLoss must hand the argmax to the capture, not just the hit."""
    import torch

    from torchtitan.experiments.retroplanner.loss import RoleCrossEntropyLoss
    from torchtitan.experiments.retroplanner.trajectory import ROLE_DECISION

    vocab, length = 32, 6
    loss_fn = RoleCrossEntropyLoss(
        RoleCrossEntropyLoss.Config(global_vocab_size=vocab, max_spans=4)
    )
    labels = torch.tensor([[3, 4, 5, 6, 7, 8]])
    pred = torch.full((1, length, vocab), -10.0)
    for i, target in enumerate(labels[0].tolist()):
        pred[0, i, target] = 10.0
    roles = torch.full((1, length), ROLE_DECISION)
    spans = torch.zeros((1, length), dtype=torch.long)

    loss_fn.capture.arm()
    loss, _ = loss_fn(pred, labels, 1.0, role_ids=roles, span_ids=spans)
    assert torch.isfinite(loss)
    rows = loss_fn.capture._rows  # captured, one per decision token
    assert len(rows) == length
    # (microbatch, chunk, span, position, pred, gold)
    assert all(row[4] == row[5] for row in rows)
    # a second microbatch must not overwrite the first: span ids restart
    loss_fn.capture.end_microbatch()
    loss_fn(pred, labels, 1.0, role_ids=roles, span_ids=spans)
    assert len(loss_fn.capture._rows) == 2 * length
    assert len({(r[0], r[2]) for r in loss_fn.capture._rows}) == 2
