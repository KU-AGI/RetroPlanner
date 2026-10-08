#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate a board corpus, report what is in it, and split it.

Every check here is for something that is SILENT at training time.  A call that
names a molecule the board never showed produces the same loss curve as a call
that names the wrong one; an episode the packer drops is simply absent.  So the
counts are taken once, written next to the run, and the run is judged against
them.

    python -m torchtitan.experiments.retroplanner.scripts.prepare_board_dataset \\
        --input .../sft_board/board.wire.jsonl \\
        --out-prefix .../sft_board/board \\
        --seq-len 16384 --val-frac 0.02

The input must be the AUGMENTED corpus (verify_harmony.py --augment): the
preamble on the row has to be the one vLLM renders, not the chat template's.
Split is by TARGET, never by instance -- a target's several routes share every
board they walk, so splitting by instance leaks the answer.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from torchtitan.experiments.retroplanner.board_trajectory import (
    aggregate_validation,
    to_board_trajectory,
    validate_board_record,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", default=None,
                    help="write <prefix>.train.jsonl / <prefix>.val.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-len", type=int, default=16384,
                    help="report how many episodes the packer would drop")
    ap.add_argument("--assets", default=None,
                    help="gpt-oss assets; enables the token-length report")
    ap.add_argument("--report", default=None, help="write the counts as json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input)]
    counts = aggregate_validation(validate_board_record(r) for r in rows)

    lengths: list[int] = []
    encoder = None
    if args.assets:
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.retroplanner.board_harmony import (
            BoardHarmonyEncoder,
        )

        tokenizer = HuggingFaceTokenizer(tokenizer_path=args.assets)
        encoder = BoardHarmonyEncoder(tokenizer)
    for r in rows:
        if encoder is not None:
            lengths.append(len(encoder.encode(to_board_trajectory(r)).input_ids) + 1)
        elif "tokens" in r:
            lengths.append(int(r["tokens"]))

    targets = sorted({r["target"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(targets)
    n_val = max(1, int(len(targets) * args.val_frac))
    val_targets = set(targets[:n_val])

    report: dict[str, float] = {f"check/{k}": v for k, v in counts.items()}
    report["data/records"] = len(rows)
    report["data/targets"] = len(targets)
    report["data/val_targets"] = len(val_targets)
    per_target = Counter(r["target"] for r in rows)
    report["data/routes_per_target_mean"] = sum(per_target.values()) / max(
        len(per_target), 1
    )
    if lengths:
        # THE SUM, not just the quantiles: BOARD_STEPS is two epochs of it,
        # `round(2 * packable tokens / (4 * seq_len))` on four ranks. Count it with this
        # encoder; concatenating the wire_* fields undercounts badly, because `wire_text`
        # is not among the fields the slim step keeps. Packable means it fits: a row over
        # seq_len is dropped by the packer, so its tokens are not trained on and must not
        # be in the total.
        # OVER EVERY ROW, val included, not the train split: that is the convention every
        # config's step count was set with. Change it only by recomputing all of them.
        packable = [n for n in lengths if n <= args.seq_len]
        report["length/packable_tokens"] = sum(packable)
        report["length/packable_rows"] = len(packable)
        lengths.sort()
        report["length/median"] = lengths[len(lengths) // 2]
        report["length/p95"] = lengths[int(0.95 * (len(lengths) - 1))]
        report["length/max"] = lengths[-1]
        dropped = sum(1 for n in lengths if n > args.seq_len)
        report["length/dropped_at_seq_len"] = dropped
        report["length/dropped_frac"] = dropped / len(lengths)

    failures = {k: v for k, v in counts.items() if not k.startswith("n_")}
    for key in sorted(report):
        print(f"{key:38} {report[key]}")
    if failures:
        print("\nFAILURES (each one is silent during training):")
        for k, v in sorted(failures.items()):
            print(f"  {k:36} {v}")
    else:
        print("\nno failures")

    if args.out_prefix:
        # Arrow infers a json column's type from the FIRST block it reads, so a
        # field that is null for the first few hundred episodes and a string
        # afterwards kills the load with "Couldn't cast array of type string to
        # null" -- and the traceback names neither the field nor the row.
        # `invalid` is exactly that: None on a clean episode, a NoMenu note on a
        # dropped one.  Normalising it here, where the split is written, keeps the
        # training input a stable schema instead of one that depends on which
        # episodes happen to land first.
        def _stable(row: dict) -> dict:
            for key in ("invalid",):
                if key in row:
                    row[key] = "" if row[key] is None else str(row[key])
            return row

        train_p = Path(f"{args.out_prefix}.train.jsonl")
        val_p = Path(f"{args.out_prefix}.val.jsonl")
        n_tr = n_va = 0
        with open(train_p, "w") as tr, open(val_p, "w") as va:
            for r in rows:
                line = json.dumps(_stable(r), ensure_ascii=False) + "\n"
                if r["target"] in val_targets:
                    va.write(line)
                    n_va += 1
                else:
                    tr.write(line)
                    n_tr += 1
        print(f"\nwrote {train_p} ({n_tr}) and {val_p} ({n_va}), split by target")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
