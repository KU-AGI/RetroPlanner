# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate, split and report on a retrosynthesis trajectory corpus.

Run this before training. It answers, per record and in aggregate:

  format   does every assistant turn have the think/act shape, and does the
           action parse against the grammar?
  gold     does the disconnection the assistant committed match the children
           the search tree recorded? (Compare with --canonicalize inchikey14:
           a plain string comparison reports tautomers as failures.)
  state    is each user message the correct successor of the previous action --
           turn index, expansion budget, the echo of an expansion, the pieces
           a commit opens?
  length   how many trajectories will the packer drop at a given seq_len?

Outputs ``<stem>.clean.train.jsonl``, ``<stem>.clean.val.jsonl``,
``<stem>.rejected.jsonl`` and ``<stem>.report.json`` next to the input, and
optionally logs the same numbers to a wandb run so the dataset a training run
consumed is recorded alongside the run itself.

    python -m torchtitan.experiments.retroplanner.scripts.prepare_dataset \\
        --input .../conv_full7_k50_paroutes.oss.jsonl \\
        --canonicalize inchikey14 --seq-len 16384 --wandb
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from torchtitan.experiments.retroplanner.trajectory import (
    aggregate,
    identity_canonicalizer,
    inchikey14_canonicalizer,
    validate_record,
)

# Issues that make a trajectory unusable as supervision, as opposed to ones
# worth counting but not worth deleting data over. A tree that lists a reagent
# twice still teaches the right disconnection; an ungrammatical action does not.
REJECT_CODES = frozenset(
    {
        "format/no_system_message",
        "format/message_role_sequence",
        "format/assistant_shape_or_grammar",
        "format/think_empty",
        "format/n_turns_mismatch",
        "action/mismatch_actions_field",
        "gold/committed_children_mismatch",
        "gold/unfinished",
        "gold/done_premature",
        "state/done_not_last",
        "state/turn_header_missing",
        "state/budget",
        "state/budget_overspent",
        "state/expand_target_not_open",
        "state/expansion_not_echoed",
        "state/candidate_missing",
        "state/rank_target_not_ready",
        "state/route_not_opened",
        "state/route_pieces_mismatch",
    }
)


# Fields the trainer reads. ``mols`` and ``tree`` are deliberately dropped:
# their keys are per-record molecule ids, and datasets.load_dataset("json")
# infers one Arrow struct for the whole file, so an open key set either errors
# out or silently nulls most rows. They are only needed by the validation that
# already ran here, and the report records what was kept.
TRAIN_FIELDS = (
    "route_idx",
    "combo_idx",
    "benchmark",
    "n_turns",
    "n_expansions",
    "actions",
    "messages",
)


def slim(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in TRAIN_FIELDS if k in record}


def token_lengths(records: list[dict[str, Any]], hf_assets_path: str, seq_len: int):
    """Encoded length of each record, and how many exceed ``seq_len``."""
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.experiments.retroplanner.harmony import HarmonyTrajectoryEncoder
    from torchtitan.experiments.retroplanner.trajectory import to_trajectory

    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=hf_assets_path)
    encoder = HarmonyTrajectoryEncoder(tokenizer, current_date="2025-06-01")
    lengths = [len(encoder.encode(to_trajectory(r))) for r in records]
    lengths.sort()
    n = len(lengths)
    return {
        "tokens_total": sum(lengths),
        "tokens_p50": lengths[n // 2],
        "tokens_p90": lengths[int(n * 0.9)],
        "tokens_p99": lengths[int(n * 0.99)],
        "tokens_max": lengths[-1],
        "seq_len": seq_len,
        "dropped_at_seq_len": sum(1 for x in lengths if x > seq_len),
        "dropped_at_seq_len_frac": sum(1 for x in lengths if x > seq_len) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-stem", type=Path, default=None)
    parser.add_argument(
        "--canonicalize",
        choices=("string", "inchikey14"),
        default="inchikey14",
        help="How to compare fragment SMILES. inchikey14 needs RDKit.",
    )
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16384,
        help="Report how many trajectories the packer would drop at this length.",
    )
    parser.add_argument(
        "--hf-assets-path",
        default=None,
        help="gpt-oss tokenizer directory. Enables the token-length report.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="retro-sft")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    stem = args.output_stem or args.input.with_suffix("").with_suffix("")
    canonicalize = (
        inchikey14_canonicalizer()
        if args.canonicalize == "inchikey14"
        else identity_canonicalizer
    )

    records = [json.loads(line) for line in args.input.open()]
    reports = [validate_record(r, canonicalize=canonicalize) for r in records]
    summary = aggregate(reports)
    summary["canonicalize"] = args.canonicalize
    summary["input"] = str(args.input)

    kept, rejected = [], []
    for record, report in zip(records, reports, strict=True):
        blocking = sorted(set(report.issues) & REJECT_CODES)
        if blocking:
            rejected.append({"route_idx": report.route_idx, "issues": blocking})
        else:
            kept.append((record, report))

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac)) if kept else 0
    val, train = kept[:n_val], kept[n_val:]

    train_path = Path(f"{stem}.clean.train.jsonl")
    val_path = Path(f"{stem}.clean.val.jsonl")
    rejected_path = Path(f"{stem}.rejected.jsonl")
    report_path = Path(f"{stem}.report.json")

    for path, rows in ((train_path, train), (val_path, val)):
        with path.open("w") as fh:
            for record, _ in rows:
                fh.write(json.dumps(slim(record)) + "\n")
    with rejected_path.open("w") as fh:
        for row in rejected:
            fh.write(json.dumps(row) + "\n")

    summary["kept"] = len(kept)
    summary["rejected"] = len(rejected)
    summary["train"] = len(train)
    summary["val"] = len(val)
    summary["reject_codes"] = REJECT_CODES and sorted(REJECT_CODES)

    if args.hf_assets_path:
        summary["length"] = token_lengths(
            [r for r, _ in kept], args.hf_assets_path, args.seq_len
        )

    report_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "reject_codes"}, indent=2)
    )
    print(
        f"\nwrote {train_path}\n      {val_path}\n      {rejected_path}\n      {report_path}"
    )

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"prep-{args.input.stem}",
            job_type="dataset",
            config={
                "input": str(args.input),
                "canonicalize": args.canonicalize,
                "val_frac": args.val_frac,
                "seq_len": args.seq_len,
            },
        )
        flat = {
            f"dataset/{k}": v for k, v in summary.items() if isinstance(v, (int, float))
        }
        flat.update(
            {f"dataset/issue/{k}": v for k, v in summary["issue_counts"].items()}
        )
        for k, v in summary.get("length", {}).items():
            flat[f"dataset/{k}"] = v
        run.summary.update(flat)
        table = wandb.Table(columns=["issue", "records"])
        for code, count in sorted(summary["issue_counts"].items()):
            table.add_data(code, count)
        run.log({"dataset/issues": table})
        artifact = wandb.Artifact(f"{args.input.stem}-clean", type="dataset")
        artifact.add_file(str(report_path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
