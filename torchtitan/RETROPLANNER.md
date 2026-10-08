# Training

SFT of gpt-oss-20b on board episodes — the checkpoint the RetroPlanner evaluation
serves. The objective is token-level cross-entropy on packed multi-turn episodes:
the assistant reasoning and actions are supervised, every input and tool-observation
token is masked, and the loss is normalised over supervised tokens across
data-parallel ranks.

It runs as a torchtitan experiment, so this directory keeps torchtitan's own
layout: our module sits at the path torchtitan expects it, and upstream is
referenced by patch rather than vendored.

```
torchtitan/
├── RETROPLANNER.md                          this file
├── patches/ours.patch                       our edits to 23 upstream files
└── torchtitan/experiments/retroplanner/     the experiment module
```

The layout is not cosmetic. The launch scripts take the torchtitan repo root as
`$(dirname "$0")/../../..` and `cd` there before training, writing
`outputs/wandb` under it. Flattened by one level that resolves above the repo,
and the run lands somewhere else entirely.

The module is `retroplanner` throughout: directory, module path, `MODULE=`, the
test file and the `experiments/__init__.py` registration inside the patch.

## Installing it

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
git apply "$RP_ROOT"/torchtitan/patches/ours.patch   # RP_ROOT: this repo
rsync -a "$RP_ROOT"/torchtitan/torchtitan/ ./torchtitan/
```

Copy the **inner** `torchtitan/` only — the outer one holds this file and the
patch, which do not belong in a torchtitan checkout. The patch is mostly
deletions: assertions and guards that do not hold on the training GPUs (B200).
It also adds `"retroplanner"` to `_supported_experiments`, without which the
module will not load.

## Running it

```bash
NGPU=4 ./torchtitan/experiments/retroplanner/run_board_sft.sh
CONFIG=sft_gpt_oss_20b_rs_smoke NGPU=4 ./torchtitan/experiments/retroplanner/run_board_sft.sh
```

`CONFIG` and `BOARD_CORPUS` default to `sft_gpt_oss_20b_rs` and `rs_slim`,
which match. They have to: the config reads the same
environment variables the launcher exports, so a corpus default left behind
trains the named config on another corpus and nothing in the log says so.
Change one, change the other.

## The configs

Every config in `config_registry.py` is `_arm_abl(corpus, steps, ckpt)` over one
shared board setup, `_board_base` (AdamW, lr 1e-5, 20 warm-up steps then cosine,
seq_len 98,304, one sequence per rank). An arm differs only in its corpus, its step
count and its checkpoint folder. Steps are two epochs of that arm's own corpus;
`scripts/prepare_board_dataset.py` reports the packable token count they come from.

| group | configs |
| --- | --- |
| the reasoning arm (the served model) | `sft_gpt_oss_20b_rs`, and `rs_smoke` (ten steps, the memory check) |
| expansion strategy, no reasoning | `mr_nr` (multi-route, parallel frontier), `mr_so` (multi-route, serial frontier), `seq_nr` (single-route, one episode per route) |
| reasoning-schema ladder | `ladder_L0` (no reasoning), `L1` (MOLECULE), `L2` (+DISCONNECTIONS), `L3_div` / `L3_ord` (+DECIDING with `diverging from` / `ordered by`), `L3` (full schema) |

All names carry the `sft_gpt_oss_20b_` prefix. The served checkpoint is
`$RP_MODEL_DIR` (see `config/env.sh`).

## Corpora

The training corpora are not in this repo. The configs read them from
`$RP_ROOT/tools/reaction-mcp/data/sft_board`;
`run_board_sft.sh` checks each dataset path before launching and stops if one is
missing, so a missing corpus surfaces as a refusal to start rather than as a wrong
run.

Build them with
`python -m torchtitan.experiments.retroplanner.scripts.prepare_board_dataset` —
[`README_board.md`](torchtitan/experiments/retroplanner/README_board.md) has the
arguments.

## Exporting a checkpoint

Use
[`scripts/export_to_hf.py`](torchtitan/experiments/retroplanner/scripts/export_to_hf.py),
not torchtitan's own `convert_to_hf.py`: on gpt-oss the upstream converter
drops every expert without erroring.
