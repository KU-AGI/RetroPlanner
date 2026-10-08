#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# gpt-oss-20b SFT on board episodes (tool calls against the AND-OR board).
#
#   ./torchtitan/experiments/retroplanner/run_board_sft.sh
#   CONFIG=sft_gpt_oss_20b_rs_smoke NGPU=8 ./.../run_board_sft.sh
#
# Extra arguments are forwarded, e.g. --training.steps 50 --optimizer.lr 5e-6

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# The torchtitan checkout this runs from is not the RetroPlanner repo (see
# RETROPLANNER.md), so the repo is located through RP_ROOT, not this file's path.
RP_ROOT=${RP_ROOT:-/mnt/data/RetroPlanner}
. "$RP_ROOT/config/env.sh"
CONDA_ENV=${CONDA_ENV:-$CONDA_ROOT/$RP_ENV_TRAIN}
BOARD_DIR=${BOARD_DIR:-$RP_MCP/data/sft_board}

export MODULE=${MODULE:-retroplanner}
export CONFIG=${CONFIG:-sft_gpt_oss_20b_rs}          # the reasoning arm
# 4, not $RP_NGPU: the board configs' step counts are epoch-matched to the rank count
# they were trained on (see config_registry.py), so a different NGPU changes the epochs.
export NGPU=${NGPU:-4}
export LOG_RANK=${LOG_RANK:-0}

export GPT_OSS_ASSETS=${GPT_OSS_ASSETS:-$RP_CACHE/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee}
export RETRO_INIT_CHECKPOINT=${RETRO_INIT_CHECKPOINT:-$RP_CACHE/gpt-oss-20b-bf16}
export RETRO_ATTN_BACKEND=${RETRO_ATTN_BACKEND:-flex}
# The corpus these default to has to match CONFIG: the config reads the same
# variables, so a default left at another corpus here silently trains the named
# config on that corpus and nothing in the log says so.
export BOARD_CORPUS=${BOARD_CORPUS:-rs_slim}
export BOARD_SFT_DATASET=${BOARD_SFT_DATASET:-$BOARD_DIR/$BOARD_CORPUS.train.jsonl}
export BOARD_SFT_VAL_DATASET=${BOARD_SFT_VAL_DATASET:-$BOARD_DIR/$BOARD_CORPUS.val.jsonl}
export BOARD_TRAINPROBE_DATASET=${BOARD_TRAINPROBE_DATASET:-$BOARD_DIR/$BOARD_CORPUS.trainprobe.jsonl}

export WANDB_PROJECT=${WANDB_PROJECT:-retroplanner-sft}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$CONFIG}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$BOARD_CORPUS}
export WANDB_RUN_JOB_TYPE=${WANDB_RUN_JOB_TYPE:-sft}
export WANDB_TAGS=${WANDB_TAGS:-retrosynthesis,harmony,board,tool-calls,multi-route}
export WANDB_DIR=${WANDB_DIR:-$REPO_ROOT/outputs/wandb}
export HF_HOME=${HF_HOME:-$RP_CACHE/huggingface}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

for f in "$BOARD_SFT_DATASET" "$BOARD_SFT_VAL_DATASET" "$BOARD_TRAINPROBE_DATASET"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; MISSING=1; }
done
if [ "${MISSING:-0}" = 1 ] || [ ! -f "$BOARD_SFT_DATASET" ]; then
    echo "Dataset not found: $BOARD_SFT_DATASET" >&2
    echo "Build it: render_board_episode.py --format harmony, then" >&2
    echo "verify_harmony.py --augment, then scripts/prepare_board_dataset.py." >&2
    exit 1
fi

export PATH="$CONDA_ENV/bin:$PATH"
cd "$REPO_ROOT"
exec ./run_train.sh "$@"
