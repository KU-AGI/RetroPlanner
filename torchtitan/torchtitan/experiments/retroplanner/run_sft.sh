#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# gpt-oss-20b SFT on packed retrosynthesis agent trajectories.
#
#   ./torchtitan/experiments/retroplanner/run_sft.sh
#   CONFIG=sft_gpt_oss_20b_rs_smoke NGPU=8 ./torchtitan/experiments/retroplanner/run_sft.sh
#
# Any extra arguments are forwarded to torchtitan, e.g.
#   ./run_sft.sh --training.steps 50 --optimizer.lr 5e-6

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# The torchtitan checkout this runs from is not the RetroPlanner repo (see
# RETROPLANNER.md), so the repo is located through RP_ROOT, not this file's path.
RP_ROOT=${RP_ROOT:-/mnt/data/RetroPlanner}
. "$RP_ROOT/config/env.sh"
CONDA_ENV=${CONDA_ENV:-$CONDA_ROOT/$RP_ENV_TRAIN}

export NGPU=${NGPU:-$RP_NGPU}
export MODULE=retroplanner
export CONFIG=${CONFIG:-sft_gpt_oss_20b_rs}
export LOG_RANK=${LOG_RANK:-0}

# gpt-oss-20b weights + tokenizer. Override to train from a different snapshot.
export GPT_OSS_ASSETS=${GPT_OSS_ASSETS:-$(echo "$RP_CACHE"/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*/)}
export RETRO_SFT_DATASET=${RETRO_SFT_DATASET:-$RP_MCP/data/train_clean/sft_distilled/conv_full7_k50_paroutes.clean.train.jsonl}
export RETRO_SFT_VAL_DATASET=${RETRO_SFT_VAL_DATASET:-${RETRO_SFT_DATASET/.train.jsonl/.val.jsonl}}

# varlen needs FA3; see the environment note in README.md before switching.
export RETRO_ATTN_BACKEND=${RETRO_ATTN_BACKEND:-flex}

# Weights & Biases. torchtitan reads these env vars in WandBLogger; set
# WANDB_MODE=offline to record locally without a network round trip.
export WANDB_PROJECT=${WANDB_PROJECT:-retroplanner-sft}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$CONFIG}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-gpt-oss-20b}
export WANDB_RUN_JOB_TYPE=${WANDB_RUN_JOB_TYPE:-sft}
# WANDB_TAGS, not WANDB_RUN_TAGS: torchtitan passes WANDB_RUN_TAGS straight
# into wandb.init(tags=...) as a string, and wandb then iterates it character
# by character, so "a,b" becomes the tags "a", ",", "b". WANDB_TAGS is wandb's
# own env var and is split on commas properly.
export WANDB_TAGS=${WANDB_TAGS:-retrosynthesis,harmony,role-loss}
export WANDB_DIR=${WANDB_DIR:-$REPO_ROOT/outputs/wandb}

export HF_HOME=${HF_HOME:-$RP_CACHE/huggingface}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# The MoE grouped matmul and varlen attention kernels dominate; keep the
# tokenizer's own threads out of the dataloader workers.
export TOKENIZERS_PARALLELISM=false

if [ ! -f "$RETRO_SFT_DATASET" ]; then
    echo "Dataset not found: $RETRO_SFT_DATASET" >&2
    echo "Run scripts/prepare_dataset.py first (see README.md)." >&2
    exit 1
fi

export PATH="$CONDA_ENV/bin:$PATH"
cd "$REPO_ROOT"
exec ./run_train.sh "$@"
