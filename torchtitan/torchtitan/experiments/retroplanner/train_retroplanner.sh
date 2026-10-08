#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# One command from a clean machine to a trained gpt-oss-20b.
#
#   ./torchtitan/experiments/retroplanner/train_retroplanner.sh
#   CONFIG=sft_gpt_oss_20b_rs ./torchtitan/.../train_retroplanner.sh
#   STEPS=1500 ./torchtitan/.../train_retroplanner.sh
#
# Defaults to sft_gpt_oss_20b_rs, the served model; CONFIG picks any config in config_registry.py.
#
# What this handles that a bare run_train.sh does not:
#
#   * the bf16 weights. The released gpt-oss-20b is mxfp4, torchtitan's
#     quantized reader does not fill the expert tensors correctly, and
#     nothing complains -- so the weights are dequantized once,
#     offline, and this rebuilds them if they are missing.
#   * waiting for the GPUs. Starting on top of a run that is still exiting
#     looks exactly like a fresh OOM.
#   * checking there is room. The final checkpoint is 39 GiB.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# The torchtitan checkout this runs from is not the RetroPlanner repo (see
# RETROPLANNER.md), so the repo is located through RP_ROOT, not this file's path.
RP_ROOT=${RP_ROOT:-/mnt/data/RetroPlanner}
. "$RP_ROOT/config/env.sh"
CONDA_ENV=${CONDA_ENV:-$CONDA_ROOT/$RP_ENV_TRAIN}
export PATH="$CONDA_ENV/bin:$PATH"
cd "$REPO_ROOT"

export CONFIG=${CONFIG:-sft_gpt_oss_20b_rs}
# 4: the board configs' step counts are 2 epochs at 4 ranks; another NGPU changes the epochs.
export NGPU=${NGPU:-4}
export TMPDIR=${TMPDIR:-$RP_CACHE/tmp}
unset CUDA_VISIBLE_DEVICES

RELEASED=${GPT_OSS_ASSETS:-$RP_CACHE/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee}
INIT=${RETRO_INIT_CHECKPOINT:-$RP_CACHE/gpt-oss-20b-bf16}
export RETRO_INIT_CHECKPOINT="$INIT"

# ---- 1. bf16 weights -------------------------------------------------------
if [ ! -f "$INIT/model.safetensors.index.json" ]; then
    echo "[1/4] dequantizing $RELEASED -> $INIT (39 GiB)"
    python -m torchtitan.experiments.retroplanner.scripts.dequantize_gpt_oss \
        --src "$RELEASED" --dest "$INIT"
else
    echo "[1/4] bf16 weights present at $INIT"
fi

# ---- 2. disk ---------------------------------------------------------------
FREE_GIB=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
echo "[2/4] disk: ${FREE_GIB} GiB free (the final checkpoint needs 39)"
if [ "$FREE_GIB" -lt 45 ]; then
    echo "      not enough room for the final save; free space and rerun" >&2
    exit 1
fi

# ---- 3. GPUs ---------------------------------------------------------------
echo "[3/4] waiting for the GPUs"
for _ in $(seq 1 240); do
    # 20 GiB, not 0: the reaction MCP server parks a couple of GiB on GPU 0 and
    # training is unaffected by it.
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>20000' | wc -l)
    procs=$(ps -eo args | grep -c "[t]orchtitan.train" || true)
    [ "$busy" -eq 0 ] && [ "$procs" -eq 0 ] && break
    sleep 10
done

# ---- 4. train --------------------------------------------------------------
echo "[4/4] $CONFIG on $NGPU GPUs"
EXTRA=()
[ -n "${STEPS:-}" ] && EXTRA+=(--training.steps "$STEPS")
exec ./torchtitan/experiments/retroplanner/run_sft.sh \
    --metrics.enable-tensorboard \
    --metrics.save-tb-folder "tb_${CONFIG}" \
    "${EXTRA[@]}" "$@"
