# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Training configs for the SFT of gpt-oss-20b on board episodes: the reasoning arm,
the expansion-strategy arms and the reasoning-schema ladder.

    NGPU=4 MODULE=retroplanner CONFIG=sft_gpt_oss_20b_rs ./run_train.sh

Every config is ``_arm_abl`` over one corpus in ``$RP_DATA/sft_board`` -- the board setup is
shared (``_board_base``), and an arm differs only in its data, its step count and where it
checkpoints. Training is token-level cross-entropy on packed multi-turn episodes, with
only the assistant reasoning and actions supervised (see loss.py).
"""


import os


# This module is rsynced into an upstream torchtitan checkout and run from there
# (see RETROPLANNER.md), so it cannot find the RetroPlanner repo through
# __file__. The roots come from the environment; they mirror config/env.sh and
# config/paths.py.
RP_ROOT = os.environ.get("RP_ROOT", "/mnt/data/RetroPlanner")
RP_CACHE = os.environ.get("RP_CACHE", "/mnt/data/.cache")
# $RP_ROOT/tools/reaction-mcp/data -- holds every corpus below.
RP_DATA = os.path.join(RP_ROOT, "tools", "reaction-mcp", "data")


from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.gpt_oss import model_registry


from .board_dataset import BoardTrajectoryDataLoader
from .compat import apply_compat_shims
from .loss import CheckpointedChunkedLoss, RoleCrossEntropyLoss
from .state_dict_adapter import RetroGptOssStateDictAdapter
from .trainer import RetroSFTTrainer, RetroValidator


# Decision spans per packed sequence -- generous for any board episode.
MAX_SPANS = 256


def attention_backend() -> str:
    """Attention backend for both configs; override with RETRO_ATTN_BACKEND.

    "flex" is the default. "varlen" routes through torch's built-in FA4 bridge on SM 10.0 (see
    ``torchtitan.tools.utils.get_cuda_flash_attention_impl``), and that bridge
    is currently broken against the installed ``flash-attn-4`` release: torch's
    ``_fa4.py`` unpacks ``out, lse = module._flash_attn_fwd(...)`` but
    ``flash_attn.cute.interface._flash_attn_fwd`` returns a 4-tuple, so every
    varlen call dies with "too many values to unpack (expected 2)". Revisit
    once torch's FA4 bridge and flash-attn-4's cute interface agree on a
    signature again.
    """
    return os.environ.get("RETRO_ATTN_BACKEND", "flex")


# The released snapshot, used for the tokenizer and as the reference the
# expert-load preflight compares against.
RELEASED_HF_ASSETS = (
    f"{RP_CACHE}/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)
DEFAULT_HF_ASSETS = os.environ.get("GPT_OSS_ASSETS", RELEASED_HF_ASSETS)


# A bf16 copy of the same weights, produced by scripts/dequantize_gpt_oss.py.
# torchtitan's mxfp4 reader does not fill the expert tensors correctly, so the
# weights are dequantized once, offline, and loaded through the ordinary reader.
DEFAULT_INIT_CHECKPOINT = os.environ.get(
    "RETRO_INIT_CHECKPOINT", f"{RP_CACHE}/gpt-oss-20b-bf16"
)


def _board_base(board: str, steps: int, ckpt: str) -> RetroSFTTrainer.Config:
    """The board SFT setup every arm shares, over the corpus at ``board``.{train,val,trainprobe}.jsonl.

    gpt-oss-20b, loaded from the bf16 copy, trained on packed board episodes with the
    role-masked loss: AdamW at lr 1e-5, 20 warm-up steps then cosine decay. The sequence holds a whole board episode -- 98,304 tokens, split into
    48 loss chunks so the logits never materialise at once -- and one sequence per rank, so
    the global batch is the rank count. The validator runs 3 steps every 25 over the held-out
    targets, plus a train-probe split to read over-fitting against.
    """
    apply_compat_shims()
    model_spec = model_registry("20b", attn_backend=attention_backend())
    model_spec.state_dict_adapter = RetroGptOssStateDictAdapter
    return RetroSFTTrainer.Config(
        model_spec=model_spec,
        hf_assets_path=DEFAULT_HF_ASSETS,
        loss=CheckpointedChunkedLoss.Config(
            num_chunks=int(os.environ.get("BOARD_NUM_CHUNKS", 48)),
            loss_fn=RoleCrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
                max_spans=MAX_SPANS,
            ),
        ),
        dataloader=BoardTrajectoryDataLoader.Config(
            dataset_path=os.environ.get("BOARD_SFT_DATASET", f"{board}.train.jsonl"),
            max_spans_per_sequence=MAX_SPANS,
            infinite=True,
        ),
        optimizer=default_adamw(lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=int(os.environ.get("BOARD_SEQ_LEN", 98304)),
            steps=int(os.environ.get("BOARD_STEPS", steps)),
            max_norm=1.0,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=5,
            enable_wandb=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder=os.environ.get("BOARD_CKPT_FOLDER", ckpt),
            initial_load_path=DEFAULT_INIT_CHECKPOINT,
            initial_load_in_hf=True,
            initial_load_in_hf_quantized=False,
            initial_load_model_only=True,
            interval=int(os.environ.get("BOARD_CKPT_INTERVAL", 1_000_000)),
            keep_latest_k=2,
            last_save_model_only=True,
            export_dtype="bfloat16",
            # the dataloader state is per-corpus; a resumed arm must not inherit another's
            exclude_from_loading=["dataloader"],
        ),
        activation_checkpoint=FullAC.Config(),
        comm=CommConfig(train_timeout_seconds=900),
        validator=RetroValidator.Config(
            enable=True,
            freq=25,
            steps=3,
            probe_dataset_path=os.environ.get(
                "BOARD_TRAINPROBE_DATASET", f"{board}.trainprobe.jsonl"),
            dataloader=BoardTrajectoryDataLoader.Config(
                dataset_path=os.environ.get("BOARD_SFT_VAL_DATASET", f"{board}.val.jsonl"),
                max_spans_per_sequence=MAX_SPANS,
                infinite=False,
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# Expansion strategy: multi-route x {parallel frontier, serial frontier}, and single-route
#
# The multi-route arms are NO-REASONING boards over the same training targets, so the
# analysis channel is absent from both and the only axis between them is how the OPEN
# frontier advances. They separate "several routes per episode" from "several molecules
# open at once". The single-route arm holds one route per episode.
#
# STEPS ARE PER ARM: two epochs of that arm's own corpus, round(2 * packable tokens /
# (4 * 98304)) on four ranks. The step counts below are fallbacks; BOARD_STEPS from the
# prep report (scripts/prepare_board_dataset.py) overrides them.
# --------------------------------------------------------------------------------------

def _arm_abl(corpus: str, steps: int, ckpt: str) -> RetroSFTTrainer.Config:
    """The shared board config over one arm's corpus -- only the data and the step count move."""
    return _board_base(f"{RP_DATA}/sft_board/{corpus}", steps, ckpt)


def sft_gpt_oss_20b_mr_nr() -> RetroSFTTrainer.Config:
    """mr_nr: multi-route, PARALLEL frontier (--multi-route --branch-queue), no reasoning.

    Several molecules sit OPEN at once, so a rank turn chooses among candidates that belong
    to different parts of the tree. This is the RetroPlanner episode without the analysis
    channel (`--analysis` defaults to none).
    """
    return _arm_abl("mr_nr_slim", 233, "checkpoint_mr_nr")


def sft_gpt_oss_20b_mr_so() -> RetroSFTTrainer.Config:
    """mr_so: multi-route, SERIAL frontier (--multi-route --branch 1), no reasoning.

    One molecule is open at a time over the same routes per target (up to eight), so a rank
    turn only ever chooses among candidates for one product. Against mr_nr this isolates
    frontier width from route count.

    The serial frontier cannot replay some episodes the parallel one keeps, so this arm and
    mr_nr do not cover exactly the same targets.
    """
    return _arm_abl("mr_so_slim", 233, "checkpoint_mr_so")


def sft_gpt_oss_20b_rs() -> RetroSFTTrainer.Config:
    """rs: the REASONING arm -- the same board as mr_nr, plus the analysis channel.

    Rendered from the same command as the no-reasoning corpus (branch-queue, multi-route,
    up to eight routes a target, menu-show 10, the four BOARD_* switches) with
    `--analysis text --thoughts` added, so this arm and mr_nr differ in the channel and in
    nothing else.

    The analysis is the MOLECULE / DISCONNECTIONS / DECIDING trace written by the teacher
    (Qwen3.8-27B at reasoning_effort=medium -- serve the eval with the same value) and
    verified against the search graph. A turn whose trace never passed verification, after
    repair and regeneration, carries no analysis, so only verified reasoning is supervised.
    """
    # 2 epochs on 4 ranks (paper Table 3).
    return _arm_abl("rs_slim", 429, "checkpoint_rs")


def sft_gpt_oss_20b_rs_smoke() -> RetroSFTTrainer.Config:
    """Ten steps of the reasoning arm at full seq_len -- the memory check before a real run."""
    config = sft_gpt_oss_20b_rs()
    config.training.steps = 10
    config.metrics = MetricsProcessor.Config(log_freq=1, enable_wandb=False)
    config.checkpoint.enable = False
    return config


def sft_gpt_oss_20b_seq_nr() -> RetroSFTTrainer.Config:
    """seq_nr: SINGLE-ROUTE expansion, no reasoning -- one episode PER ROUTE.

    The third point on the expansion axis. mr_nr holds a target's routes in one episode with
    a parallel frontier, mr_so holds them with a serial one; seq_nr does not hold them
    together at all -- each route is its own short episode, so route diversity lives across
    episodes rather than inside one rollout.

    Its corpus is much larger than the multi-route ones, because the shared prefix is paid
    once per route. BOARD_STEPS is two epochs of ITS OWN corpus -- a step count shared with
    the other arms would train it for fewer passes over its data, which is not the axis.
    """
    return _arm_abl("seq_nr_slim", 888, "checkpoint_seq_nr")


# --------------------------------------------------------------------------------------
# Reasoning-schema ladder: what does each part of the analysis form contribute?
#
# Rungs over the same targets, episodes and actions, built by deleting reasoning lines from
# the full-schema corpus, so two rungs differ only in the lines one of them does not have:
#
#   L0      no analysis channel at all
#   L1      MOLECULE
#   L2      MOLECULE + DISCONNECTIONS
#   L3-div  + DECIDING with `diverging from`
#   L3-ord  + DECIDING with `ordered by`
#   L3      + DECIDING with both -- the full schema
#
# Every rung is served the developer message that describes its own form. The split is by
# target and seeded alike, so all rungs hold out the same targets.
#
# STEPS ARE PER RUNG: two epochs of that rung's own corpus. The rungs differ in tokens
# precisely because each adds a block, so a shared step count would train the shorter
# rungs for more passes over their own data.
# --------------------------------------------------------------------------------------

def _ladder(corpus_rung: str, steps: int) -> RetroSFTTrainer.Config:
    """The shared board config over a ladder rung -- only the data and the step count move.

    ``corpus_rung`` is the rung's name on disk, which is not always its name here.
    """
    return _arm_abl(f"ladder_{corpus_rung}", steps, f"checkpoint_ladder_{corpus_rung}")


def sft_gpt_oss_20b_ladder_L0() -> RetroSFTTrainer.Config:
    """L0: no analysis channel. Steps = 2 epochs of this rung's corpus."""
    return _ladder("L0", 330)


def sft_gpt_oss_20b_ladder_L1() -> RetroSFTTrainer.Config:
    """L1: MOLECULE only."""
    return _ladder("L1", 357)


def sft_gpt_oss_20b_ladder_L2() -> RetroSFTTrainer.Config:
    """L2: MOLECULE + DISCONNECTIONS, no DECIDING."""
    return _ladder("L2_disc", 377)


def sft_gpt_oss_20b_ladder_L3_ord() -> RetroSFTTrainer.Config:
    """L3-ord: MOLECULE + DISCONNECTIONS + DECIDING with `ordered by` only."""
    return _ladder("L3_ord", 391)


def sft_gpt_oss_20b_ladder_L3_div() -> RetroSFTTrainer.Config:
    """L3-div: MOLECULE + DISCONNECTIONS + DECIDING with `diverging from` only."""
    return _ladder("L3_div", 390)


def sft_gpt_oss_20b_ladder_L3() -> RetroSFTTrainer.Config:
    """L3: the full schema, DECIDING with both `diverging from` and `ordered by`.

    Trained under its own checkpoint over the ladder's targets rather than reusing the
    reasoning arm, which covers more targets and would confound the ladder with corpus size.
    """
    return _ladder("L4", 401)
