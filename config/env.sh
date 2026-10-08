# Shell-side paths for the RetroPlanner tree. Source it, don't copy literals:
#
#     . "$(dirname "$0")/../../config/env.sh"
#
# Every value is overridable from the environment, so a different machine only
# needs to export the four roots rather than edit scripts. The names follow the
# convention the launchers use (CONDA_ROOT points at the *envs* directory, not at
# the conda prefix).

# The repo this file lives in.
RP_ROOT="${RP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where the heavy inputs live: checkpoints, vendored backend repos, stocks, caches.
# This repository itself -- they sit in place, at the paths below, and git ignores them.
# Point it at another tree only to import from it with config/import_worktree.sh.
RP_WORKTREE="${RP_WORKTREE:-$RP_ROOT}"

# Conda *envs* directory. The backends cannot share one env -- each retro model
# pins its own torch. tools/reaction-mcp/docs/ENVIRONMENTS.md is the map.
CONDA_ROOT="${CONDA_ROOT:-/mnt/data/miniconda3/envs}"

# Everything a script touches is addressed through this repo, and sits inside it --
# config/import_worktree.sh brings the heavy parts in.
RP_MCP="$RP_ROOT/tools/reaction-mcp"                   # models/ external/ data/ results/
RP_BOARD="${RP_BOARD:-$RP_ROOT/evaluation/board}"      # the board harness
RP_PROTOCOL="$RP_ROOT/evaluation/eval_protocol"        # results/ rows/ derived/
RP_BASELINE="$RP_ROOT/external/baseline"               # R-SMILES repo (rsmiles_server.py)
RP_SCI_DATA="$RP_ROOT/external/sci-data"               # gold routes, PaRoutes, stocks
RP_CHECKPOINTS="$RP_ROOT/checkpoints"                  # served LLM, single-step and forward weights

# Model/compile caches. Kept off $HOME because the images are large.
RP_CACHE="${RP_CACHE:-/mnt/data/.cache}"
export HF_HOME="${HF_HOME:-$RP_CACHE/huggingface}"
# Single-step and forward weights, read from the repo rather than ~/.cache and $HF_HOME.
export SYNTHESEUS_CACHE_DIR="${SYNTHESEUS_CACHE_DIR:-$RP_CHECKPOINTS/syntheseus}"
export REACTIONT5_MODEL="${REACTIONT5_MODEL:-$RP_CHECKPOINTS/ReactionT5v2-forward}"

export RP_ROOT RP_WORKTREE CONDA_ROOT RP_CACHE RP_MCP RP_BOARD RP_PROTOCOL RP_BASELINE RP_SCI_DATA RP_CHECKPOINTS

# ---------------------------------------------------------------------------------------
# Shared run settings. Every launcher, runner and trainer reads its defaults from here,
# so one machine layout is written down once. Override any of them from the environment.
# ---------------------------------------------------------------------------------------

# GPUs the scripts may place work on, and how many that is.
RP_GPUS="${RP_GPUS:-0,1,2,3,4,5,6,7}"
RP_NGPU="${RP_NGPU:-$(awk -F, '{print NF}' <<< "$RP_GPUS")}"

# Ports. A fleet takes BASE..BASE+N-1 for its replicas and one port for its proxy.
RP_PORT_LLM="${RP_PORT_LLM:-8300}"                  # vLLM, the served checkpoint: 8300..
RP_PORT_FORWARD="${RP_PORT_FORWARD:-8090}"          # ReactionT5v2 forward (rt axis): 8090..
RP_PORT_MENU="${RP_PORT_MENU:-9019}"                # menu_cache_proxy in front of R-SMILES
RP_PORT_RSMILES="${RP_PORT_RSMILES:-9020}"          # R-SMILES SSR proxy
RP_PORT_RSMILES_BASE="${RP_PORT_RSMILES_BASE:-10000}"
RP_PORT_LOCALRETRO="${RP_PORT_LOCALRETRO:-9021}"    # LocalRetro SSR proxy
RP_PORT_LOCALRETRO_BASE="${RP_PORT_LOCALRETRO_BASE:-11000}"
RP_PORT_TEACHER="${RP_PORT_TEACHER:-8080}"          # reasoning-generation teacher fleet: 8080..

# Replica counts.
RP_N_LLM="${RP_N_LLM:-$RP_NGPU}"                    # one vLLM replica per card
RP_N_FORWARD="${RP_N_FORWARD:-$RP_NGPU}"
RP_N_SSR="${RP_N_SSR:-32}"                          # single-step replicas per fleet

# Concurrency. RP_WORKERS is threads per process for every board / search driver;
# RP_NSHARD the processes a runner splits its targets into (the drivers are GIL-bound,
# so shards, not threads, are what scale).
RP_WORKERS="${RP_WORKERS:-8}"
RP_NSHARD="${RP_NSHARD:-8}"
RP_REASON_WORKERS="${RP_REASON_WORKERS:-16}"        # reasoning generation; a multiple of the
                                                    # teacher card count (see run_reasoning.sh)

# vLLM serving of the checkpoint.
RP_GPU_UTIL="${RP_GPU_UTIL:-0.70}"
RP_CTX="${RP_CTX:-131072}"
RP_MODEL_DIR="${RP_MODEL_DIR:-$RP_CHECKPOINTS/retroplanner}"   # the served checkpoint (GPT-OSS-20B SFT, sft_gpt_oss_20b_rs step 429)
RP_MODEL_NAME="${RP_MODEL_NAME:-retroplanner}"

# Conda envs, by role. Evaluation runs in ONE env (requirements.txt): vLLM, the R-SMILES
# servers, the board drivers, the forward model and the scorer share torch 2.9.1. Training
# needs its own (a torch nightly for torchtitan). Each role can still be pointed elsewhere.
RP_ENV="${RP_ENV:-retroplanner}"
RP_ENV_TOOLS="${RP_ENV_TOOLS:-$RP_ENV}"             # proxies, merges, small python
RP_ENV_BOARD="${RP_ENV_BOARD:-$RP_ENV}"             # board eval drivers, forward fleet
RP_ENV_SSR="${RP_ENV_SSR:-$RP_ENV}"                 # predict_standalone single-step replicas
RP_ENV_VLLM="${RP_ENV_VLLM:-$RP_ENV}"               # the vllm binary that serves the checkpoint
RP_ENV_SCORE="${RP_ENV_SCORE:-$RP_ENV}"             # table_geo.py
RP_ENV_TRAIN="${RP_ENV_TRAIN:-torchtitan}"

export RP_GPUS RP_NGPU RP_PORT_LLM RP_PORT_FORWARD RP_PORT_MENU RP_PORT_RSMILES \
  RP_PORT_RSMILES_BASE RP_PORT_LOCALRETRO RP_PORT_LOCALRETRO_BASE RP_PORT_TEACHER \
  RP_N_LLM RP_N_FORWARD RP_N_SSR RP_WORKERS RP_NSHARD RP_REASON_WORKERS \
  RP_GPU_UTIL RP_CTX RP_MODEL_DIR RP_MODEL_NAME \
  RP_ENV RP_ENV_TOOLS RP_ENV_BOARD RP_ENV_SSR RP_ENV_VLLM RP_ENV_SCORE RP_ENV_TRAIN

# rp_ports BASE N -> "BASE,BASE+1,...": the comma list a fleet of N answers on.
rp_ports() { local i o=""; for ((i = 0; i < $2; i++)); do o+="${o:+,}$(($1 + i))"; done; printf '%s' "$o"; }
# rp_urls BASE N SUFFIX -> "http://127.0.0.1:BASE/SUFFIX,..."
rp_urls() { local p o=""; for p in $(rp_ports "$1" "$2" | tr , ' '); do o+="${o:+,}http://127.0.0.1:$p$3"; done; printf '%s' "$o"; }

# rp_py <env-name> -- absolute interpreter path, or a hard failure.
#
# Worth the two lines: a sweep launched against an env that has moved produces
# logs but no result files, because every replica dies on a missing interpreter
# and the driver has nothing to write. Fail at launch, not silently at collection
# time.
rp_py() {
  local _p="$CONDA_ROOT/$1/bin/python"
  [ -x "$_p" ] || { echo "FATAL: conda env '$1' has no interpreter at $_p" >&2; return 2; }
  printf '%s' "$_p"
}
