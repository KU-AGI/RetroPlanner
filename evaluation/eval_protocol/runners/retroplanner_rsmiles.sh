#!/usr/bin/env bash
# RetroPlanner x R-SMILES on USPTO-190 -- the evaluation protocol of the paper.
#
# THE SETTING, sentence by sentence (the defaults below are exactly this):
#
#   "ends each rollout at its first complete route"   STOP_ON_SOLVE=1      --stop-episode-on-solve
#   "and restarts"                                    ITER_STOP_ON_SOLVE=0 the target keeps going
#   "spending the remaining budget on alternative     unique-molecule      ITER_MAX_BUDGET=500 counts
#    routes"                                          budget               distinct molecules: re-opening
#                                                                          is free, and it accumulates
#                                                                          across restarts
#   up to 40 rollouts                                 ITER_MAX_ROLLOUTS=40
#   "does not shuffle the environment and instead     TEMPERATURE=0.6      per-rollout seed, so restarts
#    samples its reasoning and decisions at           (fixed menu order)   differ and the run reproduces
#    temperature 0.6"
#
# ITER_MAX_CALLS (total opens, revisits included) is set out of the way, so the unique budget
# is the one that binds -- the same unit RetroAgent and Retro-R1 count. Each route is stamped
# with the unique budget at which it was found, so every budget cell up to ITER_MAX_BUDGET is
# read off this one run.
#
# What it does, in order:
#   1. frees the GPUs of any vLLM, and brings up the R-SMILES menu on $RP_PORT_MENU (RP_N_SSR root_aligned
#      replicas behind it) if it is not answering;
#   2. checks the forward fleet ($RP_PORT_FORWARD.., the rt axis) is up -- it is not restarted;
#   3. gates on menu latency: a starved SSR does not fail loudly, the menu comes back empty
#      and the harness records a chemical dead end. It refuses to start, and warns mid-run;
#   4. serves the checkpoint on RP_N_LLM vLLM replicas, shards the targets, runs the iterative
#      board eval, and merges the shards into $FINAL.
#
# Ports, GPUs, workers, envs and the served model come from config/env.sh (RP_*); every
# setting below is overridable from the environment.
set -uo pipefail
# Paths come from config/env.sh, sourced relative to this file before the cd below.
# The board harness is $RP_BOARD (evaluation/board/). Override with RP_BOARD= to run a
# different checkout.
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
[ -f "$RP_BOARD/eval_board_agent.py" ] || {
  echo "!! board harness not found -- set RP_BOARD=/path/to/evaluation/board" >&2; exit 2; }
cd "$RP_MCP"
mkdir -p logs results
PY_TOOLS=$(rp_py "$RP_ENV_TOOLS") || exit 2
PY_VERL=$(rp_py "$RP_ENV_BOARD") || exit 2
VLLM_BIN="$CONDA_ROOT/$RP_ENV_VLLM/bin/vllm"
DEV="$RP_PROTOCOL/developer/dev_retroplanner.txt"
MENU_PORT=${MENU_PORT:-$RP_PORT_MENU}                   # menu_cache_proxy over the rsmiles fleet
MENU_URL=${MENU_URL:-http://127.0.0.1:$MENU_PORT/predict}
RUNTAG=${RUNTAG:-retroplanner}
S=${OUT_ROOT:-logs}
M=${MODEL_DIR:-$RP_MODEL_DIR}
TAG=${TAG:-$RP_MODEL_NAME}
CTX=${CTX:-$RP_CTX}
NSHARD=${NSHARD:-$RP_NSHARD}
WORKERS=${WORKERS:-$RP_WORKERS}
GPU_UTIL=${GPU_UTIL:-$RP_GPU_UTIL}
IFS=, read -ra GPU_LIST <<< "$RP_GPUS"
N_LLM=$RP_N_LLM
# -- the paper's setting (see the header) --------------------------------------------------
# ends each rollout at its first complete route
STOP_ON_SOLVE=${STOP_ON_SOLVE:-1}
STOP_FLAG=$([ "$STOP_ON_SOLVE" = "1" ] && echo "--stop-episode-on-solve" || echo "")
# ... and restarts, rather than stopping the target at its first solve
ITER_STOP_ON_SOLVE=${ITER_STOP_ON_SOLVE:-0}
# spending the remaining budget on alternative routes: unique molecules, across restarts
ITER_MAX_BUDGET=${ITER_MAX_BUDGET:-500}
ITER_MAX_CALLS=${ITER_MAX_CALLS:-100000}               # total opens: kept out of the way
# up to 40 rollouts
ITER_MAX_ROLLOUTS=${ITER_MAX_ROLLOUTS:-40}
# no environment shuffle; variation across rollouts comes from sampling the agent
TEMPERATURE=${TEMPERATURE:-0.6}
# ILLEGAL_CAP: consecutive malformed actions before an episode ends. This protocol uses 5;
# the harness's own default is 30.
ILLEGAL_CAP=${ILLEGAL_CAP:-5}
FINAL=${FINAL:-results/forced_iter_${RUNTAG}_emols_div_t06_b300_ep40.jsonl}
# Menu filtering: drop invalid candidates and duplicate reactions, from a pool of 50.
export BOARD_MENU_FILTER=${BOARD_MENU_FILTER:-1} BOARD_MENU_DEDUP_RXN=${BOARD_MENU_DEDUP_RXN:-1}
export BOARD_MENU_POOL=${BOARD_MENU_POOL:-50}
export BOARD_COST_NORM=1 BOARD_ROUTES_FULL=1 BOARD_ROUTE_AXES=1
unset BOARD_ROUTE_ORDER BOARD_ANCESTRY BOARD_ROUTE_FRONT BOARD_PRICE_NOTE
log() { echo "[$(date '+%F %T')] $*"; }
kp() { local P; P=$(pgrep -f "$1" | tr '\n' ' '); [ -n "$P" ] && kill -9 $P 2>/dev/null; return 0; }

menu_ms() {
  local t0 t1
  t0=$(date +%s%N)
  curl -s -m 20 -o /dev/null -X POST "$MENU_URL" -H 'Content-Type: application/json' \
    -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O","top_n":10}' 2>/dev/null || return 1
  t1=$(date +%s%N); echo $(( (t1-t0)/1000000 ))
}
menu_ok() { curl -s -m 25 -X POST "$MENU_URL" -H 'Content-Type: application/json' \
    -d '{"smiles":"CCOC(=O)c1ccccc1","top_n":10}' 2>/dev/null | grep -c precursors; }
wait_menu() { local n; for t in $(seq 1 $1); do n=$(menu_ok)
    [ "${n:-0}" -gt 0 ] && { log "  :$MENU_PORT ready"; return 0; }; sleep 15; done
    log "  !! :$MENU_PORT not ready"; return 1; }
min_free() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
    | awk -F'[ ,]+' '{d=$3-$1; if(m==""||d<m) m=d} END{print m}'; }

grep -q "empty_retry_max = 6" "$RP_BOARD"/eval_board_agent.py || { echo "!! empty-menu retry patch missing"; exit 1; }
[ -d "$M" ] || { echo "!! checkpoint missing: $M"; exit 1; }

# -- 1. GPUs and the R-SMILES menu ---------------------------------------------------------
need=$(python3 -c "print(int(81559*$GPU_UTIL))")
kp 'vllm serve'; kp 'VLLM::EngineCore'; sleep 25
# The forward fleet and the root_aligned replicas share the cards and must not be killed, so
# the bar is vLLM's own requirement plus a small margin, not a round figure.
for t in 1 2 3; do
  fm=$(min_free); log "  min card free ${fm}MiB (need $need)"
  [ "${fm:-0}" -ge $((need + 900)) ] && break
  [ "$t" = 3 ] && { log "!! failed to free GPU -- aborting"; exit 1; }
  kp 'vllm serve'; kp 'VLLM::EngineCore'; sleep 25
done
if ! wait_menu 40; then
  log ":$MENU_PORT not responding -- restarting $RP_N_SSR root_aligned replicas"
  for ((i = 0; i < RP_N_SSR; i++)); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % RP_NGPU))]} nohup setsid "$CONDA_ROOT/$RP_ENV_SSR/bin/python" -u \
      "$RP_MCP/scripts/predict_standalone.py" root_aligned \
      --port $((RP_PORT_RSMILES_BASE + i)) --host 127.0.0.1 > logs/ra_$((RP_PORT_RSMILES_BASE + i)).log 2>&1 &
  done
  wait_menu 60 || { log "!! :$MENU_PORT failed -- aborting"; exit 1; }
fi

# -- 2. forward fleet, 3. menu latency -----------------------------------------------------
n=$(for p in $(rp_ports $RP_PORT_FORWARD $RP_N_FORWARD | tr , ' '); do curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$p/health; done | grep -c 200)
[ "$n" -ge "$RP_N_FORWARD" ] || { echo "!! forward fleet $n/$RP_N_FORWARD -- start it first (tools/reaction-mcp/scripts/feas_forward_fleet.sh)"; exit 1; }
log "forward fleet $n/$RP_N_FORWARD confirmed"
ms=$(menu_ms) || { echo "!! R-SMILES menu not responding"; exit 1; }
[ "$ms" -le 5000 ] || { echo "!! menu latency ${ms}ms > 5000 -- a starved board writes fake dead ends, aborting"; exit 1; }
log "menu latency ${ms}ms confirmed"

# -- 4. vLLM, shards, eval, merge ----------------------------------------------------------
log "vLLM $N_LLM replicas (util $GPU_UTIL, max-model-len $CTX, served-name $TAG)"
mkdir -p $S/vllm_${RUNTAG} $S/${RUNTAG}_shards
export HF_HOME=$RP_CACHE/huggingface HF_HUB_OFFLINE=1
for ((i = 0; i < N_LLM; i++)); do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % RP_NGPU))]} VLLM_CACHE_ROOT=$RP_CACHE/vllm/${RUNTAG}_$i \
  TORCHINDUCTOR_CACHE_DIR=$RP_CACHE/torchinductor/${RUNTAG}_$i \
  TRITON_CACHE_DIR=$RP_CACHE/triton/${RUNTAG}_$i \
  nohup setsid "$VLLM_BIN" serve "$M" \
    --served-model-name $TAG --host 127.0.0.1 --port $((RP_PORT_LLM + i)) \
    --tensor-parallel-size 1 --max-model-len $CTX --gpu-memory-utilization $GPU_UTIL \
    --enable-auto-tool-choice --tool-call-parser openai > $S/vllm_${RUNTAG}/rep$i.log 2>&1 &
done
r=0
for t in $(seq 1 90); do
  r=$(for p in $(rp_ports $RP_PORT_LLM $N_LLM | tr , ' '); do curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$p/v1/models; done | grep -c 200)
  [ "$r" -ge "$N_LLM" ] && { log "vLLM $N_LLM/$N_LLM ready"; break; }; sleep 20
done
[ "$r" -lt "$N_LLM" ] && { echo "!! vLLM $r/$N_LLM"; tail -6 $S/vllm_${RUNTAG}/rep0.log; exit 1; }
served=$(curl -s -m 5 http://127.0.0.1:$RP_PORT_LLM/v1/models | "$PY_TOOLS" -c \
  "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['max_model_len'], d['id'])" 2>/dev/null)
[ "$served" = "$CTX $TAG" ] || { echo "!! served '$served' != '$CTX $TAG'"; exit 1; }
log "ctx/model name match: $served"

"$PY_TOOLS" - <<PY
S="$S"
# Resume by passing only the remaining targets via TARGETS_FILE. OUTSUF writes this pass's
# output separately so it does not overwrite the existing out{i}.jsonl -- the merge below
# reads all out*.jsonl and dedups by target, so earlier passes' records survive and merge in.
L=[l for l in open("${TARGETS_FILE:-data/route_search/targets_uspto190.jsonl}") if l.strip()]
assert len(L)>0, "0 targets"
if not "${TARGETS_FILE:-}": assert len(L)==190, len(L)
for i in range($NSHARD):
    open(f"{S}/${RUNTAG}_shards/t{i}${OUTSUF:-}.jsonl","w").writelines(L[i::$NSHARD])
print("  shards", [sum(1 for _ in open(f"{S}/${RUNTAG}_shards/t{i}${OUTSUF:-}.jsonl")) for i in range($NSHARD)])
PY
U=$(rp_urls $RP_PORT_LLM $N_LLM /v1)
RT=$(rp_urls $RP_PORT_FORWARD $RP_N_FORWARD)
log "R-SMILES: stop-on-solve=$STOP_ON_SOLVE restart=$([ "$ITER_STOP_ON_SOLVE" = 0 ] && echo on || echo off) budget=$ITER_MAX_BUDGET unique rollouts<=$ITER_MAX_ROLLOUTS T=$TEMPERATURE, $NSHARD shards x $WORKERS workers"
# Collect only the board shard PIDs. A bare wait also waits for the vLLMs launched above,
# and `nohup setsid ... &` only detaches the session while the parent-child relation stays,
# so unless vLLM dies on its own it never returns.
BOARD_PIDS=()
for i in $(seq 0 $((NSHARD-1))); do
  ITER_MAX_CALLS=$ITER_MAX_CALLS ITER_MAX_ROLLOUTS=$ITER_MAX_ROLLOUTS ITER_PATIENCE=0 \
  ITER_STOP_ON_SOLVE=$ITER_STOP_ON_SOLVE ITER_MAX_BUDGET=$ITER_MAX_BUDGET \
  nohup "$PY_VERL" -W ignore \
    "$RP_BOARD"/eval_board_agent_forced_iterative.py \
    --targets $S/${RUNTAG}_shards/t$i${OUTSUF:-}.jsonl --model-url "$U" --model $TAG \
    --menu-url "$MENU_URL" --stock emols --rt "live:$RT" \
    --developer-file "$DEV" --two-stage --ctx-len $CTX \
    --budget 300 --max-turns 120 --max-tokens 3000 --reasoning medium \
    --menu-show 10 --signals q,p,rt --deciding p --handover flow --cutoff 0.05 \
    --illegal-cap $ILLEGAL_CAP $STOP_FLAG --temperature $TEMPERATURE --workers $WORKERS \
    --out $S/${RUNTAG}_shards/out$i${OUTSUF:-}.jsonl > $S/${RUNTAG}_shards/log$i${OUTSUF:-}.txt 2>&1 &
  BOARD_PIDS+=($!)
done
# Keep checking for menu starvation during the run -- a starved board silently writes fake dead ends
( while [ "$(pgrep -f 'eval_board_agent_forced_iterat''ive[.]py' | wc -l)" -gt 0 ]; do
    sleep 120; ms=$(menu_ms) || ms=99999
    [ "$ms" -gt 8000 ] && echo "[$(date +%H:%M:%S)] !! menu latency ${ms}ms -- board is starving"
  done ) > $S/${RUNTAG}_shards/menu_watch.log 2>&1 &
wait "${BOARD_PIDS[@]}"
"$PY_TOOLS" - <<MERGE
import glob, json
seen={}
for f in sorted(glob.glob("$S/${RUNTAG}_shards/out*.jsonl")):
    for l in open(f):
        if l.strip():
            try: seen[json.loads(l)["target"]]=l.rstrip()
            except Exception: pass
with open("$FINAL","w") as fh:
    for l in seen.values(): fh.write(l+chr(10))
print("  merged", len(seen), "-> $FINAL")
MERGE
log "done"
