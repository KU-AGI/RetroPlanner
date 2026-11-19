#!/usr/bin/bash
# One eval dump -> labeled_<stem>.jsonl, which is what BTR is computed from.
#
# THE AXES ARE NOT REIMPLEMENTED ANYWHERE. BTR is
#   plaus_passfrac x rt5 x (1 - min(1, cost_usd/1000))
# against the published route, so a dump has to go through the SAME three stages every
# other row of every table went through, or its number is comparable to nothing:
#
#   3  eval_dump_to_pooled.py   reshape only -- agent routes in, pooled records out
#   4  traj_route_score.py      fills the SHARED per-reaction caches (node_scores/*.json)
#   5  traj_route_label.py      route-level vector + Pareto rank -> labeled_<stem>.jsonl
#
# TWO THINGS THAT SILENTLY GIVE THE WRONG NUMBER:
#
#  * --rt-cache roundtrip_rt5 IS THE REPORTED AXIS. There are two round-trip caches on this
#    box; load_scores() hands back the Chemformer one, and the tables quote ReactionT5v2 from
#    roundtrip_rt5.json. Mixing them is not visible in any output.
#  * ROUND-TRIP COVERAGE IS NOT 100% BY DEFAULT. An unscored step is scored as a FAILURE, so
#    an arm whose pool is half-uncached loses on rt5 for being uncached rather than for being
#    wrong -- agent pools have measured 38-52% uncached while board dumps were 100%. The
#    coverage report at the end is not decoration; read it before quoting BTR.
#
# env: $RP_ENV_BOARD -- it needs syntheseus AND onnxruntime AND joblib together.
#
#   bash scripts/label_board_dump.sh $RP_PROTOCOL/results/rpfix_uspto190.jsonl rq_rpfix_uspto190
set -uo pipefail
DUMP=${1:?usage: label_board_dump.sh <dump.jsonl> <stem>}
STEM=${2:?usage: label_board_dump.sh <dump.jsonl> <stem>}
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
MCP="$RP_MCP"
PY="${LABEL_PY:-$(rp_py "$RP_ENV_BOARD")}"
TARGETS=${TARGETS:-$RP_PROTOCOL/targets_uspto190.jsonl}
_fp=${RP_PORT_FORWARD:-8090}; _fn=${RP_N_FORWARD:-8}
FWD=${REACTION_FORWARD_URL:-$(for p in $(seq "$_fp" $((_fp+_fn-1))); do printf "http://127.0.0.1:%s," $p; done | sed 's/,$//')}
L=$MCP/data/route_search
LOG=$MCP/logs/label_dump; mkdir -p "$LOG"
cd "$MCP" || exit 1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

[ -s "$DUMP" ] || { log "!! no dump at $DUMP"; exit 1; }
log "stage 3  pool   $DUMP -> pooled_$STEM.jsonl"
$PY scripts/eval_dump_to_pooled.py --dump "$DUMP" --ds "$STEM" \
    --model board-20b --algo agent --out "$L/pooled_$STEM.jsonl" \
    > "$LOG/${STEM}_pool.log" 2>&1
rc=$?; n=$(wc -l < "$L/pooled_$STEM.jsonl" 2>/dev/null || echo 0)
log "  rc=$rc routes=$n"
[ "$n" -gt 0 ] || { log "!! pooled file is empty -- nothing to score"; exit 1; }

log "stage 4a plausibility + molprice (CPU/ONNX)"
$PY -W ignore scripts/traj_route_score.py --stem "$STEM" --no-gold \
    --axis plausibility,molprice > "$LOG/${STEM}_plaus.log" 2>&1
log "  rc=$?"

# CONCURRENCY IS SIZED FROM THE POOL, because the failure is an OOM hours in.
# --rt-workers 32 --batch 64 is what the forward replicas OOM'd on once already: each
# replica balloons far past its 1.5GB of weights under sustained high concurrency, and a
# dead replica does not fail the run -- its reactions come back unscored, and an unscored
# step is scored as a FAILED one, so the arm loses BTR for an infrastructure event.
# Small pools keep the fast setting; a pool past ~100k routes drops to the combination
# that survived (8/32).  Override either with RT_WORKERS / RT_BATCH.
_np=$(wc -l < "$L/pooled_$STEM.jsonl" 2>/dev/null || echo 0)
if [ "$_np" -gt 100000 ]; then RTW=${RT_WORKERS:-8}; RTB=${RT_BATCH:-32}
else                           RTW=${RT_WORKERS:-32}; RTB=${RT_BATCH:-64}; fi
log "stage 4b round-trip on the forward fleet  (pool=$_np routes, rt-workers=$RTW batch=$RTB)"
REACTION_FORWARD_URL="$FWD" $PY -W ignore scripts/traj_route_score.py \
    --stem "$STEM" --no-gold --axis roundtrip --rt-cache roundtrip_rt5 \
    --rt-workers "$RTW" --batch "$RTB" \
    > "$LOG/${STEM}_rt.log" 2>&1
log "  rc=$?"

log "stage 4c coverage report -- an uncached step scores as a failed one"
$PY -W ignore scripts/traj_route_score.py --stem "$STEM" --no-gold \
    --axis plausibility,molprice,roundtrip --rt-cache roundtrip_rt5 --report \
    2>&1 | tail -12 | sed 's/^/    /'

log "stage 5  label -> labeled_$STEM.jsonl"
$PY -W ignore scripts/traj_route_label.py --stem "$STEM" --targets "$TARGETS" \
    > "$LOG/${STEM}_label.log" 2>&1
rc=$?; n=$(wc -l < "$L/labeled_$STEM.jsonl" 2>/dev/null || echo 0)
log "  rc=$rc labeled routes=$n"
log "done: $L/labeled_$STEM.jsonl"
