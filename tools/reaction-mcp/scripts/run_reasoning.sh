#!/bin/bash
# Teacher reasoning traces for the board episodes, on one machine. The episodes are split into
# one input shard per teacher card; each shard runs as its own process against the fleet.
#
#   ./run_reasoning.sh <side> <ncard>     # reads eps_v22_<side>_sNN.jsonl, NN = 00..ncard-1
#                                         # writes reason_v22_<side>_sNN.jsonl
#
# WORKERS MUST BE A MULTIPLE OF THE CARD COUNT. `stick_to` homes an episode on a replica with a
# per-PROCESS counter that starts at 0, and every shard process starts its own at 0. So each
# shard's W live episodes occupy homes 0..W-1 mod NCARD; if W is not a multiple of NCARD the
# low-numbered replicas carry more episodes than the rest in every shard, the shards add up in
# lockstep, and part of the fleet idles.
#
# WORKERS PER SHARD IS NOT A THROUGHPUT DIAL. `stick_to` pins an episode to one replica, so the
# number of live episodes per card sets the KV pressure on it: a late turn carries a long
# transcript plus its output, so only a few fit per card, and more workers buy queue depth,
# not throughput. Where prefix caching is on, more concurrent sequences also evict cached
# transcript prefixes, and since prefill dominates this workload each request then costs more
# than the extra concurrency wins. Raise WORKERS only after checking both the prefix-cache hit
# rate and requests/min; KV headroom alone will mislead.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
SIDE=${1:?loc or h100}; NCARD=${2:?number of cards}
cd "$(dirname "$0")"
RS=../data/route_search
P=${PY:-$(rp_py "$RP_ENV_BOARD")}
LOG=${LOGDIR:-/tmp/v21_reason}; mkdir -p $LOG
WORKERS=${WORKERS:-$RP_REASON_WORKERS}   # a multiple of NCARD (see above)
PORT0=${PORT0:-$RP_PORT_TEACHER}   # teacher fleet: PORT0..PORT0+NCARD-1

URLS=$(rp_urls $PORT0 $NCARD /v1)
up=0
for u in ${URLS//,/ }; do curl -sf -m 5 "$u/models" >/dev/null 2>&1 && up=$((up+1)); done
[ "$up" = "$NCARD" ] || { echo "!! teacher $up/$NCARD replicas answered on $URLS"; exit 1; }
echo "teacher $up/$NCARD"

COMMON="--kinds rank,open,done,final --min-words 12 --max-words-brief 45 --max-tokens 4096 \
  --base-url $URLS --model Qwen/Qwen3.8-27B --no-thinking --samples 2 --max-draws 12 --repair 3 \
  --reasoning-effort medium --variants 3"

# Shards are named s00, s01, ...; `seq -w` pads only to the widest operand, so a single-digit
# range would not be zero-padded. printf is explicit.
for i in $(seq 0 $((NCARD-1)) | while read n; do printf '%02d\n' "$n"; done); do
  IN=$RS/eps_v22_${SIDE}_s$i.jsonl
  [ -f "$IN" ] || { echo "!! missing $IN"; exit 1; }
  setsid nohup env BOARD_COST_NORM=1 BOARD_ROUTES_FULL=1 BOARD_ROUTE_AXES=1 \
                   BOARD_ROUTE_ORDER=axes_diverse \
    $P traj_route_reasoning_routeloop.py \
      --in "$IN" --out $RS/reason_v22_${SIDE}_s$i.jsonl \
      $COMMON --episode-workers $WORKERS > $LOG/${SIDE}_s$i.log 2>&1 &
done
sleep 45
echo "launched $NCARD shards x $WORKERS workers; logs in $LOG"
head -3 $LOG/${SIDE}_s00.log
