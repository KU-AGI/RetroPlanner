#!/usr/bin/env bash
# N replicas of the ReactionT5v2 forward model, one URL per replica, for the round-trip axis.
#
#   bash scripts/feas_forward_fleet.sh <N> [first_port]     # launch, prints REACTION_FORWARD_URL
#   bash scripts/feas_forward_fleet.sh stop
#
# No proxy. `reaction_mcp.scoring.roundtrip` takes a url_override per call and `feas_cost.RtScorer`
# rotates a comma-separated list, so a proxy would only add a process to keep alive and a
# second place for a 200-with-an-error-body to hide (a failed replica must never write its
# reactions into the round-trip cache as Nones).
#
# PORT 8090 AND UP, NEVER 8088. A forward server that ranks predictions from ZERO would turn
# every top-1 hit into rank 0 and the whole axis would read as "the chemistry is bad".
# `feas_cost.RtScorer.probe()` refuses to start unless each replica returns rank 1 for a
# reaction it must reproduce -- this script only keeps to a port range that avoids 8088.
#
# ENV: $RP_ENV_BOARD (transformers + a CUDA build of torch that matches the driver). The
# weights (sagawa/ReactionT5v2-forward) are read from $REACTIONT5_MODEL, the copy under
# checkpoints/ (config/env.sh), so no replica downloads anything.
#
# SIZING. The model is a T5-base, small per replica, so the constraint is free VRAM rather
# than replica count -- the GPUs may be shared with a vLLM run, and a replica that cannot
# reach CUDA falls back to CPU SILENTLY (`torch.cuda.is_available()` False, the model placed
# on CPU, the fleet reporting itself healthy at a fraction of the throughput). So each replica
# is pinned to one GPU with CUDA_VISIBLE_DEVICES and the launcher refuses a GPU with less than
# 4 GB free.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
cd "$(cd "$(dirname "$0")" && pwd)/.."                          # tools/reaction-mcp
LOG="$(pwd)/logs/feas_forward"; mkdir -p "$LOG"
PY="${FEAS_PY:-$(rp_py "$RP_ENV_BOARD")}"

if [ "${1:-}" = "stop" ]; then
  pkill -f "reactiont5_forward_server.py --port" && echo "stopped" || echo "nothing running"
  exit 0
fi

N="${1:-$RP_N_FORWARD}"
P0="${2:-$RP_PORT_FORWARD}"
mapfile -t FREE < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
IFS=, read -ra GPU_LIST <<< "$RP_GPUS"
NGPU="${#GPU_LIST[@]}"
[ "${#FREE[@]}" -gt 0 ] && [ "$NGPU" -gt 0 ] || { echo "no GPUs visible"; exit 1; }

URLS=()
launched=0
for ((i = 0; i < N; i++)); do
  gpu=${GPU_LIST[$(( i % NGPU ))]}
  if [ "${FREE[$gpu]}" -lt 4096 ]; then
    echo "!! GPU $gpu has only ${FREE[$gpu]} MiB free -- skipping replica $i. A replica that" \
         "cannot reach CUDA runs on CPU without saying so."
    continue
  fi
  port=$(( P0 + i ))
  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    echo "   port $port already listening -- reusing it"
    URLS+=("http://127.0.0.1:$port")
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" REACTIONT5_DEVICE=cuda:0 \
    nohup "$PY" scripts/reactiont5_forward_server.py --port "$port" \
    > "$LOG/rt5_$port.log" 2>&1 &
  URLS+=("http://127.0.0.1:$port")
  launched=$(( launched + 1 ))
  echo "   replica $i -> port $port on GPU $gpu (pid $!)"
done

echo "waiting for $launched replica(s) to load the model..."
for u in "${URLS[@]}"; do
  port="${u##*:}"
  for _ in $(seq 1 120); do
    curl -s --max-time 2 "$u" | grep -q reactiont5v2_forward && break
    sleep 2
  done
  curl -s --max-time 2 "$u" | grep -q reactiont5v2_forward \
    && echo "   $u ready" \
    || { echo "   $u DID NOT COME UP -- see $LOG/rt5_$port.log"; }
done

IFS=,; JOINED="${URLS[*]}"; unset IFS
echo
echo "export REACTION_FORWARD_URL='$JOINED'"
echo "# then: --feas-rt-url \"\$REACTION_FORWARD_URL\"  (RtScorer probes every replica)"
