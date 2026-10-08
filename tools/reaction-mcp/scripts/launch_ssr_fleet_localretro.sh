#!/usr/bin/env bash
# Launch N replicas of the localretro (LocalRetro) SSR server + a least-outstanding
# proxy in front, and record the proxy URL as LOCALRETRO_SSR for traj_route_search.py.
#
#   bash scripts/launch_ssr_fleet_localretro.sh [N] [proxy_port]      # defaults: RP_N_SSR, the RP_PORT_* proxy
#   bash scripts/launch_ssr_fleet_localretro.sh stop
#
# The LocalRetro twin of launch_ssr_fleet.sh: same launcher, same SSR wire contract
# ({smiles, top_n}, no task field), served by predict_standalone.py with
# syntheseus' LocalRetroModel. Its ports come from config/env.sh
# (RP_PORT_LOCALRETRO_BASE.., proxy RP_PORT_LOCALRETRO), apart from the R-SMILES fleet's, so
# the two can run at once.
#
# The note below was written for RootAlignedModel and is kept because the launcher and
# its thread/MPS settings are shared.
#
# READ THIS BEFORE RAISING N. syntheseus' RootAlignedModel canonicalises its beam
# output with `multiprocessing.Pool(multiprocessing.cpu_count())` -- forked and torn
# down ON EVERY CALL, sized to the whole machine. That is one fork of a large process
# per core per prediction, so replica count multiplies into a fork storm rather than into
# throughput, with barely any time going to the model. predict_standalone.py caps that
# pool (serial map by default, RETRO_MP_WORKERS to change it); a fleet launched against an
# unpatched server will reproduce the storm, which is why this note lives next to the N.
#
# One call is 20 augmentations x beam 10 through a small OpenNMT transformer, about one
# core and a small share of a GPU. Replicas are spread round-robin over SSR_GPUS.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
cd "$(cd "$(dirname "$0")" && pwd)"                      # tools/reaction-mcp/scripts
API="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # predict_standalone.py lives here
LOG="$(cd .. && pwd)/logs/ssr_fleet"; mkdir -p "$LOG"

# The env's torch must be built for a CUDA version the driver supports. If it is not,
# torch.cuda.is_available() comes back False, syntheseus' device=None default places the
# model on CPU, and the fleet runs entirely on CPU while reporting itself healthy.
# $RP_ENV_SSR is expected to match the driver. Override with SSR_PY if another env is wanted.
PY="${SSR_PY:-$(rp_py "$RP_ENV_SSR")}"
PROXY_PY="$(rp_py "$RP_ENV_TOOLS")"     # stdlib only, any python does
BLOCK="${SSR_BLOCK:-$RP_PORT_LOCALRETRO_BASE}"                              # replica ports BLOCK..BLOCK+N-1
GPUS="${SSR_GPUS-$RP_GPUS}"                       # set SSR_GPUS= (empty) to place on CPU

if [ "${1:-}" = "stop" ]; then
  pkill -f "predict_standalone.py localretro --port 1[0-9][0-9][0-9][0-9]" && echo "replicas stopped" || echo "no replicas"
  pkill -f "pool_proxy.py --port ${2:-$RP_PORT_LOCALRETRO}" && echo "proxy stopped" || echo "no proxy"
  exit 0
fi

N="${1:-$RP_N_SSR}"
PROXY_PORT="${2:-$RP_PORT_LOCALRETRO}"
GPU_LIST=()
[ -n "$GPUS" ] && IFS=',' read -ra GPU_LIST <<< "$GPUS"

# Each replica is single-core work; predict_standalone setdefaults OMP to 16, which
# with N processes would oversubscribe the cores.
export OMP_NUM_THREADS="${SSR_OMP:-1}" MKL_NUM_THREADS="${SSR_OMP:-1}" \
       NUMEXPR_NUM_THREADS="${SSR_OMP:-1}" OPENBLAS_NUM_THREADS="${SSR_OMP:-1}"

# CUDA MPS. Without it a card runs one process's kernels at a time, so stacking
# replicas on a GPU redistributes throughput instead of adding it: per-card throughput
# caps however many processes share it, because the work is many microsecond kernels and
# the cost is the context switch between them. MPS funnels every client through one server context per GPU so
# the kernels actually overlap. Start the daemon once with
#   CUDA_MPS_PIPE_DIRECTORY=<dir> CUDA_MPS_LOG_DIRECTORY=<dir> nvidia-cuda-mps-control -d
# and export the same CUDA_MPS_PIPE_DIRECTORY here; replicas inherit it.
if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
  export CUDA_MPS_PIPE_DIRECTORY
  echo "MPS: replicas will attach to ${CUDA_MPS_PIPE_DIRECTORY}"
else
  echo "MPS: not attached (per-card throughput will cap regardless of N)"
fi

URLS=""
for i in $(seq 0 $((N - 1))); do
  PORT=$((BLOCK + i))
  URLS="${URLS:+$URLS,}http://127.0.0.1:${PORT}/predict"
  curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && continue
  GPU=""
  [ "${#GPU_LIST[@]}" -gt 0 ] && GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  ( cd "$API" && CUDA_VISIBLE_DEVICES="$GPU" nohup setsid "$PY" -u predict_standalone.py \
      localretro --port "$PORT" --host 127.0.0.1 \
      > "$LOG/replica_${PORT}.log" 2>&1 < /dev/null & )
done
echo "launched up to $N replicas on ports ${BLOCK}..$((BLOCK + N - 1)) on ${GPUS:-CPU}"

echo "waiting for /health (each replica downloads nothing) ..."
deadline=$(( $(date +%s) + ${SSR_WARMUP_TIMEOUT:-1800} ))
live=0
while :; do
  live=0
  for i in $(seq 0 $((N - 1))); do
    curl -sf -m 2 "http://127.0.0.1:$((BLOCK + i))/health" >/dev/null 2>&1 && live=$((live + 1))
  done
  echo "  $live/$N ready"
  [ "$live" -ge "$N" ] && break
  [ "$(date +%s)" -ge "$deadline" ] && { echo "  timeout with $live/$N — continuing"; break; }
  sleep 15
done

# The proxy only knows the replica list it was given, so it is rebuilt, not adopted.
pkill -f "pool_proxy.py --port ${PROXY_PORT}" 2>/dev/null && sleep 1
# Admit MORE than one request per replica: predict_standalone runs a batch worker that
# folds whatever is queued into one model batch, so a replica held at 1 in-flight is a
# replica decoding batches of 1. Too far above N, though, the proxy's own socket backlog
# can start resetting connections.
INFLIGHT="${SSR_MAX_INFLIGHT:-$((N * 4))}"
nohup setsid "$PROXY_PY" pool_proxy.py --port "$PROXY_PORT" --backend localretro \
  --replicas "$URLS" --max-inflight "$INFLIGHT" > "$LOG/proxy_${PROXY_PORT}.log" 2>&1 < /dev/null &
sleep 3
echo "proxy -> http://127.0.0.1:${PROXY_PORT}/predict"
curl -s -m 5 "http://127.0.0.1:${PROXY_PORT}/health" | head -c 300; echo
echo
echo "export LOCALRETRO_SSR=http://127.0.0.1:${PROXY_PORT}/predict"
