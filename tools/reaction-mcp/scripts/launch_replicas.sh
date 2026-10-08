#!/usr/bin/env bash
# Launch N replicas of a warm retro backend + a least-outstanding proxy in front.
#
#   bash scripts/launch_replicas.sh <backend> <N> [proxy_port]
#
# Backends built on pool_server.serve hold a global lock around predict(), so one
# server = one concurrent request. Under a concurrent fan-out the queue, not the
# model, dominates latency. N replicas behind the proxy restore the model's real
# throughput.
#
# Each backend owns a 100-port replica block (see the case below); the proxy takes
# <proxy_port> and is what REACTION_<NAME>_URL should point at. The ORIGINAL server
# on the base port joins the rotation too (it need not be restarted).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"
cd "$(dirname "$0")"

BACKEND="${1:?backend: localretro|rsmiles}"
N="${2:-8}"
PROXY_PORT="${3:-}"

# GPU backends spread their replicas round-robin over REPLICA_GPUS (comma list of
# ids). CPU backends ignore it.
REPLICA_GPUS="${REPLICA_GPUS:-$RP_GPUS}"
# Per-backend spec: env|script|base_port|use_gpu|proxy_port|replica_block.
# Replica blocks are 100 apart and proxy ports are their own range, so no two
# backends can collide (an overlapping block would let one backend's proxy
# silently front another model).
case "$BACKEND" in
  localretro)  ENV=localretro;  SCRIPT=localretro_server.py;  BASE=8097; USE_GPU=0; PROXY_DEFAULT=9003; BLOCK=9200 ;;
  rsmiles)     ENV=rsmiles;     SCRIPT=rsmiles_server.py;     BASE=8100; USE_GPU=1; PROXY_DEFAULT=9006; BLOCK=9500 ;;
  *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac
if [ "$N" -gt 100 ]; then echo "N must be <= 100 (replica block size)" >&2; exit 2; fi
IFS=',' read -ra _GPU_LIST <<< "$REPLICA_GPUS"
PROXY_PORT="${PROXY_PORT:-$PROXY_DEFAULT}"
PY="$(rp_py "$ENV")"
PROXY_PY="$(rp_py "$RP_ENV_TOOLS")"

# Per-replica CPU thread cap. torch's default (=cores/2) spins many OpenMP threads
# per process; with many replicas that oversubscribes the machine and every call
# thrashes.
THREADS="${REPLICA_CPU_THREADS:-4}"

# Concurrency the proxy admits at once. The GPU-backed model shares the GPUs, so
# beyond ~2 concurrent requests per GPU replicas only time-slice each other.
# Cap rsmiles near that point and leave the cheap CPU localretro open.
case "$BACKEND" in
  rsmiles) MAX_INFLIGHT="${MAX_INFLIGHT:-16}" ;;
  *)       MAX_INFLIGHT="${MAX_INFLIGHT:-0}"  ;;
esac

mkdir -p ../logs
URLS="http://127.0.0.1:${BASE}/predict"
for i in $(seq 0 $((N - 1))); do
  PORT=$((BLOCK + i))
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "replica $i already up on $PORT"
  else
    if [ "$USE_GPU" = "1" ]; then
      GPU="${_GPU_LIST[$((i % ${#_GPU_LIST[@]}))]}"
    else
      GPU=""
    fi
    CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      nohup "$PY" "$SCRIPT" --port "$PORT" \
      > "../logs/replica_${BACKEND}_${PORT}.log" 2>&1 &
    echo "replica $i -> port $PORT (pid $!) gpu=${GPU:-cpu}"
  fi
  URLS="${URLS},http://127.0.0.1:${PORT}/predict"
done

echo "waiting for replicas to warm up ..."
for i in $(seq 0 $((N - 1))); do
  PORT=$((BLOCK + i))
  for _ in $(seq 1 240); do
    curl -s -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ready": *true' && break
    sleep 5
  done
done

if curl -s -m 2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "proxy already listening on $PROXY_PORT — restart it to pick up new replicas"
else
  nohup "$PROXY_PY" pool_proxy.py \
    --port "$PROXY_PORT" --backend "$BACKEND" --replicas "$URLS" \
    --max-inflight "$MAX_INFLIGHT" \
    > "../logs/proxy_${BACKEND}.log" 2>&1 &
  echo "proxy -> http://127.0.0.1:${PROXY_PORT}/predict (pid $!)"
fi

# Record the proxy URL so consumers pick it up by sourcing one file instead of each caller hardcoding ports.
NAME=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')
ENVFILE="replica_urls.env"
touch "$ENVFILE"
grep -v "^export REACTION_${NAME}_URL=" "$ENVFILE" > "${ENVFILE}.tmp" || true
echo "export REACTION_${NAME}_URL=http://127.0.0.1:${PROXY_PORT}/predict" >> "${ENVFILE}.tmp"
mv "${ENVFILE}.tmp" "$ENVFILE"

echo
echo "recorded in scripts/${ENVFILE}:"
cat "$ENVFILE"
