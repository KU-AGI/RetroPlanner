#!/usr/bin/env bash
# Main-table renderer -- USPTO-190 / ChEMBL-1000, B=50/300, imputed + solve-only.
#
# Usage:
#   bash evaluation/eval_protocol/runners/table_main.sh chembl   [outdir]
#   bash evaluation/eval_protocol/runners/table_main.sh uspto    [outdir]
#   bash evaluation/eval_protocol/runners/table_main.sh both
#
# Without SPEND, from_pooled falls back and undercharges, so build it first:
#   python evaluation/eval_protocol/runners/build_spend.py
#
# Fixed conventions live in metrics/table_geo.py itself: R-SMILES, unique-molecule budget,
# representative route = max (q_p + q_rt + u_c)/3 over the Pareto front, leaf-set dedup,
# Price = geometric mean (imputed) and median of the solved.
#
# Inputs per arm (set R1_uspto / RA_uspto / RP_uspto, R1_chembl / ... to override):
#   Retro-R1      10 USPTO runs / 4 ChEMBL runs, laid end to end
#   RetroAgent    10 experiments (within-run stamps + SPEND)
#   RetroPlanner  the board run
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../../config/env.sh"   # RP_MCP, RP_PROTOCOL, CONDA_ROOT
cd "$RP_MCP"
WHICH="${1:-both}"
OUT="${2:-results/tables_$(date +%Y%m%d_%H%M)}"
PY="$(rp_py "$RP_ENV_SCORE")" || exit 2
SPEND_DIR="${SPEND_DIR:-data/route_search/spend}"
mkdir -p "$OUT"

[ -f "$SPEND_DIR/ra_spend_uspto.json" ] || {
  echo "!! SPEND missing: $SPEND_DIR/ra_spend_uspto.json"
  echo "   python $RP_PROTOCOL/runners/build_spend.py  -- run this first"; exit 1; }

log(){ echo "[$(date '+%F %T')] $*"; }

# R1_<bench> RA_<bench> RP_<bench> pass through from the environment to table_geo.py.
run(){   # run <uspto|chembl>
  for B in 50 300; do for imp in 1 0; do
    tag=$([ "$imp" = 1 ] && echo imputed || echo solveonly)
    log "$1 b$B $tag"
    BENCH=$1 BUDGET=$B IMPUTE=$imp PRICE_CAP=1000 SPEND_DIR="$SPEND_DIR" \
      "$PY" "$RP_PROTOCOL/metrics/table_geo.py" > "$OUT/$1_b${B}_$tag.txt" 2>/dev/null
  done; done
}

case "$WHICH" in
  uspto)  run uspto ;;
  chembl) run chembl ;;
  both)   run uspto; run chembl ;;
  *) echo "usage: $0 {uspto|chembl|both} [outdir]"; exit 1 ;;
esac
log "done -> $OUT"
ls "$OUT"
