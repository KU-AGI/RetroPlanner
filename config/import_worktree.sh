#!/usr/bin/env bash
# Brings the heavy inputs -- checkpoints, single-step and scorer weights, stocks, caches,
# run outputs -- into this repo, so that nothing a run reads lives outside it:
#
#     RP_WORKTREE=/path/to/that/tree bash config/import_worktree.sh
#
# HARD LINKS, NOT SYMLINKS. On the same filesystem every file is hard-linked (`cp -al`):
# no extra space, and the copy survives the source tree being moved or deleted. Across
# filesystems it falls back to a real copy. Hard links share an inode, so a file that is
# rewritten IN PLACE (an append, an editor that does not write-and-rename) changes in both
# trees; everything the pipeline writes goes through tmp + replace, which splits them.
#
# Idempotent: an existing destination is left as it is (remove it to re-import), and a
# symlink left by the old link-based setup is replaced. .gitignore names every destination.
#
# Sources outside RP_WORKTREE, each overridable:
#   RP_CKPT_SRC        the served checkpoint (an HF export of sft_gpt_oss_20b_rs); required
#   RP_SYNTHESEUS_SRC  syntheseus' downloaded weights   (default ~/.cache/torch/syntheseus)
#   RP_RT5_SRC         ReactionT5v2-forward HF snapshot (default: under $HF_HOME/hub)
#
# R-SMILES is the default single-step model. RP_WITH_LOCALRETRO=1 also brings in
# LocalRetro (its syntheseus weights and the LocalRetro repo localretro_server.py imports).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -n "${RP_WORKTREE:-}" ] && [ "$(cd "$RP_WORKTREE" && pwd)" != "$RP_ROOT" ] || {
  echo "set RP_WORKTREE to the tree that holds the inputs" >&2; exit 2; }
W="$RP_WORKTREE/tools/reaction-mcp"
B="$RP_WORKTREE/sci-orchestrator/baseline"

# bring SRC DST: hard-link copy of SRC at DST (real copy across filesystems). -L dereferences
# symlinks in the source, so the HF snapshot's blob links become files.
bring() {
  local src="$1" dst="$2"
  [ -e "$src" ] || { echo "!! missing: $src" >&2; return 1; }
  [ -L "$dst" ] && rm "$dst"                     # the link-based setup; never `rm -r` a link
  [ -e "$dst" ] && { echo "   kept   $dst"; return 0; }
  mkdir -p "$(dirname "$dst")"
  cp -alL "$src" "$dst" 2>/dev/null || { rm -rf "$dst"; cp -aL "$src" "$dst"; }
  echo "   copied $dst <- $src"
}

# tools/reaction-mcp: model weights, vendored backends (MolPrice, MORetro, ...), the menu
# caches / feasibility overlays / runs under data/, and run outputs.
# Other single-step models (Molecular Transformer, RetroSim) are not brought in: R-SMILES is
# the single-step model of the default setup, and nothing here reads them.
SKIP=" molecular_transformer retrosim_50k retrosim_full MolecularTransformer "
for d in models external; do
  for e in "$W/$d"/*; do
    case "$SKIP" in *" $(basename "$e") "*) continue;; esac
    bring "$e" "$RP_MCP/$d/$(basename "$e")"
  done
done
for d in data results; do bring "$W/$d" "$RP_MCP/$d"; done
for d in results rows derived; do bring "$W/eval_protocol/$d" "$RP_PROTOCOL/$d"; done
# Stocks and gold routes: PaRoutes n1/n5 stocks, the eMolecules SMILES stock and the
# USPTO-190 reference routes (reaction_mcp/stock.py).
for d in paroutes retro_star; do bring "$RP_WORKTREE/sci-orchestrator/data/$d" "$RP_SCI_DATA/$d"; done
# From the baseline checkouts, only the R-SMILES repo rsmiles_server.py imports. The
# baseline arms' own run logs are not brought in: the SPEND stamps build_spend.py derived
# from them already sit in data/route_search/spend/, which is all the tables read.
bring "$B/rsmiles" "$RP_BASELINE/rsmiles"
# LocalRetro, opt-in: the upstream checkout is LocalRetro_orig; here it is LocalRetro.
if [ "${RP_WITH_LOCALRETRO:-0}" = 1 ]; then
  for p in scripts LocalTemplate models data LICENSE; do
    bring "$B/LocalRetro_orig/$p" "$RP_BASELINE/LocalRetro/$p"
  done
fi
# reaction_mcp.scoring's inputs that no model directory ships, each from wherever it was
# built (skipped when unset): the per-axis caches, MolPrice's prediction tables, SCScore.
[ -n "${RP_NODE_SCORES_SRC:-}" ]  && bring "$RP_NODE_SCORES_SRC"  "$RP_MCP/data/node_scores"
[ -n "${RP_PRICE_TABLES_SRC:-}" ] && bring "$RP_PRICE_TABLES_SRC" "$RP_MCP/data/molprice"
[ -n "${RP_SCSCORE_SRC:-}" ]      && bring "$RP_SCSCORE_SRC"      "$RP_MCP/models/scscore"

# Checkpoints.
[ -e "$RP_MODEL_DIR" ] || [ -n "${RP_CKPT_SRC:-}" ] || {
  echo "set RP_CKPT_SRC to the served checkpoint (HF export) to import it" >&2; exit 2; }
[ -e "$RP_MODEL_DIR" ] && echo "   kept   $RP_MODEL_DIR" || bring "$RP_CKPT_SRC" "$RP_MODEL_DIR"
SYN="${RP_SYNTHESEUS_SRC:-$HOME/.cache/torch/syntheseus}"
SYN_KEYS="RootAligned_backward"; [ "${RP_WITH_LOCALRETRO:-0}" = 1 ] && SYN_KEYS+=" LocalRetro_backward"
for k in $SYN_KEYS; do bring "$SYN/$k" "$SYNTHESEUS_CACHE_DIR/$k"; done
RT5="${RP_RT5_SRC:-$(ls -d "$HF_HOME"/hub/models--sagawa--ReactionT5v2-forward/snapshots/*/ 2>/dev/null | head -1)}"
bring "${RT5%/}" "$REACTIONT5_MODEL"

# Symlinks inside the copied trees that still point into RP_WORKTREE are re-aimed at the
# same path inside this repo.
find "$RP_MCP" "$RP_PROTOCOL" "$RP_BASELINE" "$RP_SCI_DATA" -type l 2>/dev/null | while read -r l; do
  t=$(readlink "$l"); case "$t" in "$RP_WORKTREE"/*) ;; *) continue;; esac
  r=${t#"$RP_WORKTREE"/}
  case "$r" in
    tools/reaction-mcp/eval_protocol/*) n="$RP_PROTOCOL/${r#tools/reaction-mcp/eval_protocol/}";;
    tools/reaction-mcp/*)               n="$RP_MCP/${r#tools/reaction-mcp/}";;
    sci-orchestrator/data/*)            n="$RP_SCI_DATA/${r#sci-orchestrator/data/}";;
    sci-orchestrator/baseline/*)        n="$RP_BASELINE/${r#sci-orchestrator/baseline/}";;
    *) echo "!! $l -> $t stays pointing outside the repo" >&2; continue;;
  esac
  ln -sfn "$n" "$l"; echo "   relink $l -> $n"
done
