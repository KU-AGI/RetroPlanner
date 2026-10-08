#!/usr/bin/env python
"""Merge per-shard draw caches into the shared one.

A parallel FRESH search cannot share one draw cache file: every process holds the
whole dict and `DrawCache._flush` rewrites the file from its own view, so the last
flush wins and the others' menus are gone.  `--cache-readonly` avoids the clobber
by giving up the write-back entirely, which is right for a replay and wrong for a
fresh search: it leaves targets with no menus on disk, so no board can be rendered
for them at all.

The fix is one writable cache per shard plus this: menus are keyed by content, so
the merge is a dict update and the only thing that needs deciding is what to do
when two shards hold the same molecule.  They agree by construction (same model,
same draw, same top_k), so the incoming entry
is kept only when the target does not have it -- first writer wins, which makes
the merge order-independent and the result reproducible.

    python scripts/merge_draw_cache.py --into data/route_search/draw_cache/rsmiles__d0__k10.json \\
        --from 'data/route_search/draw_cache/shards/rsmiles__d0__k10.sh*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True, help="the shared cache to merge into")
    ap.add_argument("--from", dest="src", required=True, help="glob over shard caches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    into = Path(args.into)
    base = json.loads(into.read_text()) if into.exists() else {}
    print(f"# {into.name}: {len(base):,} molecules", file=sys.stderr)

    added = kept = 0
    files = sorted(glob.glob(args.src))
    if not files:
        raise SystemExit(f"no shard caches matched {args.src}")
    for f in files:
        d = json.loads(Path(f).read_text())
        new = 0
        for k, v in d.items():
            if k in base:
                kept += 1
                continue
            base[k] = v
            new += 1
        added += new
        print(f"#   {Path(f).name}: {len(d):,} molecules, {new:,} new", file=sys.stderr)

    print(f"# {added:,} added · {kept:,} already present · {len(base):,} total",
          file=sys.stderr)
    if args.dry_run:
        print("# dry run -- nothing written", file=sys.stderr)
        return 0
    # tmp + replace, as DrawCache does: a partial write here loses every menu in
    # the shared cache, and the searches that produced them are not cheap.
    tmp = str(into) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(base, fh)
    os.replace(tmp, into)
    print(f"# wrote {into}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
