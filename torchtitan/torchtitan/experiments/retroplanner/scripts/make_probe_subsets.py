#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Depth-stratified probe subsets, so the held-out number is not a length average.

An episode's difficulty tracks its route's longest linear chain (`lls`): a
2-step route and an 8-step one are different tasks, and a split's mean is
dominated by whichever depth it happens to hold. Sampling a fixed number per
depth makes the train and val probes comparable to each other and stable across
runs, which is the point -- the train probe exists to be read AGAINST the val
probe, and a difference in depth mix would masquerade as memorisation.

    python -m torchtitan.experiments.retroplanner.scripts.make_probe_subsets \\
        --train .../board.train.jsonl --val .../board.val.jsonl \\
        --out-prefix .../board --per-depth 6 --depths 2-8
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def depth_of(row: dict) -> int | None:
    """How deep this episode goes, for stratifying the probes.

    A single-route episode says so in `route`; a multi-route one carries
    `route_queue` and `route` is None, so reading only `route` silently gives
    every multi-route episode a depth of None and empties the probe subsets.  The
    deepest route is the one that sets how far the episode had to search.
    """
    route = row.get("route") or {}
    for key in ("lls", "n_steps"):
        v = route.get(key)
        if isinstance(v, int):
            return v
    depths = [q.get("lls") or q.get("n_steps")
              for q in (row.get("route_queue") or [])]
    depths = [int(d) for d in depths if isinstance(d, (int, float))]
    return max(depths) if depths else None


def stratify(rows: list[dict], depths: range, per_depth: int,
             seed: int) -> tuple[list[dict], dict[int, int]]:
    buckets: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        d = depth_of(r)
        if d is not None and d in depths:
            buckets[d].append(r)
    rng = random.Random(seed)
    out, got = [], {}
    for d in depths:
        pool = buckets.get(d, [])
        rng.shuffle(pool)
        take = pool[:per_depth]
        out.extend(take)
        got[d] = len(take)
    return out, got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--per-depth", type=int, default=6)
    ap.add_argument("--depths", default="2-8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.depths.split("-"))
    depths = range(lo, hi + 1)

    for name, path in (("trainprobe", args.train), ("valprobe", args.val)):
        rows = [json.loads(line) for line in open(path)]
        subset, got = stratify(rows, depths, args.per_depth, args.seed)
        dest = Path(f"{args.out_prefix}.{name}.jsonl")
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in subset))
        short = {d: n for d, n in got.items() if n < args.per_depth}
        print(f"{name}: {len(subset)} episodes from {len(rows)} · per depth {got}"
              + (f" · SHORT of {args.per_depth} at depths {sorted(short)}" if short
                 else ""))
        print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
