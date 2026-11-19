#!/usr/bin/env python
"""Turn a closed-loop eval dump into a pooled_*.jsonl the scoring pipeline reads.

The feasibility axes are NOT reimplemented here.  A route from the agent has to be
scored by exactly the code that scores every other row of the table --
`traj_route_score.py` for the per-reaction caches and `traj_route_label.py` for the
route-level vector -- or the numbers are comparable to nothing.  So this only
reshapes: agent routes in, pooled records out.

    python scripts/eval_dump_to_pooled.py \\
        --dump ../results/board_eval_uspto190.jsonl \\
        --ds board_eval_uspto190 --model board-20b --algo agent \\
        --out ../data/route_search/pooled_board_eval_uspto190.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import traj_route_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", default="board_eval")
    ap.add_argument("--model", default="board-20b")
    ap.add_argument("--algo", default="agent")
    ap.add_argument("--draw", default="d0")
    args = ap.parse_args()

    rows, n_targets, n_stock = [], 0, 0
    for line in open(args.dump):
        rec = json.loads(line)
        n_targets += 1
        # A target that is itself in stock closes with no reaction at all.  It has
        # no route to score and counting it as a solve is stock leakage, not
        # planning, so it is carried as a flag rather than a record.
        # `routes` is the DECISIVE episode's list.  Under the iterative protocol a target
        # is searched many times and `routes_union` is what survived across those restarts,
        # deduplicated by step set -- scoring `routes` there would drop everything the
        # restarts found and report a diversity axis from one episode of twenty.  Prefer
        # the union wherever the runner wrote one.
        routes = rec.get("routes_union")
        if routes is None:
            routes = rec.get("routes") or []
        if rec.get("solved") and not routes:
            n_stock += 1
        for i, route in enumerate(routes):
            steps = [[p, list(rs)] for p, rs in route["steps"]]
            leaves = sorted({r for _, rs in steps for r in rs}
                            - {p for p, _ in steps})
            set_hash = C.set_hash(steps) if hasattr(C, "set_hash") else None
            row = {
                "target": rec["target"],
                "steps": steps,
                "n_steps": len(steps),
                "leaves": leaves,
                "sources": [[args.model, args.algo, args.draw]],
                "first_rank": i,
                "rollout": rec.get("rollout", 0),
                "ds": args.ds,
                "n_models": 1,
                "models": [args.model],
                "algos": [args.algo],
                "n_sources": 1,
                "seq_dupes_of_set": 1,
                "is_gold": False,
                "route_label": route.get("label"),
                "agent_calls": rec.get("calls"),
                "agent_turns": rec.get("turns"),
                "agent_stop": rec.get("stop"),
            }
            if set_hash:
                row["set_hash"] = set_hash
                row["seq_hash"] = set_hash
            rows.append(row)

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"# {n_targets} targets -> {len(rows)} routes "
          f"({len(rows) / max(n_targets, 1):.2f} routes/target)")
    print(f"# {n_stock} targets closed with no reaction (the target is itself in "
          f"stock) -- excluded from the route set")
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
