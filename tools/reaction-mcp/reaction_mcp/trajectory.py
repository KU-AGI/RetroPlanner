"""Reference retrosynthesis trajectory driver (the node loop).

This encodes, as runnable code, the recipe an LLM follows when it builds a route
node-by-node with ``reaction_route_target`` — so the loop is reproducible instead
of re-improvised each prompt. It is a thin driver over the shared routing core
(``routing.route_node``); the MCP tool and this driver make the same decisions.

THE RECIPE
----------
Maintain a stack of open (not-yet-resolved) molecule nodes, a visited set (cycle
guard), and a step budget. Starting from the target, pop a node and route it:

    decision = route_node(node).decision           # operational: reaches stock?
    ┌ in_stock              → leaf; nothing to do
    ├ single_step           → take chosen_disconnection; all precursors are leaves
    ├ single_then_escalate  → take chosen_disconnection; its `leaves` are done,
    │                         push its `recurse_on` precursors as new nodes
    └ multi_step_search     → node is far from stock and no single step helps;
                              mark it unresolved (the pool has no multi-step
                              engine; run scripts/traj_route_search.py for a
                              real route search)

Stop when the stack is empty (every branch bottomed out in stock ⇒ solved) or the
step budget is hit (⇒ partial, with `unresolved` / `open` nodes reported).

Requires an operational context: a configured stock (``REACTION_STOCK`` or the
``stock=`` arg) and at least one live single-step retro backend. Without them,
``route_node`` can only return the complexity *prior* (no ``chosen_disconnection``
to act on), and the driver reports the node as unresolved (``reason=prior_only``).

CLI:
    REACTION_STOCK=paroutes_n1 \
      python -m reaction_mcp.trajectory "CC(=O)Nc1ccc(-c2ccccc2)cc1"
"""
from __future__ import annotations

import os
from typing import Any

from .pool.smiles import canonical
from .routing import route_node


def plan(
    target: str,
    stock: str | None = None,
    max_steps: int = 32,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build a retrosynthesis route for ``target`` via the node loop.

    Returns ``{target, solved, n_steps, steps, leaves, unresolved, visited}``.
    ``solved`` is True iff every branch bottomed out in stock within the budget.
    """
    if stock:
        os.environ["REACTION_STOCK"] = stock

    root = canonical(target) or target
    stack: list[str] = [root]
    visited: set[str] = set()
    steps: list[dict] = []
    leaves: set[str] = set()
    unresolved: list[dict] = []

    while stack and len(steps) < max_steps:
        node = stack.pop()  # DFS: resolve one branch deep before the next
        key = canonical(node) or node
        if key in visited:
            continue
        visited.add(key)

        r = route_node(node, top_k=top_k)
        decision, basis = r["decision"], r["basis"]

        if decision == "in_stock":
            leaves.add(node)
            continue

        if basis != "operational":
            # no stock / no single-step backend -> only the complexity prior, which
            # has no disconnection to act on. Report rather than guess.
            unresolved.append({"node": node, "reason": "prior_only", "prior": r["complexity_prior"]})
            continue

        if decision in ("single_step", "single_then_escalate"):
            cd = r["chosen_disconnection"]
            steps.append({"product": node, "precursors": cd["precursors"],
                          "decision": decision, "basis": basis})
            leaves.update(cd["leaves"])
            for m in cd["recurse_on"]:
                if (canonical(m) or m) not in visited:
                    stack.append(m)

        elif decision == "multi_step_search":
            # no single-step disconnection reaches stock and the pool has no
            # multi-step engine to delegate the subtree to
            unresolved.append({"node": node, "reason": "multi_step_search"})

    solved = not unresolved and not stack
    # a leaf that is actually a pushed-but-unpopped node (budget hit) isn't buyable
    open_nodes = [n for n in stack if (canonical(n) or n) not in visited]
    return {
        "target": target,
        "solved": solved,
        "n_steps": len(steps),
        "steps": steps,
        "leaves": sorted(leaves),
        "unresolved": unresolved,
        "open": open_nodes,
        "visited": len(visited),
        "budget_hit": len(steps) >= max_steps,
    }


def _print_tree(res: dict) -> None:
    print(f"target: {res['target']}")
    print(f"solved: {res['solved']}  steps: {res['n_steps']}  "
          f"leaves: {len(res['leaves'])}  unresolved: {len(res['unresolved'])}"
          f"{'  [BUDGET HIT]' if res['budget_hit'] else ''}")
    for i, s in enumerate(res["steps"]):
        print(f"  [{i}] ({s['decision']}) {s['product']}")
        print(f"      -> {' + '.join(s['precursors'])}")
    if res["leaves"]:
        print("  stock leaves:", ", ".join(res["leaves"]))
    for u in res["unresolved"]:
        print(f"  UNRESOLVED {u['node']}  ({u['reason']})")


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="reaction-mcp trajectory driver")
    ap.add_argument("target", help="target SMILES")
    ap.add_argument("--stock", help="REACTION_STOCK value (e.g. paroutes_n1, zinc_ik14)")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="dump full result as JSON")
    args = ap.parse_args()

    res = plan(args.target, stock=args.stock, max_steps=args.max_steps,
               top_k=args.top_k)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_tree(res)
