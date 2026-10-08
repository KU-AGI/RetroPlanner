"""Single-vs-multi routing decision for one retrosynthesis node.

Shared core behind the ``reaction_route_target`` MCP tool (``server.py``) and the
reference trajectory driver (``trajectory.py``). Keeping it here means both agree
on the criterion.

The criterion is **operational** (see the calibration note in ``complexity.py``):
molecular complexity does not by itself predict route depth, so the authoritative
signal is *does a single-step disconnection reach purchasable stock?*. At a node:

  1. if the target itself is in stock            -> ``in_stock`` (leaf)
  2. run single-step retro; for each candidate precursor set, check stock:
     * some candidate is *fully* in stock         -> ``single_step``
     * best candidate is *partially* in stock      -> ``single_then_escalate``
     * nothing reaches stock                        -> ``multi_step_search``

The molecular-complexity score is carried as a *prior* and used as the decision
only when the operational probe can't run (no stock configured, or no live
single-step backend).

For an actionable node decision, ``chosen_disconnection`` names the precursor set
to take, split into ``leaves`` (buyable, done) and ``recurse_on`` (the next nodes).
"""
from __future__ import annotations

from typing import Any

from .complexity import route_smiles
from .pool import Task, aggregate, get_pool
from .stock import check_one as _stock_check_one, stock_status as _stock_status


def _operational_route(smiles: str, top_k: int) -> dict | None:
    """Operational decision: does a single-step disconnection reach stock?

    Returns ``None`` (caller falls back to the complexity prior) when routing
    can't be probed — no stock configured or no live single-step retro backend.
    """
    st = _stock_status()
    if not st.get("stock_configured"):
        return None
    if _stock_check_one(smiles).get("in_stock"):
        return {"decision": "in_stock", "target_in_stock": True,
                "chosen_disconnection": None,
                "reason": "target is already a purchasable building block"}
    pool = get_pool()
    results = pool.run(Task.RETRO, {"product": smiles, "target": smiles, "top_k": top_k})
    if not results or not any(r.ok for r in results):
        return None  # no live single-step backend -> degrade to prior
    agg = aggregate(results, top_k=top_k, backbone="consensus")
    cands = agg["candidates"]
    if not cands:
        return {"decision": "multi_step_search", "target_in_stock": False,
                "chosen_disconnection": None,
                "reason": "no single-step disconnection proposed"}

    per: list[dict] = []
    best_full: dict | None = None
    best_partial: dict | None = None
    best_partial_frac = 0.0
    best_frac = 0.0
    for c in cands:  # cands are consensus-ranked, so first full/partial hit is best
        mols = c.get("molecules", [])
        flags = [bool(_stock_check_one(m).get("in_stock")) for m in mols]
        frac = (sum(flags) / len(flags)) if flags else 0.0
        entry = {"precursors": mols, "in_stock": flags, "frac_in_stock": round(frac, 2)}
        per.append(entry)
        best_frac = max(best_frac, frac)
        if flags and all(flags):
            if best_full is None:
                best_full = entry
        elif frac > best_partial_frac:
            best_partial_frac, best_partial = frac, entry

    if best_full is not None:
        decision, chosen = "single_step", best_full
    elif best_partial is not None:
        decision, chosen = "single_then_escalate", best_partial
    else:
        decision, chosen = "multi_step_search", None

    chosen_disconnection = None
    if chosen is not None:
        pairs = list(zip(chosen["precursors"], chosen["in_stock"]))
        chosen_disconnection = {
            "precursors": chosen["precursors"],
            "in_stock": chosen["in_stock"],
            "leaves": [m for m, f in pairs if f],        # buyable -> done
            "recurse_on": [m for m, f in pairs if not f],  # -> next nodes
            "frac_in_stock": chosen["frac_in_stock"],
        }
    return {"decision": decision, "target_in_stock": False,
            "best_frac_in_stock": round(best_frac, 2),
            "chosen_disconnection": chosen_disconnection,
            "candidates_stock": per[:top_k]}


def route_node(smiles: str, top_k: int = 10, probe: bool = True) -> dict[str, Any]:
    """Route one node: operational decision (default) + complexity prior.

    Raises ``ValueError`` on an unparseable SMILES (via the complexity prior).
    """
    prior = route_smiles(smiles)  # may raise ValueError on bad SMILES
    op = _operational_route(smiles, top_k) if probe else None
    if op is not None:
        basis, decision = "operational", op["decision"]
        chosen = op.get("chosen_disconnection")
    else:
        basis = "complexity_prior" if probe else "complexity_prior(requested)"
        decision, chosen = prior["decision"], None
    return {
        "smiles": smiles,
        "decision": decision,
        "basis": basis,  # "operational" (authoritative) | "complexity_prior" (fallback)
        "chosen_disconnection": chosen,
        "operational": op,
        "complexity_prior": prior,
    }


__all__ = ["route_node"]
