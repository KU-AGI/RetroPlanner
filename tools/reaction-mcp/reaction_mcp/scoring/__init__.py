"""The scores the board puts on a candidate disconnection and on a molecule.

    plausibility   AiZynthFinder's USPTO filter policy, P(the reaction is feasible)
    roundtrip      rank at which a forward model recovers the product (ReactionT5v2 server)
    price          MolPrice, ln(USD/mmol) of a molecule (MORetro's numpy port)
    scscore        SCScore in [1, 5], synthetic complexity of a molecule

Every axis is cached per reaction or per molecule under NODE_SCORES; the board reads the
cache first and scores only what it lacks. Weights are read from where each model ships
(models/, external/), so nothing here depends on another checkout.
"""
import os
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[2]                       # tools/reaction-mcp
MODELS = MCP_ROOT / "models"
EXTERNAL = MCP_ROOT / "external"
# The per-axis caches: plausibility.json, roundtrip_rt5.json, molprice.json, ... keyed by
# rxn_key (reactions) or SMILES (molecules). RP_NODE_SCORES overrides the location.
NODE_SCORES = Path(os.environ.get("RP_NODE_SCORES") or MCP_ROOT / "data" / "node_scores")


def rxn_key(product: str, reactants) -> str:
    """The cache key of a disconnection: reactants deduplicated and sorted, not re-canonicalised."""
    return f"{product}>>" + ".".join(sorted(set(reactants)))
