"""Target-complexity routing: single-step vs multi-step retrosynthesis.

An LLM building a retrosynthesis trajectory node-by-node needs to decide, at each
partial-trajectory node, whether a one-shot **single-step** disconnection is
enough or the target is complex enough to warrant a full **multi-step tree
search**. This module scores that complexity from RDKit descriptors and returns
a decision the LLM can act on (or override).

The score is a *weighted sum* (not a mean) of five normalized complexity
dimensions, scaled to 0-100, then bucketed by two thresholds:

    score <  T_LOW               -> "single_step"            (simple: one disconnection)
    T_LOW <= score < T_HIGH      -> "single_then_escalate"   (try single, escalate if unsolved)
    score >= T_HIGH              -> "multi_step_search"       (complex: launch tree search)

`featurize` is RDKit-only (SAscore via the RDKit Contrib ``sascorer``), so it runs
in the base reaction-mcp env with no extra deps. The weights/thresholds/ramps are
the tuned defaults; override via the module constants if you recalibrate.
"""
from __future__ import annotations

import os
import sys
from typing import Any

# --- ring/macrocycle definitions (documented so the score is reproducible) ---
_MACROCYCLE_MIN_RING = 12  # smallest ring size counted as a macrocycle (IUPAC)

_SASCORER = None


def _sascorer():
    """Lazily import the RDKit Contrib SA_Score scorer (cached)."""
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig

        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer  # type: ignore

        _SASCORER = sascorer
    return _SASCORER


def featurize(smiles: str) -> dict[str, Any]:
    """Extract the complexity descriptors that drive routing.

    Returns a dict with: ``SAscore`` (1-10 synthetic-accessibility), ``n_fused``
    (bonds shared by >=2 rings), ``n_bridge`` (bridgehead atoms), ``n_spiro``
    (spiro atoms), ``fsp3`` (fraction of sp3 carbons), ``macrocycle`` (0/1, any
    ring >= 12), ``n_stereocenters`` (assigned + unassigned).

    Raises ``ValueError`` on an unparseable SMILES.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    # fused bonds: a bond shared by >= 2 SSSR rings (the degree of ring fusion)
    n_fused = sum(
        1 for b in range(mol.GetNumBonds()) if ring_info.NumBondRings(b) >= 2
    )

    try:
        stereo = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        n_stereocenters = len(stereo)
    except Exception:  # noqa: BLE001 - fall back to assigned-only count
        n_stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

    return {
        "SAscore": float(_sascorer().calculateScore(mol)),
        "n_fused": int(n_fused),
        "n_bridge": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "n_spiro": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "fsp3": float(Descriptors.FractionCSP3(mol)),
        "macrocycle": 1 if any(s >= _MACROCYCLE_MIN_RING for s in ring_sizes) else 0,
        "n_stereocenters": int(n_stereocenters),
    }


# ===========================================================================
# Routing score (weighted sum of normalized dimensions -> 0..100 -> decision)
# ===========================================================================

def _clip01(x: float) -> float:  # clamp to [0, 1]
    return max(0.0, min(1.0, x))


def _ramp(x: float, lo: float, hi: float) -> float:  # lo->0, hi->1 (clamped)
    return _clip01((x - lo) / (hi - lo))


# per-dimension weights (sum = 1.0)
W = {"sascore": 0.45, "ring": 0.25, "fsp3": 0.15, "macro": 0.10, "stereo": 0.05}

# Decision thresholds on the 0..100 score. These are a WEAK PRIOR, not the
# criterion. This complexity score does not by itself separate single-step from
# multi-step targets, because route depth is governed by the target-to-stock gap,
# not absolute complexity. So the authoritative routing decision is OPERATIONAL
# (single-step retro reaches stock? -> reaction_route_target probe); this score
# only pre-orders / provides a cheap fallback. Thresholds sit on the score's
# typical range.
T_LOW, T_HIGH = 10.0, 30.0


def routing_score(f: dict[str, Any]) -> tuple[float, dict, dict]:
    """Weighted-sum complexity score from a ``featurize`` result."""
    ring_raw = f["n_fused"] * 1.0 + f["n_bridge"] * 1.5 + f["n_spiro"] * 1.5

    # 1) normalize each dimension to [0, 1]
    comp = {
        "sascore": _ramp(f["SAscore"], 2.5, 6.5),     # SAscore 2.5~6.5
        "ring": _ramp(ring_raw, 0, 4),                # fused+bridged+spiro weighted 0~4
        "fsp3": _ramp(f["fsp3"], 0.2, 0.9),           # sp3 fraction 0.2~0.9
        "macro": float(f["macrocycle"]),              # 0 or 1
        "stereo": _ramp(f["n_stereocenters"], 0, 6),  # stereocenters 0~6
    }

    # 2) weighted sum -> 0..100 (weighted sum, NOT a mean)
    contrib = {k: W[k] * v for k, v in comp.items()}
    score = 100 * sum(contrib.values())
    return score, comp, contrib


def route(f: dict[str, Any]) -> dict[str, Any]:
    """Map a ``featurize`` result to a single/multi retrosynthesis decision."""
    score, comp, contrib = routing_score(f)
    # 3) bucket by threshold
    if score < T_LOW:
        decision = "single_step"
    elif score < T_HIGH:
        decision = "single_then_escalate"
    else:
        decision = "multi_step_search"
    return {
        "score": round(score, 1),
        "decision": decision,
        "normalized": comp,
        "contribution": contrib,
    }


def route_smiles(smiles: str) -> dict[str, Any]:
    """Convenience: ``featurize`` then ``route``, returning both."""
    features = featurize(smiles)
    result = route(features)
    result["features"] = features
    return result


__all__ = ["featurize", "routing_score", "route", "route_smiles", "W", "T_LOW", "T_HIGH"]
