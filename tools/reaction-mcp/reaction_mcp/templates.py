"""Named retro-template matching (the "which known reactions apply" layer).

Instead of asking the model to *author* a retro SMARTS (which LLMs do
unreliably), this matches a molecule against a corpus of **named** reaction
templates and returns the named reactions that could produce it. The model then
reasons over / selects grounded, human-named reactions (e.g. "Suzuki coupling",
"Buchwald-Hartwig amination", "Esterification of Carboxylic Acids").

Default corpus: Rxn-INSIGHT's named-reaction SMIRKS, vendored at
``reaction_mcp/data/named_reaction_smirks.json`` (JSON-lines: ``{name, smirks}``,
~528 SMIRKS / ~470 distinct names; ``smirks`` is forward ``reactants>>product``,
so we match the PRODUCT side). Source repo: external/Rxn-INSIGHT.

Override the corpus with ``REACTION_TEMPLATE_LIBRARY``. Two formats are accepted:
  - ``*.json``           : JSON-lines named SMIRKS (forward; product = RHS).
  - ``*.csv`` / ``*.csv.gz`` : AiZynth-style tab-separated retro templates
    (columns incl. ``retro_template`` = ``product>>reactants`` and
    ``library_occurence``; product = LHS, ranked by occurrence). The bundled
    ``models/aizynthfinder/uspto_templates.csv.gz`` (~42.5k) works but is
    unnamed — use it for breadth, the named corpus for interpretability.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from rdkit import Chem

_DEFAULT_LIBRARY = Path(__file__).resolve().parent / "data" / "named_reaction_smirks.json"

_LIB: list[dict[str, Any]] | None = None
_LIB_LOCK = threading.Lock()
_LIB_ERROR: str | None = None
_NAMED: bool = True  # True -> SMIRKS named corpus; False -> aizynth retro CSV


def _library_path() -> Path:
    env = os.getenv("REACTION_TEMPLATE_LIBRARY")
    return Path(env) if env else _DEFAULT_LIBRARY


def _load_named(path: Path) -> list[dict[str, Any]]:
    lib: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sm = str(row.get("smirks", ""))
            if ">>" not in sm:
                continue
            patt = Chem.MolFromSmarts(sm.split(">>")[-1])  # product side of forward SMIRKS
            if patt is None:
                continue
            lib.append({
                "name": str(row.get("name", "unnamed")),
                "pattern": patt,
                "template": sm,
                "occurrence": None,
            })
    return lib


def _load_aizynth(path: Path) -> list[dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    if "retro_template" not in df.columns:
        raise ValueError(f"missing 'retro_template' column: {list(df.columns)}")
    lib: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rt = str(row["retro_template"])
        if ">>" not in rt:
            continue
        patt = Chem.MolFromSmarts(rt.split(">>", 1)[0])  # product side of retro template
        if patt is None:
            continue
        lib.append({
            "name": str(row.get("classification", "") or "unclassified"),
            "pattern": patt,
            "template": rt,
            "occurrence": int(row.get("library_occurence", 0) or 0),
        })
    return lib


def _load_library() -> list[dict[str, Any]]:
    """Lazily load + compile every template's product-side SMARTS; cache."""
    global _LIB, _LIB_ERROR, _NAMED
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB or []
    with _LIB_LOCK:
        if _LIB is not None or _LIB_ERROR is not None:
            return _LIB or []
        path = _library_path()
        if not path.exists():
            _LIB_ERROR = f"template library not found: {path}"
            return []
        try:
            if path.suffix == ".json":
                _NAMED = True
                _LIB = _load_named(path)
            else:
                _NAMED = False
                _LIB = _load_aizynth(path)
        except Exception as e:  # noqa: BLE001
            _LIB_ERROR = f"failed to load template library {path}: {e}"
            return []
        return _LIB


def library_status() -> dict[str, Any]:
    lib = _load_library()
    return {"path": str(_library_path()), "named": _NAMED,
            "n_templates": len(lib), "error": _LIB_ERROR}


def match_templates(
    smiles: str,
    top_k: int = 10,
    name_filter: str | None = None,
) -> dict[str, Any]:
    """Match ``smiles`` against the named-template corpus (product side).

    Returns the named reactions that could produce the molecule, grouped by
    name. For the named corpus, results are ranked by match specificity (size of
    the matched substructure — more specific first); for an occurrence-weighted
    CSV corpus, by corpus frequency. Each result carries the reaction name, the
    template SMIRKS/SMARTS (feed to ``check_template_compatibility``), and
    matched atoms. ``name_filter`` keeps only names containing that substring
    (case-insensitive).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"invalid SMILES: {smiles!r}", "valid": False}
    lib = _load_library()
    if not lib:
        return {"error": _LIB_ERROR or "empty template library", "valid": False}

    nf = name_filter.lower() if name_filter else None
    # Collapse multiple templates sharing a name into one best entry.
    by_name: dict[str, dict[str, Any]] = {}
    for t in lib:
        name = t["name"]
        if nf and nf not in name.lower():
            continue
        matches = mol.GetSubstructMatches(t["pattern"])
        if not matches:
            continue
        best_atoms = max(len(m) for m in matches)
        cur = by_name.get(name)
        if cur is None or best_atoms > cur["_spec"] or (
            t["occurrence"] is not None and (cur["occurrence"] or 0) < (t["occurrence"] or 0)
        ):
            by_name[name] = {
                "name": name,
                "template": t["template"],
                "occurrence": t["occurrence"],
                "match_count": len(matches),
                "matched_atoms": [list(m) for m in matches[:3]],
                "_spec": best_atoms,
            }

    hits = list(by_name.values())
    # rank: by occurrence if available, else by match specificity (atoms matched)
    if any(h["occurrence"] is not None for h in hits):
        hits.sort(key=lambda h: (h["occurrence"] or 0, h["_spec"]), reverse=True)
    else:
        hits.sort(key=lambda h: (h["_spec"], h["name"]), reverse=True)
    for h in hits:
        h.pop("_spec", None)

    top = hits[: max(1, top_k)]
    return {
        "smiles": smiles,
        "n_applicable": len(hits),
        "templates": top,
        "names": [h["name"] for h in top],
        "valid": True,
        "error": None,
    }


def extract_reaction_template(mapped_smiles: str) -> dict[str, Any]:
    """Extract a reaction-specific retro template (SMARTS) from a reaction via rdchiral.

    Unlike :func:`match_templates` (which matches a molecule against a fixed
    named-reaction corpus), this derives a *fresh* retro template from the given
    reaction itself — generalizing to reactions outside the named set. The input
    must be atom-mapped ``reactants>>product``; the server maps it first.

    Returns ``{retro_template, necessary_reagent, intra_only, dimer_only}``:
      - ``retro_template``    : rdchiral ``reaction_smarts`` (product>>reactants,
                                with the leaving/attacking atoms generalized) —
                                feedable straight into template compatibility checks.
      - ``necessary_reagent`` : reagent context rdchiral judged essential (or None).
      - ``intra_only``        : template only valid intramolecularly.
      - ``dimer_only``        : template only valid as a dimerization.
    """
    if ">>" not in mapped_smiles:
        return {"valid": False, "error": "reaction_smiles must contain '>>' (R>>P)"}
    try:
        from rdchiral.template_extractor import extract_from_reaction
    except ImportError:
        return {"valid": False, "error": "rdchiral not installed"}

    left, right = mapped_smiles.split(">>", 1)
    reactants = left.split(">")[0] if ">" in left else left
    try:
        out = extract_from_reaction({"reactants": reactants, "products": right, "_id": "0"})
    except Exception as e:  # noqa: BLE001 — rdchiral raises broadly on odd inputs
        return {"valid": False, "error": f"template extraction failed: {e}"}
    if not out or not out.get("reaction_smarts"):
        return {"valid": False,
                "error": "rdchiral could not extract a template (check atom mapping)"}

    return {
        "retro_template":    out["reaction_smarts"],
        "necessary_reagent": out.get("necessary_reagent") or None,
        "intra_only":        bool(out.get("intra_only", False)),
        "dimer_only":        bool(out.get("dimer_only", False)),
        "valid":             True,
        "error":             None,
    }


__all__ = ["match_templates", "library_status", "extract_reaction_template"]
