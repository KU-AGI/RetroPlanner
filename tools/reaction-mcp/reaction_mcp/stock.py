"""Purchasable-stock membership oracle for retro leaf decisions.

Mirrors the evaluation's stock check so an LLM agent can query the SAME oracle the
search uses to decide "is this molecule a buyable leaf (stop) or must it be decomposed
further?":

  * paroutes (n1/n5)   -> exact CANONICAL SMILES match against the route-set stock
  * pdvn / fusionretro -> InChIKey[:14] SKELETON match against the ZINC stock

The active stock is chosen PER BENCHMARK via env, loaded lazily once and cached:

  REACTION_STOCK        paroutes_n1 | paroutes_n5 | zinc_ik14   (built-in aliases)
                        or  smiles:/abs/path.txt   (exact canonical SMILES stock)
                        or  ik14:/abs/path.txt     (InChIKey[:14] skeleton stock)
  REACTION_STOCK_PATH   override the file path for a built-in alias.

Unset -> the tool reports ``stock_configured: false`` instead of guessing.
"""
from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem

_REACTION_MCP = Path(__file__).resolve().parents[1]        # tools/reaction-mcp
# The shared data directory, linked into the repo as external/sci-data (config/paths.py
# SCI_DATA). The package reads the env var rather than importing config/, which an
# installed copy does not carry.
_SCI_DATA = Path(os.environ.get("RP_SCI_DATA") or _REACTION_MCP.parents[1] / "external" / "sci-data")
_PAROUTES_TEST = _SCI_DATA / "paroutes" / "test"
_ZINC_IK14 = _REACTION_MCP / "models" / "aizynthfinder" / "zinc_inchikey14.txt"
_EMOLS_IK = _REACTION_MCP / "models" / "aizynthfinder" / "emol_inchikeys.txt"   # 23M full InChIKeys
_EMOLS_SMILES = _SCI_DATA / "retro_star" / "stock_emols.txt"

# alias -> (mode, default_path). mode:
#   smiles  -> canonical SMILES exact
#   ik14    -> InChIKey first-block (14-char skeleton; stereo/charge blind, looser)
#   ikfull  -> full 27-char InChIKey (stereo+protonation; the retro*/PDVN + AiZynthFinder convention)
_ALIASES: dict[str, tuple[str, Path]] = {
    "paroutes_n1": ("smiles", _PAROUTES_TEST / "stock_n1.txt"),
    "paroutes_n5": ("smiles", _PAROUTES_TEST / "stock_n5.txt"),
    "zinc_ik14":   ("ik14",   _ZINC_IK14),
    # Retro*/PDVN eMolecules stock, full-InChIKey match (AiZynthFinder InMemoryInchiKeyQuery
    # convention). USPTO-190 and ChEMBL-1000 use this.
    "emols":       ("ikfull", _EMOLS_IK),
    "emols_smiles": ("smiles", _EMOLS_SMILES),  # retro* raw-SMILES reproduction (paper's own impl)
}

# Multi-stock cache keyed by resolved (mode, path) so one server process can
# serve several stocks at once (e.g. an RL run using the env-configured stock
# while an eval run injects a per-call ``spec``). Each entry loaded once.
_CACHE: dict[tuple[str, str], tuple[str, frozenset, str]] = {}


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _ikfull(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)          # full 27-char InChIKey
    except Exception:  # noqa: BLE001 — rdkit InChI can raise on odd inputs
        return None


def _ik14(smi: str) -> str | None:
    k = _ikfull(smi)
    return k[:14] if k else None


def _resolve_spec(spec: str | None) -> tuple[str, Path] | None:
    """Return (mode, path) for an explicit ``spec`` string, or from the
    ``REACTION_STOCK`` env when ``spec`` is None. None if unconfigured.

    ``REACTION_STOCK_PATH`` overrides an alias's path only for the env-selected
    stock (spec is None) — an explicit per-call spec is self-contained.
    """
    from_env = spec is None
    if from_env:
        spec = (os.getenv("REACTION_STOCK") or "").strip()
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec in _ALIASES:
        mode, default_path = _ALIASES[spec]
        path = Path(os.getenv("REACTION_STOCK_PATH") or default_path) if from_env else default_path
        return mode, path
    if ":" in spec:
        mode, _, raw = spec.partition(":")
        mode = mode.strip().lower()
        if mode in ("smiles", "ik14", "ikfull") and raw.strip():
            return mode, Path(raw.strip())
    raise ValueError(
        f"stock spec {spec!r} not understood; use an alias "
        f"({'/'.join(_ALIASES)}) or 'smiles:<path>' / 'ik14:<path>' / 'ikfull:<path>'"
    )


def _load(spec: str | None = None) -> tuple[str, frozenset, str] | None:
    """Load the stock for ``spec`` (or the env default), cached per (mode, path).
    None if unconfigured; raises RuntimeError on a bad/missing path."""
    resolved = _resolve_spec(spec)          # raises ValueError on a bad spec string
    if resolved is None:
        return None
    mode, path = resolved
    key = (mode, str(path))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if not path.exists():
        raise RuntimeError(f"stock file missing: {path}")
    entries = frozenset(
        ln.strip() for ln in path.read_text().splitlines() if ln.strip()
    )
    loaded = (mode, entries, f"{mode}:{path.name} ({len(entries)})")
    _CACHE[key] = loaded
    return loaded


def stock_status(spec: str | None = None) -> dict:
    """Describe the active stock (for logging / preflight). ``spec`` overrides env."""
    try:
        loaded = _load(spec)
    except (RuntimeError, ValueError) as e:
        return {"stock_configured": False, "error": str(e)}
    if loaded is None:
        return {"stock_configured": False,
                "hint": "set REACTION_STOCK=paroutes_n1|paroutes_n5|zinc_ik14 "
                        "or smiles:<path> / ik14:<path> / ikfull:<path>, "
                        "or pass stock=<spec> per call"}
    mode, entries, label = loaded
    return {"stock_configured": True, "mode": mode, "size": len(entries), "stock": label}


def check_one(smiles: str, spec: str | None = None) -> dict:
    """Membership for one SMILES against the ``spec`` stock (or env), mirroring the eval."""
    loaded = _load(spec)      # raises if misconfigured; None if unset
    if loaded is None:
        return {"smiles": smiles, "in_stock": None, "error": "no stock configured"}
    mode, entries, _ = loaded
    if mode == "ik14":
        key = _ik14(smiles)
    elif mode == "ikfull":
        key = _ikfull(smiles)
    else:
        key = _canon(smiles)
    if key is None:
        return {"smiles": smiles, "in_stock": False, "valid": False,
                "reason": "unparseable_smiles"}
    return {"smiles": smiles, "in_stock": key in entries, "valid": True,
            "match_key": key, "mode": mode}
