"""What `analyze` buys: bond changes, earned reaction names, molecule descriptors.

The board carries `q` (the expander's own confidence) and `ln$` (a price on purchasable
fragments) and nothing else. That is enough to RANK by score and not enough to say why: a
trace arguing from q alone is arguing from the number the search already sorted on. The three
things a chemist would actually cite -- which bond moves, whether the transformation has a
name, and what the fragments structurally are -- are not on the board and cannot be, because
none of them is knowable from board state.

So they are bought with one action. `analyze` is charged nothing in `COST` because it calls no
single-step model; it is RDKit, RXNMapper and a template corpus. But it is still an ACTION and
still has to be asked for, which is the point: the model that cites a bond position must have
requested the mapping first, or it is reciting facts it was never given -- the same failure as
leaking a route's outcome into the reasoning that chose it.

TWO PROVIDERS, one interface.

  `CacheEvidence`   reads the caches this pipeline already built -- evidence/bond_changes.json
                    (rxn_key -> formed/broken with atom positions), node_scores/
                    named_reaction.json (rxn_key -> tier/names/match) and
                    node_scores/mol_descriptors.json (smiles -> scaffold, rings, ...). Offline
                    SFT generation uses this: the facts are already computed for the whole
                    pool, so a board turn costs a dict lookup.
  `LiveEvidence`    calls the MCP tools (`reaction_get_bond_changes`,
                    `reaction_name_reaction`, `reaction_describe_molecules`) for molecules the
                    caches do not hold. Eval and any online rollout needs this, since a policy
                    at inference reaches states no offline pass enumerated.

A miss is reported as a miss. `bond: None` means "not measured", never "no bond changed" --
RXNMapper fails on some reactions (and fails SILENTLY on a GPU whose torch build
lacks kernels for it), and rendering that as "no bond change detected" would put a false fact
in front of the model with no way to tell it from a true one.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Optional, Protocol

SD = Path(__file__).resolve().parents[1]                  # evaluation/board
# config/paths.py, loaded by file location under its own name: the analysis pipeline has
# a module called `paths` of its own, and that one must keep the name.
_spec = importlib.util.spec_from_file_location("rp_paths", SD.parents[1] / "config" / "paths.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)
ROOT = Path(RP.MULTISTEP)                                 # tools/reaction-mcp
NODE_SCORES = Path(RP.NODE_SCORES)
BOND_CACHE = ROOT / "data" / "route_search" / "evidence" / "bond_changes.json"


def rxn_key(product: str, reactants) -> str:
    """Byte-identical to traj_route_common.rxn_key.

    Duplicated here rather than imported so `board` stays importable without the
    trajectory pipeline on the path. If either definition changes, both change.
    """
    return f"{product}>>" + ".".join(sorted(set(reactants)))


class Evidence(Protocol):
    def for_step(self, product: str, precursors) -> dict: ...
    def for_molecule(self, smiles: str) -> Optional[dict]: ...


class CacheEvidence:
    """Offline: the caches the pipeline already filled. Lazy-loaded, they are large."""

    def __init__(self, bond_path: Path = BOND_CACHE, node_scores: Path = NODE_SCORES):
        self._paths = {"bond": bond_path,
                       "named": node_scores / "named_reaction.json",
                       "desc": node_scores / "mol_descriptors.json"}
        self._c: dict[str, dict] = {}

    def _cache(self, which: str) -> dict:
        if which not in self._c:
            p = self._paths[which]
            self._c[which] = json.load(open(p)) if p.exists() else {}
        return self._c[which]

    def for_step(self, product: str, precursors) -> dict:
        k = rxn_key(product, precursors)
        bond = self._cache("bond").get(k)
        named = self._cache("named").get(k)
        return {
            # `valid: false` entries are misses, not measurements -- see the module docstring
            "bond": bond if (bond and bond.get("valid")) else None,
            "named": named,
        }

    def for_molecule(self, smiles: str) -> Optional[dict]:
        d = self._cache("desc").get(smiles)
        return d if (d and "error" not in d) else None


class LiveEvidence:
    """Online: the MCP tools, for states no offline pass enumerated.

    Batched per product, because both tools are: one call names and maps every candidate of a
    molecule, and calling them per candidate is the same work several times over.
    """

    def __init__(self, cache: Optional[CacheEvidence] = None):
        # The cache is consulted first even here: a state the offline pass did enumerate should
        # not pay for RXNMapper again, and the numbers must agree between the two paths.
        self.cache = cache or CacheEvidence()
        self._mem: dict = {}
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # RXNMapper: see cgr.py's note

    def _tools(self):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from reaction_mcp import server as S
        return S

    def prefetch(self, product: str, candidate_sets) -> None:
        """One call per tool for the whole candidate set. Call before `for_step`."""
        want = [list(c) for c in candidate_sets
                if rxn_key(product, c) not in self._mem]
        if not want:
            return
        S = self._tools()
        bonds, names = {}, {}
        try:
            r = S.reaction_get_bond_changes(product_smiles=product, candidates=want)
            for res in (r or {}).get("results") or []:
                i = res.get("candidate_idx")
                if i is not None and res.get("valid"):
                    bonds[i] = {"formed": res.get("bonds_formed"),
                                "broken": res.get("bonds_broken"),
                                "order_changed": res.get("bonds_order_changed"),
                                "mapped_rxn": res.get("mapped_rxn"), "valid": True}
        except Exception:                                             # noqa: BLE001
            pass
        try:
            r = S.reaction_name_reaction(product_smiles=product, candidates=want)
            for res in (r or {}).get("results") or []:
                i = res.get("candidate_idx")
                if i is not None:
                    names[i] = {k: res.get(k) for k in ("tier", "names", "match")}
        except Exception:                                             # noqa: BLE001
            pass
        for i, c in enumerate(want):
            self._mem[rxn_key(product, c)] = {"bond": bonds.get(i), "named": names.get(i)}

    def for_step(self, product: str, precursors) -> dict:
        k = rxn_key(product, precursors)
        if k in self._mem:
            return self._mem[k]
        hit = self.cache.for_step(product, precursors)
        if hit.get("bond") or hit.get("named"):
            return hit
        self.prefetch(product, [list(precursors)])
        return self._mem.get(k, {"bond": None, "named": None})

    def for_molecule(self, smiles: str) -> Optional[dict]:
        hit = self.cache.for_molecule(smiles)
        if hit:
            return hit
        key = ("mol", smiles)
        if key not in self._mem:
            try:
                r = self._tools().reaction_describe_molecules(smiles=[smiles])
                m = ((r or {}).get("molecules") or [{}])[0]
                self._mem[key] = None if "error" in m else m
            except Exception:                                         # noqa: BLE001
                self._mem[key] = None
        return self._mem[key]


# ------------------------------------------------------------------ derived facts
def split_shape(prod_desc: Optional[dict], prec_descs) -> dict:
    """What the disconnection does structurally: the part no single descriptor says.

    `scaffold_kept` asks whether the product's ring-and-linker TOPOLOGY survives into a
    precursor. If it does the step decorates a skeleton that already exists; if it does not the
    step builds or breaks the skeleton. Those are different decisions and the distinction is
    invisible in q.

    `size_ratio` separates a convergent coupling (two comparable fragments, ratio near 1) from a
    decoration (one small reagent, ratio near 0) -- which is what "is this a real
    simplification" turns on, and it is not the same question as heavy-atom count.
    """
    ok = [d for d in prec_descs if d]
    if not prod_desc or not ok:
        return {}
    gp = prod_desc.get("scaffold_generic")
    hv = [d.get("n_heavy") or 0 for d in ok]
    return {
        "scaffold_kept": bool(gp) and any(d.get("scaffold_generic") == gp for d in ok),
        "rings_product": prod_desc.get("n_rings"),
        "rings_precursors": sum(d.get("n_rings") or 0 for d in ok),
        "heavy_product": prod_desc.get("n_heavy"),
        "heavy_precursors": sum(hv),
        "heavy_balance": (prod_desc.get("n_heavy") or 0) - sum(hv),
        "fragments": sorted(hv, reverse=True),
        "size_ratio": round(min(hv) / max(max(hv), 1), 2) if len(hv) > 1 else 0.0,
        "stereo_product": prod_desc.get("n_stereocentres"),
        "stereo_precursors": sum(d.get("n_stereocentres") or 0 for d in ok),
    }
