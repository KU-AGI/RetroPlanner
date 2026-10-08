"""SMILES normalization shared by backends and the consensus layer.

Cross-backend voting only works if every model's output is reduced to the same
canonical string, so all candidate comparison goes through :func:`canonical`.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonical(smiles: str, *, isomeric: bool = True) -> str | None:
    """Canonical SMILES, or ``None`` if RDKit cannot parse it."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def heavy_atom_count(smiles: str) -> int | None:
    """Heavy atoms in a SMILES, or ``None`` if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
    return None if mol is None else mol.GetNumHeavyAtoms()


# Heavy atoms an unrecorded reagent may plausibly contribute to a product. Cbz, the largest
# common protecting group, brings 9; 10 therefore admits every real protection or alkylation
# while still rejecting anything that creates a skeleton out of nothing.
RETRO_MASS_SLACK = 10


def retro_conserves_mass(product: str, precursors: list[str], slack: int = RETRO_MASS_SLACK
                         ) -> bool:
    """Can these precursors supply the product's skeleton?

    Template-based retro models cannot violate this: rdchiral maps the product's atoms onto
    the precursors. Sequence models can -- asked for a deep beam they emit degenerate outputs
    like ['Br'], ['O'], ['C'] at negligible score. Harmless as ranked candidates, poisonous to
    a multi-step planner: its solve test only asks that every leaf be purchasable, single atoms
    ARE in stock, and the search then reports a solved route after one expansion.

    Unparseable input returns False -- a candidate we cannot check does not get a vote.
    """
    hp = heavy_atom_count(product)
    if hp is None or not precursors:
        return False
    total = 0
    for smi in precursors:
        h = heavy_atom_count(smi)
        if h is None:
            return False
        total += h
    return (hp - total) <= slack


def strip_atom_maps(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonical_set(smiles_list: list[str], *, isomeric: bool = True) -> list[str]:
    """Canonicalize each SMILES; drop unparseable ones; sort for a stable key."""
    out: list[str] = []
    for smi in smiles_list:
        c = canonical(smi, isomeric=isomeric)
        if c:
            out.append(c)
    return sorted(out)


def split_components(smiles: str) -> list[str]:
    """Split a dot-joined / mixture SMILES into canonical components."""
    if not smiles:
        return []
    return canonical_set(smiles.split("."))


def same_molecule(a: str, b: str) -> bool:
    ca, cb = canonical(a), canonical(b)
    return ca is not None and ca == cb


__all__ = [
    "canonical",
    "strip_atom_maps",
    "canonical_set",
    "split_components",
    "same_molecule",
]
