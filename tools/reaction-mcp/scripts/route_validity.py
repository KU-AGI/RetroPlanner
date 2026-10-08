"""Does a route conserve atoms? The one implementation, shared by the audit and the sweep.

`solved` in a multi-step run is decided by leaf-in-stock alone. Nothing in that test asks whether
a proposed disconnection could actually run, so a generative single-step model can "solve" a
target by dropping part of it: the leaves it lands on are genuinely purchasable, the metric
passes, and the route is nonsense.

Criterion (retro direction, product -> children): every element of the product must be supplied
by some child, i.e. the product's element multiset must be a sub-multiset of the union of its
children's. A step that violates it is an *atom-conjuring* step; a route with any such step is
invalid.

This lives in its own module because it is scored in more than one place, and two copies of a
scoring rule drift. Import it; do not re-derive it.

Steps whose product or a child will not parse are SKIPPED, not counted as violations: RDKit
failing to read a SMILES is not evidence that the chemistry is wrong. They are returned
separately so a route built mostly of unparseable steps cannot masquerade as a clean one.
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Molecule-level, so it survives across routes and targets within a process. Route trees repeat
# the same intermediates constantly, so caching keeps the gate cheap.
_formula_cache: dict[str, Counter | None] = {}


def formula(smi: str) -> Counter | None:
    """Element multiset of a SMILES, or None if RDKit rejects it. Implicit hydrogens are not
    atoms in an RDKit mol, so this is a heavy-atom count."""
    if smi in _formula_cache:
        return _formula_cache[smi]
    m = Chem.MolFromSmiles(smi) if smi else None
    c = Counter(a.GetSymbol() for a in m.GetAtoms()) if m else None
    _formula_cache[smi] = c
    return c


def walk_steps(node: dict):
    """Yield (product_smiles, [child_smiles, ...]) for every reaction node in a route tree."""
    kids = node.get("children") or []
    if kids:
        yield node.get("smiles"), [k.get("smiles") for k in kids]
        for k in kids:
            yield from walk_steps(k)


def audit_route(root: dict) -> tuple[int, int]:
    """(scored_steps, atom_conjuring_steps) for one route tree. Unparseable steps are skipped
    and appear in neither count -- use `audit_route_detail` when that distinction matters."""
    n, bad, _ = audit_route_detail(root)
    return n, bad


def audit_route_detail(root: dict) -> tuple[int, int, int]:
    """(scored_steps, atom_conjuring_steps, skipped_unparseable_steps)."""
    n = bad = skipped = 0
    for prod, kids in walk_steps(root):
        fp = formula(prod)
        if fp is None:
            skipped += 1
            continue
        fc: Counter = Counter()
        ok = True
        for k in kids:
            f = formula(k)
            if f is None:
                ok = False
                break
            fc += f
        if not ok:
            skipped += 1
            continue
        n += 1
        if any(fc[el] < cnt for el, cnt in fp.items()):
            bad += 1
    return n, bad, skipped


def route_is_valid(root: dict) -> bool:
    """True when no step of the route conjures atoms.

    A route with NO scorable step is not valid: either every step was unparseable, or the tree is
    a bare target with no reaction at all, and in both cases there is no evidence of conservation
    to report. Calling that valid would let an empty or unreadable route pass the gate.
    """
    n, bad, _ = audit_route_detail(root)
    return n > 0 and bad == 0
