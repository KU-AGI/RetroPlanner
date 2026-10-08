"""CGR (Condensed Graph of Reaction) analysis.

Wraps CGRtools + RDKit to produce semantic bond-change analysis.
Atom-map numbers are used internally only; outputs use FG descriptions.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdchem

BOND_SYMBOL = {1: "single", 1.5: "aromatic", 2: "double", 3: "triple"}
HYB_SYMBOL = {1: "sp3", 2: "sp2", 3: "sp", 4: "aromatic"}


def _bs(o) -> str | None:
    if o is None:
        return None
    return BOND_SYMBOL.get(o, str(o))


def _hyb(o) -> str | None:
    if o is None:
        return None
    return HYB_SYMBOL.get(o, str(o))


# ── CGRtools bond extraction ──────────────────────────────────────────────────

def _cgr_extract(mapped_smiles: str) -> dict:
    """Run CGRtools on a mapped reaction SMILES; return raw bond-change dicts."""
    try:
        from CGRtools import smiles as cgr_smiles_fn
        from CGRtools.containers import ReactionContainer
    except ImportError:
        return {"cgr_error": "CGRtools not installed"}

    parts = mapped_smiles.split(">")
    canonical = f"{parts[0]}>>{parts[2]}" if len(parts) == 3 else mapped_smiles

    try:
        rxn = cgr_smiles_fn(canonical)
    except Exception as e:
        return {"cgr_error": f"parse: {e}"}

    if not isinstance(rxn, ReactionContainer):
        return {"cgr_error": "not a ReactionContainer"}

    try:
        cgr = ~rxn
    except Exception as e:
        return {"cgr_error": f"cgr build: {e}"}

    def alabel(mn):
        a = cgr.atom(mn)
        return f"{a.atomic_symbol}:{mn}"

    formed, broken, order_changed = [], [], []
    for a1, a2, bond in cgr.bonds():
        o, p = bond.order, bond.p_order
        if o == p:
            continue
        l1, l2 = alabel(a1), alabel(a2)
        rec = {"atom1": l1, "atom2": l2,
               "reactant_order": _bs(o), "product_order": _bs(p)}
        if o is None:
            rec["change"] = "formed"
            rec["description"] = f"{l1}—{l2}: formed ({_bs(p)} bond)"
            formed.append(rec)
        elif p is None:
            rec["change"] = "broken"
            rec["description"] = f"{l1}—{l2}: broken ({_bs(o)} bond)"
            broken.append(rec)
        else:
            rec["change"] = "order_changed"
            rec["description"] = f"{l1}—{l2}: order changed ({_bs(o)} → {_bs(p)})"
            order_changed.append(rec)

    charge_changes = []
    for mn, atom in cgr.atoms():
        if atom.charge != atom.p_charge:
            charge_changes.append({
                "atom": alabel(mn),
                "reactant_charge": atom.charge,
                "product_charge": atom.p_charge,
            })

    try:
        center_atoms = [alabel(i) for i in cgr.center_atoms]
    except Exception:
        center_atoms = []

    return {
        "cgr_bond_changes": formed + broken + order_changed,
        "cgr_charge_changes": charge_changes,
        "cgr_center_atoms": center_atoms,
        "cgr_n_bonds_formed": len(formed),
        "cgr_n_bonds_broken": len(broken),
        "cgr_error": None,
    }


# ── FG classifier ─────────────────────────────────────────────────────────────

def _atom_env(mol: Chem.Mol, atom: rdchem.Atom) -> str:
    sym   = atom.GetSymbol()
    arom  = atom.GetIsAromatic()
    nbrs  = [(b.GetOtherAtom(atom).GetSymbol(),
              b.GetBondTypeAsDouble(),
              b.GetOtherAtom(atom).GetIsAromatic())
             for b in atom.GetBonds()]
    halogens = {"F", "Cl", "Br", "I"}
    nbr_syms = {s for s, _, _ in nbrs}

    if sym == "C":
        dbl_O = any(s == "O" and bt == 2.0 for s, bt, _ in nbrs)
        dbl_N = any(s == "N" and bt == 2.0 for s, bt, _ in nbrs)
        dbl_C = any(s == "C" and bt == 2.0 for s, bt, _ in nbrs)
        trp_C = any(s == "C" and bt == 3.0 for s, bt, _ in nbrs)
        trp_N = any(s == "N" and bt == 3.0 for s, bt, _ in nbrs)
        lg    = nbr_syms & halogens
        sO    = any(s == "O" and bt == 1.0 for s, bt, _ in nbrs)
        sN    = any(s == "N" and bt == 1.0 for s, bt, _ in nbrs)
        sS    = any(s == "S" and bt == 1.0 for s, bt, _ in nbrs)
        if arom:
            if sN: return "aromatic C (bonded to N)"
            if sO: return "aromatic C (bonded to O)"
            return "aromatic C"
        if trp_C: return "alkynyl C (C≡C)"
        if trp_N: return "nitrile C (C≡N)"
        if dbl_O and lg:  return f"acyl C (C=O, {'/'.join(sorted(lg))} leaving group)"
        if dbl_O and sO:  return "ester/acid carbonyl C (C=O, O neighbor)"
        if dbl_O and sN:  return "amide/carbamate carbonyl C (C=O, N neighbor)"
        if dbl_O and sS:  return "thioester carbonyl C (C=O, S neighbor)"
        if dbl_O:         return "carbonyl C (C=O)"
        if dbl_N:         return "imine/iminium C (C=N)"
        if dbl_C:         return "alkene C (C=C)"
        if lg:            return f"sp3 C (electrophilic, {'/'.join(sorted(lg))} attached)"
        if sO and sN:     return "sp3 C (O and N neighbors)"
        if sO:            return "sp3 C (O neighbor)"
        if sN:            return "sp3 C (N neighbor)"
        return "sp3 C"

    if sym == "O":
        hasH = atom.GetTotalNumHs() > 0
        dblC = any(s == "C" and bt == 2.0 for s, bt, _ in nbrs)
        cNbr = [s for s, _, _ in nbrs if s == "C"]
        if hasH: return "hydroxyl O (–OH)"
        if dblC: return "carbonyl O (C=O)"
        if len(cNbr) == 2:
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "C":
                    for b2 in other.GetBonds():
                        o2 = b2.GetOtherAtom(other)
                        if o2.GetSymbol() == "O" and b2.GetBondTypeAsDouble() == 2.0:
                            return "ester O (–O–C=O)"
            return "ether O (–O–)"
        return "O"

    if sym == "N":
        hasH   = atom.GetTotalNumHs() > 0
        cNbr   = sum(1 for s, _, _ in nbrs if s == "C")
        aromN  = any(a for _, _, a in nbrs)
        dblC   = any(s == "C" and bt == 2.0 for s, bt, _ in nbrs)
        if arom:             return "aromatic N (ring)"
        if dblC:             return "imine N (C=N)"
        if hasH and cNbr == 1: return "primary amine N (–NH2)"
        if hasH and cNbr == 2: return "secondary amine N (–NH–)"
        if not hasH and cNbr >= 2: return "tertiary amine N"
        if aromN:            return "N (adjacent to aromatic)"
        return "N"

    if sym == "S":
        dblO = sum(1 for s, bt, _ in nbrs if s == "O" and bt == 2.0)
        if dblO >= 2: return "sulfonyl S (S(=O)2)"
        if dblO == 1: return "sulfoxide S (S=O)"
        if atom.GetTotalNumHs() > 0: return "thiol S (–SH)"
        return "thioether S (–S–)"

    if sym in {"F", "Cl", "Br", "I"}:
        return f"{sym} (leaving group / halide)"
    if sym == "B":
        return "boronic/boronate B" if any(s == "O" for s, _, _ in nbrs) else "organoboron B"
    if sym == "Si": return "silyl Si"
    if sym == "P":
        return "phosphoryl P (P=O)" if any(s == "O" and bt == 2.0 for s, bt, _ in nbrs) else "phosphine P"
    return sym


def _build_fg_map(mapped_smiles: str, cgr_center_atoms: list[str]) -> dict[str, str]:
    """Build {atom_label: fg_description} for each center atom.

    Uses CGRtools to discover auto-assigned map numbers, then matches to RDKit
    atoms for _atom_env lookup.
    """
    if not mapped_smiles or not cgr_center_atoms:
        return {}

    # collect CGRtools map_num → symbol
    cgr_map: dict[int, str] = {}
    try:
        from CGRtools import smiles as cgr_smiles_fn
        parts = mapped_smiles.split(">")
        canon = f"{parts[0]}>>{parts[2]}" if len(parts) == 3 else mapped_smiles
        rxn = cgr_smiles_fn(canon)
        for mol_part in list(rxn.reactants) + list(rxn.products):
            for mn, atom in mol_part.atoms():
                cgr_map.setdefault(mn, atom.atomic_symbol)
    except Exception:
        pass

    # parse combined mol with RDKit
    parts = mapped_smiles.split(">")
    combined = f"{parts[0]}.{parts[2]}" if len(parts) == 3 else parts[0]
    mol = Chem.MolFromSmiles(combined)
    if mol is None:
        return {c: "parse error" for c in cgr_center_atoms}

    # build map_num → RDKit atom
    map_to_atom: dict[int, rdchem.Atom] = {}
    for atom in mol.GetAtoms():
        mn = atom.GetAtomMapNum()
        if mn:
            map_to_atom[mn] = atom

    # fill in auto-assigned atoms (CGRtools-only map nums) by symbol matching
    missing = {mn: sym for mn, sym in cgr_map.items() if mn not in map_to_atom}
    if missing:
        unmapped: dict[str, list[rdchem.Atom]] = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                unmapped.setdefault(atom.GetSymbol(), []).append(atom)
        for mn, sym in sorted(missing.items()):
            pool = unmapped.get(sym, [])
            if pool:
                map_to_atom[mn] = pool.pop(0)

    result: dict[str, str] = {}
    for label in cgr_center_atoms:
        try:
            mn = int(label.split(":")[1])
        except (IndexError, ValueError):
            result[label] = "unknown"
            continue
        atom = map_to_atom.get(mn)
        if atom is None:
            result[label] = "not found"
            continue
        try:
            result[label] = _atom_env(mol, mol.GetAtomWithIdx(atom.GetIdx()))
        except Exception:
            result[label] = atom.GetSymbol()
    return result


# ── Reaction type inference ───────────────────────────────────────────────────

def infer_reaction_type(bond_changes: list[dict]) -> str:
    if not bond_changes:
        return "unknown"

    halogens = {"Cl", "Br", "I", "F"}
    formed  = [b for b in bond_changes if b.get("change") == "formed"]
    broken  = [b for b in bond_changes if b.get("change") == "broken"]
    changed = [b for b in bond_changes if b.get("change") == "order_changed"]

    def sym(label: str) -> str:
        return label.split(":")[0] if ":" in label else label

    def bstr(b: dict) -> str:
        return f"{sym(b.get('atom1',''))}-{sym(b.get('atom2',''))}"

    n_f, n_b = len(formed), len(broken)

    if n_f == 1 and n_b == 1:
        fa = {sym(formed[0]["atom1"]), sym(formed[0]["atom2"])}
        ba = {sym(broken[0]["atom1"]), sym(broken[0]["atom2"])}
        fb = "-".join(sorted(fa))
        bb = "-".join(sorted(ba))

        if fa == {"C", "N"} and ba & halogens:
            return "N-alkylation (C-N formed, C-X broken)"
        if fa == {"C", "N"} and "C" in ba and "O" in ba:
            return "N-acylation / amide formation (C-N formed, C-O broken)"
        if fa == {"C", "N"} and "C" in ba and "N" in ba:
            return "N-N / N displacement (C-N formed, C-N broken)"
        if fa == {"C", "O"} and ba & halogens:
            return "O-alkylation / esterification (C-O formed, C-X broken)"
        if fa == {"C", "O"} and "O" in ba and "C" in ba:
            return "transesterification / acyl substitution (C-O formed, C-O broken)"
        if fa == {"C", "O"} and "O" in ba and "H" in ba:
            return "O-H activation / esterification (C-O formed, O-H broken)"
        if fa == {"C"} and ba & halogens:
            return "C-C coupling (C-C formed, C-X broken)"
        if fa == {"C"} and "C" in ba and "O" in ba:
            return "C-C coupling via acyl (C-C formed, C-O broken)"
        if fa == {"C", "S"}:
            return f"C-S bond formation (C-S formed, {bb} broken)"
        if fa in ({"C", "B"}, {"B", "C"}):
            return "C-B bond formation"
        if fa == {"N"} and ba & halogens:
            return "N-N coupling (N-N formed, N-X broken)"
        if fa == {"N", "O"}:
            return "N-O bond formation"
        if "O" in fa and "H" in ba:
            return "deprotection (O-H restored)"
        if "N" in fa and "H" in ba:
            return "N-deprotection (N-H restored)"
        return f"substitution ({fb} formed, {bb} broken)"

    if n_f >= 2 and n_b == 0:
        return f"addition ({', '.join(bstr(b) for b in formed)} formed)"
    if n_f == 0 and n_b >= 2:
        return f"elimination ({', '.join(bstr(b) for b in broken)} broken)"
    if n_f >= 2 and n_b >= 2:
        return f"rearrangement / multi-bond ({n_f} formed, {n_b} broken)"
    if n_f == 0 and n_b == 0 and changed:
        atoms_ch = set()
        for b in changed:
            atoms_ch.add(sym(b["atom1"])); atoms_ch.add(sym(b["atom2"]))
        if "O" in atoms_ch:
            return "oxidation/reduction (bond order change with O)"
        return "bond-order change (tautomerism or conjugation)"
    return f"reaction ({n_f} bonds formed, {n_b} broken)"


# ── Top-level semantic analysis ───────────────────────────────────────────────

def analyze(mapped_smiles: str) -> dict:
    """Full CGR analysis of a mapped reaction SMILES.

    Returns a semantic dict with no raw atom-map numbers in the output.
    """
    raw = _cgr_extract(mapped_smiles)
    if raw.get("cgr_error"):
        return {"error": raw["cgr_error"], "valid": False}

    bond_changes    = raw["cgr_bond_changes"]
    center_atoms    = raw["cgr_center_atoms"]
    fg_map          = _build_fg_map(mapped_smiles, center_atoms)
    reaction_type   = infer_reaction_type(bond_changes)

    def _enrich(b: dict) -> dict:
        a1_fg = fg_map.get(b["atom1"], b["atom1"].split(":")[0])
        a2_fg = fg_map.get(b["atom2"], b["atom2"].split(":")[0])
        a1s   = b["atom1"].split(":")[0]
        a2s   = b["atom2"].split(":")[0]
        bond_ord = b.get("product_order") or b.get("reactant_order") or "?"
        return {
            "atoms":       f"{a1s}-{a2s}",
            "bond_order":  bond_ord,
            "atom1_fg":    a1_fg,
            "atom2_fg":    a2_fg,
            "description": f"{a1s}-{a2s} {bond_ord} bond "
                           f"({'formed' if b['change'] == 'formed' else 'broken' if b['change'] == 'broken' else 'order changed'})"
                           f" ({a1_fg} ↔ {a2_fg})",
        }

    bonds_formed        = [_enrich(b) for b in bond_changes if b["change"] == "formed"]
    bonds_broken        = [_enrich(b) for b in bond_changes if b["change"] == "broken"]
    bonds_order_changed = [_enrich(b) for b in bond_changes if b["change"] == "order_changed"]

    summary_parts = []
    if bonds_formed:
        summary_parts.append("Bonds formed: " + ", ".join(b["atoms"] for b in bonds_formed))
    if bonds_broken:
        summary_parts.append("Bonds broken: " + ", ".join(b["atoms"] for b in bonds_broken))

    return {
        "reaction_type":        reaction_type,
        "bonds_formed":         bonds_formed,
        "bonds_broken":         bonds_broken,
        "bonds_order_changed":  bonds_order_changed,
        "n_bonds_formed":       raw["cgr_n_bonds_formed"],
        "n_bonds_broken":       raw["cgr_n_bonds_broken"],
        "center_atom_fg":       {k: v for k, v in fg_map.items()
                                 if "not found" not in v and v != "parse error"},
        "cgr_summary":          " | ".join(summary_parts) if summary_parts else "no bond changes detected",
        "charge_changes":       raw["cgr_charge_changes"],
        "valid":                True,
        "error":                None,
    }


# ── Batch CGR analysis for ML retro candidates ───────────────────────────────

def analyze_candidates(
    product_smiles: str,
    candidates: list,
    rxnmapper_fn=None,
) -> list[dict]:
    """Run CGR analysis on a batch of ML-predicted retro precursor sets.

    For each candidate (str ``"A.B"`` or list ``["A", "B"]``), constructs
    ``precursors>>product``, optionally atom-maps via ``rxnmapper_fn``, then
    runs the full semantic CGR analysis.  Returns one result dict per candidate,
    augmented with ``candidate_idx`` and ``candidate_smiles``.

    This is the bridge from ML retro output → structured bond-change info,
    enabling downstream rejection sampling and multi-step path search.
    """
    results: list[dict] = []
    for i, cand in enumerate(candidates):
        if isinstance(cand, str):
            precursors_str = cand
        elif isinstance(cand, (list, tuple)):
            precursors_str = ".".join(str(s) for s in cand)
        elif isinstance(cand, dict):
            mols = cand.get("molecules") or cand.get("reactants") or []
            precursors_str = ".".join(str(s) for s in mols)
        else:
            precursors_str = str(cand)

        rxn_smiles = f"{precursors_str}>>{product_smiles}"
        mapped = rxn_smiles
        mapping_confidence: float | None = None

        if rxnmapper_fn is not None and ":" not in rxn_smiles:
            try:
                res = rxnmapper_fn([rxn_smiles])
                if res:
                    mapped = res[0].get("mapped_rxn", rxn_smiles)
                    mapping_confidence = float(res[0].get("confidence", 0.0))
            except Exception:
                pass

        cgr_result = analyze(mapped)
        cgr_result["candidate_idx"] = i
        cgr_result["candidate_smiles"] = precursors_str
        cgr_result["product_smiles"] = product_smiles
        if mapping_confidence is not None:
            cgr_result["mapping_confidence"] = mapping_confidence
        results.append(cgr_result)

    return results


# ── CGR topology helpers (reaction center / shell / signature) ────────────────

def _open_cgr(mapped_smiles: str):
    """Parse a (mapped) reaction SMILES into a CGRtools CGRContainer.

    Returns ``(cgr, error)`` — ``cgr`` is None on failure.
    """
    try:
        from CGRtools import smiles as cgr_smiles_fn
        from CGRtools.containers import ReactionContainer
    except ImportError:
        return None, "CGRtools not installed"

    parts = mapped_smiles.split(">")
    canonical = f"{parts[0]}>>{parts[2]}" if len(parts) == 3 else mapped_smiles
    try:
        rxn = cgr_smiles_fn(canonical)
    except Exception as e:
        return None, f"parse: {e}"
    if not isinstance(rxn, ReactionContainer):
        return None, "not a ReactionContainer"
    try:
        return ~rxn, None
    except Exception as e:
        return None, f"cgr build: {e}"


def _center_atom_detail(cgr, fg_map: dict[str, str]) -> list[dict]:
    """Per reaction-center atom: hybridization / neighbor / charge changes + FG."""
    out: list[dict] = []
    for mn in cgr.center_atoms:
        a = cgr.atom(mn)
        label = f"{a.atomic_symbol}:{mn}"
        rec = {
            "atom": label,
            "element": a.atomic_symbol,
            "in_ring": bool(getattr(a, "in_ring", False)),
            "fg": fg_map.get(label, a.atomic_symbol),
            "hybridization": _hyb(getattr(a, "hybridization", None)),
            "neighbors": getattr(a, "neighbors", None),
            "charge": a.charge,
        }
        # dynamic (reactant -> product) changes
        p_hyb = _hyb(getattr(a, "p_hybridization", None))
        p_nbr = getattr(a, "p_neighbors", None)
        if p_hyb != rec["hybridization"]:
            rec["hybridization_change"] = f"{rec['hybridization']} -> {p_hyb}"
        if p_nbr != rec["neighbors"]:
            rec["neighbor_change"] = f"{rec['neighbors']} -> {p_nbr}"
        if a.p_charge != a.charge:
            rec["charge_change"] = f"{a.charge} -> {a.p_charge}"
        out.append(rec)
    return out


def extract_reaction_center(mapped_smiles: str, max_radius: int = 2) -> dict:
    """Extract the reaction center: atoms, bonds, atom-level changes + graded templates.

    Returns a semantic dict (no raw map numbers in the headline fields):
      - ``reaction_type``     : inferred class
      - ``center_atoms``      : per-atom {element, fg, hybridization(+change),
                                 neighbor_change, charge_change, in_ring}
      - ``center_bonds``      : bonds at the center {atoms, change, order}
      - ``center_template``   : radius-0 CGR signature (minimal reaction template)
      - ``local_environment`` : the reaction center expanded shell-by-shell into
                                its steric/electronic neighborhood — one entry per
                                radius ``1..max_radius`` (capped at 3), each an
                                augmented-substructure CGR signature
                                ``{radius, n_atoms, template}``. radius-0 is the
                                bare mechanism (``center_template``); each extra
                                shell adds one bond-layer of context that decides
                                whether a template actually applies.
      - ``n_components``      : number of disconnected reaction-center fragments
    """
    cgr, err = _open_cgr(mapped_smiles)
    if cgr is None:
        return {"error": err, "valid": False}

    raw = _cgr_extract(mapped_smiles)
    bond_changes = raw.get("cgr_bond_changes", []) if not raw.get("cgr_error") else []
    fg_map = _build_fg_map(mapped_smiles, raw.get("cgr_center_atoms", []))
    reaction_type = infer_reaction_type(bond_changes)

    center_atoms = _center_atom_detail(cgr, fg_map)

    center_bonds: list[dict] = []
    for (i, j) in cgr.center_bonds:
        b = cgr.bond(i, j)
        o, p = _bs(b.order), _bs(b.p_order)
        change = ("formed" if b.order is None else
                  "broken" if b.p_order is None else "order_changed")
        center_bonds.append({
            "atoms": f"{cgr.atom(i).atomic_symbol}-{cgr.atom(j).atomic_symbol}",
            "change": change,
            "order": f"{o} -> {p}" if change == "order_changed" else (p or o),
        })

    try:
        template = str(cgr.augmented_substructure(cgr.center_atoms, deep=0))
    except Exception:
        template = None

    # Graded local environment: reaction center + r bond-shells (r = 1..max_radius).
    # radius-0 is `center_template`; each shell adds the steric/electronic context
    # that decides whether a template actually applies.
    local_environment: list[dict] = []
    for r in range(1, max(0, min(int(max_radius), 3)) + 1):
        try:
            sub = cgr.augmented_substructure(cgr.center_atoms, deep=r)
        except Exception:
            break
        local_environment.append(
            {"radius": r, "n_atoms": sub.atoms_count, "template": str(sub)}
        )

    return {
        "reaction_type":     reaction_type,
        "center_atoms":      center_atoms,
        "center_bonds":      center_bonds,
        "center_template":   template,
        "local_environment": local_environment,
        "n_components":      len(cgr.centers_list),
        "valid":             True,
        "error":             None,
    }

